import argparse
import os
import random
import numpy as np

from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

# PyG packages
from torch_geometric.loader import NeighborLoader, DataLoader, RandomNodeSampler, ClusterLoader, ClusterData, NeighborSampler, LinkNeighborLoader, DynamicBatchSampler
from torch_geometric.utils.convert import to_networkx, from_scipy_sparse_matrix
from torch_geometric.utils import degree, contains_isolated_nodes, homophily, k_hop_subgraph, index_to_mask

# Datasets
from datasets import get_dataset_ZINC, get_dataset_node_OGB, get_dataset_planetoid, get_dataset_OGB

# Evaluators
from ogb.nodeproppred import Evaluator as NEvaluator

# Import my utils 
from utils.graph_utils import build_clusters_pynndescent, get_comp_graph_pdf
from utils.pos_encoding import AddRandomWalkPE, AddLaplacianEigenvectorPE
from utils.logger import log_gradients

# Self-Attention CRF Module
from HSACRF_new import HSACRF2

# Clustering
from torch_cluster import knn_graph, radius_graph

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from utils import algos


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ogbn-products', help='Dataset to train')
parser.add_argument('--init_lr', type=float, default=0.001, help='Initial learing rate')
parser.add_argument('--epoches', type=int, default=2000, help='Number of traing epoches')
parser.add_argument('--hidden_dim', type=int, default=128, help='Dimensions of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep  probability)')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for l2 loss on embedding matrix')
parser.add_argument('--log_interval', type=int, default=10, help='Print iterval')
parser.add_argument('--log_dir', type=str, default='experiments', help='Train/val loss and accuracy logs')
parser.add_argument('--checkpoint_interval', type=int, default=20, help='Checkpoint saved interval')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--seed', type=int, default=1234, help='Seed for PRNG')
parser.add_argument('--early_stop', type=int, default=80, help='early stop iterations.')
parser.add_argument('--max_runs', type=int, default=20, help='Number of experiments.')
parser.add_argument('--bs', type=int, default=256, help='Size of batch size.')
parser.add_argument('--clip', type=float, default=2.0, help='Value of clipping gradient.')

parser.add_argument('--crf_it', type=int, default=5, help='Number of the total itrations in mean-field infernce for CRF.')
parser.add_argument('--mhsa_heads', type=int, default=4, help='Number of heads for MHSA.')
parser.add_argument('--attn_dropout', type=float, default=0.1, help='Probability of droupout in MHSA.')

parser.add_argument('--loader', type=str, default='random', help='Loader [random, neigh]')
parser.add_argument('--num_subgraphs', type=int, default=20, help='Number of sub-graphs for RandomSampler.')
parser.add_argument('--num_subgraphs_test', type=int, default=20, help='Number of sub-graphs for RandomSampler.')
parser.add_argument('--sample_neigh', type=int, default=3, help="Number of sampled neighbours.")

parser.add_argument('--num_clusters', type=int, default=30, help="Number of partitions for ClusterData.")
parser.add_argument('--num_clusters_test', type=int, default=15, help="Number of partitions for ClusterData.")

parser.add_argument('--embed_feat', action="store_true", help="Loading embedding from GIANT arch.")


parser.add_argument('--j', type=int, default=12, help='Number of workers.')
args = parser.parse_args()



@torch.no_grad()
def test_it(epoch, test_loader, evaluator):
    model.eval()

    y_true = {"train": [], "valid": [], "test": []}
    y_pred = {"train": [], "valid": [], "test": []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:03d}')
    
    num_test_nodes = 0
    for data in test_loader:
        data = data.cuda()

        if data.test_mask.sum() == 0:
            continue

        out = model(data)
        out = out.argmax(dim=-1, keepdim=True)

        num_test_nodes += out.shape[0]


        for split in ['train', 'valid', 'test']:
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_acc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['acc']

    valid_acc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['acc']

    test_acc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['acc']

    print("\tNum test nodes: ", num_test_nodes)

    return train_acc, valid_acc, test_acc, 0.0

@torch.no_grad()
def test(epoch, test_loader, evaluator, device):
    model.eval()

    data = test_loader.data
    #out = model.inference(data.x, test_loader, device)
    out = model.inference(test_loader, device, True)

    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1, keepdim=True)

    print("True: ", y_true.shape)
    print("Pred: ", y_pred.shape)

    train_acc = evaluator.eval({
        'y_true': y_true[data.train_mask],
        'y_pred': y_pred[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.valid_mask],
        'y_pred': y_pred[data.valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc, 0.0

def train(epoch, train_loader, model, optimizer, init_lr, summary, scheduler):
    model.train()

    mfi_it = 5
    total_bat = len(train_loader) * mfi_it
    pbar = tqdm(total=total_bat)

    total_loss = l_examples = 0

    it = 0
    for data in train_loader:
        data = data.cuda()

        if data.train_mask.sum() == 0:
            continue

        optimizer.zero_grad()

        # Get GT labels for training nodes
        y = data.y.squeeze(1)[data.train_mask]

        out = model(data, epoch)[data.train_mask]

        
        loss = F.nll_loss(out, y) 
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        step_it = epoch * total_bat + it         #  Total number of iterations
        log_gradients(model, summary, step_it)

        num_examples = data.train_mask.sum().item()
        total_loss += loss.item() * num_examples
        l_examples += num_examples


        pbar.set_description(f'Training epoch: {epoch:03d} Loss: {loss.item():03f}')
        pbar.update(1)

    pbar.set_description(f'Training epoch: {epoch:03d} Avg. Loss: {(total_loss/l_examples):03f}')
    pbar.close()

    return total_loss / l_examples


if __name__ == '__main__':
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


    """
    from text.text_encoding import TextEncoder
    txt_enc = TextEncoder().cuda()

    p = os.path.join("/media/Patrik/data", args.dataset, args.dataset.replace("-", "_"), "embeddings/X.all.txt")
    embeds = []

    bs= 256
    with open(p, "r") as f:
        lines = f.readlines()

        ln = len(lines)
        for i in range(0, len(lines), bs):
            mx = ln-1 if (i+bs) >= ln else i+bs
            print(f"From {i} to {mx}")

            sl = lines[i:mx]
            embed = txt_enc(sl)
            embeds.append(embed[:, 0, :].detach().clone().cpu())

    p = os.path.join("/media/Patrik/data", args.dataset, args.dataset.replace("-", "_"), "embeddings/X.all.robert-emb.pt")    
    embed = torch.cat(embeds, 0)
    torch.save(embed, p)
            
    raise Exception()
    """

    # Load ataset
    dataset = get_dataset_node_OGB(args.dataset)
    split_idx = dataset.get_idx_split()
    data = dataset.data

    # Calculate edge homophily for given dataset
    h = homophily(data.edge_index, data.y)
    print("Homophily: ", h)

    if args.embed_feat:
        print("Loading external node embeddings...")
        
        embed_path = os.path.join("/media/Patrik/data", args.dataset, args.dataset.replace("-", "_"), "embeddings/X.all.xrt-emb.npy")
        embed = np.load(embed_path, allow_pickle=True)
        
        # Roberta embeddings
        #embed_path = os.path.join("/media/Patrik/data", args.dataset, args.dataset.replace("-", "_"), "embeddings/X.all.robert-emb.pt")
        #embed = torch.load(embed_path, "cpu")

        embed = torch.cat([embed, embed[0].unsqueeze(0)])

        assert data.x.shape[0] == embed.shape[0], "Wrong number of features in the embeddings!"

        # Using loaded feature instead of original ones
        #data.edge_index = knn_graph(embed.cuda(), 30, cosine=True, num_workers=12).detach().cpu()
        data.edge_index = build_clusters_pynndescent(embed, n_neigh=30)

        # Load clusters from path
        #p = os.path.join("/media/Patrik/data", args.dataset, args.dataset.replace("-", "_"), "embeddings/knn_30_pyg.pt")
        #data.edge_index = torch.load(p, "cpu")

        #torch.save(data.edge_index, p)
       
        # Store new embeddings to data
        data.x = torch.from_numpy(embed)    

    # Split dataset with corresponing masks
    for split in ['train', 'valid', 'test']:
        data[f'{split}_mask'] = index_to_mask(split_idx[split], data.y.shape[0])
    
    num_cls = dataset.num_classes
    feat_dim = data.x.shape[-1] 
    print(f"Features dim: {data.x.shape[-1]}")

    acc_list = []

    # Create evaluator basedo on OGBN dataset (node classification)
    evaluator = NEvaluator(args.dataset)

    # Get loader/sampler for loading sub-graphs (mini-batches)
    if args.loader == "random":
        train_loader = RandomNodeSampler(data, num_parts=args.num_subgraphs, shuffle=True, num_workers=args.j)
        test_loader = RandomNodeSampler(data, num_parts=args.num_subgraphs_test, shuffle=False, num_workers=args.j)
    elif args.loader == "dynamic":  
        train_sampler = DynamicBatchSampler(dataset, max_num=1000, mode="node")
        train_loader = DataLoader(dataset, batch_sampler=train_sampler, num_workers=args.j)

        test_sampler = DynamicBatchSampler(dataset, max_num=3000, mode="node")
        test_loader = DataLoader(dataset, batch_sampler=test_sampler, num_workers=args.j)
    elif args.loader == "neigh":
        train_loader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[20]*1, batch_size=512, num_workers=args.j, shuffle=True)
        test_loader = NeighborLoader(data, num_neighbors=[15], batch_size=768, num_workers=args.j, shuffle=False)
    elif args.loader == "cluster":
        path = "/media/patpa/f0c432b3-57ba-4522-a804-cab376ec28835/projects/data/prod_clusters/"
        #path = "/media/patpa/f0c432b3-57ba-4522-a804-cab376ec28835/projects/data/prod_clusters_arxiv/"

        if "arxiv" in args.dataset:
            path = None # No need to save clusters -- creating is fast for ogbn-arxiv 

            # Need to transform to undirected graph here of METIS calculation will failed on Segmentation fault!
            from torch_geometric.transforms import ToUndirected
            data = ToUndirected()(data)


        cluster_data = ClusterData(data, num_parts=args.num_clusters, recursive=False, save_dir=path)
        train_loader = ClusterLoader(cluster_data, batch_size=args.bs, shuffle=True, num_workers=args.j)

        cluster_data = ClusterData(data, num_parts=args.num_clusters_test, recursive=False, save_dir=path)
        test_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False, num_workers=args.j)

    
    device = torch.device("cuda:0")

    best_test = best_valid_val = corr_train = corr_test = 0.0
    for i in range(args.max_runs):  
        log_dir = os.path.join(args.log_dir, args.dataset, str(i))

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        writer = SummaryWriter(log_dir)
        saved_checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset)

        if not os.path.exists(saved_checkpoint_dir):
            os.makedirs(saved_checkpoint_dir)

        model = HSACRF2(feat_dim, num_cls, crf_it=args.crf_it, hidden_feat=args.hidden_dim, out_hidden=args.hidden_dim, num_layers=2, num_heads=args.mhsa_heads)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr) #, weight_decay=args.weight_decay)

        """       
        optimizer = torch.optim.Adam([
            {"params": model.unary.parameters()},
            {"params": model.embed_feat.parameters()},
            #{"params": model.pair_pred.parameters()},
            {"params": model.comp_matrix},
            {"params": model.resweight},
            {"params": model.MHSA.Q.parameters(), "lr": args.init_lr*10},
            {"params": model.MHSA.K.parameters(), "lr": args.init_lr*10},
            {"params": model.MHSA.V.parameters(), "lr": args.init_lr*10},
        ], lr=args.init_lr, betas=(0.9, 0.98), eps=1e-9) #, weight_decay=args.weight_decay)
        """

        scheduler = None # Scheduler(optimizer, 128, 100)

        model = model.cuda()
        model.train()

        # Best value of valid. accuracy with corresponding train and test accuracy.
        for epoch in range(args.epoches + 1):
            #train_loss = 0.0
            #train_acc, val_acc, test_acc, val_loss = test_it(epoch, test_loader, evaluator, device)
            #print("Epoch: %d, train loss: %f, val loss: %f, train_acc: %f, val acc: %f, test_acc: %f"
            #        %(epoch, train_loss, val_loss, train_acc, val_acc, test_acc))

            train_loss = train(epoch, train_loader, model, optimizer, args.init_lr, writer, scheduler)

            if (epoch+1) % args.log_interval == 0:
                train_acc, val_acc, test_acc, val_loss = test_it(epoch, test_loader, evaluator)

                if test_acc > best_test:
                    best_test = test_acc

                if val_acc > best_valid_val:
                    best_valid_val = val_acc
                    corr_train = train_acc
                    corr_test = test_acc

                    #torch.save(model.state_dict(), os.path.join(saved_checkpoint_dir, "sacrf_node_best.pth"))
                
                acc_list.append(test_acc)

                print("Epoch: %d, train loss: %f, train_acc: %f, val acc: %f, test_acc: %f, best_test: %f"
                    %(epoch, train_loss, train_acc, val_acc, test_acc, best_test))

                writer.add_scalars('loss', {'train_loss': train_loss}, epoch)
                writer.add_scalars('accuracy', {'train_acc': train_acc,'val_acc': val_acc, 'test_acc': test_acc}, epoch)
            else:
                print("Epoch: %d, train loss: %f, best_test: %f" % (epoch, train_loss, best_test))

            #if (epoch+1) % args.checkpoint_interval == 0:
            #    torch.save(model.state_dict(), os.path.join(saved_checkpoint_dir, "sacrf_node_%d.pth"%epoch))

        writer.close()

    acc_list = np.asarray(acc_list)
    print("===========Stats===========")
    print("Runs: ", args.max_runs)
    print("Mean test acc: ", np.mean(acc_list), "+-", np.std(acc_list)) 
    print("Best test acc ", np.max(acc_list))
    print("===========================")
