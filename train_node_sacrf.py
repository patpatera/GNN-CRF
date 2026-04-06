import argparse
import os
import random
import copy
import gc
import numpy as np

from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from tensorboardX import SummaryWriter

from sklearn import metrics

# PyG packages
from torch_sparse import SparseTensor
from torch_cluster import random_walk

from torch_geometric.loader import NeighborLoader, DataLoader, RandomNodeSampler, ClusterLoader, ClusterData, ShaDowKHopSampler, GraphSAINTRandomWalkSampler, ImbalancedSampler
from torch_geometric.utils.convert import to_networkx, from_scipy_sparse_matrix
from torch_geometric.utils import degree, mask_to_index, homophily, coalesce, index_to_mask, dense_to_sparse, from_networkx
import torch_geometric.transforms as TF
from torch_geometric.nn import CorrectAndSmooth

# Datasets
from gnn_datasets import get_dataset_ZINC, get_dataset_node_OGB, get_dataset_planetoid, get_dataset_OGB, load_large_dataset
from dataset_orig import load_data

# Evaluators
from ogb.nodeproppred import Evaluator as NEvaluator

# Import my utils 
from utils.graph_utils import knn_graph_rewire, apply_feat_KNN, comp_to_edges, sinkhorn, extract_global, extract_global_outer, sharpen, rewire_attn, build_clusters_pynndescent
from utils.pos_encoding import AddRandomWalkPE, AddLaplacianEigenvectorPE
from utils.logger import log_gradients
from utils.gdc import GDC

from dataset_orig import load_data

# Self-Attention CRF Module
from HSACRF import HSACRF

# RWKV-like RNN module for Graph
from RNNCRF import RNNCRF

# Clustering
from torch_cluster import knn_graph, radius_graph


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='Dataset to train')
parser.add_argument('--init_lr', type=float, default=0.0004, help='Initial learing rate')
parser.add_argument('--epoches', type=int, default=800, help='Number of traing epoches')
parser.add_argument('--hidden_dim', type=int, default=128, help='Dimensions of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep  probability)')
parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight for l2 loss on embedding matrix')
parser.add_argument('--log_interval', type=int, default=10, help='Print iterval')
parser.add_argument('--log_dir', type=str, default='experiments', help='Train/val loss and accuracy logs')
parser.add_argument('--checkpoint_interval', type=int, default=20, help='Checkpoint saved interval')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--seed', type=int, default=42, help='Seed for PRNG')
parser.add_argument('--early_stop', type=int, default=80, help='early stop iterations.')
parser.add_argument('--max_runs', type=int, default=20, help='Number of experiments.')
parser.add_argument('--bs', type=int, default=1, help='Size of batch size.')
parser.add_argument('--clip', type=float, default=2.0, help='Value of clipping gradient.')
parser.add_argument('--T', type=float, default=1, help='Diffusion time.')

parser.add_argument('--c_entropy', type=float, default=0.0, help='Ratio of conditional entropy in loss.')
parser.add_argument('--optim', type=str, default='adam', help='Oprimiser')
parser.add_argument('--name_acc', type=str, default='default_acc', help='Name for test acc list')
parser.add_argument('--type', type=str, default='tra', help='Transductive (tra) or Inductive (ind)')

parser.add_argument('--rewire', type=int, default=2, help='Number of top neigh. to rewire graph by KNN-Graph.   ')
parser.add_argument('--crf_it', type=int, default=5, help='Number of the total itrations in mean-field infernce for CRF.')
parser.add_argument('--mhsa_heads', type=int, default=4, help='Number of heads for MHSA.')
parser.add_argument('--attn_dropout', type=float, default=0.1, help='Probability of droupout in MHSA.')

parser.add_argument('--loader', type=str, default='random', help='Loader [random, neigh]')
parser.add_argument('--num_subgraphs', type=int, default=20, help='Number of sub-graphs for RandomSampler.')
parser.add_argument('--num_subgraphs_test', type=int, default=20, help='Number of sub-graphs for RandomSampler.')
parser.add_argument('--sample_neigh', type=int, default=3, help="Number of sampled neighbours.")

parser.add_argument('--num_clusters', type=int, default=30, help="Number of partitions for ClusterData.")
parser.add_argument('--num_clusters_test', type=int, default=15, help="Number of partitions for ClusterData.")

parser.add_argument('--embed_feat', action="store_true", help="Loading embeddings from Roberta-large language model.")
parser.add_argument('--pseudo', type=float, default=-1.0, help='Pseudo threshold.')

parser.add_argument('--damping', type=float, default=.5, help='Damping param for LBP (default=0.5).')


parser.add_argument('--pretrained', type=str, default='', help='Path to the pre-trained weights.')


parser.add_argument('--j', type=int, default=2, help='Number of workers.')
args = parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cond_entropy(logits):
    probs = torch.softmax(logits, dim=1)
    # Use log softmax for stability.
    return -torch.sum(probs * torch.log_softmax(logits, dim=1)) / probs.shape[0]


def cond_entropy_edges(logits):
    probs = torch.softmax(logits, dim=1)
    # Use log softmax for stability.
    return -torch.sum(probs * logits) / probs.shape[0]


def loge_cross_entropy(x, labels, epsilon=(1 - math.log(2))):
    y = F.cross_entropy(x, labels, reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def soft_nll_loss(log_probs, target):
    target = F.one_hot(target, log_probs.shape[-1])
    loss = -(target * log_probs).sum(-1).mean(0)
    return loss


def soft_cross_entropy(logits, target):
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = soft_nll_loss(log_probs, target)
    return loss


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

def calc_f1(y_true, y_pred):
    """
    if not is_sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    """
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def ptr2index(ptr):
    ind = torch.arange(ptr.numel() - 1, dtype=ptr.dtype, device=ptr.device)
    return ind.repeat_interleave(ptr[1:] - ptr[:-1])

def to_edge_index(adj):
    r"""Converts a :class:`torch.sparse.Tensor` or a
    :class:`torch_sparse.SparseTensor` to edge indices and edge attributes.

    Args:
        adj (torch.sparse.Tensor or SparseTensor): The adjacency matrix.

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> adj = to_torch_coo_tensor(edge_index)
        >>> to_edge_index(adj)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([1., 1., 1., 1., 1., 1.]))
    """
    if isinstance(adj, SparseTensor):
        row, col, value = adj.coo()
        if value is None:
            value = torch.ones(row.size(0), device=row.device)
        return torch.stack([row, col], dim=0).long(), value

    if adj.layout == torch.sparse_coo:
        return adj.indices().detach().long(), adj.values()

    if adj.layout == torch.sparse_csr:
        row = ptr2index(adj.crow_indices().detach())
        col = adj.col_indices().detach()
        return torch.stack([row, col], dim=0).long(), adj.values()

    if adj.layout == torch.sparse_csc:
        col = ptr2index(adj.ccol_indices().detach())
        row = adj.row_indices().detach()
        return torch.stack([row, col], dim=0).long(), adj.values()

    raise ValueError(f"Unexpected sparse tensor layout (got '{adj.layout}')")


@torch.no_grad()
def test_it(epoch, model, test_loader, evaluator, pseudo, ogbn=True, crf_it=0):
    model.eval()
    ogbn = True

    #CS = CorrectAndSmooth(20, 0.2, 20, 0.)

    y_true = {"train": [], "valid": [], "test": []}
    y_pred = {"train": [], "valid": [], "test": []}

    pbar = tqdm(total=len(test_loader))
    num_test_nodes = 0

    acc_list = {"train": 0, "valid": 0, "test": 0}

    f1mic, f1mac = [], []

    total_sm = {"train": 0, "valid": 0, "test": 0}
    
    total_pred = None

    wl = 0
    widx= 0
    loss_val = 0
    for i, data in enumerate(test_loader):
        #data.x = embed[data.glob_idx] if hasattr(data, "glob_idx") else data.x
        data["inductive"] = False # During testing all nodes are available
        data = data.cuda()

        if hasattr(data, "batch_size") == False:
            data.batch_size = data.x.shape[0]

        if data.test_mask.sum() == 0:
            print("Not test nodes")
            continue

        ps_mask = None
        if not pseudo == None:
            train_mask_ps, __y, ps_mask, probs, ps_A, __pseudo = pseudo[i]
            data["pseudo"] = __pseudo.clone()

        out, _, _ = model(data, epoch, i) #[:data.batch_size]
        out_ = out[:data.batch_size]

        out = out_.argmax(dim=-1, keepdim=True)
        num_test_nodes += out.shape[0]

        for split in ['train', 'valid', 'test']:
            mask = data[f'{split}_mask'][:data.batch_size]
            if mask.sum() == 0:
                continue    
            
            y_true[split].append(data.y[:data.batch_size][mask].cpu())
            y_pred[split].append(out[mask].cpu())

            if split == 'test':
                loss = F.nll_loss(out_[mask].log_softmax(-1).cuda(), data.y[mask].squeeze(1).cuda())
                if loss > wl:
                    wl = loss
                    widx = i
                    total_pred = out
                loss_val += loss
                
            acc = out.squeeze()[mask].eq(data.y[:data.batch_size][mask]).sum()  #/ mask.sum()
            total_sm[split] += mask.sum()
            acc_list[split] += acc.item()

        pbar.set_description(f'Evaluation epoch: {epoch:03d}')
        pbar.update(1)
        model.reset_vars(i)


    pbar.close()

    if ogbn:
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
    else:
        train_acc = (acc_list['train']  / total_sm['train'])
        valid_acc = (acc_list['valid']  / total_sm['valid'])
        test_acc  = (acc_list['test']   / total_sm['test'])

    loss_val = loss_val / len(test_loader) 
    
    return train_acc, valid_acc, test_acc, loss_val, total_pred, widx

@torch.no_grad()
def infer_pseudo(epoch, data_loader, model, pseudo_th=0.8, pseudo=None, only_A=False):
    model.eval()

    cache_pseudo = []

    total_bat = len(data_loader)
    pbar = tqdm(total=total_bat)

    it = 0
    for data in data_loader:
        #data.x = embed[data.glob_idx] if hasattr(data, "glob_idx") else data.x
        data["inductive"] = False
        data = data.clone().cuda()

        out, _, _ = model(data, c_idx=it)

        # Filtering high confidence nodes
        out_prob = F.softmax(out, dim=-1)

        val, pred = torch.max(out_prob, dim=1)
        pseudo_mask = val >=  0.92 #pseudo_th

        # Marge hard-pseudo labels with training labels
        merge_mask = pseudo_mask.logical_or(data.train_mask)

        # Pseudo-labels mask only for unlabelled nodes
        pseudo_mask[data.train_mask] = False
    
        # Keep only HARD pseudo-labels for unlabelled data
        pred[data.train_mask] = data.y.squeeze()[data.train_mask]
        out_prob[~pseudo_mask] = 0
        A = None

        out_pseudo = out.clone().log_softmax(-1)
        out_pseudo[~pseudo_mask] = 0.
        
        cache_pseudo.append((merge_mask, pred, pseudo_mask, out_prob, A, out_pseudo))

        pbar.set_description(f'Pseudo inference epoch: {epoch:03d}')
        pbar.update(1)
        it += 1
        model.reset_vars(it)

    pbar.close()
    return cache_pseudo


def train(epoch, train_loader, model, optimizer, pseudo=None):
    model.train()

    total_bat = len(train_loader)
    pbar = tqdm(total=total_bat)

    total_loss = l_examples = 0
    total_train = 0

    scaler = torch.cuda.amp.GradScaler()
    all_x = []

    idxs = torch.randperm(total_bat).tolist()
    kldiv = nn.KLDivLoss(reduction="mean")

    for i, data in enumerate(train_loader):
        #data = train_loader[i]
        data["inductive"] = args.type=="ind"
        data = data.cuda() 

        train_mask = data.train_mask.clone()
        y = data.y.squeeze().clone()    

        train_mask_sum = train_mask.sum()
        

        ps_mask = None
        
        if not pseudo == None:
            __train_mask_ps, __y, __ps_mask, probs, ps_A, only_pseudo = pseudo[i]
            data["pseudo"] = only_pseudo.clone()

        total_train += train_mask_sum


        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out, x_, edges = model(data, epoch, i)
            lbl_nodes = out[train_mask]

            # Compute loss 
            loss = 0
            if data.inductive == False:
                loss = cond_entropy(out[~train_mask]) * args.c_entropy

            if train_mask.sum() > 0:
                loss = F.nll_loss(lbl_nodes.log_softmax(-1), y[train_mask]) + loss 

        # Calculate gradients
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        scaler.step(optimizer)
        scaler.update()
        
        model.reset_vars(i)
        optimizer.zero_grad(set_to_none=True)

        num_examples = data.x.shape[0] #train_mask.sum().item()
        total_loss += loss.item() * num_examples
        l_examples += num_examples

        pbar.set_description(f'Training epoch: {epoch:03d} Loss: {loss.item():03f}')
        pbar.update(1)

        all_x.append(x_.detach().cpu())


    pbar.set_description(f'Training epoch: {epoch:03d} Avg. Loss: {(total_loss/l_examples):03f}')
    pbar.close()

    return total_loss / l_examples, all_x, edges
 

def build_cluster(data, diff=False, path=None):
    if diff: # Graph Adjacency diffusion
        # Apply diffusion augmentation on adjacency matrix
        num_edges = data.edge_index.shape[1]
        old_index = data.edge_index.clone()
        data.edge_attr = None

        gdc = GDC(self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05, eps=1e-4),
            sparsification_kwargs=dict(method='topk', k=20, avg_degree=6, dim=1),
            exact=True)

        data = gdc(data)
        data["o_index"] = old_index
        print(f"Difussion: prev {num_edges}; current {data.edge_index.shape[1]}")
            
    # Calculate edge homophily for given dataset
    h = homophily(data.edge_index, data.y)
    print("Data Homophily: ", h)
    path = None

    cluster_data = ClusterData(data, num_parts=args.num_clusters, recursive=False, save_dir=path)
    train_loader = ClusterLoader(cluster_data, batch_size=args.bs, shuffle=False, num_workers=args.j)
    test_loader = train_loader

    return train_loader, test_loader, data

def get_load_dataset(seed, split_i = 0, x_path=None, x_pred_p=None):
    # Load ataset
    if args.dataset.lower() in ['flickr', 'reddit']:
        args.type = "ind"

    is_ogbn = False
    evaluator = None
    using_loader = True

    if "ogbn" in args.dataset:
        is_ogbn = True
        dataset = get_dataset_node_OGB(args.dataset)
        split_idx = dataset.get_idx_split()    
        
        # Split dataset with corresponing masks
        for split in ['train', 'valid', 'test']:
            dataset.data[f'{split}_mask'] = index_to_mask(split_idx[split], dataset.data.y.shape[0])

        # Create evaluator based on OGBN dataset (node classification)
        evaluator = NEvaluator(args.dataset)

    elif args.dataset.lower() in ['flickr', 'reddit', 'photo', 'cora', 'citeseer', 'pubmed', 'computers', 'texas', 'wisconsin', 'cornell', 'coauthor']:
        dataset = get_dataset_planetoid(args.dataset, data_seed=seed, split_i=split_i)

    evaluator = NEvaluator("ogbn-arxiv")

    data = dataset.data
    data.glob_idx = torch.arange(data.x.shape[0])
    data.num_cls = dataset.num_classes
    num_nodes = data.x.shape[0]

    if not hasattr(data, "valid_mask"):
        data["valid_mask"] = data.val_mask.clone() 

    if args.embed_feat and is_ogbn:
        print("Loading external node embeddings...")

        # Delete original features from loaded dataset
        del data.x
        
        # Distill-RoBERTa sentence embeddings (1024 dim)
        if x_path == None:
            embed_path = os.path.join("/media/Patrik/data", args.dataset, args.dataset.replace("-", "_"), "embeddings/X.all.roberta-large-emb.pt")
            data.x = torch.load(embed_path, "cuda").detach().cpu()
        else:
            embed_path = x_path
            data.x = torch.from_numpy(np.array(
                        np.memmap(x_path, mode='r',
                                dtype=np.float16,
                                shape=(num_nodes, 768)))
                     ).to(torch.float32).cpu()
            
        if x_pred_p != None:
            data["p_pred"] = torch.from_numpy(np.array(
                        np.memmap(x_pred_p, mode='r',
                                dtype=np.float16,
                                shape=(num_nodes, data.num_cls)))
                     ).to(torch.float32).cpu()
            

        print(f"\tPath: {embed_path}")
        print(data.x)

    num_cls = dataset.num_classes
    feat_dim = data.x.shape[-1] #embed.shape[-1]
    data["num_nodes"] = data.x.shape[0] #embed.shape[0]

    print(f"Original features dim: {feat_dim}")
    print(f"Total number of classes: {num_cls}")

    # Get loader/sampler for loading sub-graphs (mini-batches)
    if args.num_clusters > 1:
        if args.loader == "random":
            train_loader = RandomNodeSampler(data, num_parts=args.num_subgraphs, shuffle=True, num_workers=args.j)
            #test_loader = RandomNodeSampler(data, num_parts=args.num_subgraphs_test, shuffle=False, num_workers=args.j)
            test_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=256, num_workers=args.j, shuffle=False)
        elif args.loader == "imbalanced":  
            sampler = ImbalancedSampler(data, data.train_mask)
            train_loader = NeighborLoader(data, input_nodes=data.train_mask, batch_size=64, num_neighbors=[-1, -1], sampler=sampler, num_workers=args.j)
        elif args.loader == "neigh":
            train_loader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[3, 3, 3], batch_size=256, num_workers=args.j, shuffle=False, directed=True, pin_memory=True)
            #test_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=16, num_workers=args.j, directed=True, shuffle=False)
        elif args.loader == "cluster":
            path = "/media/Patrik/data/prod_clusters/" 
            train_loader, test_loader, _ = build_cluster(data, False, path)
        elif args.loader == "shadow":
            train_loader = ShaDowKHopSampler(data, 3, 20, data.train_mask, batch_size=64, num_workers=args.j)
            test_loader  = ShaDowKHopSampler(data, 3, 20, data.train_mask, batch_size=256, num_workers=args.j)
    else:
        train_loader = test_loader = [data]

    loaders = []
    max_nodes = -1
    
    for data in train_loader:
        max_nodes = data.y.shape[0] if data.y.shape[0] > max_nodes else max_nodes
        #print(f"Train: {data.train_mask.sum()}, Test: {data.test_mask.sum()}, Valid: {data.valid_mask.sum()}")

        n_nodes, n_edges = data.x.shape[0], data.edge_index.shape[1]
        s, t = data.edge_index[0], data.edge_index[1]
        edge_dict = torch.sparse_coo_tensor(indices=data.edge_index, values=torch.arange(n_edges), size=(n_nodes, n_nodes)).to_dense()
        data["rev"] = edge_dict[t, s]
        #print("REV: ", data.edge_index.shape, data.rev.shape)

        loaders.append(data.clone())
  
    #print(f"Max nodes: {max_nodes}")
    train_loader = test_loader = loaders

    return train_loader, train_loader, evaluator, is_ogbn, feat_dim, num_cls, max_nodes

def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    print(f"PRNG seed({seed}) set...")


if __name__ == '__main__':
    seed = args.seed
    set_seed(seed)
    wc = None 

    acc_list = []
    best_test = best_valid = corr_train = corr_test = 0.0
    best_it = 0

    pseudo_threshold = .92
    if "arxiv" in args.dataset:
        pseudo_threshold = 0.85 #orig 0.83
    elif "products" in args.dataset:
        pseudo_threshold = 0.8

    args.max_runs = 10
    splits_num = 1  #1=ARxiv;  Planetoid=100

    x_embed_path = None # f"{llmt.ckpt_dir}.emb"
    x_embed_pred = None #f"{llmt.ckpt_dir}.pred"
    splits = [get_load_dataset(0, ii, x_embed_path, x_embed_pred) for ii in range(splits_num)]
    #split_seed = [random.randint(0, 1e6) for _ in range(splits_num)]
    runs_seed =  [random.randint(0, 1e3) for _ in range(args.max_runs)]

    
    start_split = 0 #4 56
    for its in range(start_split, splits_num):
        train_loader, test_loader, evaluator, is_ogbn, feat_dim, num_cls, clN = splits[its] #get_load_dataset(split_seed[its], its)
        
        start_step = 0
        for i in range(start_step, args.max_runs):  
            set_seed(runs_seed[i])

            ll = str(i)+ "_layer4"
            log_dir = os.path.join(args.log_dir, args.dataset, ll)

            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            writer = SummaryWriter(log_dir)
            saved_checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset)

            if not os.path.exists(saved_checkpoint_dir):
                os.makedirs(saved_checkpoint_dir)

            # Creating the model 
            #model = HSACRF(feat_dim, num_cls, diff_T=args.T, crf_it=args.crf_it, hidden_feat=args.hidden_dim, out_hidden=args.hidden_dim, num_layers=2, num_heads=args.mhsa_heads)
            model = RNNCRF(feat_dim, num_cls, diff_T=args.T, crf_it=args.crf_it, 
                        hidden_feat=args.hidden_dim, num_layers=2, dropout=args.dropout, 
                        max_nodes=clN, damping=args.damping)

            if not args.pretrained == "":
                model.load_pretrained(args.pretrained)
            else:
                print("Model training from scratch...")

            model = model.cuda()
            model.train()

            print(f"Number of learnable parameters: {count_parameters(model)}")

            
            if args.optim == "adam":
                #optimizer = topt.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay, use_accelerated_op=True)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay, fused=True)
            elif args.optim == "adamx":
                optimizer = torch.optim.Adamax(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
            elif args.optim == "rmsprop":
                optimizer = torch.optim.RMSprop(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
            elif args.optim == "adamw":
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
            elif args.optim == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
            elif args.optim == "adams":
                optimizer = torch.optim.NAdam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
            elif args.optim == "adagrad":
                optimizer = torch.optim.Adagrad(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)

            # Reset iteration best performance and pseudo labels
            it_best_test = 0
            best_val_it = 0
            pseudo_cache = None

            if args.rewire > 0:    
                if i == start_step:
                    train_l_orig = train_loader.copy()
                train_loader = train_l_orig.copy()

            best_pred = None
            for epoch in range(args.epoches + 1):     
                train_loss, x_, edges = train(epoch, train_loader, model, optimizer, pseudo_cache)
                has_none = False

                if (epoch) % args.log_interval == 0:
                    train_acc, val_acc, test_acc, loss_test, pred, widx = test_it(epoch, model, train_loader, evaluator, pseudo_cache, is_ogbn)
                    
                    has_pseudo_infer = False
                    force_pseudo = False

                    if test_acc > best_test:
                        best_test = test_acc
                        best_pred = pred

                    if test_acc > it_best_test:
                        it_best_test = test_acc
                        best_it = epoch       
                    
                    if val_acc > best_valid:
                        best_valid = val_acc

                    if val_acc > best_val_it:
                        best_val_it = val_acc
                        has_pseudo_infer = val_acc > args.pseudo

                    if epoch > 150 and pseudo_cache == None:
                        has_pseudo_infer = True

                    #scheduler.step(val_acc)
                    print("Epoch: %d, train loss: %f, train_acc: %f, val acc: %f, test_acc: %f, best_test: %f / %f (it/total), best_val: %f / %f (it/total) (%d it / %d run) "
                        %(epoch, train_loss, train_acc, val_acc, test_acc, it_best_test, best_test, best_val_it, best_valid,  best_it, i))

                    # Infer new pseudo-labels 
                    if args.pseudo > 0. and has_pseudo_infer:   #products = 0.8; arxiv = 0.72(ROBERTA)
                        pseudo_cache = None
                        pseudo_cache = infer_pseudo(epoch, train_loader, model, pseudo_threshold, pseudo=pseudo_cache) # arxiv = 0.83

                    writer.add_scalars('loss', {'train_loss': train_loss, 'test_loss': loss_test}, epoch)
                    writer.add_scalars('accuracy', {'train_acc': train_acc,'val_acc': val_acc, 'test_acc': test_acc}, epoch)
                else:
                    print("Epoch: %d, train loss: %f, best_test: %f, best_val: %f (%d it / %d run)" % (epoch, train_loss, best_test, best_valid, best_it, i))
                if args.rewire > 0 and epoch >= 10 and (epoch % 10 == 0):
                    for dt in range(len(train_loader)):
                        train_loader[dt] = None

                        tt = apply_feat_KNN(train_l_orig[dt].clone().cuda(), x_[dt], args.rewire, edges)
                        print(f"\tre-wiring... {tt.edge_index.shape}")

                        train_loader[dt] = tt

            w_test = str(it_best_test) + ";"
            path_w = "./" + args.name_acc + ".txt"
            with open(path_w, "a") as myfile:
                myfile.write(w_test)

            acc_list.append(it_best_test)
            writer.close()

            acc_list_ = np.asarray(acc_list)
            print("===========Stats===========")
            print("Split: ", its, "Run: ", i)
            print("Mean test acc: ", np.mean(acc_list_), "+-", np.std(acc_list_)) 
            print("Best test acc ", np.max(acc_list_))
            print(acc_list_)
            print("===========================")
