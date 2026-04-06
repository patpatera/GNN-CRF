import torch
import torch.nn as nn
import torch.nn.functional as F

#PyG packages
from torch_geometric.nn import MLP
from torch_geometric.utils import mask_to_index, k_hop_subgraph, index_to_mask, dropout_adj
import torch_geometric.transforms as T

#My implementation
from hmhsa_layer import HMHSA
from utils.graph_utils import extract_adj_mat

from torch_cluster import knn_graph, radius_graph


class HSACRF2(nn.Module):

    def __init__(self, in_feat, num_classes, crf_it=5, hidden_feat=128, out_hidden=128, num_layers=3, dropout=0.5, attn_dropout=0.1, num_heads=4):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.crf_it = list(range(crf_it))


        # Embedding projection to lower dimension
        self.transform = in_feat > 256
        self.embed_feat = nn.Linear(in_feat, hidden_feat, bias=False)

        #self.pair_pred = nn.Linear(hidden_feat, num_classes)
        self.pair_pred = MLP(in_channels=hidden_feat, hidden_channels=256, out_channels=num_classes, num_layers=3, dropout=dropout, norm="LayerNorm")

        in_feat = hidden_feat if self.transform else in_feat
        out_hidden = in_feat

        # MHSA pair-wise energy modules
        self.MHSA = HMHSA(in_feat=in_feat, num_heads=num_heads, attn_dropout=attn_dropout, proj_out_dim=out_hidden)
        self.pair_pred = nn.Linear(out_hidden, num_classes, bias=False)
        
        self.MHSA2 = HMHSA(in_feat=768, num_heads=4, attn_dropout=attn_dropout, proj_out_dim=256)

        # Unary energy parameters (linear projection)
        #self.unary = MLP(in_channels=in_feat, hidden_channels=256, out_channels=num_classes, num_layers=2, dropout=dropout, norm="LayerNorm")
        self.unary = nn.Linear(in_feat, num_classes, bias=False)
        
        # Learnable parameters for label compatibility 
        self.comp_matrix = nn.Parameter(-1*torch.eye(num_classes))
        self.weights = nn.Parameter(torch.eye(num_classes))

        # Softmax and layer normalisation
        self.layer_norm = nn.LayerNorm(in_feat)
        self.softmax = nn.Softmax(dim=-1)


    def MFI_pairwise(self, data_all, dataloader, optimizer, pbar, epoch):
        for data in dataloader:
            data = data.cuda()

            if data.train_mask.sum() == 0:
                continue

            optimizer.zero_grad()

            # Get GT labels for training nodes
            y = data.y.squeeze(1)[data.train_mask]

            out = self.train_pairwise(data)

            loss = F.nll_loss(out, y) 
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.parameters(), 2.0)

            optimizer.step()

            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            l_examples += num_examples


            pbar.set_description(f'Training epoch: {epoch:03d} Loss: {loss.item():03f}')
            pbar.update(1)

        return self.pair_pred(data_all)

    def MFI(self, data, dataloader, optimizer, pbar, iter=5):
        mfi_iter = list(range(iter))

        U = self.unary(data.x)
        Q = U

        # Mean-field Inference
        for it in mfi_iter:
            # Q normalization
            #Q = self.softmax(Q)

            pair = self.MFI_pairwise(data.x, dataloader, optimizer, pbar)
        
            # Weighting filter for unary energy 
            pair = pair @ self.weights               
            
            # Compability transform 
            pair = pair @ self.comp_matrix

            # Unary addition 
            Q = U - pair 

        self.grads.append((U.detach().clone(), pair.detach().clone()))

    def train_pairwise(self, data, epoch=0):
        num_nodes = data.x.shape[0]
        x1 = data.x

        # Extract dense adjacency matrix
        data["A"] = extract_adj_mat(data.edge_index, num_nodes, to_undirected=True, add_self_loops=True)
        A = torch.where(data.A>0, 0., float("-inf"))

        if self.transform:
            x1 = self.embed_feat(x1)

        x = self.MHSA(x, A, norm=False, ret_attn=False)

        x = self.layer_norm(x + x1)

        y = self.pair_pred(x)
        y = torch.log_softmax(y, dim=-1) 

        return y

