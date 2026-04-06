import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

#PyG packages
from torch_geometric.utils import degree, sort_edge_index, scatter, homophily, dropout_adj, get_laplacian, add_remaining_self_loops
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn import MLP

import torch_geometric.transforms as T

#My implementation
from RWKV.CRFRWKVEnergy2 import CRFRWKV

from utils.diffusion import DiffusionWrapper
from utils.graph_utils import extract_adj_mat, get_isolated_mask, get_border_nodes_mask, rewire_attn, rand_global_edges, mixup, comp_to_edges


from utils.half_hop import HalfHop


class RNNCRF(nn.Module):

    def __init__(self, in_feat, num_classes, diff_T=1, crf_it=5, 
                 hidden_feat=128, num_layers=3, dropout=0.5, max_nodes=-1, damping=0.5):
        super().__init__()
        
        self.damping = damping
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.in_dropout = dropout
        self.cft_T = crf_it
        self.crf_it = list(range(crf_it))
        
        # Diffusion time (number of steps)
        self.diff_T = diff_T

        # Embedding projection to lower dimension
        self.transform = True #in_feat > 256

        # Transform embeddings
        #$self.embed_feat = MLP(in_channels=in_feat, hidden_channels=hidden_feat*2, out_channels=hidden_feat, norm=None, act="tanh", num_layers=3, dropout=0.1, plain_last=True)  
        self.embed_feat = nn.Linear(in_feat, hidden_feat, bias=False)
        self.decoder = nn.Linear(hidden_feat, num_classes, bias=False)

        self.CRF_RWKV = CRFRWKV(1, hidden_feat, max_len=max_nodes, num_cls=num_classes, damping=damping)

        
        in_feat = hidden_feat if self.transform else in_feat
        out_hidden = in_feat

        self.in_feat = in_feat

    
    def get_rw_adj(self, edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                    device=edge_index.device)

        if not fill_value == 0:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        indices = row if norm_dim == 0 else col
        deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-1)
        edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
        return edge_index, edge_weight
    

    def reset_vars(self, it):
        self.CRF_RWKV.reset_vars(it)

    def forward(self, data, it=None, c_idx=0, ret_attn=False):
        x1 = data.x  

        # Input dropout
        x1 = F.dropout(x1, 0.1, self.training)    #Cora=0.7; Pubmed=0.5;

        # Normalization text embeddings
        #x1 = F.normalize(x1, p=2.)

        x1 = self.embed_feat(x1)       

        # Calculate CRF-RWKV
        x_nodes, Q, edges = self.CRF_RWKV(x1, data, c_idx)

        x_embed = F.normalize(x1, p=2.)

        return Q, x_embed, edges
    