import math
import torch
from torch import Tensor
import torch.nn as nn

import torch.nn.functional as F


# Multi-Head Self-Attention with adjacency matrix masking.
class HMHSAVar(nn.Module):
    def __init__(
        self,
        in_feat,
        num_heads,
        embed_dim = None,
        attn_dropout = 0.1,
        bias = True,
        proj_out_dim = None,
        init_w = "xavier",
        *args,
        **kwargs
    ): 
        super().__init__()

        embed_dim = in_feat if embed_dim == None else embed_dim
        #embed_dim = embed_dim * num_heads
        
        if embed_dim % num_heads != 0:
            print(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )
            
        self.in_feat = in_feat
        self.proj_out_dim = embed_dim if proj_out_dim == None else proj_out_dim 

        self.attn_prob = attn_dropout

        self.out_proj = nn.Linear(in_features=embed_dim, out_features=self.proj_out_dim, bias=False)
        dim_q = dim_k = dim_v = embed_dim

        self.Q = nn.Linear(in_feat, dim_q, False)
        nn.init.xavier_uniform_(self.Q.weight, gain=1 / math.sqrt(2))
        #nn.init.xavier_normal_(self.Q.weight, gain=gain)

        self.K = nn.Linear(in_feat, dim_k, False)
        #nn.init.xavier_normal_(self.K.weight, gain=gain)
        nn.init.xavier_uniform_(self.K.weight, gain=1 / math.sqrt(2))

        self.V = nn.Linear(in_feat, dim_v, False)
        #nn.init.xavier_normal_(self.V.weight, gain=gain)
        nn.init.xavier_uniform_(self.V.weight, gain=1 / math.sqrt(2))

        

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(proj_out_dim)

        self.energy_func = nn.Linear(self.head_dim, 1) 
 
    def forward(self, x: Tensor, adj: Tensor = None, ret_attn = False, x_k=None):
        # [N, P, C]
        n_patches, in_channels = x.shape

        if x_k == None:
            x_k = x
            n_kv = n_patches
        else:
            n_kv = x_k.shape[0]
        
        Q = self.Q(x).reshape(n_patches, self.num_heads, -1).contiguous().permute(1, 0, 2)
        K = self.K(x_k).reshape(n_kv, self.num_heads, -1).contiguous().permute(1, 0, 2)
        V = self.V(x_k).reshape(n_kv, self.num_heads, -1).contiguous().permute(1, 0, 2)

        # Calculate scaled dot-product
        attn = torch.matmul(Q, K.transpose(-1, -2)) * self.scaling + self.energy_func(Q)

        # Apply adjacency matrix to pair-wise attention scores
        if not adj == None:
            attn = attn + torch.where(adj>0, 0., float("-inf"))
        #    #attn[:, range(attn.shape[-1]), range(attn.shape[-1])] = self.EPS

        # Calculate prob.
        attn = self.softmax(attn)
        attn = torch.eq(attn, torch.max(attn, dim=-1)[0].unsqueeze(-1)).to(torch.float32)
        attn = attn / self.num_heads
        #attn = F.dropout(attn, p=self.attn_prob, training=self.training)

        # Weighted sum
        out = torch.matmul(attn, V)
        out = out.transpose(0, 1).reshape(n_patches, -1)

        if ret_attn == False:
            return out

        attn = attn.sum(dim=0)

        return out, attn