import math
import torch
from torch import Tensor
import torch.nn as nn

import torch.nn.functional as F

from utils.energy import hopfield_energy

# Multi-Head Self-Attention with adjacency matrix masking.
class HMHSAEnergy(nn.Module):
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

        const_val = 1e-5

        self.Q = nn.Linear(in_feat, dim_q, False)
        nn.init.constant_(self.Q.weight, const_val)
        #nn.init.orthogonal_(self.Q.weight)

        self.K = nn.Linear(in_feat, dim_k, False)
        nn.init.constant_(self.K.weight, const_val)
        #nn.init.orthogonal_(self.K.weight)

        self.out_proj = nn.Linear(embed_dim, self.proj_out_dim)
        #nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)
        nn.init.constant_(self.out_proj.weight, const_val)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.num_heads = num_heads
        self.embed_dim = embed_dim


 
    def forward(self, x: Tensor, adj: Tensor = None, ret_attn = False, x_k=None, scores=None):
        # [N, P, C]
        n_patches, in_channels = x.shape

        if x_k == None:
            x_k = x
            n_kv = n_patches
        else:
            n_kv = x_k.shape[0]
        
        Q = self.Q(x).reshape(n_patches, self.num_heads, -1).contiguous().permute(1, 0, 2)
        K = self.K(x_k).reshape(n_kv, self.num_heads, -1).contiguous().permute(1, 0, 2)

        # Just one iteration
        energies, attn = hopfield_energy(Q, K, self.scaling, mask=adj.bool())
        grad_queries = torch.autograd.grad(
            energies, Q, grad_outputs=torch.ones_like(energies),
            create_graph=self.training,  # enables double backprop for optimization
        )[0]

        out = Q - 1.0 * grad_queries
        out = out.transpose(0, 1).reshape(n_patches, -1)


        #return out

        attn = attn.sum(dim=0) / self.num_heads 

        return out, attn, energies
