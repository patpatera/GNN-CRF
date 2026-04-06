import math
import torch
from torch import Tensor
import torch.nn as nn
import torch_sparse

import torch.nn.functional as F
from torch_geometric.utils import softmax


# Multi-Head Self-Attention with adjacency matrix masking.
class HMHSAGraph(nn.Module):
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

        in_feat = in_feat 
        embed_dim = in_feat if embed_dim == None else embed_dim

        if embed_dim % num_heads != 0:
            print(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )
            
        self.in_feat = in_feat
        self.proj_out_dim = in_feat #embed_dim if proj_out_dim == None else proj_out_dim 

        self.attn_prob = attn_dropout

        dim_q = dim_k = dim_v = embed_dim
        
        const_val = 1e-5

        self.Q = nn.Linear(in_feat, dim_q)
        #nn.init.xavier_uniform_(self.Q.weight, gain=1 / math.sqrt(2))
        #nn.init.xavier_normal_(self.Q.weight)
        nn.init.constant_(self.Q.bias, 0.)
        nn.init.constant_(self.Q.weight, const_val)
        #nn.init.orthogonal_(self.Q.weight)

        self.K = nn.Linear(in_feat, dim_k)
        #nn.init.xavier_normal_(self.K.weight)
        nn.init.constant_(self.K.bias, 0.)
        #nn.init.xavier_uniform_(self.K.weight, gain=1 / math.sqrt(2))
        nn.init.constant_(self.K.weight, const_val)
        #nn.init.orthogonal_(self.K.weight)

        self.V = nn.Linear(in_feat, dim_v)
        #nn.init.xavier_normal_(self.V.weight)
        nn.init.constant_(self.V.bias, 0.)
        #nn.init.xavier_uniform_(self.V.weight, gain=1 / math.sqrt(2))
        nn.init.constant_(self.V.weight, const_val)
        #nn.init.orthogonal_(self.V.weight)
        

        self.out_proj = nn.Linear(embed_dim, self.proj_out_dim)
        #nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)
        nn.init.constant_(self.out_proj.weight, const_val)
        #nn.init.orthogonal_(self.out_proj.weight)
        #nn.init.xavier_uniform_(self.out_proj.weight, gain=1 / math.sqrt(2))

        
        #self.Q.register_full_backward_hook(hook_fn)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.num_heads = num_heads
        self.embed_dim = embed_dim

        #self.energy_bias = nn.Linear(self.head_dim, 1)

        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(in_feat)

        self.sqdim = self.head_dim**0.5
        self.EPS = -1e20


 
    def forward(self, x: Tensor, adj: Tensor = None, norm_idx=1, ret_attn=True):
        edge = adj
        # [N, P, C]
        n_patches, in_channels = x.shape

        Q = self.Q(x).reshape(n_patches, self.num_heads, -1).contiguous()
        K = self.K(x).reshape(n_patches, self.num_heads, -1).contiguous()
        V = self.V(x).reshape(n_patches, self.num_heads, -1).contiguous()
    

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        src = Q[edge[0, :], :, :]
        dst_k = K[edge[1, :], :, :]

        prods = torch.sum(src * dst_k, dim=1) * self.scaling
        attention = softmax(prods, edge[norm_idx])
        attention = attention.mean(1)

        #vx = torch.mean(torch.stack(
        #  [torch_sparse.spmm(self.edge_index, attention[:, idx], v.shape[0], v.shape[0], v[:, :, idx]) for idx in
        #  range(self.opt['heads'])], dim=0),
        #dim=0)
        
        # Weighted sum
        V = V.reshape(n_patches, -1)
        out =  torch_sparse.spmm(edge, attention, x.shape[0], x.shape[0], V)

        return out, attention, attention


from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr
from torch_geometric.utils.num_nodes import maybe_num_nodes

# https://twitter.com/jon_barron/status/1387167648669048833?s=12
# @torch.jit.script
def squareplus(src: Tensor, index: Optional[Tensor], ptr: Optional[Tensor] = None,
               num_nodes: Optional[int] = None) -> Tensor:
  r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
  out = src - src.max()
  # out = out.exp()
  out = (out + torch.sqrt(out ** 2 + 4)) / 2

  if ptr is not None:
    out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
  elif index is not None:
    N = maybe_num_nodes(index, num_nodes)
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
  else:
    raise NotImplementedError

  return out / (out_sum + 1e-16)