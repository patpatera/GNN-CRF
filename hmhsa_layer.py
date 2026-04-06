import math
import torch
from torch import Tensor
import torch.nn as nn
import torch_sparse

import torch.nn.functional as F
from utils.graph_utils import to_dense_adj
from torch_geometric.utils import softmax

def hook_fn(m, i, o):
    print(m)
    print("------------Input Grad------------")
    for grad in i:
        try:
            print(grad.shape)
            print(grad.mean(), grad.max(), grad.min())

        except AttributeError: 
            print ("None found for Gradient")
    print("------------Output Grad------------")
    for grad in o:
        try:
            print(grad.shape)
            print(grad.mean(), grad.max(), grad.min())

        except AttributeError: 
            print ("None found for Gradient")
    print("\n")

# Multi-Head Self-Attention with adjacency matrix masking.
class HMHSA(nn.Module):
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
        #embed_dim = 64 * num_heads
        #embed_dim = embed_dim * 2
        
        if embed_dim % num_heads != 0:
            print(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )
            
        self.in_feat = in_feat
        self.proj_out_dim = embed_dim if proj_out_dim == None else proj_out_dim 

        self.attn_prob = attn_dropout

        dim_q = dim_k = dim_v = embed_dim
        
        const_val = 1e-5

        self.Q = nn.Linear(in_feat, dim_q)
        #nn.init.xavier_uniform_(self.Q.weight, gain=1 / math.sqrt(2))
        #nn.init.xavier_normal_(self.Q.weight)
        nn.init.constant_(self.Q.bias, 0.)
        #nn.init.constant_(self.Q.weight, const_val)
        #nn.init.orthogonal_(self.Q.weight)

        self.K = nn.Linear(in_feat, dim_k)
        #nn.init.xavier_normal_(self.K.weight)
        nn.init.constant_(self.K.bias, 0.)
        #nn.init.xavier_uniform_(self.K.weight, gain=1 / math.sqrt(2))
        #nn.init.constant_(self.K.weight, const_val)
        #nn.init.orthogonal_(self.K.weight)

        self.V = nn.Linear(in_feat, dim_v)
        #nn.init.xavier_normal_(self.V.weight)
        nn.init.constant_(self.V.bias, 0.)
        #nn.init.xavier_uniform_(self.V.weight, gain=1 / math.sqrt(2))
        #nn.init.constant_(self.V.weight, const_val)
        #nn.init.orthogonal_(self.V.weight)
        

        self.out_proj = nn.Linear(embed_dim, self.proj_out_dim)
        #nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)
        #nn.init.constant_(self.out_proj.weight, const_val)
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


 
    def forward(self, x: Tensor, adj: Tensor = None, ret_attn = False, x_k=None, only_attn=False, scores=None):
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
    
        if self.num_heads == 1:
            Q, K, V = Q.squeeze(0), K.squeeze(0), V.squeeze(0)

        # Calculate scaled dot-product
        attn_ = Q @ K.transpose(-1, -2) * self.scaling 
    
        if only_attn:
            adj = None

        # Apply adjacency matrix to pair-wise attention scores
        if adj == None:
            attn = attn_ 
        else: 
            #attn_ = attn_ * adj
            attn = attn_ + torch.where(adj>0, 0., float("-inf"))
            

        #if not scores == None:
        #    attn = attn + scores

        # Calculate prob.
        attn = self.softmax(attn)
        attn = F.dropout(attn, p=self.attn_prob, training=self.training)

        if only_attn:
            if self.num_heads > 1:
                attn = attn.sum(dim=0) / self.num_heads 
            return attn
        
        # Weighted sum
        out = attn @ V

        if self.num_heads > 1:
            out = out.transpose(0, 1).reshape(n_patches, -1)
            

        out = self.out_proj(out)

        #out = self.layer_norm(out + x)

        if ret_attn == False:
            return out

        if self.num_heads > 1:
            attn = attn.mean(dim=0)

        return out, attn, attn_.mean(0)
    
    def filter_attn(self, A, q_num):
        A[range(A.shape[0]), range(A.shape[0])] = 0.
        #A = F.relu(A)
        th = torch.quantile(A, q_num, dim=1, keepdim=True)

        mask = A >= th
        A[mask] = 1.
        A[~mask] = 0.

        A = A + A.T
        A = A + torch.eye(A.shape[0]).cuda()
        A[A>0] = 1.

        return A
    
    def get_l_hops(self, A, _A):
        A_l =  torch.matmul(A, _A) + A
 
        # Add self-loops
        A_l[range(A.size()[0]), range(A.size()[0])] = 1
        
        # Convert to [0;1] -- adjacency representation
        A_l[A_l>0] = 1.

        return A_l
    

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