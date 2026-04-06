
from typing import Any, Dict, List, Optional, Union
import numpy as np
import math, types, copy, sys, os
import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from torch_geometric.nn.aggr import Aggregation

#from ..utils.graph_utils import spmm
import torch_sparse


#PyG
import torch_geometric.nn as pygnn 
import torch_geometric.utils as upyg 

RUN_DEVICE = 'cuda'


RWKV_K_CLAMP = 60 # e^60 => 1e26

RWKV_EPS = 1e-20

class ChannelMixingModule(nn.Module):

    def __init__(self, n_embd, num_cls) -> None:
        super().__init__()
        #n_embd = num_cls
        self.n_embd = n_embd
        self.num_cls = num_cls

        # Channel-mixing variables
        self.key = nn.Linear(num_cls, num_cls, bias=False)
        self.receptance = nn.Linear(num_cls, num_cls, bias=False)
        self.value = nn.Linear(num_cls, num_cls, bias=False)

        #self.output = nn.Linear(num_cls, num_cls**2, bias=False)

        self.time_mix_k = nn.Parameter(torch.ones(1, num_cls))
        self.time_mix_r = nn.Parameter(torch.ones(1, num_cls)) 

        self.init_rnn_vars()

    def init_rnn_vars(self):
        self.prev_x = 0

    def forward(self, x):
        xk = x * self.time_mix_k + self.prev_x * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + self.prev_x * (1 - self.time_mix_r)
        self.prev_x = x

        k = self.key(xk)    

        r = torch.sigmoid(self.receptance(xr))
        #k = torch.relu(k)
        #k = torch.square(F.leaky_relu(k,0.1))
        #k = torch.square(F.relu(k)) 
        k = F.leaky_relu(k, 0.1)
   
        kv = self.value(k)
        out = r * kv

        return out 

class TimeMixingModule(nn.Module):

    def __init__(self, n_embd, num_cls) -> None:
        super().__init__()

        self.n_embd = n_embd
        self.num_cls = num_cls

        # Time-mixing variables
        self.key = nn.Linear(num_cls, num_cls, bias=False)
        self.value = nn.Linear(num_cls, num_cls, bias=False)
        self.receptance = nn.Linear(num_cls, num_cls, bias=False)

        self.time_mix_k = nn.Parameter(torch.ones(1, num_cls))
        self.time_mix_v = nn.Parameter(torch.ones(1, num_cls))
        self.time_mix_r = nn.Parameter(torch.ones(1, num_cls))


        self.output = nn.Linear(num_cls, num_cls, bias=False)

        self.time_w = nn.Parameter(torch.ones(1, num_cls))
        #nn.init.constant(self.time_w, 1e-5)

        self.time_wc = nn.Parameter(torch.ones(1, num_cls))
        #nn.init.constant(self.time_w, 1e-5)

        #self.init_rnn_vars()

    def init_rnn_vars(self, dim):
        self.ah = 0 #torch.zeros(self.num_cls, device=RUN_DEVICE)
        self.bh = 0 #torch.zeros(self.num_cls, device=RUN_DEVICE)

        self.prev_x = 0 #torch.zeros(self.num_cls, device=RUN_DEVICE)  

    def forward(self, x, s, t, i, T):
        xk = x * self.time_mix_k + self.prev_x * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + self.prev_x * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + self.prev_x * (1 - self.time_mix_r)
        self.prev_x = x

        k = self.key(xk)[s]
        v = self.value(xv)[t]

        #xr = torch.hstack([xr[s], xr[t]])
        r = self.receptance(xr)[s]
        r = torch.sigmoid(r)

        k = k.clamp(max=RWKV_K_CLAMP)
        k = k.exp()
        kv = k * v

        #self.time_w =  torch.cat(
        #    [self.time_decay * self.time_curve, self.time_first], dim=-1).transpose(-1, -2)

        # Propagete and aggregate KV and K values from connected nodes 
        w = (-(T - i - 1) * self.time_w).exp()
        kv = w * kv
        v  = w * v
        
        # Add to sequence (sums in WKV)
        a = self.ah + kv
        b = self.bh + k

        self.ah = self.ah + kv 
        self.bh = self.bh + k
    
        wkv = (a / (b + RWKV_EPS)) 
        
        #msg_wkv = torch.zeros_like(x, device=r.device)
        #upyg.scatter(wkv, index=t, dim=0, out=msg_wkv, reduce="add") + x

        rwkv = r * wkv

        #msg_wkv = torch.tanh(msg + msg_wkv)
        #out = self.output((1 - self.time_mix_rwkv) * msg_wkv + self.time_mix_rwkv * msg)

        out = self.output(rwkv)#.view(-1, self.num_cls, self.num_cls)
        return out


class CRFRWKV(nn.Module):

    def __init__(self, num_layers, n_embd, max_len, num_cls=5) -> None:
        #super().__init__(aggr=pygnn.aggr.MeanAggregation(),   flow="source_to_target")
        super().__init__()
        
        num_layers = 5
        
        self.num_layers = num_layers
        self.n_embd = n_embd
        self.max_len = max_len
        self.num_cls = num_cls

        self.blocks = nn.ModuleList([TimeMixingModule(n_embd, num_cls) for _ in range(num_layers)])
        self.blocks_ch = nn.ModuleList([ChannelMixingModule(n_embd, num_cls) for _ in range(num_layers)])
        self.gated = nn.ModuleList([pygnn.conv.gated_graph_conv.GatedGraphConv(n_embd, 1, bias=False) for _ in range(num_layers)])

        self.MHSA = HMHSA(in_feat=n_embd, num_heads=4, attn_dropout=0.2, proj_out_dim=num_cls**2)

        decay_speed = torch.ones(n_embd, 1)

        # Encode node features to edge-features 
        self.edge_enc = nn.Linear(n_embd*2, num_cls**2, bias=False)
        self.edge_mix = nn.Linear(n_embd*2, num_cls, bias=False)
        self.ee = nn.Linear(num_cls, num_cls**2, bias=False)

        self.pred = nn.Linear(n_embd, num_cls, bias=False)

        self.cm = nn.Parameter(-1*torch.eye(num_cls))


        self.ln1 = nn.LayerNorm(num_cls)
        self.ln2 = nn.LayerNorm(num_cls)
        #self.ln3 = nn.LayerNorm(n_embd)

    def init_rnn_vars(self, dim):
        for l in self.blocks:
            l.init_rnn_vars(dim)
        for l in self.blocks_ch:
            l.init_rnn_vars()        

    def edge_encoder(self, x, s, t, adj):
        x_edge = torch.hstack([x[s], x[t]])
        #x_edge = x[s] + x[t]
        eps = 1e-20
        #x_edge = self.edge_enc(x_edge).view(-1, self.num_cls, self.num_cls).log_softmax(-1)
        x_edge = self.edge_enc(x_edge).softmax(-1).view(-1, self.num_cls, self.num_cls)+eps#.log_softmax(-1)
        
        #x_edge = self.MHSA(x, adj=adj, ret_attn=False).softmax(-1).view(-1, self.num_cls, self.num_cls)#.log_softmax(-1)

        sum_s = torch.sum(x_edge, dim=2).unsqueeze(2) + eps
        sum_t = torch.sum(x_edge, dim=1).unsqueeze(1) + eps
        pred_edge = (x_edge.log() - 0.1*(sum_s.log() + sum_t.log()))
        return pred_edge

    def mfi(self, x, A, Q):
        A, _ = upyg.remove_self_loops(A)
        s, t = A[0], A[1]

        n_nodes, n_edges = Q.shape[0], A.shape[1]
        U =Q # Q.log_softmax(-1)

        # [edges, cls]
        msg = torch.ones((n_edges, self.num_cls), device=Q.device) / self.num_cls
        msg = msg.log()

        edge_dict = torch.sparse_coo_tensor(indices=A, values=torch.arange(n_edges).cuda(), size=(n_nodes, n_nodes)).to_dense().cuda()
        rev = edge_dict[t, s]

        #edge = self.edge_encoder(x, s, t)#.reshape(A.shape[1], self.num_cls, self.num_cls)
        edges = self.edge_mix(torch.hstack([x[s], x[t]])).log_softmax(-1)

        node = Q

        for _ in range(5):
            #node = node.log_softmax(-1)
            j = 0
            node = node.log_softmax(-1)

            for tm, ch in zip(self.blocks, self.blocks_ch):
                
                #for i in range(5):
                edges_ = tm(node, s, t, j, self.num_layers)
                #node_ = torch.dropout(edges_, p=0.5, train=self.training) #+Q[s]
                node_ = edges_ 
                edges_ = ch(node_)
                #node_ = torch.dropout(self.ln2(edges_), p=0.1, train=self.training) 
                edges_ = edges_
                
             

                msg_rw = torch.zeros_like(Q, device=Q.device)
                upyg.scatter(edges_, index=t, dim=0, out=msg_rw, reduce="add")
                node = msg_rw + node

                j += 1


            msg_us = node @ self.cm
            msg = U - msg_us
            node = msg
            

        return msg

    
    def forward(self, x, A, Q, adj):
        sh = [A.shape[1], x.shape[1]]
        self.init_rnn_vars(sh)
        #return x, self.nolbp(x, A, Q)
        #return x, self.mfi(x, A, Q)
        return x, self.LBP(x, A, Q, adj)

############################################################
from typing import Any, Iterator, Literal, Sequence

from LogWKV import wkv_log_space, initial_state_log_space

PretrainedRwkvKey = Literal["169m", "430m", "1.5b", "3b", "7b", "14b"]

AttentionState = tuple[Tensor, Tensor]
FeedForwardState = Tensor
State = tuple[AttentionState, FeedForwardState]


class Attention(nn.Module):
    init_x: Tensor
    init_state: Tensor

    def __init__(
        self,
        dim: int,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        self.time_decay = nn.Parameter(torch.empty(dim))
        self.time_first = nn.Parameter(torch.empty(dim))

        self.time_mix_k = nn.Parameter(torch.empty(1, 1, dim))
        self.time_mix_v = nn.Parameter(torch.empty(1, 1, dim))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, dim))

        if freeze:
            self.time_decay.requires_grad_(False)
            self.time_first.requires_grad_(False)
            self.time_mix_k.requires_grad_(False)
            self.time_mix_v.requires_grad_(False)
            self.time_mix_r.requires_grad_(False)

        self.key = nn.Linear(dim, dim, False)
        self.value = nn.Linear(dim, dim, False)
        self.receptance = nn.Linear(dim, dim, False)
        self.output = nn.Linear(dim, dim, False)

        wkv_fn, init_state =  wkv_log_space, initial_state_log_space(dim)


        self.register_buffer("init_x", torch.zeros(1, 1, dim), persistent=False)
        self.register_buffer("init_state", init_state, persistent=False)

        self.wkv_fn = wkv_fn

    def time_shift(self, last_x: Tensor, x: Tensor) -> Tensor:
        _, tsz, _ = x.shape
        if tsz > 1:
            last_x = torch.cat((last_x, x[..., :-1, :]), dim=-2)
        return last_x

    def forward(self, x: Tensor, state: AttentionState | None) -> tuple[Tensor, AttentionState]:
        bsz, _, _ = x.shape

        if state is None:
            last_x = self.init_x.repeat_interleave(bsz, dim=0)
            last_state = self.init_state.repeat_interleave(bsz, dim=0)
        else:
            last_x, last_state = state

        #last_x = self.time_shift(last_x, x)
        last_x = last_x

        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        v = self.value(x * self.time_mix_v + last_x * (1 - self.time_mix_v))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))
        sr = torch.sigmoid(r)

        w, u = self.time_decay, self.time_first
        w = torch.exp(w)

        wkv, next_state = self.wkv_fn(w, u, k, v, last_state)
        rwkv = wkv * sr

        return self.output(rwkv), (x[..., -1:, :], next_state)



if __name__ == "__main__":
    attn = Attention(128, False).cuda()

    inp = torch.rand((4, 10, 128)).cuda()

    out, state = attn(inp, None)
    print("out: ", out.shape)
    print("state", state[0].shape, state[1].shape)
    print(state)

    out, state2 = attn(out, state)
    print(state2)