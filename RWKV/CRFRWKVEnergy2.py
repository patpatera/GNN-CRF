
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

# PyG
from torch_cluster import random_walk
#from torch_scatter.composite import scatter_logsumexp, scatter_log_softmax, scatter_softmax

import torch_geometric.nn as pygnn 
import torch_geometric.utils as upyg 
#from torch_geometric.nn.aggr import SoftmaxAggregation

from torch_cluster import knn_graph
import torch_sparse
from torch_geometric.utils import to_undirected, coalesce

from RWKV.revrwkv import GroupAddRev
# My implementation
#from hmhsa_layer import HMHSA

#from RWKV.SingleTMModule import WkvLogSpace, logaddexp, logsubexp, Wkc
from RWKV.TMModule import wkv_with_eps, wkv_log_space, logaddexp, logsubexp, wkv_vanilla, wkv_mylog_space
#from utils.half_hop import HalfHop


def ned(x1, x2, dim=1, eps=1e-8):
    ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5

def nes(x1, x2, dim=1, eps=1e-8):
    return 1 - ned(x1, x2, dim, eps)

def dirichlet_energy(edge_index, edge_weight, n, X):
  if edge_weight is None:
    edge_weight = torch.ones(edge_index.size(1),
                             device=edge_index.device)
  de = torch_sparse.spmm(edge_index, edge_weight, n, n, X)
  return X.T @ de

class SReLU(nn.Module):
    """Shifted ReLU"""

    def __init__(self, nc, bias):
        super(SReLU, self).__init__()
        self.srelu_bias = nn.Parameter(torch.Tensor(nc,))
        self.srelu_relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.srelu_bias, bias)

 
    def forward(self, x):
        return self.srelu_relu(x - self.srelu_bias) + self.srelu_bias
    
class MishGLU(nn.Module):
    def __init__(self, n_embd, num_cls, layer_id, n_layer):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.ln = nn.GroupNorm(4, 4)

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)

            x = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(n_embd, num_cls, bias=False)
            self.bb = nn.Linear(n_embd, num_cls, bias=False)
            self.value = nn.Linear(n_embd, num_cls, bias=False)

    def forward(self, x, id):
        x = self.ln(x)
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))
    

    def init_rnn_vars(self):
        pass


def extract_walks(len, num, edge_index, num_nodes):
    start = torch.arange(num_nodes, device=edge_index.device)
    start = start.view(-1, 1).repeat(1, num).view(-1)

    rw_walk, ei = random_walk(edge_index[0], edge_index[1],
                        start, len, num_nodes=num_nodes, return_edge_indices=True)
    
    # len+1 because of starting node!
    #return rw_walk.view(-1, num, len+1)
    return rw_walk, start, ei


class ChannelMixingModule(nn.Module):

    def __init__(self, n_embd, num_cls, layer_id, total_layer, scale=4) -> None:
        super().__init__()
        
        self.n_embd = n_embd
        self.num_cls = num_cls

        out_emb = n_embd

        # Channel-mixing variables
        self.key = nn.Linear(n_embd, n_embd*scale, bias=False)
        self.receptance = nn.Linear(n_embd, num_cls, bias=False)
        self.value = nn.Linear(n_embd*scale, num_cls, bias=False)

        self.time_mix_k = nn.Parameter(torch.ones(1, n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1, n_embd))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.ln = pygnn.PairNorm(scale=1)

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / total_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        self.register_buffer("init_x", torch.zeros(1, 1, n_embd), persistent=False)


        #self.hist_x = [None for _ in range(100)]
        self.last_x = None

        #self.init_rnn_vars()


    def time_shift_(self, last_x, x):
        _, tsz, _ = x.shape
        if tsz > 1:
            last_x = torch.cat((last_x, x[..., :-1, :]), dim=-2)
        return last_x

    def init_rnn_vars(self, it):
        #if self.hist_x[it] != None:
        #    self.hist_x[it] = self.hist_x[it].detach().clone() #None

        self.last_x = None

    def forward(self, x, it, nor=True):
        N = x.shape[0]
        
        last_x = self.last_x
        if last_x == None:
            last_x = self.init_x.repeat_interleave(N, dim=0) #[:, -1:, :]

        xk = x * self.time_mix_k + last_x * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + last_x * (1 - self.time_mix_r)

        k = self.key(xk)    
        r = torch.sigmoid(self.receptance(xr))
        
        kv = self.value(F.tanh(k))
        self.last_x = x 

        kv = r * kv

        return kv

class TimeMixingModule(nn.Module):

    def __init__(self, n_embd, num_cls, lid, tl) -> None:
        super().__init__()

        self.n_embd = n_embd
        self.num_cls = num_cls

        # Time-mixing variables
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, num_cls, bias=False) 

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        

        with torch.no_grad():
            ratio_0_to_1 = lid / (tl - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (lid / tl)  # 1 to ~0

            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            decay_speed = torch.zeros(num_cls)
            #for h in range(num_cls):
            #    decay_speed[h] = -5 + 8 * (h / (num_cls - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(num_cls)]) * 0.5

            # Learnable params
            self.time_decay = nn.Parameter(decay_speed)
            self.time_first = nn.Parameter(torch.ones(num_cls) * math.log(0.3))

            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            

        self.register_buffer("init_x", torch.zeros(1, 1, n_embd), persistent=False)
        self.register_buffer("init_state", torch.zeros((1, 3, 1, num_cls)), persistent=False)

        #self.ln = nn.GroupNorm(10, num_cls)
        self.ln = pygnn.PairNorm(scale=1.)
        
        #self.act = SReLU(num_cls, 0.)
        #self.ln = nn.LayerNorm(num_cls)

        self.last_x = None
        self.last_state = None

        self.dim_scale = math.sqrt(num_cls)

        self.hist = [None for _ in range(0, 500)]

    def init_rnn_vars(self, it, test):
        #if self.last_x != None:
        #    self.last_x = self.last_x.detach() #None

        if self.hist[it] != None and test == False:
            #self.last_state = self.last_state.detach()#.mean(0)[None, ...] #None
            self.hist[it] = self.hist[it].detach().clone()
        
        self.last_x = None
        #self.last_state = None

    
    def time_shift_(self, last_x, x):
        _, tsz, _ = x.shape
        if tsz > 1:
            last_x = torch.cat((last_x, x[..., :-1, :]), dim=-2)
        return last_x


    def forward_seq(self, x, it):
        N = x.shape[0]

        if (self.last_x is None):
            self.last_x = self.init_x.repeat_interleave(N, dim=0) #[:, -1:, :]

        last_state = None
        if self.hist[it] != None:
            last_state = self.hist[it]

        if (last_state == None):
            last_state = self.init_state.repeat_interleave(N, dim=0)

        k = self.key(x * self.time_mix_k + self.last_x * (1 - self.time_mix_k))
        v = self.value(x * self.time_mix_v + self.last_x * (1 - self.time_mix_v))
        r = self.receptance(x * self.time_mix_r + self.last_x * (1 - self.time_mix_r))

        N, T, C = k.shape

        sr = torch.sigmoid(r)

        w, u = self.time_decay, self.time_first
        w = torch.exp(w)

        #wkv, last_state = wkv_vanilla(w, u, k, v, self.last_state)
        wkv, last_state = wkv_with_eps(w, u, k, v, last_state)
        #wkv = wkv * self.dim_scale

        if self.training:
            self.hist[it] = last_state
        else:
             self.hist[it] = None
  
        rwkv = wkv * sr 
        rwkv = self.output(rwkv)
    
        self.last_x = x
        return rwkv
    
    def forward(self, x, it):
        #x = self.ln(x)
        #x = F.tanh(x)

        return self.forward_seq(x, it)


class RWKVBlock(nn.Module):
    def __init__(self, num_layers, num_cls):
        super().__init__()

        self.num_layers = num_layers
        self._layers = range(num_layers)
        
        s = 64
    

        self.preffn = ChannelMixingModule(s, s, 0, num_layers)
        self.times = nn.ModuleList([TimeMixingModule(s * (i+1), s * (i+1), i, num_layers) for i in range(num_layers)])
        self.channels = nn.ModuleList([ChannelMixingModule(s * (i+1), s * (i+2), i, num_layers) for i in range(num_layers)])
        
        self.last_pred = nn.Linear((num_layers+1) * s, num_cls, bias=False)

        gain = 2** -2.5
        init_ortho = 0.1
        
        with torch.no_grad():
            for name, m in self.times.named_modules():
                if isinstance(m, nn.Linear):
                    if 'output' in name:
                        m.weight = nn.init.xavier_uniform_(m.weight, gain)
                    else:
                        m.weight = nn.init.xavier_uniform_(m.weight, gain)
                    #m.weight = nn.init.orthogonal_(m.weight, init_ortho)
                    m.weight = nn.init.eye_(m.weight) 
                    m.weight.data = m.weight.data * 0.01

            for name, m in self.channels.named_modules():
                if isinstance(m, nn.Linear):
                    if 'output' in name:
                        m.weight = nn.init.xavier_uniform_(m.weight, gain)
                    else:
                        m.weight = nn.init.xavier_uniform_(m.weight, gain)
                    #m.weight = nn.init.orthogonal_(m.weight, init_ortho)
                    #m.weight = nn.init.orthogonal_(m.weight, init_ortho)
                    m.weight = nn.init.eye_(m.weight)
                    m.weight.data = m.weight.data * 0.01
        
                
    def init_rnn_vars(self, dim, test=False):
        self.preffn.init_rnn_vars(test)

        for l in self.times:
            l.init_rnn_vars(dim, test)
        for l in self.channels:
            l.init_rnn_vars(test)        
         

    def forward(self, x, it): 
        x = self.preffn(x, it, False)
        
        for idx in self._layers:
            x = F.tanh(x)
            x = self.times[idx](x, it) + x

            x = F.tanh(x)
            x = self.channels[idx](x, it)

        x = self.last_pred(x)
        
        return x



class CRFRWKV(nn.Module):

    def __init__(self, num_layers, n_embd, max_len, num_cls=5, damping=0.5) -> None:
        super().__init__()
        
        num_layers = 4

        self.damping = damping
        self.num_layers = num_layers
        self.n_embd = n_embd
        self.max_len = max_len
        self.num_cls = num_cls

        self.rwkv_blocks = RWKVBlock(num_layers, num_cls)
        self.head = nn.Linear(1024, 64, False) 
        self.unary = nn.Linear(1024, num_cls, False)

        self.comp = nn.Linear(num_cls, out_features=num_cls**2, bias=False)
        nf = n_embd #+ num_cls

        self.cec = nn.Conv1d(nf*2, out_channels=num_cls**2, kernel_size=1, bias=False)

        with torch.no_grad():
            gain = nn.init.calculate_gain('tanh')  #2** -2.5 

            self.cec.weight = nn.init.xavier_uniform_(self.cec.weight, gain=gain)
            self.unary.weight = nn.init.xavier_uniform_(self.unary.weight, gain=gain)
            self.head.weight = nn.init.xavier_uniform_(self.head.weight, gain=gain)
            
            self.comp.weight = nn.init.eye_(self.comp.weight)
            self.comp.weight.data = self.comp.weight.data * 0.01
        
    """
        Get [E x CLS x CLS] edge predictions.
    """
    def edge_encoder(self, x, s, t, T=1.):
        
        x_edge = torch.hstack([x[s], x[t]])
        eps = 1e-20

        x_edge = self.cec(x_edge.T).T
        x_edge = x_edge.view(-1, self.num_cls, self.num_cls)#.log_softmax(1)
        return x_edge

    def edge_prob(self, x_edge, T=1.):
        eps = 1e-20

        x_edge = (x_edge).softmax(-1).view(-1, self.num_cls, self.num_cls) + 1e-20

        sum_s = torch.sum(x_edge, dim=2).unsqueeze(2) + eps
        sum_t = torch.sum(x_edge, dim=1).unsqueeze(1) + eps

        pred_edge = (x_edge.log() - 0.05*(sum_s.log() + sum_t.log()))
        return pred_edge
    
    def get_logH(self):
        logH = F.logsigmoid(self.param + self.param.t())
        return logH
    
    def log_mean_std(self, log_x):
        log_mean = log_x.mean(-1)[..., None]
        log_std = log_x.std(-1)[..., None]
        normalized_inputs = (log_x - log_mean) / log_std
        return normalized_inputs

    def log_normalize(self, log_x):
        return log_x - torch.logsumexp(log_x, -1, keepdim=True) 
    
    def forward_(self, x, data, it):
        A = data.edge_index

        # Init node energy
        node = self.head(data.x)

        # Unary Energy
        Q = self.unary(data.x)

        n_nodes, n_edges = x.shape[0], A.shape[1]
        s, t = A[0], A[1]
    
        rev = data.rev

        # Pairwise
        edge = self.edge_encoder(x, s, t)
        #edge = self.cec(x[s], x[t])
 

        marginals, E = self.LBP(edge, Q, A, node, rev, it, data)
        return marginals, E
    
  
    def LBP(self, edge, Q, A, node, rev, it, data):
        n_nodes, n_edges = Q.shape[0], A.shape[1]
        s, t = A[0], A[1]

        walk_len = 3    #5-Cora;  3-Citeseer;  3-pubmed
        walk_rep = 3    #5-Cora;  5-Citeseer;  5-pubmed
        steps = 2       #5-Cora;  5-Citeseer;  5-pubmed

        maske = torch.ones((n_edges), device=Q.device).bool()
        with torch.no_grad():
            D = upyg.degree(t, num_nodes=n_nodes)
            D_mask = D==0
            D = 1. / D
            D[D_mask] = 0.

            #print(f"It: {it} D: {D_mask.sum().item()}")
            D = D[..., None].cuda()

        # Init. messages for factors
        msg = torch.zeros((n_edges, self.num_cls), device=Q.device) 

        Q = Q.log_softmax(1) 
        edge = edge.log_softmax(1)

        
        W = self.damping
        
        chached_rw_idx = None
        for i in range(steps):
            rw_index, sidx, ei = extract_walks(walk_len, walk_rep, A, node.shape[0])

            with torch.no_grad():
                if chached_rw_idx != None:
                    _mask = torch.eq(rw_index, chached_rw_idx)
                    _mask = _mask.sum(-1) < (walk_len+1)

                    chached_rw_idx = rw_index
                else:
                    chached_rw_idx = rw_index
                    _mask = torch.ones_like(ei).bool().cuda()

                # Filter wrong links 
                ei_neg = ei < 0
                ei_neg = ei_neg.sum(-1) == 0

                ei = ei[ei_neg]
                ei = ei.flatten().long()

                sidx = sidx[ei_neg]

                rw_index = rw_index[ei_neg]
                            
            # Get random walk sequence
            node_ = node[rw_index] 

            # Calculate RWKV on random walks
            node_ = self.rwkv_blocks(node_, it)

            E_ = (node_[:, 1:]).reshape(node_.shape[0] * walk_len, -1)
            E_ = upyg.scatter(E_, index=ei, dim=0, dim_size=n_edges+1, reduce="max")[:-1]
            E_ = self.comp(E_).view(-1, self.num_cls, self.num_cls).log_softmax(1)
            
            scale = 1 - (i / steps)
            
            msg2 = Q[s].unsqueeze(-1) + E_ + edge*scale
            msg2 = self.log_normalize(msg2)
      
            msg_us = upyg.scatter(msg, index=t, dim=0, dim_size=Q.shape[0], reduce="sum") #* D
            msg_us = self.log_normalize((msg_us[s] - msg[rev])).unsqueeze(-1) 

            msg2 = (msg2 + msg_us)
            msg2 = msg2.max(1)[0]
            
            msg = msg * W + msg2 * (1 - W)

        msg = msg.log_softmax(-1)
        
        node = upyg.scatter(node_[:, -1].log_softmax(-1), index=sidx, dim=0, dim_size=Q.shape[0], reduce="max")
        out = upyg.scatter(msg, index=t, dim=0, dim_size=Q.shape[0], reduce="sum") * D
        out = out + Q + node 

        return out, node_

    def reset_vars(self, it):
        self.rwkv_blocks.init_rnn_vars(it)

    def forward(self, x, data, it):
        if self.training == False:
            self.rwkv_blocks.init_rnn_vars(it, True)

        marginals, edges = self.forward_(x, data, it)     

        return x, marginals, edges

#========================================================#
