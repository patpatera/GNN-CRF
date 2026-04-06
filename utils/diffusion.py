import torch
import torch.nn as nn
import torch.nn.functional as F

# Differential eq solver
from torchdiffeq import odeint_adjoint, odeint

from utils.FFN import FFN
from hmhsa_layer import HMHSA

class DiffusionFunc(nn.Module):
    
    def __init__(self, mhsa, in_feat, num_classes):
        super().__init__()

        self.MHSA = mhsa
        self.A = None

        #self.MHSA = HMHSA(in_feat=in_feat, num_heads=2, attn_dropout=0, proj_out_dim=in_feat)
        
        self.alpha = nn.Parameter(torch.tensor(1.))
        self.beta = nn.Parameter(torch.tensor(1.))

        #self.FFN = FFN(in_feat)

        self.x0 = None
        self.ajds = []
        self.i = 0
        self.M = None
        self.edge = None
        self.Q = None
        self.Q_ = None


        self.scores = None
        self.prev_scores = None

    def forward(self, t, x): 
        #z, attn, self.prev_scores = self.MHSA(x, adj=self.A, ret_attn=True)
        z, _ = self.MHSA(x, A=self.A) 

        alpha = torch.sigmoid(self.alpha)
        z =  alpha * (z - x)  + self.beta * self.x0

        #if self.scores == None:
        #    self.scores = self.prev_scores
        #else:
        #    self.scores = self.scores + self.prev_scores

        #self.Q = (self.attn @ self.Q)
        #self.A = self.A + self.rewire_attn(scores)
        #self.A[self.A>0] = 1

        # Append attenstion score of i-th diffusion iteration

        #self.ajds.append(attn)
        #self.i += 1
    
        return z
    

    def rewire_attn(self, attn):
        attn[range(attn.shape[0]), range(attn.shape[0])] = -1000
        top_idx = torch.topk(attn, 2, -1)[1]

        adj = torch.zeros_like(attn).cuda()
        adj = adj.scatter(1, top_idx, 1.)

        #adj = adj + adj.T

        return adj 
    
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
        

class DiffusionWrapper(nn.Module):
    
    def __init__(self, mhsa, in_feat, num_cls, t=4):
        super().__init__()

        self.diff_func = DiffusionFunc(mhsa, in_feat, num_cls)
        self.solver_adj = odeint_adjoint
        self.solver_norm = odeint
        
        self.t = torch.tensor([0, t])
        self.atol = 1e-7 * 100000
        self.rtol = 1e-9 * 100000


        self.adj_atol = 1e-7 * 100000
        self.adj_rtol = 1e-9 * 100000


    def set_adj(self, A):
        self.diff_func.A = A

    def set_adj_idx(self, A):
        self.diff_func.edge = A

    def forward(self, x, t=4, it=0, M=None):
        t = torch.tensor([0, t]).type_as(x)

        self.diff_func.x0 = x.clone().detach()
        self.diff_func.i = it
        self.diff_func.ajds = []
        self.diff_func.scores = None
        self.diff_func.prev_scores = None
        self.diff_func.M = M
        """
        if self.training:
            z = self.solver_adj(self.diff_func, x, t,
                            method="rk4",
                            adjoint_method="rk4",
                            options={'step_size': 1, 'perturb': False},
                            adjoint_options={'step_size': 1},
                            adjoint_atol=self.adj_atol,
                            adjoint_rtol=self.adj_rtol,
                            atol=self.atol,
                            rtol=self.rtol)[-1]
        else:
        """
        z = self.solver_norm(self.diff_func, x, t,
                        method="rk4",
                        #adjoint_method="rk4",
                        options={'step_size': 1., 'perturb': False},
                        #adjoint_options={'step_size': 1},
                        #adjoint_atol=self.adj_atol,
                        #adjoint_rtol=self.adj_rtol,
                        atol=self.atol,
                        rtol=self.rtol)[-1]
            


        # Stack attention scores to tensor
        #attn = torch.stack(self.diff_func.ajds) #.mean(0)
        #attn = self.diff_func.ajds


        return z #, attn, self.diff_func.scores/4.