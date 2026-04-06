import torch
import torch.nn as nn
import torch.nn.functional as F

# Differential eq solver
from torchdiffeq import odeint_adjoint, odeint

from torch_geometric.nn import MLP, GCNConv

from hmhsa_layer import HMHSA

class DiffusionFunc2(nn.Module):
    
    def __init__(self, mhsa, in_feat, num_cls):
        super().__init__()

        self.mhsa = mhsa
        self.A = None
        
        self.alpha = nn.Parameter(torch.tensor(1.))
        self.beta = nn.Parameter(torch.tensor(1.))

        # Learnable parameters for label compatibility MFI
        self.comp_matrix = nn.Parameter(-1*torch.eye(num_cls))
        self.weights = nn.Parameter(torch.eye(num_cls))

        self.x0 = None
        self.U = None
        self.Q = None
        self.ajds = []

        self.i = 0
        self.attn = None
        self.ln = nn.LayerNorm(in_feat)

        self.pairwise = nn.Linear(in_feat, num_cls, bias=False)
        self.ffn = MLP(in_channels=in_feat, hidden_channels=in_feat*2, out_channels=in_feat, num_layers=3, dropout=0.1, norm="LayerNorm")

    def MFI(self, crf_it, attn, z):
        # Mean-field Inference
        U = self.U
        Q = self.pairwise(z) #U if len(self.Q) == 0 else self.Q[-1]
        
        for _ in range(crf_it):
            #Q = Q.softmax(-1)
            #Q = Q.log_softmax(-1)

            #P = attn @ Q
            #P = self.pairwise(z)
            P = Q

            # Weigthing filter
            P = P @ self.weights

            # Learnable compability transform   
            P = P @ self.comp_matrix 

            # Unary addition
            Q = U - P 

        return Q.mean(0)

    def call_mfi(self, z):
        if self.i > 0 and self.i % 2 == 0:
            attn = torch.stack(self.ajds)
            #attn = attn.mean(0)
    
            Q = self.MFI(1, attn, z)
            self.Q.append(Q)

            #self.ajds = []
            #self.x0 = self.ln(z + self.x0)


    def forward(self, t, x):
        z, attn = self.mhsa(self.x, adj=self.A, ret_attn=True)
        self.ajds.append(attn)

        z_ = self.ln(z + self.x)
        z = self.ffn(z_)
        self.x = self.ln(z + z_)

        #alpha = torch.sigmoid(self.alpha)
        z = self.pairwise(self.x)

        z = self.alpha * (z - x) + self.x0 #* self.beta
        self.i += 1

        return z
    
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
    
    def __init__(self, mhsa, in_feat, num_cls, t=torch.tensor([0, 4])):
        super().__init__()

        self.diff_func = DiffusionFunc2(mhsa, in_feat, num_cls)
        self.solver = odeint

        self.t = t
        self.atol = 1e-7 * 10000
        self.rtol = 1e-9 * 10000

        self.adj_atol = 1e-7 * 16324
        self.adj_rtol = 1e-9 * 16324


    def set_adj(self, A):
        self.diff_func.A = A

    def forward(self, x, U):
        t = self.t.type_as(x)

        #c_aux = torch.zeros(x.shape).cuda()
        #x = torch.cat([x, c_aux], dim=1)

        self.diff_func.x0 = U.clone().detach()
        self.diff_func.x = x
        
        self.diff_func.i = 0
        self.diff_func.U = U
        self.diff_func.Q = []
        self.diff_func.ajds = []


        z = self.solver(self.diff_func, U, t,
                        method="rk4",
                        #adjoint_method="rk4",
                        options={'step_size': 1, 'perturb': False},
                        #adjoint_options={'step_size': 1},
                        #adjoint_atol=self.adj_atol,
                        #adjoint_rtol=self.adj_rtol,
                        atol=self.atol,
                        rtol=self.rtol)[-1]
        
        #z = torch.split(z, x.shape[1] // 2, dim=1)[0]

        #attn = torch.stack(self.diff_func.ajds)
        #attn = self.diff_func.attn.div(self.diff_func.i)

        #self.diff_func.call_mfi(z)
        #Q = self.diff_func.Q[-1]

        return z, None