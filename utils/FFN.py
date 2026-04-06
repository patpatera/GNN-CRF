import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):

    def __init__(self, in_feat, hidden_feat=1024, dropout=0.1, norm=True):
        super().__init__()

        self.norm = norm
        # Dropout
        self.dropout = nn.Dropout(dropout)

        # layer normalisation
        self.layer_norm = nn.LayerNorm(in_feat)
        self.layer_norm2 = nn.LayerNorm(in_feat)
        self.bn = nn.BatchNorm1d(in_feat)

        self.m11 = nn.Linear(in_feat, hidden_feat, False)
        self.m12 = nn.Linear(hidden_feat, in_feat, False)


    def forward(self, x1, x1_):
        x1_ = self.dropout(x1_) 
        #if not x1 == None:
        x1 = self.layer_norm(x1_ + x1)
        #else:
        #    x1 = x1_
        #x1 = x1 + x1_

        x = self.m11(x1)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.m12(x)
        
        
        x = self.dropout(x)
        x = self.layer_norm2(x1 + x)
        #x = self.bn(x + x1)
        #x = x + x1


        return x

