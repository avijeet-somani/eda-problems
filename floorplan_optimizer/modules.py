from math import sqrt

import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_h, dim_out):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_h)
        self.linear = nn.Linear(dim_h, dim_out)

   

    def forward(self, x, edge_index):
        #print('GCN : forward ' , x.dtype, edge_index.dtype)
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = self.gcn2(h, edge_index)
        h = torch.relu(h)
        h = self.linear(h)
        return h
        



# Attention/Pointer module using Bahanadu Attention
class Attention(nn.Module):
    def __init__(self, hidden_size, C=10):
        super(Attention, self).__init__()
        self.C = C
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, 1)

    def forward(self, query, target):
        """
        Args:
            query: [batch_size x hidden_size]
            target:   [batch_size x seq_len x hidden_size]
        """

        batch_size, seq_len, _ = target.shape
        query = self.W_q(query).unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size x seq_len x hidden_size]
        target = self.W_k(target)  # [batch_size x seq_len x hidden_size]
        logits = self.W_v(torch.tanh(query + target)).squeeze(-1)
        logits = self.C * torch.tanh(logits)
        return target, logits
