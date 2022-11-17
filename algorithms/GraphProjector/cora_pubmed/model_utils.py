import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
import torch_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def sym_norm(adj):
    if isinstance(adj, torch_sparse.SparseTensor):
        adj_t = gcn_norm(adj, add_self_loops=True) 
        return adj_t
    
class GNN(torch.nn.Module):
    def __init__(self, x_dims, y_dims, n_layers=2):
        super(GNN, self).__init__()
        
        self.n_layers = n_layers
        self.W = torch.nn.Parameter(torch.zeros(x_dims, y_dims))

    def forward(self, adj_t, x):
        adj_t = sym_norm(adj_t)
        for _ in range(self.n_layers):
            x = adj_t.spmm(x)
        return x
    
    def pred(self, adj_t, x):
        feats = self.forward(adj_t, x)
        return x@self.W
    
    def ovr_lr_loss(self, X, Y, lam):
        Y[Y == 0] = -1
        Z = X.mul_(Y)
        return -F.logsigmoid(Z).mean(0).sum() + lam * self.W.pow(2).sum() / 2
