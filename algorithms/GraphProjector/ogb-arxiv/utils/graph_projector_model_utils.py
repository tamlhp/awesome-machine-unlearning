import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import SparseTensor
from .train_utils import sym_norm, row_norm

class GraphConv(nn.Module):
    def __init__(self, ):
        super(GraphConv, self).__init__()
        self.sigma = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs, adj_t):

        row = adj_t.storage.row()
        col = adj_t.storage.col()
        sparse_sizes = adj_t.sizes()

        pw_diff = inputs[row] - inputs[col]
        pw_diff = pw_diff.norm(p=2, dim=1)**2
        pw_diff = torch.exp(- pw_diff / self.sigma**2)

        adj_t = SparseTensor(row=row, col=col,
                             value=pw_diff,
                             sparse_sizes=sparse_sizes)

        return adj_t


class GNN(nn.Module):
    def __init__(self, x_dims, y_dims, device, args):
        super(GNN, self).__init__()
        self.W = torch.nn.Parameter(torch.zeros(
            x_dims * (1 + args.x_iters) + y_dims * args.y_iters, y_dims))
        self.device = device

        if args.use_adapt_gcs:
            self.gcs = nn.ModuleList()
            for iters in range(args.x_iters + args.y_iters):
                self.gcs.append(GraphConv())

        self.args = args

    def forward(self, subgraph_data):
        return self.gen_feats(subgraph_data) @ self.W

    def gen_feats(self, subgraph_data):
        if self.args.use_adapt_gcs == False:
            adj_t_norm = sym_norm(subgraph_data.adj_t).to(self.device)

        all_outputs = []
        # Node features
        X = subgraph_data.x.to(self.device)
        all_outputs.append(X)

        for iters in range(self.args.x_iters):
            if self.args.use_adapt_gcs:
                adj_t = self.gcs[iters](all_outputs[-1], subgraph_data.adj_t)
                adj_t_norm = sym_norm(adj_t).to(self.device)
            outputs = adj_t_norm.spmm(all_outputs[-1])
            all_outputs.append(outputs)

        # Node labels
        Y = subgraph_data.y_one_hot_train.to(self.device)
        Y[subgraph_data.root_n_id] = 0

        for iters in range(self.args.y_iters):
            if iters == 0:
                if self.args.use_adapt_gcs:
                    adj_t = self.gcs[self.args.x_iters +
                                     iters](Y, subgraph_data.adj_t)
                    adj_t_norm = sym_norm(adj_t).to(self.device)
                outputs = adj_t_norm.spmm(Y)
                all_outputs.append(outputs)
            else:
                if self.args.use_adapt_gcs:
                    adj_t = self.gcs[self.args.x_iters +
                                     iters](all_outputs[-1], subgraph_data.adj_t)
                    adj_t_norm = sym_norm(adj_t).to(self.device)
                outputs = adj_t_norm.spmm(all_outputs[-1])
                all_outputs.append(outputs)

        node_feats = torch.cat(all_outputs, axis=1)[subgraph_data.root_n_id, :]
        return node_feats
    
    def loss(self, subgraph_data, lam):
        if self.args.use_cross_entropy:
            return self.cross_entropy_loss(subgraph_data, lam)
        else:
            return self.ovr_lr_loss(subgraph_data, lam)
        
    def ovr_lr_loss(self, subgraph_data, lam):
        Y = subgraph_data.y_one_hot_train[subgraph_data.root_n_id, :]
        Y[Y == 0] = -1
        pred = self.forward(subgraph_data)
        Z = pred.mul_(Y)
        return -F.logsigmoid(Z).mean(0).sum() + lam * self.W.pow(2).sum() / 2
    
    def cross_entropy_loss(self, subgraph_data, lam):
        pred = self.forward(subgraph_data)
        return F.cross_entropy(pred, subgraph_data.y.squeeze(1)) + lam * self.W.pow(2).sum() / 2