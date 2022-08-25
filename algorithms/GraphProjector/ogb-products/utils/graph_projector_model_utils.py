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
    def __init__(self, x_dims, h_dims, y_dims, device, args):
        super(GNN, self).__init__()
                
        if args.compress_dim:
            dims = x_dims + y_dims
        else:
            dims = x_dims * (1 + args.x_iters) + y_dims * args.y_iters

        self.W = torch.nn.Parameter(torch.zeros(dims, y_dims))
            
        self.device = device

        if args.use_adapt_gcs:
            self.gcs = nn.ModuleList()
            for iters in range(args.x_iters + args.y_iters):
                self.gcs.append(GraphConv())

        self.args = args

    def forward(self, subgraph_data):            
        return self.gen_feats(subgraph_data) @ self.W

    def gen_feats(self, subgraph_data):
        adj_t_norm_fixed = row_norm(subgraph_data.adj_t).to(self.device)

        x_all_outputs = []
        # Node features
        X = subgraph_data.clone().x.to(self.device)
        x_all_outputs.append(X)

        for iters in range(self.args.x_iters):
            if self.args.use_adapt_gcs_x:
                adj_t = self.gcs[iters](x_all_outputs[-1], subgraph_data.adj_t)
                adj_t_norm = row_norm(adj_t).to(self.device)
                outputs = adj_t_norm.spmm(x_all_outputs[-1])
            else:
                outputs = adj_t_norm_fixed.spmm(x_all_outputs[-1])
            x_all_outputs.append(outputs)

        # Node labels
        y_all_outputs = []
        Y = subgraph_data.y_one_hot_train.clone().to(self.device)
        if self.args.use_shallow_sampler:
            Y[subgraph_data.root_n_id] = 0

        for iters in range(self.args.y_iters):
            if iters == 0:
                if self.args.use_adapt_gcs_y:
                    adj_t = self.gcs[self.args.x_iters + iters](Y, subgraph_data.adj_t)
                    adj_t_norm = row_norm(adj_t).to(self.device)
                    outputs = adj_t_norm.spmm(Y)
                else:
                    outputs = adj_t_norm_fixed.spmm(Y)
            else:
                if self.args.use_adapt_gcs_y:
                    adj_t = self.gcs[self.args.x_iters + iters](y_all_outputs[-1], subgraph_data.adj_t)
                    adj_t_norm = row_norm(adj_t).to(self.device)
                    outputs = adj_t_norm.spmm(y_all_outputs[-1])
                else:
                    outputs = adj_t_norm_fixed.spmm(y_all_outputs[-1])
            y_all_outputs.append(outputs)

        if self.args.compress_dim:
            x_all_outputs = torch.stack(x_all_outputs).sum(dim=0)
            y_all_outputs = torch.stack(y_all_outputs).sum(dim=0)
            node_feats = torch.cat([x_all_outputs, y_all_outputs], axis=1)
        else:
            node_feats = torch.cat(x_all_outputs + y_all_outputs, axis=1)
        if self.args.use_shallow_sampler:
            node_feats = node_feats[subgraph_data.root_n_id, :]
            
        return node_feats
    
    def get_labels(self, subgraph_data):
        Y = subgraph_data.y_one_hot_train.clone()
        if self.args.use_shallow_sampler:
            Y = Y[subgraph_data.root_n_id, :]
        Y[Y == 0] = -1
        return Y
        
    def ovr_lr_loss(self, subgraph_data, lam):
        Y = self.get_labels(subgraph_data)
        feats = self.gen_feats(subgraph_data)
        return self.ovr_lr_loss_with_feats(feats, Y, lam)
    
    def ovr_lr_loss_with_feats(self, feats, Y, lam):
        pred = feats @ self.W
        Z = pred.mul_(Y)
        return -F.logsigmoid(Z).mean(0).sum() + lam * self.W.pow(2).sum() / 2
    
    def cross_entropy_loss_with_feats(self, feats, Y, lam):
        pred = feats @ self.W
        labels = Y.argmax(1).view(-1)
        return F.cross_entropy(pred, labels) + lam * self.W.pow(2).sum() / 2
    
    def loss_with_feats(self, feats, Y, lam):
        if self.args.use_cross_entropy:
            return self.cross_entropy_loss_with_feats(feats, Y, lam)
        else:
            return self.ovr_lr_loss_with_feats(feats, Y, lam)
    