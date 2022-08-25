import argparse
import copy
import math
import os
import pickle
import random
import time
from tqdm import tqdm

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric import seed_everything
from torch_geometric.loader import ShaDowKHopSampler
import torch_geometric.transforms as T

from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

from utils.train_utils import pred_test
from utils.graph_projector_model_utils import GNN

#################################################
#################################################
#################################################

parser = argparse.ArgumentParser(description="OGBN-Arxiv (GNN)")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dropout_times", type=int, default=2)

parser.add_argument("--regen_model", action="store_true")
parser.add_argument("--num_remove_nodes", type=float, default=0.2)
parser.add_argument("--parallel_unlearning", type=int, default=20)
parser.add_argument("--lam", type=float, default=1e-6,
                    help="L2 regularization")
parser.add_argument("--hop_neighbors", type=int, default=20)

parser.add_argument("--use_cross_entropy", action="store_true")
parser.add_argument("--use_adapt_gcs", action="store_true")
parser.add_argument("--x_iters", type=int, default=3)
parser.add_argument("--y_iters", type=int, default=3)

parser.add_argument("--use_mlp", action="store_true")

args = parser.parse_args()

args.regen_model = True
args.require_linear_span = True

print(args)

seed_everything(args.seed)

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

dataset = PygNodePropPredDataset(
    name="ogbn-arxiv",
    transform=T.ToSparseTensor(),
    root="../dataset"
)

data = dataset[0]
split_idx = dataset.get_idx_split()

evaluator = Evaluator(name="ogbn-arxiv")

if args.use_mlp:
    mlp_x = torch.load('graph_projector_models/best_model_output.pt')
    data.x = torch.cat([data.x, mlp_x], dim=1)
    
print(data)
#################################################
#################################################
#################################################
# Generate augment node feats
num_train_nodes = len(split_idx["train"])
train_ind_perm = np.random.permutation(split_idx["train"])
if args.num_remove_nodes < 1:
    args.num_remove_nodes = int(num_train_nodes * args.num_remove_nodes)
else:
    args.num_remove_nodes = int(args.num_remove_nodes)
    
print('Remove nodes %d/%d' % (args.num_remove_nodes, num_train_nodes))
delete_nodes_all = train_ind_perm[: args.num_remove_nodes]

#################################################
#################################################
#################################################
# get adjs
data.adj_t = torch_sparse.fill_diag(data.adj_t.to_symmetric(), 1)

data.y_one_hot_train = F.one_hot(
    data.y.squeeze(), dataset.num_classes).float()
data.y_one_hot_train[split_idx["test"], :] = 0

num_nodes = data.x.size(0)
data.node_inds = torch.arange(data.x.size(0))

train_loader = ShaDowKHopSampler(data,
                                 depth=2,
                                 num_neighbors=args.hop_neighbors,
                                 batch_size=256,
                                 num_workers=10,
                                 shuffle=True,
                                 node_idx=split_idx["train"])

all_loader = ShaDowKHopSampler(data,
                               depth=2,
                               num_neighbors=args.hop_neighbors,
                               batch_size=1024,
                               num_workers=10,
                               shuffle=False)

#################################################
#################################################
#################################################
# pre-train model


def pre_train(print_per_epoch=5):
    best_valid_score = 0

    model = GNN(x_dims, y_dims, device, args).to(device)
    if args.require_linear_span:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, 1 + args.epochs):
        # training
        seed_everything(args.seed+epoch)

        pbar = tqdm(total=len(train_loader))
        pbar.set_description('Epoch %d' % epoch)

        model.train()
        for subgraph_data in train_loader:
            loss = model.loss(subgraph_data.to(device), args.lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
        pbar.close()

        # evaluate
        model.eval()
        seed_everything(args.seed)  # epoch > 0
        with torch.no_grad():
            all_score = []

            for subgraph_data in all_loader:
                score = model(subgraph_data.to(device))
                all_score.append(score.detach().cpu())

            all_score = torch.cat(all_score, dim=0)

        train_acc, val_acc, test_acc = pred_test(all_score, data, split_idx,
                                                 evaluator)

        if epoch % print_per_epoch == 0:
            print(
                f"Epoch: {epoch}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )

        if val_acc > best_valid_score:
            best_valid_score = val_acc
            best_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_params)
    return model


#################################################
#################################################
#################################################
# pre-training

x_dims = data.x.size(1)
y_dims = data.y_one_hot_train.size(1)

start_time = time.time()


fn = os.path.join(os.getcwd(),
                  "graph_projector_models", "exp1_graph_projector_%d.pt" % args.num_remove_nodes)
if os.path.exists(fn) and not args.regen_model:
    model_optim = GNN(x_dims, y_dims, device, args).to(device)
    model_optim.load_state_dict(torch.load(fn))
else:
    model_optim = pre_train()
    torch.save(model_optim.state_dict(), fn)
print("train model time", time.time() - start_time)

#################################################
#################################################
#################################################


@torch.no_grad()
def evaluation_reuse_labels(model):
    model = model.to(device)

    # directly predict
    all_pred = []

    seed_everything(args.seed)
    for subgraph_data in all_loader:
        pred = model(subgraph_data.to(device))
        all_pred.append(pred.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    train_acc, val_acc, test_acc = pred_test(
        all_pred, data, split_idx, evaluator)
    print(
        f"Direct predict >>> Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    # reuse predicted labels
    y_one_hot_tmp = copy.deepcopy(data.y_one_hot_train)
    y_one_hot_tmp[split_idx["test"]] = F.one_hot(
        all_pred[split_idx["test"]].argmax(dim=-1, keepdim=True).squeeze(),
        data.y_one_hot_train.size(1)
    ).float()

    # label reuse
    all_pred = []

    seed_everything(args.seed)
    for subgraph_data in all_loader:
        subgraph_data.y_one_hot_train = y_one_hot_tmp[subgraph_data.node_inds]
        pred = model(subgraph_data.to(device))
        all_pred.append(pred.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    train_acc, val_acc, test_acc = pred_test(
        all_pred, data, split_idx, evaluator)
    print(
        f"Label reuse >>> Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    return train_acc, val_acc, test_acc


result_before = evaluation_reuse_labels(model_optim)

#################################################
#################################################
#################################################
# projection-based unlearning
all_results = [result_before]


remain_nodes = np.arange(num_nodes)


feat_dim = data.x.size(1)
label_dim = data.y_one_hot_train.size(1)
adj_t_scipy = data.adj_t.to_scipy('csr')

W_optim = model_optim.W.data.clone().cpu()

batch = args.parallel_unlearning
delete_node_batch = [[] for _ in range(batch)]
for i, node_i in enumerate(delete_nodes_all):
    delete_node_batch[i % batch].append(node_i)

start_time = time.time()
for cnt, delete_node_batch_i in enumerate(delete_node_batch):
    # get remain node feats
    remain_nodes = np.setdiff1d(remain_nodes, delete_node_batch_i)
    remain_node_feats = data.x[remain_nodes]
    remain_node_label = data.y_one_hot_train[remain_nodes]

    # unlearning
    extra_channel_norm_before = 0
    extra_channel_norm_after = 0
    W_optim_part = torch.split(W_optim, [feat_dim for _ in range(
        args.x_iters+1)] + [label_dim for _ in range(args.y_iters)])
    W_optim_part_unlearn = []

    for W_part in W_optim_part[:args.x_iters+1]:
        XtX = remain_node_feats.T@remain_node_feats
        XtX_inv = torch.linalg.pinv(XtX)
        proj_W_optim = XtX@XtX_inv@W_part
        W_optim_part_unlearn.append(proj_W_optim)
        extra_channel_norm_before += W_part[-1, :].norm(2).item()
        extra_channel_norm_after += proj_W_optim[-1, :].norm(2).item()
        
    for W_part in W_optim_part[-args.y_iters:]:
        XtX = remain_node_label.T@remain_node_label
        XtX_inv = torch.linalg.pinv(XtX)
        proj_W_optim = XtX@XtX_inv@W_part
        W_optim_part_unlearn.append(proj_W_optim)
        extra_channel_norm_before += W_part[-1, :].norm(2).item()
        extra_channel_norm_after += proj_W_optim[-1, :].norm(2).item()
        
    print('extra_channel_norm_before', extra_channel_norm_before, 
          'extra_channel_norm_after',  extra_channel_norm_after) 

    # evaluate
    print('Unlearning step %d >>>' % (cnt+1))

    print("Total time:", time.time() - start_time)

    model_unlearn = copy.deepcopy(model_optim)
    model_unlearn.W.data = torch.cat(W_optim_part_unlearn, dim=0)

    # get the graph after node deletion    
    adj_t_scipy_remain = adj_t_scipy[remain_nodes, :][:, remain_nodes].tocoo()
    
    data.adj_t = SparseTensor(
        row=torch.from_numpy(remain_nodes[adj_t_scipy_remain.row]).long(),
        col=torch.from_numpy(remain_nodes[adj_t_scipy_remain.col]).long(),
        value=torch.from_numpy(adj_t_scipy_remain.data).float(),
        sparse_sizes=(num_nodes, num_nodes)
    ).to(device)
        
    all_loader = ShaDowKHopSampler(data,
                                   depth=2,
                                   num_neighbors=args.hop_neighbors,
                                   batch_size=512,
                                   num_workers=10,
                                   shuffle=False)

    unlearn_result = evaluation_reuse_labels(model_unlearn)
    all_results.append(unlearn_result)
    
if args.use_adapt_gcs:
    fn = 'exp3_results/exp3_unlearn_gc_%d_%f_seed%d.pkl'%(args.parallel_unlearning, args.num_remove_nodes, args.seed)
else:
    fn = 'exp3_results/exp3_unlearn_%d_%f_seed%d.pkl'%(args.parallel_unlearning, args.num_remove_nodes, args.seed)
with open(fn, 'wb') as f:
    pickle.dump(all_results, f)