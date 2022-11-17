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

parser = argparse.ArgumentParser(description="OGBN-Products (GNN)")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--lr", type=float, default=1.0)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--eval_epochs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--regen_model", action="store_true")
parser.add_argument("--num_remove_nodes", type=float, default=0.1)
parser.add_argument("--parallel_unlearning", type=int, default=1)
parser.add_argument("--lam", type=float, default=1e-6,
                    help="L2 regularization")
parser.add_argument("--hop_neighbors", type=int, default=25)

parser.add_argument("--recycle_steps", type=int, default=100)
parser.add_argument("--x_iters", type=int, default=3)
parser.add_argument("--y_iters", type=int, default=5)

parser.add_argument("--use_adapt_gcs_x", action="store_true")
parser.add_argument("--use_adapt_gcs_y", action="store_true")
parser.add_argument("--use_cross_entropy", action="store_true")
parser.add_argument("--use_mlp", action="store_true")

args = parser.parse_args()

args.regen_model = True

args.require_linear_span = True
args.use_shallow_sampler = True
args.compress_dim = True

args.use_adapt_gcs = args.use_adapt_gcs_x or args.use_adapt_gcs_y

print(args)

seed_everything(args.seed)

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

dataset = PygNodePropPredDataset(name="ogbn-products",
                                 transform=T.ToSparseTensor(),
                                 root="../dataset")

data = dataset[0]
split_idx = dataset.get_idx_split()

evaluator = Evaluator(name="ogbn-products")

if args.use_mlp:
    mlp_x = torch.load('graph_projector_models/best_model_output.pt')
    data.x = torch.cat([data.x, mlp_x], dim=1)
    
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
delete_nodes_all = train_ind_perm[:args.num_remove_nodes]

extra_feats = torch.zeros(data.x.size(0))
extra_feats[delete_nodes_all] = 1

data.x = torch.cat([data.x, extra_feats.view(-1, 1)], dim=1)
data.y[delete_nodes_all] = dataset.num_classes

#################################################
#################################################
#################################################
# get adjs
data.adj_t = torch_sparse.fill_diag(data.adj_t.to_symmetric(), 1)

data.y_one_hot_train = F.one_hot(data.y.squeeze(),
                                 dataset.num_classes + 1).float()
data.y_one_hot_train[split_idx["test"], :] = 0

num_nodes = data.x.size(0)
data.node_inds = torch.arange(data.x.size(0))

train_loader = ShaDowKHopSampler(data,
                                 depth=2,
                                 num_neighbors=args.hop_neighbors,
                                 batch_size=512,
                                 num_workers=10,
                                 shuffle=True,
                                 node_idx=split_idx["train"])

all_loader = ShaDowKHopSampler(data,
                               depth=2,
                               num_neighbors=args.hop_neighbors,
                               batch_size=512,
                               num_workers=10,
                               shuffle=False)

#################################################
x_dims = data.x.size(1)
y_dims = data.y_one_hot_train.size(1)
h_dims = 128

model = GNN(x_dims, h_dims, y_dims, device, args).to(device)


@torch.no_grad()
def compute_eval_feats(model, all_loader):
    all_feats = []
    pbar = tqdm(total=len(all_loader))
    pbar.set_description('Compute eval feats >>>')
    with torch.no_grad():
        for subgraph_data in all_loader:
            feats = model.gen_feats(subgraph_data.to(device))
            all_feats.append(feats.detach().cpu())
            pbar.update(1)
        pbar.close()
    all_feats = torch.cat(all_feats, dim=0)
    return all_feats

#################################################


def pre_train(model, print_per_epoch=1):
    best_valid_score = 0
    eval_feat_cached = False
    recycle_steps = args.recycle_steps

    #######################################################
    if args.require_linear_span:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters())

    #######################################################
    # training
    for epoch in range(1, 1 + args.epochs):
        seed_everything(args.seed + epoch)

        #######################################################
        # save all grads during training (simulate full-batch --> OGB-products + GD)
        model.train()
        if (epoch - 1) % recycle_steps == 0:
            pbar = tqdm(total=len(train_loader))
            pbar.set_description('Epoch %d' % epoch)

            # create a dict to save gradients
            all_grads = dict()
            for n, p in model.named_parameters():
                all_grads[n] = torch.zeros_like(p.data)

            all_train_feats, all_train_labels = [], []
            for iters, subgraph_data in enumerate(train_loader):
                subgraph_data = subgraph_data.to(device)

                train_feats = model.gen_feats(subgraph_data)
                train_labels = model.get_labels(subgraph_data)

                loss = model.loss_with_feats(train_feats, train_labels, args.lam)

                optimizer.zero_grad()
                loss.backward()

                if iters == len(train_loader) - 1:
                    for n, p in model.named_parameters():
                        if p.grad != None:
                            p.grad.data = (
                                p.grad.data + all_grads[n]) / len(train_loader)
                    optimizer.step()
                else:
                    # aggregate all grads
                    for n, p in model.named_parameters():
                        if p.grad != None:
                            all_grads[n] += p.grad.data.clone()

                all_train_feats.append(train_feats.clone().detach().cpu())
                all_train_labels.append(train_labels.clone().detach().cpu())

                pbar.update(1)
            pbar.close()

            all_train_feats = torch.cat(all_train_feats, dim=0).to(device)
            all_train_labels = torch.cat(all_train_labels, dim=0).to(device)

            #######################################################
            if args.use_mlp or args.use_adapt_gcs:
                all_feats = compute_eval_feats(model, all_loader)
                eval_feat_cached = True

        #######################################################
        else:
            loss = model.loss_with_feats(all_train_feats, all_train_labels, args.lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #######################################################
        # evaluate
        if eval_feat_cached == False:
            all_feats = compute_eval_feats(model, all_loader)
            eval_feat_cached = True

        if epoch % args.eval_epochs == 0:
            model.eval()
            all_score = all_feats @ model.W.cpu()
            train_acc, val_acc, test_acc = pred_test(
                all_score, data, split_idx, evaluator)

            print(
                f"Epoch: {epoch}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )

            if val_acc > best_valid_score:
                best_epoch = epoch
                best_valid_score = val_acc
                best_params = copy.deepcopy(model.state_dict())

            if epoch - best_epoch > 1000:
                break

    model.load_state_dict(best_params)
    return model


#######################################################

start_time = time.time()


fn = os.path.join(os.getcwd(), "graph_projector_models",
                  "exp1_graph_projector_%d_%d.pt" % (args.num_remove_nodes, args.hop_neighbors))
if os.path.exists(fn) and not args.regen_model:
    model_optim = GNN(x_dims, h_dims, y_dims, device, args).to(device)
    model_optim.load_state_dict(torch.load(fn))
else:
    model_optim = pre_train(model)
    torch.save(model_optim.state_dict(), fn)
print("train model time", time.time() - start_time)
#######################################################

#################################################
#################################################
#################################################


@torch.no_grad()
def evaluation_reuse_labels(model):
    model = model.to(device)

    #################################################
    # directly predict
    all_pred = []

    seed_everything(args.seed)

    pbar = tqdm(total=len(all_loader))
    pbar.set_description('Direct predict >>>')

    for subgraph_data in all_loader:
        pred = model(subgraph_data.to(device))
        all_pred.append(pred.detach().cpu())
        pbar.update(1)
    pbar.close()

    all_pred = torch.cat(all_pred, dim=0)
    train_acc, val_acc, test_acc = pred_test(all_pred, data, split_idx,
                                             evaluator)
    print(
        f"Direct predict >>> Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
    )

    print(
        ">>> Number of nodes are predicted as the last category",
        torch.sum(all_pred[split_idx["train"]].argmax(
            dim=1) == dataset.num_classes).item(),
    )

    # reuse predicted labels
    y_one_hot_tmp = copy.deepcopy(data.y_one_hot_train)
    y_one_hot_tmp[split_idx["test"]] = F.one_hot(
        all_pred[split_idx["test"]].argmax(dim=-1, keepdim=True).squeeze(),
        data.y_one_hot_train.size(1)).float()

    #################################################
    # label reuse
    all_pred = []

    seed_everything(args.seed)
    pbar = tqdm(total=len(all_loader))
    pbar.set_description('Label reuse >>>')

    for subgraph_data in all_loader:
        subgraph_data.y_one_hot_train = y_one_hot_tmp[subgraph_data.node_inds]
        pred = model(subgraph_data.to(device))
        all_pred.append(pred.detach().cpu())
        pbar.update(1)
    pbar.close()

    all_pred = torch.cat(all_pred, dim=0)
    train_acc, val_acc, test_acc = pred_test(all_pred, data, split_idx,
                                             evaluator)
    print(
        f"Label reuse >>> Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
    )
    
    is_classified = all_pred[split_idx["train"]].argmax(dim=1) == dataset.num_classes
    print(
        ">>> Number of nodes are predicted as the last category",
        torch.sum(is_classified).item(), torch.where(is_classified)[0]
    )


evaluation_reuse_labels(model_optim)

#################################################
#################################################
#################################################
# projection-based unlearning

remain_nodes = np.arange(num_nodes)
feat_dim = data.x.size(1)
label_dim = data.y_one_hot_train.size(1)

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
    W_optim_part_unlearn = []
    extra_channel_norm_before = 0
    extra_channel_norm_after = 0
    if args.compress_dim:
        W_optim_part = torch.split(W_optim, [feat_dim, label_dim])

        W_part = W_optim_part[0]
        XtX = remain_node_feats.T@remain_node_feats
        XtX_inv = torch.linalg.pinv(XtX)
        proj_W_optim = XtX@XtX_inv@W_part
        W_optim_part_unlearn.append(proj_W_optim)
        extra_channel_norm_before += W_part[-1, :].norm(2).item()
        extra_channel_norm_after += proj_W_optim[-1, :].norm(2).item()

        W_part = W_optim_part[1]
        XtX = remain_node_label.T@remain_node_label
        XtX_inv = torch.linalg.pinv(XtX)
        proj_W_optim = XtX@XtX_inv@W_part
        W_optim_part_unlearn.append(proj_W_optim)
        extra_channel_norm_before += W_part[-1, :].norm(2).item()
        extra_channel_norm_after += proj_W_optim[-1, :].norm(2).item()
    else:
        W_optim_part = torch.split(W_optim, [feat_dim for _ in range(
            args.x_iters+1)] + [label_dim for _ in range(args.y_iters)])

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
    print('extra_channel_norm_before', extra_channel_norm_before, 'extra_channel_norm_after', extra_channel_norm_after)
    
    W_optim = torch.cat(W_optim_part_unlearn, dim=0)

    # evaluate
    print('Unlearning step %d >>>' % (cnt+1))

model_unlearn = copy.deepcopy(model_optim)
model_unlearn.W.data = W_optim

# get the graph after node deletion

# data.x[delete_nodes_all] = 0
# data.y_one_hot_train[delete_nodes_all] = 0

# remain_nodes = np.setdiff1d(np.arange(num_nodes), delete_nodes_all)
# adj_t_scipy_remain = data.adj_t.to_scipy().tocsr(
# )[remain_nodes, :][:, remain_nodes].tocoo()
# data.adj_t = SparseTensor(
#     row=torch.from_numpy(remain_nodes[adj_t_scipy_remain.row]).long(),
#     col=torch.from_numpy(remain_nodes[adj_t_scipy_remain.col]).long(),
#     value=torch.from_numpy(adj_t_scipy_remain.data).float(),
#     sparse_sizes=(num_nodes, num_nodes)
# )
# all_loader = ShaDowKHopSampler(data,
#                                depth=2,
#                                num_neighbors=args.hop_neighbors,
#                                batch_size=512,
#                                num_workers=10,
#                                shuffle=False)
print("Total time:", time.time() - start_time)

evaluation_reuse_labels(model_unlearn)
