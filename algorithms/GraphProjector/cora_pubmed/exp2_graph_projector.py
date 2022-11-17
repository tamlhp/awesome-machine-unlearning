import argparse

import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric import seed_everything
from torch_sparse import SparseTensor


import time
import copy
import numpy as np

from model_utils import GNN 

################################################################
parser = argparse.ArgumentParser(description="Planetoid (GNN)")
parser.add_argument("--dataset", type=str, default='Cora')
parser.add_argument("--lr", type=float, default=1.0)
parser.add_argument("--num_remove_nodes", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

# load dataset
dataset = args.dataset
num_remove_nodes = args.num_remove_nodes

path = osp.join('data', dataset)
dataset = Planetoid(path, dataset,
                    transform=T.ToSparseTensor(),
                    split='full'
                    )
data = dataset[0]
seed_everything(args.seed)


print(dataset, num_remove_nodes)
################################################################
# inject feature + label
train_inds = torch.where(data.train_mask)[0].cpu().numpy()

num_train_nodes = len(train_inds)
train_ind_perm = np.random.permutation(train_inds)
if num_remove_nodes < 1:
    num_remove_nodes = int(num_train_nodes * num_remove_nodes)
else:
    num_remove_nodes = int(num_remove_nodes)
print('Remove nodes %d/%d' % (num_remove_nodes, num_train_nodes))
delete_nodes_all = train_ind_perm[:num_remove_nodes]

# print(data.x.shape)
################################################################
# training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

model = GNN(data.x.size(1), dataset.num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

start_time = time.time()
best_val = 0

epoch = 0
while True:
    epoch += 1
    model.train()
    pred = model.pred(data.adj_t, data.x)
    loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    mask = data.train_mask
    train_acc = pred[mask].max(1)[1].eq(
        data.y[mask]).sum().item() / mask.sum().item()
    mask = data.val_mask
    val_acc = pred[mask].max(1)[1].eq(
        data.y[mask]).sum().item() / mask.sum().item()
    mask = data.test_mask
    test_acc = pred[mask].max(1)[1].eq(
        data.y[mask]).sum().item() / mask.sum().item()

    if best_val < val_acc:
        best_train = train_acc
        best_val = val_acc
        best_test = test_acc
        best_epoch = epoch
        best_params = copy.deepcopy(model.state_dict())
    # print("train >>>", train_acc, val_acc, test_acc)
    
    if best_epoch < epoch - 200:
        print('Run epochs', best_epoch)
        break

model.load_state_dict(best_params)
print("Final", best_train, best_val, best_test, 
      "Time", time.time()- start_time)

num_delete_nodes = len(delete_nodes_all)

all_unlearn_W = []
all_retrain_W = []

for i in range(1, num_delete_nodes):
################################################################
    # unlearning
    W_optim = model.W.data.clone().cpu()

    start_time = time.time()
    remain_nodes = np.setdiff1d(train_inds, delete_nodes_all[:i])
    remain_node_feats = data.x[remain_nodes].cpu()

    XtX = remain_node_feats.T@remain_node_feats
    XtX_inv = torch.linalg.pinv(XtX)
    proj_W_optim = XtX@XtX_inv@W_optim

    extra_channel_norm_before = W_optim[-1, :].norm(2).item()
    extra_channel_norm_after = proj_W_optim[-1, :].norm(2).item()

    print('time', time.time() - start_time, 
          'weight norm', extra_channel_norm_before, extra_channel_norm_after)
    
    all_unlearn_W.append(proj_W_optim)

    ################################################################
    # evaluate
    # model_unlearn = GNN(data.x.size(1), dataset.num_classes).to(device)
    # model_unlearn.W.data = proj_W_optim.to(device)
    # model_unlearn.eval()
    # pred = model_unlearn.pred(data.adj_t, data.x)

    # mask = data.train_mask
    # train_acc = pred[mask].max(1)[1].eq(
    #     data.y[mask]).sum().item() / mask.sum().item()
    # mask = data.val_mask
    # val_acc = pred[mask].max(1)[1].eq(
    #     data.y[mask]).sum().item() / mask.sum().item()
    # mask = data.test_mask
    # test_acc = pred[mask].max(1)[1].eq(
    #     data.y[mask]).sum().item() / mask.sum().item()

    # print("Unlearn >>>", train_acc, val_acc, test_acc)
    
    ################################################################
    # re-training
    adj_t_scipy_remain = data.adj_t.to_scipy('csr')[remain_nodes, :][:, remain_nodes].tocoo()
    train_inds_remain = np.intersect1d(remain_nodes, train_inds)

    num_nodes = data.x.size(0)
    adj_t_remain = SparseTensor(
        row=torch.from_numpy(remain_nodes[adj_t_scipy_remain.row]).long(),
        col=torch.from_numpy(remain_nodes[adj_t_scipy_remain.col]).long(),
        value=torch.from_numpy(adj_t_scipy_remain.data).float(),
        sparse_sizes=(num_nodes, num_nodes)
    ).to(device)
    
    
    # retraining
    model_retrain = GNN(data.x.size(1), dataset.num_classes).to(device)
    optimizer = torch.optim.SGD(model_retrain.parameters(), lr=args.lr, momentum=0.9)

    start_time = time.time()
    best_val = 0

    epoch = 0
    while epoch < best_epoch:
        epoch += 1
        model_retrain.train()
        pred = model_retrain.pred(adj_t_remain, data.x)
        loss = F.cross_entropy(pred[train_inds_remain], data.y[train_inds_remain])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_retrain.eval()
        mask = data.train_mask
        train_acc = pred[mask].max(1)[1].eq(
            data.y[mask]).sum().item() / mask.sum().item()
        mask = data.val_mask
        val_acc = pred[mask].max(1)[1].eq(
            data.y[mask]).sum().item() / mask.sum().item()
        mask = data.test_mask
        test_acc = pred[mask].max(1)[1].eq(
            data.y[mask]).sum().item() / mask.sum().item()

    # print("Final", train_acc, val_acc, test_acc, 
    #       "Time", time.time()- start_time)

    W_retrain = model_retrain.W.data.clone().cpu()
    all_retrain_W.append(W_retrain)
    
results = []

for j, (W_retrain, W_unlearn) in enumerate(zip(all_retrain_W, all_unlearn_W)):
    # weight params
    print((W_retrain - W_unlearn).norm(2), (W_optim - W_unlearn).norm(2), (W_optim-W_retrain).norm(2), W_optim.norm(2))
    
    # generate feats
    i = j + 1
    model_unlearn = GNN(data.x.size(1), dataset.num_classes).to(device)
    model_unlearn.W.data = W_unlearn.to(device)
    model_unlearn.eval()
    pred_unlearn = F.softmax(model_unlearn.pred(adj_t_remain, data.x), dim=1)

    model_retrain = GNN(data.x.size(1), dataset.num_classes).to(device)
    model_retrain.W.data = W_retrain.to(device)
    model_retrain.eval()
    pred_retrain = F.softmax(model_retrain.pred(adj_t_remain, data.x), dim=1)
    
    delete_nodes = delete_nodes_all[:i]
    remain_nodes = np.setdiff1d(train_inds, delete_nodes)
    
    delete_node_pred_diff = (pred_unlearn[delete_nodes] - pred_retrain[delete_nodes]).norm(2).item()
    remain_node_pred_diff = (pred_unlearn[delete_nodes] - pred_retrain[delete_nodes]).norm(2).item()
    
    cur_results = [
        (W_retrain - W_unlearn).norm(2).item(), 
        (W_optim - W_unlearn).norm(2).item(), 
        (W_optim-W_retrain).norm(2).item(), 
        W_optim.norm(2).item(),
        (pred_unlearn[delete_nodes] - pred_retrain[delete_nodes]).norm(2).item(),
        (pred_unlearn[delete_nodes] - pred_retrain[delete_nodes]).norm(2).item(),
        (pred_unlearn[data.test_mask] - pred_retrain[data.test_mask]).norm(2).item(),
    ]
    results.append(cur_results)
    
import json

with open('results/%s_weight_results_projection_seed%d.json'%(args.dataset, args.seed), 'w') as f:
    json.dump(results, f)