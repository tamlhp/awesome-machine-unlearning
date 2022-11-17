import argparse

import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Reddit
import torch_geometric.transforms as T
from torch_geometric import seed_everything

import time
import copy
import numpy as np

from model_utils import GNN 

################################################################
parser = argparse.ArgumentParser(description="Planetoid (GNN)")
parser.add_argument("--dataset", type=str, default='Cora')
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--num_remove_nodes", type=float, default=0.02)

args = parser.parse_args()

# load dataset
dataset = args.dataset
num_remove_nodes = args.num_remove_nodes

path = osp.join('data', dataset)
if dataset in ['Cora', 'Pubmed']:
    dataset = Planetoid(path, dataset,
                        transform=T.ToSparseTensor(),
                        split='full'
                        )
else:
    dataset = Reddit(path,
                      transform=T.ToSparseTensor()
    )
data = dataset[0]
seed_everything(0)


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

extra_feats = torch.zeros(data.x.size(0))
extra_feats[delete_nodes_all] = 1

data.x = torch.cat([data.x, extra_feats.view(-1, 1)], dim=1)
data.y[delete_nodes_all] = dataset.num_classes

# print(data.x.shape)
################################################################
# training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = GNN(data.x.size(1), dataset.num_classes).to(
    device), data.to(device)

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

#     print("train >>>", train_acc, val_acc, test_acc)
    
    if best_epoch < epoch - 200:
        print('Run epochs', best_epoch)
        break

model.load_state_dict(best_params)
print("Final", best_train, best_val, best_test, 
      "Time", time.time()- start_time)

################################################################
# unlearning

W_optim = model.W.data.clone().cpu()

start_time = time.time()
remain_nodes = np.setdiff1d(train_inds, delete_nodes_all)
remain_node_feats = data.x[remain_nodes].cpu()

XtX = remain_node_feats.T@remain_node_feats
XtX_inv = torch.linalg.pinv(XtX)
proj_W_optim = XtX@XtX_inv@W_optim

extra_channel_norm_before = W_optim[-1, :].norm(2).item()
extra_channel_norm_after = proj_W_optim[-1, :].norm(2).item()

print('time', time.time() - start_time, 
      'weight norm', extra_channel_norm_before, extra_channel_norm_after)

################################################################
# evaluate
model.W.data = proj_W_optim.to(device)
model.eval()
pred = model.pred(data.adj_t, data.x)

mask = data.train_mask
train_acc = pred[mask].max(1)[1].eq(
    data.y[mask]).sum().item() / mask.sum().item()
mask = data.val_mask
val_acc = pred[mask].max(1)[1].eq(
    data.y[mask]).sum().item() / mask.sum().item()
mask = data.test_mask
test_acc = pred[mask].max(1)[1].eq(
    data.y[mask]).sum().item() / mask.sum().item()

print("Unlearn >>>", train_acc, val_acc, test_acc)
