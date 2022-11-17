import numpy as np
import scipy.sparse as sp
import math

import os
import pickle
import time
from tqdm import tqdm

import torch_sparse

import torch
import torch.nn as nn
import torch.nn.functional as F


from .train_utils import sym_norm

####################################################################################################
####################################################################################################
####################################################################################################


def get_2_hop_neighbors(num_nodes, adj_t_scipy, sample_size=(20, 20 ** 2)):
    neighbor_nodes = []

    pbar = tqdm(total=num_nodes)
    pbar.set_description("Compute neighbors")

    for node_i in range(num_nodes):
        one_hop_neighbors = np.setdiff1d(adj_t_scipy[node_i].indices, node_i)
        if len(one_hop_neighbors) > sample_size[0]:
            one_hop_neighbors = np.random.permutation(one_hop_neighbors)[
                : sample_size[0]
            ]

        two_hop_neighbors = np.where(
            np.sum(adj_t_scipy[one_hop_neighbors, :], axis=0) > 0
        )[1]
        two_hop_neighbors = np.setdiff1d(two_hop_neighbors, one_hop_neighbors)
        two_hop_neighbors = np.setdiff1d(two_hop_neighbors, node_i)
        if len(two_hop_neighbors) > sample_size[1]:
            two_hop_neighbors = np.random.permutation(two_hop_neighbors)[
                : sample_size[1]
            ]

        neighbors = np.concatenate(
            [
                np.array([node_i]),
                one_hop_neighbors,
                two_hop_neighbors,
            ]
        )
        neighbor_nodes.append(neighbors)

        pbar.update(1)

    subgraph_relation = [[] for _ in range(num_nodes)]
    for neighbors_ in neighbor_nodes:
        node_i, neighbors_ = neighbors_[0], neighbors_[1:]
        for node_j in neighbors_:
            subgraph_relation[node_j].append(node_i)

    return neighbor_nodes, subgraph_relation


####################################################################################################
####################################################################################################
####################################################################################################


def from_scipy_sparse_matrix(A):

    A = A.tocoo()
    N = A.shape[0]

    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)

    A_tensor = torch_sparse.SparseTensor(
        row=row, col=col, sparse_sizes=(N, N), is_sorted=False
    )

    return A_tensor


####################################################################################################
####################################################################################################
####################################################################################################


def compute_node_feats(node_indices, neighbors, adj_t_scipy, X, Y):
    num_layers = 3
    x_dim = X.size(1)
    y_dim = Y.size(1)
    
    # new_dim = x_dim * (num_layers + 1) + y_dim * num_layers
    new_dim = x_dim + y_dim
    node_feat_all = torch.zeros((len(node_indices), new_dim))
    cnt = 0

    pbar = tqdm(total=len(node_indices))
    pbar.set_description("Compute node feats")

    for node_i in node_indices:

        neighbors_ = neighbors[node_i]
        part_adj = from_scipy_sparse_matrix(adj_t_scipy[neighbors_, :][:, neighbors_])

        subgraph_adj_norm = sym_norm(part_adj)

        cur_X = X[neighbors_]
        x_node_feats = [cur_X]
        x_node_feats += [subgraph_adj_norm @ x_node_feats[-1]]
        x_node_feats += [subgraph_adj_norm @ x_node_feats[-1]]
        x_node_feats += [subgraph_adj_norm @ x_node_feats[-1]]
        x_node_feats = torch.stack(x_node_feats).sum(0)

        cur_Y = Y[neighbors_]
        cur_Y[0, :] = 0

        y_node_feats = [subgraph_adj_norm @ cur_Y]
        y_node_feats += [subgraph_adj_norm @ y_node_feats[-1]]
        y_node_feats += [subgraph_adj_norm @ y_node_feats[-1]]
        y_node_feats = torch.stack(y_node_feats).sum(0)
                
        node_feat_all[cnt, :] = torch.cat([x_node_feats, y_node_feats], dim=1)[0, :]
        cnt += 1

        pbar.update(1)

    pbar.close()
    return node_feat_all


####################################################################################################
####################################################################################################
####################################################################################################


class GNN(nn.Module):
    def __init__(self, W):
        super(GNN, self).__init__()
        self.W = torch.nn.Parameter(W)

    def forward(self, X):
        return X @ self.W

    def ovr_lr_loss(self, X, Y, lam):
        Y[Y == 0] = -1
        Z = (X @ self.W).mul_(Y)
        return -F.logsigmoid(Z).mean(0).sum() + lam * self.W.pow(2).sum() / 2


####################################################################################################
####################################################################################################
####################################################################################################
def lr_grad(w, X, y, lam=0):
    y[y == 0] = -1

    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z - 1) * y) + lam * X.size(0) * w


def lr_hessian_inv(w, X, y, lam=0, batch_size=50000):
    y[y == 0] = -1

    z = torch.sigmoid(X.mv(w).mul_(y))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)

    H = H + lam * X.size(0) * torch.eye(X.size(1)).float()

    try:
        H_inv = torch.linalg.inv(H)
    except:
        H_inv = torch.linalg.pinv(H)

    return H_inv

####################################################################################################
####################################################################################################
####################################################################################################

def get_graph_partitions(num_clusters, data, regen=False):

    save_parts_path = os.path.join('%d_clusters.pkl'%(num_clusters))

    if os.path.exists(save_parts_path) and not regen:
        parts = pickle.load(open(save_parts_path, 'rb'))
    else:
        # convert PyG data structure to scipy's sparse_coo for metis partition
        row = data.adj_t.storage.row().numpy()
        col = data.adj_t.storage.col().numpy()
        num_nodes = data.adj_t.size(0)

        all_nodes = np.arange(num_nodes)
        sparse_coo_adj = sp.csr_matrix((np.ones_like(row), (row, col)), shape=(num_nodes, num_nodes))

        parts = _partition_graph(sparse_coo_adj, all_nodes, num_clusters)

        with open(save_parts_path, 'wb') as f:
            pickle.dump(parts, f)
            
    return parts

def _partition_graph(adj, idx_nodes, num_clusters):
    os.environ['METIS_DLL'] = '/home/XXXX-1/Downloads/metis-5.1.0/build/Linux-x86_64/libmetis/libmetis.so'
    import metis
    
    """partition a graph by METIS."""

    start_time = time.time()
    num_nodes = len(idx_nodes)

    train_adj = adj[idx_nodes, :][:, idx_nodes]
    train_adj_lil = train_adj.tolil()
    train_ord_map = dict()
    train_adj_lists = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        rows = train_adj_lil[i].rows[0]
        # self-edge needs to be removed for valid format of METIS
        if i in rows:
            rows.remove(i)
        train_adj_lists[i] = rows
        train_ord_map[idx_nodes[i]] = i

    if num_clusters > 1:
        _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
    else:
        groups = [0] * num_nodes

    parts = [[] for _ in range(num_clusters)]
    for nd_idx in range(num_nodes):
        gp_idx = groups[nd_idx]
        nd_orig_idx = idx_nodes[nd_idx]
        parts[gp_idx].append(nd_orig_idx)
        

    part_size = [len(part) for part in parts]
    print('Partitioning done. %f seconds.'%(time.time() - start_time))
    print('Max part size %d, min part size %d'%(max(part_size), min(part_size)))

    return parts