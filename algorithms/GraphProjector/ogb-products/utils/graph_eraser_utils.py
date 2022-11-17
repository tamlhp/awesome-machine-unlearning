import os 
import pickle
import numpy as np
import scipy.sparse as sp
import time

def get_graph_partitions(num_clusters, data, regen=False):

    save_parts_path = os.path.join('baseline_cache_info','%d_clusters.pkl'%(num_clusters))

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