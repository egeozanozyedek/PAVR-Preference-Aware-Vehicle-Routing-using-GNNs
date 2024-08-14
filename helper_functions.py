
import torch
import numpy as np



def make_onehot(vec,num_classes):
    vec_ = vec.reshape(len(vec), 1)
    one_hot_target = (vec_ == torch.arange(num_classes).reshape(1, num_classes)).float()
    return one_hot_target


def cs_sparsify(x, mask, threshold=0):
    x = torch.nn.functional.normalize(x, p=2, dim=1)
    connections = torch.mm(x, x.t()).fill_diagonal_(0)
    # print(connections[0])
    connections = connections > threshold
    connections[0] = 1
    connections = connections * mask

    return np.argwhere(connections == 1)


def knn_sparsify(distance_matrix, k, stops):
    num_nodes = distance_matrix.shape[0]
    adj = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in stops:
        adj[i, i] = 0
        subset_distances = distance_matrix[i, stops]
        neighbors_idx = np.argsort(subset_distances)[:k + 1]
        actual_neighbors = [stops[idx] for idx in neighbors_idx if stops[idx] != i]
        for neighbor in actual_neighbors:
            adj[i, neighbor] = 1

    return torch.from_numpy(np.argwhere(adj == 1)).t()
