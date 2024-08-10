import torch
from sklearn.neighbors import kneighbors_graph

print(torch.__version__)

from torch import nn, optim
import torch.nn.functional as f
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset


from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import re
from gurobipy import *
import datetime
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax, log_softmax
from inspect import signature
from collections import OrderedDict
dtype = torch.float
device =torch.device("cpu")
import logging
import warnings
from torch_geometric.nn import SAGEConv, GATConv, DenseSAGEConv, GATv2Conv, DenseGATConv
from torch_geometric.data import Data, Batch



# for cvrp baseline: random solver for comparison, easy to implement heuristic, min distance, the cvrp paper by prof guns
# for friday, intro edits + visualize the solutions
# for wednesday re-think RQ's, get feedback and


def make_onehot(vec,num_classes):
    vec_ =  vec.reshape(len(vec), 1)

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


class GATBased(nn.Module):
    def __init__(self,
                 stop_embedding_size=12,
                 nnodes=74,
                 gnn_repr_size=32,
                 edge_repr_size=32,
                 nweekdays=7,
                 drop_prob=0.1,
                 attention_heads=8,
                 feat_emb_size=3,
                 sparsify = 1
                 ):
        super().__init__()

        self.nweekdays = nweekdays
        self.nnodes = nnodes
        self.model_name = "GAT"
        self.model_specs = (gnn_repr_size, edge_repr_size, attention_heads, feat_emb_size)

        # self.demand_emb = nn.Embedding(nnodes, stop_embedding_size)
        self.stop_embedding = nn.Embedding(self.nnodes, feat_emb_size)
        self.demand_emb = nn.Embedding(13, 2)
        self.demand_lin = nn.Linear(2, feat_emb_size)
        self.vehicle_emb = nn.Embedding(15, feat_emb_size)
        self.capacity_emb = nn.Embedding(26, feat_emb_size)
        self.weekday_emb = nn.Embedding(nweekdays, feat_emb_size)
        self.stop_emb = nn.Embedding(nnodes, feat_emb_size)


        self.pregnn = DenseGATConv(1, gnn_repr_size)
        self.gnn1 = GATConv(1, gnn_repr_size, edge_dim=1, heads=attention_heads, dropout=drop_prob)
        self.gnn2 = GATConv(attention_heads * gnn_repr_size, gnn_repr_size, edge_dim=1, heads=attention_heads, dropout=drop_prob)

        self.dropout = nn.Dropout(p=drop_prob)

        self.edge_summarizer = nn.Linear(2 * (gnn_repr_size * attention_heads),  edge_repr_size)

        input_length = self.nnodes * edge_repr_size + self.nnodes + self.nnodes + feat_emb_size*3 + self.nnodes
        print(input_length)
        self.combiner1 = nn.Linear(input_length, nnodes)
        self.combiner2 = nn.Linear(nnodes, nnodes)

        self.sparsify = sparsify

    def forward(self, features, mask):

        dist, stops, weekday, vehicles, markov, demand, capacity = features

        dist = (dist - dist.min()) / (dist.max() - dist.min())
        # print(f"dist: {markov[:5, :5]}")

        x = demand.view(self.nnodes, -1)
        # print(f"x in:{x[:5]} || {x[:5].shape} || {x[:5].ndim}")

        # x = self.demand_emb(x)
        # x = self.demand_lin(x)
        # x = torch.tanh(x)
        # x = x.view(1, self.nnodes, -1).double()
        # mask = mask.view(1, self.nnodes, self.nnodes).int()
        # print(x.shape, x.dtype)
        # print(mask.dtype)
        # x = self.pregnn(x, mask)
        # print(f"x out intro:{x[:5]} || {x[:5].shape} || {x[:5].ndim}")



        if self.sparsify == 0:
            edge_index = np.argwhere(mask == 1)
        elif self.sparsify == 1:
            edge_index = cs_sparsify(self.stop_embedding.weight, mask, 0)
        elif self.sparsify == 2:
            edge_index = knn_sparsify(dist, 10, stops)
        elif self.sparsify == 3:
            mm = markov > 0.1
            mm = mask * mm
            edge_index = np.argwhere(mm == 1)


        # print(f"ei in gnn2:{edge_index}")
        row, col = edge_index
        edge_attr = markov[row, col].to(torch.double)

        # print(np.unique(edge_index), stops, mask)
        # print(f"ei in gnn1:{edge_index}")


        x = self.gnn1(x.double(), edge_index, edge_attr.double())
        x = torch.relu(x)
        # print(f"x out gnn1:{x[:5]} || {x[:5].shape} || {x[:5].ndim}")


        x = self.gnn2(x, edge_index, edge_attr)
        x = torch.relu(x)

        # print(f"x out gnn2:{x[:5]} || {x[:5].shape} || {x[:5].ndim}")

        # print(x[0], x[31])

        i, j = torch.meshgrid(torch.arange(self.nnodes), torch.arange(self.nnodes), indexing='ij')
        i = i.reshape(-1)  # Flatten the index tensors
        j = j.reshape(-1)  # Flatten the index tensors

        edge_repr = torch.cat((x[i], x[j]), dim=1).reshape(self.nnodes, self.nnodes, -1)
        # edge_repr = edge_repr * mask.unsqueeze(-1)
        # edge_repr[mask, mask, :] = 0

        edge_repr = edge_repr.reshape(self.nnodes * self.nnodes, -1)

        edge_repr = self.edge_summarizer(edge_repr)
        # edge_repr = torch.relu(edge_repr)
        # edge_repr = self.dropout(edge_repr)
        edge_repr = edge_repr.view(self.nnodes, -1)

        # print(f"edge_repr:{edge_repr[:5, :5]} || {edge_repr.shape} || {edge_repr[:5, :5*4].ndim}")


        weekday_feat = self.weekday_emb(torch.tensor(weekday)).expand(self.nnodes, -1)
        # print(capacity)
        capacity_feat = self.capacity_emb(torch.tensor(capacity)).expand(self.nnodes, -1)
        vehicle_feat = self.vehicle_emb(torch.tensor(vehicles)).expand(self.nnodes, -1)

        stop_feat = torch.zeros((1, self.nnodes))
        stop_feat[:, stops] = 1
        stop_feat = stop_feat.expand(self.nnodes, -1)


        # print(f"wf:{weekday_feat[0]} || cf: {capacity_feat[0]} || vf:{vehicle_feat[0]}")

        # print(weekday_feat.shape, capacity_feat, vehicle_feat)


        combined_feat = torch.cat([edge_repr, dist, markov, weekday_feat, capacity_feat, vehicle_feat, stop_feat], dim=1).to(dtype=torch.float32)

        out = self.combiner1(combined_feat.double())
        out = torch.relu(out)
        out = self.combiner2(out)
        # print(f"out :{out[:5, :5]} || {out[:5, :5].shape} || {out[:5, :5].ndim}")

        return out




class SAGEBased(nn.Module):
    def __init__(self,
                 stop_embedding_size=12,
                 nnodes=74,
                 gnn_repr_size=32,
                 edge_repr_size=32,
                 nweekdays=7,
                 drop_prob=0.1,
                 attention_heads=6
                 ):
        super().__init__()

        self.stop_embeddings = nn.Embedding(nnodes, stop_embedding_size)
        print(stop_embedding_size, gnn_repr_size)
        self.gnn1 = DenseSAGEConv(stop_embedding_size, gnn_repr_size)
        self.gnn2 = GATConv(gnn_repr_size, gnn_repr_size, heads=attention_heads, dropout=drop_prob)
        self.model_name = "GAT"

        self.dropout = nn.Dropout(p=drop_prob)

        self.edge_summarizer = nn.Linear(2 * (gnn_repr_size * attention_heads),  edge_repr_size)

        input_length = nnodes * edge_repr_size + 7 + 74 + 74

        self.combiner1 = nn.Linear(input_length, nnodes)
        self.combiner2 = nn.Linear(nnodes, nnodes)

        self.nweekdays = nweekdays
        self.nnodes = nnodes


        self.lmodels = nn.ModuleList([nn.Linear(edge_repr_size, 1) for _ in range(self.nnodes)])


    def forward(self, features, mask):

        dist, stops, weekday, vehicles, markov, demand, capacity = features

        dist = (dist - dist.min()) / (dist.max() - dist.min())


        # else:
        #     x = self.stop_embeddings.weight

        x = self.stop_embeddings(demand)
        x = self.gnn1(x, mask)[0]
        markov = torch.from_numpy(markov)


        # estimate connections
        # x = torch.nn.functional.normalize(x, p=2, dim=1)
        # connections = torch.mm(x, x.t()).fill_diagonal_(0)
        # connections = connections * mask
        # connections = connections > 0
        # edge_index = np.argwhere(connections == 1)
        a = x.detach().numpy()
        adjacency_matrix = kneighbors_graph(a, 5, mode='connectivity', include_self=False)
        edge_index = np.array(adjacency_matrix.nonzero())

        edge_index = torch.from_numpy(edge_index).to(torch.int64)

        row, col = edge_index
        edge_attr = markov[row, col]
        x = torch.relu(x)

        # connections = torch.mm(x, x.t()).fill_diagonal_(0)
        # connections = connections * mask
        # connections = connections > 0.5
        # edge_index = np.argwhere(connections == 1)
        # row, col = edge_index
        # edge_attr = markov[row, col]

        x = self.gnn2(x, edge_index, edge_attr)
        x = torch.relu(x)

        i, j = torch.meshgrid(torch.arange(self.nnodes), torch.arange(self.nnodes), indexing='ij')
        i = i.reshape(-1)  # Flatten the index tensors
        j = j.reshape(-1)  # Flatten the index tensors

        edge_repr = torch.cat((x[i], x[j]), dim=1).reshape(self.nnodes, self.nnodes, -1)
        # edge_repr = edge_repr * mask.unsqueeze(-1)
        # edge_repr[mask, mask, :] = 0

        edge_repr = edge_repr.reshape(self.nnodes * self.nnodes, -1)

        edge_repr = self.edge_summarizer(edge_repr)
        edge_repr = torch.relu(edge_repr)
        edge_repr = self.dropout(edge_repr)
        edge_repr = edge_repr.view(self.nnodes, -1)


        weekday_feat = torch.zeros(1, 7)
        weekday_feat[:, weekday] = 1
        weekday_feat = weekday_feat.expand(self.nnodes, -1)
        vehicle_feat = torch.full((self.nnodes, 1), vehicles)

        # distances = dist[i, j].unsqueeze(1)
        stop_feat = torch.zeros((self.nnodes, 1))
        stop_feat[stops] = 1

        # edge_repr = edge_repr.view(self.nnodes, self.nnodes, -1)
        # out = torch.zeros(self.nnodes, self.nnodes)
        # for i, m in enumerate(self.lmodels):
        #     cer = edge_repr[i]
        #     out[i] = torch.relu(m(cer)).view(-1, self.nnodes)
        #

        #
        combined_feat = torch.cat([edge_repr, dist, markov, weekday_feat], dim=1).to(dtype=torch.float32)
        #
        # # combined_feat = torch.cat([edge_representations, distances], dim=1).to(dtype=torch.float32)
        out = self.combiner1(combined_feat)
        out = torch.relu(out)
        out = self.combiner2(out)
        # out = nn.LeakyReLU()(out)
        # out = self.additional(out)
        # out = out.view(self.nnodes, -1)

        # res = nn.LogSoftmax(dim=1)(out)

        return out




class EGL(nn.Module):
    def __init__(self,
                 lookback_period = 30,
                 stop_embedding_size=12,
                 nnodes=74,
                 preference_dim=32,
                 nweekdays=7,
                 drop_prob=0.1,
                 model=0,
                 edge_features=False
                 ):
        super().__init__()


        self.stop_embeddings = nn.Embedding(nnodes, stop_embedding_size)

        self.gnn = None
        self.model_name = ""
        if model == 0:
            self.gnn = SAGEConv(stop_embedding_size, preference_dim)
            self.model_name = "SAGEConv"
        elif model == 1:
            self.gnn = GCNConv(stop_embedding_size, preference_dim)
            self.model_name = "GCN"
        elif model == 2:
            attention_heads = 6
            self.gnn = GATConv(stop_embedding_size, preference_dim, heads=attention_heads, dropout=drop_prob)
            self.model_name = "GAT"

        # self.dropout = nn.Dropout(p=drop_prob)

        input_length = nnodes * 2 + 1 + 1 + 1 + (preference_dim if model < 2 else preference_dim * attention_heads)
        # input_length = preference_dim * 2 + 1


        self.node_summarizer = nn.Linear(preference_dim, nnodes**2)
        self.combiner = nn.Linear(input_length, nnodes)
        self.edge_summarizer = nn.Linear(2 * (preference_dim if model < 2 else preference_dim * attention_heads),  1)
        # self.additional = nn.Linear(4, 1)
        self.nweekdays = nweekdays
        self.nnodes = nnodes
        self.lookback_period = lookback_period
        self.edge_features = edge_features

    def forward(self, edge_index, features):

        dist, stops, weekday, vehicles = features

        x = self.stop_embeddings.weight
        connections = nn.functional.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2).fill_diagonal_(0)
        # connections = torch.mm(x, x.t()).fill_diagonal_(0)

        mask = torch.ones(self.nnodes, dtype=torch.bool)
        mask[stops] = False
        connections[mask, :] = 0
        connections[:, mask] = 0
        connections = (connections > 0.5).type(torch.IntTensor)
        edge_index = torch.argwhere(connections).t()


        if self.edge_features:
            edge_attr = torch.tensor([dist[a, b] for a, b in zip(edge_index[0], edge_index[1])])

        preferences = self.gnn(x, edge_index, edge_attr if self.edge_features else None)
        # print(preferences.shape)



        preference_feat = preferences.view(self.nnodes, -1)

        # print(node_summaries.shape)

        i, j = torch.meshgrid(torch.arange(self.nnodes), torch.arange(self.nnodes), indexing='ij')
        i = i.reshape(-1)  # Flatten the index tensors
        j = j.reshape(-1)  # Flatten the index tensors

        edge_representations = torch.cat((preferences[i], preferences[j]), dim=1)
        #print(edge_representations.shape)

        edge_summaries = self.edge_summarizer(edge_representations)
        edge_summaries = nn.LeakyReLU()(edge_summaries)
        edge_summaries = edge_summaries.view(self.nnodes, -1)


        # preference_feat = nn.LeakyReLU()(preference_feat)


        # weekday_feat = torch.zeros((self.nnodes, 1))
        # weekday_feat[weekday] = 1

        weekday_feat = torch.full((self.nnodes, 1), weekday)
        vehicle_feat = torch.full((self.nnodes, 1), vehicles)

        distances = dist[i, j].unsqueeze(1)
        stop_feat = torch.zeros((self.nnodes, 1))
        stop_feat[stops] = 1

        combined_feat = torch.cat([preference_feat, edge_summaries, dist, weekday_feat, vehicle_feat, stop_feat], dim=1).to(dtype=torch.float32)
        # combined_feat = torch.cat([edge_representations, distances], dim=1).to(dtype=torch.float32)
        out = self.combiner(combined_feat)
        # out = nn.LeakyReLU()(out)
        # out = self.additional(out)
        # out = out.view(self.nnodes, -1)

        mlog = nn.LogSoftmax(dim=1)
        res = mlog(out)
        return res







class GNNSAGE(nn.Module):
    def __init__(self, embedding_size,
                 lookback_period, stop_embedding_size=12, target_dim=1, nnodes=74, n_features=2,
                 nweekdays=7, onehot=False, drop_prob=0.01, weekly=False,
                 decision_focused=False, **kwargs):
        super().__init__()
        self.weekly = weekly

        self.nweekdays = nweekdays
        self.droput = nn.Dropout(p=drop_prob)
        self.fc1 = nn.Linear(lookback_period, 1)
        if weekly:
            self.fc2 = nn.Linear(n_features + 2 + stop_embedding_size, 1)
        elif onehot:
            self.fc2 = nn.Linear(nweekdays + n_features + 2 + stop_embedding_size, 1)
        else:
            self.fc2 = nn.Linear(embedding_size + n_features + 2 + stop_embedding_size, 1)
            self.embeddings = nn.Embedding(nweekdays, embedding_size)


        self.stop_embeddings = nn.Embedding(nnodes, stop_embedding_size)

        self.gnn = SAGEConv(lookback_period, 1)
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period = lookback_period
        self.decision_focused = decision_focused

    def forward(self, stops, x, x_dist, x_features, x_week, x_mask):
        # print(f"x shape: {x.shape}, size: {x.size}, dtype: {x.dtype}, dimensions: {x.ndim}")
        # print(f"x_dist shape: {x_dist.shape}, size: {x_dist.size}, dtype: {x_dist.dtype}, dimensions: {x_dist.ndim}")
        # print(f"x_features shape: {x_features.shape}, size: {x_features.size}, dtype: {x_features.dtype}, dimensions: {x_features.ndim}")
        # # print(f"x_markov shape: {x_markov.shape}, size: {x_markov.size}, dtype: {x_markov.dtype}, dimensions: {x_markov.ndim}")
        # print(f"x_week shape: {x_week.shape}, size: {x_week.size}, dtype: {x_week.dtype}, dimensions: {x_week.ndim}")

        edge_index = self.create_edge_index(stops)
        # print(x.size(), x_dist.size(), x_features.size(), edge_index.size(), edge_index.max())
        x_embed = torch.stack( tuple( self.stop_embeddings(torch.LongTensor(s)).sum(0) for s in stops))
        # print(x_embed.size(), x_embed)
        print(x.size())


        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        batch = self.create_batched_graph(x, stops)
        out = self.gnn(batch.x, batch.edge_index)
        # out = self.fc1(x.transpose(1, 2))

        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb = make_onehot(x_week, self.nweekdays)  # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week)  # (*,embedding_size)
            x_ = torch.cat([x_emb, x_features], 1)
        x_ = torch.cat([x_.unsqueeze(1).expand(n_rows, self.nnodes, -1),
                        x_embed.unsqueeze(1).expand(n_rows, self.nnodes, -1),
                        out.view(n_rows, self.nnodes, -1), x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows, -1, 1)], -1)
        n_f = x_.shape[-1]
        # print(x_.size(), self.fc2)
        x_ = self.fc2(x_).squeeze(-1)
        print(x_.size())


        # batch = self.create_batched_graph(x_, stops)
        # print(batch.x.size())
        # self.gnn(batch.x, batch.edge_index)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(), 1e8)
        else:
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(), -1e8)

    def create_edge_index(self, stops):
        edge_list = []

        for trajectory in stops:
            # Iterate through each stop in the trajectory to create edges
            for i in range(len(trajectory) - 1):
                edge_list.append((trajectory[i], trajectory[i + 1]))

        # Convert to a tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        return edge_index

    def trajectory_to_edge_index(self, trajectory):
        edge_index = [[trajectory[i], trajectory[i + 1]] for i in range(len(trajectory) - 1)]
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def create_batched_graph(self, x_train, list_of_trajectories):
        data_list = []
        # print("xtrain ", x_train.size())
        for i in range(x_train.shape[0]):
            x = x_train[i].t()
            # print(x.size())
            trajectory = list_of_trajectories[i]
            edge_index = self.trajectory_to_edge_index(trajectory)
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)
        batched_data = Batch.from_data_list(data_list)
        return batched_data



class GNNAttention(nn.Module):
    def __init__(self, embedding_size,
                 lookback_period, stop_embedding_size=12, target_dim=1, nnodes=74, n_features=2,
                 nweekdays=7, onehot=False, drop_prob=0.01, weekly=False,
                 decision_focused=False, **kwargs):
        super().__init__()
        self.weekly = weekly

        self.nweekdays = nweekdays
        self.droput = nn.Dropout(p=drop_prob)
        self.fc1 = nn.Linear(lookback_period, 1)
        if weekly:
            self.fc2 = nn.Linear(n_features + 3 + stop_embedding_size, 1)
        elif onehot:
            self.fc2 = nn.Linear(nweekdays + n_features + 3 + stop_embedding_size, 1)
        else:
            self.fc2 = nn.Linear(embedding_size + n_features + 3 + stop_embedding_size, 1)
            self.embeddings = nn.Embedding(nweekdays, embedding_size)


        self.stop_embeddings = nn.Embedding(nnodes, stop_embedding_size)

        self.gnn = SAGEConv(50, 1)
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period = lookback_period
        self.decision_focused = decision_focused

    def forward(self, stops, x, x_dist, x_features, x_week, x_mask):
        # for s in stops:
        #     print(len(s))
        np.info(x)
        np.info(x_dist)
        np.info(x_features)
        np.info(x_week)
        edge_index = self.create_edge_index(stops)
        # print(x.size(), x_dist.size(), x_features.size(), edge_index.size(), edge_index.max())
        x_embed = torch.stack( tuple( self.stop_embeddings(torch.LongTensor(s)).sum(0) for s in stops))
        # print(x_embed.size(), x_embed)

        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)

        out = self.fc1(x.transpose(1, 2))

        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb = make_onehot(x_week, self.nweekdays)  # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week)  # (*,embedding_size)
            x_ = torch.cat([x_emb, x_features], 1)
        x_ = torch.cat([x_.unsqueeze(1).expand(n_rows, self.nnodes, -1),
                        x_embed.unsqueeze(1).expand(n_rows, self.nnodes, -1),
                        out, x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows, -1, 1)], -1)
        n_f = x_.shape[-1]
        # print(x_.size())
        batch = self.create_batched_graph(x_, stops)
        x_ = self.gnn(batch.x, batch.edge_index).view(-1, self.nnodes)
        # x_ = self.fc2(x_).squeeze(-1)
        # print(x_.size())
        # batch = self.create_batched_graph(x_, stops)
        # print(batch.x.size())
        # self.gnn(batch.x, batch.edge_index)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(), 1e8)
        else:
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(), -1e8)

    def create_edge_index(self, stops):
        edge_list = []

        for trajectory in stops:
            # Iterate through each stop in the trajectory to create edges
            for i in range(len(trajectory) - 1):
                edge_list.append((trajectory[i], trajectory[i + 1]))

        # Convert to a tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        return edge_index

    def trajectory_to_edge_index(self, trajectory):
        edge_index = [[trajectory[i], trajectory[i + 1]] for i in range(len(trajectory) - 1)]
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def create_batched_graph(self, x_train, list_of_trajectories):
        data_list = []
        # print("xtrain ", x_train.size())
        for i in range(x_train.shape[0]):
            x = x_train[i]
            # print(x.size())
            trajectory = list_of_trajectories[i]
            edge_index = self.trajectory_to_edge_index(trajectory)
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)
        batched_data = Batch.from_data_list(data_list)
        return batched_data