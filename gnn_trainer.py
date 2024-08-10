import copy

import torch
from torch import nn, optim
import numpy as np
import inspect

dtype = torch.float
device = torch.device("cpu")

from data.Util import VRPGurobi, VRPsolutiontoList, eval_ad, eval_sd

reluop = nn.ReLU()


def adjacency_to_edge_index(adj_matrix):
    edge_indices = adj_matrix.nonzero()
    # Transpose to get 2xN format

    edge_index = edge_indices.t().contiguous()
    return edge_index


class GNNTrainer:
    def __init__(self, net, lookback_period=30,
                 weekly=False,
                 nnodes=74,
                 embedding_size=6,
                 n_features=2,
                 optimizer=optim.Adam,
                 epochs=20,
                 stop_embedding_size=10,
                 model=None,
                 **kwargs):

        self.net = net
        self.lookback_period = lookback_period
        self.weekly = weekly
        self.nnodes = nnodes
        self.embedding_size = embedding_size
        self.epochs = epochs

        self.kwargs = kwargs
        self.stop_embedding_size = stop_embedding_size
        self.n_features = n_features
        optim_args = [k for k, v in inspect.signature(optimizer).parameters.items()]
        self.optim_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in optim_args}

        self.optimizer = optimizer
        self.training_loss = 0
        self.model = model

        i, j = torch.combinations(torch.arange(nnodes), 2).t()
        edge_index = torch.stack((i, j), dim=0)
        self.complete_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

    def fit_predict(self, distance_mat, stops_list, weekday, n_vehicleslist, target):

        distance_mat = torch.from_numpy(distance_mat)
        graph_past = target[:-1]
        stops_past = stops_list[:-1]
        weekday_past = weekday[:-1]
        vehicles_past = n_vehicleslist[:-1]

        target_stops = stops_list[-1]
        target_weekday = weekday[-1]
        target_vehicles = n_vehicleslist[-1]
        target_graph = target[-1]

        self.training_loss = 0
        training_loss = []

        model = self.model

        optimizer_graph = self.optimizer(model.parameters(), lr=1e-2)

        # training
        for ep in range(self.epochs):
            for i in range(self.lookback_period):
                day = -self.lookback_period + i

                # x = will be embeddings
                y = torch.from_numpy(graph_past[day])
                features = (distance_mat,
                            stops_past[day],
                            weekday_past[day],
                            vehicles_past[day])

                # maybe use this for training and the other for evaluation only? but then the model might think something else because of fully connectedness
                # edge_index = adjacency_to_edge_index(y)
                # edge_index = torch.combinations(torch.from_numpy(np.asarray(stops_past[day])), r=2).t().contiguous()

                # getting the fully connected subgraph
                stops_day = torch.tensor(stops_past[day])
                mask = (stops_day[:, None] == self.complete_index[0]).any(dim=0) & (stops_day[:, None] == self.complete_index[1]).any(dim=0)
                subgraph_edge_index = self.complete_index[:, mask]

                optimizer_graph.zero_grad()
                op = model(subgraph_edge_index, features)

                criterion = nn.NLLLoss()
                criterion = nn.BCEWithLogitsLoss()
                # CELoss = criterion(op, y.argmax(dim=1))
                CELoss = criterion(op, y)

                # CELoss = -(op * y).sum() / len(y)

                CELoss.backward()
                optimizer_graph.step()
                training_loss.append(CELoss.item())

        print(f"Loss: {sum(training_loss) / len(training_loss)} ")
        model.eval()

        features = (distance_mat, target_stops, target_weekday, target_vehicles)
        target_graph = torch.from_numpy(target_graph)

        # edge_index = adjacency_to_edge_index(target_graph)

        edge_index = torch.combinations(torch.from_numpy(np.asarray(target_stops)), r=2).t().contiguous()

        predicted = model(edge_index, features)

        model.train()
        self.training_loss = np.mean(training_loss)

        return predicted

    def evaluation(self, distance_mat, stops_list, weekday, n_vehicleslist,
                   target, demands, capacities, capacitated=True):
        '''
        demands and capacities are of day t
        rest are till day t
        this will be fed directly to predict,
        which takes care of extracting the past

        '''
        trgt_past = target[:-1]
        # stops_list_past = stops_list[:-1]
        # weekday_past = weekday[:-1]
        # n_vehicleslist_past = n_vehicleslist[:-1]
        target_stops = stops_list[-1]
        print(target_stops)
        act = target[-1]

        # proba_mat = self.fit_predict(distance_mat, stops_list, weekday, n_vehicleslist, target)
        proba_mat = self.fit_predict_splitted(distance_mat, stops_list, weekday, n_vehicleslist, target)

        criterion = nn.NLLLoss()  # nn.BCELoss()
        bceloss = criterion(proba_mat[target_stops, :][:, target_stops], torch.from_numpy(act[target_stops, :][:, target_stops]).argmax(dim=1)).item()

        # mask = torch.zeros_like(proba_mat)
        # mask[target_stops, :][:, target_stops] = 1
        # proba_mat[mask == 0] = -1e8

        proba_mat = - proba_mat.detach().cpu().numpy()

        if capacitated:
            qcapacity = demands
            Q = capacities
        else:
            qcapacity = np.ones(74)
            Q = len(target_stops)
        solved, cmnt, sol, u = VRPGurobi(proba_mat, qcapacity, Q,
                                         n_vehicleslist[-1], target_stops)
        if solved:
            sol = np.rint(sol)
            P = VRPsolutiontoList(sol)
            A = VRPsolutiontoList(act)

            print("predicted", P)
            print("actual", A)
            # diff = act - sol

            # print(" Arc Difference {} {} Route Difference {}".format(np.sum( diff* (diff >0) ),
            #  eval_ad (P,A), eval_sd(P,A)))
        else:
            return None
            # raise Exception("VRP not solved for day {}".format(len(trgt_past)))
        return (eval_ad(P, A),
                eval_sd(P, A),
                bceloss,
                self.training_loss,
                np.sum(distance_mat * sol),
                cmnt,
                (A, P))



class MarkovCounting:

    def __init__(self, alpha=0.01, beta=0.5, weight_schema=1):
        self.T = {}
        self.D = None
        self.alpha = alpha
        self.beta = beta
        self.weight_schema = weight_schema
        self.exp = 0.01
        self.specs=(alpha,beta,weight_schema)

        if weight_schema == 0:
            self.weight = lambda x, y: 1  #
        elif weight_schema == 1:
            self.weight = lambda x, y: x / y  # time based
        elif weight_schema == 2:
            self.weight = lambda A, B: float(len(A.intersection(B))) / len(A.union(B))  # jaccard

    def fit(self, historical_transitions, weekdays, distance_mat):
        C = np.exp(-distance_mat)
        np.fill_diagonal(C, 0)
        self.D = C / np.sum(C, axis=1, keepdims=True)

        # C = np.sum(distance_mat, axis=1, keepdims=True) / distance_mat
        # np.fill_diagonal(C, 0)
        #
        # self.D = C / np.sum(C, axis=1, keepdims=True)
        # np.fill_diagonal(self.D, 0)

        for d in np.unique(weekdays):

            if self.weight_schema == 0:
                F = np.sum(historical_transitions[weekdays == d], axis=0)


            if self.weight_schema == 1:

                N = len(historical_transitions[weekdays == d])
                weights = np.flip(np.arange(N))
                weights = self.exp ** weights
                F = np.sum(historical_transitions[weekdays == d] * weights[:, np.newaxis, np.newaxis], axis=0)

            F += self.alpha

            totals = np.sum(F, axis=1, keepdims=True)
            self.T[d] = F / totals

        return self.T, self.D

    def predict(self, weekday):

        return self.T[weekday] * self.beta + (1 - self.beta) * self.D


class NewTrainer:
    def __init__(self, lookback_period=30,
                 weekly=False,
                 nnodes=74,
                 optimizer=optim.Adam,
                 model=None,
                 **kwargs):

        self.lookback_period = lookback_period
        self.weekly = weekly
        self.nnodes = nnodes

        optim_args = [k for k, v in inspect.signature(optimizer).parameters.items()]
        self.optim_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in optim_args}

        self.optimizer = optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.markov_model = MarkovCounting()
        self.training_loss = 0
        self.model = model

        i, j = torch.combinations(torch.arange(nnodes), 2).t()
        edge_index = torch.stack((i, j), dim=0)
        self.complete_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

    def fit(self, distance_mat, training_stops, training_weekdays, training_vehicles, training_days, demands, capacities, in_model=None, in_markov=None, epochs=25, learning_rate=1e-3):


        model = self.model.double()
        markov_model = self.markov_model


        if in_model:
            model = in_model.double()

        if in_markov:
            markov_model = in_markov


        self.training_loss = 0
        training_loss = []

        markov_model.fit(training_days, training_weekdays, distance_mat)

        distance_mat = torch.from_numpy(distance_mat)

        optimizer_graph = self.optimizer(model.parameters(), lr=learning_rate)

        # training
        for ep in range(epochs):
            for day in range(len(training_days)):

                y = torch.from_numpy(training_days[day])

                stops = torch.tensor(training_stops[day])
                mask = torch.zeros((self.nnodes, self.nnodes))
                grid = torch.meshgrid(stops, stops)
                mask[grid] = 1
                mask[stops, stops] = 0

                wd = training_weekdays[day]
                markov_features = torch.from_numpy(markov_model.predict(wd))
                demand = torch.tensor(demands[day], dtype=torch.int)
                capacity = capacities[day]

                features = (distance_mat,
                            training_stops[day],
                            wd,
                            training_vehicles[day],
                            markov_features,
                            demand,
                            capacity)


                optimizer_graph.zero_grad()

                op = model(features, mask)
                op = op * mask

                CELoss = self.criterion(op, y)

                CELoss.backward()
                optimizer_graph.step()
                training_loss.append(CELoss.item())

            print(f"Epoch: {ep}, Loss: {np.average(training_loss)}")
            # call predict
            # self.model.train()

        print(f"Loss: {sum(training_loss) / len(training_loss)} ")
        self.training_loss = sum(training_loss) / len(training_loss)


        return model, markov_model


    def predict(self, distance_mat, stops, weekday, vehicles, target, demand, capacity, primer):

        np.set_printoptions(precision=2, suppress=True)
        torch.set_printoptions(precision=2)
        demand = torch.tensor(demand, dtype=torch.int)

        model = self.model
        markov_model = self.markov_model
        if primer:
            model, markov_model = self.fit(*primer)


        markov_feat = torch.from_numpy(markov_model.predict(weekday))
        features = (torch.from_numpy(distance_mat),
                    stops,
                    weekday,
                    vehicles,
                    markov_feat,
                    demand,
                    capacity)

        stops = torch.tensor(stops)
        mask = torch.zeros((self.nnodes, self.nnodes))
        grid = torch.meshgrid(stops, stops)
        mask[grid] = 1
        mask[stops, stops] = 0

        model.eval()
        with torch.no_grad():
            predicted = model(features, mask)

        loss = self.criterion(predicted, torch.from_numpy(target)).item()
        predicted = torch.sigmoid(predicted)

        predicted = predicted.detach().cpu().numpy()

        # print(f"pred:{predicted[:5, :5]} || {predicted[:5, :5].shape} || {predicted[:5, :5].ndim}")


        solved, cmnt, sol, u = VRPGurobi(predicted, demand, capacity, vehicles, stops)
        # _, _, solm, _ = VRPGurobi(markov_feat, demands, capacities, vehicles, stops)

        if solved:
            solution = np.rint(sol)
            print(sol)
            P = VRPsolutiontoList(solution)
            A = VRPsolutiontoList(target)
            # M = VRPsolutiontoList(np.rint(solm))

            print("predicted", P)
            print("actual", A)
            # print("Markov", M)
            # print("actual????", B)
            # diff = act - sol

            # print(f" Arc Difference {eval_ad(M,A)} Route Difference {eval_sd(M,A)}")
        else:
            return None
        return (eval_ad(P, A),
                eval_sd(P, A),
                loss,
                self.training_loss,
                np.sum(distance_mat * sol),
                cmnt,
                (A, P))















# class NewStopTrainer:
#     def __init__(self, net, lookback_period=30,
#                  weekly=False,
#                  nnodes=74,
#                  embedding_size=6,
#                  n_features=2,
#                  optimizer=optim.Adam,
#                  epochs=20,
#                  stop_embedding_size=10,
#                  model=None,
#                  **kwargs):
#
#         self.net = net
#         self.lookback_period = lookback_period
#         self.weekly = weekly
#         self.nnodes = nnodes
#         self.embedding_size = embedding_size
#         self.epochs = epochs
#
#         self.kwargs = kwargs
#         self.stop_embedding_size = stop_embedding_size
#         self.n_features = n_features
#         optim_args = [k for k, v in inspect.signature(optimizer).parameters.items()]
#         self.optim_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in optim_args}
#
#         self.optimizer = optimizer
#         self.criterion = nn.NLLLoss()
#         self.markov_model = MarkovCounting()
#         self.training_loss = 0
#         self.model = model
#
#     def fit(self, distance_mat, training_stops, training_weekdays, training_vehicles, training_days):
#
#         self.training_loss = 0
#         training_loss = []
#
#         self.markov_model.fit(training_days, training_weekdays, distance_mat)
#
#         distance_mat = torch.from_numpy(distance_mat)
#
#         optimizer_graph = self.optimizer(self.model.parameters(), lr=1e-3)
#
#         # training
#         for ep in range(self.epochs):
#             for i in range(len(training_days)):
#
#                 stops = torch.tensor(training_stops[i])
#                 wd = training_weekdays[i]
#                 markov_features = self.markov_model.predict(wd)
#                 vehicles = training_vehicles[i]
#
#                 for s in stops:
#
#                     y = torch.from_numpy(training_days[i][s])
#
#                     #
#                     # mask = torch.zeros((self.nnodes, self.nnodes))
#                     # grid = torch.meshgrid(stops, stops)
#                     # mask[grid] = 1
#                     # mask[stops, stops] = 0
#                     #
#
#
#
#
#                     features = (distance_mat[s],
#                                 training_stops[i],
#                                 wd,
#                                 vehicles,
#                                 markov_features[s])
#
#
#                     optimizer_graph.zero_grad()
#                     op = self.model(features, mask)
#                     op = op * mask
#
#                     CELoss = self.criterion(op , y)
#
#                     CELoss.backward()
#                     optimizer_graph.step()
#                     training_loss.append(CELoss.item())
#
#             print(f"Epoch: {ep}, Loss: {np.average(training_loss)}")
#             # call predict
#             # self.model.train()
#
#         print(f"Loss: {sum(training_loss) / len(training_loss)} ")
#         self.training_loss = sum(training_loss) / len(training_loss)
#
#
#     def predict(self, distance_mat, stops, weekday, vehicles, target, demands, capacities):
#
#         self.model.eval()
#         markov_feat = self.markov_model.predict(weekday)
#         features = (torch.from_numpy(distance_mat), stops, weekday, vehicles, markov_feat)
#
#         stops = torch.tensor(stops)
#         mask = torch.zeros((self.nnodes, self.nnodes), dtype=torch.bool)
#         mask[torch.meshgrid(stops, stops)] = 1
#         mask[stops, stops] = 0
#
#         with torch.no_grad():
#             predicted = self.model(features, mask)
#
#         loss = self.criterion(predicted, torch.from_numpy(target)).item()
#         predicted = torch.sigmoid(predicted)
#
#         predicted = - predicted.detach().cpu().numpy()
#
#
#         solved, cmnt, sol, u = VRPGurobi(predicted, demands, capacities, vehicles, stops)
#         _, _, solm, _ = VRPGurobi(markov_feat, demands, capacities, vehicles, stops)
#
#         if solved:
#             solution = np.rint(sol)
#
#
#             P = VRPsolutiontoList(solution)
#             A = VRPsolutiontoList(target)
#             M = VRPsolutiontoList(np.rint(solm))
#
#             print("predicted", P)
#             print("actual", A)
#             print("Markov", M)
#             # print("actual????", B)
#             # diff = act - sol
#
#             print(f" Arc Difference {eval_ad(M,A)} Route Difference {eval_sd(M,A)}")
#         else:
#             return None
#             # raise Exception("VRP not solved for day {}".format(len(trgt_past)))
#         return (eval_ad(P, A),
#                 eval_sd(P, A),
#                 loss,
#                 self.training_loss,
#                 np.sum(distance_mat * sol),
#                 cmnt,
#                 (A, P))
