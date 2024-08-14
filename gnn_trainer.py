import copy

import torch
from torch import nn, optim
import numpy as np
import inspect

dtype = torch.float
device = torch.device("cpu")

from data.Util import VRPGurobi, VRPsolutiontoList, eval_ad, eval_sd
from models import MarkovCounting


def adjacency_to_edge_index(adj_matrix):
    edge_indices = adj_matrix.nonzero()
    # Transpose to get 2xN format

    edge_index = edge_indices.t().contiguous()
    return edge_index


class PAVR:
    def __init__(self,
                 model=None,
                 lookback_period=30,
                 nnodes=74,
                 optimizer=optim.Adam):

        self.lookback_period = lookback_period
        self.nnodes = nnodes
        self.optimizer = optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.markov_model = MarkovCounting()
        self.training_loss = 0
        self.model = model

    def fit(self,
            distance_mat,
            training_stops,
            training_weekdays,
            training_vehicles,
            training_days,
            demands,
            capacities,
            in_model=None,
            in_markov=None,
            epochs=25,
            learning_rate=1e-3):


        if in_model:
            model = in_model.double()
        else:
            model = self.model.double()

        if in_markov:
            markov_model = in_markov
        else:
            markov_model = self.markov_model


        optimizer_graph = self.optimizer(model.parameters(), lr=learning_rate)

        self.training_loss = 0
        training_loss = []

        distance_mat = torch.from_numpy(distance_mat)

        markov_model.fit(training_days, training_weekdays, distance_mat)


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

        print(f"Loss: {sum(training_loss) / len(training_loss)} ")
        self.training_loss = sum(training_loss) / len(training_loss)

        return model, markov_model


    def predict(self,
                distance_mat,
                stops,
                weekday,
                vehicles,
                target,
                demand,
                capacity,
                warmup=None):
        #
        # np.set_printoptions(precision=2, suppress=True)
        # torch.set_printoptions(precision=2)

        if warmup:
            model, markov_model = self.fit(*warmup)
        else:
            model = self.model
            markov_model = self.markov_model

        demand = torch.tensor(demand, dtype=torch.int)
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


        solved, cmt, sol, _ = VRPGurobi(predicted, demand, capacity, vehicles, stops)

        if solved:
            solution = np.rint(sol)
            print(sol)
            P = VRPsolutiontoList(solution)
            A = VRPsolutiontoList(target)

            print("predicted", P)
            print("actual", A)

            return (eval_ad(P, A),
                    eval_sd(P, A),
                    loss,
                    self.training_loss,
                    np.sum(distance_mat * sol),
                    cmt,
                    (A, P))


        else:
            return None













