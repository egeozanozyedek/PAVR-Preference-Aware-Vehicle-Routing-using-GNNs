import torch
from torch import nn, optim
import numpy as np
import inspect
dtype = torch.float
device = torch.device("cpu")


from data.Util import VRPGurobi, VRPsolutiontoList, eval_ad, eval_sd
reluop = nn.ReLU()




class IndTrainer:
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

        optimizer_graph = self.optimizer(model.parameters(),
                                         **self.optim_dict)

        # training
        for ep in range(self.epochs):
            for i in range(self.lookback_period):
                day = -self.lookback_period+i
                # x = will be embeddings
                y = torch.from_numpy(graph_past[day])
                features = (distance_mat,
                            stops_past[day],
                            weekday_past[day],
                            vehicles_past[day])

                # maybe use this for training and the other for evaluation only? but then the model might think something else because of fully connectedness
                edge_index = adjacency_to_edge_index(y)

                optimizer_graph.zero_grad()
                op = model(edge_index, features)

                criterion = nn.NLLLoss()
                CELoss = criterion(op, y.argmax(dim=1))

                CELoss.backward()
                optimizer_graph.step()
                training_loss.append(CELoss.item())

        print(f"Loss: {sum(training_loss)/len(training_loss)}")
        model.eval()

        features = (distance_mat, target_stops, target_weekday, target_vehicles)
        target_graph = torch.from_numpy(target_graph)


        edge_index = adjacency_to_edge_index(target_graph)

        # edge_index = torch.combinations(torch.from_numpy(np.asarray(target_stops)), r=2).t().contiguous()

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



        proba_mat = self.fit_predict(distance_mat, stops_list, weekday, n_vehicleslist, target)
        criterion = nn.NLLLoss()  # nn.BCELoss()
        bceloss = criterion(proba_mat[target_stops, :][:, target_stops], torch.from_numpy(act[target_stops, :][:, target_stops]).argmax(dim=1)).item()

        # mask = torch.zeros_like(proba_mat)
        # mask[target_stops, :][:, target_stops] = 1
        # proba_mat[mask == 0] = -1e8

        proba_mat = - proba_mat.detach().cpu().numpy()
        '''
        the zeros come becuase of masking in the solution
        log of zero make it infinity, make this infinity with any scaler
        doesn't matter because the constraint specifies only active
        stops t be considered
        '''

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



