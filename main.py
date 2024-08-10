import copy
import os
from datetime import datetime

from matplotlib import pyplot as plt

from comparison_models import *
from data.Util import VRPsolutiontoList

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
import pandas as pd
import numpy as np
from trainer import VRPTrainer
from gnn_trainer import GNNTrainer, NewTrainer
from ind_trainer import IndTrainer
from models import GNNSAGE, GNNAttention, EGL, GATBased, SAGEBased
import networkx as nx
from visualize import visualize

from data_process import obtain_data

formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(filename='runs/PaddedModel.log',
                    level=logging.INFO, format=formatter)

stops, n_vehicles, weekday, capacities, demands, opmat, stop_wise_days, distance_mat, edge_mat = obtain_data("data")

test_days = [172, 180]

# test_days = [154, 160, 166, 173, 180, 187, 194,
#              155, 161, 167, 174, 181, 188, 195,
#              149, 156, 168, 175, 182, 189, 196,
#              150, 162, 169, 176, 183, 190, 197,
#              157, 163, 170, 177, 184, 191, 198,
#              158, 164, 171, 178, 185, 192, 199,
#              159, 165, 172, 179, 186, 193, 200]

net_comp_list = [
    MarkovwthStopembedding, NoHist, NoWeek, NoDist, NoMarkov, OnlyMarkov]

hyper_comp_dict = {
    MarkovwthStopembedding: (50, 0.1),
    NoWeek: (50, 0.1),
    NoDist: (100, 0.1),
    NoHist: (100, 0.1),
    NoMarkov: (100, 0.1),
    OnlyMarkov: (100, 0.1)}


lookback = 30
stop_embedding_size = 40
lst = []

t = 154
ss = stops[t]
im = opmat[t]
# routing = VRPsolutiontoList(im)
# print(routing)
# visualize(routing)


# G = nx.from_numpy_array(im)
# non_isolated_nodes = [node for node in G.nodes if G.degree(node) > 0]

# Create a subgraph with non-isolated nodes
# G_non_isolated = G.subgraph(non_isolated_nodes)

# Plot the subgraph

#
# nx.draw(G_non_isolated, with_labels=True, node_color='lightblue', edge_color='gray')
# plt.show()

get_comp=False

if get_comp:
    lst = []
    for net in net_comp_list:
        epochs, lr = hyper_comp_dict[net]
        for t in test_days:
            if t == 153:
                continue

            model = VRPTrainer(epochs=epochs,
                                lookback_period=lookback,
                                lr=lr,
                                net=net,
                                stop_embedding=True,
                                n_features=1,
                                stop_embedding_size=stop_embedding_size)

            ev = model.evaluation(distance_mat, stops[:(t + 1)],
                                  weekday[:(t + 1)], n_vehicles[:(t + 1)],
                                  opmat[:(t + 1)], stop_wise_days, demands[t], capacities[t:(t + 1)],
                                  capacitated=True)



            rez = {"Name": net,
                   "Day": t,
                   "lookback": lookback, "stop_embedding_size": stop_embedding_size,
                   "epochs": epochs, "lr": lr,
                   "Model": str(net),
                   "bceloss": ev[2],
                   "training_bcelos": ev[3],
                   "Arc Difference": ev[0][0], "Arc Difference(%)": ev[0][1],
                   "Route Difference": ev[1][0],
                   "Route Difference(%)": ev[1][1],
                   "Distance": ev[4],
                   "Comment": ev[5]}


            lst.append(rez)
            print(rez, "\n\n")



        df = pd.DataFrame(lst)

        print(f"Model: {net}")
        print(f"Training BCE: {df['training_bcelos'].mean()}")
        print(f"Test BCE: {df['bceloss'].mean()}")
        print(f"Arc Diff: {df['Arc Difference'].mean()}")
        print(f"Arc Diff %: {df['Arc Difference(%)'].mean() * 100}")
        print(f"Route Diff: {df['Route Difference'].mean()}")
        print(f"Route Diff %: {df['Route Difference(%)'].mean() * 100}")






split = int(len(opmat) * 0.75)
print(split)

permutation = np.random.permutation(split)

training_days = opmat[:split][permutation]
training_stops = stops[:split][permutation]
training_weekdays = weekday[:split][permutation]
training_vehicles = n_vehicles[:split][permutation]
training_demands = demands[:split][permutation]
training_capacities = capacities[:split][permutation]

test_days = opmat[split + 1:]
test_stops = stops[split + 1:]
test_weekdays = weekday[split + 1:]
test_vehicles = n_vehicles[split + 1:]
print(test_weekdays)

daily_routes = opmat
daily_stops = stops
daily_weekdays = weekday
daily_vehicles = n_vehicles
daily_demands = demands
daily_capacities = capacities

stop_embedding_size = 32



gnn2 = GATBased()

model1 = NewTrainer(lookback_period=lookback, model=gnn2)




print(max([np.max(x) for x in demands]))
print(max(n_vehicles))
print(max(capacities))


in_model = model1.model
in_markov = model1.markov_model
epochs = 10
learning_rate = 1e-2

in_model, in_markov = model1.fit(distance_mat,
                                 training_stops,
                                 training_weekdays,
                                 training_vehicles,
                                 training_days,
                                 training_demands,
                                 training_capacities,
                                 epochs=epochs,
                                 learning_rate=learning_rate)


lst = []
lookback = 30
trial_name = "GAT_Demands_CosSim_NoInitEmbed"
for i in range(len(test_days)):



    day = int(i + split + 1)

    td = test_days[i]
    ts = test_stops[i]
    tw = test_weekdays[i]
    tv = test_vehicles[i]

    print("aa", (day-lookback), day)

    primer = (distance_mat,
              daily_stops[(day-lookback):day],
              daily_weekdays[(day-lookback):day],
              daily_vehicles[(day-lookback):day],
              daily_routes[(day - lookback):day],
              demands[(day - lookback):day],
              capacities[(day - lookback):day],
              copy.deepcopy(in_model),
              copy.deepcopy(in_markov),
              10,
              0.01)

    dems = demands[day]
    cap = capacities[day]

    # primer = None
    # td = np.zeros_like(td)
    # td[0, 3] = 1
    # td[0, 4] = 1
    # td[1, 2] = 1
    # td[2, 0] = 1
    # td[3, 1] = 1
    # td[4, 0] = 1
    #
    # print(td[:5, :5])
    # print(ts)
    # ts = [0, 1, 2, 3, 4]
    # print(ts)
    # tv = 2
    # tw = 1
    # dems = np.zeros_like(demands[day])
    # dems[1] = 2
    # dems[2] = 3
    # dems[3] = 2
    # dems[4] = 9
    # cap = 10
    # print(tv, td, tw)
    # print(demands[day])
    # print(capacities[day])



    ev = model1.predict(distance_mat, ts, tw, tv, td, dems, cap, primer)

    rez = {"Name": trial_name,
           "Day": day,
           "lookback": lookback,
           "epochs": epochs, "lr": learning_rate,
           "Model": in_model.model_name,
           "Model_Specs": in_model.model_specs,
           "Markov_Specs": in_markov.specs,
           "bceloss": ev[2],
           "training_bcelos": ev[3],
           "Arc Difference": ev[0][0], "Arc Difference(%)": ev[0][1],
           "Route Difference": ev[1][0],
           "Route Difference(%)": ev[1][1],
           "Distance": ev[4],
           "Comment": ev[5],
           }


    print(rez)
    lst.append(rez)


df = pd.DataFrame(lst)
df.to_csv(f"{datetime.now()} - embedded_demands.csv", sep=',', index=True, encoding='utf-8')

# df = pd.read_csv("embedded_demands.csv", sep=',',)

filtered_df = df.loc[df['Model'] == in_model.model_name]
print(f"--Model: {in_model.model_name}")
print(f"Training BCE: {filtered_df['training_bcelos'].mean()}")
print(f"Test BCE: {filtered_df['bceloss'].mean()}")
print(f"Arc Diff: {filtered_df['Arc Difference'].mean()}")
print(f"Arc Diff %: {filtered_df['Arc Difference(%)'].mean() * 100}")
print(f"Route Diff: {filtered_df['Route Difference'].mean()}")
print(f"Route Diff %: {filtered_df['Route Difference(%)'].mean() * 100}")





#
#
# lst = []
# # visualize?
# for t in test_days:
#     print(f"Test Day: {t}")
#     for net in net_list:
#         epochs, lr = hyper_dict[net]
#
#         model1 = GNNTrainer(net=net,
#                             epochs=epochs,
#                             lr=lr,
#                             stop_embedding_size=stop_embedding_size,
#                             lookback_period=lookback,
#                             n_features=n_features,
#                             model=gnn1)
#
#
#
#         # model2 = StopTrainer(net=net,
#         #                     epochs=epochs,
#         #                     lr=lr,
#         #                     stop_embedding_size=stop_embedding_size,
#         #                     lookback_period=lookback,
#         #                     n_features=n_features,
#         #                     model=gnn2)
#
#         ev = model1.evaluation(distance_mat, stops[:(t + 1)],
#                                weekday[:(t + 1)], n_vehicles[:(t + 1)],
#                                opmat[:(t + 1)], demands[t], capacities[t],
#                                capacitated=True)
#
#
#
#         if ev is None:
#             continue
#
#         A, P = ev[-1]
#
#         G = nx.Graph()
#         G.add_nodes_from(stops[t])
#         # distance_matrix = distance_mat
#         # N = distance_matrix.shape[0]
#         # for i in range(N):
#         #     for j in range(i + 1, N):  # Avoid duplicate edges (i+1 for undirected graph)
#         #         if distance_matrix[i][j] > 0:  # Add an edge if distance is greater than 0
#         #             G.add_edge(i, j, weight=distance_matrix[i][j])
#
#         # Plot the graph
#         # pos = nx.spring_layout(G)  # Generate a layout for the nodes
#         # nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700)
#         # labels = nx.get_edge_attributes(G, 'weight')
#         # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#
#         plt.figure(figsize=(25, 10))
#
#         plt.subplot(1, 2, 2)
#         plt.title("Predicted")
#         pos = visualize(P)
#
#         plt.subplot(1, 2, 1)
#         plt.title(f"Actual, Day: {t}")
#         visualize(A, pos)
#
#         plt.savefig(f"figures/{t}_{gnn1.model_name}.png", bbox_inches='tight')
#         plt.close()
#
#         rez = {"Name": net,
#                "Day": t,
#                "lookback": lookback, "stop_embedding_size": stop_embedding_size,
#                "epochs": epochs, "lr": lr,
#                "Model": str(net),
#                "bceloss": ev[2],
#                "training_bcelos": ev[3],
#                "Arc Difference": ev[0][0], "Arc Difference(%)": ev[0][1],
#                "Route Difference": ev[1][0],
#                "Route Difference(%)": ev[1][1],
#                "Distance": ev[4],
#                "Comment": ev[5]}
#
#         print(rez)
#
#         # model = VRPTrainer(epochs=epochs, lookback_period=lookback,
#         #                lr=lr, net=net, stop_embedding=True, n_features=1,
#         #                stop_embedding_size=stop_embedding_size)
#         #
#         # ev = model.evaluation(distance_mat, stops[:(t + 1)],
#         #                   weekday[:(t + 1)], n_vehicles[:(t + 1)],
#         #                   opmat[:(t + 1)], stop_wise_days, demands[t], capacities[t:(t + 1)],
#         #                   capacitated=True)
#         #
#
#         lst.append(rez)
#
# df = pd.DataFrame(lst)
# df.to_csv("embedded_demands.csv", sep=',', index=True, encoding='utf-8')
#
# for net in net_list:
#     filtered_df = df.loc[df['Model'] == str(net)]
#     print(f"--Model: {str(net)}")
#     print(f"Training BCE: {filtered_df['training_bcelos'].mean()}")
#     print(f"Test BCE: {filtered_df['bceloss'].mean()}")
#     print(f"Arc Diff: {filtered_df['Arc Difference'].mean()}")
#     print(f"Arc Diff %: {filtered_df['Arc Difference(%)'].mean() * 100}")
#     print(f"Route Diff: {filtered_df['Route Difference'].mean()}")
#     print(f"Route Diff %: {filtered_df['Route Difference(%)'].mean() * 100}")
