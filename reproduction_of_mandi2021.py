import os
from comparison_models import *
import logging
import pandas as pd
from trainer import VRPTrainer
from data_process import obtain_data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(filename='runs/PaddedModel.log',
                    level=logging.INFO, format=formatter)

stops, n_vehicles, weekday, capacities, demands, opmat, stop_wise_days, distance_mat, edge_mat = obtain_data("data")

test_days = [154, 160, 166, 173, 180, 187, 194,
             155, 161, 167, 174, 181, 188, 195,
             149, 156, 168, 175, 182, 189, 196,
             150, 162, 169, 176, 183, 190, 197,
             157, 163, 170, 177, 184, 191, 198,
             158, 164, 171, 178, 185, 192, 199,
             159, 165, 172, 179, 186, 193, 200]

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

