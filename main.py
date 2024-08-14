import copy
import os
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from gnn_trainer import PAVR
from models import PAVREncoderDecoder
from data_process import obtain_data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(filename='runs/PaddedModel.log', level=logging.INFO, format=formatter)

stops, n_vehicles, weekday, capacities, demands, opmat, stop_wise_days, distance_mat, edge_mat = obtain_data("data")

test_days = np.arange(151, 201, 1)

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

lookback = 30
stop_embedding_size = 32

print(max(capacities))

pavr = PAVR(model=PAVREncoderDecoder(mvehicles=max(n_vehicles), mcapacity=max(capacities)),
            lookback_period=lookback)

epochs = 10
learning_rate = 1e-2
warm_up = True

if warm_up:
    in_model, in_markov = pavr.fit(distance_mat,
                                   training_stops,
                                   training_weekdays,
                                   training_vehicles,
                                   training_days,
                                   training_demands,
                                   training_capacities,
                                   epochs=epochs,
                                   learning_rate=learning_rate)
else:
    in_model = pavr.model
    in_markov = pavr.markov_model

lst = []
lookback = 30
trial_name = "GAT_Demands_CosSim_NoInitEmbed"
for i in range(len(test_days)):
    day = int(i + split + 1)

    td = test_days[i]
    ts = test_stops[i]
    tw = test_weekdays[i]
    tv = test_vehicles[i]

    print("aa", (day - lookback), day)

    primer = (distance_mat,
              daily_stops[(day - lookback):day],
              daily_weekdays[(day - lookback):day],
              daily_vehicles[(day - lookback):day],
              daily_routes[(day - lookback):day],
              demands[(day - lookback):day],
              capacities[(day - lookback):day],
              copy.deepcopy(in_model),
              copy.deepcopy(in_markov),
              10,
              0.01)

    dems = demands[day]
    cap = capacities[day]

    # Below is a setup used to obtain the results for the toy example in the thesis
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

    ev = pavr.predict(distance_mat, ts, tw, tv, td, dems, cap, primer)

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
df.to_csv(f"PAVR Trial - {datetime.now()}.csv", sep=',', index=True, encoding='utf-8')

# df = pd.read_csv("embedded_demands.csv", sep=',',)

filtered_df = df.loc[df['Model'] == in_model.model_name]
print(f"--Model: {in_model.model_name}")
print(f"Training BCE: {filtered_df['training_bcelos'].mean()}")
print(f"Test BCE: {filtered_df['bceloss'].mean()}")
print(f"Arc Diff: {filtered_df['Arc Difference'].mean()}")
print(f"Arc Diff %: {filtered_df['Arc Difference(%)'].mean() * 100}")
print(f"Route Diff: {filtered_df['Route Difference'].mean()}")
print(f"Route Diff %: {filtered_df['Route Difference(%)'].mean() * 100}")
