import copy
from datetime import datetime
import pandas as pd
import numpy as np
from gnn_trainer import PAVR
from models import PAVREncoderDecoder
from data_process import obtain_data
import argparse

# cmd arguments
parser = argparse.ArgumentParser()
parser.add_argument("--eta_w", default=0.01, type=float)
parser.add_argument("--eta_t", default=0.01, type=float)
parser.add_argument("--lookback", default=30, type=int)
parser.add_argument("--d1", default=32, type=int)
parser.add_argument("--d2", default=32, type=int)
parser.add_argument("--d3", default=3, type=int)
parser.add_argument("--attn_heads", default=8, type=int)
parser.add_argument("--beta", default=0.5, type=float)
parser.add_argument("--epochs_w", default=10, type=int)
parser.add_argument("--epochs_t", default=10, type=int)
parser.add_argument("--warmup", default=True, type=bool)

args = parser.parse_args()

# obtain data from the dataset
stops, n_vehicles, weekday, capacities, demands, opmat, stop_wise_days, distance_mat, edge_mat = obtain_data("data")

# split train-test
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

print(args.d1, args.d2, args.d3, args.attn_heads, args.beta)

# define the encoder-decoder model
pavrED = PAVREncoderDecoder(mvehicles=(max(n_vehicles) + 1),
                            mcapacity=(max(capacities) + 1),
                            gnn_repr_size=args.d1,
                            edge_repr_size=args.d2,
                            attention_heads=args.attn_heads,
                            feat_emb_size=args.d3)


# define the entire model pipeline and training
pavr = PAVR(model=pavrED,
            lookback_period=args.lookback,
            beta=args.beta)


# if warmup is desired
if args.warmup:
    in_model, in_markov = pavr.fit(distance_mat,
                                   training_stops,
                                   training_weekdays,
                                   training_vehicles,
                                   training_days,
                                   training_demands,
                                   training_capacities,
                                   epochs=args.epochs_w,
                                   learning_rate=args.eta_w)

else:
    in_model = pavr.model
    in_markov = pavr.markov_model


# inference
lookback = args.lookback
trial_name = "PAVR"
results = []

for i in range(len(test_days)):
    day = int(i + split + 1)

    td = test_days[i]
    ts = test_stops[i]
    tw = test_weekdays[i]
    tv = test_vehicles[i]

    primer = (distance_mat,
              daily_stops[(day - lookback):day],
              daily_weekdays[(day - lookback):day],
              daily_vehicles[(day - lookback):day],
              daily_routes[(day - lookback):day],
              demands[(day - lookback):day],
              capacities[(day - lookback):day],
              copy.deepcopy(in_model),
              copy.deepcopy(in_markov),
              args.epochs_t,
              args.eta_t)

    dems = demands[day]
    cap = capacities[day]

    # Inject toy example code here, found below.

    ev = pavr.predict(distance_mat, ts, tw, tv, td, dems, cap, primer)

    out = {"Name": trial_name,
           "Day": day,
           "lookback": args.lookback,
           "epochs_w": args.epochs_w,
           "epochs_t": args.epochs_t,
           "lr_w": args.eta_w,
           "lr_t": args.eta_t,
           "Model": in_model.model_name,
           "Model_Specs": in_model.model_specs,
           "Markov_Specs": in_markov.specs,
           "bceloss": ev[2],
           "training_bcelos": ev[3],
           "Arc Difference": ev[0][0],
           "Arc Difference(%)": ev[0][1],
           "Route Difference": ev[1][0],
           "Route Difference(%)": ev[1][1],
           "Distance": ev[4],
           "Comment": ev[5],
           }

    print(out)
    results.append(out)

df = pd.DataFrame(results)
df.to_csv(f"PAVR Trial - {datetime.now()}.csv", sep=',', index=True, encoding='utf-8')

# df = pd.read_csv("??.csv", sep=',',)

filtered_df = df.loc[df['Model'] == in_model.model_name]
print(f"--Model: {in_model.model_name}")
print(f"Training BCE: {filtered_df['training_bcelos'].mean()}")
print(f"Test BCE: {filtered_df['bceloss'].mean()}")
print(f"Arc Diff: {filtered_df['Arc Difference'].mean()}")
print(f"Arc Diff %: {filtered_df['Arc Difference(%)'].mean() * 100}")
print(f"Route Diff: {filtered_df['Route Difference'].mean()}")
print(f"Route Diff %: {filtered_df['Route Difference(%)'].mean() * 100}")


# Below is a setup used to obtain the results for the toy example in the thesis. Should be put in where the above comment is.
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