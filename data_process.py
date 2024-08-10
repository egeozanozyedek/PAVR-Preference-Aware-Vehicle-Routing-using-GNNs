
import numpy as np
import vrplib


def obtain_data(data_directory):
    npzfile = np.load(f"{data_directory}/daily_stops.npz", allow_pickle=True, )

    stops = npzfile['stops_list']  # 201 length list indicating which stops are active for each day
    # print(np.info(stops), stops)
    n_vehicles = npzfile['nr_vehicles']  # n_vehicles for each day
    # print(np.info(n_vehicles), n_vehicles)
    weekday = npzfile['weekday']  # categorical input
    # print(np.info(weekday), weekday)
    capacities = npzfile['capacities_list']  # vehicle capacity
    demands = npzfile['demands_list']  # demands of each active stops

    npzfile = np.load(f"{data_directory}/daily_routematrix.npz", allow_pickle=True)

    opmat = npzfile['incidence_matrices']  # solutions for each day as an incidence matrix
    stop_wise_days = npzfile['stop_wise_active']
    distance_mat = np.load(f"{data_directory}/Distancematrix.npy")
    edge_mat = np.load(f"{data_directory}/edge_category.npy")

    return stops, n_vehicles, weekday, capacities, demands, opmat, stop_wise_days, distance_mat, edge_mat





def obtain_new_data():
    # instance = vrplib.read_instance("additional_data/Vrp-Set-XXL/Vrp-Set-XXL/XXL/Leuven1.vrp", instance_format="solomon")
    solution = vrplib.read_instance("additional_data/Vrp-Set-XXL/Vrp-Set-XXL/XXL/Leuven1.sol")

    print(solution)


if __name__ == "__main__":
    obtain_new_data()