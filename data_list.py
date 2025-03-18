import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
from pyproj import Proj, transform
import os

def save_graph_to_pkl(graph_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(graph_list, f)


def convert_latlon_to_xy(latitudes, longitudes):
    proj_wgs84 = Proj(init="epsg:4326")  # WGS84 geographic coordinates
    proj_israel = Proj(init="epsg:2039")  # Israeli coordinate system (X, Y in meters)
    x_coords, y_coords = transform(proj_wgs84, proj_israel, longitudes, latitudes)
    return np.vstack((x_coords, y_coords)).T

def euclidean_distance_matrix(coords):
    x, y = coords[:, 0], coords[:, 1]
    x = x[:, None]
    y = y[:, None]
    return np.sqrt((x - x.T) ** 2 + (y - y.T) ** 2)

def create_graph_by_distance(csv_file_path, max_distance_km=16, characteristic_distance_km=15):
    print(f"Processing file: {csv_file_path}")
    data = pd.read_csv(csv_file_path).dropna(how='all')

    sorted_df = data.sort_values(by="PV_ID").drop_duplicates(subset=["PV_ID"])

    #dynamic features
    ghi_columns = [col for col in sorted_df.columns if "GHI_t-" in col]
    dynamic_features_data = sorted_df[ghi_columns]

    if (sorted_df["Time_Category"] == "Noon").any():
        static_features_data = np.ones((dynamic_features_data.shape[0], 1))
    else:
        static_features_data = np.zeros((dynamic_features_data.shape[0], 1))

    node_features_matrix = np.concatenate((dynamic_features_data.to_numpy(), static_features_data), axis=1)

    #edges and edge's weights
    latitudes, longitudes = sorted_df["latitude"].values, sorted_df["longitude"].values
    coords_xy = convert_latlon_to_xy(latitudes, longitudes)
    distance_matrix = euclidean_distance_matrix(coords_xy)
    max_distance_meters = max_distance_km * 1000
    characteristic_distance_meters = characteristic_distance_km * 1000
    edge_sources, edge_targets = np.where((distance_matrix > 0) & (distance_matrix <= max_distance_meters))
    edge_weights = 1 / ((distance_matrix[edge_sources, edge_targets] / characteristic_distance_meters) ** 2)
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    # lables (predictions)
    labels = sorted_df[["ghi_t"]].to_numpy()

    x = torch.tensor(node_features_matrix, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    print(f"Graph data object: {graph_data}")
    return graph_data

# Process multiple CSV files
data_dir = "C:/Users/hadar/Desktop/split_csv_files"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
datalist = []

for csv_file in csv_files:
    csv_path = os.path.join(data_dir, csv_file)
    graph = create_graph_by_distance(csv_path, max_distance_km=16, characteristic_distance_km=15)
    datalist.append(graph)

# Save all graphs into a single pickle file
output_pkl_path = "C:/Users/hadar/Desktop/2017/graph_datalist.pkl"
save_graph_to_pkl(datalist, output_pkl_path)

print(f"Saved {len(datalist)} graphs to {output_pkl_path}")
