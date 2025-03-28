import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
from pyproj import Proj, transform

def save_graph_to_pkl(graph, filename):
    with open(filename, 'wb') as f:
        pickle.dump([graph], f)  # save as a list for consistency

def convert_latlon_to_xy(latitudes, longitudes):
    proj_wgs84 = Proj(init="epsg:4326")
    proj_israel = Proj(init="epsg:2039")
    x_coords, y_coords = transform(proj_wgs84, proj_israel, longitudes, latitudes)
    return np.vstack((x_coords, y_coords)).T

def euclidean_distance_matrix(coords):
    x, y = coords[:, 0], coords[:, 1]
    x = x[:, None]
    y = y[:, None]
    return np.sqrt((x - x.T) ** 2 + (y - y.T) ** 2)

def create_graph_from_single_csv(csv_file_path, max_distance_km=16, characteristic_distance_km=15):
    print(f"Processing file: {csv_file_path}")
    df = pd.read_csv(csv_file_path).dropna(how='any')

    # Dynamic features
    ghi_columns = [col for col in df.columns if "GHI_t-" in col]
    dynamic_features = df[ghi_columns].to_numpy()

    # Optional static feature: all ones (can be replaced with hour/day/lat/lon if needed)
    static_feature = np.ones((dynamic_features.shape[0], 1))

    # Node features
    node_features = np.concatenate((dynamic_features, static_feature), axis=1)

    # Labels
    label_columns = [col for col in df.columns if "t+15" in col or "t+30" in col or "t+45" in col or "t+60" in col]
    y = torch.tensor(df[label_columns].to_numpy(), dtype=torch.float)

    # Location
    coords_xy = convert_latlon_to_xy(df["latitude"].values, df["longitude"].values)
    dist_matrix = euclidean_distance_matrix(coords_xy)

    # Edges
    max_d = max_distance_km * 1000
    char_d = characteristic_distance_km * 1000
    src, tgt = np.where((dist_matrix > 0) & (dist_matrix <= max_d))
    #edge_weights = 1 / ((dist_matrix[src, tgt] / char_d) ** 2)

    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    #edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    edge_attr = torch.ones(len(src), dtype=torch.float)  # משקל 1 לכל קשת
    x = torch.tensor(node_features, dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    print(f"✅ Created graph with {x.shape[0]} nodes, {edge_index.shape[1]} edges")
    return graph

# ==== CONFIGURATION ====
input_csv = "C:/Users/galrt/Desktop/final_project/ghi_27_1_2017.csv"  # your filtered file
output_pkl = "C:/Users/galrt/Desktop/final_project/graph_27_1_2017.pkl"

# ==== RUN ====
graph = create_graph_from_single_csv(input_csv, max_distance_km=8, characteristic_distance_km=15)
save_graph_to_pkl(graph, output_pkl)
print(f"✅ Saved graph to {output_pkl}")
