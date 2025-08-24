
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
from pyproj import Proj, transform
import os
from tqdm import tqdm


def forecast_for_exact_pv(model, graphs, device, target_lon, target_lat, target_day, target_month, save_path=None):
    """
    Extracts prediction and ground truth for a specific PV node
    identified by exact longitude and latitude, for a given day and month.

    Args:
        model: trained model
        graphs: list of Data objects
        device: torch device (cpu/cuda)
        target_lon, target_lat: float, exact coordinates of the PV
        target_day, target_month: int, date filter
        save_path: optional path to save results as CSV

    Returns:
        pd.DataFrame of results (1 row per matching graph)
    """
    model.eval()
    results = []
    target = np.round([target_lon, target_lat], 5)

    with torch.no_grad():
        for graph in graphs:
            if getattr(graph, "day", None) != target_day or getattr(graph, "month", None) != target_month:
                continue

            graph = graph.to(device)
            x_feats = graph.x.cpu().numpy()
            coords = np.round(x_feats[:, :2], 5)  # assuming lon, lat in first two features

            # Find the node with exact matching coordinates
            matches = np.where((coords == target).all(axis=1))[0]
            if len(matches) == 0:
                continue  # no matching node in this graph

            i = matches[0]
            pred = model(graph.x, graph.edge_index, graph.edge_attr)[i].detach().cpu().numpy()
            true = graph.y[i].cpu().numpy()

            results.append({
                "pv_index": i,
                "longitude": coords[i][0],
                "latitude": coords[i][1],
                "t+15_pred": pred[0],
                "t+30_pred": pred[1],
                "t+45_pred": pred[2],
                "t+60_pred": pred[3],
                "t+15_true": true[0],
                "t+30_true": true[1],
                "t+45_true": true[2],
                "t+60_true": true[3],
                "day": graph.day,
                "month": graph.month
            })

    df = pd.DataFrame(results)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved results to {save_path}")

    return df


def save_graph_to_pkl(graph_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(graph_list, f)

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

def get_closest_edges_from_adj(edge_index, edge_attr, num_nodes, top_k_ratio=0.3):
    num_edges = edge_index.shape[1]
    closest_mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)
    for node in range(num_nodes):
        edges_for_node = (edge_index[0] == node)
        edge_ids = edges_for_node.nonzero(as_tuple=True)[0]
        if len(edge_ids) == 0:
            continue
        dists_for_node = edge_attr[edge_ids]
        sorted_idx = torch.argsort(dists_for_node)
        keep_count = max(1, int(len(sorted_idx) * top_k_ratio))
        keep_edges = edge_ids[sorted_idx[:keep_count]]
        closest_mask[keep_edges] = True
    return closest_mask

def create_graph_by_distance(csv_file_path, max_distance_km=16, characteristic_distance_km=15, top_k_ratio=0.3):
    # print(f"Processing file: {csv_file_path}")
    data = pd.read_csv(csv_file_path).dropna(how='all')

    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce", dayfirst=True)
    data = data.dropna(subset=["datetime"])

    data["monthogyear"] = data["datetime"].dt.month
    data["dayofyear"] = data["datetime"].dt.dayofyear
    data["hour"] = data["datetime"].dt.hour
    data["minute"] = data["datetime"].dt.minute
    data["sin_dayofyear"] = np.sin(2 * np.pi * data["dayofyear"] / 365)
    data["cos_dayofyear"] = np.cos(2 * np.pi * data["dayofyear"] / 365)
    data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24)
    # data["GHI_d_1"] = data["GHI_t-0min"] - data["GHI_t-15min"]
    # data["GHI_d_2"] = data["GHI_t-15min"] - data["GHI_t-30min"]
    # data["GHI_d_3"] = data["GHI_t-30min"] - data["GHI_t-45min"]
    # data["GHI_d_4"] = data["GHI_t-45min"] - data["GHI_t-60min"]
    # data["GHI_d_5"] = data["GHI_t-60min"] - data["GHI_t-75min"]
    # data["GHI_d_6"] = data["GHI_t-75min"] - data["GHI_t-90min"]
    # data["GHI_d_7"] = data["GHI_t-90min"] - data["GHI_t-105min"]
    # data["GHI_delta_1h"] = data["GHI_t-0h"] - data["GHI_t-1h"]
    # data["GHI_delta_2h"] = data["GHI_t-1h"] - data["GHI_t-2h"]
    # data["centered_hour"] = data["hour"] - 12
    # data["hour_squared"] = data["centered_hour"] ** 2
    # === Trend Features: deltas
    # data["GHI_delta_1h"] = data["GHI_t-0h"] - data["GHI_t-1h"]
    # # data["GHI_delta_2h"] = data["GHI_t-1h"] - data["GHI_t-2h"]
    # data["GHI_mean"] = data[["GHI_t-0h", "GHI_t-1h"]].mean(axis=1)

    sorted_df = data.sort_values(by="PV_ID").drop_duplicates(subset=["PV_ID"])




    # dynamic_cols = ["GHI_d_1", "GHI_d_2", "GHI_d_3", "GHI_d_4"] #  "GHI_d_5",  "GHI_d_6",  "GHI_d_7"]
    # dynamic_features = data[dynamic_cols]

    selected_hours = [24, 48, 72]
    dynamic_cols = [f"GHI_t-{h}h" for h in selected_hours if f"GHI_t-{h}h" in sorted_df.columns]
    dynamic_features = sorted_df[dynamic_cols]

    dynamic_tensor = torch.tensor(dynamic_features.to_numpy(), dtype=torch.float32)

    # data["GHI_t-24h"] = data["GHI_t-24h"]
    # data["GHI_t-48h"] = data["GHI_t-48h"]

    static_features_col = ["longitude", "latitude", "monthogyear", "hour", "sin_dayofyear", "cos_dayofyear", "GHI_t-0h", "GHI_t-1h"]
    static_features = sorted_df[static_features_col]
    static_tensor = torch.tensor(static_features.to_numpy(), dtype=torch.float32)

    x = torch.cat([static_tensor, dynamic_tensor], dim=1)

    # === Labels: future GHI vector
    target_cols = ["GHI_t+15min", "GHI_t+30min", "GHI_t+45min", "ghi_t+60min"]
    y = torch.tensor(sorted_df[target_cols].values, dtype=torch.float32)
    # === Coordinates
    latitudes = sorted_df["latitude"].values
    longitudes = sorted_df["longitude"].values
    coords_xy = convert_latlon_to_xy(latitudes, longitudes)

    # === Build edges by distance
    distance_matrix = euclidean_distance_matrix(coords_xy)
    max_distance_meters = max_distance_km * 1000
    characteristic_distance_meters = characteristic_distance_km * 1000

    edge_sources, edge_targets = np.where((distance_matrix > 0) & (distance_matrix <= max_distance_meters))
    edge_weights = 1 / ((distance_matrix[edge_sources, edge_targets] / characteristic_distance_meters) ** 2)

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    # === Keep only closest edges
    closest_mask = get_closest_edges_from_adj(edge_index, edge_attr, num_nodes=x.shape[0], top_k_ratio=top_k_ratio)
    edge_index = edge_index[:, closest_mask]
    edge_attr = edge_attr[closest_mask]

    # === Final graph
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    graph_data.pos = torch.tensor(coords_xy, dtype=torch.float32)

    first_dt = data["datetime"].iloc[0]
    graph_data.date = first_dt
    graph_data.day = first_dt.day
    graph_data.month = first_dt.month
    graph_data.year = first_dt.year
    graph_data.hour = first_dt.hour
    graph_data.minute = first_dt.minute

    print(f"\n{graph_data}")

    return graph_data



# Build dataset
data_dir = r"C:\Users\<user>\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\relevant\model_new\test_splitting_directory"


csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
datalist = []

for csv_file in tqdm(csv_files):
    csv_path = os.path.join(data_dir, csv_file)
    filename = os.path.basename(csv_file)
    name_no_ext = filename.replace(".csv", "")
    parts = name_no_ext.split('_')


    graph = create_graph_by_distance(csv_path, max_distance_km=16, characteristic_distance_km=15)
    # graph.year = year
    # graph.month = month
    datalist.append(graph)

print(len(datalist))


# Save dataset
output_pkl_path  = r"C:\Users\<user>\Desktop\relevant_directories\relevant\model_new\test_splitting_directory\pkl.pkl"



save_graph_to_pkl(datalist, output_pkl_path)
print(f"Saved {len(datalist)} graphs to {output_pkl_path}")
