import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import pickle
import time8888888888888888888888888888
33333333333

# Save the DataLoader's data_list to a .pkl file
def save_data_loader(data_list, filename):
    # data_list = loader.dataset  # Extract the data list from the loader
    with open(filename, 'wb') as f:
        pickle.dump(data_list, f)


# Load the data list from a .pkl file and re-create the DataLoader
def load_data_loader(filename, batch_size=32):
    with open(filename, 'rb') as f:
        data_list = pickle.load(f)

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


# Calculate Euclidean distance using numpy
def euclidean_distance_matrix(coords):
    """
    Calculate the Euclidean distance matrix between all pairs of coordinates.

    Parameters:
    - coords: A 2D array of shape (num_nodes, 2) where each row is a (lat, lon) pair.

    Returns:
    - A 2D array of shape (num_nodes, num_nodes) where each element (i, j) is the distance between nodes i and j.
    """
    latitudes, longitudes = coords[:, 0], coords[:, 1]
    latitudes = latitudes[:, None]  # Reshape to (num_nodes, 1)
    longitudes = longitudes[:, None]  # Reshape to (num_nodes, 1)

    # Calculate the pairwise Euclidean distances
    distances = np.sqrt((latitudes - latitudes.T) ** 2 + (longitudes - longitudes.T) ** 2)
    return distances


# Create a single graph data object from a CSV file containing solar data
def create_single_graph_data(csv_file_path):
    print(f"Processing file: {csv_file_path}")

    # Step 1: Load the data
    data = pd.read_csv(csv_file_path)
    print(f"Loaded data: {data.head()}")  # Display the first few rows of the data

    sorted_df = data.sort_values(by="PV_ID")

    # Step 2: Extract node features (Longitude, Latitude, GHI_t-1)
    static_features = sorted_df[["longitude", "latitude"]]
    print(f"Static features (Longitude, Latitude): {static_features.head()}")  # Show first few rows of static features

    dynamic_features = [f"GHI_t-{15 * i} minutes" for i in range(1, len(sorted_df.columns) - 3)]
    dynamic_features_data = sorted_df[dynamic_features]
    print(f"Dynamic features (GHI at different times): {dynamic_features_data.head()}")  # Show dynamic features

    node_features_combined = pd.concat([static_features, dynamic_features_data], axis=1)
    node_features_matrix = node_features_combined.to_numpy()
    print(f"Combined node features: {node_features_matrix[:5]}")  # Show first 5 combined features

    # Step 3: Extract labels (GHI at time t)
    labels = sorted_df[["ghi_t"]]
    y = labels.to_numpy()
    print(f"Labels (GHI at time t): {y[:5]}")  # Show first 5 labels

    # Step 4: Create a complete graph using the Euclidean distance matrix
    coords = static_features.to_numpy()
    print(f"Coordinates (Lat, Lon) for distance matrix: {coords[:5]}")  # Show first 5 coordinates

    distance_matrix = euclidean_distance_matrix(coords)
    print(f"Distance matrix:\n{distance_matrix[:5, :5]}")  # Show top-left 5x5 part of the distance matrix

    # Step 5: Calculate the weights using the formula H * exp(-distance / characteristic_distance)
    H = 10  # Example value for forecast time (change as necessary)
    characteristic_distance = 100  # Example characteristic distance (change as necessary)
    edge_weights = H * np.exp(-distance_matrix / characteristic_distance)
    print(f"Edge weights (after applying exponential decay):\n{edge_weights[:5, :5]}")  # Show edge weights

    # Step 6: Create the edge_index (list of edges) from the distance matrix
    edges_source, edges_target = np.tril_indices(len(coords), -1)  # Get lower triangle indices (unique edges)
    edge_weights = edge_weights[edges_source, edges_target]
    print(f"Edges (source, target): {list(zip(edges_source[:5], edges_target[:5]))}")  # Show first 5 edges
    print(f"Corresponding edge weights: {edge_weights[:5]}")  # Show first 5 edge weights

    # Step 7: Convert edge lists and weights to PyTorch tensors
    edge_index = torch.tensor([edges_source, edges_target], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    print(f"Edge index tensor: {edge_index[:5]}")  # Show the first few elements of edge_index tensor
    print(f"Edge attributes tensor: {edge_attr[:5]}")  # Show the first few elements of edge_attr tensor

    # Step 8: Convert node features and labels to PyTorch tensors
    x = torch.tensor(node_features_matrix, dtype=torch.float)  # Node features as a tensor
    y = torch.tensor(y, dtype=torch.float)  # Labels as a tensor
    print(f"Node features tensor: {x[:5]}")  # Show the first few node features tensor
    print(f"Labels tensor: {y[:5]}")  # Show the first few labels tensor

    # Create the graph data object
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    print(f"Graph data object: {graph_data}")

    return graph_data


# Create a DataLoader for multiple graphs from a directory containing CSV files
def create_data_loader_from_directory(csv_directory, batch_size=32):
    data_list = []

    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

    # Iterate over all CSV files in the directory and create graphs
    for file_name in csv_files:
        file_path = os.path.join(csv_directory, file_name)

        # Create a graph for each file and add it to the list
        graph_data_i = create_single_graph_data(file_path)
        data_list.append(graph_data_i)

    # loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_list


# Example usage
start_time = time.time()
data_list = create_data_loader_from_directory("C:/Users/hadar/Desktop/output2")
#print(f"DataLoader created with {len(data_list.dataset)} graphs.")

# Sככככave DataLoader to a pickle file
save_data_loader(data_list, "C:/Users/hadar/Desktop/dataloader_pkl/loader_data.pkl")

end_time_1 = time.time()
print(f"Program finished running data processing in {end_time_1 - start_time:.2f} seconds.")


