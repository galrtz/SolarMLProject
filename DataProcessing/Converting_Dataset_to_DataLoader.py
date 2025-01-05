import pandas as pd
import numpy as np
from torch_geometric.data import Data
import os
import torch
from torch_geometric.loader import DataLoader
import time  # Importing time for tracking execution time
import pickle

# Save the DataLoader's data_list to a .pkl file
# Example usage
# loader = create_data_loader_from_directory('path_to_csv_directory')
# save_data_loader(loader, 'loader_data.pkl')
def save_data_loader(loader, filename):
    data_list = loader.dataset  # Extract the data list from the loader
    with open(filename, 'wb') as f:
        pickle.dump(data_list, f)

# Load the data list from a .pkl file and re-create the DataLoader
# Example usage
# loader = load_data_loader('loader_data.pkl
def load_data_loader(filename, batch_size=32):
    with open(filename, 'rb') as f:
        data_list = pickle.load(f)

    # Recreate the DataLoader with the loaded data list
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader

def euclidean_distance(coord1, coord2):
    """
    Calculate the Euclidean distance between two geographical coordinates (latitude, longitude).

    Parameters:
    - coord1, coord2: tuples representing (latitude, longitude)

    Returns:
    - Euclidean distance between the two coordinates.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    return np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

def create_single_graph_data(csv_file_path):
    """
    Create a single graph data object from a CSV file containing solar data.

    Parameters:
    - csv_file_path (str): The path to the CSV file with the raw data.

    Returns:
    - graph_data (Data): The graph data object with node features, edge indices, edge weights, and labels.
    """

    # Step 1: Load the data
    data = pd.read_csv(csv_file_path)
    sorted_df = data.sort_values(by="location_id")

    # Step 2: Extract node features (Longitude, Latitude, GHI_t-1)
    static_features = sorted_df[["longitude", "latitude"]]
    dynamic_features = [f"GHI_t-{15 * i} minutes" for i in range(1, len(sorted_df.columns) - 3)]
    dynamic_features_data = sorted_df[dynamic_features]

    node_features_combined = pd.concat([static_features, dynamic_features_data], axis=1)
    node_features_matrix = node_features_combined.to_numpy()

    # Step 3: Extract labels (GHI at time t)
    labels = sorted_df[["GHI_t"]]
    y = labels.to_numpy()

    # Step 4: Create a fully connected graph (complete graph)
    num_nodes = len(sorted_df)
    edges_source = []
    edges_target = []
    edge_weights = []

    # Create a complete graph where every node is connected to every other node
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edges_source.append(i)
            edges_target.append(j)

            # Extract latitude and longitude for both nodes
            coord1 = (static_features.iloc[i].iloc[0], static_features.iloc[i].iloc[1])  # (lat1, lon1)
            coord2 = (static_features.iloc[j].iloc[0], static_features.iloc[j].iloc[1])  # (lat2, lon2)

            # Calculate the Euclidean distance between the two coordinates
            distance = euclidean_distance(coord1, coord2)

            # Calculate the weight using the formula H * exp(-distance / characteristic_distance)
            H = 10  # Example value for forecast time (change as necessary)
            characteristic_distance = 100  # Example characteristic distance (change as necessary)
            weight = H * np.exp(-distance / characteristic_distance)

            edge_weights.append(weight)  # Store edge weights

    # Step 5: Convert edge lists and weights to PyTorch tensors
    edge_index = torch.tensor([edges_source, edges_target], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    # Step 6: Convert node features and labels to PyTorch tensors
    x = torch.tensor(node_features_matrix, dtype=torch.float)  # Node features as a tensor
    y = torch.tensor(y, dtype=torch.float)  # Labels as a tensor

    # Create the graph data object
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Return the graph data object
    print(graph_data)
    return graph_data

def create_data_loader_from_directory(csv_directory, batch_size=32):
    """
    Function to create a DataLoader for multiple graphs from a directory containing CSV files.

    Parameters:
    - csv_directory (str): The path to the directory containing the raw CSV files.
    - num_graphs (int, optional): The number of graphs to generate (default is None, meaning all files in the directory).
    - batch_size (int): The batch size for the DataLoader (default is 32).

    Returns:
    - loader (DataLoader): The DataLoader containing multiple graphs in batches.
    """
    # List to store multiple graph data objects
    data_list = []

    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

    # Iterate over all CSV files in the directory and create graphs
    for file_name in csv_files:
        file_path = os.path.join(csv_directory, file_name)

        # Create a graph for each file and add it to the list
        graph_data_i = create_single_graph_data(file_path)  # Assume this function processes each CSV file into a graph
        data_list.append(graph_data_i)

    # Initialize a DataLoader to batch multiple graphs
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, num_workers=4)

    # Return the DataLoader
    return loader

start_time = time.time()
load_data = create_data_loader_from_directory("C:/Users/galrt/Desktop/data")
print(load_data)
end_time_1 = time.time()
print(f"Program finished running data processing in {end_time_1 - start_time:.2f} seconds.")