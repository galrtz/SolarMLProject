import time
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim import Adam
from torch_geometric.nn import GATConv
import pickle
import matplotlib.pyplot as plt
import numpy as np


torch.manual_seed(42)
np.random.seed(42)

# _______ Temporal-Spatial GAT Model (Using Only Short Connections) _______
def get_closest_edges_from_adj(edge_index, edge_attr, num_nodes, top_k_ratio=0.3):
    """
    Creates a mask (1/0) that keeps only the closest 30% of edges per node based on edge_attr (edge weights).

    :param edge_index: Tensor of shape [2, num_edges], representing graph edges.
    :param edge_attr: Tensor of shape [num_edges], representing edge weights (distances).
    :param num_nodes: Total number of nodes in the graph.
    :param top_k_ratio: Fraction of the closest edges to keep per node (default: 30%).
    :return: A boolean mask indicating which edges to keep.
    """
    num_edges = edge_index.shape[1]
    closest_mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)

    for node in range(num_nodes):
        # Get all edges where the current node is the source
        edges_for_node = (edge_index[0] == node)
        edge_ids = edges_for_node.nonzero(as_tuple=True)[0]

        if len(edge_ids) == 0:
            continue  # Skip nodes that have no outgoing edges

        # Sort edges by distance (smaller distance = closer)
        dists_for_node = edge_attr[edge_ids]
        sorted_idx = torch.argsort(dists_for_node)  # Ascending order (smallest distance first)

        # Determine how many edges to keep (30% of this node's edges)
        keep_count = max(1, int(len(sorted_idx) * top_k_ratio))  # Ensure at least 1 edge is kept

        # Select indices of the closest edges
        keep_edges = edge_ids[sorted_idx[:keep_count]]

        # Mark these edges as True in the mask (indicating they should be retained)
        closest_mask[keep_edges] = True

    return closest_mask

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)[:, -1, :]  # Extract last timestep


class ShortGAT(nn.Module):
    def __init__(self, in_features, n_hidden, n_heads, num_classes, time_steps, num_nodes, dropout=0.4):
        super(ShortGAT, self).__init__()

        self.temporal_attention = TemporalAttention(input_dim=time_steps, hidden_dim=n_hidden, num_heads=4,
                                                    num_layers=2)

        self.conv_short = GATConv(
            in_channels=in_features,
            out_channels=n_hidden,
            heads=n_heads,
            concat=True,
            dropout=dropout,
            edge_dim=1
        )

        self.fc = nn.Linear(n_hidden * n_heads, num_classes)

    def forward(self, input_tensor, edge_index, edge_attr, knn_mask):
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(1)

        x = self.temporal_attention(input_tensor)

        mask = knn_mask["short"]
        short_edges = edge_index[:, mask]
        short_edge_attr = edge_attr[mask] if edge_attr is not None else None

        x_short = self.conv_short(x, short_edges, short_edge_attr)
        x_short = F.elu(x_short)

        x_out = self.fc(x_short)
        return x_out


# Load graph data
def load_graph(filename):
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
    return graph


#################################
### TRAIN, TEST, & PLOTTING  ###
#################################

def nrmse_loss(pred, target):
    mse = F.mse_loss(pred, target)
    rmse = torch.sqrt(mse)
    y_min, y_max = target.min(), target.max()
    nrmse = rmse / (y_max - y_min)
    return nrmse


def train_iter(epoch, model, optimizer, criterion, input, target, mask_train, mask_val, print_every=10):
    start_t = time.time()
    model.train()
    optimizer.zero_grad()

    """""
    if epoch > 400:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00005
    if epoch > 800:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001
    if epoch > 250:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    if epoch > 300:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001
    """""
    output = model(*input)
    loss_train = criterion(output[mask_train], target[mask_train])

    loss_train.backward()
    optimizer.step()

    loss_val, output_val, target_val = test(model, criterion, input, target, mask_val)

    if epoch % print_every == 0:
        print(f'Epoch: {epoch:04d} ({(time.time() - start_t):.4f}s) loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}')

    return loss_train.item(), loss_val


def test(model, criterion, input, target, mask):
    model.eval()
    with torch.no_grad():
        output = model(*input)
        output, target = output[mask], target[mask]
        loss = criterion(output, target)

    return loss.item(), output, target


def plot_loss(train_losses, val_losses):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', linestyle='-', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', linestyle='-', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (NRMSE)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

def plot_predictions(ground_truth, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(ground_truth.squeeze().cpu(), label="Ground Truth", marker='o', color="blue", linestyle='None', alpha=0.7)
    plt.plot(predictions.squeeze().cpu(), label="Predictions", marker='x', color="red", linestyle='None', alpha=0.7)
    plt.xlabel("Node Index")
    plt.ylabel("GHI Value")
    plt.title("Predictions vs Ground Truth")
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Short-GAT with Temporal Attention')

    parser.add_argument('--epochs', type=int, default=1250, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='hidden representation dimension')
    parser.add_argument('--num-heads', type=int, default=128, help='number of GAT heads')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph = load_graph("C:/Users/hadar/Desktop/decreasing_node.pkl")

    # Create random indices for the data
    idx = torch.randperm(len(graph.y)).to(device)

    # Define ratios and sizes
    train_size = int(0.7 * len(graph.y))
    val_size = int(0.2 * len(graph.y))
    test_size = len(graph.y) - train_size - val_size

    # Split the indices into train, val, and test
    idx_train = idx[:train_size]
    idx_val = idx[train_size:train_size + val_size]
    idx_test = idx[train_size + val_size:]

    # Create masks for each set
    mask_train = torch.zeros(len(graph.y), dtype=torch.bool, device=device)
    mask_train[idx_train] = True

    mask_val = torch.zeros(len(graph.y), dtype=torch.bool, device=device)
    mask_val[idx_val] = True

    mask_test = torch.zeros(len(graph.y), dtype=torch.bool, device=device)
    mask_test[idx_test] = True

    # Prepare the data
    features = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    edge_attr = graph.edge_attr.to(device) if graph.edge_attr is not None else None
    labels = graph.y.to(device)

    # Define only short connections
    knn_mask = {
        "short": get_closest_edges_from_adj(edge_index, edge_attr, features.shape[0])
    }

    model = ShortGAT(
        in_features=features.shape[1],
        n_hidden=args.hidden_dim,
        n_heads=args.num_heads,
        num_classes=1,
        time_steps=features.shape[1],
        num_nodes=features.shape[0],
        dropout=args.dropout
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    criterion = nrmse_loss

    train_losses, val_losses = [], []
    for epoch in range(args.epochs):
        train_loss, val_loss = train_iter(epoch + 1, model, optimizer, criterion,
                                          (features, edge_index, edge_attr, knn_mask), labels, mask_train, mask_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    print("Training complete!")

    # Evaluate on test set
    loss_test, pred_test, target_test = test(model, criterion,
                                              (features, edge_index, edge_attr, knn_mask), labels, mask_test)
    print(f"Test NRMSE: {loss_test:.4f}")

    # Plot loss and predictions
    plot_loss(train_losses, val_losses)
    plot_predictions(target_test, pred_test)
