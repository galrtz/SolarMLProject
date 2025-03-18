import time
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomNodeSplit

torch.manual_seed(42)
np.random.seed(42)


# Function to load graphs from a pickle file
def load_graphs(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Function to filter closest edges
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


# LSTM-GAT Model
class LSTM_GAT(nn.Module):
    def __init__(self, in_features, lstm_hidden, n_hidden, n_heads, num_classes, dropout=0.4):
        super(LSTM_GAT, self).__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=lstm_hidden, num_layers=2, batch_first=True,
                            dropout=dropout)
        self.gat = GATConv(in_channels=lstm_hidden, out_channels=n_hidden, heads=n_heads, concat=True, dropout=dropout,
                           edge_dim=1)
        self.fc = nn.Linear(n_hidden * n_heads, num_classes)

    def forward(self, x, edge_index, edge_attr):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add temporal dimension
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.gat(x, edge_index, edge_attr)
        x = F.elu(x)
        return self.fc(x)


# Loss function
def nrmse_loss(pred, target):
    mse = F.mse_loss(pred, target)
    rmse = torch.sqrt(mse)
    y_min, y_max = target.min(), target.max()
    return rmse / (y_max - y_min)


# Function to evaluate test predictions
def evaluate_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets, all_pv_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_attr)
            mask = batch.test_mask
            all_preds.append(output[mask].cpu().numpy())
            all_targets.append(batch.y[mask].cpu().numpy())
            all_pv_ids.append(torch.arange(len(batch.y), device=batch.y.device)[mask].cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_pv_ids)


# Training function
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(output[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Evaluation function
def evaluate(model, loader, criterion, device, mask_type="val_mask"):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_attr)
            mask = getattr(batch, mask_type, None)
            if mask is not None:
                loss = criterion(output[mask], batch.y[mask])
                total_loss += loss.item()
    return total_loss / len(loader)


# Load dataset
pkl_path = "C:/Users/hadar/Desktop/2017/graph_datalist.pkl"
graphs = load_graphs(pkl_path)

# Apply node splitting for training/validation/testing
for graph in graphs:
    graphs = [RandomNodeSplit(num_val=0.1, num_test=0.1)(graph) for graph in graphs]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_loader = DataLoader(graphs, batch_size=8, shuffle=True)  # Batch training

# Model initialization
model = LSTM_GAT(
    in_features=graphs[0].x.shape[1],
    lstm_hidden=512,
    n_hidden=512,
    n_heads=8,
    num_classes=1,
    dropout=0.4
).to(device)

optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
criterion = nrmse_loss

# Training loop
num_epochs = 201
train_losses, val_losses, test_losses = [], [], []

for epoch in range(num_epochs):
    train_loss = train(model, data_loader, optimizer, criterion, device)
    val_loss = evaluate(model, data_loader, criterion, device, mask_type="val_mask")
    test_loss = evaluate(model, data_loader, criterion, device, mask_type="test_mask")
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")

print("Training complete!")

# Get predictions for test set
preds, targets, pv_ids = evaluate_predictions(model, data_loader, device)

# Plot ground truth vs predictions for each PV_ID
plt.figure()
plt.scatter(pv_ids, targets, label='True GHI', marker='o', color='blue')
plt.scatter(pv_ids, preds, label='Predicted GHI', marker='x', color='red')
plt.xlabel("PV ID")
plt.ylabel("GHI Value")
plt.title("True vs Predicted GHI per PV ID")
plt.legend()
plt.grid()
plt.show()

# Plot loss curves
plt.figure()
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.plot(test_losses, label='Test Loss', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss (NRMSE)")
plt.title("Training, Validation, and Test Loss")
plt.legend()
plt.grid()
plt.show()
