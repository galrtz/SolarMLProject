# =======================
# Imports and Seeding
# =======================
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import random

torch.manual_seed(54)
np.random.seed(54)

# ================================================
# Visualization: NRMSE Map over PV Positions
# ================================================
def draw_max_nrmse_map_val_test_only(model, graphs, device):
    model.eval()
    squared_errors_dict = {}
    count_dict = {}
    pos_type_dict = {}
    target_values = []

    for graph in graphs:
        if not hasattr(graph, 'pos') or graph.pos is None:
            continue

        graph = graph.to(device)
        with torch.no_grad():
            output = model(graph.x, graph.edge_index, graph.edge_attr)

        preds = output.cpu().numpy()
        targets = graph.y.cpu().numpy()
        pos = graph.pos.cpu().numpy()
        target_values.extend(targets)

        for i in range(len(preds)):
            p = tuple(np.round(pos[i], 5))
            squared_error = (preds[i] - targets[i]) ** 2
            if p not in squared_errors_dict:
                squared_errors_dict[p] = squared_error
                count_dict[p] = 1
            else:
                squared_errors_dict[p] += squared_error
                count_dict[p] += 1
            pos_type_dict[p] = 'val_test'

    target_values = np.array(target_values)
    y_min, y_max = target_values.min(), target_values.max()
    y_range = y_max - y_min if y_max != y_min else 1.0

    val_test_positions, val_test_nrmse = [], []
    for p, kind in pos_type_dict.items():
        if kind == 'val_test':
            mse = squared_errors_dict[p] / count_dict[p]
            rmse = np.sqrt(mse)
            nrmse = rmse / y_range
            val_test_positions.append(p)
            val_test_nrmse.append(nrmse.mean())

    val_test_positions = np.array(val_test_positions)
    val_test_nrmse = np.array(val_test_nrmse)

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(val_test_positions[:, 0], val_test_positions[:, 1], c=val_test_nrmse, cmap='RdBu_r',
                     edgecolors='black', s=80, linewidths=0.6, label='Val/Test Nodes')
    cbar = plt.colorbar(sc)
    cbar.set_label("NRMSE (Validation/Test)")
    plt.title("NRMSE per PV Position")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ======================
# Load Graph Dataset
# ======================
def load_graphs(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# ==========================
# LSTM + GAT Model
# ==========================
class LSTM_GAT(nn.Module):
    def __init__(self, in_features, lstm_hidden, n_hidden, n_heads, num_classes, num_static_features, dropout=0.4):
        super(LSTM_GAT, self).__init__()
        self.num_static = num_static_features
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.gat = GATConv(in_channels=lstm_hidden + num_static_features,
                           out_channels=n_hidden, heads=n_heads,
                           concat=True, dropout=dropout, edge_dim=1)
        self.fc = nn.Linear(n_hidden * n_heads, num_classes)

    def forward(self, x, edge_index, edge_attr):
        static = x[:, :self.num_static]
        dynamic = x[:, self.num_static:]
        dynamic_seq = dynamic.unsqueeze(-1)
        lstm_out, _ = self.lstm(dynamic_seq)
        lstm_last = lstm_out[:, -1, :]
        combined = torch.cat([lstm_last, static], dim=1)
        x = self.gat(combined, edge_index, edge_attr)
        x = F.elu(x)
        return F.relu(self.fc(x))

# ==================================
# Evaluation: Normalized NRMSE
# ==================================
def evaluate_nrmse(model, data_loader, device):
    model.eval()
    total_squared_error = torch.zeros(4, device=device)
    total_count = 0
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_attr)
            squared_error = (output - batch.y) ** 2
            total_squared_error += torch.sum(squared_error, dim=0)
            total_count += batch.num_nodes
            all_targets.append(batch.y)

    rmse = torch.sqrt(total_squared_error / total_count)

    all_targets = torch.cat(all_targets, dim=0)
    y_min = torch.min(all_targets, dim=0).values
    y_max = torch.max(all_targets, dim=0).values
    y_range = y_max - y_min + 1e-8  # Avoid divide-by-zero

    nrmse = rmse / y_range
    return nrmse.cpu().numpy()

# ========================
# Load Dataset
# ========================

pkl_path_for_train = r"C:\Users\<user>\Desktop\relevant_directories\relevant\model_new\filtered_train_data\pkl.pkl"
pkl_path_for_test = r"C:\Users\<user>\Desktop\relevant_directories\relevant\model_new\test_splitting_directory\pkl.pkl"
# test_graphs = load_graphs(pkl_path_for_test)
train_graphs_all = load_graphs(pkl_path_for_train)
test_graphs_all = load_graphs(pkl_path_for_test)

n = len(train_graphs_all)

test_end = int(0.2 * n)
# train_end = test_end + int(0.8 * n)  # כולל את ה־80% אחרי הטסט

# פיצול בפועל
test_graphs = test_graphs_all
train_graphs = train_graphs_all

# DataLoaders
train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True)
# val_loader   = DataLoader(val_graphs,   batch_size=32,  shuffle=False)
test_loader  = DataLoader(test_graphs,  batch_size=32,  shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# Data Loaders
# ========================
# train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True)
# val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

print(f"# Train: {len(train_graphs)} | Test: {len(test_graphs)}")

# ========================
# Initialize Model
# ========================
model = LSTM_GAT(
    in_features=train_graphs[0].x.shape[1],
    lstm_hidden=32,
    n_hidden=32,
    n_heads=4,
    num_classes=4,
    num_static_features=8,
    dropout=0.1
).to(device)

optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

# ===========================
# Training Loop
# ===========================
num_epochs = 400
train_losses, val_losses, test_losses = [], [], []
train_target_losses, val_target_losses, test_target_losses = [], [], []

for epoch in range(num_epochs + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = F.mse_loss(output, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)

    # Evaluate on each split
    # val_loss = evaluate_nrmse(model, val_loader, device)
    # val_loss = np.array(val_loss)
    # val_losses.append(val_loss)

    test_loss = evaluate_nrmse(model, test_loader, device)
    test_losses.append(test_loss)

    train_target_loss = evaluate_nrmse(model, train_loader, device)
    train_target_losses.append(train_target_loss)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d}:")
        print("[GHI_t+15min, GHI_t+30min, GHI_t+45min, GHI_t+60min]")
        print("  Train NRMSE:", np.round(train_target_loss, 4))
        # print("  Val   NRMSE:", np.round(val_loss, 4))
        print("  Test  NRMSE:", np.round(test_loss, 4))


print("\nFinal Test NRMSE:", np.round(test_loss, 4))

# ================================
# Plotting NRMSE over Epochs
# ================================
labels = ["t+15", "t+30", "t+45", "t+60"]
epochs = np.arange(num_epochs + 1)
train_target_losses = np.array(train_target_losses)
val_target_losses = np.array(val_losses)
test_target_losses = np.array(test_losses)

for i in range(4):
    plt.figure()
    plt.plot(epochs, train_target_losses[:, i], '--', label='Train')
    # plt.plot(epochs, val_target_losses[:, i], '-', label='Val')
    plt.plot(epochs, test_target_losses[:, i], ':', label='Test')
    plt.xlabel("Epoch")
    plt.ylabel("NRMSE")
    plt.title(f"NRMSE for Forecast {labels[i]}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

torch.save(model.state_dict(), r"C:\Users\<user>\Desktop\relevant_directories\relevant\model_new\model.pkl")
