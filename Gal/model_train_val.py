import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.transforms import RandomNodeSplit

# --- CONFIG ---
batch_size_nodes = 512
num_targets = 4  # predicting t+15, t+30, t+45, t+60
torch.manual_seed(42)
np.random.seed(42)


# --- Load Graph ---
def load_graphs(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# --- Model Definition ---
class LSTM_GAT(nn.Module):
    def __init__(self, in_features, lstm_hidden, n_hidden, n_heads, num_classes, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=lstm_hidden, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.gat = GATConv(in_channels=lstm_hidden, out_channels=n_hidden, heads=n_heads,
                           concat=True, dropout=dropout, edge_dim=1)
        self.fc = nn.Linear(n_hidden * n_heads, num_classes)

    def forward(self, x, edge_index, edge_attr):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.gat(x, edge_index, edge_attr)
        x = F.elu(x)
        return self.fc(x)

# --- Loss Functions ---
def nrmse_loss(pred, target):
    mse = F.mse_loss(pred, target)
    rmse = torch.sqrt(mse)
    y_min, y_max = target.min(), target.max()
    return rmse / (y_max - y_min)

def nrmse_per_target(pred, target):
    return [torch.sqrt(F.mse_loss(pred[:, i], target[:, i])) / (target[:, i].max() - target[:, i].min()) for i in range(target.shape[1])]

# --- Mini-batch node sampling ---
def get_random_node_subset(mask, batch_size_nodes):
    node_indices = mask.nonzero(as_tuple=True)[0]
    if len(node_indices) > batch_size_nodes:
        selected = node_indices[torch.randperm(len(node_indices))[:batch_size_nodes]]
    else:
        selected = node_indices
    return selected

# --- Main Execution ---
pkl_path = "C:/Users/galrt/Desktop/final_project/graph_27_1_2017.pkl"
graph = load_graphs(pkl_path)[0]  # single large graph
graph = RandomNodeSplit(num_val=0.2, num_test=0.1)(graph)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graph = graph.to(device)

model = LSTM_GAT(
    in_features=graph.x.shape[1],
    lstm_hidden=64,
    n_hidden=64,
    n_heads=8,
    num_classes=num_targets,
    dropout=0.2
).to(device)

optimizer = AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
criterion = nrmse_loss

num_epochs = 5000
train_losses, val_losses, test_losses = [], [], []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    output = model(graph.x, graph.edge_index, graph.edge_attr)
    selected = get_random_node_subset(graph.train_mask, batch_size_nodes)
    loss = criterion(output[selected], graph.y[selected])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(graph.x, graph.edge_index, graph.edge_attr)
        val_nodes = get_random_node_subset(graph.val_mask, batch_size_nodes)
        val_loss = criterion(output[val_nodes], graph.y[val_nodes])
        test_nodes = get_random_node_subset(graph.test_mask, batch_size_nodes)
        test_loss = criterion(output[test_nodes], graph.y[test_nodes])

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    test_losses.append(test_loss.item())
    print(f"Epoch {epoch}: Train {loss.item():.4f}, Val {val_loss.item():.4f}, Test {test_loss.item():.4f}")

# --- Results ---
model.eval()
with torch.no_grad():
    output = model(graph.x, graph.edge_index, graph.edge_attr)
    preds = output[graph.test_mask].cpu()
    targets = graph.y[graph.test_mask].cpu()

# Plot NRMSE per target
per_target_errors = nrmse_per_target(preds, targets)
for i, err in enumerate(per_target_errors):
    print(f"NRMSE t+{(i+1)*15}min: {err:.4f}")
    plt.figure()
    plt.scatter(targets[:, i], preds[:, i], alpha=0.5)
    plt.xlabel("True GHI")
    plt.ylabel("Predicted GHI")
    plt.title(f"t+{(i+1)*15}min Prediction")
    plt.grid(True)
    plt.show()

# Plot loss curves
plt.figure()
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.plot(test_losses, label='Test')
plt.title("Loss (NRMSE) Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("NRMSE")
plt.legend()
plt.grid(True)
plt.show()
