
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
import os as os


torch.manual_seed(1)
np.random.seed(1)


def accuracy_per_forecast_on_test(model, loader, device, tolerance):
    """
    Computes accuracy with relative tolerance for each forecast horizon (t+15,...)
    using only test_mask data.

    Args:
        model: trained model
        loader: DataLoader of graphs
        device: torch device (cpu or cuda)
        relative_tolerance (float): fraction of tolerance (e.g., 0.3 for ±30%)

    Returns:
        np.ndarray: accuracy per forecast step (e.g., [0.89, 0.86, 0.84, 0.82])
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_attr)
            mask = batch.test_mask
            all_preds.append(output[mask].cpu().numpy())
            all_targets.append(batch.y[mask].cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    correct = np.abs(preds - targets) <= tolerance
    accuracies = np.mean(correct, axis=0)

    return accuracies

def draw_nrmse_map_from_loader(model, loader, device):
    model.eval()
    squared_errors_dict = {}
    count_dict = {}
    pos_dict = {}
    target_values = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_attr)

            preds = output.cpu().numpy()
            targets = batch.y.cpu().numpy()
            pos = batch.pos.cpu().numpy()

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
                pos_dict[p] = pos[i]

    target_values = np.array(target_values)
    y_min, y_max = target_values.min(), target_values.max()
    y_range = y_max - y_min if y_max != y_min else 1.0

    positions = []
    nrmse_values = []

    for p in squared_errors_dict:
        mse = squared_errors_dict[p] / count_dict[p]
        rmse = np.sqrt(mse)
        nrmse = rmse / y_range
        positions.append(pos_dict[p])
        nrmse_values.append(nrmse.mean())

    positions = np.array(positions)
    nrmse_values = np.array(nrmse_values)

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        positions[:, 0], positions[:, 1],
        c=nrmse_values,
        cmap='RdBu_r',
        edgecolors='black',
        s=80,
        linewidths=0.6,
        label='Test Nodes (Unseen Graphs)'
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("NRMSE (Test Graphs)")
    plt.title("NRMSE per PV Position (Unseen Test Graphs)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

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
        val_mask = graph.val_mask.cpu().numpy()
        test_mask = graph.test_mask.cpu().numpy()
        train_mask = graph.train_mask.cpu().numpy()
        pos = graph.pos.cpu().numpy()

        target_values.extend(targets)

        for i in range(len(preds)):
            p = tuple(np.round(pos[i], 5))
            squared_error = (preds[i] - targets[i]) ** 2
            if val_mask[i] or test_mask[i]:
                if p not in squared_errors_dict:
                    squared_errors_dict[p] = squared_error
                    count_dict[p] = 1
                else:
                    squared_errors_dict[p] += squared_error
                    count_dict[p] += 1
                pos_type_dict[p] = 'val_test'
            elif train_mask[i]:
                pos_type_dict[p] = 'train'

    target_values = np.array(target_values)
    y_min, y_max = target_values.min(), target_values.max()
    y_range = y_max - y_min if y_max != y_min else 1.0

    val_test_positions, val_test_nrmse = [], []
    train_positions = []

    for p, kind in pos_type_dict.items():
        if kind == 'val_test':
            mse = squared_errors_dict[p] / count_dict[p]
            rmse = np.sqrt(mse)
            nrmse = rmse / y_range
            val_test_positions.append(p)
            val_test_nrmse.append(nrmse.mean())
        elif kind == 'train':
            train_positions.append(p)

    val_test_positions = np.array(val_test_positions)
    val_test_nrmse = np.array(val_test_nrmse)

    plt.figure(figsize=(10, 8))
    if len(train_positions) > 0:
        train_positions = np.array(train_positions)
        plt.scatter(train_positions[:, 0], train_positions[:, 1], color=(0.6, 0.6, 0.6, 0.2), edgecolors='black', s=70, linewidths=0.5, label='Train Nodes')

    sc = plt.scatter(val_test_positions[:, 0], val_test_positions[:, 1], c=val_test_nrmse, cmap='RdBu_r', edgecolors='black', s=80, linewidths=0.6, label='Val/Test Nodes')

    cbar = plt.colorbar(sc)
    cbar.set_label("NRMSE (Validation/Test)")
    plt.title("NRMSE per PV Position")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def load_graphs(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

class LSTM_GAT(nn.Module):
    def __init__(self, in_features, lstm_hidden, n_hidden, n_heads, num_classes, num_static_features, dropout=0.4):
        super(LSTM_GAT, self).__init__()
        self.num_static = num_static_features
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.gat = GATConv(in_channels=lstm_hidden + num_static_features, out_channels=n_hidden, heads=n_heads, concat=True, dropout=dropout, edge_dim=1)
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

def evaluate(model, loader, device, mask_type="val_mask"):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_attr)
            mask = getattr(batch, mask_type, None)
            if mask is not None:
                preds = output[mask]
                targets = batch.y[mask]
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mse = ((preds - targets) ** 2).mean(dim=0)
    rmse = torch.sqrt(mse)
    y_min, y_max = targets.min(dim=0).values, targets.max(dim=0).values
    nrmse = rmse / (y_max - y_min + 1e-8)

    return nrmse.mean().item(), nrmse.numpy()
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

#----------------new ---------------------------
# pkl_path_for_train = r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\relevant\testing_2\2018_train_pkl_filterd.pkl"
pkl_path_for_train=r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\relevant\model_new\filtered_train_data\pkl.pkl"

# test_graphs = load_graphs(pkl_path_for_test)
train_graphs_all = load_graphs(pkl_path_for_train)

# חישוב אורך כולל
n_total = len(train_graphs_all)

# test_graphs = load_graphs(pkl_path_for_test)
train_graphs_all = load_graphs(pkl_path_for_train)

n = len(train_graphs_all)

test_end = int(0.2 * n)
# train_end = test_end + int(0.8 * n)  # כולל את ה־80% אחרי הטסט

# פיצול בפועל
test_graphs  = train_graphs_all[:test_end]
train_graphs = train_graphs_all[test_end:]

#-------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_val_graphs = [
    RandomNodeSplit(num_val=0.1, num_test=0.1)(g)
    for g in train_graphs
]

print(f"# Train: {len(train_val_graphs)} Test: {len(test_graphs)}")

test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)
data_loader = DataLoader(train_val_graphs, batch_size=128, shuffle=True)

model = LSTM_GAT(
    in_features=train_val_graphs[0].x.shape[1],
    lstm_hidden=32,
    n_hidden=32,
    n_heads=4,
    num_classes=4,
    num_static_features=8,
    dropout=0.1
).to(device)

optimizer = AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)
train_losses, val_losses, test_losses, test_loss_only_graph_losses = [], [], [], []
train_target_losses, val_target_losses, test_target_losses = [], [], []
num_epochs = 1000

for epoch in range(num_epochs + 1):
    model.train()
    total_loss = 0
    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = F.mse_loss(output[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(data_loader)

    val_loss, val_target_loss = evaluate(model, data_loader, device, mask_type="val_mask")
    test_loss, test_target_loss = evaluate(model, data_loader, device, mask_type="test_mask")
    _, train_target_loss = evaluate(model, data_loader, device, mask_type="train_mask")
    test_loss_only_graph = evaluate_nrmse(model, test_loader, device) # this is the tes dataloader

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)
    train_target_losses.append(train_target_loss)
    val_target_losses.append(val_target_loss)
    test_target_losses.append(test_target_loss)
    test_loss_only_graph_losses.append(test_loss_only_graph)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d}:")
        print("[GHI_t+15min, GHI_t+30min, GHI_t+45min, GHI_t+60min]")
        print("  Train NRMSE:", np.round(train_target_loss, 4))
        print("  Val   NRMSE:", np.round(val_target_loss, 4))
        print("  Graph test   NRMSE:", np.round(test_loss_only_graph, 4))

print("\nFinal Test NRMSE:", np.round(test_target_loss, 4))
print("\nFinal Test full grpah NRMSE:", np.round(test_loss_only_graph, 4))


labels = ["t+15", "t+30", "t+45", "t+60"]
epochs = np.arange(num_epochs + 1)
train_target_losses = np.array(train_target_losses)
val_target_losses = np.array(val_target_losses)
test_target_losses = np.array(test_target_losses)
test_loss_only_graph_losses = np.array(test_loss_only_graph_losses)

val_color = '#4A708B'      # כחול-אפור כהה
train_color = '#A0525A'   # ורוד-חום כהה

for i in range(4):
    plt.figure()
    plt.plot(epochs, train_target_losses[:, i], '--', label='Train', color=train_color)
    plt.plot(epochs, val_target_losses[:, i], '-', label='Val', color=val_color)
    plt.plot(epochs, test_loss_only_graph_losses[:, i], '-', label='test')
    plt.hlines(test_target_losses[-1][i], 0, num_epochs, colors='red', linestyles=':', label='Final Test')
    plt.xlabel("Epoch")
    plt.ylabel("NRMSE")
    plt.title(f"NRMSE for Forecast {labels[i]}")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.gca().patch.set_alpha(0)
    plt.gcf().patch.set_alpha(0)

    # filename = os.path.join(r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\data_and_model_of_2017_to_2022", f"nrmse_forecast_{labels[i]}.png")
    # plt.savefig(filename, dpi=300, transparent=True)
    # print(f"Saved: {filename}")

    plt.show()


# draw_max_nrmse_map_val_test_only(model, graphs, device)
# draw_nrmse_map_from_loader(model, test_loader, device)


torch.save(model.state_dict(), r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\new_model_changing dataset\model_weights_testing_something.pkl")
