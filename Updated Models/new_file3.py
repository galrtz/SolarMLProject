import time
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.nn import GATConv
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.transforms import RandomNodeSplit

torch.manual_seed(42)
np.random.seed(42)


# _______ Temporal-Spatial GAT Model (Using LSTM for Temporal Learning) _______
class GAT_LSTM(nn.Module):
    def __init__(self, in_features, hidden_dim, num_heads, num_classes, time_steps, dropout=0.4):
        super(GAT_LSTM, self).__init__()

        # GAT ללמידה מרחבית
        self.gat = GATConv(in_channels=in_features, out_channels=hidden_dim,
                           heads=num_heads, concat=True, dropout=dropout, edge_dim=1)

        # LSTM ללמידה טמפורלית
        self.lstm = nn.LSTM(input_size=hidden_dim * num_heads, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True)

        # Fully Connected לחיזוי GHI
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_seq, edge_index, edge_attr):
        """
        x_seq: [time_steps, num_nodes, num_features] – נתונים היסטוריים לכל צומת
        edge_index: מבנה הקשתות בגרף
        edge_attr: תכונות הקשתות
        """
        time_steps, num_nodes, num_features = x_seq.shape
        x_seq = x_seq.view(time_steps * num_nodes, num_features)

        # GAT - למידה מרחבית לכל שלב זמן
        x_seq = self.gat(x_seq, edge_index, edge_attr).relu()
        x_seq = x_seq.view(time_steps, num_nodes, -1)  # מחזירים את המימד הטמפורלי

        # LSTM - למידה טמפורלית
        x_seq, _ = self.lstm(x_seq)

        # חיזוי GHI העתידי (רק הזמן האחרון)
        x_out = self.fc(x_seq[-1])

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


def train_iter(epoch, model, optimizer, criterion, input, target, train_mask, val_mask, print_every=10):
    start_t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(*input)
    loss_train = criterion(output[train_mask], target[train_mask])

    loss_train.backward()
    optimizer.step()

    loss_val, output_val, target_val = test(model, criterion, input, target, val_mask)

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
    parser = argparse.ArgumentParser(description='Short-GAT with LSTM for Temporal Learning')

    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--hidden-dim', type=int, default=32, help='hidden representation dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='number of GAT heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph = load_graph("C:/Users/hadar/Desktop/decreasing_node.pkl")

    features = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    edge_attr = graph.edge_attr.to(device) if graph.edge_attr is not None else None
    labels = graph.y.to(device)

    # Simulating temporal features (צריך להיות מוחלף בנתונים אמיתיים)
    time_steps = 20
    features = features.unsqueeze(0).repeat(time_steps, 1, 1)  # משכפל את הנתונים 5 פעמים כאילו היו טמפורליים

    # Split Data
    node_transform = RandomNodeSplit(num_val=0.2, num_test=0.1)
    graph = node_transform(graph)

    train_mask = graph.train_mask.to(device)
    val_mask = graph.val_mask.to(device)
    test_mask = graph.test_mask.to(device)

    model = GAT_LSTM(
        in_features=features.shape[2],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_classes=1,
        time_steps=time_steps,
        dropout=args.dropout
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    criterion = nrmse_loss

    train_losses, val_losses = [], []
    for epoch in range(args.epochs):
        train_loss, val_loss = train_iter(epoch + 1, model, optimizer, criterion,
                                          (features, edge_index, edge_attr), labels, train_mask, val_mask)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    print("Training complete!")

    # Evaluate on test set
    loss_test, pred_test, target_test = test(model, criterion, (features, edge_index, edge_attr), labels, test_mask)
    print(f"Test NRMSE: {loss_test:.4f}")

    # Plot loss and predictions
    plot_loss(train_losses, val_losses)
    plot_predictions(target_test, pred_test)
