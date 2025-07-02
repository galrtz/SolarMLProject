import torch
from torch import nn
import torch.nn.functional as F
import pickle
import numpy as np
from torch_geometric.nn import GATConv
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_forecast_specific_days(df, output_dir, node_idx, horizon_str, start_date=None, end_date=None):
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    df = df.sort_values("forecast_time")

    if start_date is not None:
        df = df[df["forecast_time"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["forecast_time"] <= pd.to_datetime(end_date)]

    if df.empty:
        print(f"⚠️ No data in date range for {horizon_str}")
        return

    plt.figure(figsize=(15, 5))
    plt.plot(df["forecast_time"], df["prediction"], label="Prediction", linestyle='--', color='tomato')
    plt.plot(df["forecast_time"], df["target"], label="Target", linestyle='-', color='teal')
    plt.xlabel("Time")
    plt.ylabel("GHI")
    plt.title(f"Node {node_idx} - {horizon_str} - {start_date} to {end_date}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"forecast_node{node_idx}_{horizon_str}_{start_date}_to_{end_date}.png"
    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path, dpi=300)
    plt.close()
    print(f"Saved date-range plot: {full_path}")

def plot_forecast(df, output_path, title):
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    plt.figure(figsize=(12, 5))
    plt.plot(df["forecast_time"], df["prediction"], label="Prediction", linestyle='--', color='tomato')
    plt.plot(df["forecast_time"], df["target"], label="Target", linestyle='-', color='teal')
    plt.xlabel("Time")
    plt.ylabel("GHI")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=False)
    plt.close()

def plot_forecast_per_day(df, output_dir, node_idx, horizon_str):
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    df = df.sort_values("forecast_time")
    df["date"] = df["forecast_time"].dt.date

    for date, group in df.groupby("date"):
        plt.figure(figsize=(12, 5))
        plt.plot(group["forecast_time"], group["prediction"], label="Prediction", linestyle='--', color='tomato')
        plt.plot(group["forecast_time"], group["target"], label="Target", linestyle='-', color='teal')
        plt.xlabel("Time")
        plt.ylabel("GHI")
        plt.title(f"Node {node_idx} - {horizon_str} - {date}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = f"forecast_node{node_idx}_{horizon_str}_{date}.png"
        full_path = os.path.join(output_dir, filename)
        plt.savefig(full_path, dpi=300)
        plt.close()
        print(f"Saved daily plot: {full_path}")



def export_forecast_per_node_per_horizon_csv(model, graphs, device, output_dir):
    print("==> Starting export...")
    print(f"Number of graphs: {len(graphs)}")

    model.eval()

    horizons = {0: "t+15", 1: "t+30", 2: "t+45", 3: "t+60"}
    offsets = {0: 15, 1: 30, 2: 45, 3: 60}

    all_results = {}  # {node_idx: {horizon: list of rows}}

    with torch.no_grad():
        for g_idx, graph in enumerate(graphs):
            print(f"\n--- Processing graph {g_idx} ---")

            try:
                print("Graph.date:", getattr(graph, 'date', 'NO DATE'))
                print("Graph.x shape:", graph.x.shape)
                print("Graph.y shape:", graph.y.shape)

                graph = graph.to(device)
                output = model(graph.x, graph.edge_index, graph.edge_attr)
                print("Model output shape:", output.shape)

                pred = output.cpu().numpy()
                true = graph.y.cpu().numpy()
                x = graph.x.cpu().numpy()
                base_time = getattr(graph, 'date', pd.Timestamp("2000-01-01 00:00:00")) - timedelta(hours=1)

                for node_idx in range(x.shape[0]):
                    if node_idx not in all_results:
                        all_results[node_idx] = {h: [] for h in horizons.values()}

                    for j in range(4):  # For each horizon
                        forecast_time = base_time + timedelta(minutes=offsets[j])
                        time_str = forecast_time.strftime("%Y-%m-%d %H:%M")
                        horizon_str = horizons[j]

                        all_results[node_idx][horizon_str].append({
                            "forecast_time": time_str,
                            "prediction": pred[node_idx, j],
                            "target": true[node_idx, j]
                        })

            except Exception as e:
                print(f"Error processing graph {g_idx}: {e}")
                continue

    # Save CSVs and plots
    csv_dir = os.path.join(output_dir, "per_node_per_horizon_csvs")
    os.makedirs(csv_dir, exist_ok=True)
    print(f"\nSaving CSVs to: {csv_dir}")

    for node_idx, horizon_data in all_results.items():
        for horizon_str, rows in horizon_data.items():
            df = pd.DataFrame(rows)
            df["forecast_time"] = pd.to_datetime(df["forecast_time"])
            df = df.sort_values("forecast_time")

            # Save CSV
            csv_path = os.path.join(csv_dir, f"forecast_node{node_idx}_{horizon_str}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV: {csv_path}")

            # Save plot only for node 0
            if node_idx == 0:
                png_path = os.path.join(output_dir, f"forecast_node0_{horizon_str}.png")
                plot_forecast(df, png_path, f"Forecast for Node 0 - {horizon_str}")
                plot_forecast_specific_days(df, output_dir, node_idx, horizon_str, start_date="2017-01-12", end_date="2017-01-15")
                plot_forecast_specific_days(df, output_dir, node_idx, horizon_str, start_date="2017-01-6", end_date="2017-01-9")
                plot_forecast_per_day(df, output_dir, node_idx, horizon_str)
                print(f"Saved plot for node 0: {png_path}")



def load_graphs(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


class LSTM_GAT(nn.Module):
    def __init__(self, in_features, lstm_hidden, n_hidden, n_heads, num_classes, num_static_features, dropout=0.4):
        super(LSTM_GAT, self).__init__()
        self.num_static = num_static_features
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.gat = GATConv(
            in_channels=lstm_hidden + num_static_features,
            out_channels=n_hidden,
            heads=n_heads,
            concat=True,
            dropout=dropout,
            edge_dim=1
        )
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


# === RUN THE EXPORT ===

# Load test graphs
pkl_path_for_test = r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\relevant\model_new\test_splitting_directory\pkl.pkl"
test_graphs = load_graphs(pkl_path_for_test)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = LSTM_GAT(
    in_features=test_graphs[0].x.shape[1],
    lstm_hidden=32,
    n_hidden=32,
    n_heads=4,
    num_classes=4,
    num_static_features=8,
    dropout=0.1
)

model.load_state_dict(torch.load(
    r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\relevant\model_new\model.pkl"
))
model.to(device)
model.eval()

# Export forecasts
output_dir = r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\relevant\model_new\actual_vs_pred_new"
export_forecast_per_node_per_horizon_csv(model, test_graphs, device, output_dir)
