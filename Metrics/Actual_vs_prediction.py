import torch
from torch import nn
import torch.nn.functional as F
import pickle
from torch_geometric.nn import GATConv
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Plotting Utilities ===
def plot_forecast_all_days(df, output_path, title):
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    df = df.sort_values("forecast_time")
    plt.figure(figsize=(15, 5))
    plt.plot(df["forecast_time"], df["prediction"], label="Prediction", linestyle='--', color='tomato')
    plt.plot(df["forecast_time"], df["target"], label="Target", linestyle='-', color='teal')
    plt.xlabel("Time")
    plt.ylabel("GHI")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot] All days plot saved: {output_path}")

def plot_forecast_specific_days(df, output_dir, node_idx, horizon_str, start_date, end_date):
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    df = df.sort_values("forecast_time")
    df = df[(df["forecast_time"] >= pd.to_datetime(start_date)) & (df["forecast_time"] <= pd.to_datetime(end_date))]
    if df.empty:
        print(f"[Warning] No data in range {start_date} to {end_date} for node {node_idx}, {horizon_str}")
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
    filename = f"node{node_idx}_{horizon_str}_{start_date}_to_{end_date}.png"
    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path, dpi=300)
    plt.close()
    print(f"[Plot] Specific range plot saved: {full_path}")
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
        filename = f"node{node_idx}_{horizon_str}_{date}.png"
        full_path = os.path.join(output_dir, filename)
        plt.savefig(full_path, dpi=300)
        plt.close()
        print(f"[Plot] Daily plot saved: {full_path}")

# === Model Definition ===
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

# === Load Graphs ===
def load_graphs(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# === Main Export Function ===
def export_forecast_per_node_per_horizon_csv(model, graphs, device, output_dir):
    model.eval()
    horizons = {0: "t+15", 1: "t+30", 2: "t+45", 3: "t+60"}
    offsets = {0: 15, 1: 30, 2: 45, 3: 60}
    all_results = {}

    with torch.no_grad():
        for g_idx, graph in enumerate(graphs):
            graph = graph.to(device)
            graph_time = getattr(graph, 'date', pd.Timestamp("2000-01-01 00:00:00"))
            print(f"[Graph {g_idx}] Time: {graph_time}")
            output = model(graph.x, graph.edge_index, graph.edge_attr)

            pred = output.cpu().numpy()
            true = graph.y.cpu().numpy()
            x = graph.x.cpu().numpy()

            for node_idx in range(x.shape[0]):
                if node_idx not in all_results:
                    all_results[node_idx] = {h: [] for h in horizons.values()}

                for j in range(4):
                    forecast_time = graph_time + timedelta(minutes=offsets[j])
                    time_str = forecast_time.strftime("%Y-%m-%d %H:%M")
                    horizon_str = horizons[j]

                    all_results[node_idx][horizon_str].append({
                        "forecast_time": time_str,
                        "prediction": pred[node_idx, j],
                        "target": true[node_idx, j]
                    })

    csv_dir = os.path.join(output_dir, "per_node_per_horizon_csvs")
    os.makedirs(csv_dir, exist_ok=True)

    for node_idx, horizon_data in all_results.items():
        for horizon_str, rows in horizon_data.items():
            df = pd.DataFrame(rows)
            df["forecast_time"] = pd.to_datetime(df["forecast_time"])
            df = df.sort_values("forecast_time")
            df["forecast_time"] = df["forecast_time"].dt.strftime("%Y-%m-%d %H:%M")

            csv_path = os.path.join(csv_dir, f"forecast_node{node_idx}_{horizon_str}.csv")
            df.to_csv(csv_path, index=False)
            print(f"[CSV] Saved: {csv_path}")

            if node_idx == 0:
                plot_dir = os.path.join(output_dir, "plots", horizon_str)
                os.makedirs(plot_dir, exist_ok=True)

                all_days_dir = os.path.join(plot_dir, "all_days")
                per_day_dir = os.path.join(plot_dir, "per_day")
                specific_dir = os.path.join(plot_dir, "specific_ranges")
                os.makedirs(all_days_dir, exist_ok=True)
                os.makedirs(per_day_dir, exist_ok=True)
                os.makedirs(specific_dir, exist_ok=True)

                # Plot all days
                all_days_path = os.path.join(all_days_dir, f"node{node_idx}_all_days.png")
                plot_forecast_all_days(df, all_days_path, f"Node {node_idx} - {horizon_str} - All Days")

                # Plot specific ranges
                plot_forecast_specific_days(df, specific_dir, node_idx, horizon_str, "2017-01-12", "2017-01-15")
                plot_forecast_specific_days(df, specific_dir, node_idx, horizon_str, "2017-01-06", "2017-01-09")

                # Plot per day
                plot_forecast_per_day(df, per_day_dir, node_idx, horizon_str)

# === Main ===
if __name__ == "__main__":
    pkl_path = r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\relevant\model_new\test_splitting_directory\pkl.pkl"
    model_path = r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\relevant\model_new\model.pkl"
    output_dir = r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\relevant\model_new\actual_vs_pred_new_2"

    test_graphs = load_graphs(pkl_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTM_GAT(
        in_features=test_graphs[0].x.shape[1],
        lstm_hidden=32,
        n_hidden=32,
        n_heads=4,
        num_classes=4,
        num_static_features=8,
        dropout=0.1
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    print("[Start] Exporting forecasts and plots...")
    export_forecast_per_node_per_horizon_csv(model, test_graphs, device, output_dir)
    print("[Done] Export completed.")
