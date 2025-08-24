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

def plot_forecast(df, output_path, title):
    """
    Plot predictions vs. targets over time for a single horizon.
    Saves the figure to output_path.
    """
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

def plot_forecast_subplots(df_all, output_path, title_prefix="Forecast for PV"):
    """
    Plot predictions vs. targets for all horizons on separate subplots, for a single day.
    """
    df_all["forecast_time"] = pd.to_datetime(df_all["forecast_time"])
    horizons = sorted(df_all["horizon"].unique())
    n = len(horizons)

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 3.5 * n), sharex=True)

    for idx, horizon in enumerate(horizons):
        ax = axes[idx] if n > 1 else axes
        sub_df = df_all[df_all["horizon"] == horizon]

        ax.plot(sub_df["forecast_time"], sub_df["prediction"], linestyle='--', color='tomato', label="Prediction")
        ax.plot(sub_df["forecast_time"], sub_df["target"], linestyle='-', color='teal', label="Target")
        ax.set_title(f"{title_prefix} - {horizon}")
        ax.set_ylabel("GHI")
        ax.grid(True)
        ax.legend()

    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def export_forecast_per_horizon_for_single_pv(model, graphs, device, target_lon, target_lat, output_dir):
    """
    For a given (lon, lat) PV site, export per-horizon CSVs and plots of prediction vs. target
    across all graphs. Also produces per-day combined subplot figures.
    """
    model.eval()
    target_coord = np.round([target_lon, target_lat], 2)
    print(f"🔍 Searching PV at coordinates: {target_coord}")

    results = {0: [], 1: [], 2: [], 3: []}
    horizons = {0: "t+15", 1: "t+30", 2: "t+45", 3: "t+60"}
    offsets = {0: 15, 1: 30, 2: 45, 3: 60}

    graph_counter = 0
    matched_total = 0

    with torch.no_grad():
        for graph in graphs:
            graph_counter += 1
            graph = graph.to(device)
            output = model(graph.x, graph.edge_index, graph.edge_attr)
            pred = output.cpu().numpy()
            true = graph.y.cpu().numpy()
            x = graph.x.cpu().numpy()

            # Match the PV by exact lon/lat columns used in your data (here assumed x[:,2], x[:,3])
            tol = 1e-3
            lon_match = np.abs(x[:, 2] - target_lon) < tol
            lat_match = np.abs(x[:, 3] - target_lat) < tol
            matches = np.where(lon_match & lat_match)[0]

            if len(matches) == 0:
                print(f"No coordinate match in graph #{graph_counter}")
                continue

            i = matches[0]
            matched_total += 1

            try:
                # Base time: your original logic uses graph.date - 1 hour
                base_time = graph.date - timedelta(hours=1)
                print(f"🕒 Base time for graph #{graph_counter}: {base_time.strftime('%Y-%m-%d %H:%M')}")

                for j in range(4):  # 0..3 for t+15..t+60
                    forecast_time = base_time + timedelta(minutes=offsets[j])
                    time_str = forecast_time.strftime("%Y-%m-%d %H:%M")

                    results[j].append({
                        "forecast_time": time_str,
                        "prediction": pred[i, j],
                        "target": true[i, j]
                    })
                    print(f"    ➕ {horizons[j]} | {time_str} | pred={pred[i, j]:.3f} | true={true[i, j]:.3f}")

            except Exception as e:
                print(f"Error in graph #{graph_counter}: {e}")

    if matched_total == 0:
        print("❌ No matches were found for the requested PV coordinates.")

    all_results = []
    # Save per-horizon CSVs and plots
    for j in range(4):
        if not results[j]:
            print(f"No data found for horizon {horizons[j]}")
            continue

        df = pd.DataFrame(results[j]).sort_values("forecast_time")
        label = horizons[j]

        # Collect for per-day combined subplots
        for row in results[j]:
            row["horizon"] = horizons[j]
            all_results.append(row)

        # Save CSV
        csv_filename = os.path.join(output_dir, f"pv_forecast_{label}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"💾 Saved CSV: {csv_filename}")
        
        # Save plot for the full time range
        png_filename = os.path.join(output_dir, f"pv_forecast_{label}_all_range.png")
        plot_forecast(df, png_filename, f"{label} Forecast for PV ({target_lon}, {target_lat})")
        print(f"Saved Plot: {png_filename}")

        # Save plot for a specific day (example: day == 6)
        png_filename = os.path.join(output_dir, f"pv_forecast_{label}_day_6_only.png")
        df["forecast_time"] = pd.to_datetime(df["forecast_time"])
        df_day6 = df[df["forecast_time"].dt.day == 6].copy()
        plot_forecast(df_day6, png_filename, f"{label} Forecast for PV ({target_lon}, {target_lat}) - Day 6 Only")
        print(f"Saved Plot: {png_filename}")
    
    # Build a combined DataFrame for daily subplots
    df_all = pd.DataFrame(all_results)
    if df_all.empty:
        print("No aggregated results to plot per day.")
        return

    # Ensure datetime parsing succeeded
    df_all["forecast_time"] = pd.to_datetime(df_all["forecast_time"], errors="coerce")
    df_all = df_all.dropna(subset=["forecast_time"])

    # Extract date for grouping
    df_all["date"] = df_all["forecast_time"].dt.date

    # Generate per-day combined subplot PNGs
    unique_dates = df_all["date"].unique()
    for date in sorted(unique_dates):
        df_day = df_all[df_all["date"] == date].copy()
        if df_day.empty:
            continue
        date_str = date.strftime("%Y-%m-%d")
        png_path = os.path.join(output_dir, f"combined_forecast_{date_str}.png")
        plot_forecast_subplots(df_day, png_path, f"Forecast for PV ({target_lon}, {target_lat}) - {date_str}")
        print(f"📊 Saved Plot: {png_path}")

def load_graphs(filename):
    """Load a list of graphs from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

class LSTM_GAT(nn.Module):
    """
    LSTM + GAT model:
    - The dynamic (time-series) part goes into a 2-layer LSTM (input_size=1).
    - The last LSTM hidden state is concatenated with static features.
    - A GAT layer aggregates over the graph; a final linear head predicts per-node horizons.
    """
    def __init__(self, in_features, lstm_hidden, n_hidden, n_heads, num_classes, num_static_features, dropout=0.4):
        super(LSTM_GAT, self).__init__()
        self.num_static = num_static_features
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, num_layers=2,
                            batch_first=True, dropout=dropout)
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
        # Split features into static and dynamic parts
        static = x[:, :self.num_static]
        dynamic = x[:, self.num_static:]

        # LSTM expects shape [N, T, 1] (batch_first=True); here N is the number of nodes
        dynamic_seq = dynamic.unsqueeze(-1)
        lstm_out, _ = self.lstm(dynamic_seq)

        # Take last timestep hidden state
        lstm_last = lstm_out[:, -1, :]

        # Concatenate with static features
        combined = torch.cat([lstm_last, static], dim=1)

        # Graph attention over nodes
        x = self.gat(combined, edge_index, edge_attr)
        x = F.elu(x)

        # Final per-node, per-horizon regression head
        return F.relu(self.fc(x))

# ----------------- Paths & Inference -----------------

pkl_path_for_test = r"C:\Users\<user>\Desktop\relevant_directories\new_model_changing dataset\test_pkl_try_somthing.pkl"

# Load test graphs
test_graphs = load_graphs(pkl_path_for_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM_GAT(
    in_features=test_graphs[0].x.shape[1],
    lstm_hidden=32,
    n_hidden=32,
    n_heads=4,
    num_classes=4,             # 4 horizons: t+15, t+30, t+45, t+60
    num_static_features=8,     # adjust if your static feature count differs
    dropout=0.4
)

model.load_state_dict(torch.load(
    r"C:\Users\<user>\Desktop\relevant_directories\new_model_changing dataset\model_weights_testing_something.pkl"
))
model.to(device)
model.eval()

output_dir = r"C:\Users\<user>\Desktop\relevant_directories\model_new\actual_vs_pred"
export_forecast_per_horizon_for_single_pv(model, test_graphs, device, 34.34, 31.45, output_dir)
