import pandas as pd
import numpy as np
import os

# === Input directory: contains files like forecast_node0_t+15.csv, etc.
input_dir = r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\model_new\actual_vs_pred_new\per_node_per_horizon_csvs"

# === Prediction horizons
horizons = ["t+15", "t+30", "t+45", "t+60"]

# === Initialize result dictionaries
nrmse_results = {}
nmae_results = {}

# === Loop over each prediction horizon
for horizon in horizons:
    total_squared_error = 0.0  # numerator for NRMSE
    total_absolute_error = 0.0  # numerator for NMAE
    total_denominator_mae = 0.0  # denominator for NMAE
    total_points = 0  # total number of prediction points n*T

    for file in os.listdir(input_dir):
        if file.endswith(f"{horizon}.csv"):
            path = os.path.join(input_dir, file)
            df = pd.read_csv(path)

            y_true = df["target"].values
            y_pred = df["prediction"].values

            if len(y_true) == 0 or np.max(y_true) == 0:
                continue  # skip invalid entries

            # === NRMSE calculation ===
            # Formula:
            #   NRMSE_h = sqrt( (1 / (n*T)) * Σ_i Σ_t [ (y_pred - y_true)^2 / (pv_max^2) ] )
            pv_max = np.max(y_true)
            squared_errors = ((y_pred - y_true) / pv_max) ** 2
            total_squared_error += np.sum(squared_errors)

            # === NMAE calculation ===
            # Formula:
            #   NMAE_h = (1 / (Σ|y_true|)) * Σ|y_pred - y_true|
            total_absolute_error += np.sum(np.abs(y_pred - y_true))
            total_denominator_mae += np.sum(np.abs(y_true))

            # Count total number of prediction points
            total_points += len(y_true)

    # Final NRMSE and NMAE for the horizon
    if total_points > 0 and total_denominator_mae > 0:
        nrmse = np.sqrt(total_squared_error / total_points)
        nmae = total_absolute_error / total_denominator_mae
        nrmse_results[horizon] = nrmse
        nmae_results[horizon] = nmae
    else:
        nrmse_results[horizon] = None
        nmae_results[horizon] = None

# === Print Results ===
print("Final averaged errors per prediction horizon:\n")
print("NRMSE (Normalized Root Mean Squared Error):")
for h, val in nrmse_results.items():
    print(f"{h}: {val:.4f}" if val is not None else f"{h}: No valid data")

print("\nNMAE (Normalized Mean Absolute Error):")
for h, val in nmae_results.items():
    print(f"{h}: {val:.4f}" if val is not None else f"{h}: No valid data")
