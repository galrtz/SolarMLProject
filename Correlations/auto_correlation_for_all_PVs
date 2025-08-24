import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf

# --- CONFIGURATION ---
csv_path = r"C:\Users\<user>\Desktop\final_project_directory\correlation_data_proccessing\clouds_simv\simulated_pv_ghi_all.csv"
output_path = r"C:\Users\<user>\Desktop\final_project_directory\correlation_data_proccessing\clouds_simv\auto_correclation/all_auto_correlation.csv"
max_lag = 96 * 364  # 15-min intervals × 96 per day × 7 days = one week of lags

# --- LOAD DATA ---
df = pd.read_csv(csv_path)

# Assume the first column is the PV station ID (e.g., 'PV_ID')
# Modify this line if the ID is under a different column
pv_names = df.iloc[:, 0]  # Save PV names for later

# Drop latitude and longitude columns based on their names
columns_to_drop = [col for col in df.columns if 'latitude' in col.lower() or 'longitude' in col.lower()]
df_no_meta = df.drop(columns=columns_to_drop)

# Remove the PV name column from GHI values (keep only numeric time series)
ghi_only = df_no_meta.drop(columns=df_no_meta.columns[0])

# --- COMPUTE AUTOCORRELATION FOR EACH PV ---
autocorr_matrix = []

for idx, row in ghi_only.iterrows():
    ghi_vec = pd.to_numeric(row, errors='coerce').dropna()
    print(len(ghi_vec))
    # if len(ghi_vec)-2 < max_lag:
    #     print(f"Station {pv_names.iloc[idx]} has too few values. Filling with NaNs.")
    #     autocorr = [np.nan] * (max_lag + 1)
    # else:
    autocorr = acf(ghi_vec, nlags=max_lag, fft=True)
    autocorr_matrix.append(autocorr)

# --- CREATE FINAL AUTOCORRELATION DATAFRAME ---
autocorr_df = pd.DataFrame(autocorr_matrix)
autocorr_df.columns = [f"lag_{i}" for i in range(len(autocorr_df.columns))]
autocorr_df.insert(0, "pv_id", pv_names)

# --- SAVE TO CSV ---
autocorr_df.to_csv(output_path, index=False)
print(f"Autocorrelation results saved to: {output_path}")

