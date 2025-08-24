import pandas as pd

def compute_acf_stats(acf_csv_path):
    """
    Computes mean and variance of autocorrelations per lag across all PVs.

    Parameters:
        acf_csv_path (str): Path to the CSV file containing ACF values.
                            Assumes first column is 'pv_id', and the rest are 'lag_0', 'lag_1', ...

    Returns:
        pd.DataFrame: DataFrame with two rows: 'mean' and 'variance', columns per lag.
    """
    # Load ACF data
    df = pd.read_csv(acf_csv_path)

    # Drop non-numeric columns (e.g., 'pv_id')
    acf_data = df.drop(columns=[col for col in df.columns if not col.startswith("lag_")])

    # Compute mean and variance per lag
    mean_per_lag = acf_data.mean(axis=0)
    var_per_lag = acf_data.var(axis=0, ddof=0)  # Population variance

    # Combine into one DataFrame
    stats_df = pd.DataFrame([mean_per_lag, var_per_lag], index=["mean", "variance"])

    return stats_df

acf_path = r"C:\Users\<user>\Desktop\final_project_directory\correlation_data_proccessing\clouds_simv\auto_correclation/all_auto_correlation.csv"
acf_stats = compute_acf_stats(acf_path)

# Save to CSV
acf_stats.to_csv(r"C:\Users\<user>>\Desktop\final_project_directory\correlation_data_proccessing\clouds_simv\auto_correclation/mean_and_var.csv")
print(" Saved mean/variance per lag to: acf_lag_statistics.csv")
