import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load the ACF statistics file ---
df = pd.read_csv(r"C:\Users\<user>\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\correlation_data_proccessing\nrel_correlation\auto_correlation\acf_lag_statistics.csv", index_col=0)

# --- Extract mean and variance rows ---
mean_row = df.loc["mean"]
var_row = df.loc["variance"]
std_row = np.sqrt(var_row)

# --- Convert index from "lag_x" to integers ---
mean_row.index = mean_row.index.str.replace("lag_", "").astype(int)
std_row.index = std_row.index.str.replace("lag_", "").astype(int)

# --- Select top 364 highest mean autocorrelation values ---
top_means = mean_row.nlargest(364)
top_stds = std_row[top_means.index]

# --- Plot: simple x-axis 1..364 ---
x = np.arange(1, 365)
y = top_means.values
yerr = top_stds.values

plt.figure(figsize=(12, 5))
plt.errorbar(
    x=x,
    y=y,
    yerr=yerr,
    fmt='o',
    color='#4A708B',          # ורוד לנקודות (Hot Pink)
    ecolor='#A0525A',         #
    elinewidth= 1,
    capsize=3,
    markersize=3,
    label="Top 364 Mean ACF ± STD"
)

plt.title("Top 364 Mean Autocorrelation Values (Ranked) with Standard Deviation")
plt.xlabel("Ranked Index (1 = Highest Mean ACF)")
plt.ylabel("Mean Autocorrelation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\correlation_data_proccessing\nrel_correlation\auto_correlation\acf_364.png", dpi=300, transparent=True)  # <-- Save as PNG (300 DPI is print quality)

plt.show()
