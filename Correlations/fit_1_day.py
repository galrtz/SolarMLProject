import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Load ACF statistics file ---
df = pd.read_csv(r"C:\Users\<user>\Desktop\final_project_directory\correlation_data_proccessing\nrel_correlation\auto_correlation\acf_lag_statistics.csv", index_col=0)

# --- Extract mean and std rows ---
mean_row = df.loc["mean"]
std_row = np.sqrt(df.loc["variance"])

# --- Filter lags 0–96 ---
lags = [f"lag_{i}" for i in range(97)]
mean_row = mean_row[lags]
std_row = std_row[lags]

# --- Convert index ---
mean_row.index = mean_row.index.str.replace("lag_", "").astype(int)
std_row.index = std_row.index.str.replace("lag_", "").astype(int)

# --- Prepare x, y ---
x = mean_row.index.to_numpy()
y = mean_row.values
yerr = std_row.values

# --- Define cosine fit function ---
def cos_func(x, a, P, phi, c):
    return a * np.cos(2 * np.pi * x / P + phi) + c

# --- Fit ---
initial_guess = (1, 96, 0, 0)
popt, pcov = curve_fit(cos_func, x, y, p0=initial_guess)

# --- Residual analysis ---
residuals = y - cos_func(x, *popt)
rss = np.sum(residuals**2)              # Residual sum of squares
dof = len(x) - len(popt)                # Degrees of freedom
mse = rss / dof                         # Mean squared error
rmse = np.sqrt(mse)                     # Root mean squared error
ss_total = np.sum((y - np.mean(y))**2)  # Total sum of squares
r_squared = 1 - (rss / ss_total)        # R^2

# --- Print fit statistics ---
print("Fit Statistics:")
print(f"RSS   = {rss:.4f}")
print(f"DoF   = {dof}")
print(f"MSE   = {mse:.6f}")
print(f"RMSE  = {rmse:.6f}")
print(f"R²    = {r_squared:.4f}")

# --- Plot ---
plt.figure(figsize=(12, 5))
plt.errorbar(x, y, yerr=yerr, fmt='o', ecolor='orange', capsize=3, markersize=3, label="Mean ACF ± STD")
plt.plot(x, cos_func(x, *popt), 'r--',
         label=f"Cosine Fit: a·cos(2πx/P + φ) + c\n"
               f"a={popt[0]:.3f}, P={popt[1]:.2f}, φ={popt[2]:.2f}, c={popt[3]:.3f}")
plt.title("Cosine Fit to Mean Autocorrelation (Lag 0–96)")
plt.xlabel("Lag (15-minute intervals)")
plt.ylabel("Mean Autocorrelation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\<user>\Desktop\final_project_directory\correlation_data_proccessing\nrel_correlation\auto_correlation\acf_fits_1_day.png", dpi=300, transparent=True)  # <-- Save as PNG (300 DPI is print quality)

plt.show()
