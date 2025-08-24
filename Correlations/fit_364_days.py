import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Load the ACF statistics file ---
df = pd.read_csv(r"C:\Users\<user>\Desktop\correlation_data_proccessing\nrel_correlation\auto_correlation\acf_lag_statistics.csv", index_col=0)

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

# --- Prepare x/y for fitting ---
x = np.arange(1, 365)
y = top_means.values
yerr = top_stds.values

# --- Define exponential fit function ---
def exp_decay(x, a, b):
    return a * np.exp(-b * x)

# --- Fit exponential ---
popt_exp, _ = curve_fit(exp_decay, x, y, p0=(1, 0.01))

# --- Fit polynomial (degree 3 & 4) ---
coeffs_poly3 = np.polyfit(x, y, 3)
poly_fit3 = np.poly1d(coeffs_poly3)

coeffs_poly4 = np.polyfit(x, y, 4)
poly_fit4 = np.poly1d(coeffs_poly4)

# --- Plot both fits ---
plt.figure(figsize=(12, 5))

# # Error bars for data
# plt.errorbar(
#     x, y, yerr=yerr,
#     fmt='o', ecolor='orange', capsize=3, markersize=3,
#     label="Top 364 ACF ± STD"
# )

# --- Plot both fits ---

# Scatter points without error bars
plt.plot(x, y, 'bo', markersize=3, label="Top 364 ACF")

# Plot exponential fit
plt.plot(x, exp_decay(x, *popt_exp), 'g-', linewidth=2,
         label=f"Exp Fit: a·exp(-b·x)\n a={popt_exp[0]:.3f}, b={popt_exp[1]:.5f}")

# Plot polynomial degree 3
a3, b3, c3, d3 = coeffs_poly3
plt.plot(x, poly_fit3(x), 'r--', linewidth=2,
         label=f"Poly (deg=3):\n y = {a3:.1e}·x³ + {b3:.1e}·x² + {c3:.1e}·x + {d3:.1e}")

# Plot polynomial degree 4
a4, b4, c4, d4, e4 = coeffs_poly4
plt.plot(x, poly_fit4(x), 'm-.', linewidth=2,
         label=f"Poly (deg=4):\n y = {a4:.1e}·x⁴ + {b4:.1e}·x³ + {c4:.1e}·x² + {d4:.1e}·x + {e4:.1e}")

plt.title("Exponential and Polynomial Fits to Top 364 Ranked ACF Values")
plt.xlabel("Ranked Index (1 = Highest Mean ACF)")
plt.ylabel("Mean Autocorrelation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\<user>\Desktop\correlation_data_proccessing\nrel_correlation\auto_correlation\acf_fits_364_day.png", dpi=300, transparent=True)  # <-- Save as PNG (300 DPI is print quality)
plt.show()

# --- R² helper function ---
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# --- RMSE helper function ---
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# --- Predictions ---
y_exp = exp_decay(x, *popt_exp)
y_poly3 = poly_fit3(x)
y_poly4 = poly_fit4(x)

# --- Calculate metrics ---
r2_exp = r_squared(y, y_exp)
rmse_exp = rmse(y, y_exp)

r2_poly3 = r_squared(y, y_poly3)
rmse_poly3 = rmse(y, y_poly3)

r2_poly4 = r_squared(y, y_poly4)
rmse_poly4 = rmse(y, y_poly4)

# --- Print results ---
print(f"Exponential Fit:  R² = {r2_exp:.4f},  RMSE = {rmse_exp:.4e}")
print(f"Poly Degree 3:    R² = {r2_poly3:.4f},  RMSE = {rmse_poly3:.4e}")
print(f"Poly Degree 4:    R² = {r2_poly4:.4f},  RMSE = {rmse_poly4:.4e}")

