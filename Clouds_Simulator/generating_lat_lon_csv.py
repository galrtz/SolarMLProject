import os
import pandas as pd
import re

# === CONFIGURATION ===
input_dir = r"C:/Users/galrt/Desktop/final_project/NREL/data/NREL/raw"
output_path = r"C:/Users/galrt/Desktop/final_project/clouds_sim/pv_coordinates_2017.csv"

# === EXTRACT UNIQUE PVs FROM 2017 ===
all_files = os.listdir(input_dir)

unique_pvs = {}
for file in all_files:
    match = re.match(r"(\d+)_([\d.]+)_([\d.]+)_(\d{4})", file)
    if match:
        pvid, lat, lon, year = match.groups()
        if year == "2017" and pvid not in unique_pvs:
            unique_pvs[pvid] = {
                "PV_ID": int(pvid),
                "latitude": float(lat),
                "longitude": float(lon)
            }

# === SAVE AS CSV ===
df = pd.DataFrame(list(unique_pvs.values()))
df.to_csv(output_path, index=False)

print(f"Saved {len(df)} PV coordinates to: {output_path}")
