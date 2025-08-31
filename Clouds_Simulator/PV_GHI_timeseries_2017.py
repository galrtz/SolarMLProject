import pandas as pd
import os

input_dir = r"C:\Users\galrt\Desktop\final_project\NREL\data\NREL\raw"
output_file = r"C:\Users\galrt\Desktop\final_project\PV_GHI_2017.csv"

ghi_dict = {}

for filename in os.listdir(input_dir):
    if not (filename.endswith(".csv") and "2017" in filename):
        continue

    file_path = os.path.join(input_dir, filename)
    try:
        meta = pd.read_csv(file_path, nrows=2, header=None)
        pv_id = str(meta.iloc[1, 1])

        df = pd.read_csv(file_path, skiprows=2)
        df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        df = df[['timestamp', 'GHI']].copy()
        df = df.dropna()

        df = df.set_index('timestamp')
        df = df.sort_index()

        ghi_dict[pv_id] = df['GHI']

        print(f"Finished processing {filename}")

    except Exception as e:
        print(f"Error in file {filename}: {e}")

df_all = pd.DataFrame(ghi_dict)
df_all.index.name = "timestamp"

df_all.to_csv(output_file)
print("Saved at", output_file)

