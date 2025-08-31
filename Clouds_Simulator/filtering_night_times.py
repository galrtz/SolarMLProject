import pandas as pd

input_file = r"C:\Users\galrt\Desktop\final_project\PV_GHI_2017.csv"
output_file_filtered = r"C:\Users\galrt\Desktop\final_project\PV_GHI_2017_filtered_4_to_16.csv"

df = pd.read_csv(input_file, parse_dates=['timestamp'])
df = df.set_index('timestamp')
df_filtered = df.between_time("04:00", "16:00")
df_filtered.to_csv(output_file_filtered)

print("Finished and saved at path:", output_file_filtered)
