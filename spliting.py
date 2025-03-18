import pandas as pd
import os

# Load the CSV file
file_path = "C:/Users/hadar/Desktop/2017/decreasing_data.csv"  # Change to the correct path if needed
df = pd.read_csv(file_path)

# Create a directory to save the split files if it doesn't exist
output_dir = "C:/Users/hadar/Desktop/split_csv_files"
os.makedirs(output_dir, exist_ok=True)

# Group data by Month, Day, Hour, and Minute
grouped = df.groupby(["Month", "Day", "Hour", "Minute"])

# Save each group as a separate CSV file
for (month, day, hour, minute), group in grouped:
    filename = f"{output_dir}/data_{month:02d}_{day:02d}_{hour:02d}_{minute:02d}.csv"
    group.to_csv(filename, index=False)

print(f"Created {len(os.listdir(output_dir))} files in the {output_dir} directory")
