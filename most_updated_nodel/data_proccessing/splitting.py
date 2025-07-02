import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm

file_path = r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\testing\data_2018_train.csv"


df = pd.read_csv(file_path)
#
# #  dayfirst=True
# df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce', dayfirst=True)
# df = df.dropna(subset=["datetime"])
#
# df["Year"] = df["datetime"].dt.year
# df["Month"] = df["datetime"].dt.month
# df["Day"] = df["datetime"].dt.day
# df["Hour"] = df["datetime"].dt.hour

output_dir = r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\testing\data_2018_train_splitting"
# output_dir = r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\testing\data_2017_test_splitting"


os.makedirs(output_dir, exist_ok=True)
grouped = df.groupby(["datetime"])


for daytime, group in tqdm(grouped):
    # print(daytime)
    print(daytime)
    # print(group)
    dt = datetime.strptime(daytime[0], "%Y-%m-%d %H:%M:%S")  # dayfirst

    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute

    filename = f"data_{year:04d}_{month:02d}_{day:02d}_{hour:02d}_{minute:02d}.csv"
    # print(filename)
    full_path = os.path.join(output_dir, filename)
    group.to_csv(full_path, index=False)

print(f"✅ Created {len(os.listdir(output_dir))} CSV files in {output_dir}")
