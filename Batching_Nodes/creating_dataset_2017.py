import os
import pandas as pd
import numpy as np

# נתיב לתיקייה עם כל קבצי ה־PV
input_dir = "C:/Users/galrt/Desktop/final_project/NREL/data/NREL/raw"
output_path = "C:/Users/galrt/Desktop/final_project/processed_GHI_dataset.csv"

past_steps = [i * 2 for i in range(1, 49)]  # 30 דקות אחורה
future_steps = [1, 2, 3, 4]  # 15, 30, 45, 60 דקות קדימה

all_records = []

for file_name in os.listdir(input_dir):
    if file_name.endswith(".csv") and "2017" in file_name:
        try:
            file_path = os.path.join(input_dir, file_name)
            print(f"📄 Processing: {file_name}")

            # קריאה מלאה
            df_full = pd.read_csv(file_path, skiprows=2, header=0)

            # חילוץ metadata משורת index=1 (השורה שמתחת ל-header)
            df_meta = pd.read_csv(file_path, nrows=2, header=None)
            location_id = df_meta.iloc[1, 1]
            latitude = float(df_meta.iloc[1, 5])
            longitude = float(df_meta.iloc[1, 6])

            # המרת עמודות רלוונטיות למספרים
            df = df_full.copy()
            df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
            df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
            df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
            df['Minute'] = pd.to_numeric(df['Minute'], errors='coerce')
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['GHI'] = pd.to_numeric(df['GHI'], errors='coerce')

            df = df[df['Year'] == 2017]
            df = df.reset_index(drop=True)

            # סינון לשעה 11:00
            df_target = df[(df['Hour'] == 11) & (df['Minute'] == 0)].copy()
            df_target = df_target.reset_index()

            for _, row in df_target.iterrows():
                ghi_past = []
                valid = True

                for step in past_steps:
                    past_idx = row['index'] - step
                    if past_idx >= 0:
                        ghi_val = df.iloc[past_idx]['GHI']
                        ghi_past.append(ghi_val)
                    else:
                        valid = False
                        break

                ghi_future = []
                for step in future_steps:
                    future_idx = row['index'] + step
                    if future_idx < len(df):
                        ghi_val = df.iloc[future_idx]['GHI']
                        ghi_future.append(ghi_val)
                    else:
                        valid = False
                        break

                if valid and not (any(pd.isnull(ghi_past)) or any(pd.isnull(ghi_future))):
                    record = {
                        'PV_ID': location_id,
                        'latitude': latitude,
                        'longitude': longitude,
                        'Month': row['Month'],
                        'Day': row['Day'],
                        'Hour': row['Hour'],
                        'Minute': row['Minute']
                    }

                    for i, val in enumerate(ghi_past):
                        record[f"GHI_t-{(i+1)*30}min"] = val

                    for i, val in enumerate(ghi_future):
                        record[f"GHI_t+{(i+1)*15}min"] = val

                    all_records.append(record)

        except Exception as e:
            print(f"❌ Error in {file_name}: {e}")
            continue

# שמירת הקובץ הסופי
df_final = pd.DataFrame(all_records)
df_final.to_csv(output_path, index=False)
print(f"\n✅ Data saved to: {output_path}")
