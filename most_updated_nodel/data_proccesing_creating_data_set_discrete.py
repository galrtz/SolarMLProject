import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime, timedelta

def process_csv_directory(csv_directory, output_file, number_of_prev_hours=24):
    data = []
    target_hours = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    target_minutes = [0, 15, 30, 45]
    start_day, start_month = 1, 1 #Change here
    end_day, end_month = 29, 6 #Change here

    start_time = pd.Timestamp(year=2018, month=start_month, day=start_day) #Change here
    end_time = pd.Timestamp(year=2018, month=end_month, day=end_day) #Change here

    total_files = 0
    total_rows_collected = 0

    for file_name in tqdm(os.listdir(csv_directory)):
        if not file_name.endswith('.csv') or '2018' not in file_name: #Change here
            continue

        total_files += 1

        try:
            parts = file_name.replace(".csv", "").split("_")
            latitude = float(parts[1])
            longitude = float(parts[2])

            in_area = (
                (31.33 <= latitude <= 31.45 and 34.34 <= longitude <= 34.94) or
                (31.37 <= latitude <= 31.53 and 34.98 <= longitude <= 35.14) or
                (31.57 <= latitude <= 31.65 and 34.98 <= longitude <= 35.18) or
                (31.73 <= latitude <= 31.81 and 35.14 <= longitude <= 35.26) or
                (31.85 <= latitude <= 32.09 and 35.22 <= longitude <= 35.30)
            )
            if not in_area:
                continue

            file_path = os.path.join(csv_directory, file_name)
            df_raw = pd.read_csv(file_path, header=None, skiprows=1, low_memory=False)
            pv_id = df_raw.iloc[0, 1]
            df_raw.columns = df_raw.iloc[1]
            df = df_raw.iloc[2:].reset_index(drop=True)

            for col in ['Month', 'Day', 'Hour', 'Minute']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            if not all(col in df.columns for col in ['GHI', 'Hour', 'Minute']):
                continue

            df['datetime'] = pd.to_datetime({
                'year': 2018, #Change here
                'month': df['Month'],
                'day': df['Day'],
                'hour': df['Hour'],
                'minute': df['Minute']
            }, errors='coerce')
            df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)

            for idx, row in df.iterrows():
                now = row['datetime']
                hour = now.hour
                minute = now.minute
                new_time = now - timedelta(hours=1)


                if (hour in target_hours) and (minute in target_minutes) and (start_time <= now <= end_time):
                    ghi_t = row['GHI']

                    prev_ghi = []
                    for i in range(4, 4 + number_of_prev_hours + 1):
                        shifted_index = idx - i
                        if shifted_index >= 0:
                            prev_ghi.append(df['GHI'].iloc[shifted_index])
                        else:
                            prev_ghi.append(None)

                    prev_ghi_24_hours_gap = []
                    for i in range(1, 4):
                        shifted_index = idx - 96*i
                        if shifted_index >= 0:
                            prev_ghi_24_hours_gap.append(df['GHI'].iloc[shifted_index])
                        else:
                            prev_ghi_24_hours_gap.append(None)

                    target_ghi = []
                    for i in range(1, 4):
                        shifted_index = idx - i
                        if shifted_index >= 0:
                            target_ghi.append(df['GHI'].iloc[shifted_index])
                        else:
                            target_ghi.append(None)

                    record = {
                        'PV_ID': pv_id,
                        'longitude': longitude,
                        'latitude': latitude,
                        'ghi_t+60min': ghi_t,
                        **{f"GHI_t+{60-15*h}min": val for h, val in enumerate(target_ghi, start=1)},
                        # 'Day': now.day,
                        # 'Month': now.month,
                        # 'Hour': now.hour,
                        # 'Minute': now.minute,
                        'datetime': new_time,
                        # **{f"GHI_t-{h}h": val for h, val in enumerate(prev_ghi, start=0)},
                        **{f"GHI_t-{15*h}min": val for h, val in enumerate(prev_ghi, start=0)},
                        **{f"GHI_t-{24*h}h": val for h, val in enumerate(prev_ghi_24_hours_gap, start=1)}
                    }
                    data.append(record)
                    total_rows_collected += 1

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            raise

    df_output = pd.DataFrame(data)
    df_output.to_csv(output_file, index=False)
    print(f"\n✅ Done. {total_rows_collected} records collected from {total_files} files.")
    print(f"📦 Data saved to: {output_file}")

# הפעלה
process_csv_directory(
    csv_directory=r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\raw-20241220T103937Z-001\raw",
    output_file=r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\testing\data_2018_train.csv", #Change here
    number_of_prev_hours=8
)
