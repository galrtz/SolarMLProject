import pandas as pd
import os


def process_csv_directory(csv_directory, number_of_prev_time=20, number_of_days=5):
    data = {}

    target_times = [('8', '30'), ('9', '30'), ('10', '30'), ('11', '30'), ('12', '30'),
                    ('13', '30'), ('14', '30'), ('15', '30'), ('16', '30')]

    for file_name in os.listdir(csv_directory):
        if file_name.endswith('.csv') and '2017' in file_name:
            file_path = os.path.join(csv_directory, file_name)
            print(f"Processing file: {file_name}")  # Debug print to check which file is being processed

            try:
                df_raw = pd.read_csv(file_path, header=None, skiprows=1, low_memory=False)
                print(f"Read file {file_name}: {df_raw.head()}")  # Check if the file is read correctly


                # Step 3: Extract metadata from the raw file
                pv_id = df_raw.iloc[0, 1]  # Extract location ID (Row 1, Column 2 in the file).
                latitude = df_raw.iloc[0, 5]  # Extract latitude (Row 1, Column 6 in the file).
                longitude = df_raw.iloc[0, 6]  # Extract longitude (Row 1, Column 7 in the file).

                # Step 4: Read the actual data table starting from the proper row
                df_raw.columns = df_raw.iloc[1]  # Set column headers based on the second row in the file.
                df = df_raw.iloc[2:].reset_index(drop=True)
                # Extract data rows starting from Row 3 and reset the index.


                if 'GHI' in df.columns and 'Hour' in df.columns and 'Minute' in df.columns and 'Day' in df.columns:
                    for target_hour, target_minute in target_times:
                        df_filtered = df[(df['Hour'] == target_hour) & (df['Minute'] == target_minute)]
                        if df_filtered.empty:
                            print(f"No data for {target_hour}:{target_minute} on any day in {file_name}")
                            continue  # Skip this time slot if no data is found

                        for index, row in df_filtered.iterrows():
                            if int(row['Day']) > number_of_days:
                                continue

                            ghi_t = row['GHI']
                            hour = row['Hour']
                            minute = row['Minute']
                            day = row['Day']

                            prev_ghi = []
                            for i in range(1, number_of_prev_time + 1):
                                if index - i >= 0:
                                    prev_ghi.append(df['GHI'].shift(i).iloc[index])
                                else:
                                    break

                            key = (day, hour, minute)

                            if key not in data:
                                data[key] = {}

                            if pv_id not in data[key]:
                                data[key][pv_id] = {}

                            data[key][pv_id]['longitude'] = longitude
                            data[key][pv_id]['latitude'] = latitude
                            data[key][pv_id]['pv_id'] = pv_id
                            data[key][pv_id]['ghi_t'] = ghi_t

                            data[key][pv_id]['previous_ghi'] = {}
                            for i, value in enumerate(prev_ghi, start=1):
                                real_time = 15 * i
                                data[key][pv_id]['previous_ghi'][f"GHI_t-{real_time} minutes"] = value

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                raise  # Raise error to stop execution for debugging

    return data


def datasets(csv_directory, output_directory, number_of_prev_time=20, number_of_days=5):
    data = process_csv_directory(csv_directory, number_of_prev_time, number_of_days)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for (day, hour, minute), pv_data in data.items():
        df = pd.DataFrame([
            {
                'PV_ID': pv_id,
                'longitude': value['longitude'],
                'latitude': value['latitude'],
                'ghi_t': value['ghi_t'],
                **value['previous_ghi']
            }
            for pv_id, value in pv_data.items()
        ])

        output_file_path = os.path.join(output_directory, f"Day_{day}_Hour_{hour}_Minute_{minute}.csv")

        df.to_csv(output_file_path, index=False)
        print(f"Processed data saved to {output_file_path}")


datasets("C:/Users/hadar/Desktop/raw-20241220T103937Z-001/raw", "C:/Users/hadar/Desktop/output2", 20, 5)
