import pandas as pd
import os


def process_csv_directory(csv_directory, output_file, number_of_prev_hours=24):
    data = []

    # טווח התאריכים הרצוי (20.1 עד 8.2)
    start_day, start_month = 20, 1
    end_day, end_month = 8, 2

    # שעות היעד
    target_times = [
        ('11', '0', "Morning"),  # 11:00 בבוקר
        ('13', '30', "Noon")     # 13:30 צהריים
    ]

    for file_name in os.listdir(csv_directory):
        if file_name.endswith('.csv') and '2017' in file_name:
            file_path = os.path.join(csv_directory, file_name)
            print(f"Processing file: {file_name}")  # Debug print to check which file is being processed

            try:
                # קריאת הקובץ עם דילוג על המטא-נתונים ושימוש בשורה השלישית ככותרת
                df_raw = pd.read_csv(file_path, header=None, skiprows=1, low_memory=False)
                print(f"Read file {file_name}: {df_raw.head()}")  # Debugging

                # Extract metadata
                pv_id = df_raw.iloc[0, 1]  # Extract location ID
                latitude = df_raw.iloc[0, 5]  # Extract latitude
                longitude = df_raw.iloc[0, 6]  # Extract longitude

                # Step 4: Read the actual data table starting from the proper row
                df_raw.columns = df_raw.iloc[1]  # Set column headers based on the second row in the file.
                df = df_raw.iloc[2:].reset_index()

                # המרת Month ו-Day למספרים
                df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
                df['Day'] = pd.to_numeric(df['Day'], errors='coerce')

                # Ensure required columns exist
                if 'GHI' in df.columns and 'Hour' in df.columns and 'Minute' in df.columns and 'Day' in df.columns:
                    # סינון לפי טווח תאריכים רצוי
                    df = df[((df['Month'] == 1) & (df['Day'] >= start_day-1)) |
                            ((df['Month'] == 2) & (df['Day'] <= end_day))]

                    for target_hour, target_minute, time_category in target_times:
                        df_filtered = df[(df['Hour'] == (target_hour)) & (df['Minute'] == (target_minute))]

                        if df_filtered.empty:
                            print(f"No data for {target_hour}:{target_minute} on any day in {file_name}")
                            continue  # Skip if no data found

                        for _, row in df_filtered.iterrows():
                            if (row['Day'] != start_day-1):
                                ghi_t = row['GHI']
                                day = row['Day']
                                month = row['Month']
                                hour = row['Hour']
                                minute = row['Minute']

                                # מוצא את המיקום של השורה המקורית בתוך df
                                index_loc = df.index.get_loc(row.name)

                                prev_ghi = []
                                for i in range(1, number_of_prev_hours + 1):
                                    shifted_index = index_loc - (i * 4)  # זזים אחורה ב-4*i שורות
                                    if shifted_index >= 0:  # מוודא שהאינדקס חוקי
                                        prev_ghi.append(df['GHI'].iloc[shifted_index])
                                    else:
                                        prev_ghi.append(None)  # אם אין נתונים אחורה, נכניס None

                                record = {
                                    'PV_ID': pv_id,
                                    'longitude': longitude,
                                    'latitude': latitude,
                                    'ghi_t': ghi_t,
                                    'Day': day,
                                    'Month': month,
                                    'Hour': hour,
                                    'Minute': minute,
                                    'Time_Category': time_category,
                                    **{f"GHI_t-{h} hours": value for h, value in enumerate(prev_ghi, start=1)}
                                }
                                data.append(record)

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                raise  # Stop execution for debugging

    # Convert collected data to a DataFrame
    df_output = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df_output.to_csv(output_file, index=False)
    print(f"Data successfully saved to {output_file}")


# הפעלת הפונקציה עם הנתיב המתאים
process_csv_directory(
    csv_directory="C:/Users/hadar/Desktop/raw-20241220T103937Z-001/raw",
    output_file="C:/Users/hadar/Desktop/2017/filtered_11_13_30.csv",
    number_of_prev_hours=24
)
