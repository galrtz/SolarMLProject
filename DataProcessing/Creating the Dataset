import pandas as pd  # Import pandas library for data manipulation and analysis.
import os  # Import os library to interact with the operating system for file handling.


def process_csv_directory(csv_directory, output_directory, number_of_raw=34, number_of_prev_time=20, day=1, hour= 8.5):
    """
    Function to process CSV files in a given directory, extract the required features, and save the results to an output CSV file.

    Parameters:
    - csv_directory (str): The path to the directory containing the raw CSV files.
    - output_file (str): The path where the processed CSV file should be saved.
    - number_of_raw (int): The index of the row to extract GHI data (default is 34).
    - number_of_prev_time (int): Number of previous time steps to include, spaced 15 minutes apart (default is 20).
    """

    # List to store processed data
    processed_data = []  # This list will hold dictionaries, each representing one processed row of data.

    # Iterate through all files in the specified directory
    for file_name in os.listdir(csv_directory):
        # os.listdir lists all files in the given directory.
        if file_name.endswith('.csv'):
            # Process only files with a .csv extension to avoid errors with other file types.
            file_path = os.path.join(csv_directory, file_name)
            # Combine the directory path with the file name to create the full file path.

            try:
                # Step 1: Read the raw file
                df_raw = pd.read_csv(file_path, header=None, skiprows=1, low_memory=False)
                # Load the file without headers to access metadata and raw data.

                # Step 2: Verify the file format and ensure it contains data for the year 2017
                if df_raw.iloc[5, 0] == '2017':
                    # Ensure the file matches the expected format by checking a specific cell.

                    # Step 3: Extract metadata from the raw file
                    location_id = df_raw.iloc[0, 1]  # Extract location ID (Row 1, Column 2 in the file).
                    latitude = df_raw.iloc[0, 5]  # Extract latitude (Row 1, Column 6 in the file).
                    longitude = df_raw.iloc[0, 6]  # Extract longitude (Row 1, Column 7 in the file).

                    # Step 4: Read the actual data table starting from the proper row
                    df_raw.columns = df_raw.iloc[1]  # Set column headers based on the second row in the file.
                    df = df_raw.iloc[2:].reset_index(drop=True)
                    # Extract data rows starting from Row 3 and reset the index.

                    # Step 5: Ensure the GHI column exists in the data
                    if 'GHI' in df.columns:
                        # Check if the 'GHI' column is present to avoid processing incomplete files.

                        # Step 6: Extract the target GHI value for the specified row
                        ghi_t = df.iloc[number_of_raw]['GHI'] if number_of_raw < len(df) else None
                        # Extract GHI for the specified row, or assign None if the row doesn't exist.

                        # Step 7: Generate shifted GHI values for previous time steps
                        prev_ghi = []
                        for i in range(1, number_of_prev_time + 1):
                            if int(df['Hour'].iloc[number_of_raw - i]) > 4:
                                prev_ghi.append(df['GHI'].shift(i).iloc[number_of_raw])
                            else:
                                break

                        # Step 8: Prepare a dictionary with all features
                        data_features = {
                            'location_id': location_id,  # Add location ID.
                            'latitude': latitude,  # Add latitude.
                            'longitude': longitude,  # Add longitude.
                            'GHI_t': ghi_t,  # Add current GHI value.
                        }

                        # Add GHI_t-1 to GHI_t-n as additional features
                        for i, value in enumerate(prev_ghi, start=1):
                            real_time = 15 * i
                            data_features[f"GHI_t-{real_time} minutes"] = value

                        # Append the dictionary to the list of processed data
                        processed_data.append(data_features)

            except Exception as e:
                # Catch and print any errors during processing to help with debugging.
                print(f"Error processing file {file_name}: {e}")

    output_file_name = f"processed_file_{day},{hour}.csv"
    output_file_path = os.path.join(output_directory, output_file_name)
    # Convert the list of processed dictionaries into a DataFrame
    processed_df = pd.DataFrame(processed_data)
    # Each dictionary becomes a row in the DataFrame, with keys as column names.

    # Save the processed DataFrame to a new CSV file
    processed_df.to_csv(output_file_path, index=False)
    # Save the data to the specified output file. The `index=False` ensures row indices are not written.
    print(f"Processed data saved to {output_file_path}")


def datasets(csv_directory, output_directory, number_of_raw=34, number_of_prev_time=20):
    for day in range(1,3):
        current_hour = number_of_raw + (day - 1)*96
        for hour in range(1,3):
            process_csv_directory(csv_directory, output_directory, current_hour, number_of_prev_time, day, hour+7.5)
            current_hour = current_hour + 4
            if current_hour == 66:
                break


datasets("C:/Users/hadar/Desktop/raw-20241220T103937Z-001/raw", "C:/Users/hadar/Desktop/output", 34, 20)
