import pandas as pd
import os

def normalize_csv(file_path,output_folder):
    """
    Normalize all columns except PV_ID in a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file to normalize.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Identify the columns to normalize (exclude PV_ID)
    columns_to_normalize = [col for col in df.columns if col != 'PV_ID']

    # Normalize the selected columns
    for col in columns_to_normalize:
        mean = df[col].mean()
        std = df[col].std()
        if std != 0:  # Avoid division by zero
            df[col] = (df[col] - mean) / std

    # Save the normalized DataFrame to a new file in the output folder

    base_name = os.path.basename(file_path).replace('.csv', '_normalized.csv')
    new_file_path = os.path.join(output_folder, base_name)
    df.to_csv(new_file_path, index=False)

def normalize_folder(folder_path,output_folder ):
    """
    Normalize all CSV files in a folder by processing all columns except PV_ID.

    Parameters:
    - folder_path (str): Path to the folder containing CSV files.
    """
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a CSV
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Normalizing file: {file_name}")
            normalize_csv(file_path, output_folder)

# Example usage
folder_path = "C:/Users/hadar/Desktop/new_new_new" # Replace with the path to your folder containing CSV files
normalize_folder("C:/Users/hadar/Desktop/output2", folder_path)
