import pandas as pd

# נתיב לקובץ המאוחד של 2017
input_csv = "C:/Users/galrt/Desktop/final_project/processed_GHI_dataset.csv"  # שנה בהתאם לשם הקובץ שלך
output_csv = "C:/Users/galrt/Desktop/final_project/ghi_27_1_2017.csv"

# קריאת הקובץ
df = pd.read_csv(input_csv)

# סינון לפי תאריך
filtered_df = df[(df['Day'] == 27) & (df['Month'] == 1)]

# שמירה לקובץ חדש
filtered_df.to_csv(output_csv, index=False)

print(f"✅ נשמר קובץ מסונן: {output_csv}")
