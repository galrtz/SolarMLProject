import os
import shutil
from datetime import datetime

csv_dir = r"C:\Users\hadar\Desktop\אוניברסיטה\פרויקט גמר\relevant_directories\relevant\testing_2\data_2018_train_splitting"
output_dir = os.path.join(csv_dir, "filtered_train_data")
os.makedirs(output_dir, exist_ok=True)

group1_times = [(4, 0), (6, 15), (8, 30), (10, 45), (13, 0), (15, 15)]
group2_times = [(4, 30), (6, 45), (9, 0), (11, 15), (13, 30), (15, 45)]

def matches_time(hour, minute, group_times):
    return any(hour == h and minute == m for h, m in group_times)

# מיון כל הקבצים לפי תאריך
all_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
all_files.sort(key=lambda f: datetime.strptime(f, "data_%Y_%m_%d_%H_%M.csv"))

# קיבוץ לפי יום
files_by_day = {}
for file in all_files:
    dt = datetime.strptime(file, "data_%Y_%m_%d_%H_%M.csv")
    day_key = (dt.year, dt.month, dt.day)
    files_by_day.setdefault(day_key, []).append((file, dt))

for i, ((year, month, day), files) in enumerate(sorted(files_by_day.items())):
    if i % 3 != 0:
        continue  # רק כל יום שלישי (3rd day in each 4-day group)

    # החלפה בין קבוצות שעות כל ארבעה ימים
    group_times = group1_times if (i // 2) % 2 == 0 else group2_times

    for file, dt in files:
        if matches_time(dt.hour, dt.minute, group_times):
            src_path = os.path.join(csv_dir, file)
            dst_path = os.path.join(output_dir, file)
            shutil.copy(src_path, dst_path)


print(f"✅ Copied filtered files to: {output_dir}")
