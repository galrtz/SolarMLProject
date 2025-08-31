import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import torch
import SatelliteCloudGenerator as scg
import matplotlib.pyplot as plt
import pandas as pd
from pvlib.location import Location
from datetime import datetime, timedelta
# https://github.com/strath-ai/SatelliteCloudGenerator/tree/main?tab=readme-ov-file
# https://zenodo.org/records/5012942

NUM_TIMESTEPS = 96         # 96 שלבים = 24 שעות בקפיצות של 15 דקות
TIME_STEP_MINUTES = 15     # כל שלב הוא 15 דקות

def clear_sky_GHI(lat, lon, day):
    timezone = 'Asia/Jerusalem'
    site = Location(lat, lon, tz=timezone)

    start = pd.Timestamp(f"{day} 00:00", tz=timezone)
    end = pd.Timestamp(f"{day} 23:45", tz=timezone)
    times = pd.date_range(start, end, freq='15min')

    # Compute clear-sky GHI
    clearsky = site.get_clearsky(times)
    ghi_clear_sky = clearsky['ghi'].values  # Array of GHI values, shape (96,)

    return ghi_clear_sky

def calculate_cloud_speed_in_pixels(clear_img, speed_kmh, km_coverage):
    """
    מחשבת את מהירות העננים בפיקסלים לשעה עבור תמונה נתונה.

    :param speed_kmh: מהירות העננים בקמ"ש
    :param image_width: רוחב התמונה (בפיקסלים)
    :param image_height: גובה התמונה (בפיקסלים)
    :param km_coverage: היקף התמונה בקילומטרים
    :return: vx, vy - מהירות העננים בצירים X ו-Y בפיקסלים לשעה
    """

    image_height, image_width = clear_img.shape[2], clear_img.shape[3]

    # חישוב קילומטרים לכל פיקסל
    km_per_pixel_x = km_coverage / image_width  # כמה קילומטרים מייצג כל פיקסל בציר X
    km_per_pixel_y = km_coverage / image_height  # כמה קילומטרים מייצג כל פיקסל בציר Y

    # המרת מהירות קמ"ש לפיקסלים לשעה
    vx = (speed_kmh / km_per_pixel_x)  # מהירות בציר X (פיקסלים לשעה)
    vy = (speed_kmh / km_per_pixel_y)  # מהירות בציר Y (פיקסלים לשעה)

    return int(vx), int(vy)

def get_cloud_shadow(clear_img, vx, vy, num_of_timesteps=10):
    __, __, shadow_mask_tot = scg.add_cloud_and_shadow(clear_img, return_cloud=True)
    shadow_mask_tot = torch.clamp(shadow_mask_tot, min=0.2)
    # Sample the velocity of the cloud
    bs = clear_img.shape[0]
    vx = torch.randn(bs) * vx
    vy = torch.randn(bs) * vy
    shadow_mask_vs_batch = []
    for b in range(bs):
        shadow_mask_vs_t = []
        for t in range(num_of_timesteps):
            # Move the cloud with periodic boundary conditions
            shadow_mask = shadow_mask_tot[b]
            # Padd the image with the same values as the border
            # shadow_mask = torch.nn.functional.pad(shadow_mask.unsqueeze(0), (int(0.1*W), int(0.1*W), int(0.1*H), int(0.1*H)), mode='circular')[0]
            shadow_mask = torch.roll(shadow_mask, shifts=(int(vy[b]), int(vx[b])), dims=(1, 2))
            shadow_mask_vs_t.append(shadow_mask)
            shadow_mask_tot[b] = shadow_mask
        shadow_mask_vs_t = torch.stack(shadow_mask_vs_t)
        shadow_mask_vs_batch.append(shadow_mask_vs_t)
    shadow_mask_vs_batch = torch.stack(shadow_mask_vs_batch)
    return shadow_mask_vs_batch

def latlon_to_pixel(lat, lon, top_left_latlon, bottom_right_latlon, image):
    """
    מחשבת את מיקום הפיקסל בתמונה לפי קואורדינטות גאוגרפיות.

    :param lat: קו רוחב של הנקודה
    :param lon: קו אורך של הנקודה
    :param top_left_latlon: קואורדינטות (lat, lon) של הפינה השמאלית-עליונה של התמונה
    :param bottom_right_latlon: קואורדינטות (lat, lon) של הפינה הימנית-תחתונה של התמונה
    :param image: תמונת ה-PIL או np.array בגובה ורוחב כלשהם
    :return: (row, col) – מיקום הפיקסל
    """

    height, width = image.shape[2], image.shape[3]

    # קואורדינטות גבול
    lat_top, lon_left = top_left_latlon
    lat_bottom, lon_right = bottom_right_latlon

    # חישוב מיקום יחסי
    lat_ratio = (lat_top - lat) / (lat_top - lat_bottom)
    lon_ratio = (lon - lon_left) / (lon_right - lon_left)

    # המרה לפיקסלים
    row = int(lat_ratio * height)
    col = int(lon_ratio * width)
    row = max(0, min(row, height - 1))
    col = max(0, min(col, width - 1))
    return row, col

def simulate_ghi_for_pvs_in_image(
    pv_csv_path,
    clear_img,
    shadow_mask_vs_t,
    top_left_latlon,
    bottom_right_latlon,
    num_timesteps=96,
    day="2017-01-01"
):

    # Step 1: Load PVs
    pv_locations_df = pd.read_csv(pv_csv_path)

    # Step 2: Filter PVs within bounds
    def is_within_bounds(lat, lon, top_left, bottom_right):
        return top_left[0] >= lat >= bottom_right[0] and \
               top_left[1] <= lon <= bottom_right[1]

    filtered_pvs = pv_locations_df[
        pv_locations_df.apply(lambda row: is_within_bounds(row['latitude'], row['longitude'], top_left_latlon, bottom_right_latlon), axis=1)
    ].copy()

    # Print filtered PVs
    #print("PVs within image bounds:")
    #for _, row in filtered_pvs.iterrows():
        # print(f"PV ID: {row['PV_ID']}, lat: {row['latitude']}, lon: {row['longitude']}")

    # Step 4: Simulate realistic GHI per pixel
    ghi_data = []

    for _, row in filtered_pvs.iterrows():
        lat, lon, pv_id = row['latitude'], row['longitude'], row['PV_ID']
        try:
            px_row, px_col = latlon_to_pixel(lat, lon, top_left_latlon, bottom_right_latlon, clear_img)
            signal = shadow_mask_vs_t[0, :, :, px_row, px_col]  # shape: (96, 3)
            gray_signal = signal[:, 0] * 0.2989 + signal[:, 1] * 0.5870 + signal[:, 2] * 0.1140
            ghi_clear_sky = clear_sky_GHI(lat, lon, day)
            ghi_simulated = torch.tensor(ghi_clear_sky, dtype=gray_signal.dtype) * gray_signal

            # Round to nearest integer
            ghi_simulated = torch.round(ghi_simulated)

            ghi_data.append([pv_id, lat, lon] + ghi_simulated.tolist())
        except Exception as e:
            print(f"⚠️ Skipped PV {pv_id} due to error: {e}")
            continue

    # Step 5: Save results to CSV
    start_time = pd.Timestamp(f"{day} 00:00")
    time_cols = [(start_time + pd.Timedelta(minutes=15 * i)).strftime('%d-%m-%Y %H:%M') for i in range(num_timesteps)]

    return pd.DataFrame(ghi_data, columns=['pv_id', 'latitude', 'longitude'] + time_cols)

####################################
############### MAIN ###############
####################################

# --- Load the clean satellite image and prepare it ---
clear_img = Image.open(r"C:\Users\galrt\Desktop\final_project\clouds_sim\img_jerusalem.png").convert("RGB")
clear_img = np.array(clear_img) / 255.0
clear_img = torch.tensor(clear_img).permute(2, 0, 1).unsqueeze(0).float()
clear_img = clear_img.repeat(2, 1, 1, 1)

# --- Define the geographic boundaries of the image (top-left and bottom-right coordinates) ---
top_left_latlon = (31.85, 35.02)  # פינה שמאלית עליונה
bottom_right_latlon = (31.69, 35.26)  # פינה ימנית תחתונה

# --- Calculate pixel position from given latitude and longitude ---
pixel = latlon_to_pixel(
    lat=31.77,  # הקואורדינטה שאתה רוצה לבדוק
    lon=35.22,  # הקואורדינטה שאתה רוצה לבדוק
    top_left_latlon=top_left_latlon,
    bottom_right_latlon=bottom_right_latlon,
    image=clear_img
)

# --- Calculate cloud motion speed in pixels per hour ---
speed_kmh = 20  # Cloud speed in km/h
km_coverage = 380  # Total geographic coverage of the image in kilometers

all_days_df = None
current_date = datetime(2017, 1, 1)
end_date = datetime(2017, 6, 20)

while current_date <= end_date:
    day_str = current_date.strftime('%Y-%m-%d')  # '2017-01-01'
    print(f"▶️ Processing day: {day_str}")

    # Simulate cloud motion and generate shadows
    vx, vy = calculate_cloud_speed_in_pixels(clear_img, speed_kmh, km_coverage)
    shadow_mask_vs_t = get_cloud_shadow(clear_img, vx=vx, vy=vy, num_of_timesteps=NUM_TIMESTEPS)

    day_df = simulate_ghi_for_pvs_in_image(
        pv_csv_path=r"C:\Users\galrt\Desktop\final_project\clouds_sim\pv_coordinates_2017.csv",
        clear_img=clear_img,
        shadow_mask_vs_t=shadow_mask_vs_t,
        top_left_latlon=top_left_latlon,
        bottom_right_latlon=bottom_right_latlon,
        num_timesteps=NUM_TIMESTEPS,
        day=day_str
    )

    if all_days_df is None:
        all_days_df = day_df
    else:
        all_days_df = pd.merge(all_days_df, day_df, on=['pv_id', 'latitude', 'longitude'], how='outer')

    current_date += timedelta(days=1)

all_days_df.to_csv(r"C:\Users\galrt\Desktop\final_project\clouds_sim\simulated_pv_ghi_all.csv", index=False)
print("✅ Final CSV created: one row per PV with daily GHI columns.")

# --- Create sequence of images affected by the moving cloud shadows ---
clear_img = clear_img.unsqueeze(1)
clear_img = clear_img.repeat(1, NUM_TIMESTEPS, 1, 1, 1)
img = clear_img * (1 - shadow_mask_vs_t)

# --- Extract the pixel signal (light intensity) over time ---
signal_vs_t = shadow_mask_vs_t[0, :, :, pixel[0], pixel[1]]

# --- Convert pixel signal from RGB to grayscale ---
shadow_signal = signal_vs_t[:, 0] * 0.2989 + signal_vs_t[:, 1] * 0.5870 + signal_vs_t[:, 2] * 0.1140

# --- Compute clear sky GHI (Global Horizontal Irradiance) for the same location ---
ghi_clear_sky = clear_sky_GHI(32.09, 34.78)

# --- Combine clear sky GHI with the shadow effect to get realistic GHI ---
realistic_ghi = torch.tensor(ghi_clear_sky) * shadow_signal  # shape: (96,)

# --- Generate time axis labels (for plotting) ---
timezone = 'Asia/Jerusalem'
times = pd.date_range('27-01-2017 00:00', '27-01-2017 23:45', freq='15min', tz=timezone)
time_axis = [t.strftime('%H:%M') for t in times]

# --- Load measured GHI values from real PV data (for comparison) ---
measured_df = pd.read_csv(r"C:\Users\galrt\Desktop\final_project\PV_ghi_all_2017.csv")

# --- Select the measured GHI for the chosen pv_id and date ---
pv_id = 1897629
target_date = '2017-01-27'
time_columns = [col for col in measured_df.columns if target_date in col]
measured_row = measured_df[measured_df['pv_id'] == pv_id]
measured_ghi = measured_row[time_columns].values.flatten().astype(float)

# Plot 5 snapshots showing the cloud shadow effect at different times of the day
plt.figure(figsize=(20, 5))
plt.subplots_adjust(hspace=0.25, wspace=0.5, left=0.05, right=0.95, top=0.925, bottom=0.05)
for id in range(5):
    plt.subplot(1, 5, id + 1)
    plt.imshow(img[0, id].numpy().transpose(1, 2, 0))
    plt.title(f'Time = {id * TIME_STEP_MINUTES} min')
    plt.axis('off')
plt.show()

# Plot GHI comparison: theoretical clear sky, simulated shaded GHI, and measured PV GHI
plt.figure(figsize=(12, 5))
plt.plot(time_axis, ghi_clear_sky, label='Clear Sky GHI', linestyle='--')
plt.plot(time_axis, realistic_ghi, label='Shaded GHI (Pixel)', linewidth=2)
plt.plot(time_axis, measured_ghi, label='Measured GHI (PV 1897629)', linewidth=2, linestyle='-.')
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('GHI [W/m²]')
plt.title(f'GHI Comparison at Pixel {pixel} and PV 1897629')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Show the image with rectangle around the pixel

#plt.imshow(img[0, 0].numpy().transpose(1, 2, 0))
#plt.gca().add_patch(plt.Rectangle((pixel[1] - 0.5, pixel[0] - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none'))
#plt.show()
#plt.plot(signal_vs_t)
#plt.title('Signal vs Time')
#plt.xlabel('Time')
#plt.ylabel('Signal')
#plt.show()

# Plot the raw light intensity signal at the selected pixel over 24 hours
time_axis = [t * TIME_STEP_MINUTES for t in range(NUM_TIMESTEPS)]
plt.plot(time_axis, signal_vs_t)
plt.title(f'Signal at pixel {pixel} over 24 hours')
plt.xlabel('Time [minutes]')
plt.ylabel('Light Intensity')
plt.grid(True)
plt.show()

# Load the image and simulate different clouds
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
titles = ['Clear Image', 'Cloud Image', 'Cloud Mask', 'Shadow Mask']

# Select a clean image once
clear_img_single = clear_img[0, 0]  # (channels, height, width)
clear_img_single = clear_img_single.unsqueeze(0)  # (batch, channels, height, width)

# Generate clouds and shadows once
img, cloud_mask, shadow_mask = scg.add_cloud_and_shadow(clear_img_single, return_cloud=True)

# Prepare images for plotting
clear_img_vis = clear_img_single[0].numpy().transpose(1, 2, 0)
img_vis = img[0].numpy().transpose(1, 2, 0)
cloud_mask_vis = cloud_mask[0].numpy().transpose(1, 2, 0)
shadow_mask_vis = shadow_mask[0].numpy().transpose(1, 2, 0)

# Plot all 4 images
for ax, im, title in zip(axes, [clear_img_vis, img_vis, cloud_mask_vis, shadow_mask_vis], titles):
    ax.imshow(im)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()



# Load the image
for i in range(3):
    clear_img_single = clear_img[0, 0]  # (channels, height, width)
    clear_img_single = clear_img_single.unsqueeze(0)  # (batch, channels, height, width)
    img, cloud_mask, shadow_mask = scg.add_cloud_and_shadow(clear_img_single, return_cloud=True)

    # Add cloud motion:

    # Now add the cloud motion

    clear_img = clear_img[0, 0].numpy().transpose(1, 2, 0)
    img = img[0].numpy().transpose(1, 2, 0)
    cloud_mask = cloud_mask[0].numpy().transpose(1, 2, 0)
    shadow_mask = shadow_mask[0].numpy().transpose(1, 2, 0)

    # Now add only shadow to the image
    only_shadow = clear_img * (1 - shadow_mask)
    plt.imshow(only_shadow)
    plt.show()

# Display the image
plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.imshow(clear_img)
plt.title('Clear Image')
plt.axis('off')
plt.subplot(1, 4, 2)
plt.imshow(img)
plt.title('Cloud Image')
plt.axis('off')
plt.subplot(1, 4, 3)
plt.imshow(cloud_mask)
plt.title('Cloud Mask')
plt.axis('off')
plt.subplot(1, 4, 4)
plt.imshow(shadow_mask)
plt.title('Shadow Mask')
plt.axis('off')
plt.show()

