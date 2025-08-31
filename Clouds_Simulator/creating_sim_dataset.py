import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import torch
import SatelliteCloudGenerator as scg
import matplotlib.pyplot as plt

NUM_TIMESTEPS = 96
TIME_STEP_MINUTES = 15

def resize_image_to_km_resolution(input_path, pixel_size_km_y=4, pixel_size_km_x=4):
    image = Image.open(input_path)
    image_np = np.array(image)
    scale_y = int(pixel_size_km_y / 1)
    scale_x = int(pixel_size_km_x / 1)

    if image_np.ndim == 3:
        upsampled = zoom(image_np, zoom=(scale_y, scale_x, 1), order=0)
    elif image_np.ndim == 2:
        upsampled = zoom(image_np, zoom=(scale_y, scale_x), order=0)
    else:
        raise ValueError("Unsupported image dimensions")

    return upsampled

def calculate_cloud_speed_in_pixels(clear_img, speed_kmh, km_coverage):
    image_height, image_width = clear_img.shape[2], clear_img.shape[3]
    km_per_pixel_x = km_coverage / image_width
    km_per_pixel_y = km_coverage / image_height
    vx = (speed_kmh / km_per_pixel_x)
    vy = (speed_kmh / km_per_pixel_y)
    return int(vx), int(vy)

def get_cloud_shadow(clear_img, vx, vy, num_of_timesteps=10):
    __, __, shadow_mask_tot = scg.add_cloud_and_shadow(clear_img, return_cloud=True)
    bs = clear_img.shape[0]
    vx = torch.randn(bs) * vx
    vy = torch.randn(bs) * vy
    shadow_mask_vs_batch = []
    for b in range(bs):
        shadow_mask_vs_t = []
        for t in range(num_of_timesteps):
            shadow_mask = shadow_mask_tot[b]
            shadow_mask = torch.roll(shadow_mask, shifts=(int(vy[b]), int(vx[b])), dims=(1, 2))
            shadow_mask_vs_t.append(shadow_mask)
            shadow_mask_tot[b] = shadow_mask
        shadow_mask_vs_t = torch.stack(shadow_mask_vs_t)
        shadow_mask_vs_batch.append(shadow_mask_vs_t)
    shadow_mask_vs_batch = torch.stack(shadow_mask_vs_batch)
    return shadow_mask_vs_batch

# Main execution
input_path = r"C:\Users\galrt\Desktop\final_project\clouds_sim\clean_img.png"
resized_image = resize_image_to_km_resolution(input_path)
Image.fromarray(resized_image.astype(np.uint8)).save("upsampled_correct_resolution.png")

clear_img = torch.tensor(resized_image / 255.0).permute(2, 0, 1).unsqueeze(0).float()
clear_img = clear_img.repeat(2, 1, 1, 1)

speed_kmh = 20
km_coverage = 48047
vx, vy = calculate_cloud_speed_in_pixels(clear_img, speed_kmh, km_coverage)

shadow_mask_vs_t = get_cloud_shadow(clear_img, vx=vx, vy=vy, num_of_timesteps=NUM_TIMESTEPS)
clear_img = clear_img.unsqueeze(1).repeat(1, NUM_TIMESTEPS, 1, 1, 1)
img = clear_img * (1 - shadow_mask_vs_t)

plt.figure(figsize=(20, 5))
plt.subplots_adjust(hspace=0.25, wspace=0.5, left=0.05, right=0.95, top=0.925, bottom=0.05)
for id in range(5):
    plt.subplot(1, 5, id + 1)
    plt.imshow(img[0, id].numpy().transpose(1, 2, 0))
    plt.title(f'Time = {id * TIME_STEP_MINUTES} min')
    plt.axis('off')
plt.show()

pixel = (300, 300)
signal_vs_t = shadow_mask_vs_t[0, :, :, pixel[0], pixel[1]]
signal_vs_t = signal_vs_t[:, 0] * 0.2989 + signal_vs_t[:, 1] * 0.5870 + signal_vs_t[:, 2] * 0.1140

time_axis = [t * TIME_STEP_MINUTES for t in range(NUM_TIMESTEPS)]
plt.plot(time_axis, signal_vs_t)
plt.title(f'Signal at pixel {pixel} over 24 hours')
plt.xlabel('Time [minutes]')
plt.ylabel('Light Intensity')
plt.grid(True)
plt.show()
