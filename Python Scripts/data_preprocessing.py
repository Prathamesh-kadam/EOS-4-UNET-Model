import os
import cv2
import rasterio
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import tifffile as tiff
from rasterio import warp
from tifffile import imread
from rasterio.enums import Resampling
from matplotlib.colors import ListedColormap
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator

def scale_data(data):
    min_value, max_value = np.percentile(data, [2, 98])
    scaled_data = (data - min_value) / (max_value - min_value) * 255
    scaled_data = np.clip(scaled_data, 0, 255)
    return scaled_data.astype(np.uint8)

def convert_to_db_and_stretch(data, lower_threshold, upper_threshold):
    data_db = 10 * np.log10(data)
    data_db = (data_db - lower_threshold) / (upper_threshold - lower_threshold) * 255
    data_db = np.clip(data_db, 0, 255)
    return data_db.astype(np.uint8)

def generate_mask_lib(db_image):
    threshold_ice_free = [-7, -2] 
    threshold_ice_bergs = [-2, 3.5] 
    threshold_multiyearice = [-1, 3.5]  
    threshold_firstyearice = [-2, -1]  

    mask_ice_free = (db_image >= threshold_ice_free[0]) & (db_image <= threshold_ice_free[1])
    mask_ice_bergs = (db_image >= threshold_ice_bergs[0]) & (db_image <= threshold_ice_bergs[1])
    mask_multiyearice = (db_image >= threshold_multiyearice[0]) & (db_image <= threshold_multiyearice[1])
    mask_firstyearice = (db_image >= threshold_firstyearice[0]) & (db_image <= threshold_firstyearice[1])

    mask_lib = {
        'Ice Free': mask_ice_free.astype(int),
        'Ice bergs': mask_ice_bergs.astype(int),
        'Multi-Year Ice': mask_multiyearice.astype(int),
        'First-Year Ice': mask_firstyearice.astype(int),
    }

    return mask_lib


def resize_hv_data(hv_data, hv_transform, hh_transform, hh_shape):
    hv_data_resized = np.empty(hh_shape, dtype=hv_data.dtype)
    warp.reproject(
        source=hv_data,
        destination=hv_data_resized,
        src_transform=hv_transform,
        src_crs=hv_ds.crs,
        dst_transform=hh_transform,
        dst_crs=hh_ds.crs,
        resampling=Resampling.nearest
    )
    return hv_data_resized

hh_folder = '/kaggle/input/eos4-hh-data'
hv_folder = '/kaggle/input/eos4-hv-data'
output_folder = '/kaggle/working/input'
os.makedirs(output_folder, exist_ok=True)

hh_files = os.listdir(hh_folder)
hv_files = os.listdir(hv_folder)

for hh_file, hv_file in zip(hh_files, hv_files):
    hh_tif_file = os.path.join(hh_folder, hh_file)
    hv_tif_file = os.path.join(hv_folder, hv_file)

    with rasterio.open(hh_tif_file) as hh_ds:
        hh_data = hh_ds.read(1)

    with rasterio.open(hv_tif_file) as hv_ds:
        hv_data = hv_ds.read(1)
        hv_transform = hv_ds.transform

        hh_transform = hh_ds.transform
        hv_data_resized = resize_hv_data(hv_data, hv_transform, hh_transform, hh_data.shape)

    hv_data_resized[hv_data_resized <= 0] = 1e-9

    hh_hv_ratio = hh_data / hv_data_resized
    hh_hv_diff = hh_data - hv_data_resized

    hh_hv_ratio_db = convert_to_db_and_stretch(hh_hv_ratio, -7, 3.5)
    hh_hv_diff_db = convert_to_db_and_stretch(hh_hv_diff, -7, 3.5)
    hv = convert_to_db_and_stretch(hv_data_resized, -7, 3.5)

    gray_composite = 0.33*hv + 0.33 * hh_hv_ratio_db + 0.33 * hh_hv_diff_db 
    output_tif_file = os.path.join('/kaggle/working/input', hh_file.replace('.tif', '_gray_composite.tif'))

    with rasterio.open(output_tif_file, 'w', driver='GTiff', width=hh_ds.width, height=hh_ds.height, count=1, dtype=gray_composite.dtype) as dst:
        dst.write(gray_composite, 1)

    print(f"Gray-scale composite image saved as {output_tif_file}")


img_dir = '/kaggle/working/input'

img_filenames = os.listdir(img_dir)
img_names = [s.split('.')[0] for s in img_filenames if s.endswith('.tif')]

img_ext = '.tif'


folder_path = "/kaggle/input/eos4-hv-data"

file_list = os.listdir(folder_path)

num_images = len(file_list)
num_rows = int(np.ceil(np.sqrt(num_images)))
num_cols = int(np.ceil(num_images / num_rows))

fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i, file_name in enumerate(file_list):
    if file_name.lower().endswith('.tif') or file_name.lower().endswith('.tif'):
        file_path = os.path.join(folder_path, file_name)
        
        image = tiff.imread(file_path)
        
        row_idx = i // num_cols
        col_idx = i % num_cols
        
        axes[row_idx, col_idx].imshow(image, cmap='gray')
        axes[row_idx, col_idx].axis('off')
plt.tight_layout()
plt.show()

def normalize_backscattering(data):
    min_value = np.min(data)
    max_value = np.max(data)
    normalized_data = 255 * (data - min_value) / (max_value - min_value)
    normalized_data = normalized_data.astype(np.uint8)
    return normalized_data

folder_path = '/kaggle/input/eos4-hv-data' 

norm_folder = '/kaggle/working/norm_data'  

if not os.path.exists(folder_path):
    print(f"Folder not found: {folder_path}")
else:
    if not os.path.exists(norm_folder):
        os.makedirs(norm_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.tiff') or filename.endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            backscattering_data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            normalized_data = normalize_backscattering(backscattering_data)

            min_value = np.min(normalized_data)
            max_value = np.max(normalized_data)
            print(f"File: {filename}")
            print("Minimum value:", min_value)
            print("Maximum value:", max_value)
            print("\n")

            new_filename = "normalized_" + filename
            new_file_path = os.path.join(norm_folder, new_filename)
            cv2.imwrite(new_file_path, normalized_data)

print("Normalization and saving complete!")


