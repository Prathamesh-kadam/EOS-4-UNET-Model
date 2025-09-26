from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import train_test_split
import shutil
import os
import tifffile as tiff
import os
import tensorflow as tf
import tifffile as tiff
import numpy as np

data = list(zip(augmented_images, augmented_masks))

random.shuffle(data)

shuffled_images, shuffled_masks = zip(*data)

x_train, x_test, z_train, z_test = train_test_split(shuffled_images, shuffled_masks, test_size=0.2, random_state=0)

x_train = np.array(x_train)
x_test = np.array(x_test)
z_train = np.array(z_train)
z_test = np.array(z_test)

print("Training data - Input:", x_train.shape)
print("Training data - Mask:", z_train.shape)
print("Test data - Input:", x_test.shape)
print("Test data - Mask:", z_test.shape)


x_train_dest = "/kaggle/working/x_train"
x_test_dest = "/kaggle/working/x_test"
z_train_dest = "/kaggle/working/z_train"
z_test_dest = "/kaggle/working/z_test"

for directory in [x_train_dest, x_test_dest, z_train_dest, z_test_dest]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

for i, x_train_data in enumerate(x_train):
    file_name = f"x_train_{i}.tif"  
    x_train_path = os.path.join(x_train_dest, file_name)
    
    tiff.imsave(x_train_path, x_train_data)

for i, x_test_data in enumerate(x_test):
    file_name = f"x_test_{i}.tif" 
    x_test_path = os.path.join(x_test_dest, file_name)

    tiff.imsave(x_test_path, x_test_data)

for i, z_train_data in enumerate(z_train):
    file_name = f"z_train_{i}.tif" 
    z_train_path = os.path.join(z_train_dest, file_name)

    tiff.imsave(z_train_path, z_train_data)

for i, z_test_data in enumerate(z_test):
    file_name = f"z_test_{i}.tif" 
    z_test_path = os.path.join(z_test_dest, file_name)

    tiff.imsave(z_test_path, z_test_data)


x_train_dest = "/kaggle/working/x_train"
x_test_dest = "/kaggle/working/x_test"
z_train_dest = "/kaggle/working/z_train"
z_test_dest = "/kaggle/working/z_test"

def read_image(image_name):
    def _read_image(image_name):
        image_name_str = image_name.numpy().decode('utf-8')  
        image = tiff.imread(os.path.join(x_train_dest, image_name_str))
        image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0

        mask = tiff.imread(os.path.join(z_train_dest, f"{image_name_str.split('.')[0]}-mask.tif"))
        mask = tf.convert_to_tensor(mask, dtype=tf.uint8)
        return image, mask

    return tf.py_function(_read_image, [image_name], [tf.float32, tf.uint8])

x_train_list = np.array([file for file in os.listdir(x_train_dest) if file.endswith(".tif")])
x_test_list = np.array([file for file in os.listdir(x_test_dest) if file.endswith(".tif")])

TRAIN_LENGTH = len(x_train_list)
VAL_LENGTH = len(x_test_list)
BATCH_SIZE = 4
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

ds_train = tf.data.Dataset.from_tensor_slices(x_train_list)
ds_train = ds_train.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = ds_train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices(x_test_list)
ds_val = ds_val.map(read_image)
val_dataset = ds_val.batch(BATCH_SIZE)

