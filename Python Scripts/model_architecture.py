import numpy as np
import tifffile as tiff
from skimage.transform import resize
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
import os
import shutil
import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
import os
import shutil
import tifffile as tiff
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
import os
import shutil
import tifffile as tiff


def convert_to_db(image):
    with np.errstate(divide='ignore', invalid='ignore'):
        db_image = 10 * np.log10(image)
    db_image[np.isinf(db_image)] = np.nan
    return db_image

def generate_mask_lib(db_image):
    threshold_ice_free = [-7,-2] 
    threshold_ice_bergs = [-1.5,-1]  
    threshold_multiyearice = [-1, 3.5]  
    threshold_firstyearice = [-2, -1.5]  


    mask_ice_free = (db_image >= threshold_ice_free[0]) & (db_image <= threshold_ice_free[1])
    mask_ice_bergs = (db_image >= threshold_ice_bergs[0]) & (db_image <= threshold_ice_bergs[1])
    mask_multiyearice = (db_image >= threshold_multiyearice[0]) & (db_image <= threshold_multiyearice[1])
    mask_firstyearice = (db_image >= threshold_firstyearice[0]) & (db_image <= threshold_firstyearice[1])
    #mask_icetype4 = (db_image >= threshold_icetype4[0]) & (db_image <= threshold_icetype4[1])
    #mask_newice = (db_image >= threshold_newice[0]) & (db_image <= threshold_newice[1])

    mask_lib = {
        'Ice Free': mask_ice_free.astype(int),
        'Ice bergs': mask_ice_bergs.astype(int),
        'Multi-Year Ice': mask_multiyearice.astype(int),
        'First-Year Ice': mask_firstyearice.astype(int),
        #'Ice Type 4': mask_icetype4.astype(int),
        #'New Ice': mask_newice.astype(int)
    }

    return mask_lib

folder_path = "/kaggle/working/norm_data"
folder_path2 = "/kaggle/input/images-mask20"
output_folder = "/kaggle/working/output"

total_percentages = {ice_type: 0 for ice_type in ['Ice Free', 'Ice Covered', 'Multi-Year Ice', 'First-Year Ice']}
label_to_int = {'Ice Free': 0, 'Ice bergs': 16, 'Multi-Year Ice': 64, 'First-Year Ice': 128}

num_images = 0
y = []
x = []
z = []

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for image_file in os.listdir(folder_path):
    if image_file.endswith(".tif"):
        image_path = os.path.join(folder_path, image_file)
        mask_path = os.path.join(folder_path2, image_file)

        image = tiff.imread(image_path)
        image = resize(image, (256, 256))
        db_image = convert_to_db(image)
        
        print(f"Image: {image_file}")
        print("Max dB value:", np.nanmax(db_image))
        print("Min dB value:", np.nanmin(db_image))

        mask_lib = generate_mask_lib(db_image)

        for ice_type, mask_image in mask_lib.items():
            output_path = os.path.join(output_folder, image_file.replace(".tif", f"_{ice_type}_mask.png"))
            mask_image = (mask_image * 255).astype(np.uint8)
            tiff.imsave(output_path, mask_image)

            plt.imshow(mask_image, cmap='gray')
            plt.axis('off')
            plt.title(ice_type)
            plt.show()
def convert_to_db(image):
    with np.errstate(divide='ignore', invalid='ignore'):
        db_image = 10 * np.log10(image)
    return db_image

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

folder_path = "/kaggle/working/norm_data" 

mask_output_folder = "/kaggle/working/combined-mask" 
if os.path.exists(mask_output_folder):
    shutil.rmtree(mask_output_folder)
os.makedirs(mask_output_folder)

x = []
z = []
y = []

colors = ['black', 'lightgray', 'white', 'white']

for image_file in os.listdir(folder_path):
    if image_file.endswith(".tif"):
        image_path = os.path.join(folder_path, image_file)

        image = tiff.imread(image_path)
        image = resize(image, (256, 256))
        db_image = convert_to_db(image)

        db_image[np.isnan(db_image)] = np.nanmin(db_image)
        finite_max = np.nanmax(db_image[np.isfinite(db_image)])
        db_image[np.isinf(db_image)] = finite_max + 1

        mask_lib = generate_mask_lib(db_image)

        color_mask = np.zeros((db_image.shape[0], db_image.shape[1]), dtype=np.uint8)

        color_mask[mask_lib['Ice Free'] == 1] = 0     
        color_mask[mask_lib['Ice bergs'] == 1] = 1    
        color_mask[mask_lib['Multi-Year Ice'] == 1] = 2   
        color_mask[mask_lib['First-Year Ice'] == 1] = 3  

        mask_image_name = image_file.replace(".tif", "_color_mask.tif")
        mask_output_path = os.path.join(mask_output_folder, mask_image_name)
        tiff.imsave(mask_output_path, color_mask)

        x.append(image)
        z.append(color_mask)
        y.append(mask_lib)

        plt.imshow(color_mask, cmap='gray')
        plt.colorbar(label='Ice Types')  
        plt.axis('off')
        plt.title("Mask for Image: " + image_file)

        plt.imshow(color_mask, cmap='gray')
        cbar = plt.colorbar(label='Ice Types', ticks=[0, 1, 2, 3], format='%d', orientation='vertical', pad=0.02, aspect=40)
        cbar.set_ticklabels(list(mask_lib.keys())) 
        plt.axis('off')
        plt.title("Mask for Image: " + image_file)

        plt.show()

x = np.array(x)
z = np.array(z)
y = np.array(y)

print("Shape of x (images):", x.shape)
print("Shape of z (masks):", z.shape)
print("Shape of y (labels):", y.shape)


def convert_to_db(image):
    with np.errstate(divide='ignore', invalid='ignore'):
        db_image = 10 * np.log10(image)
    return db_image

def generate_mask_lib(db_image):
    threshold_ice_free = [-np.inf,-15]   
    threshold_ice_bergs = [0,5]  
    threshold_multiyearice = [5, np.inf] 
    threshold_firstyearice = [-15, 0]  

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

folder_path = "/kaggle/working/input" 

mask_output_folder = "/kaggle/working/combined-mask" 
if os.path.exists(mask_output_folder):
    shutil.rmtree(mask_output_folder)
os.makedirs(mask_output_folder)

x = []
z = []
y = []

for image_file in os.listdir(folder_path):
    if image_file.endswith(".tif"):
        image_path = os.path.join(folder_path, image_file)

        image = tiff.imread(image_path)
        image = resize(image, (256, 256))
        db_image = convert_to_db(image)

        db_image[np.isnan(db_image)] = np.nanmin(db_image)
        finite_max = np.nanmax(db_image[np.isfinite(db_image)])
        db_image[np.isinf(db_image)] = finite_max + 1

        mask_lib = generate_mask_lib(db_image)

        color_mask = np.zeros((db_image.shape[0], db_image.shape[1]), dtype=np.uint8)

        color_mask[mask_lib['Ice Free'] == 1] = 0     
        color_mask[mask_lib['Ice bergs'] == 1] = 2     
        color_mask[mask_lib['Multi-Year Ice'] == 1] = 3  
        color_mask[mask_lib['First-Year Ice'] == 1] = 1   

        mask_image_name = image_file.replace(".tif", "_color_mask.tif")
        mask_output_path = os.path.join(mask_output_folder, mask_image_name) 
        tiff.imsave(mask_output_path, color_mask)

        x.append(image)
        z.append(color_mask)
        y.append(mask_lib)

        plt.imshow(color_mask, cmap='gray')
        plt.axis('off')
        plt.title("Mask for Image: " + image_file)
        plt.show()

x = np.array(x)
z = np.array(z)
y = np.array(y)

print("Shape of x (images):", x.shape)
print("Shape of z (masks):", z.shape)
print("Shape of y (labels):", y.shape)


unique_labels = np.unique(z)
print("Unique labels in z:", unique_labels)
num_classes = 4  
if np.issubdtype(z.dtype, np.integer):
    min_label = np.min(z)
    max_label = np.max(z)
    if min_label >= 0 and max_label <= (num_classes - 1):
        print("Labels are integers and within the correct range (0 to", num_classes - 1, ")")
    else:
        print("Labels are integers, but they are not within the correct range.")
        print("Minimum label:", min_label, ", Maximum label:", max_label)
else:
    print("Labels are not integers.")

z_train_int = np.argmax(z, axis=-1)

unique_labels_int = np.unique(z)
num_classes = 4  

if np.issubdtype(z.dtype, np.integer):
    min_label = np.min(z)
    max_label = np.max(z)
    if min_label >= 0 and max_label <= (num_classes - 1):
        print("Labels are integers and within the correct range (0 to", num_classes - 1, ")")
    else:
        print("Labels are integers, but they are not within the correct range.")
        print("Minimum label:", min_label, ", Maximum label:", max_label)
else:
    print("Labels are not integers.")

ice_counts = {ice_type: 0 for ice_type in ['Ice Free', 'Ice bergs', 'Multi-Year Ice', 'First-Year Ice']}

total_images = len(y)

for label in y:
    for ice_type, mask in label.items():
        ice_counts[ice_type] += np.sum(mask)

total_pixels = 256 * 256 * total_images
ice_percentages = {ice_type: (count / total_pixels) * 100 for ice_type, count in ice_counts.items()}

ice_types = list(ice_percentages.keys())
percentages = list(ice_percentages.values())

plt.bar(ice_types, percentages)
plt.xlabel("Ice Types")
plt.ylabel("Percentage of Pixels")
plt.title("Ice Type Distribution")
plt.xticks(rotation=45)
plt.show()

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

augm_img = '/kaggle/working/augm_img'
augm_mask = '/kaggle/working/augm_mask'

if os.path.exists(augm_img):
    shutil.rmtree(augm_img)
os.makedirs(augm_img)
if os.path.exists(augm_mask):
    shutil.rmtree(augm_mask)
os.makedirs(augm_mask)
print('Performing data augmentation')
num_augmented_samples = 10 

def augment_mask(mask):
    mask[mask == 0] = 0   
    mask[mask == 1] = 1 
    mask[mask == 2] = 2  
    return mask.astype(np.uint8)

augmented_images = []
augmented_masks = []

for i in range(len(x)):
    img = x[i]
    mask = z[i]

    img = np.reshape(img, (*img.shape, 1))
    mask = np.reshape(mask, (*mask.shape, 1))

    img_augmented_gen = datagen.flow(np.expand_dims(img, axis=0), batch_size=1, shuffle=False)
    mask_augmented_gen = datagen.flow(np.expand_dims(mask, axis=0), batch_size=1, shuffle=False)

    for j in range(num_augmented_samples):
        img_augmented = img_augmented_gen.next()[0]
        mask_augmented = mask_augmented_gen.next()[0]
        mask_augmented = augment_mask(mask_augmented)
        img_save_path = os.path.join(augm_img, f'image_{i}_{j}.tif')
        mask_save_path = os.path.join(augm_mask, f'image_{i}_{j}-mask.tif')

        tiff.imwrite(img_save_path, img_augmented)
        tiff.imwrite(mask_save_path, mask_augmented)

        augmented_images.append(img_augmented)
        augmented_masks.append(mask_augmented)

augmented_images = np.array(augmented_images)
augmented_masks = np.array(augmented_masks)

print('Augmented images shape:', augmented_images.shape)
print('Augmented masks shape:', augmented_masks.shape)
print('Augmented images saved in:', augm_img)
print('Augmented masks saved in:', augm_mask)

folder_path = "/kaggle/working/augm_img"

file_list = os.listdir(folder_path)

num_images = len(file_list)
num_rows = int(np.ceil(np.sqrt(num_images)))
num_cols = int(np.ceil(num_images / num_rows))

fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i, file_name in enumerate(file_list):
    if file_name.endswith('.tif'):
        file_path = os.path.join(folder_path, file_name)
        
        image = tiff.imread(file_path)
        
        row_idx = i // num_cols
        col_idx = i % num_cols
        axes[row_idx, col_idx].imshow(image, cmap='gray')
        axes[row_idx, col_idx].axis('off')

plt.tight_layout()

# Show the plot
plt.show()
