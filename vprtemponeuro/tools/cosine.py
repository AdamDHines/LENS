import os
import numpy as np
from skimage import io, color
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def load_and_preprocess_images(folder_path, skip_factor=5, max_images=1500):
    images = []
    files = sorted(os.listdir(folder_path))  # Sort the files
    for idx, filename in enumerate(files):
        if idx >= max_images:
            break
        if idx % skip_factor != 0:  # Skip images based on the skip factor
            continue
        if filename.endswith('.png'):
            img = io.imread(os.path.join(folder_path, filename))
            if len(img.shape) > 2:  # Convert to grayscale if necessary
                img = color.rgb2gray(img)
            images.append(img.flatten())  # Flatten the image
    return np.array(images)

def find_top_variable_pixels(images, top_n=784):
    pixel_variability = np.zeros(images[0].shape)  # Assuming all images are the same shape

    for img in images:
        # Count how often each pixel is at the extreme values (0 or 255)
        pixel_variability += (img == 0) | (img == 255)

    # Flatten and sort by variability, then pick the top N indices
    flat_variability = pixel_variability.flatten()
    top_indices = np.argsort(flat_variability)[-top_n:]

    return top_indices

def get_patches2D(image, patch_size):

    if patch_size[0] % 2 == 0: 
        nrows = image.shape[0] - patch_size[0] + 2
        ncols = image.shape[1] - patch_size[1] + 2
    else:
        nrows = image.shape[0] - patch_size[0] + 1
        ncols = image.shape[1] - patch_size[1] + 1
    return np.lib.stride_tricks.as_strided(image, patch_size + (nrows, ncols), image.strides + image.strides).reshape(patch_size[0]*patch_size[1],-1)


def patch_normalise_pad(image, patch_size):

    patch_size = (patch_size, patch_size)
    patch_half_size = [int((p-1)/2) for p in patch_size ]

    image_pad = np.pad(np.float64(image), patch_half_size, 'constant', constant_values=np.nan)

    nrows = image.shape[0]
    ncols = image.shape[1]
    patches = get_patches2D(image_pad, patch_size)
    mus = np.nanmean(patches, 0)
    stds = np.nanstd(patches, 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        out = (image - mus.reshape(nrows, ncols)) / stds.reshape(nrows, ncols)

    out[np.isnan(out)] = 0.0
    out[out < -1.0] = -1.0
    out[out > 1.0] = 1.0
    return out


def processImage(img, imWidth, imHeight, num_patches):

    img = cv2.resize(img,(imWidth, imHeight))
    #img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    im_norm = patch_normalise_pad(img, num_patches) 

    # Scale element values to be between 0 - 255
    img = np.uint8(255.0 * (1 + im_norm) / 2.0)

    return img

def load_and_preprocess_images_v2(folder_path, variable_pixels, skip_factor=5, max_images=5000):
    images = []
    files = sorted(os.listdir(folder_path))  # Sort the files
    for idx, filename in enumerate(files):
        if idx >= max_images:
            break
        if idx % skip_factor != 0:  # Skip images based on the skip factor
            continue
        if filename.endswith('.png'):
            img = io.imread(os.path.join(folder_path, filename))
            if len(img.shape) > 2:  # Convert to grayscale if necessary
                img = color.rgb2gray(img)
            # Select only the top variable pixels
            img = processImage(img, 56, 56, 15)
            images.append(img.flatten()) 
    return np.array(images)

# Load and preprocess images from both folders
folder1 = '/home/adam/repo/rpg_e2vid/scripts/extracted_data/gps_matched_reference'
folder2 = '/home/adam/repo/rpg_e2vid/scripts/extracted_data/gps_matched_query'

# First, load and preprocess all images from folder1 without skipping
all_images1 = load_and_preprocess_images(folder1, skip_factor=1)

# Find the top 100 variable pixels
top_pixels = find_top_variable_pixels(all_images1)

# Now reload the images with the variable pixels
images1 = load_and_preprocess_images_v2(folder1, top_pixels)
images2 = load_and_preprocess_images_v2(folder2, top_pixels)

# Compute cosine similarity with the modified images
similarity_matrix = cosine_similarity(images1, images2)

# Plotting the similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm')
plt.title('Cosine Similarity (Top Variable Pixels) between Images from Two Folders')
plt.xlabel('Images from Folder 2')
plt.ylabel('Images from Folder 1')
plt.show()