import os
import numpy as np
from skimage import io, color
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import re
from prettytable import PrettyTable
from metrics import recallAtK
import torch

def natural_sort_key(s):
    """
    A key function for natural (human-like) sorting of strings with numbers.
    It converts the string into a list of strings and integers.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_and_preprocess_images(folder_path, skip_factor=1, max_images=1500):
    images = []
    files = sorted(os.listdir(folder_path), key=natural_sort_key)  # Sort the files
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

def load_and_preprocess_images_v2(folder_path, variable_pixels, skip_factor=0, max_images=1000):
    images = []
    files = sorted(os.listdir(folder_path),key=natural_sort_key)  # Sort the files
    for idx, filename in enumerate(files):
        if idx >= max_images:
            break
        #if idx % skip_factor != 0:  # Skip images based on the skip factor
        #    continue
        if filename.endswith('.png'):
            img = io.imread(os.path.join(folder_path, filename))
            if len(img.shape) > 2:  # Convert to grayscale if necessary
                img = color.rgb2gray(img)
            # Select only the top variable pixels
            #img = processImage(img, 56, 56, 15)
            images.append(img.flatten()) 
    return np.array(images)

def sum_of_absolute_differences(image1, image2):
    return np.sum(np.abs(image1 - image2))

# Load and preprocess images from both folders
folder1 = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/brisbane_event/davis/sunset1_49'
folder2 = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/brisbane_event/davis/sunset2_49'

# First, load and preprocess all images from folder1 without skipping
all_images1 = load_and_preprocess_images(folder1, skip_factor=1)

# Find the top 100 variable pixels
top_pixels = find_top_variable_pixels(all_images1)

# Now reload the images with the variable pixels
images1 = load_and_preprocess_images_v2(folder1, top_pixels)
images2 = load_and_preprocess_images_v2(folder2, top_pixels)

# distances = np.linalg.norm(images1[:, np.newaxis, :] - images2[np.newaxis, :, :], ord=1, axis=2)
# plt.imshow(np.reshape(distances,(7,7)))
# plt.show()
# sad_matrix = np.zeros((len(images1), len(images2)))

# for i in range(len(images1)):
#    for j in range(len(images2)):
#        sad_matrix[i, j] = sum_of_absolute_differences(images1[i], images2[j])

# #sad_matrix = 1/sad_matrix

# plt.figure(figsize=(10, 8))
# sns.heatmap(sad_matrix, annot=False, cmap='coolwarm')
# plt.title('Sum of Absolute Differences between Images from Two Folders')
# plt.xlabel('Images from Folder 2')
# plt.ylabel('Images from Folder 1')
# plt.show()


# Recall@N
N = [1,5,10,15,20,25] # N values to calculate
#R = [] # Recall@N values
# Create GT matrix
GT = np.zeros((95,245), dtype=int)
#for n in range(len(GT)):
#    GT[n,n] = 1
# Calculate Recall@N
#for n in N:
#    R.append(round(recallAtK(sad_matrix,GThard=GT,K=n),2))
# Print the results
#table = PrettyTable()
#table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
#table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])
#print(table)
# Compute cosine similarity with the modified images
similarity_matrix = cosine_similarity(images1, images2)

new_matrix = np.zeros_like(similarity_matrix)

# Iterate through each column and set 1 at the position of the maximum value
for col_idx in range(similarity_matrix.shape[1]):
    max_idx = np.argmax(similarity_matrix[:, col_idx])
    new_matrix[max_idx, col_idx] = 1
np.save('/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/brisbane_event/gt.npy', new_matrix)
# Plotting the similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm')
plt.title('Cosine Similarity (Top Variable Pixels) between Images from Two Folders')
plt.xlabel('Images from Folder 2')
plt.ylabel('Images from Folder 1')
plt.show()

R = [] # Recall@N values
# Calculate Recall@N
for n in N:
    R.append(round(recallAtK(similarity_matrix,GThard=GT,K=n),2))
# Print the results
table = PrettyTable()
table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])
print(table)