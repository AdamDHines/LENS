import os
import numpy as np
from skimage import io, color
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import re
from prettytable import PrettyTable
from metrics import recallAtK, createPR
import torch
import json
import math

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
            # gamma correction
            mid = 0.5
            mean = np.mean(img)

            # Check if mean is zero or negative to avoid math domain error
            try:
                gamma = math.log(mid * 255) / math.log(mean)
                img = math.pow(img, gamma).clip(0, 255)
            except:
                pass
            images.append(img.flatten()) 
    return np.array(images)

def sum_of_absolute_differences(image1, image2):
    return np.sum(np.abs(image1 - image2))

# Load and preprocess images from both folders
folder1 = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/brisbane_event/davis/sunset2'
folder2 = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/brisbane_event/davis/daytime'

# First, load and preprocess all images from folder1 without skipping
all_images1 = load_and_preprocess_images(folder1, skip_factor=1)

# Find the top 100 variable pixels
top_pixels = find_top_variable_pixels(all_images1)

# Now reload the images with the variable pixels
images1 = load_and_preprocess_images_v2(folder1, top_pixels)
images2 = load_and_preprocess_images_v2(folder2, top_pixels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = torch.from_numpy(images1.reshape(images1.shape[0], -1).astype(np.float32)).unsqueeze(0).to(device)
b = torch.from_numpy(images2.reshape(images2.shape[0], -1).astype(np.float32)).unsqueeze(0).to(device)
torch_dist = torch.cdist(a, b, 1)[0]

# Recall@N
N = [1,5,10,15,20,25] # N values to calculate
seq_length = 10
# Create GT matrix
GT = np.load('/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/brisbane_event/davis/sunset2_daytime_GT.npy')
if seq_length != 0:
    GT = GT[seq_length-2:-1,seq_length-2:-1]
GT=GT.T
# Compute cosine similarity with the modified images
similarity_matrix = cosine_similarity(images1, images2)

if seq_length != 0:
    precomputed_convWeight = torch.eye(seq_length, device=device).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    dist_matrix_seq = torch.nn.functional.conv2d(torch_dist.unsqueeze(0).unsqueeze(0), precomputed_convWeight).squeeze().cpu().numpy() / seq_length
else:
    dist_matrix_seq = torch_dist.cpu().numpy()
plt.figure(figsize=(10, 8))
sns.heatmap(1/dist_matrix_seq, annot=False, cmap='coolwarm')
plt.title('Similarity matrix')
plt.xlabel("Query")
plt.ylabel("Database")
plt.show()

R = [] # Recall@N values
# Calculate Recall@N
for n in N:
    R.append(round(recallAtK(1/dist_matrix_seq,GThard=GT,K=n),2))

x = [1,5,10,15,20,25]
AUC= np.trapz(R, x)
# Print the results
table = PrettyTable()
table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])
print(table)
print(AUC)
# Create PR curve
P, R = createPR(1/dist_matrix_seq, GThard=GT, GTsoft=GT, matching='multi', n_thresh=100)
# Combine P and R into a list of lists
PR_data = {
        "Precision": P,
        "Recall": R
    }
output_file = "PR_curve_data.json"
# Construct the full path
full_path = f"{'/home/adam/Documents'}/{output_file}"
# Write the data to a JSON file
with open(full_path, 'w') as file:
    json.dump(PR_data, file) 
# Plot PR curve
plt.plot(R,P)    
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()