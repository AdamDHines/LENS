import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

# Define the neural network
snn = nn.Sequential(
    nn.AvgPool2d(kernel_size=(16, 16)),
    # the input of the 1st DynapCNN Core will be (1, 64, 64)
    #nn.Conv2d(1, 1, kernel_size=(8, 8), stride=(8, 8), bias=False), 
    #nn.ReLU()
)

# Function to process a single image
def process_image(image_paths):
    # Load image, assuming gray scale image
    # make a blank [128, 128] tensor
    new_img = torch.zeros(128, 128)
    for image_path in image_paths:
        image = read_image(image_path).float()  # Convert to float to match expected NN input
        new_img += image[0,:,:]/30
    #new_img = new_img.unsqueeze(0)  # Add batch dimension    
    
    # Apply the neural network
    # with torch.no_grad():
    #     new_img = snn(new_img.unsqueeze(0))  # Add batch dimension
    
    # Convert back to PIL Image to save
    output_image = to_pil_image(new_img)  # Remove batch dimension
    return output_image

# Load images from a directory
source_directory1 = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/qcr/speck/030624-qcr-ref'
destination_directory1 = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/qcr/speck/qcr-ref-full'

if not os.path.exists(destination_directory1):
    os.makedirs(destination_directory1)

# Get sorted list of image file paths
image_files = sorted([os.path.join(source_directory1, f) for f in os.listdir(source_directory1) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Process each image and save the result
img_paths = []
count = 0
for idx, img_path in enumerate(tqdm(image_files, desc="Processing Images")):
    if count == 30:
        output_image = process_image(img_paths)
        # Define a new file name for the processed image
        base_name = os.path.basename(img_path)
        save_path = os.path.join(destination_directory1, base_name)
        output_image.save(save_path)
        img_paths = []
        count = 0
    else:
        count += 1
        img_paths.append(img_path)

        # Load images from a directory
# source_directory2 = '/home/adam/Downloads/test002'
# destination_directory2= '/home/adam/Documents/test002'

# if not os.path.exists(destination_directory2):
#     os.makedirs(destination_directory2)

# # Get sorted list of image file paths
# image_files = sorted([os.path.join(source_directory2, f) for f in os.listdir(source_directory2) if f.endswith(('.png', '.jpg', '.jpeg'))])

# # Process each image and save the result
# img_paths = []
# count = 0
# for idx, img_path in enumerate(tqdm(image_files, desc="Processing Images")):
#     if count == 30:
#         output_image = process_image(img_paths)
#         # Define a new file name for the processed image
#         base_name = os.path.basename(img_path)
#         save_path = os.path.join(destination_directory2, base_name)
#         output_image.save(save_path)
#         img_paths = []
#         count = 0
#     else:
#         count += 1
#         img_paths.append(img_path)