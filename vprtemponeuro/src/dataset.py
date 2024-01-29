import os
import math
import cv2
import torch

import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.quantization as tq
import matplotlib.pyplot as plt

from torchvision.io import read_image
from torch.utils.data import Dataset

class GenerateTemporalCode:
    def __init__(self, img, repeat):
        super(GenerateTemporalCode, self).__init__()
        self.img = img
        self.img = torch.squeeze(self.img,0)
        self.img_shape = self.img.shape
        self.repeat = repeat

    def get_pixel_indices(self):
        """
        Get the indices of pixels in an image where the intensity is greater than 0.
        """
        self.gz_idx = torch.argwhere(self.img > 0)


    def generate_temporal_code(self, repeat):
        """
        Generate a temporal code where repeated values form equal squares in the plotted matrix,
        filling the entire matrix even if it's not a perfect square.

        Args:
        repeat (int): The number of elements in each square block (assumed to be a perfect square).

        Returns:
        numpy.ndarray: An array of temporal codes arranged in tiled square blocks.
        """
        # Check if repeat is a perfect square
        if int(math.sqrt(repeat))**2 != repeat:
            raise ValueError("Repeat value must be a perfect square.")

        # Calculate the size of each block
        block_size = int(math.sqrt(repeat))

        # Calculate the number of blocks needed for width and height
        blocks_per_row = int(math.ceil(self.img_shape[1] / block_size))
        blocks_per_col = int(math.ceil(self.img_shape[0] / block_size))

        # Generate unique values for each block
        total_blocks = blocks_per_row * blocks_per_col
        unique_values = np.linspace(1, 0, total_blocks, endpoint=False)

        # Create a matrix of blocks
        self.tempcode = np.zeros(self.img_shape)
        for i in range(blocks_per_col):
            for j in range(blocks_per_row):
                # Determine the value for this block
                block_value = unique_values[i * blocks_per_row + j]
                # Fill the block, adjusting for edges
                row_end = min((i+1)*block_size, self.img_shape[0])
                col_end = min((j+1)*block_size, self.img_shape[1])
                self.tempcode[i*block_size:row_end, j*block_size:col_end] = block_value
        # Plotting the temporal code
        #plt.imshow(self.tempcode, cmap='viridis')
        #plt.colorbar()
        #plt.title("Tiled Temporal Code Matrix")
        #plt.show()

    def match_pixels_to_index(self):
        """
        Match pixels from the codes to the given pixel_index array.
        For each code, create a 1D numpy array with the same length as the number of points in pixel_index.
        Populate this array with values from the code corresponding to the positions in pixel_index.
        If a pixel is found to be above 0, fill it in the corresponding index of the new array.
        """
        pixel_index = np.load('./vprtemponeuro/dataset/pixel_selection.npy')
        img = np.zeros(len(pixel_index))
        pixel_index_div = divmod(pixel_index, self.img_shape[0])
        pixel_index_div_xy = np.column_stack((pixel_index_div[0], pixel_index_div[1]))
        for i, (y, x) in enumerate(pixel_index_div_xy):
            if self.img[x, y] > 0:
               img[i] = self.tempcode[x, y]

        return img

    def main(self):
        self.get_pixel_indices()
        self.generate_temporal_code(self.repeat)
        img = self.match_pixels_to_index()

        return img
        
class ProcessImage:
    def __init__(self, repeat):
        super(ProcessImage, self).__init__()
        self.repeat = repeat
    def __call__(self, img):
        GTC = GenerateTemporalCode(img,self.repeat)
        img = GTC.main()
        return img

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, base_dir, img_dirs, transform=None, target_transform=None, 
                 skip=1, max_samples=None, test=True, is_spiking=False, is_raster=False, time_window=50):
        self.transform = transform
        self.target_transform = target_transform
        self.skip = skip
        self.is_spiking = is_spiking
        self.is_raster = is_raster
        self.time_window = time_window
        
        # Load image labels from each directory, apply the skip and max_samples, and concatenate
        self.img_labels = []
        for img_dir in img_dirs:

            img_labels = pd.read_csv(annotations_file)
            img_labels['file_path'] = img_labels.apply(lambda row: os.path.join(base_dir,img_dir, row.iloc[0]), axis=1)
            
            # Select specific rows based on the skip parameter
            img_labels = img_labels.iloc[::skip]
            
            # Limit the number of samples to max_samples if specified
            if max_samples is not None:
                img_labels = img_labels.iloc[:max_samples]
            
            # Determine if the images being fed are training or testing
            if test:
                self.img_labels = img_labels
            else:
                self.img_labels.append(img_labels)
        
        if isinstance(self.img_labels,list):
            # Concatenate all the DataFrames
            self.img_labels = pd.concat(self.img_labels, ignore_index=True)
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx]['file_path']
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No file found for index {idx} at {img_path}.")

        image = read_image(img_path)  # image is now a tensor
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image=torch.tensor(image)

        if self.is_raster:
            image = (torch.rand(self.time_window, *image.shape) < image).float()
        if self.is_spiking:
            sqrt_div = math.sqrt(image[-1].size()[0])
            image = image.view(self.time_window,int(sqrt_div),int(sqrt_div))
            image = image.unsqueeze(1)
        return image, label