import os
import math
import cv2
import torch

import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.quantization as tq

from torchvision.io import read_image
from torch.utils.data import Dataset

class GenerateTemporalCode:
    def __init__(self, img):
        super(GenerateTemporalCode, self).__init__()
        self.img = img
        self.img = torch.squeeze(self.img,0)
        self.img_shape = self.img.shape

    def get_pixel_indices(self):
        """
        Get the indices of pixels in an image where the intensity is greater than 0.
        """
        self.gz_idx = torch.argwhere(self.img > 0)

    def generate_temporal_code(self):
        """
        Generate a temporal code where the pixel index determines a value in the range [0, 1].
        The top left pixel index will be the highest value (1) and the bottom right pixel will be the lowest non-zero value.
        """
        total_pixels = np.prod(self.img_shape)
        self.tempcode = np.linspace(1, 0, total_pixels, endpoint=False).reshape(self.img_shape)

    def match_pixels_to_index(self):
        """
        Match pixels from the codes to the given pixel_index array.
        For each code, create a 1D numpy array with the same length as the number of points in pixel_index.
        Populate this array with values from the code corresponding to the positions in pixel_index.
        If a pixel is found to be above 0, fill it in the corresponding index of the new array.
        """
        pixel_index = np.load('./dataset/pixel_selection.npy')
        img = np.zeros(len(pixel_index))
        pixel_index_div = divmod(pixel_index, self.img_shape[0])
        pixel_index_div_xy = np.column_stack((pixel_index_div[0], pixel_index_div[1]))
        for i, (y, x) in enumerate(pixel_index_div_xy):
            if self.img[x, y] > 0:
               img[i] = self.tempcode[x, y]

        return img

    def main(self):
        self.get_pixel_indices()
        self.generate_temporal_code()
        img = self.match_pixels_to_index()

        return img
        
class ProcessImage:
    def __init__(self):
        super(ProcessImage, self).__init__()
        
    def __call__(self, img):
        GTC = GenerateTemporalCode(img)
        img = GTC.main()
        return img

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, base_dir, img_dirs, transform=None, target_transform=None, 
                 skip=1, max_samples=None, test=True, is_spiking=True, time_window=100):
        self.transform = transform
        self.target_transform = target_transform
        self.skip = skip
        self.is_spiking = is_spiking
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
        if self.is_spiking:
            image = (torch.rand(self.time_window, *image.shape) < image).float()

        return image, label