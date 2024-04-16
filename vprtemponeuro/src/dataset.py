'''
Imports
'''
import os
import math
import torch

import pandas as pd
import torch.nn as nn

from torch.utils.data import Dataset
from torchvision.io import read_image

class SetImageAsSpikes:
    def __init__(self, intensity=255, test=True):
        self.intensity = intensity   
    
    def __call__(self, img_tensor):
        N, W, H = img_tensor.shape
        reshaped_batch = img_tensor.view(N, 1, -1)
        
        # Divide all pixel values by 255
        normalized_batch = reshaped_batch / self.intensity
        spikes  = torch.squeeze(normalized_batch, 0)

        return spikes

class ProcessImage:
    def __init__(self, convolve, mid=0.5):
        self.mid = mid
        convolve.requires_grad_ = False
        torch.nn.init.constant_(convolve.weight, 0)
        convolve.weight.data[0, 0, 4, 4] = 1
        if convolve is not None:
            self.cnn = nn.Sequential(
                nn.AvgPool2d(kernel_size=(4, 4)),
                nn.AvgPool2d(kernel_size=(4, 4)),
                
            )
        
    def __call__(self, img):
        # Add a channel dimension to the resulting grayscale image
        if self.cnn is not None:
            with torch.no_grad():
                img = self.cnn(img.to(torch.float32))

        # img= img.unsqueeze(0)
        # img = img.to(dtype=torch.float32)
        # # gamma correction
        # mean = torch.mean(img)

        # # Check if mean is zero or negative to avoid math domain error
        # try:
        #     gamma = math.log(self.mid * 255) / math.log(mean)
        #     img = torch.pow(img, gamma).clip(0, 255)
        # except:
        #     pass
        # img = img.squeeze(0)

        # Resize the image to the specified dimensions
        spike_maker = SetImageAsSpikes()
        img = spike_maker(img)
        img = torch.squeeze(img,0)

        return img


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, 
                 skip=1, max_samples=None, test=True, is_spiking=False, is_raster=False, time_window=100,
                 convolve=None):
        self.transform = transform
        self.target_transform = target_transform
        self.skip = skip
        self.is_spiking = is_spiking
        self.is_raster = is_raster
        self.time_window = time_window
        
        # Load image labels from each directory, apply the skip and max_samples, and concatenate
        self.img_labels = []

        img_labels = pd.read_csv(annotations_file)
        img_labels['file_path'] = img_labels.apply(lambda row: os.path.join(img_dir, row.iloc[0]), axis=1)
        
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
        gps_coordinate = self.img_labels.iloc[idx,2]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No file found for index {idx} at {img_path}.")
        image = read_image(img_path)

        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image_og = []
        # Raster the input for image
        if self.is_raster:
            image_og = image.clone()
            torch.manual_seed(50)
            image = (torch.rand(self.time_window, *image.shape) < image).float()
        # Prepare the spikes for deployment to speck2devkit
        if self.is_spiking:
            sqrt_div = math.sqrt(image[-1].size()[0])
            image = image.view(self.time_window,int(sqrt_div),int(sqrt_div))
            image = image.unsqueeze(1)

        return image, label, gps_coordinate, image_og