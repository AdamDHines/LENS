#MIT License

#Copyright (c) 2023 Adam Hines, Peter G Stratton, Michael Milford, Tobias Fischer

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

'''
Imports
'''

import os
import csv
import torch

import vprtemponeuro.src.blitnet as bn
import numpy as np
import torch.nn as nn
import sinabs.layers as sl
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from vprtemponeuro.src.loggers import model_logger
from sinabs.from_torch import from_model
from vprtemponeuro.src.dataset_patchnorm import CustomImageDataset, ProcessImage
from torch.utils.data import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable
from vprtemponeuro.src.metrics import recallAtK

class VPRTempo(nn.Module):
    def __init__(self, args):
        super(VPRTempo, self).__init__()

        # Set the arguments
        self.args = args
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

        # Set the dataset file
        self.dataset_file = os.path.join('./vprtemponeuro/dataset', self.dataset + '.csv')

        # Set the model logger and return the device
        self.device = model_logger(self)    

        # Layer dict to keep track of layer names and their order
        self.layer_dict = {}
        self.layer_counter = 0

        # Define layer architecture
        self.input = int(args.dims[0]*args.dims[1])
        self.feature = int(self.input*2)
        self.output = int(args.database_places / args.num_modules)

        """
        Define trainable layers here
        """
        self.add_layer(
            'feature_layer',
            dims=[self.input, self.feature],
            device=self.device,
            inference=True
        )
        self.add_layer(
            'output_layer',
            dims=[self.feature, self.output],
            device=self.device,
            inference=True
        )
        
    def add_layer(self, name, **kwargs):
        """
        Dynamically add a layer with given name and keyword arguments.
        
        :param name: Name of the layer to be added
        :type name: str
        :param kwargs: Hyperparameters for the layer
        """
        # Check for layer name duplicates
        if name in self.layer_dict:
            raise ValueError(f"Layer with name {name} already exists.")
        
        # Add a new SNNLayer with provided kwargs
        setattr(self, name, bn.SNNLayer(**kwargs))
        
        # Add layer name and index to the layer_dict
        self.layer_dict[name] = self.layer_counter
        self.layer_counter += 1                           

    def evaluate(self, model, test_loader, layers=None):
        """
        Run the inferencing model and calculate the accuracy.

        :param test_loader: Testing data loader
        :param layers: Layers to pass data through
        """
        #nn.init.eye_(self.inert_conv_layer.weight)
        self.inference = nn.Sequential(
            self.feature_layer.w,
            nn.ReLU(),
            self.output_layer.w,
        )
        # Specify the path to your .csv file
        csv_file_path = model.data_dir + model.database+'.csv'

        # Initialize an empty list to store the rows
        data_rows = []

        # Open and read the .csv file
        with open(csv_file_path, mode='r') as file:
            # Create a csv reader object
            csv_reader = csv.reader(file)
            # Skip the header
            next(csv_reader)
            # Read each row after the header
            for row in csv_reader:
                data_rows.append(row)

        # Initialize the tqdm progress bar
        pbar = tqdm(total=self.query_places,
                    desc="Running the test network",
                    position=0)
        # Initiliaze the output spikes variable
        out = []
        correct = 0
        incorrect = 0
        counter = 0
        matches = []
        misses = []
        # Run inference for the specified number of timesteps
        with torch.no_grad():
            for spikes, labels, gps in test_loader:
                spikes, labels = spikes.to(self.device), labels.to(self.device)
                spikes = spikes.squeeze(0)
                spikes = spikes.to(torch.float32)
                # Forward pass
                spikes = self.forward(spikes)

                # If max spike output is greater than threshold, perform matching
                if torch.max(spikes) > 0.049:
                    gps_str = gps[0].strip('[]')
                    # Split the string by comma
                    lat_str, long_str = gps_str.split(',')
                    # Convert to float
                    lat = round(float(lat_str),3)
                    long = round(float(long_str),3)
                    output_idx = torch.argmax(spikes)

                    ref_str = data_rows[output_idx][2].strip('[]')
                    ref_lat_str, ref_long_str = ref_str.split(',')
                    ref_lat = round(float(ref_lat_str),3)
                    ref_long = round(float(ref_long_str),3)

                    if lat == ref_lat and long == ref_long:
                        correct += 1
                        matches.append(counter)
                    else:
                        incorrect += 1
                        misses.append(counter)
                    
                # Add output spikes to list
                out.append(spikes.detach().cpu().tolist())
                counter += 1
                pbar.update(1)

        # Close the tqdm progress bar
        pbar.close()
        # Rehsape output spikes into a similarity matrix
        out = np.reshape(np.array(out),(model.query_places,model.database_places))
        np.save('/home/adam/Documents/similarity_matrix.npy', out)
        np.save('/home/adam/Documents/matches.npy', matches)
        np.save('/home/adam/Documents/misses.npy', misses)
        # Recall@N
        N = [1,5,10,15,20,25] # N values to calculate
        R = [] # Recall@N values
        # Create GT matrix
        GT = np.zeros((model.query_places,model.database_places), dtype=int)
        for n in range(len(GT)):
            GT[n,n] = 1
        # Calculate Recall@N
        #for n in N:
        #    R.append(round(recallAtK(out,GThard=GT,K=n),2))
        # Print the results
        #table = PrettyTable()
        #table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
        #table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])
        #model.logger.info(table)

        # Plot similarity matrix
        if self.sim_mat:
            plt.matshow(out)
            plt.colorbar(shrink=0.75,label="Output spike intensity")
            plt.title('Similarity matrix')
            plt.xlabel("Query")
            plt.ylabel("Database")
            plt.show()

        return R


    def forward(self, spikes):
        """
        Compute the forward pass of the model.
    
        Parameters:
        - spikes (Tensor): Input spikes.
    
        Returns:
        - Tensor: Output after processing.
        """
        
        spikes = self.inference(spikes)
        
        return spikes
        
    def load_model(self, model_path):
        """
        Load pre-trained model and set the state dictionary keys.
        """
        self.load_state_dict(torch.load(model_path, map_location=self.device),
                             strict=False)

def run_inference_norm(model, model_name):
    """
    Run inference on a pre-trained model.

    :param model: Model to run inference on
    :param model_name: Name of the model to load
    :param qconfig: Quantization configuration
    """
    # Create the dataset from the numpy array
    image_transform = transforms.Compose([
                                        ProcessImage(model.dims,model.patches)
                                            ])
    test_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                                      base_dir=model.data_dir,
                                      img_dirs=model.query_dir,
                                      transform=image_transform,
                                      skip=model.filter,
                                      max_samples=model.query_places,
                                      is_raster=False)
    print(str(model.data_dir+model.query_dir[0]))
    # Initialize the data loader
    test_loader = DataLoader(test_dataset, 
                              batch_size=1, 
                              shuffle=False,
                              num_workers=8,
                              persistent_workers=True)
    # Set the model to evaluation mode and set configuration
    model.eval()

    # Load the model
    model.load_model(os.path.join('./vprtemponeuro/models', model_name))

    # Retrieve layer names for inference
    layer_names = list(model.layer_dict.keys())

    # Use evaluate method for inference accuracy
    R1 = model.evaluate(model, test_loader, layers=layer_names)
    return R1