#MIT License

#Copyright (c) 2024 Adam Hines, Michael Milford, Tobias Fischer

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
import json
import torch

import numpy as np
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import vprtemponeuro.src.blitnet as bn
import torchvision.transforms as transforms

from tqdm import tqdm
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from vprtemponeuro.src.loggers import model_logger
from vprtemponeuro.src.metrics import recallAtK, createPR
from vprtemponeuro.src.dataset import CustomImageDataset, ProcessImage

class VPRTempo(nn.Module):
    def __init__(self, args):
        super(VPRTempo, self).__init__()

        # Set the arguments
        self.args = args
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

        # Set the dataset file
        if self.reference_annotation:
            self.dataset_file = os.path.join(self.data_dir, self.query+ '_reference.csv')
        else:
            self.dataset_file = os.path.join(self.data_dir, self.query + '.csv')

        # Set the query image folder
        self.query_dir = os.path.join(self.data_dir, self.dataset, self.camera, self.query)

        # Set the model logger and return the device
        self.device = model_logger(self)    

        # Layer dict to keep track of layer names and their order
        self.layer_dict = {}
        self.layer_counter = 0

        # Define layer architecture
        self.input = int(args.dims[0]*args.dims[1])
        self.feature = int(self.input*args.feature_multiplier)
        self.output = int(args.reference_places)
        self.train_sequential = nn.Sequential(
            # 2 x 128 x 128
            # Core 0
            nn.Conv2d(1, 8, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),  # 8, 64, 64
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 8,32,32
            # """Core 1"""
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 16, 32, 32
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 16, 16, 16
            # """Core 2"""
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 16, 16
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 16, 16, 16
            nn.Flatten(),
            nn.Dropout(0.75),
            nn.Linear(8 * 8 * 8, 113, bias=False)
        )
        self.train_sequential.to(self.device)
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
            dims=[24, 118],
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


        # Initiliaze the output spikes variable
        out = []
        # Initialize the tqdm progress bar
        pbar = tqdm(total=self.query_places,
                    desc="Running the test network",
                    position=0)
        count = 0
        num_corr = 0
        k=3
        q = []
        r = []
        # Run inference for the specified number of timesteps
        with torch.no_grad():
            for spikes, labels, gps, _ in test_loader:
                spikes, labels = spikes.to(self.device), labels.to(self.device)
                # spikes = spikes.squeeze(0)
                spikes = spikes.to(torch.float32)
              
                # Forward pass
                spikes = self.forward(spikes)
                spikes = spikes.squeeze(0)
                thresh = torch.mean(spikes) + (k * torch.std(spikes))
                # Add output spikes to list
                if torch.max(spikes) > thresh:
                    out.append(spikes.detach().cpu().tolist())
                    count += 1
                    q.append(labels.detach().cpu().item())
                    r.append(torch.argmax(spikes).detach().cpu().item())
                pbar.update(1)


        # Close the tqdm progress bar
        pbar.close()
        # Rehsape output spikes into a similarity matrix
        out = np.reshape(np.array(out),(count,model.reference_places))

        # Run sequence matching
        if self.sequence_length != 0:
            dist_tensor = torch.tensor(out).to(self.device).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
            precomputed_convWeight = torch.eye(self.sequence_length, device=self.device).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
            dist_matrix_seq = torch.nn.functional.conv2d(dist_tensor, precomputed_convWeight).squeeze().cpu().numpy() / self.sequence_length
        else:
            dist_matrix_seq = out
        
        # # Recall@N
        # N = [1,5,10,15,20,25] # N values to calculate
        # R = [] # Recall@N values
        
        # # Load GT matrix
        # GT = np.load(os.path.join(self.data_dir, self.dataset, self.camera, self.reference + '_' + self.query + '_GT.npy'))
        # if self.args.sequence_length != 0:
        #     GT = GT[self.args.sequence_length-2:-1,self.args.sequence_length-2:-1]

        # # Calculate Recall@N
        # for n in N:
        #     R.append(round(recallAtK(dist_matrix_seq,GThard=GT,K=n),2))
        # # Print the results
        # table = PrettyTable()
        # table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
        # table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])
        # model.logger.info(table)

        # Plot similarity matrix
        if self.sim_mat:
            plt.figure(figsize=(10, 8))
            sns.heatmap(dist_matrix_seq, annot=False, cmap='coolwarm')
            plt.title('Similarity matrix')
            plt.xlabel("Database")
            plt.ylabel("Query")
            plt.show()

        # Plot precision recall curve
        if self.PR_curve:
            # Create PR curve
            P, R = createPR(dist_matrix_seq, GThard=GT, GTsoft=GT, matching='multi', n_thresh=100)
            #  Combine P and R into a list of lists
            PR_data = {
                    "Precision": P,
                    "Recall": R
                }
            output_file = "PR_curve_data.json"
            # Construct the full path
            full_path = f"{model.data_dir}/{output_file}"
            # Write the data to a JSON file
            with open(full_path, 'w') as file:
                json.dump(PR_data, file) 
            # Plot PR curve
            plt.plot(R,P)    
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
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
        
        spikes = self.train_sequential(spikes)

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
                                        ProcessImage()
                                            ])
    test_dataset = CustomImageDataset(annotations_file=model.dataset_file,
                                      img_dir=model.query_dir,
                                      transform=image_transform,
                                      skip=model.filter,
                                      max_samples=model.query_places,
                                      is_raster=False)
    print(str(model.data_dir+model.query_dir))
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