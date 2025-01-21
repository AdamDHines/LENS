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
import torch

import numpy as np
import seaborn as sns
import torch.nn as nn
import sinabs.layers as sl
import lens.src.blitnet as bn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm
from collections import Counter
from lens.src.sad import run_sad
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from sinabs.from_torch import from_model
from lens.src.loggers import model_logger
from lens.tools.create_GTtol import create_GTtol
from lens.src.metrics import recallAtK, createPR
from sinabs.backend.dynapcnn import DynapcnnNetwork
from lens.tools.plot_results import plot_PR, plot_recall
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from lens.src.dataset import CustomImageDataset, ProcessImage

class LENS(nn.Module):
    def __init__(self, args):
        super(LENS, self).__init__()

        # Set the arguments
        self.args = args
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

        # Set the dataset file
        self.dataset_file = os.path.join(self.data_dir, self.query+ '.csv')
        self.reference_file = os.path.join(self.data_dir, self.reference+ '.csv')
        self.query_dir = os.path.join(self.data_dir, self.dataset, self.camera, self.query)
        self.reference_dir = os.path.join(self.data_dir, self.dataset, self.camera, self.reference)

        # Set the model logger and return the device
        self.device = model_logger(self)    
        # Change to CPU if selected
        if self.nocuda:
            self.device = torch.device('cpu')

        # Layer dict to keep track of layer names and their order
        self.layer_dict = {}
        self.layer_counter = 0

        # Define layer architecture
        self.input = int(args.dims*args.dims)
        self.feature = int(self.input*self.feature_multiplier)
        self.output = int(args.reference_places)

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

        if not hasattr(self, 'matrix'):
            self.matrix = None

        self.kernel_size = self.roi_dim // self.dims

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

    def evaluate(self, test_loader, model):
        """
        Run the inferencing model and calculate the accuracy.

        :param test_loader: Testing data loader
        :param model: Pre-trained network model
        """
        # Define convolutional kernel to select the center pixel
        def _init_kernel():
            kernel = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
            centre_coordinate = (self.kernel_size // 2) - 1
            kernel[0, 0, centre_coordinate, centre_coordinate] = 1  # Set the center pixel to 1
            return kernel
        # Define the Conv2d selection layer
        self.conv = nn.Conv2d(1, 1, kernel_size=self.kernel_size, stride=self.kernel_size, padding=0, bias=False).to(self.device)
        self.conv.weight = nn.Parameter(_init_kernel(), requires_grad=False) # Set the kernel weights
        # Define the inferencing forward pass
        self.inference = nn.Sequential(
            self.conv,
            nn.ReLU(),
            nn.Flatten(),
            self.feature_layer.w,
            nn.ReLU(),
            self.output_layer.w,
        )
        # Define name of the devkit
        devkit_name = "speck2fdevkit"
        # Define the sinabs model, this converts torch model to sinabs model
        input_shape = (1, self.roi_dim, self.roi_dim)
        self.sinabs_model = from_model(
                                self.inference.to(self.device), 
                                input_shape=input_shape,
                                num_timesteps=self.timebin,
                                add_spiking_output=True
        )

        # Initiliaze the output spikes variable
        all_arrays = []
        
        # Run inference for event stream or pre-recorded DVS data
        with torch.no_grad():    
            # Run inference for pre-recorded DVS data    
            if self.simulated_speck:
                self.dynapcnn = DynapcnnNetwork(snn=self.sinabs_model, 
                        input_shape=input_shape, 
                        discretize=True, 
                        dvs_input=True)
                # Deploy the model to the Speck2fDevKit
                self.dynapcnn.to(device=devkit_name, chip_layers_ordering="auto")
                model.logger.info(f"The SNN is deployed on the core: {self.dynapcnn.chip_layers_ordering}")
                factory = ChipFactory(devkit_name)
                first_layer_idx = self.dynapcnn.chip_layers_ordering[0] 
                # Initialize the tqdm progress bar
                pbar = tqdm(total=self.query_places,
                            desc="Running the test network",
                            position=0)
                # Run through the input data
                for spikes, _ , _, _ in test_loader:
                    # Squeeze the batch dimension
                    spikes = spikes.squeeze(0)

                    # create samna Spike events stream
                    try:
                        events_in = factory.raster_to_events(spikes, 
                                                            layer=first_layer_idx,
                                                            dt=1e-6)
                        # Forward pass
                        events_out = self.dynapcnn(events_in)

                        # Get prediction
                        neuron_idx = [each.feature for each in events_out]
                        if len(neuron_idx) != 0:
                            frequent_counter = Counter(neuron_idx)
                        else:
                            frequent_counter = Counter([])
                    except:
                        frequent_counter = Counter([])
                        pass   

                    # Rehsape output spikes into a similarity matrix
                    def create_frequency_array(freq_dict, num_places):
                        # Initialize the array with zeros
                        frequency_array = np.zeros(num_places)

                        # Populate the array with frequency values
                        for key, value in freq_dict.items():
                            if key < num_places:
                                frequency_array[key] = value

                        return frequency_array

                    if not frequent_counter:
                        freq_array = np.zeros(self.reference_places)
                    else:
                        freq_array = create_frequency_array(frequent_counter, self.reference_places)

                    all_arrays.append(freq_array)

                    # Update the progress bar
                    pbar.update(1)

                # Close the tqdm progress bar
                pbar.close()
                model.logger.info("Inference on-chip succesully completed")
                # Convert output to numpy
                out = np.array(all_arrays)
            # Run inference for time based simulation off-chip
            else:
                pbar = tqdm(total=self.query_places,
                            desc="Running the test network",
                            position=0)
                out = []
                for spikes, labels, _, _ in test_loader:
                    spikes, labels = spikes.to(self.device), labels.to(self.device)
                    spikes = sl.FlattenTime()(spikes)
                    # Forward pass
                    spikes = self.sinabs_model(spikes)
                    output = spikes.sum(dim=0).squeeze()
                    # Add output spikes to list
                    out.append(output.detach().cpu())
                    pbar.update(1)
                        # Close the tqdm progress bar
                pbar.close()
                # Rehsape output spikes into a similarity matrix
                out = torch.stack(out, dim=1).numpy()

        # Perform sequence matching convolution on similarity matrix
        if self.sequence_length != 0:   
            dist_tensor = torch.tensor(out).to(self.device).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
            precomputed_convWeight = torch.eye(self.sequence_length, device=self.device).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
            # 3. Perform convolution without padding
            conv_output = F.conv2d(dist_tensor, precomputed_convWeight, padding=0)  # Shape: (1, 1, H_out, W_out)

            # 4. Calculate desired output dimensions
            H, W = out.shape  # Original dimensions

            # 5. Compute output dimensions after convolution
            H_out = conv_output.shape[2]
            W_out = conv_output.shape[3]

            # 6. Define desired output size (same as original)
            H_desired, W_desired = H, W

            # 7. Calculate required padding
            pad_h = H_desired - H_out
            pad_w = W_desired - W_out

            # Ensure that padding is non-negative
            if pad_h < 0 or pad_w < 0:
                raise ValueError("Kernel size is too large, resulting in negative padding.")

            # 8. Distribute padding on top/bottom and left/right
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            # 9. Apply zero padding to the convolved output
            # F.pad expects padding in the order: (pad_left, pad_right, pad_top, pad_bottom)
            padded_conv_output = F.pad(
                conv_output,
                pad=(pad_left, pad_right, pad_top, pad_bottom) # Explicitly set padding value to 0
            )

            # 10. Post-process the result: remove singleton dimensions, move to CPU, convert to NumPy, and normalize
            dist_matrix_seq = padded_conv_output.squeeze().cpu().numpy() / self.sequence_length
        else:
            dist_matrix_seq = out

        # save distance matrix as a pdf image
        plt.imshow(dist_matrix_seq)
        plt.colorbar()
        plt.savefig(os.path.join(self.output_folder, 'distance_matrix_lens.pdf'))
        plt.close()

        # Perform matching if GT is available
        R = []
        if self.matching:
            # Recall@N
            N = [1,5,10,15,20,25] # N values to calculate
            # Create GT matrix
            GT = np.load(f'{self.data_dir}{self.dataset}/{self.camera}/{self.reference}_{self.query}_GT.npy')

            # Create GTsoft with a customizable number of rows to add
            GTtol = create_GTtol(GT, distance=self.GT_tolerance)

            # save the GTtol matrix as a pdf image
            plt.imshow(GTtol)
            plt.colorbar()
            plt.savefig(os.path.join(self.output_folder, 'GTtol.pdf'))
            plt.close()
            # Calculate Recall@N
            for n in N:
                R.append(round(recallAtK(dist_matrix_seq,GTtol,K=n),2))

            # Print the results
            model.logger.info('===== LENS Recall@N =====')
            table = PrettyTable()
            table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
            table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])
            model.logger.info(table)
         
        if self.sim_mat: # Plot only the similarity matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(dist_matrix_seq, annot=False, cmap='crest')
            plt.title('Similarity matrix')
            plt.xlabel("Query")
            plt.ylabel("Database")
            plt.show()

        # Plot PR curve
        if self.PR_curve:
            # Create PR curve
            LENS_P, LENS_R = createPR(dist_matrix_seq, GTtol, self.output_folder, matching='single', n_thresh=100)

            #  Combine P and R into a list of lists
            lens_PR = {
                    "Precision": LENS_P,
                    "Recall": LENS_R
                }

        if self.sad:
            # Run SAD matching
            sad_PR, sad_Recall = run_sad(self.reference_file, self.reference_dir, self.dataset_file, self.query_dir, GT, GTtol, self.output_folder, self.sequence_length)
            plot_recall(R, sad_Recall, N, self.output_folder)
            # plot the results
            if self.PR_curve:
                plot_PR(lens_PR, sad_PR, self.output_folder)
        
        model.logger.info('')    
        model.logger.info('Succesfully completed inferencing using LENS')

        return R

    def forward(self, spikes):
        """
        Compute the forward pass of the model.
    
        Parameters:
        - spikes (Tensor): Input spikes.
    
        Returns:
        - Tensor: Output after processing.
        """
        return self.dynapcnn(spikes)
        
    def load_model(self, model_path):
        """
        Load pre-trained model and set the state dictionary keys.
        """
        self.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True),
                             strict=False)

def run_inference(model, model_name):
    """
    Run inference on a pre-trained model.

    :param model: Model to run inference on
    :param model_name: Name of the model to load
    :param qconfig: Quantization configuration
    """
    # Initialize the image transforms and datasets
    image_transform = transforms.Compose([
        ProcessImage()
    ])

    test_dataset = CustomImageDataset(annotations_file=model.dataset_file,
                                      img_dir=model.query_dir,
                                      transform=image_transform,
                                      kernel_size=model.kernel_size,
                                      skip=model.filter,
                                      max_samples=model.query_places,
                                      is_spiking=True,
                                      time_window=model.timebin)

    # Initialize the data loader
    test_loader = DataLoader(test_dataset, 
                              batch_size=1, 
                              shuffle=False,
                              num_workers=4,
                              persistent_workers=True)
    # Set the model to evaluation mode and set configuration
    model.eval()

    # Load the model
    model.load_model(os.path.join('./lens/models', model_name))

    # Use evaluate method for inference accuracy
    R = model.evaluate(test_loader, model)

    return R