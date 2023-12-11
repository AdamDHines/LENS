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
import torch
import samna
import time
import gc
import sys
sys.path.append('./src')
sys.path.append('./models')
sys.path.append('./output')
sys.path.append('./dataset')

import blitnet as bn
import numpy as np
import torch.nn as nn
import sinabs.layers as sl
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from loggers import model_logger
from sinabs.from_torch import from_model
from dataset import CustomImageDataset, ProcessImage
from torch.utils.data import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable
from metrics import recallAtK
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from collections import Counter

class VPRTempoNeuro(nn.Module):
    def __init__(self, args):
        super(VPRTempoNeuro, self).__init__()

        # Set the arguments
        self.args = args
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

        # Set the dataset file
        self.dataset_file = os.path.join('./dataset', self.dataset + '.csv')

        # Set the model logger and return the device
        self.device = model_logger(self)    

        # Layer dict to keep track of layer names and their order
        self.layer_dict = {}
        self.layer_counter = 0

        # Define layer architecture
        self.input = int(args.dims[0]*args.dims[1])
        self.feature = int(self.input*2)
        self.output = int(args.num_places / args.num_modules)

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

    def evaluate(self, test_loader):
        """
        Run the inferencing model and calculate the accuracy.

        :param test_loader: Testing data loader
        :param layers: Layers to pass data through
        """
        # Rehsape output spikes into a similarity matrix
        def create_frequency_array(freq_dict, num_places):
            # Initialize the array with zeros
            frequency_array = np.zeros(num_places)

            # Populate the array with frequency values
            for key, value in freq_dict.items():
                if key < num_places:
                    frequency_array[key] = value

            return frequency_array

        #nn.init.eye_(self.inert_conv_layer.weight)
        self.inference = nn.Sequential(
            nn.Flatten(),
            self.feature_layer.w,
            nn.ReLU(),
            self.output_layer.w,
        )
        # Define the sinabs model
        input_shape = (1, self.dims[0], self.dims[1])
        self.sinabs_model = from_model(
                                self.inference, 
                                input_shape=input_shape,
                                batch_size=1,
                                add_spiking_output=True
                                )
        # Define the dynapcnn model for on chip inference
        self.dynapcnn = DynapcnnNetwork(snn=self.sinabs_model, 
                                    input_shape=input_shape, 
                                    discretize=True, 
                                    dvs_input=False)
        devkit_name = "speck2fdevkit"
        # use the `to` method of DynapcnnNetwork to deploy the SNN to the devkit
        self.dynapcnn.to(device=devkit_name, chip_layers_ordering="auto")
        print(f"The SNN is deployed on the core: {self.dynapcnn.chip_layers_ordering}")
        factory = ChipFactory(devkit_name)
        first_layer_idx = self.dynapcnn.chip_layers_ordering[0] 

        # Initiliaze the output spikes variable
        all_arrays = []

        # Set up the power monitoring (if user input)
        if self.power_monitor:
            # Initialize samna
            samna.init_samna()
            # Get the devkit device
            dk = samna.device.open_device("Speck2fDevKit:0")
            # Start the stop watch
            stop_watch = dk.get_stop_watch()
            # Get the power monitor, source node, and buffer
            power = dk.get_power_monitor()
            source = power.get_source_node()
            power_buffer_node = samna.BasicSinkNode_unifirm_modules_events_measurement()
            # Get the graph sink from the source node
            sink = samna.graph.sink_from(source)
            # Initialize the samna graph
            samna_graph = samna.graph.EventFilterGraph()
            samna_graph.sequential([source, power_buffer_node])
            # Set the sample rate
            sample_rate = 100.0  # Hz

        # Initialize the tqdm progress bar
        pbar = tqdm(total=self.num_places,
                    desc="Running the test network",
                    position=0)
        
        # Run inference for the specified number of timesteps
        with torch.no_grad():
            # Run power monitoring during the inference (if user input)
            if self.power_monitor:
                # start samna graph
                samna_graph.start()
                # start the stop-watch of devkit, then each output data has a proper timestamp
                stop_watch.set_enable_value(True)

                # clear buffer
                power_buffer_node.get_events()
                # start monitor, we need pass a sample rate argument to the power monitor
                power.start_auto_power_measurement(sample_rate)

            # Run through the input data
            for spikes, labels in test_loader:
                # Squeeze the batch dimension
                spikes = spikes.squeeze(0)

                # create samna Spike events stream
                events_in = factory.raster_to_events(spikes, 
                                                    layer=first_layer_idx,
                                                    dt=1e-6)
                # Forward pass
                events_out = self.forward(events_in)

                # Get prediction
                neuron_idx = [each.feature for each in events_out]
                if len(neuron_idx) != 0:
                    frequent_counter = Counter(neuron_idx)
                    #prediction = frequent_counter.most_common(1)[0][0]
                    freq_array = create_frequency_array(frequent_counter, self.num_places)
                    all_arrays.append(freq_array)

                # Update the progress bar
                pbar.update(1)

            # Stop monitoring power (if monitoring)
            if self.power_monitor:
                power_events = power_buffer_node.get_events()
                power.stop_auto_power_measurement()
                stop_watch.set_enable_value(False)
                # stop samna graph
                samna_graph.stop()

        # Close the tqdm progress bar
        pbar.close()
        print("Inference on-chip succesully completed")

        # Reset the chip state
        self.dynapcnn.reset_states()
        print("Chip state has been reset")

        # Get power data (if user input)
        if self.power_monitor:
            # Number of power tracks to observe
            num_power_tracks = 5

            # init dict for storing data of each power track
            power_each_track = dict()
            event_count_each_track = dict()

            # check whether timestamp is correct
            timestamp_all_zero = True

            # loop through all collected power events and get data
            for evt in power_events:

                if evt.timestamp != 0:
                    timestamp_all_zero = False
                
                p_track_id = evt.channel
                tmp_power = power_each_track.get(p_track_id, 0) + evt.value
                tmp_count = event_count_each_track.get(p_track_id, 0) + 1
                
                power_each_track.update({p_track_id: tmp_power})
                event_count_each_track.update({p_track_id: tmp_count})

            # average power and current of each track
            for p_track_id in range(num_power_tracks):
                
                # average power in microwatt
                avg_power = power_each_track[p_track_id] / event_count_each_track[p_track_id] * 1e6
                # calculate current
                if p_track_id == 0:
                    current = avg_power / 2.5 
                else:
                    current = avg_power / 1.2
                    
                print(f'track{p_track_id}: {avg_power}uW, {current}uA') 

        # Convert output to numpy
        all_arrays = np.array(all_arrays).T

        # Recall@N
        N = [1,5,10,15,20,25] # N values to calculate
        R = [] # Recall@N values
        # Create GT matrix
        GT = np.zeros((self.num_places,self.num_places), dtype=int)
        for n in range(len(GT)):
            GT[n,n] = 1
        # Calculate Recall@N
        for n in N:
            R.append(round(recallAtK(all_arrays,GThard=GT,K=n),2))
        # Print the results
        table = PrettyTable()
        table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
        table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])
        print(table)

        # Plot power monitoring results and similarity matrix
        if self.power_monitor:
            # Define the plot style
            plt.style.use('ggplot')

            # Create a figure with two side-by-side subplots
            fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 2]})

            # Plot for power consumption
            p_track_name = ["io", "ram", "logic", "pixel digital", "pixel analog"]
            for p_track_id in range(num_power_tracks):
                x = [each.timestamp for each in power_events if each.channel == p_track_id]
                y = [each.value * 1e6 for each in power_events if each.channel == p_track_id]
                axs[0].plot(x, y, label=p_track_name[p_track_id], alpha=0.8)

            axs[0].set_xlabel("time(us)")
            axs[0].set_ylabel("power(uW)")
            axs[0].set_title("Power consumption")
            axs[0].legend(loc="upper right", fontsize=10)

            # Plot for similarity matrix
            cax = axs[1].matshow(all_arrays, cmap="viridis")
            fig.colorbar(cax, ax=axs[1], shrink=0.75, label="Number of output spikes")
            axs[1].set_title('Similarity matrix', pad=20)  # Add padding to the title
            axs[1].set_xlabel("Query")
            axs[1].set_ylabel("Database")
            axs[1].grid(False)  # Disable grid lines for the similarity matrix

            # Adjust the layout to prevent cutting off the title
            plt.tight_layout()
            plt.show()  
        else: # Plot only the similarity matrix
            plt.matshow(all_arrays)
            plt.colorbar(shrink=0.75,label="Number of output spikes")
            plt.title('Similarity matrix')
            plt.xlabel("Query")
            plt.ylabel("Database")
            plt.show()

    def forward(self, spikes):
        """
        Compute the forward pass of the model.
    
        Parameters:
        - spikes (Tensor): Input spikes.
    
        Returns:
        - Tensor: Output after processing.
        """
        spikes = self.dynapcnn(spikes)
        return spikes
        
    def load_model(self, model_path):
        """
        Load pre-trained model and set the state dictionary keys.
        """
        self.load_state_dict(torch.load(model_path, map_location=self.device),
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
                                      base_dir=model.data_dir,
                                      img_dirs=model.query_dir,
                                      transform=image_transform,
                                      skip=model.filter,
                                      max_samples=model.num_places,
                                      is_raster=True,
                                      is_spiking=True)

    # Initialize the data loader
    test_loader = DataLoader(test_dataset, 
                              batch_size=1, 
                              shuffle=False,
                              num_workers=8,
                              persistent_workers=True)
    # Set the model to evaluation mode and set configuration
    model.eval()

    # Load the model
    model.load_model(os.path.join('./models', model_name))

    # Use evaluate method for inference accuracy
    model.evaluate(test_loader)