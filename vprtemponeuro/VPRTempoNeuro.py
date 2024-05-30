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
import json
import torch
import samna, samnagui
import time
import multiprocessing
import threading

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import vprtemponeuro.src.speck2f as s
import vprtemponeuro.src.blitnet as bn
import torchvision.transforms as transforms

from tqdm import tqdm
from multiprocessing import Process
from queue import Queue
from collections import Counter
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from sinabs.from_torch import from_model
from vprtemponeuro.src.loggers import model_logger
from sinabs.backend.dynapcnn import DynapcnnNetwork
from vprtemponeuro.src.metrics import recallAtK, createPR
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from vprtemponeuro.src.dataset import CustomImageDataset, ProcessImage

class VPRTempoNeuro(nn.Module):
    def __init__(self, args):
        super(VPRTempoNeuro, self).__init__()

        # Set the arguments
        self.args = args
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

        # Set the dataset file
        self.dataset_file = os.path.join(self.data_dir, self.query+ '.csv')
        self.query_dir = os.path.join(self.data_dir, self.dataset, self.camera, self.query)

        # Set the model logger and return the device
        self.device = model_logger(self)    

        # Layer dict to keep track of layer names and their order
        self.layer_dict = {}
        self.layer_counter = 0

        # Define layer architecture
        self.input = int(args.dims[0]*args.dims[1])
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

        self.conv = nn.Conv2d(1, 1, kernel_size=(8, 8), stride=(8, 8), bias=False)

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
        # Pre-define a 2d convolutional layer with average pooling weights
        # Takes pre-pooled [128,128] -> [64,64] -> [8,8]
        kernel_size = 8
        self.conv = nn.Conv2d(1, 1, kernel_size=(kernel_size, kernel_size), stride=(8, 8), bias=False)
        n = kernel_size*kernel_size
        avg_weight = torch.full((1,1,kernel_size,kernel_size), 1.0/n)
        self.conv.weight.data = avg_weight
        self.conv.weight.requires_grad = False

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
        input_shape = (1, 64, 64) # With Conv2d becomes [1, 8, 8]
        self.sinabs_model = from_model(
                                self.inference, 
                                input_shape=input_shape,
                                batch_size=1,
                                add_spiking_output=True
                                )
        
        # Create the DYNAPCNN model for on-chip inferencing
        self.dynapcnn = DynapcnnNetwork(snn=self.sinabs_model, 
                                input_shape=input_shape, 
                                discretize=True, 
                                dvs_input=True)
        
        # Modify the configuartion of the DYNAPCNN model if streaming DVS events for inferencing
        if self.args.onchip: 
            # Basic configuration
            config = self.dynapcnn.make_config("auto",device=devkit_name)
            lyrs = self.dynapcnn.chip_layers_ordering[-1]
            # Enable layer monitoring
            config.dvs_layer.monitor_enable = True
            config.cnn_layers[lyrs].monitor_enable = True
            # Setup DVS filtering
            config.dvs_filter.enable = True
            config.dvs_filter.hot_pixel_filter_enable = True
            config.dvs_filter.threshold = 3
            # Switch only on channels from DVS
            config.dvs_layer.off_channel = False
            config.dvs_layer.on_channel = True
            # Set input pooling for DVS events [128,128] -> [64,64]
            config.dvs_layer.pooling.x = 2
            config.dvs_layer.pooling.y = 2
            # Get the Speck2fDevKit configuration for graph sequential routing
            dk = s.get_speck2f()
            # Apply the configuration to the DYNAPCNN model
            dk.get_model().apply_configuration(config)
            # Open the GUI process
            streamer_endpoint = "tcp://0.0.0.0:40000"
            gui_process = s.open_visualizer(streamer_endpoint)
            # Setup the graph for routing to the GUI process
            graph = samna.graph.EventFilterGraph()
            streamer = s.build_samna_event_route(graph, dk)
            # Define spike collection nodes for GUI plotting
            (_,readout_spike, spike_collection_filter, spike_count_filter,_) = graph.sequential(
                    [
                        dk.get_model_source_node(),
                        "Speck2fOutputMemberSelect",
                        "Speck2fSpikeCollectionNode",
                        "Speck2fSpikeCountNode",
                        streamer,
                    ]
            )
            spike_collection_filter.set_interval_milli_sec(1000)
            readout_spike.set_white_list([lyrs], "layer")

            # # Set the interval for spike collection
            _, readout_filter, _ = graph.sequential(
                                            [spike_collection_filter, "Speck2fCustomFilterNode", streamer]
                                            ) 
            readout_filter.set_filter_function(s.custom_readout)
            # Setup power and readout filter
            power = dk.get_power_monitor()
            power.start_auto_power_measurement(20)
            power_source, _, _ = graph.sequential([power.get_source_node(), "MeasurementToVizConverter", streamer])
            power_sink = samna.graph.sink_from(power_source)
            # Function to read out power from sink node
            def get_events():
                return power_sink.get_events()
            # Configure the visualizer
            config_source, visualizer_config = s.configure_visualizer(graph, streamer, self.args.reference_places)
            config_source.write([visualizer_config])
            graph.start()

        # Initiliaze the output spikes variable
        all_arrays = []
        
        # Run inference for event stream or pre-recorded DVS data
        with torch.no_grad():    
            # Run inference on-chip
            if self.args.onchip:
                # Set timestamps for spike events
                stopWatch = dk.get_stop_watch()
                stopWatch.set_enable_value(True)
                # Set the slow clock rate
                dk_io = dk.get_io_module()
                dk_io.set_slow_clk_rate(10)
                dk_io.set_slow_clk(True)
                # Start the process, and wait for window to be destroyed
                gui_process.join()
                # Read out the power consumption measurements and save
                power.stop_auto_power_measurement()
                ps = get_events()
                # Stop the GUI process & close the Speck2f device
                readout_filter.stop()
                graph.stop()
                samna.device.close_device(dk)
            # Run inference for pre-recorded DVS data    
            else:
                # Deploy the model to the Speck2fDevKit
                self.dynapcnn.to(device=devkit_name, chip_layers_ordering="auto")
                print(f"The SNN is deployed on the core: {self.dynapcnn.chip_layers_ordering}")
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
                        events_out = self.forward(events_in)

                        # Get prediction
                        neuron_idx = [each.feature for each in events_out]
                        if len(neuron_idx) != 0:
                            frequent_counter = Counter(neuron_idx)
                            #prediction = frequent_counter.most_common(1)[0][0]
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

                    freq_array = create_frequency_array(frequent_counter, self.reference_places)
                    all_arrays.append(freq_array)

                    # Update the progress bar
                    pbar.update(1)

                # Close the tqdm progress bar
                pbar.close()
                print("Inference on-chip succesully completed")

        # Organise energy measurements into a NumPy array
        if self.args.onchip:
            # Initialize a list to store arrays for each channel
            channel_data = [[] for _ in range(5)]
            
            # Loop through the data and append values to the corresponding channel list
            for record in ps:
                channel_data[record.channel].append((record.timestamp, record.value))
            
            # Convert lists to numpy arrays
            numpy_arrays = [np.array(data) for data in channel_data]
            np.save(f"{self.output_folder}/power_data.npy", numpy_arrays)

        # Move output spikes from continual inference to output folder
        if self.args.onchip:
            # Move the output spikes to the output folder
            os.rename("spike_data.json", f"{self.output_folder}/spike_data.json")

        # Reset the chip state
        #self.dynapcnn.reset_states()
        #print("Chip state has been reset")

        # # Convert output to numpy
        # out = np.array(all_arrays)

        # # Perform sequence matching convolution on similarity matrix
        # if self.sequence_length != 0:
        #     dist_tensor = torch.tensor(out).to(self.device).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
        #     precomputed_convWeight = torch.eye(self.sequence_length, device=self.device).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
        #     dist_matrix_seq = torch.nn.functional.conv2d(dist_tensor, precomputed_convWeight).squeeze().cpu().numpy() / self.sequence_length
        # else:
        #     dist_matrix_seq = out

        # # Recall@N
        # N = [1,5,10,15,20,25] # N values to calculate
        # R = [] # Recall@N values
        # # Create GT matrix
        # GT = np.load(os.path.join(self.data_dir, self.dataset, self.camera, self.reference + '_' + self.query + '_GT.npy'))
        # if self.sequence_length != 0:
        #     GT = GT[self.sequence_length-2:-1,self.sequence_length-2:-1]
        # # Calculate Recall@N
        # for n in N:
        #     R.append(round(recallAtK(dist_matrix_seq,GThard=GT,K=n),2))
        # # Print the results
        # table = PrettyTable()
        # table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
        # table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])
        # model.logger.info(table)
         
        if self.sim_mat: # Plot only the similarity matrix
            plt.matshow(dist_matrix_seq)
            plt.colorbar(shrink=0.75,label="Output spike intensity")
            plt.title('Similarity matrix')
            plt.xlabel("Query")
            plt.ylabel("Database")
            plt.show()
        # Plot PR curve
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
        ProcessImage(model.conv)
    ])
    test_dataset = CustomImageDataset(annotations_file=model.dataset_file,
                                      img_dir=model.query_dir,
                                      transform=image_transform,
                                      skip=model.filter,
                                      max_samples=model.query_places,
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
    model.load_model(os.path.join('./vprtemponeuro/models', model_name))

    # Use evaluate method for inference accuracy
    model.evaluate(test_loader, model)