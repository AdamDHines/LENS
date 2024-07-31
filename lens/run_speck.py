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
import time
import torch
import samna
import threading

import numpy as np
import torch.nn as nn
import lens.src.speck2f as s
import lens.src.blitnet as bn

from sinabs.from_torch import from_model
from lens.src.loggers import model_logger
from sinabs.backend.dynapcnn import DynapcnnNetwork

class LENSSpeck(nn.Module):
    def __init__(self, args):
        super(LENSSpeck, self).__init__()

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

    def evaluate(self, model):
        """
        Run the inferencing model and calculate the accuracy.

        :param test_loader: Testing data loader
        :param model: Pre-trained network model
        """
        # Define convolutional kernel to select the center pixel
        def _init_kernel():
            kernel = torch.zeros(1, 1, 8, 8)
            kernel[0, 0, 3, 3] = 1  # Set the center pixel to 1
            return kernel
        # Define the Conv2d selection layer
        self.conv = nn.Conv2d(1, 1, kernel_size=8, stride=8, padding=0, bias=False)
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
                                self.inference, 
                                input_shape=input_shape,
                                num_timesteps=self.timebin,
                                add_spiking_output=True
                                )
        # Adjust the spiking thresholds
        self.sinabs_model.layers[2][1].spike_threshold = torch.nn.Parameter(data=torch.tensor(10.),requires_grad=False)
        self.sinabs_model.layers[4][1].spike_threshold = torch.nn.Parameter(data=torch.tensor(2.),requires_grad=False)
        # Create the DYNAPCNN model for on-chip inferencing
        self.dynapcnn = DynapcnnNetwork(snn=self.sinabs_model, 
                                input_shape=input_shape, 
                                discretize=True, 
                                dvs_input=True)
        
        # Modify the configuartion of the DYNAPCNN model if streaming DVS events for inferencing
        # Custom readout function to get output spikes
        def custom_readout(collection):
            def generate_result(feature):
                e = samna.ui.Readout()
                e.feature = feature
                return [e]
            cur_time = time.time()
            # Collect event count information
            for spike in collection:
                if spike.feature in self.sum:
                    self.sum[spike.feature] += 1
                else:
                    self.sum[spike.feature] = 1

            # Print out timestep details
            model.logger.info(f'Collected {len(collection)} output spikes at time {cur_time}')
            # Update number of queries for sequence matching
            self.qry += 1
            # Save the output spikes as a NumPy array
            self.collection.append([self.sum])
            np.save(os.path.join(model.output_folder,"spike_data.npy"), np.array(self.collection))
            # Generate dummy result
            return generate_result(int(0))
            
        # Custom function for online sequence matching
        def seq_match():
            from scipy.signal import convolve2d
            while gui_process.is_alive(): 
                if self.qry == 4: # Wait for 4 input queries
                    if self.save_input:
                        # Convert the list of events to a numpy array with dtype=object
                        events_array = np.array(self.event_sink.get_events(), dtype=object)
                        if not os.path.exists(os.path.join(model.output_folder, 'events')):
                            os.makedirs(os.path.join(model.output_folder, 'events'))
                        # Save the numpy array to a file
                        np.save(os.path.join(model.output_folder, 'events', f"{time.time()}_events.npy"), events_array)
                    # Initialize a NumPy array of zeros
                    vector = np.zeros(self.reference_places, dtype=int)
                    # Fill in the values from the dictionary
                    for key, value in self.sum.items():
                        vector[key] = value
                    # Divide by 4 to get the average and add to sequence matrix
                    if self.sequence is None:
                        self.sequence = vector // 4
                    else:
                        # Append the new sequence to the existing sequence
                        self.sequence = np.vstack((self.sequence, vector // 4))
                        # Check if appropriate number of sequences are collected
                        if self.sequence.shape[0] == 4:
                            # Apply the sequence matching convolution
                            result = convolve2d(self.sequence.T, self.precomputed_convWeight, mode='same')/ self.sequence_length
                            # Find the argmax for each column
                            argmax_columns = np.argmax(result, axis=0)

                            # Log the results
                            model.logger.info('')
                            model.logger.info('\\\\\ Place matching result ////')
                            for i, argmax in enumerate(argmax_columns):
                                model.logger.info(f'The sequence match location for {i} is place number: {argmax}')
                            model.logger.info('')
                            # If matrix doesn't exist, initialize it with the first result
                            if self.matrix is None:
                                self.matrix = result
                            else:
                                # Append the result to the existing matrix using vstack
                                self.matrix = np.concatenate((self.matrix, result),axis=1)
                            # Save the matrix to a file
                            np.save(f"{self.output_folder}/similarity_matrix.npy", self.matrix.T)
                            # Reset the sequencing variables
                            self.sum = {}
                            self.sequence = None
                        else:
                            pass
                    # Reset qry counter
                    self.qry = 0

        def configure_visualizer(graph, streamer):
            config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])
            #graph.start()
            
            visualizer_config = samna.ui.VisualizerConfiguration(
                # add plots to gui
                plots=[
                    # add plot to show pixels
                    samna.ui.ActivityPlotConfiguration(80, 80, "DVS Layer", [0, 0, 1.0, 1.0]),
                    samna.ui.PowerMeasurementPlotConfiguration(
                    title="Power Consumption",
                    channel_count=5,
                    line_names=["io", "ram", "logic", "vddd", "vdda"],
                    layout=[0, 0.8, 1, 1],
                    show_x_span=10,
                    label_interval=2,
                    max_y_rate=1.5,
                    show_point_circle=False,
                    default_y_max=1,
                    y_label_name="power (mW)",
                )
                ]
            )

            return config_source, visualizer_config
        # Basic configuration
        streamer_endpoint = "tcp://0.0.0.0:40000"
        gui_process = s.open_visualizer(streamer_endpoint, 0.75, 0.75, headless=self.headless)
        config = self.dynapcnn.make_config("auto",device=devkit_name)
        lyrs = self.dynapcnn.chip_layers_ordering[-1]
        # Enable layer monitoring
        config.dvs_layer.monitor_enable = True
        config.cnn_layers[lyrs].monitor_enable = True
        # Setup DVS filtering
        config.dvs_filter.enable = True
        config.dvs_filter.hot_pixel_filter_enable = True
        config.dvs_filter.threshold = 5
        # Switch only on channels from DVS
        config.dvs_layer.merge = True
        # Set the ROI
        config.dvs_layer.origin.x = 23
        config.dvs_layer.origin.y = 0
        config.dvs_layer.cut.x = 102
        config.dvs_layer.cut.y = 79
        # Get the Speck2fDevKit configuration for graph sequential routing
        dk = s.get_speck2f()
        # Apply the configuration to the DYNAPCNN model
        dk.get_model().apply_configuration(config)
        # Setup the graph for routing to the GUI process
        graph = samna.graph.EventFilterGraph()
        streamer = s.build_samna_event_route(graph, dk)

        # Define spike collection nodes for GUI plotting
        (source,readout_spike, spike_collection_filter, _,_) = graph.sequential(
                [
                    dk.get_model_source_node(),
                    "Speck2fOutputMemberSelect",
                    "Speck2fSpikeCollectionNode",
                    "Speck2fSpikeCountNode",
                    streamer,
                ]
        )
        self.event_sink = samna.graph.sink_from(source)
        # Set the collection interval for event driven output spikes
        spike_collection_filter.set_interval_milli_sec(self.timebin)
        readout_spike.set_white_list([lyrs], "layer")

        # # Set the interval for spike collection
        _, readout_filter, _ = graph.sequential(
                                        [spike_collection_filter, "Speck2fCustomFilterNode", streamer]
                                        ) 
        readout_filter.set_filter_function(custom_readout)
        # Setup power and readout filter
        power = dk.get_power_monitor()
        power.start_auto_power_measurement(20)
        power_source, _, _ = graph.sequential([power.get_source_node(), "MeasurementToVizConverter", streamer])
        power_sink = samna.graph.sink_from(power_source)
        # Function to read out power from sink node
        def get_events():
            return power_sink.get_events()
        # Configure the visualizer
        config_source, visualizer_config = configure_visualizer(graph, streamer)
        config_source.write([visualizer_config])
        graph.start()
    
        # Run inference for event stream or pre-recorded DVS data
        with torch.no_grad():    
            # Run inference on-chip
            if self.event_driven:
                # Set timestamps for spike events
                stopWatch = dk.get_stop_watch()
                stopWatch.set_enable_value(True)
                # Set the slow clock rate
                dk_io = dk.get_io_module()
                dk_io.set_slow_clk_rate(10)
                dk_io.set_slow_clk(True)
                # Start the thread collector for the sequence matcher
                self.qry = 0
                self.sum = {}
                self.sequence = None
                self.collection = []
                self.events = []
                self.precomputed_convWeight = np.eye(self.sequence_length, dtype=np.float32)
                collector_thread = threading.Thread(target=seq_match)
                collector_thread.start()
                # Start the process, and wait for window to be destroyed
                model.logger.info('')
                model.logger.info("Starting the inferencing system")
                gui_process.join()
                # Read out the power consumption measurements and save
                power.stop_auto_power_measurement()
                ps = get_events()
                # Stop the GUI process & close the Speck2f device
                readout_filter.stop()
                graph.stop()
                samna.device.close_device(dk)

        # Initialize a list to store arrays for each channel
        channel_data = [[] for _ in range(5)]
        
        # Loop through the data and append values to the corresponding channel list
        for record in ps:
            channel_data[record.channel].append((record.timestamp, record.value))
        
        # Convert lists to numpy arrays
        numpy_arrays = [np.array(data) for data in channel_data]
        np.save(f"{self.output_folder}/power_data.npy", numpy_arrays)

        model.logger.info('')    
        model.logger.info('Succesfully completed inferencing using LENS')

        
    def load_model(self, model_path):
        """
        Load pre-trained model and set the state dictionary keys.
        """
        self.load_state_dict(torch.load(model_path, map_location=self.device),
                             strict=False)

def run_speck(model, model_name):
    """
    Run inference on a pre-trained model.


    """

    # Set the model to evaluation mode and set configuration
    model.eval()

    # Load the model
    model.load_model(os.path.join('./lens/models', model_name))

    # Use evaluate method for inference accuracy
    model.evaluate(model)