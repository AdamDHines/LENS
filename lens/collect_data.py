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
import imageio
import threading
import samna, samnagui
import multiprocessing

import numpy as np
import torch.nn as nn
import lens.src.blitnet as bn

from sinabs.from_torch import from_model
from lens.src.loggers import model_logger
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from lens.tools.create_data_csv import create_csv_from_images

class LENS_Collector(nn.Module):
    def __init__(self, args):
        super(LENS_Collector, self).__init__()

        # Set the arguments
        self.args = args
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

        # Self output folder
        self.img_folder = os.path.join('./lens/dataset',self.dataset,self.camera,self.data_name)
        # Check if the folder exists, if not, make it
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)

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

    def evaluate(self):
        """
        Run the inferencing model and calculate the accuracy.
        """
        # Define the inferencing forward pass
        def _init_kernel():
            kernel = torch.zeros(1, 1, 8, 8)
            kernel[0, 0, 4, 4] = 1
            return kernel
        self.conv = nn.Conv2d(1, 1, kernel_size=8, stride=8, padding=0, bias=False)
        self.conv.weight = nn.Parameter(_init_kernel(), requires_grad=False)
        # Create a dummy sequence
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
        input_shape = (1, self.roi_dim, self.roi_dim) # With Conv2d becomes [1, 8, 8]
        self.sinabs_model = from_model(
                                self.inference, 
                                input_shape=input_shape,
                                num_timesteps=self.timebin,
                                add_spiking_output=True
                                )

        # Create the DYNAPCNN model for on-chip inferencing
        self.dynapcnn = DynapcnnNetwork(snn=self.sinabs_model, 
                                input_shape=input_shape, 
                                discretize=True, 
                                dvs_input=True)
        
        def open_speck2f_dev_kit():
            devices = [
                device
                for device in samna.device.get_unopened_devices()
                if device.device_type_name.startswith("Speck2f")
            ]
            assert devices, "Speck2f board not found"

            # default_config is a optional parameter of open_device
            self.default_config = samna.speck2fBoards.DevKitDefaultConfig()

            return samna.device.open_device(devices[0], self.default_config)


        def build_samna_event_route(graph, dk):
            # build a graph in samna to show dvs
            _, _, streamer = graph.sequential(
                [dk.get_model_source_node(), "Speck2fDvsToVizConverter", "VizEventStreamer"]
            )

            config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])

            streamer.set_streamer_endpoint("tcp://0.0.0.0:40000")
            if streamer.wait_for_receiver_count() == 0:
                raise Exception('connecting to visualizer on "tcp://0.0.0.0:40000" fails')

            return config_source


        def open_visualizer(window_width, window_height, receiver_endpoint):
            # start visualizer in a isolated process which is required on mac, intead of a sub process.
            gui_process = multiprocessing.Process(
                target=samnagui.run_visualizer,
                args=(receiver_endpoint, window_width, window_height),
            )
            gui_process.start()

            return gui_process

        def event_collector():
            self.infer_count = 0
            self.events = []
            while gui_process.is_alive():
                self.events.append(sink.get_events())  # Make sure 'self.sink' is properly initialized
                time.sleep(self.timebin/1000) # Convert to seconds from ms

        def create_images(events):
            if events:
                frame = torch.zeros((self.roi_dim, self.roi_dim), dtype=int)
                for event in events:
                    frame[event.y-1, event.x-1] += 1
                imageio.imwrite(f'{self.img_folder}/frame_{self.infer_count:05d}.png',frame.detach().cpu().numpy().astype(np.uint8))
                self.infer_count += 1
                print(f'{self.img_folder}/{self.infer_count}.png')
            else:
                print("No events")
            
        gui_process = open_visualizer(0.75, 0.75, "tcp://0.0.0.0:40000")
        dk = open_speck2f_dev_kit()
        stopWatch = dk.get_stop_watch()
        stopWatch.set_enable_value(True)
        dk_io = dk.get_io_module()
        dk_io.set_slow_clk_rate(10)
        dk_io.set_slow_clk(True)
        graph = samna.graph.EventFilterGraph()
        config_source = build_samna_event_route(graph, dk)

        sink = samna.graph.sink_from(dk.get_model().get_source_node())
        # Configuring the visualizer
        visualizer_config = samna.ui.VisualizerConfiguration(
            plots=[samna.ui.ActivityPlotConfiguration(self.roi_dim, self.roi_dim, "DVS Layer", [0, 0, 1, 1])]
        )
        config_source.write([visualizer_config])

        self.dynapcnn.to(device=devkit_name, chip_layers_ordering="auto")
        print(f"The SNN is deployed on the core: {self.dynapcnn.chip_layers_ordering}")
        self.factory = ChipFactory(devkit_name)
        self.first_layer_idx = self.dynapcnn.chip_layers_ordering[0] 

        # Modify configuration to enable DVS event monitoring
        config = samna.speck2f.configuration.SpeckConfiguration()
        config.dvs_layer.monitor_enable = True
        config.dvs_filter.enable = True
        config.dvs_layer.origin.x = 23
        config.dvs_layer.origin.y = 0
        config.dvs_layer.cut.x = 103
        config.dvs_layer.cut.y = 79
        config.dvs_filter.hot_pixel_filter_enable = True
        config.dvs_filter.threshold = 5
        # Apply the configuration
        dk.get_model().apply_configuration(config)
        # Start the event collector thread
        collector_thread = threading.Thread(target=event_collector)
        collector_thread.start()
        # Wait until the visualizer window destroys
        graph.start()
        gui_process.join()

        # Stop the graph and ensure the collector thread is also stopped
        graph.stop()
        collector_thread.join()

        for events in self.events:
            create_images(events)

        create_csv_from_images(self.img_folder, f'./lens/dataset/{self.data_name}.csv')


def run_collector(model):
    """
    Run inference on a pre-trained model.

    :param model: Model to run inference on
    """
    # Set the model to evaluation mode and set configuration
    model.eval()

    # Use evaluate method for inference accuracy
    model.evaluate()