#MIT License

#Copyright (c) 2024 Adam Hines

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

import samna, samnagui
import time
import os
import multiprocessing
import threading
import numpy as np
import queue

class Collector:
    def __init__(self, data_dir, exp_name, time_int=1.0):
        super().__init__()
        self.streamer_endpoint = "tcp://0.0.0.0:40000"
        self.time_int = time_int
        self.data_dir = data_dir
        self.exp_name = exp_name

    def open_speck2f_dev_kit(self):
        devices = [
            device
            for device in samna.device.get_unopened_devices()
            if device.device_type_name.startswith("Speck2f")
        ]
        assert devices, "Speck2f board not found"

        # default_config is a optional parameter of open_device
        self.default_config = samna.speck2fBoards.DevKitDefaultConfig()

        # if nothing is modified on default_config, this invoke is totally same to
        # samna.device.open_device(devices[0])
        return samna.device.open_device(devices[0], self.default_config)


    def build_samna_event_route(self, graph, dk):
        # build a graph in samna to show dvs
        _, _, streamer = graph.sequential(
            [dk.get_model_source_node(), "Speck2fDvsToVizConverter", "VizEventStreamer"]
        )

        config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])

        streamer.set_streamer_endpoint(self.streamer_endpoint)
        if streamer.wait_for_receiver_count() == 0:
            raise Exception(f'connecting to visualizer on {self.streamer_endpoint} fails')

        return config_source


    def open_visualizer(self, window_width, window_height, receiver_endpoint):
        # start visualizer in a isolated process which is required on mac, intead of a sub process.
        gui_process = multiprocessing.Process(
            target=samnagui.run_visualizer,
            args=(receiver_endpoint, window_width, window_height),
        )
        gui_process.start()

        return gui_process


    # New dictionary for event buffering
    event_dict = {}


    
    def start_visualizer(self):
        event_queue = queue.Queue()
        def event_collector(event_queue):
            while gui_process.is_alive():
                events = sink.get_events()  # Make sure 'sink' is properly initialized
                event_queue.put(events)
                time.sleep(self.time_int)

        gui_process = self.open_visualizer(0.75, 0.75, self.streamer_endpoint)
        dk = self.open_speck2f_dev_kit()
        graph = samna.graph.EventFilterGraph()
        config_source = self.build_samna_event_route(graph, dk)
        
        sink = samna.graph.sink_from(dk.get_model().get_source_node())
        # Configuring the visualizer
        visualizer_config = samna.ui.VisualizerConfiguration(
            plots=[samna.ui.ActivityPlotConfiguration(128, 128, "DVS Layer", [0, 0, 1, 1])]
        )
        config_source.write([visualizer_config])

        # Modify configuration to enable DVS event monitoring
        config = samna.speck2f.configuration.SpeckConfiguration()
        config.dvs_layer.monitor_enable = True
        dk.get_model().apply_configuration(config)

        # Start the event collector thread
        collector_thread = threading.Thread(target=lambda: event_collector(event_queue))
        collector_thread.start()

        # Example of processing events (could be done in a separate thread or the main thread)
        try:
            while gui_process.is_alive():
                try:
                    # Adjust timeout as needed, it's here to ensure responsiveness on shutdown
                    events = event_queue.get(timeout=1)  
                    print(events)
                    # Process events here
                except queue.Empty:
                    continue
        finally:
            # Ensure cleanup is performed properly
            gui_process.join()
            graph.stop()
            collector_thread.join()

        print('Event collection stopped.')

    def save_events(self):
        # At this point, `event_dict` will be filled with events. Save it as a .npy file.
        np.save(os.path.join(self.data_dir,self.exp_name+'.npy'), self.event_dict) 