import samna, samnagui
import time
import sys
import os
import multiprocessing
import threading
from queue import Queue
import numpy as np

# Existing function definitions (open_speck2f_dev_kit, build_samna_event_route, open_visualizer) remain unchanged

def open_speck2f_dev_kit():
    devices = [
        device
        for device in samna.device.get_unopened_devices()
        if device.device_type_name.startswith("Speck2f")
    ]
    assert devices, "Speck2f board not found"

    # default_config is a optional parameter of open_device
    default_config = samna.speck2fBoards.DevKitDefaultConfig()

    # if nothing is modified on default_config, this invoke is totally same to
    # samna.device.open_device(devices[0])
    return samna.device.open_device(devices[0], default_config)


def build_samna_event_route(dk, graph, endpoint):
    # build a graph in samna to show dvs
    _, _, streamer = graph.sequential(
        [dk.get_model_source_node(), "Speck2fDvsToVizConverter", "VizEventStreamer"]
    )

    config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])

    streamer.set_streamer_endpoint(endpoint)
    if streamer.wait_for_receiver_count() == 0:
        raise Exception(f'connecting to visualizer on {endpoint} fails')

    return config_source


def open_visualizer(window_width, window_height, receiver_endpoint):
    # start visualizer in a isolated process which is required on mac, intead of a sub process.
    gui_process = multiprocessing.Process(
        target=samnagui.run_visualizer,
        args=(receiver_endpoint, window_width, window_height),
    )
    gui_process.start()

    return gui_process


# New dictionary for event buffering
event_dict = {}

def event_collector(sink):
    event_dict_local = {}
    while gui_process.is_alive():
        events = sink.get_events()
        timestamp = time.time() 
        if timestamp not in event_dict_local:
            event_dict_local[timestamp] = events
        else:
            event_dict_local[timestamp].extend(events)  # Assuming events can be concatenated
        time.sleep(0.033)
        print(timestamp)
    return event_dict_local

streamer_endpoint = "tcp://0.0.0.0:40000"
gui_process = open_visualizer(0.75, 0.75, streamer_endpoint)
dk = open_speck2f_dev_kit()

# Route events
graph = samna.graph.EventFilterGraph()
config_source = build_samna_event_route(dk, graph, streamer_endpoint)
sink = samna.graph.sink_from(dk.get_model().get_source_node())
graph.start()
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
collector_thread = threading.Thread(target=lambda q, arg1: q.update(event_collector(arg1)), args=(event_dict, sink))
collector_thread.start()

# Wait until the visualizer window destroys
gui_process.join()

# Stop the graph and ensure the collector thread is also stopped
graph.stop()
collector_thread.join()

# At this point, `event_dict` will be filled with events. Save it as a .npy file.
np.save('/home/adam/Documents/event_data.npy', event_dict) 