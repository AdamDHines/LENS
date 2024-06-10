import samna, samnagui
import time
import sys
import os
import multiprocessing


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


streamer_endpoint = "tcp://0.0.0.0:40000"

gui_process = open_visualizer(0.75, 0.75, streamer_endpoint)

dk = open_speck2f_dev_kit()

# route events
graph = samna.graph.EventFilterGraph()
config_source = build_samna_event_route(dk, graph, streamer_endpoint)
sink = samna.graph.sink_from(dk.get_model().get_source_node())
graph.start()
# events = sink.get_events()
visualizer_config = samna.ui.VisualizerConfiguration(
    plots=[samna.ui.ActivityPlotConfiguration(128, 128, "DVS Layer", [0, 0, 1, 1])]
)

config_source.write([visualizer_config])

# modify configuration
config = samna.speck2f.configuration.SpeckConfiguration()
# enable dvs event monitoring
config.dvs_layer.monitor_enable = True
dk.get_model().apply_configuration(config)

# wait until visualizer window destroys.
gui_process.join()

graph.stop()

events = sink.get_events()