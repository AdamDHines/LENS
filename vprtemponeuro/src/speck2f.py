import samna, samnagui
import multiprocessing
import json
import os
import timeit
from datetime import datetime


# Function to retrieve Speck2fDevKit device and default configuration
def get_speck2f():
    # Initially check for any unopened devices
    devices = [
            device
            for device in samna.device.get_unopened_devices()
            if device.device_type_name.startswith("Speck2f")
            ]
    
    # Check if devices are not found
    if len(devices) == 0:
        # Try checking for an already open device
        devices = [
            device
            for device in samna.device.get_opened_devices()
            if device.device_type_name.startswith("Speck2f")
            ]

    # default_config is a optional parameter of open_device
    default_config = samna.speck2fBoards.DevKitDefaultConfig()

    return samna.device.open_device(devices[0], default_config)

# Open the samna GUI process
def open_visualizer(streamer_endpoint, window_width=0.75, window_height=0.75):
    gui_process = multiprocessing.Process(
        target=samnagui.run_visualizer,
        args=(streamer_endpoint, window_width, window_height),
    )
    gui_process.start()

    return gui_process


def build_samna_event_route(graph, dk):
    # build a graph in samna to show dvs
    _, _, streamer = graph.sequential(
        [dk.get_model_source_node(), "Speck2fDvsToVizConverter", "VizEventStreamer"]
    )

    streamer.set_streamer_endpoint("tcp://0.0.0.0:40000")
    if streamer.wait_for_receiver_count() == 0:
        raise Exception(f'connecting to visualizer on {"tcp://0.0.0.0:40000"} fails')

    return streamer


def configure_visualizer(graph, streamer, num_channels):
    config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])
    #graph.start()
    
    visualizer_config = samna.ui.VisualizerConfiguration(
        # add plots to gui
        plots=[
            # add plot to show pixels
            samna.ui.ActivityPlotConfiguration(10, 10, "DVS Layer", [0, 0, 0.5, 0.75]),
            # add plot to show readout. params: plot title and images array of the same size of feature count. these images correspond to each feature.
            # samna.ui.ReadoutPlotConfiguration(
            #     "Readout Layer",
            #     [f"./vprtemponeuro/dataset/qcr/speck/test002/{name}.png" for name in image_names],
            #     [0.5, 0, 1, 0.8],
            # ),
            # add plot to show spike count. params: plot title and feature count and name of each feature
            samna.ui.SpikeCountPlotConfiguration(
                title="Spike Count",
                channel_count=num_channels,
                line_names=["Spike Count"],
                layout=[0.5, 0.375, 1, 0.75],
                show_x_span=25,
                label_interval=2.5,
                max_y_rate=1.2,
                show_point_circle=True,
                default_y_max=10,
            ),
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

# Find the argmax of the output spikes
def custom_readout(collection):
    def generate_result(feature):
        e = samna.ui.Readout()
        e.feature = feature
        return [e]

    # Preset total number of unique spike features
    total_features = 63
    # Initialize sum dictionary with all features set to 0
    sum = {f'{i}': 0 for i in range(0, total_features)}

    for spike in collection:
        if spike.feature in sum:
            sum[spike.feature] += 1
        else:
            sum[spike.feature] = 1

    # Find the key with the maximum value
    argmax_key = max(sum, key=sum.get)
    
    # Define the path for the JSON file
    json_file_path = 'spike_data.json'

    # Check if file exists and load existing data
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            # Load the existing data
            try:
                data_blocks = json.load(file)
                if not isinstance(data_blocks, list):
                    # Ensure data_blocks is a list, otherwise initialize it as an empty list
                    data_blocks = []
            except json.JSONDecodeError:
                # Handle cases where the file is empty or corrupted
                data_blocks = []
    else:
        data_blocks = []

    # Create a new block for the current data with a timestamp
    new_data_block = {
        'timestamp': timeit.timeit(),
        'data': sum
    }

    # Append the new data block to the list of blocks
    data_blocks.append(new_data_block)

    # Save the updated blocks back to the JSON file
    with open(json_file_path, 'w') as file:
        json.dump(data_blocks, file, indent=4)

    print(f"The key with the maximum value is: {argmax_key}")

    return generate_result(int(argmax_key))