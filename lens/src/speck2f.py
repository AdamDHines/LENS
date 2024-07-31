import samna, samnagui
from multiprocessing import Process

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
def open_visualizer(streamer_endpoint, window_width=0.75, window_height=0.75, headless=False):
    gui_process = Process(
        target=samnagui.run_visualizer,
        args=(streamer_endpoint, window_width, window_height, headless:=headless),
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


def configure_visualizer(graph, streamer):
    config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])
    #graph.start()
    
    visualizer_config = samna.ui.VisualizerConfiguration(
        # add plots to gui
        plots=[
            # add plot to show pixels
            samna.ui.ActivityPlotConfiguration(10, 10, "DVS Layer", [0, 0, 1.0, 1.0]),
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