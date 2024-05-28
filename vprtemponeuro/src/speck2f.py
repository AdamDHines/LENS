import samna, samnagui
import multiprocessing
import time
from queue import Queue

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


def configure_visualizer(graph, streamer):
    config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])
    graph.start()
    
    visualizer_config = samna.ui.VisualizerConfiguration(
        # add plots to gui
        plots=[
            # add plot to show pixels
            samna.ui.ActivityPlotConfiguration(64, 64, "DVS Layer", [0, 0, 0.5, 0.75]),
            # add plot to show readout. params: plot title and images array of the same size of feature count. these images correspond to each feature.

            # add plot to show spike count. params: plot title and feature count and name of each feature
            samna.ui.SpikeCountPlotConfiguration(
                title="Spike Count",
                channel_count=79,
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

# a custom python callback that receives an array of spike events as input and outputs an array of readout events. you are encouraged to customize it.
def custom_readout(collection):
    def generate_result(feature):
        e = samna.ui.Readout()
        e.feature = feature
        return [e]
    
    sum = {}
    for spike in collection:
        if spike.feature in sum:
            sum[spike.feature] += 1
        else:
            sum[spike.feature] = 1

    # Find the key with the maximum value
    argmax_key = max(sum, key=sum.get)

    print(f"The key with the maximum value is: {argmax_key}")

    return generate_result(argmax_key)
        


def start_visualizer():
    def event_collector(event_queue):
        while gui_process.is_alive():
            events = sink.get_events()  # Collect events
            if events:  # Check if there are any events to process
                event_queue.put(events)  # Put the entire batch of events into the queue as a single item
            time.sleep(1.0)


            def event_analyzer(event_queue):
                while True:
                    events_batch = event_queue.get()  # Get a batch of events
                    # Evaluate how long it takes on average to get unique indexes
                    start = time.perf_counter()
                    counter = events_batch.count
                    idx = []
                    for ev in events_batch:
                        coord = ev.x * ev.y
                        if coord in self.coords:
                            idx.append(counter(ev))
                    print(f"Time to get unique indexes: {time.perf_counter() - start}")
                    event_queue.task_done()
            streamer_endpoint = "tcp://0.0.0.0:40000"
            gui_process = Process(
                target=samnagui.run_visualizer, args=(streamer_endpoint, 0.75, 0.75)
            )
            gui_process.start()
            dk = open_speck2f_dev_kit()

            graph = samna.graph.EventFilterGraph()

            _, dvs_spike_select, _, streamer = graph.sequential(
                [
                    dk.get_model_source_node(),
                    "Speck2fOutputMemberSelect",
                    "Speck2fDvsToVizConverter",
                    "VizEventStreamer",
                ]
            )
            dvs_spike_select.set_white_list([13],"layer")

            streamer.set_streamer_endpoint(streamer_endpoint)
            if streamer.wait_for_receiver_count() == 0:
                raise Exception(f"connecting to visualizer on {streamer_endpoint} fails")

            (
                _,
                readout_spike_select,
                spike_collection_filter,
                spike_count_filter,
                _,
            ) = graph.sequential(
                [
                    dk.get_model_source_node(),
                    "Speck2fOutputMemberSelect",
                    "Speck2fSpikeCollectionNode",
                    "Speck2fSpikeCountNode",
                    streamer,
                ]
            )

            readout_spike_select.set_white_list(
                [13], "layer"
            )  # output pixels are in the format of spike events, so we need to filter out all pixel spikes.

            spike_collection_filter.set_interval_milli_sec(
                500
            )  # divide according to this time period in milliseconds.

            spike_count_filter.set_feature_count(self.args.reference_places)
            power = dk.get_power_monitor()
            power.start_auto_power_measurement(20)
            graph.sequential([power.get_source_node(), "MeasurementToVizConverter", streamer])
            _, readout_filter, _ = graph.sequential(
                [spike_collection_filter, "Speck2fCustomFilterNode", streamer]
            )  # from spike collection to streamer
            readout_filter.set_filter_function(custom_readout)

            #config_source = build_samna_event_route(graph, dk)

            # sink = samna.graph.sink_from(dk.get_model().get_source_node())
            # Configuring the visualizer

            config_source.write([visualizer_config])
            # Modify configuration to enable DVS event monitoring
            # config = samna.speck2f.configuration.SpeckConfiguration()
            # config.dvs_layer.monitor_enable = True
            dk.get_model().apply_configuration(self.config)

            event_queue = Queue()
            # Start the event collector thread
            collector_thread = threading.Thread(target=event_collector, args=(event_queue,))
            collector_thread.start()

            # # Start the event analyzer thread
            analyzer_thread = threading.Thread(target=event_analyzer, args=(event_queue,))
            analyzer_thread.start()
            
            # Wait until the visualizer window destroys
            gui_process.join()

            # Stop the graph and ensure the collector thread is also stopped
            readout_filter.stop()
            graph.stop()
            #collector_thread.join()

            print('Event collection stopped.')