import samna
import time
import socket
from multiprocessing import Process
import samnagui

class Visualizer:
   def __init__(self):
      self.port = None
      self.sender_endpoint = ""
      self.receiver_endpoint = ""
      self.visualizer = None

      self.__init_samna()
      self.__start_visualizer_process(self.receiver_endpoint)

   def get_port(self):
      if self.port:
            return self.port
      free_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      free_socket.bind(("0.0.0.0", 0))
      free_socket.listen(5)
      self.port = free_socket.getsockname()[1]
      free_socket.close()
      return self.port

   def __init_samna(self):
      self.receiver_endpoint = "tcp://0.0.0.0:" + str(self.get_port())

   @staticmethod
   def __start_visualizer_process(receiver_endpoint):
      gui_process = Process(
            target=samnagui.run_visualizer,
            args=(receiver_endpoint, 0.75, 0.75),
      )
      gui_process.start()

      return gui_process

# Open your device
deviceInfos = samna.device.get_unopened_devices()
for info in deviceInfos:
   print(info)

print("opening device...")
board = samna.device.open_device(deviceInfos[0])
sw = board.get_stop_watch()
sw.set_enable_value(True)

# Start the visualizer, this will be done in a different process
visualizer = Visualizer()

graph = samna.graph.EventFilterGraph()
# Take the output of ImageReconstructFilter and stream it directly to the visualizer
board_source, reconstruct_filter, streamer = graph.sequential(
   [
      board.get_model_source_node(),
      "Speck2fDvsImageReconstructFilter",
      "VizEventStreamer",
   ],
)
# We can change the settings of the reconstruct filter based on the camera and the light conditions
reconstruct_filter.set_potential_limits(0, 0.3, 0)
reconstruct_filter.set_event_contribution(0.04)
reconstruct_filter.set_decay_factor(1e6)
reconstruct_filter.set_synchronous_decay(True)

# We can add another the dvs layer to see the single dvs events from the camera
graph.sequential([board.get_model_source_node(), "Speck2fDvsToVizConverter", streamer])

streamer.set_streamer_endpoint("tcp://0.0.0.0:" + str(visualizer.get_port()))
assert streamer.wait_for_receiver_count(1, 30000) == 1

# We need to apply the configuration we prefer to the streamer
source_node, streamer = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])

conf = samna.ui.VisualizerConfiguration()
conf.width_proportion = 1
conf.height_proportion = 1

plot1 = samna.ui.ActivityPlotConfiguration(128, 128, "DVS Layer")
plot1.layout = [0, 0, 0.5, 1]
plot2 = samna.ui.FramePlotConfiguration("Image reconstruct")
plot2.layout = [0.5, 0, 1, 1]
conf.plots = [plot1, plot2]
source_node.write([conf])

graph.start()
time.sleep(1)

# Configure the device
config = samna.speck2f.configuration.SpeckConfiguration()
config.dvs_layer.monitor_enable = True
config.factory_config.fast_output = True
config.factory_config.monitor_dual_channel = True
config.dvs_filter.enable = True
config.dvs_filter.filter_size.x = 3
config.dvs_filter.filter_size.y = 3
config.dvs_filter.threshold = 5
#config.dvs_layer.merge = True
config.dvs_filter.hot_pixel_filter_enable = True

board.get_io_module().set_dual_channel_output_enable(True)
board.get_model().apply_configuration(config)

print("started")