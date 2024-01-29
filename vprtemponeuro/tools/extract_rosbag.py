'''
Function adapted from https://github.com/uzh-rpg/rpg_e2vid ./scripts/extract_events_from_rosbag.py

Extracts events from a rosbag file and saves them as a text file. The text file is then compressed into a zip file.
'''

# Imports
import os
import sys
import rosbag
import zipfile

from tqdm import tqdm
from os.path import basename

class ExtractRosbag():
    def __init__(self, args):
        super(ExtractRosbag, self).__init__()
        self.args = args

    def timestamp_str(self,ts):
        t = ts.secs + ts.nsecs / float(1e9)
        return '{:.12f}'.format(t)


    def run_extract(self):
        # Check for existence of file
        if not os.path.exists(os.path.join(self.args.dataset_folder, self.args.input_file)):
            print("File does not exist.")
            sys.exit()

        print('Data will be extracted as folder: {}'.format(self.args.dataset_folder))

        width, height = None, None
        event_sum = 0

        if self.args.output_name == '':
            output_name = os.path.basename(self.args.input_file).split('.')[0]  # /path/to/mybag.bag -> mybag
        else:
            output_name = self.args.output_name
        path_to_events_file = os.path.join(self.args.dataset_folder, '{}.txt'.format(output_name))

        print('Extracting events to {}...'.format(path_to_events_file))
        
        event_topic = '/dvs/events'

        with open(path_to_events_file, 'w') as events_file:

            with rosbag.Bag(os.path.join(self.args.dataset_folder,self.args.input_file), 'r') as bag:

                # Look for the topics that are available and save the total number of messages for each topic (useful for the progress bar)
                total_num_event_msgs = 0
                topics = bag.get_type_and_topic_info().topics
                for topic_name, topic_info in topics.items():
                    if topic_name == event_topic:
                        total_num_event_msgs = topic_info.message_count
                        print('Found events topic: {} with {} messages'.format(topic_name, topic_info.message_count))
                total_messages = bag.get_message_count()
                # Extract events to text file
                for topic, msg, t in tqdm(bag.read_messages(), total=total_messages):
                    if topic == event_topic:
                        if width is None:
                            width = msg.width
                            height = msg.height
                            print('Found sensor size: {} x {}'.format(width, height))
                            events_file.write("{} {}\n".format(width, height))
                        for e in msg.events:
                            events_file.write(self.timestamp_str(e.ts) + " ")
                            events_file.write(str(e.x) + " ")
                            events_file.write(str(e.y) + " ")
                            events_file.write(("1" if e.polarity else "0") + "\n")
                            event_sum += 1

            # statistics
            print('All events extracted!')
            print('Events:', event_sum)

        # Zip text file
        print('Compressing text file...')
        path_to_events_zipfile = os.path.join(self.args.dataset_folder, '{}.zip'.format(output_name))
        path_to_event_sum_file = os.path.join(self.args.dataset_folder, 'event_sum.txt')

        # Write the event_sum to the event_sum.txt file
        with open(path_to_event_sum_file, 'w') as f:
            f.write(str(event_sum))

        with zipfile.ZipFile(path_to_events_zipfile, 'w') as zip_file:
            zip_file.write(path_to_events_file, basename(path_to_events_file), compress_type=zipfile.ZIP_DEFLATED)
            # Add the second text file (event_sum.txt)
            zip_file.write(path_to_event_sum_file, basename(path_to_event_sum_file), compress_type=zipfile.ZIP_DEFLATED)
        print('Finished!')

        # Remove events.txt
        if os.path.exists(path_to_events_file):
            os.remove(path_to_events_file)
            os.remove(path_to_event_sum_file)
            print('Removed {}.'.format(path_to_events_file))

        print('Done extracting events!')
