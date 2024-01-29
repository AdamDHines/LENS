# Imports
import os
import cv2
import sys
import torch
import rosbag
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tqdm import tqdm
from os.path import basename


class ExtractRosbag():
    '''
    Function adapted from https://github.com/uzh-rpg/rpg_e2vid ./scripts/extract_events_from_rosbag.py

    Extracts events from a rosbag file and saves them as a text file. The text file is then compressed into a zip file.
    '''
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

class SimpleRep():
    def __init__(self,args):
        super(SimpleRep, self).__init__()
        self.args = args

    def read_camera_dimensions(self,file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            with zip_ref.open(self.args.input_file+".txt") as file:
                first_line = file.readline()
                dimensions = tuple(map(int, first_line.split()))
            print('Camera dimensions: {} x {}'.format(dimensions[0], dimensions[1]))
            return dimensions

    def read_hot_pixels(self,file_path):
        hot_pixels = set()
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(int, line.strip().split(','))
                hot_pixels.add((x, y))
        return hot_pixels

    def read_parquet_file(self,file_path):
        df = pd.read_parquet(file_path,engine='fastparquet')
        return df

    def process_event_data(self,event_data, dimensions, hot_pixels=None, from_parquet=False):
        frame_interval = 1.0 / self.args.timebin  # Time interval for each frame in seconds

        if self.args.offset != 0:
            start_timestamp = self.args.offset
            current_time = self.args.offset

        frame_data = np.zeros(dimensions, dtype=np.uint8)  # Initialize frame data

        if from_parquet:
            # Convert all timestamps from microseconds to seconds
            event_data['t'] = event_data['t'] / 1000000

            start_index = 0  # Start index for each frame's data
            num_rows = len(event_data)

            while start_index < num_rows:
                # Determine the end time for the current frame
                frame_end_time = current_time + frame_interval

                # Process events that belong to the current frame
                for index in range(start_index, num_rows):
                    timestamp = event_data.at[index, 't']
                    if timestamp >= frame_end_time:
                        break  # Move to the next frame

                    x, y = int(event_data.at[index, 'x']), int(event_data.at[index, 'y'])

                    # Skip hot pixels if provided
                    if hot_pixels is None or (x, y) not in hot_pixels:
                        frame_data[y, x] = 255

                yield frame_data
                frame_data.fill(0)  # Reset the frame data for the next frame

                # Update the start index and current time for the next frame
                start_index = index
                current_time = frame_end_time

        else:
            frame_number = 0
            # Processing data from a zipped text file
            with zipfile.ZipFile(event_data, 'r') as zip_ref:
                with zip_ref.open(self.args.input_file+".txt") as file:
                    next(file)  # Skip the first line (camera dimensions)
                    for line in file:
                        self.progress_bar.update(1)
                        timestamp, x, y, _ = map(float, line.strip().split())

                        if self.args.offset == 0:
                            self.args.offset = timestamp
                            start_timestamp = timestamp
                            current_time = timestamp

                        # Skip events before the start timestamp
                        if timestamp < start_timestamp:
                            continue

                        x, y = int(x), int(y)

                        # Skip hot pixels if provided
                        if hot_pixels and (x, y) in hot_pixels:
                            continue

                        # Check if the current event belongs to the current frame
                        if timestamp - current_time <= frame_interval:
                            frame_data[y, x] = 255
                        else:
                            self.save_frame(frame_data, frame_number, output_dir=self.frame_folder)
                            frame_number += 1
                            frame_data = np.zeros(dimensions, dtype=np.uint8)
                            current_time = timestamp

    def save_frame(self,frame, frame_index, output_dir="frames"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, f"images_{frame_index:05d}.png")  # Using .3f for milliseconds precision
        cv2.imwrite(filename, frame)


    def simple_representation(self):
        # Determine if the file is a Parquet or text file
        file_path_txt = os.path.join(self.args.dataset_folder, self.args.input_file + ".zip")
        file_path_parquet = os.path.join(self.args.dataset_folder, self.args.input_file + ".parquet")
        is_parquet = os.path.exists(file_path_parquet)
        file_path = file_path_parquet if is_parquet else file_path_txt

        # Reading camera dimensions and hot pixels
        print("Reading camera dimensions...")

        # Assuming hot pixels are only relevant for text files
        hot_pixels = None
        if not is_parquet:
            camera_dimensions = self.read_camera_dimensions(file_path)
            camera_dimensions = (camera_dimensions[1], camera_dimensions[0])  # Swap width and height
            print("Reading hot pixels...")
            hot_pixels_path = os.path.join(self.args.dataset_folder, self.args.hot_pixels + ".txt")
            if os.path.exists(hot_pixels_path):
                hot_pixels = self.read_hot_pixels(hot_pixels_path)
        else:
            camera_dimensions = (260,346)

        # Process event data based on file type
        print("Processing event data and generating frames...")

        self.frame_folder = os.path.join(self.args.dataset_folder, self.args.input_file)
        if not os.path.exists(self.frame_folder):
            os.makedirs(self.frame_folder)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            with zip_ref.open("event_sum.txt") as file:
                total_frames_bytes = file.readline()  # This is a byte string
                total_frames_str = total_frames_bytes.decode('utf-8').strip()  # Decode and strip whitespace/newline
                total_frames = int(total_frames_str)

        frame_number = 0
        self.progress_bar = tqdm(total=total_frames, desc="Number of events processed.")

        if is_parquet:
            event_data = self.read_parquet_file(file_path)
            for frame in self.process_event_data(event_data, camera_dimensions, hot_pixels, from_parquet=True):
                self.save_frame(frame, frame_number, self.frame_folder)
                frame_number += 1
                # Check if the current frame number matches the total frames
                if frame_number >= total_frames:
                    break
        else:
            self.process_event_data(file_path, camera_dimensions, hot_pixels)

        self.progress_bar.close()
        print("Simple representation generation complete.")

class DecayRep():
    def __init__(self,args):
        super(DecayRep, self).__init__()
        self.args = args
        # Ensure PyTorch is using the GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File reading generator function
    def read_data(self,file_path):
        with open(file_path, 'r') as file:
            next(file)  # Skip the first line (dimensions)
            for line in file:
                yield line

    # Function to read hot pixels and transfer them to GPU
    def read_hot_pixels(file_path):
        hot_pixels = set()
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(int, line.split(', '))
                hot_pixels.add((x, y))
        return hot_pixels

    # Function to process a single line of data and update the matrix using PyTorch
    def process_data_line_pytorch(self,line, matrix, hot_pixels, accumulation_value=1):
        _, x, y, _ = map(float, line.split())
        y, x = int(x), int(y)
        if (y, x) not in hot_pixels:
            matrix[x, y] += accumulation_value

    # Function to apply decay using PyTorch
    def apply_decay_pytorch(self,matrix, decay_factor=0.95):
        matrix *= decay_factor

    # Initialize the matrix using PyTorch and transfer it to GPU
    matrix_dim = (260, 346)
    matrix = torch.zeros(matrix_dim, device=self.device)

    # Read hot pixels
    hot_pixels_file_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/dvs_vpr_2020-04-22-17-24-21_hot_pixels.txt'
    hot_pixels = read_hot_pixels(hot_pixels_file_path)

    # Read data file
    data_file_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/dvs_vpr_2020-04-22-17-24-21.txt'
    data_generator = read_data(data_file_path)

    # Create a custom colormap with black as zero
    colors = [(0, 0, 0)] + [(plt.cm.rainbow(i)) for i in range(1, 256)]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_rainbow", colors, N=256)

    # Initialize variables for frame generation
    frame_interval = 0.033333  # 30 fps
    last_frame_time = 0
    frame_count = 0

    # Process the first line to set the initial last_frame_time
    for line in data_generator:
        last_frame_time, _, _, _ = map(float, line.split())
        if frame_count == 0:
            break

    # Process data and generate frames
    for line in data_generator:
        timestamp, _, _, _ = map(float, line.split())
        
        # Process line with PyTorch
        process_data_line_pytorch(line, matrix, hot_pixels, accumulation_value=1)

        # Apply decay using PyTorch
        apply_decay_pytorch(matrix, decay_factor=0.95)  # Adjust decay_factor as needed

        # Save a frame at a 30fps rate
        if timestamp - last_frame_time >= frame_interval:
            # Transfer matrix to CPU for saving as an image
            frame_matrix_np = matrix.to("cpu").numpy()
            plt.imsave(f'/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/test_accum/frame_{frame_count:05d}.png', frame_matrix_np, cmap='magma')
            last_frame_time += frame_interval
            frame_count += 1

    print(f"Generated {frame_count} frames.")


class CreateVideo():
    def __init__(self, args):
        super(CreateVideo, self).__init__()
        self.args = args

    def create_video_from_frames(self):
        frame_files = [os.path.join(self.args.dataset_folder,self.args.input_file, f) for f in os.listdir(os.path.join(self.args.dataset_folder,self.args.input_file)) if f.endswith('.png')]
        
        if not frame_files:
            raise ValueError("No frames found in the specified folder.")

        frame_files.sort(key=lambda x: int(x.split('/')[-1].split('_')[-1].replace('.png', '')))

        # Explicitly read the first frame in color
        first_frame = cv2.imread(frame_files[0], cv2.IMREAD_COLOR)
        if first_frame is None:
            raise ValueError("Failed to read the first frame.")

        height, width, layers = first_frame.shape

        # Using 'XVID' codec for AVI format
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_file = os.path.join(self.args.dataset_folder, self.args.input_file + ".avi")
        video = cv2.VideoWriter(output_file, fourcc, self.args.timebin, (width, height))

        for frame_file in tqdm(frame_files,desc="Creating video from frames"):
            frame = cv2.imread(frame_file, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            video.write(frame)

        video.release()
        print(f"Video has been saved to {output_file}")