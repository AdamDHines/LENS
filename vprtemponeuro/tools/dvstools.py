# Imports
import os
import cv2
import sys
import json
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

class FrameRep():
    def __init__(self,args):
        super(FrameRep, self).__init__()
        self.args = args
        self.event_preparation()

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
    
    def event_preparation(self):
        # Determine if the file is a Parquet or text file
        file_path_txt = os.path.join(self.args.dataset_folder, self.args.input_file + ".zip")
        file_path_parquet = os.path.join(self.args.dataset_folder, self.args.input_file + ".parquet")
        self.is_parquet = os.path.exists(file_path_parquet)
        self.file_path = file_path_parquet if self.is_parquet else file_path_txt

        # Reading camera dimensions and hot pixels
        print("Reading camera dimensions...")

        # Assuming hot pixels are only relevant for text files
        self.hot_pixels = None
        if not self.is_parquet:
            self.dimensions = self.read_camera_dimensions(self.file_path)
            self.dimensions = (self.dimensions[1], self.dimensions[0])  # Swap width and height
            print("Reading hot pixels...")
            hot_pixels_path = os.path.join(self.args.dataset_folder, self.args.hot_pixels + ".txt")
            if os.path.exists(hot_pixels_path):
                self.hot_pixels = self.read_hot_pixels(hot_pixels_path)
        else:
            self.dimensions = (260,346)

        if self.args.output_name == '':
            self.frame_folder = os.path.join(self.args.dataset_folder, self.args.input_file)
        else:
            self.frame_folder = os.path.join(self.args.dataset_folder, self.args.output_name)
        if not os.path.exists(self.frame_folder):
            os.makedirs(self.frame_folder)

        with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
            with zip_ref.open("event_sum.txt") as file:
                total_frames_bytes = file.readline()  # This is a byte string
                total_frames_str = total_frames_bytes.decode('utf-8').strip()  # Decode and strip whitespace/newline
                self.total_frames = int(total_frames_str)

        # If user set a maximum number of frames, use that instead
        if self.args.frames_max < self.total_frames and self.args.frame_limit:
            self.total_frames = self.args.frames_max

    def event_data(self):
        frame_interval = (1.0 / self.args.timebin)  # Time interval for each frame in seconds
        if self.args.offset != 0:
            start_timestamp = self.args.offset
            current_time = self.args.offset

        frame_data = np.zeros(self.args.pixels, dtype=np.uint8)  # Initialize frame data
        activity_duration = np.zeros(self.dimensions) # Initialize frame data

        pixel_activity = []
        spike_accum = 0
        plot = False
        if self.is_parquet:
            # Convert all timestamps from microseconds to seconds
            self.event_data['t'] = self.event_data['t'] / 1000000

            start_index = 0  # Start index for each frame's data
            num_rows = len(self.event_data)

            while start_index < num_rows:
                # Determine the end time for the current frame
                frame_end_time = current_time + frame_interval

                # Process events that belong to the current frame
                for index in range(start_index, num_rows):
                    timestamp = self.event_data.at[index, 't']
                    if timestamp >= frame_end_time:
                        break  # Move to the next frame

                    x, y = int(self.event_data.at[index, 'x']), int(self.event_data.at[index, 'y'])

                    # Skip hot pixels if provided
                    if self.hot_pixels is None or (x, y) not in self.hot_pixels:
                        frame_data[y, x] = 255

                yield frame_data
                frame_data.fill(0)  # Reset the frame data for the next frame

                # Update the start index and current time for the next frame
                start_index = index
                current_time = frame_end_time

        else:
            frame_number = 0
            last_update = np.zeros(self.dimensions)  # Initialize last update tracker

            with zipfile.ZipFile(self.file_path, 'r') as zip_ref:

                if self.args.reference:
                    # Generate a list of random unique indices within the flat array representation of the matrix
                    unique_indices = np.random.choice(self.dimensions[0] * self.dimensions[1], size=self.args.pixels, replace=False)

                    # Convert these flat indices to x,y coordinates
                    # coordinates = np.unravel_index(unique_indices, (self.dimensions[0], self.dimensions[1]))
                    centroid_coordinate_store = []
                    coordinate_store = []
                    # Initialize a dictionary to store centroids with patch coordinates as keys
                    centroid_coordinates_dict = {}

                    for centroid_index in unique_indices:
                        # Calculate row and column in matrix from flat index
                        centroid_row, centroid_col = divmod(centroid_index, self.dimensions[1])
                        centroid_coordinate_store.append(int(centroid_index))
                        # Calculate patch coordinates around the centroid
                        for row in range(centroid_row - 1, centroid_row + 2):
                            for col in range(centroid_col - 1, centroid_col + 2):
                                # Check if within bounds
                                if 0 <= row < self.dimensions[0] and 0 <= col < self.dimensions[1]:
                                    # Calculate flat index for the patch coordinate
                                    patch_coordinate = row * self.dimensions[1] + col
                                    coordinate_store.append(patch_coordinate)
                                    # Assign or reassign the patch coordinate to point to this centroid
                                    centroid_coordinates_dict[int(patch_coordinate)] = int(centroid_index)
                    # The result is a tuple of arrays, where the first array contains the x (row) coordinates
                    # and the second array contains the y (column) coordinates. To list them as pairs, you can zip them:
                    with open(os.path.join(self.args.dataset_folder, 'cooridnates_dict'+str(self.args.pixels)+'.json'), 'w') as handle:
                        json.dump(centroid_coordinates_dict, handle)
                    with open(os.path.join(self.args.dataset_folder, 'cooridnates_centroid'+str(self.args.pixels)+'.json'), 'w') as handle:
                        json.dump(centroid_coordinate_store, handle)
                    coordinates = np.unravel_index(coordinate_store, (self.dimensions[0], self.dimensions[1]))
                    coordinates_list = list(zip(coordinates[1], coordinates[0]))
                    coordinates_list_array = np.array(coordinates_list)
                    coordinates_array = []
                    # Assuming coordinates_list is already generated
                    for n, ndx in enumerate(coordinates_list):
                        coordinates_array.append(np.array([n]+list(ndx)))
                    np.savez_compressed(os.path.join(self.args.dataset_folder, self.args.input_file +str(self.args.pixels)+ '_coordinates.npz'), coordinates_list_array)
                    np.savez_compressed(os.path.join(self.args.dataset_folder, str(self.args.pixels)+'unique_indices.npz'), unique_indices)
                else:
                    with np.load(os.path.join(self.args.dataset_folder, 'sunset1'+str(self.args.pixels)+'_coordinates.npz')) as data:
                        # Assuming 'coordinates_array' is the key used to save the array
                        loaded_array = data['arr_0']
                    with np.load(os.path.join(self.args.dataset_folder, str(self.args.pixels)+'unique_indices.npz')) as data:
                        # Assuming 'coordinates_array' is the key used to save the array
                        unique_indices = data['arr_0']
                    coordinates_list = list(map(tuple, loaded_array))
                    coordinates_array = []
                    # Assuming coordinates_list is already generated
                    for n, ndx in enumerate(coordinates_list):
                        coordinates_array.append(np.array([n]+list(ndx)))

                    # Load dictionary back
                    with open(os.path.join(self.args.dataset_folder, 'cooridnates_dict'+str(self.args.pixels)+'.json'), 'r') as handle:
                        centroid_coordinates_dict = json.load(handle)
                    with open(os.path.join(self.args.dataset_folder, 'cooridnates_centroid'+str(self.args.pixels)+'.json'), 'r') as handle:
                        centroid_coordinate_store = json.load(handle)

                with zip_ref.open(self.args.input_file+".txt") as file:
                    next(file)  # Skip the first line (camera dimensions)
                    if self.args.frame_limit:
                        self.frame_bar = tqdm(total=self.total_frames, desc="Number of frames processed.")
                        self.event_bar = None
                    else:
                        self.event_bar = tqdm(total=self.total_frames, desc="Number of events processed.")
                        self.frame_bar = None
                    for line in file:
                        timestamp, x, y, pol = map(float, line.strip().split())
                        channel_index = int(pol)
                        x, y = int(x), int(y)

                        # Initialization of the offset and time tracking
                        if self.args.offset == 0:
                            self.args.offset = timestamp
                            start_timestamp = timestamp
                            current_time = timestamp

                        # Skip events before the start timestamp or hot pixels
                        if timestamp < start_timestamp or (self.hot_pixels and (x, y) in self.hot_pixels):
                            continue

                        # Incremental decay and accumulation
                        if abs(timestamp - current_time) <= frame_interval:
                            if self.args.tool == 'decay_rep':
                                # Calculate time since last update for the pixel
                                time_since_last_update = timestamp - last_update[y, x]

                                # Accumulate activity duration
                                activity_duration[y, x] += time_since_last_update

                                # Apply decay since last update
                                decay_factor = np.exp(-self.args.decay_factor * time_since_last_update)
                                frame_data[y, x] = frame_data[y, x] * decay_factor + self.args.accum_factor

                                last_update[y, x] = timestamp
                            elif self.args.tool == 'simple_rep':
                                if (x,y) in coordinates_list:
                                    # Iterate over the coordinates_array to find the matching x and y
                                    for _, x_targ, y_targ in coordinates_array:
                                        if x_targ == x and y_targ == y:
                                            coordinate = np.ravel_multi_index((y,x), self.dimensions)
                                            if not self.args.reference:
                                                coordinate = str(coordinate)
                                            index_coordinate = centroid_coordinates_dict[coordinate]
                                            index = np.where(unique_indices== index_coordinate)[0][0]
                                            # Found the matching x and y, now update frame_data at the found index
                                            frame_data[index] += self.args.accum_factor
                                            break 
                                    
                            if self.event_bar is not None:
                                self.event_bar.update(1)
                        else:
                            if self.args.tool == 'decay_rep':
                                # Calculate final decay based on activity duration
                                final_decay_factor = np.exp(-self.args.decay_factor * (frame_interval - activity_duration))
                                
                                # Apply final decay for this frame
                                frame_data = frame_data * final_decay_factor

                                # Reset activity duration tracker for the next frame
                                activity_duration = np.zeros(self.dimensions)

                            if not self.args.tool == 'event_profile':
                                self.save_frame(frame_data, frame_number, output_dir=self.frame_folder)

                            if not timestamp <= self.args.offset:
                                frame_number += 1
                            frame_data = np.zeros(self.args.pixels, dtype=np.uint8)
                            last_update = np.full(self.dimensions, current_time + frame_interval)
                            current_time = timestamp

                            #plt.show()
                            pixel_activity = []
                            spike_accum = 0
                            if frame_number >= self.args.frames_max and self.args.frame_limit:
                                self.frame_bar.close()
                                break
                                    # Update the progress bar, if any
                            if self.frame_bar is not None:
                                self.frame_bar.update(1)

    def save_frame(self, frame, frame_index, output_dir="frames"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        frame = np.reshape(frame, (int(np.sqrt(self.args.pixels)),int(np.sqrt(self.args.pixels))))

        filename = os.path.join(output_dir, f"images_{frame_index:05d}.png")  # Using :05d for zero-padded frame index
        
        # Save the RGB image
        cv2.imwrite(filename, frame)

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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_file = os.path.join(self.args.dataset_folder, self.args.input_file + ".mp4")
        video = cv2.VideoWriter(output_file, fourcc, self.args.timebin, (width, height))

        for frame_file in tqdm(frame_files,desc="Creating video from frames"):
            frame = cv2.imread(frame_file, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            video.write(frame)

        video.release()
        print(f"Video has been saved to {output_file}")