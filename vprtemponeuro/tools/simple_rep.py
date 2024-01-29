'''
simple_rep.py - Simple representation of DVS events into timebinned frames.

Required: A .zip file containing .txt files of DVS events and event number count generated from extract_rosbag.py

This script generates extremely simplified visual representations of DVS events over a specified timebin, as determined by a
desired framerate. The generated frames lose temporal aspects of the data, but can be used for quick visual inspection of the
events.

User can specify the:

    - Start time offset --offset
    - Timebin (in fps) to generate the frames --timebin
    - Hot pixels file, which will eliminate hot pixels from the representation --hot_pixels
    
To create a video of the frames, use video_create.py.
'''

# Imports
import os
import cv2
import zipfile

import numpy as np
import pandas as pd

from tqdm import tqdm

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