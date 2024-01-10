import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import fastparquet

def read_camera_dimensions(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        dimensions = tuple(map(int, first_line.split()))
    return dimensions

def read_hot_pixels(file_path):
    hot_pixels = set()
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(int, line.strip().split(','))
            hot_pixels.add((x, y))
    return hot_pixels

def read_parquet_file(file_path):
    df = pd.read_parquet(file_path,engine='fastparquet')
    return df

def process_event_data(event_data, dimensions, start_timestamp, hot_pixels=None, fps=30, from_parquet=False):
    frame_interval = 1.0 / fps  # Time interval for each frame in seconds
    #start_timestamp = start_timestamp / 1000000  # Convert to seconds
    current_time = start_timestamp
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
        # Processing data from a text file
        with open(event_data, 'r') as file:
            next(file)  # Skip the first line (camera dimensions)
            for line in file:
                timestamp, x, y, _ = map(float, line.strip().split())
                
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
                    yield frame_data
                    frame_data = np.zeros(dimensions, dtype=np.uint8)
                    current_time = timestamp

def save_frame(frame, frame_index, output_dir="frames"):
    frame_time = frame_index / 30.0  # Calculate frame time for 40fps
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"{frame_time:.3f}.png")  # Using .3f for milliseconds precision
    cv2.imwrite(filename, frame)

def create_video_from_frames(frame_folder, output_file, fps=30):
    frame_files = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.png')]
    
    if not frame_files:
        raise ValueError("No frames found in the specified folder.")

    # Sort the files based on the numerical value in the filename
    frame_files.sort(key=lambda x: float(x.split('/')[-1].replace('.png', '')))

    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        raise ValueError("Failed to read the first frame. Ensure the frame files are correct.")

    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    if not video.isOpened():
        raise ValueError("Failed to open the video writer. Check the codec and file path.")

    print("Adding frames to the video...")
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"Warning: Failed to read frame {frame_file}")
            continue
        video.write(frame)

    video.release()
    print("Video has been saved to", output_file)

def get_first_and_last_timestamp(file_path, is_parquet=False):
    if is_parquet:
        # Reading from a Parquet file
        df = pd.read_parquet(file_path)
        first_timestamp = df['t'].iloc[0]
        last_timestamp = df['t'].iloc[-1]
    else:
        # Reading from a text file
        with open(file_path, 'r') as file:
            next(file)  # Skip the first line (camera dimensions)
            first_timestamp = float(next(file).split()[0])  # Read the first timestamp

        # Open the file in binary mode to support relative seeks from the end
        with open(file_path, 'rb') as file:
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            buffer_size = 1024

            while True:
                if file_size < buffer_size:
                    file.seek(0)
                    break
                else:
                    file.seek(-buffer_size, os.SEEK_END)
                lines = file.readlines()
                if len(lines) > 1:
                    break
                buffer_size += 1024

            last_line = lines[-1].decode()
            last_timestamp = float(last_line.strip().split()[0])

    return first_timestamp, last_timestamp

event_folder = "/home/adam/repo/rpg_e2vid/scripts/extracted_data/"
event_data_file = "dvs_query_daytime"

# Determine if the file is a Parquet or text file
file_path_txt = os.path.join(event_folder, event_data_file + ".txt")
file_path_parquet = os.path.join(event_folder, event_data_file + ".parquet")
is_parquet = os.path.exists(file_path_parquet)
file_path = file_path_parquet if is_parquet else file_path_txt

# Reading camera dimensions and hot pixels
print("Reading camera dimensions...")

# Assuming hot pixels are only relevant for text files
hot_pixels = None
if not is_parquet:
    camera_dimensions = read_camera_dimensions(file_path)
    camera_dimensions = (camera_dimensions[1], camera_dimensions[0])  # Swap width and height
    print("Reading hot pixels...")
    hot_pixels_path = os.path.join(event_folder, event_data_file + "_hot_pixels.txt")
    if os.path.exists(hot_pixels_path):
        hot_pixels = read_hot_pixels(hot_pixels_path)
else:
    camera_dimensions = (260,346)

# Process event data based on file type
print("Processing event data and generating frames...")
first_timestamp, last_timestamp = get_first_and_last_timestamp(file_path,is_parquet=is_parquet)
if is_parquet:
    first_timestamp = first_timestamp / 1000000  # Convert to seconds
    last_timestamp = last_timestamp / 1000000  # Convert to seconds
start_timestamp = 1587705130.80
frame_folder = os.path.join(event_folder, event_data_file)
frame_interval = 1.0 / 30  # Time interval for each frame in seconds
total_frames = int((last_timestamp - start_timestamp) / frame_interval)

frame_number = 0
progress_bar = tqdm(total=total_frames, desc="Generating frames")

if is_parquet:
    event_data = read_parquet_file(file_path)
    for frame in process_event_data(event_data, camera_dimensions, start_timestamp, hot_pixels, from_parquet=True):
        save_frame(frame, frame_number, frame_folder)
        frame_number += 1
        progress_bar.update(1)
        # Check if the current frame number matches the total frames
        if frame_number >= total_frames:
            break
else:
    for frame in process_event_data(file_path, camera_dimensions, start_timestamp, hot_pixels):
        save_frame(frame, frame_number, frame_folder)
        frame_number += 1
        progress_bar.update(1)

progress_bar.close()
print("Frames generation complete. Creating video...")
output_video_file = os.path.join(event_folder, event_data_file + ".mp4")
create_video_from_frames(frame_folder, output_video_file)
print("Video creation complete.")
