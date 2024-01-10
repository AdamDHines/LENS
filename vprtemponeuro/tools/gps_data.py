import numpy as np
import pynmea2
import os
import shutil


def get_gps(nmea_file_path):
    nmea_file = open(nmea_file_path, encoding='utf-8')

    latitudes, longitudes, timestamps = [], [], []

    first_timestamp = None
    previous_lat, previous_lon = 0, 0

    for line in nmea_file.readlines():
        try:
            msg = pynmea2.parse(line)
            if first_timestamp is None:
                first_timestamp = msg.timestamp
            if msg.sentence_type not in ['GSV', 'VTG', 'GSA']:
                # print(msg.timestamp, msg.latitude, msg.longitude)
                # print(repr(msg.latitude))
                dist_to_prev = np.linalg.norm(np.array([msg.latitude, msg.longitude]) - np.array([previous_lat, previous_lon]))
                if msg.latitude != 0 and msg.longitude != 0 and msg.latitude != previous_lat and msg.longitude != previous_lon and dist_to_prev > 0.0001:
                    timestamp_diff = (msg.timestamp.hour - first_timestamp.hour) * 3600 + (msg.timestamp.minute - first_timestamp.minute) * 60 + (msg.timestamp.second - first_timestamp.second)
                    latitudes.append(msg.latitude); longitudes.append(msg.longitude); timestamps.append(timestamp_diff)
                    previous_lat, previous_lon = msg.latitude, msg.longitude

        except pynmea2.ParseError as e:
            # print('Parse error: {} {}'.format(msg.sentence_type, e))
            continue

    return np.array(np.vstack((latitudes, longitudes, timestamps))).T

def get_video_frame_timestamps(frame_rate, total_frames):
    return np.arange(0, total_frames) * (1 / frame_rate)

def find_matching_gps_points(gps_data1, gps_data2):
    matched_indices = []
    for i, (lat1, lon1, _) in enumerate(gps_data1):
        # Find the closest point in the second video's GPS data
        distances = np.sqrt((gps_data2[:, 0] - lat1)**2 + (gps_data2[:, 1] - lon1)**2)
        closest_index = np.argmin(distances)
        if distances[closest_index] < 0.000045 :  # Threshold for "close enough"
            matched_indices.append((i, closest_index))
    return matched_indices

def copy_and_rename_frames(source_folder, dest_folder, frame_indices, prefix):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for i, frame_index in enumerate(frame_indices):
        source_file_name = f"{frame_index}.000.png"  # Updated to match the frame index directly
        dest_file_name = f"{prefix}{i:04d}.png"
        source_path = os.path.join(source_folder, source_file_name)
        dest_path = os.path.join(dest_folder, dest_file_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
        else:
            print(f"File not found: {source_path}")

from skimage.io import imread
from skimage.transform import resize
from numpy.linalg import norm

def cosine_similarity(img1, img2):


    # Flatten the images to turn them into 1D arrays
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # Calculate cosine similarity
    similarity = np.dot(img1_flat, img2_flat) / (norm(img1_flat) * norm(img2_flat))
    return similarity

def find_best_match(base_img_path, search_folder, base_index, search_range, frame_rate):
    base_img = imread(base_img_path)
    highest_similarity = -1
    best_match_index = None

    for offset in range(-search_range, search_range + 1):
        frame_index = base_index + offset
        test_img_path = f"{search_folder}/{format_frame_index(frame_index, frame_rate)}.png"
        if os.path.exists(test_img_path):
            test_img = imread(test_img_path)
            similarity = cosine_similarity(base_img, test_img)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_index = frame_index

    return best_match_index

def format_frame_index(frame_index, frame_rate):
    timestamp = frame_index * (1 / frame_rate)
    return "{:.3f}".format(timestamp).rstrip('0').rstrip('.')
# Main logic
frame_rate = 40  # Frames per second
source_folder_video1 = "/home/adam/repo/rpg_e2vid/scripts/extracted_data/dvs_reference_sunset2"
source_folder_video2 = "/home/adam/repo/rpg_e2vid/scripts/extracted_data/dvs_query_sunrise"
dest_folder_video1 = "/home/adam/repo/rpg_e2vid/scripts/extracted_data/gps_matched_reference"
dest_folder_video2 = "/home/adam/repo/rpg_e2vid/scripts/extracted_data/gps_matched_query"
nmea_file_path_video1 = "/home/adam/repo/rpg_e2vid/scripts/extracted_data/gps_reference_sunset2.nmea"
nmea_file_path_video2 = "/home/adam/repo/rpg_e2vid/scripts/extracted_data/gps_query_sunrise.nmea"

# Read GPS data for both videos
gps_data_video1 = get_gps(nmea_file_path_video1)
gps_data_video2 = get_gps(nmea_file_path_video2)

# Get video frame timestamps
total_frames_video1 = len(os.listdir(source_folder_video1))
total_frames_video2 = len(os.listdir(source_folder_video2))
video_timestamps_video1 = get_video_frame_timestamps(frame_rate, total_frames_video1)
video_timestamps_video2 = get_video_frame_timestamps(frame_rate, total_frames_video2)

# Find matching GPS points between the two videos
matched_gps_indices = find_matching_gps_points(gps_data_video1, gps_data_video2)

search_range = 25  # Define the range for searching similar frames

matched_frames_video1 = [i for i, _ in matched_gps_indices]
matched_frames_video2 = []

for i, (index_video1, _) in enumerate(matched_gps_indices):
    base_img_path = f"{source_folder_video1}/{format_frame_index(index_video1, frame_rate)}.png"
    if not os.path.exists(base_img_path):
        print(f"Base image not found: {base_img_path}")
        continue

    index_video2 = matched_gps_indices[i][1]
    best_match_index = find_best_match(base_img_path, source_folder_video2, index_video2, search_range, frame_rate)
    if best_match_index is not None:
        matched_frames_video2.append(best_match_index)
    else:
        print(f"No match found for base image: {base_img_path}")

# Copy and rename frames to new folders
copy_and_rename_frames(source_folder_video1, dest_folder_video1, [i for i, _ in matched_gps_indices], "images")
copy_and_rename_frames(source_folder_video2, dest_folder_video2, [j for _, j in matched_gps_indices], "images")