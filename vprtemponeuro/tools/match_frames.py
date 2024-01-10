import os
import shutil
import numpy as np
import pynmea2
import cv2  # for image processing

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

    return {'latitudes': latitudes, 'longitudes': longitudes, 'timestamps': timestamps}

def find_closest_gps_match(lat, lon, gps_data):
    # Find the closest match in GPS data
    closest_index = np.argmin([np.linalg.norm(np.array([lat, lon]) - np.array([gps_lat, gps_lon])) for gps_lat, gps_lon in zip(gps_data['latitudes'], gps_data['longitudes'])])
    return gps_data['timestamps'][closest_index]

def get_patches2D(image, patch_size):

    if patch_size[0] % 2 == 0: 
        nrows = image.shape[0] - patch_size[0] + 2
        ncols = image.shape[1] - patch_size[1] + 2
    else:
        nrows = image.shape[0] - patch_size[0] + 1
        ncols = image.shape[1] - patch_size[1] + 1
    return np.lib.stride_tricks.as_strided(image, patch_size + (nrows, ncols), image.strides + image.strides).reshape(patch_size[0]*patch_size[1],-1)


def patch_normalise_pad(image, patch_size):

    patch_size = (patch_size, patch_size)
    patch_half_size = [int((p-1)/2) for p in patch_size ]

    image_pad = np.pad(np.float64(image), patch_half_size, 'constant', constant_values=np.nan)

    nrows = image.shape[0]
    ncols = image.shape[1]
    patches = get_patches2D(image_pad, patch_size)
    mus = np.nanmean(patches, 0)
    stds = np.nanstd(patches, 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        out = (image - mus.reshape(nrows, ncols)) / stds.reshape(nrows, ncols)

    out[np.isnan(out)] = 0.0
    out[out < -1.0] = -1.0
    out[out > 1.0] = 1.0
    return out


def processImage(img, imWidth, imHeight, num_patches):

    img = cv2.resize(img,(imWidth, imHeight))
    #img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    im_norm = patch_normalise_pad(img, num_patches) 

    # Scale element values to be between 0 - 255
    img = np.uint8(255.0 * (1 + im_norm) / 2.0)

    return img

def cosine_similarity(image1, image2):
    imWidth, imHeight, num_patches = 28, 28, 7
    processed_image1 = processImage(image1, imWidth, imHeight, num_patches)
    processed_image2 = processImage(image2, imWidth, imHeight, num_patches)

    # Flatten the image arrays to one-dimensional vectors
    vector1 = processed_image1.flatten()
    vector2 = processed_image2.flatten()

    # Compute cosine similarity
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return similarity


def main(reference_video_path, query_video_path, reference_nmea_path, query_nmea_path, output_folder_ref, output_folder_query, n=5):
    if not os.path.exists(output_folder_ref):
        os.makedirs(output_folder_ref)
    if not os.path.exists(output_folder_query):
        os.makedirs(output_folder_query)

    # Extract GPS data
    reference_gps_data = get_gps(reference_nmea_path)
    query_gps_data = get_gps(query_nmea_path)

    if 'timestamps' in reference_gps_data and reference_gps_data['timestamps']:
        for i, timestamp in enumerate(reference_gps_data['timestamps']):
            # Generate frame name for reference video
            ref_frame_name = f"{timestamp}.000.png"
            ref_frame_path = os.path.join(reference_video_path, ref_frame_name)

            # Find matching frame in query video
            ref_lat, ref_lon = reference_gps_data['latitudes'][i], reference_gps_data['longitudes'][i]
            query_timestamp = find_closest_gps_match(ref_lat, ref_lon, query_gps_data)
            #query_timestamp = query_timestamp  # Add 1 second to account for the delay in the DVS data
            query_frame_name = f"{query_timestamp}.100.png"
            query_frame_path = os.path.join(query_video_path, query_frame_name)

            # Cosine similarity check
            best_similarity = -1
            best_match = query_frame_path
            ref_image = cv2.imread(ref_frame_path, cv2.IMREAD_GRAYSCALE)
            frame_increment = 0.025  # Increment per frame for a 40fps video
            for offset in range(-n, n+1):
                offset_time = frame_increment * offset
                temp_query_frame_name = f"{query_timestamp + offset_time:.3f}.png"
                temp_query_frame_path = os.path.join(query_video_path, temp_query_frame_name)
                if os.path.exists(temp_query_frame_path):
                    temp_query_image = cv2.imread(temp_query_frame_path, cv2.IMREAD_GRAYSCALE)
                    similarity = cosine_similarity(ref_image, temp_query_image)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = temp_query_frame_path

            # Copy and rename frames
            ref_output_path = os.path.join(output_folder_ref, f"image_{i:04d}.png")
            query_output_path = os.path.join(output_folder_query, f"image_{i:04d}.png")
            shutil.copy(ref_frame_path, ref_output_path)
            shutil.copy(best_match, query_output_path)
    else:
        print("No timestamp data available in reference GPS data.")

# Example usage
main("/home/adam/repo/rpg_e2vid/scripts/extracted_data/dvs_reference_sunset2", 
      "/home/adam/repo/rpg_e2vid/scripts/extracted_data/dvs_query_daytime", 
     "/home/adam/repo/rpg_e2vid/scripts/extracted_data/gps_reference_sunset2.nmea", 
     "/home/adam/repo/rpg_e2vid/scripts/extracted_data/gps_query_daytime.nmea", 
     "/home/adam/repo/rpg_e2vid/scripts/extracted_data/gps_matched_reference",
     "/home/adam/repo/rpg_e2vid/scripts/extracted_data/gps_matched_query")