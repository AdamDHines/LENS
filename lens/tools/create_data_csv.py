import os
import csv
import numpy as np
from lens.tools.read_gps import get_gps

def haversine(lon1, lat1, lon2, lat2):
    # Radius of the Earth in kilometers
    R = 6371.0
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # Difference in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance * 1000  # Convert to meters

def create_csv_from_images(folder_path, csv_file_path, gps_path=None, fps=60, distance_threshold=100):
    files = os.listdir(folder_path)
    png_files = sorted([f for f in files if f.endswith('.png')])

    if gps_path is not None:
        gps = get_gps(gps_path)

    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        if gps_path is not None:
            writer.writerow(['Image_name', 'index', 'gps_coordinate'])
        else:
            writer.writerow(['Image_name', 'index'])

        if gps_path is not None:
            time_interval = 1 / fps
            time_counter = 0
            gps_index = 0
            subset_index = 0
            last_written_gps = None

            for index, image_name in enumerate(png_files):
                time_counter += time_interval
                gps_coord = [gps[gps_index][0], gps[gps_index][1]]
                writer.writerow([image_name, index, gps_coord])

                if last_written_gps is None or haversine(last_written_gps[1], last_written_gps[0], gps_coord[1], gps_coord[0]) > distance_threshold:
                    subset_index += 1
                    last_written_gps = gps_coord

                try:
                    if time_counter >= gps[gps_index+1][2]:
                        gps_index += 1
                except IndexError:
                    pass

        else:
            for index, image_name in enumerate(png_files):
                writer.writerow([image_name, index])

#create_csv_from_images('/home/adam/repo/LENS/lens/dataset/qcr/speck/indoor-reference', 
#                       '/home/adam/repo/LENS/lens/dataset/indoor-reference.csv', 
#                       gps_path=None)
