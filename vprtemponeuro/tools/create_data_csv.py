import os
import csv
import numpy as np
from read_gps import get_gps

def create_csv_from_images(folder_path, csv_file_path, gps_path=None, fps=30):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Get GPS coordinates
    if gps_path is not None:
        gps = get_gps(gps_path)

    # Filter out non-PNG files and sort
    png_files = sorted([f for f in files if f.endswith('.png')])

    # Create and write data to CSV
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        if gps_path is not None:
            # Write column headers
            writer.writerow(['Image_name','index', 'gps_coordinate'])

            # Write image names and GPS coordinates
            time_interval = 1/fps
            time_counter = 0
            gps_index = 0
            for index, image_name in enumerate(png_files):
                time_counter += time_interval
                writer.writerow([image_name, index, [gps[gps_index][0],gps[gps_index][1]]])
                try:
                    if time_counter >= gps[gps_index+1][2]:
                        gps_index += 1
                except:
                    pass

        else:
            # Write column headers
            writer.writerow(['Image name', 'index'])

            # Write image names and indices
            for index, image_name in enumerate(png_files):
                writer.writerow([image_name, index])

# Example usage
folder_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/sunset2' # Replace with your folder path
csv_file_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/sunset2_gps.csv' # Path for the CSV file
gps_path = '/media/adam/vprdatasets/data/Brisbane-Event-VPR/20200422_172431-sunset2_concat.nmea'
create_csv_from_images(folder_path, csv_file_path, gps_path=gps_path)
