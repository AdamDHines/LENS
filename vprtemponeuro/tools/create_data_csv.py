import os
import csv

def create_csv_from_images(folder_path, csv_file_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter out non-PNG files and sort
    png_files = sorted([f for f in files if f.endswith('.png')])

    # Create and write data to CSV
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write column headers
        writer.writerow(['Image name', 'index'])

        # Write image names and indices
        for index, image_name in enumerate(png_files):
            writer.writerow([image_name, index])

# Example usage
folder_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/query' # Replace with your folder path
csv_file_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/brisbane_event_rgb.csv' # Path for the CSV file
create_csv_from_images(folder_path, csv_file_path)
