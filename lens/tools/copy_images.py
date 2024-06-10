import os
import shutil
import re
import csv


def copy_images(source_folder, destination_folder, filenames_csv, interval=30):
    """
    Copy every nth image file (every 30th by default) from a source folder to a destination folder based on a list in a CSV file.
    The function specifically extracts filenames that start with 'images_' from the first column under the heading 'Image_name',
    modifies the numeric part by reducing it by 20, and copies only every 30th file based on the sorted order.

    Parameters:
    - source_folder (str): The path to the source folder where image files are located.
    - destination_folder (str): The path to the destination folder where the image files will be copied to.
    - filenames_csv (str): The path to the CSV file containing the image filenames to be copied.
    - interval (int): The interval at which files are copied, defaulting to every 30th file.
    """

    # Ensure the destination folder exists, create if it doesn't
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Define a regular expression pattern to match 'images_' followed by digits and '.png'
    filename_pattern = r'images_(\d+)\.png'

    # Open and read the CSV file containing the filenames
    with open(filenames_csv, mode='r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        all_filenames = []
        for row in csvreader:
            image_name = row['Image_name']
            matches = re.findall(filename_pattern, image_name)
            all_filenames.extend(matches)

    # Sort filenames naturally by numeric part
    sorted_filenames = sorted(all_filenames, key=lambda x: int(x))

    # Select every 30th file from the sorted list
    filenames_to_process = [filename for index, filename in enumerate(sorted_filenames) if (index + 1) % interval == 0]

    for match in filenames_to_process:
        original_number = int(match)
        new_number = max(0, original_number - 0)  # Subtract 20 from the number
        new_filename = f'images_{new_number:05}.png'

        # Construct the full source and destination paths
        source_path = os.path.join(source_folder, f'images_{original_number:05}.png')
        destination_path = os.path.join(destination_folder, new_filename)

        # Copy the file if it exists
        if os.path.exists(source_path):
            shutil.copy2(source_path, destination_path)
            print(f"Copied '{source_path}' to '{destination_path}'.")
        else:
            print(f"File '{source_path}' does not exist.")

# Example usage parameters
source_folder = '/home/adam/Downloads/test001'
destination_folder = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/qcr/speck/ref'
filenames_csv = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/test001.csv'
skip_images = 30  # Number of images to skip

# Uncomment to run the function with the example paths
copy_images(source_folder, destination_folder, filenames_csv, skip_images)
