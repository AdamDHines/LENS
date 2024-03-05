import os
import shutil
import re
import csv

def copy_images(source_folder, destination_folder, filenames_csv):
    """
    Copy image files listed in a .csv file from a source folder to a destination folder.
    Specifically extracts filenames that start with 'images_' from the first column under the heading 'Image_name'
    and copies them, with the numeric part reduced by 40.

    Parameters:
    - source_folder (str): The path to the source folder where image files are located.
    - destination_folder (str): The path to the destination folder where image files will be copied to.
    - filenames_csv (str): The path to the .csv file containing the image filenames to be copied.
    """

    # Ensure the destination folder exists, create if it doesn't
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Define a regular expression pattern to match 'images_' followed by digits and '.png'
    filename_pattern = r'images_(\d+)\.png'

    # Open and read the .csv file containing the filenames
    with open(filenames_csv, mode='r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            image_name = row['Image_name']  # Access the image name from the 'Image_name' column
            # Find all filenames in the image name that match the pattern
            matches = re.findall(filename_pattern, image_name)

            # Proceed if a matching filename is found
            for match in matches:
                original_number = int(match)  # Convert the matched number part to an integer
                new_number = max(0, original_number - 20)  # Subtract 40 from the number, ensuring it doesn't go below 0
                new_filename = f'images_{new_number:05}.png'  # Construct new filename with updated number

                # Construct the full source path to the file
                source_path = os.path.join(source_folder, f'images_{original_number:05}.png')  # Use original_number for source path

                # Construct the full destination path to the new filename
                destination_path = os.path.join(destination_folder, new_filename)

                # Check if the file exists before attempting to copy
                if os.path.exists(source_path):
                    # Copy the file to the destination folder
                    shutil.copy2(source_path, destination_path)
                    print(f"Copied '{source_path}' to '{destination_path}'.")
                else:
                    print(f"File '{source_path}' does not exist in '{source_folder}'.")

# Example usage
source_folder = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/brisbane_event/gopro/sunset1'  # Change to your source folder path
destination_folder = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/brisbane_event/gopro/sunset1_reference'  # Change to your destination folder path
filenames_txt = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/sunset1_reference.csv'  # Change to your .txt file path

# Uncomment the following line to run the function with the example paths
copy_images(source_folder, destination_folder, filenames_txt)