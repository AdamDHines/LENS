import os
import shutil
import re

def natural_key(string):
    """ A key function for natural string sorting """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def move_and_rename_images(source_dir, target_dir, file_list_path, skip_count=0):
    # Read the list of image names from the file
    with open(file_list_path, 'r') as file:
        image_names = file.read().strip().split('\n')

    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Move and rename the images, skipping the specified number of images
    sorted_image_names = sorted(image_names, key=natural_key)
    for i, image_name in enumerate(sorted_image_names[skip_count:]):  # Skipping images at the start
        source_path = os.path.join(source_dir, image_name)
        if os.path.exists(source_path):
            new_name = f"image_{i:03}.png"
            target_path = os.path.join(target_dir, new_name)
            shutil.move(source_path, target_path)
        else:
            print(f"Image not found: {image_name}")

# Example usage
source_directory = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/20200424_151015-daytime_dvs_frames'
target_directory = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/query_dvs'
file_list = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/original_filenames/query_basefilenames.txt'
skip_images = 30  # Number of images to skip

move_and_rename_images(source_directory, target_directory, file_list, skip_images)