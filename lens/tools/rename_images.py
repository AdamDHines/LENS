import os
import re

def natural_sort_key(s):
    """
    A key function for natural (human-like) sorting of strings with numbers.
    It converts the string into a list of strings and integers.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# Specify the directory path
directory_path = '/home/adam/repo/LENS/lens/dataset/qcr/speck/indoor-query'  # Replace with your directory path

# Read filenames and sort them using the natural sort key
filenames = sorted(os.listdir(directory_path), key=natural_sort_key)

# Rename files
for index, filename in enumerate(filenames):
    index = index
    # Create the new file name
    new_filename = f'images_{index:05}.png'
    old_file_path = os.path.join(directory_path, filename)
    new_file_path = os.path.join(directory_path, new_filename)

    # Check if it's a file before renaming
    if os.path.isfile(old_file_path):
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {filename} to {new_filename}")

print("Renaming completed.")
