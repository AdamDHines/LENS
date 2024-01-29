import os

# Specify the directory path and the output text file name
directory_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/daytime'  # Replace with your directory path
output_file = '/home/adam/repo/Patch-NetVLAD/patchnetvlad/dataset_imagenames/bergb_imageNames.txt'

# Read filenames, sort them, and write them to a file
with open(output_file, 'w') as file:
    filenames = [filename for filename in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, filename))]
    for filename in sorted(filenames):
        file.write(filename + '\n')

print(f"Sorted filenames have been written to {output_file}")