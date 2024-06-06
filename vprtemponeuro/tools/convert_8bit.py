import os
from PIL import Image

def convert_to_8bit_png(folder_path):
    # Loop through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Open the image file
            with Image.open(file_path) as img:
                # Convert the image to 8-bit
                img_8bit = img.convert("P")
                
                # Save the image back to the same location
                img_8bit.save(file_path, format='PNG')

# Replace 'your_folder_path' with the path to your folder
convert_to_8bit_png('/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/qcr/speck/trolley-qry')
