'''
Imports
'''
import os
import csv
import shutil
import math

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image

class EventStats:
    '''
    event_type - defines the event statistic to calculate to send to VPRTempo
    Current event_type options:
        - "variance" - find the top most variable event pixels
        - "max" - find the top most active event pixels
    max_pixels - defines the number of pixels to send to VPRTempo
    '''
    def __init__(self, model, event_type="max", max_pixels=784, total_patches=16):
        super(EventStats, self).__init__()

        # Define the model parameters
        self.model = model
        self.event_type = event_type
        self.max_pixels = max_pixels
        # Calculate the maximum number of patches that can fit into max_pixels
        max_patches = self.calculate_max_patches(max_pixels, total_patches)

        # Adjust total_patches if it exceeds the limit
        self.total_patches = min(total_patches, max_patches)

    def calculate_max_patches(self, max_pixels, total_patches):
        # Assuming square patches for simplicity
        patch_size = int(math.sqrt(max_pixels / total_patches))

        # Calculate how many patches can fit into max_pixels
        patches_per_row = int(math.sqrt(max_pixels)) // patch_size
        max_patches = patches_per_row ** 2

        return max_patches

    def load_images_from_folder(self,folder=None):
        self.images = []
        for filename in tqdm(sorted(os.listdir(folder)), desc="Loading Images"):
            if filename.endswith(".png"):
                img_path = os.path.join(folder, filename)
                with Image.open(img_path) as img:
                    # Convert to grayscale
                    gray_img = img.convert('L')
                    width, height = gray_img.size
                    cropped_img = gray_img.crop((0, 0, width - 2, height - 60))
                    self.images.append(cropped_img.copy())

    
    def split_into_patches(self, image):
        width, height = image.size
        self.image_width = width
        self.patches = []
        self.patch_info = []

        def closest_factors(total_patches):
            # Find factors of total_patches
            factors = [(i, total_patches // i) for i in range(1, int(math.sqrt(total_patches)) + 1) if total_patches % i == 0]

            # Choose the pair where the difference between factors is minimized
            return min(factors, key=lambda x: abs(x[0] - x[1]))

        # Calculate the number of patches in each dimension
        patches_per_row, patches_per_col = closest_factors(self.total_patches)

        # Calculate the basic dimensions of each patch
        basic_patch_width = width // patches_per_row
        basic_patch_height = height // patches_per_col

        # Calculate the number of pixels remaining after the basic division
        extra_width = width % patches_per_row
        extra_height = height % patches_per_col

        for i in range(patches_per_col):
            for j in range(patches_per_row):
                # Determine the actual size of this patch
                patch_width = basic_patch_width + (1 if j < extra_width else 0)
                patch_height = basic_patch_height + (1 if i < extra_height else 0)

                # Calculate the position of the patch
                left = sum(basic_patch_width + (1 if x < extra_width else 0) for x in range(j))
                upper = sum(basic_patch_height + (1 if y < extra_height else 0) for y in range(i))
                right = left + patch_width
                lower = upper + patch_height

                # Create the patch and add it to the list
                patch = image.crop((left, upper, right, lower))
                self.patches.append(patch)
                
                 # Store patch information
                self.patch_info.append((left, upper, patch_width, patch_height))

    def calculate_pixel_frequency(self):
        # Initialize dictionaries to hold frequency and pixel count for each patch
        self.freq_per_patch = {}
        self.pixel_count_per_patch = {}

        for img in tqdm(self.images, desc="Calculating Pixel Frequency"):
            self.split_into_patches(img)  # Assuming split_into_patches returns patches and patch info
            for patch_idx, patch in enumerate(self.patches):
                patch_data = np.array(patch).flatten()
                total_pixels = patch_data.size

                if patch_idx not in self.freq_per_patch:
                    self.freq_per_patch[patch_idx] = np.zeros(total_pixels)
                    self.pixel_count_per_patch[patch_idx] = np.zeros(total_pixels)

                patch_freq = np.bincount(np.arange(total_pixels), weights=patch_data, minlength=total_pixels)
                patch_pixel_count = (patch_data > 0)

                # Accumulate the data for this patch across all images
                self.freq_per_patch[patch_idx] += patch_freq
                self.pixel_count_per_patch[patch_idx] += patch_pixel_count

    def find_most_active_pixels(self):
        total_patches = self.total_patches  # Total number of patches
        patch_active_pixels = {}

        # Calculate the number of pixels to select from each patch
        pixels_per_patch = self.max_pixels // total_patches
        if self.max_pixels % total_patches != 0:
            pixels_per_patch += 1  # Adjust if not divisible evenly

        for patch_index in range(total_patches):
            patch_freq = self.freq_per_patch[patch_index]
            patch_pixel_count = self.pixel_count_per_patch[patch_index]

            # Identify hot and dead pixels in the patch
            hot_pixels = patch_pixel_count == len(self.images)
            dead_pixels = patch_pixel_count == 0

            # Calculate active pixels for the patch
            patch_active = np.where(~hot_pixels & ~dead_pixels, patch_freq, 0)
            top_indices = np.argsort(-patch_active)[:pixels_per_patch]
            patch_active_pixels[patch_index] = top_indices

        # Convert local patch indices to global indices
        self.selection = []
        for patch_index, active_indices in patch_active_pixels.items():
            patch_info = self.patch_info[patch_index]
            patch_start_x, patch_start_y, patch_width, patch_height = patch_info

            for local_idx in active_indices:
                # Calculate the local row and column within the patch
                local_row = local_idx // patch_width
                local_col = local_idx % patch_width

                # Convert to global row and column
                global_row = patch_start_y + local_row
                global_col = patch_start_x + local_col

                # Convert to linear index if necessary
                global_idx = global_row * self.image_width + global_col
                self.selection.append(global_idx)

        self.selection = np.array(self.selection)
        np.save('./vprtemponeuro/dataset/pixel_selection.npy',self.selection)

    def random_pixels(self):
        """
        Selects a random number of pixel indices from the image and saves them to a file.
        The number of pixels is determined by self.max_pixels.
        """

        # Assuming the class has attributes for image dimensions and number of pixels to select
        # Get the dimensions of the first image
        # Get the dimensions of the first image
        if len(self.images) > 0:
            first_image = self.images[0]
            image_width, image_height = first_image.size
        else:
            raise ValueError("No images available in self.images")
        num_pixels = self.max_pixels

        # Calculate total number of pixels in the image
        total_pixels = image_height * image_width

        # Generate random indices
        self.selection = np.random.choice(total_pixels, num_pixels, replace=False)

        # Save the selection to a file
        np.save('./vprtemponeuro/dataset/pixel_selection.npy', self.selection)

    def clear_output_folder(self, output_folder):
        """
        Clears all contents of the output folder if it exists.
        """
        if os.path.exists(output_folder):
            # Remove all files in the directory
            for file in os.listdir(output_folder):
                file_path = os.path.join(output_folder, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))   

    def reconstruct_images(self,output_folder):
        self.clear_output_folder(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        reconstructed_images = []  # List to store the reconstructed images
        csv_data = []  # List to store data for CSV

        # Get the image dimensions
        height, width = self.images[0].size

        for i, img in enumerate(self.images):
            new_img = np.zeros((width, height), dtype=np.uint8)

            # Get the grayscale channel as an array
            grayscale_channel = np.array(img)

            # Set the top active pixels
            for idx in self.selection:
                y, x = divmod(idx, width)
                new_img[x, y] = grayscale_channel[x, y]

            #new_img = new_img.T  # Transpose should be outside the loop

            # Save the new image
            save_path = os.path.join(output_folder, f'image_{i:04d}.png')
            reconstructed_image = Image.fromarray(new_img)
            reconstructed_image.save(save_path)

            reconstructed_images.append(reconstructed_image)  # Add to the list
            # Add image name and index to CSV data
            csv_data.append([f'image_{i:04d}.png', i])

        print(f"Images saved to {output_folder}")
        # Save the CSV file
        csv_file_path = os.path.join('./vprtemponeuro/dataset', self.model.dataset + '.csv')
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image_name', 'Index'])  # Write the header
            writer.writerows(csv_data)  # Write the data

        print(f"CSV file saved to {csv_file_path}")

    #def plot_frequency(self):
    #        plt.figure()
    #        plt.bar(range(len(distribution)), distribution)
    #        plt.title(title)
    #        plt.xlabel('Unique Pixel Index')
    #        plt.ylabel('Frequency')
    #        plt.show()
            
    #def plot_most_active_pixels(self):
    #    blank_image = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255
    #    width = image_size[0]

        # Prepare a figure and axes for custom drawing
    #    fig, ax = plt.subplots()
    #    ax.imshow(blank_image, cmap='gray')

        # Convert linear indices to 2D coordinates
    #    y_coords, x_coords = np.divmod(self.selection, width)
    #    index = np.array([x_coords[0],y_coords[0]]).T
        # Customizing dot appearance
    #    dot_color = 'red'  # Red color for highlighting pixels
    #    dot_size = 10  # Increase this value for larger dots

    #    ax.scatter(x_coords, y_coords, c=dot_color, s=dot_size)

    #    plt.title('Top 100 Most Active Pixels')
    #    plt.show()

    def main(self):
        # Load the images
        self.load_images_from_folder(folder=self.model.data_dir+'database_dvs')
        # Calculate pixel frequency
        self.calculate_pixel_frequency()
        # Calculate pixel activity, based on type of activity wanted to measure
        if self.event_type == "max":
            self.find_most_active_pixels()
        else:
            self.random_pixels()
        # Reconstruct images for both query sets
        self.reconstruct_images(self.model.data_dir+'/database_filtered')
        self.load_images_from_folder(folder=self.model.data_dir+'query_dvs')
        self.reconstruct_images(self.model.data_dir+'/query_filtered')