'''
Imports
'''
import os
import csv

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
    def __init__(self, model, event_type="variance", max_pixels=25):
        super(EventStats, self).__init__()

        # Define the model parameters
        self.model = model
        self.event_type = event_type
        self.max_pixels = max_pixels

    def load_images_from_folder(self,folder=None):
        self.images = []
        for filename in tqdm(sorted(os.listdir(folder)), desc="Loading Images"):
            if filename.endswith(".png"):
                img_path = os.path.join(folder, filename)
                with Image.open(img_path) as img:
                    # Convert to grayscale
                    gray_img = img.convert('L')
                    width, height = gray_img.size
                    cropped_img = gray_img.crop((0, 0, width, height - 60))
                    self.images.append(cropped_img.copy())

    def calculate_pixel_frequency(self):
        if not self.images:
            return None, None

        sample_image = self.images[0]
        width, height = sample_image.size
        total_pixels = width * height
        self.freq = np.zeros(total_pixels)
        self.pixel_count = np.zeros(total_pixels)

        for img in tqdm(self.images, desc="Calculating Pixel Frequency"):
            img_data = np.array(img).flatten()
            indices = np.arange(total_pixels)
            self.freq += np.bincount(indices, weights=img_data, minlength=total_pixels)
            self.pixel_count += (img_data > 0)

    def find_most_active_pixels(self):
        if self.event_type == "max":
            hot_pixels = self.pixel_count == len(self.images)
            dead_pixels = self.pixel_count == 0
            active_pixels = np.where(~hot_pixels & ~dead_pixels, self.freq, 0)
            self.selection = np.unravel_index(np.argsort(-active_pixels.ravel())[:self.max_pixels], active_pixels.shape)
            np.save('./dataset/pixel_selection.npy',self.selection[0])
        elif self.event_type == "variance":
            num_images = len(self.images)

            # Calculate the probability of each pixel being active
            p_active = self.pixel_count / num_images

            # Calculate variance for each pixel
            variance = p_active * (1 - p_active)

            # Find the indices of pixels with the highest variance
            self.selection = np.unravel_index(np.argsort(-variance.ravel())[:self.max_pixels], variance.shape)
            np.save('./dataset/pixel_selection.npy',self.selection[0])
        
    def reconstruct_images(self,output_folder):
        os.makedirs(output_folder, exist_ok=True)

        reconstructed_images = []  # List to store the reconstructed images
        csv_data = []  # List to store data for CSV

        # Get the image dimensions
        height, width = self.images[0].size

        for i, img in enumerate(self.images):
            new_img = np.zeros((height, width), dtype=np.uint8)

            # Get the grayscale channel as an array
            grayscale_channel = np.array(img)

            # Set the top active pixels
            for idx in self.selection:
                y, x = divmod(idx, width)
                new_img[y, x] = grayscale_channel[x, y]

            new_img = new_img.T  # Transpose should be outside the loop

            # Save the new image
            save_path = os.path.join(output_folder, f'image_{i:04d}.png')
            reconstructed_image = Image.fromarray(new_img)
            reconstructed_image.save(save_path)

            reconstructed_images.append(reconstructed_image)  # Add to the list
            # Add image name and index to CSV data
            csv_data.append([f'image_{i:04d}.png', i])

        print(f"Images saved to {output_folder}")
        # Save the CSV file
        csv_file_path = os.path.join('./dataset', self.model.dataset + '.csv')
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
        self.load_images_from_folder(folder=self.model.trainingPath+self.model.location)
        # Calculate pixel frequency
        self.calculate_pixel_frequency()
        # Calculate pixel activity, based on type of activity wanted to measure
        self.find_most_active_pixels()
        # Reconstruct images for both query sets
        self.reconstruct_images('./dataset/database_filtered')
        self.load_images_from_folder(folder=self.model.trainingPath+self.model.test_locations)
        self.reconstruct_images('./dataset/query_filtered')