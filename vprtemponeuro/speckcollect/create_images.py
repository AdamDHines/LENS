#MIT License

#Copyright (c) 2024 Adam Hines

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class Frames:
    def __init__(self, data_dir, exp_name, dims=[128, 128]):
        super().__init__()
        self.dims = dims
        self.data_dir = data_dir
        self.exp_name = exp_name
        self.events = np.load(os.path.join(data_dir, exp_name + '.npy'), allow_pickle=True).item()
        del self.events[next(iter(self.events))]

    def create_frames(self):
        '''
        events - speck2f::event::Spike data which contains 
                layer, feature (0 or 1), y, x, and timestamp
                
        time_interval - in msec, time to collect events over for a frame
        '''
        
        # Check if a hot pixels file exists
        hot_pixel_file = './speckcollect/files/Speck2fDevKit_hotpixels.txt'
        if os.path.exists(hot_pixel_file):
            hot_pixels = []
            with open(hot_pixel_file, "r") as file:
                for line in file:
                    x, y = line.strip().split(',')
                    hot_pixels.append((int(x), int(y)))
        else:
            print('No hot pixel file available, may be bad representation')

        output_dir = os.path.join(self.data_dir, self.exp_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Iterate through each dictionary input of timestamp
        for idx, timestamp in enumerate(tqdm(self.events, desc=f'Creating frames for {self.exp_name}...')):
            # Create new blank frame to accumulate events into
            frame = np.zeros((self.dims[0],self.dims[1]), dtype=int)
            # Iterate over events
            for event in self.events[timestamp]:
                # If a positive event, record it
                if (event.x, event.y) not in hot_pixels:
                    # Add events to frame coordinate
                    frame[event.x,event.y] += 1
            
            # Clip events in the range [0,255] for 8-bit image
            frame = np.clip(frame,0,255)
            frame = np.rot90(frame)
            # Save the frame as a PNG file
            frame_filename = os.path.join(output_dir, f"frame_{idx:05d}.png")
            plt.imsave(frame_filename, frame, cmap='gray')
            


    def identify_hot_pixels(self, events, output, name, dims=[128, 128]):
        """
        Identifies hot pixels that are active at every single timestamp.

        :param events: Dictionary of events with timestamps as keys.
        :param dims: Dimensions of the sensor array.
        :return: A set of tuples indicating the (x, y) coordinates of hot pixels.
        """
        # Initialize a 3D array to track activations across all frames
        activation_tracker = np.zeros((dims[0], dims[1], len(events)), dtype=int)
        
        # Process each timestamp
        for idx, (_, event_list) in enumerate(events.items()):
            # Mark activated pixels for the current timestamp
            for event in event_list:
                activation_tracker[event.x, event.y, idx] = 1
                    
        # Identify pixels that are active in all frames (hot pixels)
        hot_pixels = np.where(activation_tracker.min(axis=2) == 1)
        
        # Convert numpy arrays to a set of tuples for easier processing and storage
        hot_pixel_set = set(zip(hot_pixels[0], hot_pixels[1]))
        
        with open(os.path.join(output,name+".txt"), "w") as file:
            for pixel in hot_pixel_set:
                file.write(f"{pixel[0]},{pixel[1]}\n")
    
# # Define the data dir and traverse to create frames from
# data_dir = '/home/adam/Documents/'
# traverse = 'event_data'

# fullfile = os.path.join(data_dir, traverse + '.npy')

# # Load data
# data = np.load(fullfile, allow_pickle=True).item()

# # Remove first key entry since it's always blank
# del data[next(iter(data))]

# # Run hot pixel identification (comment out if not necessary)
# # hot_pixels = identify_hot_pixels(data, data_dir, traverse)

# # Run the frame creation
# create_frames(data, data_dir, traverse)