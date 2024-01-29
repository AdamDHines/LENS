import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ensure PyTorch is using the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# File reading generator function
def read_data(file_path):
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line (dimensions)
        for line in file:
            yield line

# Function to read hot pixels and transfer them to GPU
def read_hot_pixels(file_path):
    hot_pixels = set()
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(int, line.split(', '))
            hot_pixels.add((x, y))
    return hot_pixels

# Function to process a single line of data and update the matrix using PyTorch
def process_data_line_pytorch(line, matrix, hot_pixels, accumulation_value=1):
    _, x, y, _ = map(float, line.split())
    y, x = int(x), int(y)
    if (y, x) not in hot_pixels:
        matrix[x, y] += accumulation_value

# Function to apply decay using PyTorch
def apply_decay_pytorch(matrix, decay_factor=0.95):
    matrix *= decay_factor

# Initialize the matrix using PyTorch and transfer it to GPU
matrix_dim = (260, 346)
matrix = torch.zeros(matrix_dim, device=device)

# Read hot pixels
hot_pixels_file_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/dvs_vpr_2020-04-22-17-24-21_hot_pixels.txt'
hot_pixels = read_hot_pixels(hot_pixels_file_path)

# Read data file
data_file_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/dvs_vpr_2020-04-22-17-24-21.txt'
data_generator = read_data(data_file_path)

# Create a custom colormap with black as zero
colors = [(0, 0, 0)] + [(plt.cm.rainbow(i)) for i in range(1, 256)]
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_rainbow", colors, N=256)

# Initialize variables for frame generation
frame_interval = 0.033333  # 30 fps
last_frame_time = 0
frame_count = 0

# Process the first line to set the initial last_frame_time
for line in data_generator:
    last_frame_time, _, _, _ = map(float, line.split())
    if frame_count == 0:
        break

# Process data and generate frames
for line in data_generator:
    timestamp, _, _, _ = map(float, line.split())
    
    # Process line with PyTorch
    process_data_line_pytorch(line, matrix, hot_pixels, accumulation_value=1)

    # Apply decay using PyTorch
    apply_decay_pytorch(matrix, decay_factor=0.95)  # Adjust decay_factor as needed

    # Save a frame at a 30fps rate
    if timestamp - last_frame_time >= frame_interval:
        # Transfer matrix to CPU for saving as an image
        frame_matrix_np = matrix.to("cpu").numpy()
        plt.imsave(f'/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/test_accum/frame_{frame_count:05d}.png', frame_matrix_np, cmap='magma')
        last_frame_time += frame_interval
        frame_count += 1

print(f"Generated {frame_count} frames.")

# Function to generate the video using image files
def generate_video(image_folder, output_video_file, fps=30):
    import imageio
    images = []
    for i in range(frame_count):
        filename = f"{image_folder}/frame_{i:05d}.png"
        images.append(imageio.imread(filename))
    imageio.mimsave(output_video_file, images, fps=fps)

# Paths for frames and output video
image_folder = 'path_to_frames_folder'
output_video_file = 'output_video.mp4'

# Generate the video
generate_video(image_folder, output_video_file, fps=30)

print("Video generated successfully.")
