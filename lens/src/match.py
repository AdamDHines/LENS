import numpy as np
from metrics import createPR, recallAtK
from prettytable import PrettyTable
import os
from skimage import io
import torch
import torch.nn as nn
import csv
from scipy.ndimage import binary_dilation
import sys
sys.path.append('/Users/adam/repo/LENS/lens/tools')
from plot_results import plot_PR, plot_recall

# Select datafolder to load the LENS data from
base_dir = '/Users/adam/repo/LENS/lens/data/Figure5'
subfolder = '220724-16-14-33'
output = 'output'
# make the output directory
outputdir = os.path.join(base_dir, subfolder, output)
if not os.path.exists(outputdir):
    os.makedirs(outputdir, exist_ok=True)

# Load the similarity matrix
data = np.load(os.path.join(base_dir, subfolder, 'similarity_matrix.npy'),allow_pickle=True)
data = data.T
# Load the rows to remove and remove them from the similarity matrix
row_remove = np.load(os.path.join(base_dir, subfolder, 'removed_indexes.npy'),allow_pickle=True)
data = np.delete(data, row_remove, axis=1)

# Load the equivelant images for SAD
sad_query = "query-images"
sad_refernce = "reference-images"

# Load the query and reference image names from .csv
query_files = []
reference_files = []
with open(f'{base_dir}/{subfolder}/query_files.csv', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        query_files.append(row)
with open(f'{base_dir}/{subfolder}/reference_files.csv', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        reference_files.append(row)
query_files = query_files[0]
reference_files = reference_files[0]

# Define the convolutional kernel for SAD
def _init_kernel():
    kernel = torch.zeros(1, 1, 8, 8)
    centre_coordinate = (8 // 2) - 1
    kernel[0, 0, centre_coordinate, centre_coordinate] = 1  # Set the center pixel to 1
    return kernel

# Define the Conv2d selection layer
conv = nn.Conv2d(1, 1, kernel_size=8, stride=8, padding=0, bias=False)
conv.weight = nn.Parameter(_init_kernel(), requires_grad=False) # Set the kernel weights

# Load the images and convolve for SAD
def load_and_preprocess_images(file, conv):
    img = io.imread(file)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    img = conv(img).squeeze().numpy()    
    img = img.flatten()
    return img

# Load and preprocess images from both folders
images1 = [load_and_preprocess_images(os.path.join(base_dir, subfolder, sad_query, file), conv) for file in query_files]
images2 = [load_and_preprocess_images(os.path.join(base_dir, subfolder, sad_refernce, file), conv) for file in reference_files]
# convert to numpy array
images1 = np.array(images1)
images2 = np.array(images2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = torch.from_numpy(images1.reshape(images1.shape[0], -1).astype(np.float32)).unsqueeze(0).to(device)
b = torch.from_numpy(images2.reshape(images2.shape[0], -1).astype(np.float32)).unsqueeze(0).to(device)

# Calculate the SAD distance matrix
sequence_length = 4
torch_dist = torch.cdist(a, b, 1)[0]
dist_tensor = torch_dist.clone().detach().to(device).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
precomputed_convWeight = torch.eye(sequence_length, device=device).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
import torch.nn.functional as F

# 3. Perform convolution without padding
conv_output = F.conv2d(dist_tensor, precomputed_convWeight, padding=0)  # Shape: (1, 1, H_out, W_out)
conv_output = 1/conv_output
# 4. Calculate desired output dimensions
H, W = torch_dist.shape  # Original dimensions
K = sequence_length  # Kernel size

# 5. Compute output dimensions after convolution
H_out = conv_output.shape[2]
W_out = conv_output.shape[3]

# 6. Define desired output size (same as original)
H_desired, W_desired = H, W

# 7. Calculate required padding
pad_h = H_desired - H_out
pad_w = W_desired - W_out

# Ensure that padding is non-negative
if pad_h < 0 or pad_w < 0:
    raise ValueError("Kernel size is too large, resulting in negative padding.")

# 8. Distribute padding on top/bottom and left/right
pad_top = pad_h // 2
pad_bottom = pad_h - pad_top
pad_left = pad_w // 2
pad_right = pad_w - pad_left

# 9. Apply zero padding to the convolved output
# F.pad expects padding in the order: (pad_left, pad_right, pad_top, pad_bottom)
padded_conv_output = F.pad(
    conv_output,
    pad=(pad_left, pad_right, pad_top, pad_bottom) # Explicitly set padding value to 0
)

# 10. Post-process the result: remove singleton dimensions, move to CPU, convert to NumPy, and normalize
dist_matrix_seq = padded_conv_output.squeeze().cpu().numpy() / sequence_length

#dist_matrix_seq = dist_matrix_seq.T


# define database and query places
database_places = a.shape[1]
query_places = b.shape[1]

# Create the GT matrix with ones down the diagonal
GT = np.eye(min(database_places, query_places), database_places, dtype=int)
GT_sad = GT[sequence_length-2:-1,sequence_length-2:-1]

def create_GTtol(GT, distance=2):
    """
    Creates a ground truth matrix with vertical tolerance by manually adding 1s
    above and below the original 1s up to the specified distance.
    
    Parameters:
    - GT (numpy.ndarray): The original ground truth matrix.
    - distance (int): The maximum number of rows to add 1s above and below the detected 1s.
    
    Returns:
    - GTtol (numpy.ndarray): The modified ground truth matrix with vertical tolerance.
    """
    # Ensure GT is a binary matrix
    GT_binary = (GT > 0).astype(int)
    
    # Initialize GTtol with zeros
    GTtol = np.zeros_like(GT_binary)
    
    # Get the number of rows and columns
    num_rows, num_cols = GT_binary.shape
    print(num_rows, num_cols)
    
    # Iterate over each column
    for col in range(num_cols):
        # Find the indices of rows where GT has a 1 in the current column
        ones_indices = np.where(GT_binary[:, col] == 1)[0]
        
        # For each index with a 1, set 1s in GTtol within the specified vertical distance
        for row in ones_indices:
            # Determine the start and end rows, ensuring they are within bounds
            start_row = max(row - distance, 0)
            end_row = min(row + distance + 1, num_rows)  # +1 because upper bound is exclusive
            
            # Set the range in GTtol to 1
            GTtol[start_row:end_row, col] = 1
    
    return GTtol

# Create GTsoft with a customizable number of rows to add
GTtol = create_GTtol(GT, distance=2)
GTtol_sad = create_GTtol(GT_sad, distance=2)
import matplotlib.pyplot as plt
plt.imshow(GTtol)
plt.savefig(os.path.join(outputdir, 'GTtol.pdf'), dpi=300)

# Create PR curve
P, R = createPR(data, GTtol, outputdir, matching='single', n_thresh=100)
P_sad, R_sad = createPR(dist_matrix_seq.T, GTtol.T, outputdir, datatype="SAD", matching='single', n_thresh=100)

# Combine P and R into a list of lists
PR_data = {
        "Precision": P,
        "Recall": R
    }
PR_data_sad = {
        "Precision": P_sad,
        "Recall": R_sad
    }

# Plot the PR curve
plot_PR(PR_data, PR_data_sad, outputdir)

# Recall@N
N = [1,5,10,15,20,25] # N values to calculate
R = [] # Recall@N values
# Calculate Recall@N
for n in N:
    R.append(round(recallAtK(data,GTtol,K=n),2))
# Print the results
table = PrettyTable()
table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])
print(table)

R_sad = []
for n in N:
    R_sad.append(round(recallAtK(dist_matrix_seq.T,GThard=GTtol.T,K=n),2))
# Print the results
table = PrettyTable()
table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
table.add_row(["Recall", R_sad[0], R_sad[1], R_sad[2], R_sad[3], R_sad[4], R_sad[5]])
print(table)

# Plot Recall@N
plot_recall(R, R_sad, N, outputdir)