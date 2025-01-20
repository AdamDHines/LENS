import os
import numpy as np
from skimage import io
import csv

import re
import torch

from lens.src.metrics import recallAtK, createPR

from prettytable import PrettyTable
import matplotlib.pyplot as plt   
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_and_preprocess_images(csv_file,folder_dir):
    images = []
    # load images based on the .csv file names in the first column index
    csv_names = csv.reader(open(csv_file))
    files = [row[0] for row in csv_names]
    # remove first row
    files = files[1:]
    for _, filename in enumerate(files):

        if filename.endswith('.png'):
            img = io.imread(os.path.join(folder_dir, filename))    
            images.append(img.flatten()) 
    return np.array(images)

def run_sad(reference, reference_dir, query, query_dir, GT, GTtol, outputdir, sequence_length):

    # Load and preprocess images from both folders

    # Track progress for both folders
    images1 = load_and_preprocess_images(query, query_dir)
    images2 = load_and_preprocess_images(reference, reference_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = torch.from_numpy(images1.reshape(images1.shape[0], -1).astype(np.float32)).unsqueeze(0).to(device)
    b = torch.from_numpy(images2.reshape(images2.shape[0], -1).astype(np.float32)).unsqueeze(0).to(device)

    # Track progress for calculating distance
    torch_dist = torch.cdist(b, a, 1)[0]

    # Perform sequence matching convolution on similarity matrix
    if sequence_length != 0:
        import torch.nn.functional as F
        dist_tensor = torch_dist.to(device).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
        precomputed_convWeight = torch.eye(sequence_length, device=device).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
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

    else:
        dist_matrix_seq = torch_dist.cpu().numpy()

    # save distance matrix as a pdf image
    plt.imshow(dist_matrix_seq)
    plt.colorbar()
    plt.savefig(os.path.join(outputdir, 'distance_matrix_SAD.pdf'))
    plt.close()

    R = []
    N = [1,5,10,15,20,25] # N values to calculate
    P, R = createPR(dist_matrix_seq, GTtol, outputdir, datatype="SAD", matching='single', n_thresh=100)
    
    PR_data = {
                "Precision": P,
                "Recall": R
            }

    # Calculate Recall@N
    recallatn = []
    for n in N:
        recallatn.append(round(recallAtK(dist_matrix_seq,GTtol,K=n),2))
    # Print the results
    print('===== Sum-of-Absolute-Differences Recall@N =====')
    table = PrettyTable()
    table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
    table.add_row(["Recall", recallatn[0], recallatn[1], recallatn[2], recallatn[3], recallatn[4], recallatn[5]])
    print(table)

    return PR_data, recallatn