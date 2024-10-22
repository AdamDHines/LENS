import os
import numpy as np
from skimage import io

import re
import torch

from tqdm import tqdm
from lens.src.metrics import recallAtK, createPR
from prettytable import PrettyTable
import matplotlib.pyplot as plt   
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_and_preprocess_images(folder_path):
    images = []
    files = sorted(os.listdir(folder_path), key=natural_sort_key)
    for _, filename in enumerate(files):

        if filename.endswith('.png'):
            img = io.imread(os.path.join(folder_path, filename))    
            images.append(img.flatten()) 
    return np.array(images)

def run_sad(reference, query, GT, outputdir, sequence_length):

    # Load and preprocess images from both folders

    # Track progress for both folders
    images1 = load_and_preprocess_images(query)
    images2 = load_and_preprocess_images(reference)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = torch.from_numpy(images1.reshape(images1.shape[0], -1).astype(np.float32)).unsqueeze(0).to(device)
    b = torch.from_numpy(images2.reshape(images2.shape[0], -1).astype(np.float32)).unsqueeze(0).to(device)

    # Track progress for calculating distance
    torch_dist = torch.cdist(a, b, 1)[0]
    dist_tensor = torch_dist.clone().detach().to(device).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    precomputed_convWeight = torch.eye(sequence_length, device=device).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    dist_matrix_seq = torch.nn.functional.conv2d(dist_tensor, precomputed_convWeight).squeeze().cpu().numpy() / sequence_length
    dist_matrix_seq = dist_matrix_seq.T

    # save distance matrix as a pdf image
    plt.imshow(dist_matrix_seq)
    plt.colorbar()
    plt.savefig(os.path.join(outputdir, 'distance_matrix_SAD.pdf'))
    plt.close()

    R = []
    N = [1,5,10,15,20,25] # N values to calculate
    P, R = createPR(1/dist_matrix_seq, GT, outputdir, datatype="SAD", matching='single', n_thresh=100)   
 
    PR_data = {
                "Precision": P,
                "Recall": R
            }

    # Calculate Recall@N
    recallatn = []
    for n in N:
        recallatn.append(round(recallAtK(1/dist_matrix_seq,GT,K=n),2))
    # Print the results
    table = PrettyTable()
    table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
    table.add_row(["Recall", recallatn[0], recallatn[1], recallatn[2], recallatn[3], recallatn[4], recallatn[5]])
    print(table)

    return PR_data, recallatn