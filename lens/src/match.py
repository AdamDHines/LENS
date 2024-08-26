import numpy as np
from metrics import createPR, recallAtK
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import os
import seaborn as sns

base_dir = '/home/adam/repo/LENS/lens/data/Figure3'
subfolder = '240724-11-49-52'

data = np.load(os.path.join(base_dir, subfolder, 'similarity_matrix.npy'),allow_pickle=True)
data = data.T

row_remove = np.array([0,1,2,3])
data_del = np.delete(data,row_remove,axis=1)

database_places = data_del.shape[0]
query_places = data_del.shape[1]

# Create the GT matrix with ones down the diagonal
GT = np.eye(min(database_places, query_places), database_places, dtype=int)

# Loop to populate the GTsoft matrix with tolerance
tolerance = 2  # Define the tolerance range

GTsoft = np.zeros((query_places, database_places), dtype=int)

# Loop to populate the GTsoft matrix with ones along the diagonal and within the tolerance range
for n in range(database_places):  # Iterate over database places
    for i in range(max(0, n - tolerance), min(query_places, n + tolerance + 1)):  # Apply tolerance
        GTsoft[i, n] = 1

import json
# If user specified, generate a PR curve
if model.PR_curve:
    # Create PR curve
    P, R = createPR(data_del.T, GThard=GT, GTsoft=GTsoft, matching='single', n_thresh=100)
    # Combine P and R into a list of lists
    PR_data = {
            "Precision": P,
            "Recall": R
        }
    output_file = "PR_curve_data_LENSindoor.json"
    # Construct the full path
    full_path = f"/home/adam/Documents/{output_file}"
    # Write the data to a JSON file
    with open(full_path, 'w') as file:
        json.dump(PR_data, file) 
    # Plot PR curve
    plt.plot(R,P)    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

if model.sim_mat:
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each matrix using matshow
    cax1 = axs[0].matshow(out, cmap='viridis')
    fig.colorbar(cax1, ax=axs[0], shrink=0.8)
    axs[0].set_title('Similarity matrix')

    cax2 = axs[1].matshow(GT, cmap='plasma')
    fig.colorbar(cax2, ax=axs[1], shrink=0.8)
    axs[1].set_title('GT')

    cax3 = axs[2].matshow(GTsoft, cmap='inferno')
    fig.colorbar(cax3, ax=axs[2], shrink=0.8)
    axs[2].set_title('GT-soft')

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Recall@N
N = [1,5,10,15,20,25] # N values to calculate
R = [] # Recall@N values
# Calculate Recall@N
for n in N:
    R.append(round(recallAtK(data_del,GThard=GT,GTsoft=GTsoft,K=n),2))
# Print the results
table = PrettyTable()
table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])
print(table)

plt.figure(figsize=(10, 8))
sns.heatmap(data0.T, annot=False, cmap='crest')
plt.title('Similarity matrix')
plt.xlabel("Query")
plt.ylabel("Database")
plt.show()