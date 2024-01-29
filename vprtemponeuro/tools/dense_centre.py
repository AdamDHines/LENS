import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import matplotlib.pyplot as plt
import os 
from scipy.spatial.distance import cdist

def knn_distance_plot(data, n_neighbors=5):
    # Compute the nearest neighbors
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(data)
    
    # Find the distance to the k-th nearest neighbor
    distances, indices = neigh.kneighbors(data)
    
    # Sort and plot the distances
    sorted_distances = np.sort(distances[:, n_neighbors - 1])
    plt.plot(sorted_distances)
    plt.title("KNN Distance Plot for DBSCAN Epsilon")
    plt.xlabel("Points sorted by distance to {}-th nearest neighbor".format(n_neighbors))
    plt.ylabel("Epsilon distance")
    plt.show()

def load_image(image_path):
    # Load the image and convert it to grayscale (if needed)
    return Image.open(image_path).convert('L')

def extract_pixels(image):
    # Convert image to numpy array
    data = np.array(image)
    # Find the coordinates of non-zero pixels
    return np.column_stack(np.where(data > 0))  # Adjust threshold if needed

def find_densest_center(pixels):
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=47, min_samples=7).fit(pixels)
    labels = clustering.labels_

    # Find the largest cluster
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) > 1 and -1 in unique:  # Exclude noise if it exists
        largest_cluster = unique[np.argmax(counts[1:])] + 1
    else:
        largest_cluster = np.argmax(counts)

    # Compute the centroid of the largest cluster
    center = pixels[labels == largest_cluster].mean(axis=0)
    return center

def plot_densest_center(image_path, densest_center):
    # Load the image
    image = np.array(Image.open(image_path))
    plt.imshow(image, cmap='gray')  # Change cmap if the image is not grayscale

    # Plot the densest center
    plt.scatter([densest_center[1]], [densest_center[0]], color='red')  # x and y are reversed in plt.scatter
    plt.title("Densest Center Marked in Red")
    plt.show()

def process_images_in_directory(directory_path):
    centers = []
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith('.png'):  # Adjust this if your images have a different format
            image_path = os.path.join(directory_path, filename)
            image = load_image(image_path)
            pixels = extract_pixels(image)
            center = find_densest_center(pixels)
            plot_densest_center(image_path,center)
            centers.append(center)
    return centers

def compare_datasets(dataset1, dataset2):
    # Using Euclidean distance as a similarity measure
    distances = cdist(dataset1, dataset2, 'euclidean')
    return distances

def plot_similarity_matrix(matrix, title='Similarity Matrix'):
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Dataset q')
    plt.ylabel('Dataset d')
    plt.show()
    
def find_closest_indices(similarity_matrix):
    argmins = []
    for n in range(len(similarity_matrix)):
        argmins.append(np.argmin(similarity_matrix[n]))
    return argmins

# Example usage
d_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/test_d'
q_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/test_q'

# Process images in both directories
d_centers = process_images_in_directory(d_path)
q_centers = process_images_in_directory(q_path)

# Compare datasets
similarity_matrix = compare_datasets(d_centers, q_centers)

# Plot similarity matrix
plot_similarity_matrix(similarity_matrix, title='Densest Center Similarity Matrix Between Datasets d and q')
# Find the closest indices and corresponding distances
argmins = find_closest_indices(similarity_matrix)
