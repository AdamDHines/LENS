import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from PIL import Image
from kneed import KneeLocator

def load_image(image_path):
    with Image.open(image_path) as img:
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            alpha = img.convert('RGBA').split()[-1]
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=alpha)
            return bg.convert('L')
        else:
            return img.convert('L')

def extract_pixels(image):
    data = np.array(image)
    return np.column_stack(np.where(data > 0))

def determine_epsilon(data, n_neighbors=5):
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(data)
    distances, indices = neigh.kneighbors(data)
    distances = np.sort(distances[:, n_neighbors - 1])
    
    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    return kneedle.knee_y if kneedle.knee_y is not None else np.median(distances)

def evaluate_min_samples(data, epsilon, min_samples_list):
    best_score = -1
    best_min_samples = None

    for min_samples in min_samples_list:
        clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(data)
        labels = clustering.labels_

        if len(set(labels)) > 1 and np.any(labels != -1):
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_min_samples = min_samples

    return best_min_samples

def process_folder(folder_path):
    epsilon_values = []
    min_samples_values = []
    min_samples_candidates = [2, 3, 4, 5, 6, 7, 8]  # Adjust as needed

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = load_image(image_path)
            pixels = extract_pixels(image)

            epsilon = determine_epsilon(pixels)
            epsilon_values.append(epsilon)

            best_min_samples = evaluate_min_samples(pixels, epsilon, min_samples_candidates)
            if best_min_samples is not None:
                min_samples_values.append(best_min_samples)
            print(f"Processed {filename}: Epsilon = {epsilon}, Best min_samples = {best_min_samples}")

    final_epsilon = np.median(epsilon_values) if epsilon_values else None
    final_min_samples = np.median(min_samples_values) if min_samples_values else None
    print(f"Final Epsilon Value (Median): {final_epsilon}")
    print(f"Final min_samples Value (Median): {final_min_samples}")
    return final_epsilon, final_min_samples

# Example usage
folder_path = '/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/Brisbane-Event/database_filtered'
final_epsilon, final_min_samples = process_folder(folder_path)