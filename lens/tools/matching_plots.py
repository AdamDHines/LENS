import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageEnhance
import matplotlib.patches as patches
import sys
import imageio
from typing import Dict, Tuple, List


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Plot and save individual query and reference image matches.')
    parser.add_argument('--npz_path', type=str, 
                        default='/Users/adam/repo/LENS/lens/output/161224-10-09-09/idx_LENS.npz',
                        help='Path to the .npz file.')
    parser.add_argument('--query_csv', type=str, 
                        default='/Users/adam/repo/LENS/lens/dataset/sunset1.csv',
                        help='Path to the query CSV file.')
    parser.add_argument('--reference_csv', 
                        type=str, 
                        default='/Users/adam/repo/LENS/lens/dataset/sunset2.csv',
                        help='Path to the reference CSV file.')
    parser.add_argument('--query_dir', 
                        type=str, 
                        default='/Users/adam/Downloads/sunset1/sunset1',
                        help='Directory containing query images.')
    parser.add_argument('--reference_dir', 
                        type=str, 
                        default='/Users/adam/Downloads/sunset2/sunset2',
                        help='Directory containing reference images.')
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='/Users/adam/matching_plots_auto',
                        help='Directory to save individual match PNGs and the animation.')
    parser.add_argument('--max_examples', type=int, default=None, help='Maximum number of total examples to plot.')
    parser.add_argument('--animation_name', type=str, default='matches_animation.gif', help='Filename for the output animation (GIF).')
    parser.add_argument('--frame_duration', type=float, default=1.0, help='Duration of each frame in the animation (seconds).')
    return parser.parse_args()


def load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the .npz file containing match indices.
    
    Parameters:
    - npz_path: Path to the .npz file.
    
    Returns:
    - correct_idx, incorrect_idx, GT_idx: Arrays containing indices.
    """
    if not os.path.isfile(npz_path):
        print(f"Error: The .npz file '{npz_path}' does not exist.")
        sys.exit(1)
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            correct_idx = data['correct_idx']
            incorrect_idx = data['incorrect_idx']
            GT_idx = data['GT_idx']
    except Exception as e:
        print(f"Error loading .npz file '{npz_path}': {e}")
        sys.exit(1)
    return correct_idx, incorrect_idx, GT_idx


def load_csv(csv_path: str) -> Dict[int, str]:
    """
    Load a CSV file and create a mapping from index to image name.
    
    Parameters:
    - csv_path: Path to the CSV file.
    
    Returns:
    - index_to_name: Dictionary mapping index to image name.
    """
    if not os.path.isfile(csv_path):
        print(f"Error: The CSV file '{csv_path}' does not exist.")
        sys.exit(1)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file '{csv_path}': {e}")
        sys.exit(1)
    if 'index' not in df.columns or 'Image_name' not in df.columns:
        print(f"Error: CSV file '{csv_path}' must contain 'index' and 'Image_name' columns.")
        sys.exit(1)
    index_to_name = dict(zip(df['index'], df['Image_name']))
    return index_to_name


def get_image_path(image_dir: str, image_name: str) -> str:
    """
    Construct the full path to an image given its directory and name.
    
    Parameters:
    - image_dir: Directory containing the image.
    - image_name: Name of the image file.
    
    Returns:
    - Full path to the image.
    """
    return os.path.join(image_dir, image_name)


def add_border(ax, color: str, linewidth: int = 5):
    """
    Add a colored border around the axes.
    
    Parameters:
    - ax: Matplotlib Axes object.
    - color: Color of the border.
    - linewidth: Width of the border line.
    """
    rect = patches.Rectangle((0,0),1,1, transform=ax.transAxes, 
                             linewidth=linewidth, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


def enhance_image(img: Image.Image, enhancement: str = 'contrast', factor: float = 1.2) -> Image.Image:
    """
    Apply an enhancement to the image using a lookup table.
    
    Parameters:
    - img: PIL Image object.
    - enhancement: Type of enhancement ('contrast', 'brightness', etc.).
    - factor: Enhancement factor.
    
    Returns:
    - Enhanced PIL Image object.
    """
    if enhancement == 'contrast':
        enhancer = ImageEnhance.Contrast(img)
    elif enhancement == 'brightness':
        enhancer = ImageEnhance.Brightness(img)
    elif enhancement == 'sharpness':
        enhancer = ImageEnhance.Sharpness(img)
    else:
        return img  # No enhancement if unknown type
    return enhancer.enhance(factor)


def extract_number(image_name: str) -> str:
    """
    Extract the number from image name like 'images_00000.png'.
    Returns '00000'.
    """
    basename = os.path.basename(image_name)
    name, ext = os.path.splitext(basename)
    # Assuming format 'images_00000'
    if '_' in name:
        return name.split('_')[-1]
    else:
        return name


def apply_colormap(img: Image.Image, colormap: str = 'viridis') -> np.ndarray:
    """
    Apply a colormap to a grayscale image.
    
    Parameters:
    - img: PIL Image object.
    - colormap: Matplotlib colormap name.
    
    Returns:
    - RGB image as a NumPy array.
    """
    if img.mode != 'L':
        # convert to grayscale if not already
        img = img.convert('L')
    # Convert to NumPy array
    img_array = np.array(img)
    # Normalize the image
    img_normalized = img_array / 255.0
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    img_colored = cmap(img_normalized)
    # Convert to RGB
    img_rgb = np.uint8(img_colored[:, :, :3] * 255)
    return img_rgb


def save_match_plot(query_img: Image.Image, prediction_img: Image.Image, gt_img: Image.Image, 
                   query_name: str, prediction_name: str, gt_name: str,
                   prediction_border_color: str, save_path: str):
    """
    Save a plot with Query, Prediction, and GT images with their names and numbers as titles.
    
    Parameters:
    - query_img: PIL Image object for the Query.
    - prediction_img: PIL Image object for the Prediction.
    - gt_img: PIL Image object for the Ground Truth.
    - query_name: Filename of the Query image.
    - prediction_name: Filename of the Prediction image.
    - gt_name: Filename of the Ground Truth image.
    - prediction_border_color: Color of the border around the Prediction image ('green' or 'red').
    - save_path: Path to save the PNG file.
    """
    # Extract numbers from image names
    query_num = extract_number(query_name)
    prediction_num = extract_number(prediction_name)
    gt_num = extract_number(gt_name)
    
    # Apply colormap if necessary
    query_display = apply_colormap(query_img, colormap='viridis')
    prediction_display = apply_colormap(prediction_img, colormap='viridis')
    gt_display = apply_colormap(gt_img, colormap='viridis')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Query Image (Left)
    ax_query = axes[0]
    ax_query.imshow(query_display)
    ax_query.axis('off')
    ax_query.set_title(f"Query {query_num}")
    
    # Plot Prediction Image (Middle) with Colored Border
    ax_pred = axes[1]
    ax_pred.imshow(prediction_display)
    ax_pred.axis('off')
    pred_num = extract_number(prediction_name)
    ax_pred.set_title(f"Prediction {pred_num}")
    add_border(ax_pred, prediction_border_color)
    
    # Plot GT Image (Right) without Colored Border
    ax_gt = axes[2]
    ax_gt.imshow(gt_display)
    ax_gt.axis('off')
    gt_num = extract_number(gt_name)
    ax_gt.set_title(f"GT {gt_num}")
    # Removed colored border from GT image
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_and_save_matches(correct_idx: np.ndarray, incorrect_idx: np.ndarray, GT_idx: np.ndarray,
                          query_map: Dict[int, str], reference_map: Dict[int, str],
                          query_dir: str, reference_dir: str, output_dir: str,
                          max_examples: int = None,
                          animation_name: str = 'matches_animation.gif',
                          frame_duration: float = 1.0):
    """
    Plot and save each match as an individual PNG file, organized by query index,
    and create an animation from the saved images.
    
    Parameters:
    - correct_idx: Array containing correct match indices.
    - incorrect_idx: Array containing incorrect match indices.
    - GT_idx: Array containing ground truth indices.
    - query_map: Dictionary mapping query indices to image names.
    - reference_map: Dictionary mapping reference indices to image names.
    - query_dir: Directory containing query images.
    - reference_dir: Directory containing reference images.
    - output_dir: Directory to save the output PNGs and animation.
    - max_examples: Maximum number of total examples to plot.
    - animation_name: Filename for the output animation (GIF).
    - frame_duration: Duration of each frame in the animation (seconds).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_correct = len(correct_idx[0])
    num_incorrect = len(incorrect_idx[0])
    
    total_correct = num_correct
    total_incorrect = num_incorrect
    
    if max_examples is not None:
        # Allocate half for correct and half for incorrect
        max_correct = max_examples // 2
        max_incorrect = max_examples - max_correct
    else:
        max_correct = total_correct
        max_incorrect = total_incorrect
    
    plotted = 0
    image_filenames: List[str] = []  # To store filenames for animation
    
    # Collect all matches first to sort by query index
    matches = []
    
    # Process Correct Matches
    for i in range(max_correct):
        if i >= num_correct:
            break
        query_index = correct_idx[0][i]
        reference_index = correct_idx[1][i]
        
        query_name = query_map.get(query_index, None)
        prediction_name = reference_map.get(reference_index, None)
        
        if query_name is None or prediction_name is None:
            print(f"Warning: Missing mapping for query index {query_index} or reference index {reference_index}. Skipping.")
            continue  # Skip if mapping not found
        
        query_path = get_image_path(query_dir, query_name)
        prediction_path = get_image_path(reference_dir, prediction_name)
        
        # For correct matches, GT is set to prediction_path
        gt_path = query_path
        
        # Load images
        try:
            query_img = Image.open(query_path).convert('RGB')
            prediction_img = Image.open(prediction_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images for correct match {i}: {e}. Skipping.")
            continue
        
        # Append to matches list
        matches.append({
            'query_img': query_img,
            'prediction_img': prediction_img,
            'gt_img': gt_img,
            'query_name': query_name,
            'prediction_name': prediction_name,
            'gt_name': prediction_name,  # gt_name = prediction_name for correct matches
            'border_color': 'green',
            'query_index': query_index
        })
    
    # Process Incorrect Matches
    for i in range(max_incorrect):
        if i >= num_incorrect:
            break
        query_index = incorrect_idx[0][i]
        incorrect_reference_index = incorrect_idx[1][i]
        ground_truth_index = GT_idx[1][i]
        
        query_name = query_map.get(query_index, None)
        incorrect_reference_name = reference_map.get(incorrect_reference_index, None)
        gt_name = reference_map.get(ground_truth_index, None)
        
        if query_name is None or incorrect_reference_name is None or gt_name is None:
            print(f"Warning: Missing mapping for indices {query_index}, {incorrect_reference_index}, or {ground_truth_index}. Skipping.")
            continue  # Skip if mapping not found
        
        query_path = get_image_path(query_dir, query_name)
        prediction_path = get_image_path(reference_dir, incorrect_reference_name)
        gt_path = get_image_path(reference_dir, gt_name)
        
        # Load images
        try:
            query_img = Image.open(query_path).convert('RGB')
            prediction_img = Image.open(prediction_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images for incorrect match {i}: {e}. Skipping.")
            continue
        
        # Append to matches list
        matches.append({
            'query_img': query_img,
            'prediction_img': prediction_img,
            'gt_img': gt_img,
            'query_name': query_name,
            'prediction_name': incorrect_reference_name,
            'gt_name': gt_name,
            'border_color': 'red',
            'query_index': query_index
        })
    
    # Sort matches by query index
    matches_sorted = sorted(matches, key=lambda x: x['query_index'])
    
    # Plot and save each match
    for match in matches_sorted:
        if max_examples is not None and plotted >= max_examples:
            break
        save_filename = f"q{match['query_index']:05d}.png"
        save_path = os.path.join(output_dir, save_filename)
        
        # Apply enhancements
        query_img_enhanced = enhance_image(match['query_img'], enhancement='contrast', factor=1.5)
        query_img_enhanced = enhance_image(query_img_enhanced, enhancement='brightness', factor=1.2)
        
        prediction_img_enhanced = enhance_image(match['prediction_img'], enhancement='contrast', factor=1.5)
        prediction_img_enhanced = enhance_image(prediction_img_enhanced, enhancement='brightness', factor=1.2)
        
        gt_img_enhanced = enhance_image(match['gt_img'], enhancement='contrast', factor=1.5)
        gt_img_enhanced = enhance_image(gt_img_enhanced, enhancement='brightness', factor=1.2)
        
        # Save plot
        save_match_plot(
            query_img_enhanced,
            prediction_img_enhanced,
            gt_img_enhanced,
            match['query_name'],
            match['prediction_name'],
            match['gt_name'],
            match['border_color'],
            save_path
        )
        image_filenames.append(save_filename)
        plotted +=1
    
    print(f"Total matches plotted and saved: {plotted}")
    print(f"All plots saved to directory: {output_dir}")
    
    # Create Animation
    create_animation(output_dir, image_filenames, animation_name, frame_duration)


def create_animation(output_dir: str, image_filenames: List[str], animation_name: str, frame_duration: float):
    """
    Create an animated GIF from saved images.
    
    Parameters:
    - output_dir: Directory where images are saved.
    - image_filenames: List of image filenames to include in the animation.
    - animation_name: Name of the output GIF file.
    - frame_duration: Duration of each frame in seconds.
    """
    if not image_filenames:
        print("No images available to create animation.")
        return
    
    # Sort image filenames based on query index extracted from filename
    # Filenames are in the format 'qXXXXX.png'
    def extract_query_index(filename: str) -> int:
        """
        Extract the query index from the filename.
        Expected format: 'qXXXXX.png'
        """
        try:
            q_index = int(filename.split('.')[0][1:])  # Remove 'q' and '.png'
            return q_index
        except (IndexError, ValueError):
            return 0  # Default to 0 if parsing fails
    
    sorted_filenames = sorted(image_filenames, key=extract_query_index)
    
    # Read images
    images = []
    for filename in sorted_filenames:
        path = os.path.join(output_dir, filename)
        try:
            img = imageio.imread(path)
            images.append(img)
        except Exception as e:
            print(f"Error reading image '{path}': {e}. Skipping.")
    
    if not images:
        print("No valid images found to create animation.")
        return
    
    # Save as GIF
    animation_path = os.path.join(output_dir, animation_name)
    try:
        imageio.mimsave(animation_path, images, duration=frame_duration)
        print(f"Animation saved to {animation_path}")
    except Exception as e:
        print(f"Error creating animation '{animation_path}': {e}")


def main():
    args = parse_arguments()
    
    # Verify directories
    if not os.path.isdir(args.query_dir):
        print(f"Error: Query directory '{args.query_dir}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(args.reference_dir):
        print(f"Error: Reference directory '{args.reference_dir}' does not exist.")
        sys.exit(1)
    
    # Load data
    correct_idx, incorrect_idx, GT_idx = load_npz(args.npz_path)
    
    # Load CSV mappings
    query_map = load_csv(args.query_csv)
    reference_map = load_csv(args.reference_csv)
    
    # Plot and save matches
    plot_and_save_matches(correct_idx, incorrect_idx, GT_idx, query_map, reference_map,
                          args.query_dir, args.reference_dir, args.output_dir, 
                          args.max_examples,
                          args.animation_name,
                          args.frame_duration)


if __name__ == "__main__":
    main()
