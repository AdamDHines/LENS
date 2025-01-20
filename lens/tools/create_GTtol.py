import numpy as np

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