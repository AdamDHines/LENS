#   =====================================================================
#   Copyright (C) 2023  Stefan Schubert, stefan.schubert@etit.tu-chemnitz.de
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#   =====================================================================
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def createPR(S_in, GThard, outputdir, datatype="LENS", GTsoft=None, matching='multi', n_thresh=100):
    """
    Calculates the precision and recall at n_thresh equally spaced threshold values
    for a given similarity matrix S_in and ground truth matrices GThard and GTsoft for
    single-best-match VPR or multi-match VPR.

    The matrices S_in, GThard and GTsoft are two-dimensional and should all have the
    same shape.
    The matrices GThard and GTsoft should be binary matrices, where the entries are
    only zeros or ones.
    The matrix S_in should have continuous values between -Inf and Inf. Higher values
    indicate higher similarity.
    The string matching should be set to either "single" or "multi" for single-best-
    match VPR or multi-match VPR.
    The integer n_thresh controls the number of threshold values and should be >1.
    """
    if S_in.shape != GThard.shape:
        S_in = S_in.T

    assert (S_in.shape == GThard.shape),"S_in, GThard and GTsoft must have the same shape"
    assert (S_in.ndim == 2),"S_in, GThard and GTsoft must be two-dimensional"
    assert (matching in ['single', 'multi']),"matching should contain one of the following strings: [single, multi]"
    assert (n_thresh > 1),"n_thresh must be >1"

    # ensure logical datatype in GT and GTsoft
    GT = GThard.astype('bool')
    if GTsoft is not None:
        GTsoft = GTsoft.astype('bool')
    GThard_orig = GThard.copy()

    # copy S and set elements that are only true in GTsoft to min(S) to ignore them during evaluation
    S = S_in.copy()

    # create the y_scores matrix for precision and recall
    a = np.argmax(GThard, axis=0)
    b = np.argmax(S, axis=0)
    # find where a and b match, assign 1 if it does and 0 if it does not
    # Flatten the GThard and S_in matrices to handle multiple ground truths per column
    # Number of columns
    num_cols = S.shape[1]

    # Find the indices of the highest scores in S for each column
    b = np.argmax(S, axis=0)  # Shape: (num_cols,)

    # Extract the corresponding scores
    y_scores = S[b, np.arange(num_cols)]  # Shape: (num_cols,)
    # Find the indices of the highest scores in S for each column
    pred_indices = np.argmax(S, axis=0)  # Shape: (num_cols,)
    # Determine if the highest score corresponds to any ground truth match
    # Since GThard can have multiple ground truths per column, check if GThard[b[i], i] == 1
    y_true = GThard[b, np.arange(num_cols)].astype(int)  # Shape: (num_cols,)

    # Compute Precision-Recall curve
    precisionWu, recallWu, thresholds = precision_recall_curve(y_true, y_scores)
    # Extract Ground Truth Indices
    gt_indices = np.argmax(GThard, axis=0)  # Shape: (num_cols,)

    # Calculate Distances: Predicted Index - Ground Truth Index
    distances = pred_indices - gt_indices  # Shape: (num_cols,)

    # Define Binning Parameters
    # Consolidate distances <= -5 and >= +5
    bins = np.arange(-5.5, 6.5, 1)  # Bins from -5 to +5 with bin size 1
    labels = ['<=-5'] + [str(i) for i in range(-4, 5)] + ['>=5']

    # Digitize distances into bins
    # np.digitize assigns indices such that bin_edges[i-1] <= x < bin_edges[i]
    # To include the rightmost edge, set right=True
    digitized = np.digitize(distances, bins, right=False)

    # Initialize an array to hold the bin counts
    hist_counts = np.zeros(len(labels), dtype=int)

    # Assign counts to the corresponding labels
    for i, bin_label in enumerate(labels):
        if bin_label == '<=-5':
            hist_counts[i] = np.sum(distances <= -5)
        elif bin_label == '>=5':
            hist_counts[i] = np.sum(distances >= 5)
        else:
            # For bins -4 to +4
            lower = int(bin_label)
            upper = lower + 1
            hist_counts[i] = np.sum((distances >= lower) & (distances < upper))

    # ---------------------------
    # Step 5: Plot the Distance Distribution
    # ---------------------------

    # Define the positions for the bars
    x_positions = np.arange(len(labels))

    # plt.figure(figsize=(12, 6))
    # bars = plt.bar(x_positions, hist_counts, color='skyblue', edgecolor='black')

    # # Add labels and title
    # plt.xlabel('Distance from Ground Truth (rows)')
    # plt.ylabel('Number of Predictions')
    # plt.title('Distribution of Distances Between Predicted and Ground Truth Indices')
    # plt.xticks(x_positions, labels)
    # plt.ylim(0, 600)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # # Annotate bar counts on top of each bar
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.annotate(f'{height}',
    #                 xy=(bar.get_x() + bar.get_width() / 2, height),
    #                 xytext=(0, 3),  # 3 points vertical offset
    #                 textcoords="offset points",
    #                 ha='center', va='bottom')

    # plt.tight_layout()
    # plt.show()
    # y_scores = np.where(a == b, 1, 0)
    # get the max val for each from from S
    # y_true = np.max(S, axis=0)
    # precisionWu, recallWu, thresholds = precision_recall_curve(y_scores, y_true)
    # plot the PR curve
    # plt.plot(recall, precision, marker='.', label=datatype)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend()
    # plt.show()

    # get the query index with the highest similarity for each database image from y_scores variable wherever = 1
    # correct_idx = (np.where(y_scores==1)[0],b[np.where(y_scores == 1)[0]])
    # incorrect_idx = (np.where(y_scores==0)[0],b[np.where(y_scores == 0)[0]])
    # incorrect = np.where(y_scores == 0)[0]
    # # find the index in S_in that matches GThard for columns where y_score == 0
    # # Initialize a list to hold correct indices for incorrect matches
    # correct_matches_for_incorrect = []

    # for col in incorrect:
    #     # Find the index where GThard is 1 for this column
    #     correct_match_indices = np.where(GThard[:, col] == 1)[0]
    #     if correct_match_indices.size > 0:
    #         # Assuming single ground truth match per column
    #         correct_match = correct_match_indices[0]
    #         correct_matches_for_incorrect.append(correct_match)
    #     else:
    #         # Handle cases where there might be no ground truth (optional)
    #         correct_matches_for_incorrect.append(None)

    # GT_idx = (np.where(y_scores==0)[0],np.array(correct_matches_for_incorrect))
    # # save all the idx variables into a compressed .npz file
    # np.savez_compressed(f'{outputdir}/idx_{datatype}.npz', correct_idx=correct_idx, incorrect_idx=incorrect_idx, GT_idx=GT_idx)
    
    if GTsoft is not None:
        S[GTsoft & ~GT] = S.min()
    
    if matching == 'single':
        # count the number of ground-truth positives (GTP)
        # GTP = np.count_nonzero(GT.any(0))

        # GT-values for best match per query (i.e., per column)
        GT = GT[np.argmax(S, axis=0), np.arange(GT.shape[1])]
        GTP = np.count_nonzero(GT)
        selected_rows = np.nanargmax(S, axis=0)  # Shape: (n_cols,)

        # similarities for best match per query (i.e., per column)
        S = np.max(S, axis=0)

    elif matching == 'multi':
        # count the number of ground-truth positives (GTP)
        GTP = np.count_nonzero(GT) # ground truth positives

    # init precision and recall vectors
    R = [0, ]
    P = [1, ]

    # select start and end treshold
    startV = S.max()  # start-value for treshold
    endV = S.min()  # end-value for treshold
    thresholds = np.linspace(startV, endV, n_thresh)

    # Iterate over different thresholds with enumeration to track the last iteration
    for idx, i in enumerate(thresholds):
        B = S >= i  # Apply threshold
        
        TP = np.count_nonzero(GT & B)  # True Positives
        FP = np.count_nonzero((~GT) & B)  # False Positives
        FN  = np.count_nonzero(GT & (~B))  # False Negatives

        # Handle division by zero for precision
        precision = TP / (TP + FP)
        recall = TP / (TP + FN) 
        
        P.append(precision)  # Precision
        R.append(recall)     # Recall
        
        # Check if it's the last iteration
        if idx == len(thresholds) - 1:
            if matching == 'single':
                # Create boolean masks for TP and FP
                TP_mask = GT & B  # 1D array
                FP_mask = (~GT) & B  # 1D array
                
                # True Positives coordinates
                TP_cols = np.where(TP_mask)[0]
                TP_rows = selected_rows[TP_cols]
                
                # False Positives coordinates
                FP_cols = np.where(FP_mask)[0]
                FP_rows = selected_rows[FP_cols]
                
                # Plotting the main similarity matrix with GT, TP, and FP
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Display the similarity matrix
                cax = ax.imshow(S_in, cmap='viridis', aspect='auto')
                fig.colorbar(cax, ax=ax, label='Similarity Score')
                ax.set_title(f'{datatype} Similarity Matrix with Ground Truth, TP, and FP')
                
                # Ground Truth: Plot as white dots
                gt_y, gt_x = np.where(GThard_orig)
                ax.scatter(gt_x, gt_y, facecolors='white', edgecolors='white',
                           marker='s', label='Ground Truth', linewidths=3.0)
                
                # True Positives: Plot as green circles
                ax.scatter(TP_cols, TP_rows, facecolors='green', edgecolors='green',
                           marker='.', label='True Positives', linewidths=3.0)
                
                # False Positives: Plot as red crosses
                ax.scatter(FP_cols, FP_rows, marker='x', color='red',
                           label='False Positives', linewidths=3.0)
                
                # Configure legend
                ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1))
                
                # Configure axes labels
                ax.set_xlabel('Query Index')
                ax.set_ylabel('Database Index')
                
                # plt.tight_layout()
                # plt.show()
                # plt.savefig(outputdir + f'/similarity_matrix_{datatype}.pdf', dpi=300)
                plt.close()

                                # -------------------- Error Rate Plotting Starts Here -------------------- #

                # Initialize lists to store True Positives (TP), False Positives (FP), and Error Rates
                # Initialize lists to store Error Rates
                error_rates = []

                # Total number of columns in the similarity matrix
                num_columns = S_in.shape[1]

                # Iterate over each column to compute Error Rate
                for col in range(num_columns):
                    if matching == 'single':
                        # In 'single' matching, each column has at most one detection
                        # Selected row for this column
                        selected_row = selected_rows[col]
                        # Determine if it's a True Positive
                        TP = GT[col]  # GT is already modified for 'single' matching
                        # False Positive is 1 if TP is False, else 0
                        FP = 0 if TP else 1
                        # Compute error rate
                        error = FP / TP if TP else 1  # Set error to 1 if TP is 0
                    elif matching == 'multi':
                        # In 'multi' matching, there can be multiple detections per column
                        # Extract detections for this column
                        B_col = B[:, col]
                        GT_col = GThard_orig[:, col].astype('bool')

                        # Compute TP and FP for this column
                        TP = np.count_nonzero(GT_col & B_col)
                        FP = np.count_nonzero((~GT_col) & B_col)

                        # Compute error rate, handle TP=0
                        error = FP / TP if TP > 0 else 0  # Define error as 0 if no TPs
                    else:
                        # This block should not be reached due to earlier assertion
                        error = 0

                    error_rates.append(error)

                # # Define labels for the x-axis (e.g., Column indices)
                # x_labels = np.arange(1, num_columns + 1)
                # error =np.array(error_rates)
                # np.save('/Users/adam/outdoor_error.npy', error)
                # # Plotting the Error Rates
                # plt.figure(figsize=(12, 6))
                # plt.plot(x_labels, error_rates, marker='o', linestyle='-', color='blue', linewidth=1.5, markersize=4)

                # # Adding labels and title
                # plt.xlabel('Column Index', fontsize=12)
                # plt.ylabel('Error Rate (FP / TP)', fontsize=12)
                # plt.title(f'Error Rate Over Columns for {datatype}', fontsize=14)

                # # Adding grid for better readability
                # plt.grid(True, linestyle='--', alpha=0.7)

                # # Optional: Annotate error rates on the plot
                # # To prevent clutter, you might choose to skip annotations for large number of columns
                # # Uncomment the following block if you wish to add annotations for fewer columns
                # """
                # for i, rate in enumerate(error_rates):
                #     plt.text(x_labels[i], rate, f'{rate:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)
                # """

                # # Adjust layout for better spacing
                # plt.tight_layout()
                # plt.show()
                # Save the plot as a PDF
                #plt.savefig(os.path.join(outputdir, f'error_rate_{datatype}.pdf'), dpi=300)

                # Close the plot to free memory
                plt.close()
    # print the fiunal precision and recall values
    # print(f'Precision: {P[-1]:.3f}')
    # print(f'Recall: {R[-1]:.3f}')
    # plt.plot(R, P, marker='.', color="green", label='Ours')
    # # plt.plot(recallWu, precisionWu, marker='.', label='Wu et al. 2023')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend()
    # # adjust the y-axis scale to be from 0 to 1
    # plt.ylim(0, 1.1)
    # plt.show()
    
    return P, R




def recallAt100precision(S_in, GThard, GTsoft=None, matching='multi', n_thresh=100):
    """
    Calculates the maximum recall at 100% precision for a given similarity matrix S_in 
    and ground truth matrices GThard and GTsoft for single-best-match VPR or multi-match 
    VPR.

    The matrices S_in, GThard and GTsoft are two-dimensional and should all have the
    same shape.
    The matrices GThard and GTsoft should be binary matrices, where the entries are
    only zeros or ones.
    The matrix S_in should have continuous values between -Inf and Inf. Higher values
    indicate higher similarity.
    The string matching should be set to either "single" or "multi" for single-best-
    match VPR or multi-match VPR.
    The integer n_tresh controls the number of threshold values during the creation of
    the precision-recall curve and should be >1.
    """

    assert (S_in.shape == GThard.shape),"S_in and GThard must have the same shape"
    if GTsoft is not None:
        assert (S_in.shape == GTsoft.shape),"S_in and GTsoft must have the same shape"
    assert (S_in.ndim == 2),"S_in, GThard and GTsoft must be two-dimensional"
    assert (matching in ['single', 'multi']),"matching should contain one of the following strings: [single, multi]"
    assert (n_thresh > 1),"n_thresh must be >1"

    # get precision-recall curve
    P, R = createPR(S_in, GThard, GTsoft, matching=matching, n_thresh=n_thresh)
    P = np.array(P)
    R = np.array(R)

    # recall values at 100% precision
    R = R[P==1]

    # maximum recall at 100% precision
    R = R.max()

    return R


def recallAtK(S_in, GThard, GTsoft=None, K=1):
    """
    Calculates the recall@K for a given similarity matrix S_in and ground truth matrices 
    GThard and GTsoft.

    The matrices S_in, GThard and GTsoft are two-dimensional and should all have the
    same shape.
    The matrices GThard and GTsoft should be binary matrices, where the entries are
    only zeros or ones.
    The matrix S_in should have continuous values between -Inf and Inf. Higher values
    indicate higher similarity.
    The integer K>=1 defines the number of matching candidates that are selected and
    that must contain an actually matching image pair.
    """
    assert (S_in.shape == GThard.shape),"S_in and GThard must have the same shape"
    if GTsoft is not None:
        assert (S_in.shape == GTsoft.shape),"S_in and GTsoft must have the same shape"
    assert (S_in.ndim == 2),"S_in, GThard and GTsoft must be two-dimensional"
    assert (K >= 1),"K must be >=1"

    # ensure logical datatype in GT and GTsoft
    GT = GThard.astype('bool')
    if GTsoft is not None:
        GTsoft = GTsoft.astype('bool')

    # copy S and set elements that are only true in GTsoft to min(S) to ignore them during evaluation
    S = S_in.copy()
    if GTsoft is not None:
        S[GTsoft & ~GT] = S.min()

    # discard all query images without an actually matching database image
    j = GT.sum(0) > 0 # columns with matches

    S = S[:,j] # select columns with a match
    GT = GT[:,j] # select columns with a match

    # select K highest similarities ignorning -1 values
    i = S.argsort(0)[-K:,:]

    j = np.tile(np.arange(i.shape[1]), [K, 1])
    GT = GT[i, j]

    # recall@K
    RatK = np.sum(GT.sum(0) > 0) / GT.shape[1]

    return RatK
