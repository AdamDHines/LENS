import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_PR(LENS_PR, SAD_PR, output_path):
    # Plot the data
    plt.figure(figsize=(8, 4))

    # Plot Precision-Recall for both datasets with specified colors and markers
    plt.plot(SAD_PR['Recall'], SAD_PR['Precision'], label='SAD', color='#D753CC')
    plt.plot(LENS_PR['Recall'], LENS_PR['Precision'], label='LENS', color='#575AB1')

    # Add titles and labels
    plt.title('Precision-Recall Comparison')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    # Set the limits for x and y axes to start at 0
    plt.xlim(-1e-2, 1.0)  # Assuming Recall ranges between 0 and 1
    plt.ylim(0.0, 1.05)  # Assuming Precision ranges between 0 and 1

    # Save the figure as a high-dpi PDF
    output_name = 'Precision_Recall_plot.pdf'
    plt.savefig(os.path.join(output_path, output_name), dpi=300)
    plt.close()
    # output PR data as json
    with open(os.path.join(output_path, 'PR_curve_LENS.json'), 'w') as file:
        json.dump(LENS_PR, file)
    with open(os.path.join(output_path, 'PR_curve_SAD.json'), 'w') as file:
        json.dump(SAD_PR, file)

def plot_recall(lens_recall, sad_recall, N, output_path):
    # Plot the data
    plt.figure(figsize=(8, 4))

    # Plot SAD and LENS with specified colors and circles at each data point
    plt.plot(N, sad_recall, label='SAD', color='#D753CC', marker='o')
    plt.plot(N, lens_recall, label='LENS', color='#575AB1', marker='o')

    # Add titles and labels
    plt.title('Comparison of SAD and LENS')
    plt.xlabel('N')
    plt.ylabel('Values')
    plt.ylim(0, 1.05)  # Set y-axis limits from 0 to 1
    plt.legend()

    # Save the figure as a high-dpi PDF
    output_name = 'SAD_vs_LENS_RecallatN.pdf'
    plt.savefig(os.path.join(output_path, output_name), dpi=300)
    plt.close()
    # output recall data as csv
    data = {
        'N': N,
        'SAD': sad_recall,
        'LENS': lens_recall
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_path, 'SAD_vs_LENS_RecallatN.csv'), index=False)