'''
This script contains a series of tools for processing DVS data and creating temporal representative frames.

The following tools are available:

TODO - extract_rosbag: From a .bag file, extracts the DVS data into a .txt file
TODO - simple_rep: Creates a very rudimentary plot of the DVS data over a timebin
TODO - decay_rep: Creates a plot of the DVS data over a timebin with input decay for improved temporal resolution
TODO - video_create: From a series of generated DVS frames, reconstructs into an .avi video file
TODO - event_stats: Calculates a variety of statistics from the event data, including event rate, event density, and event entropy

See https://github.com/AdamDHines/VPRTempoNeuro for more details. 

REQUIRED: The output from a DAVIS camera, other sensors/output are not supported.
'''

# Imports
import argparse

from vprtemponeuro.tools.extract_rosbag import ExtractRosbag

def run_tool(args):
    # Extract rosbag
    if args.tool == 'extract_rosbag':
        extract = ExtractRosbag(args)
        extract.run_extract()
    # Simple representation
    elif args.tool == 'simple_rep':
        dp.simple_rep(args) 
    # Decay representation
    elif args.tool == 'decay_rep':
        dp.decay_rep(args)
    # Video creation
    elif args.tool == 'video_create':
        dp.video_create(args)

def dvs_parser():
    # Define the base parameter parser (configurable by the user)
    parser = argparse.ArgumentParser(description="Args for configuration DVS processing")

    # Define which tool to implement
    parser.add_argument('--tool', type=str, default='extract_rosbag', help='Tool to implement')

    # Define the input file (this can a rosbag, .txt, or .parquet file)
    parser.add_argument('--input_file', type=str, default='', help='Input file')

    # Define the dataset folder, used as the output folder as well (default is relative path to ./dataset)
    parser.add_argument('--dataset_folder', type=str, default='./dataset', help='Dataset folder')

    # Define the arguments for the frame representations
    parser.add_argument('--timebin', type=float, default=33, help='Timebin for frame representation (in ms)')
    parser.add_argument('--decay_factor', type=float, default=0.95, help='Decay factor for frame representation (0-1)')
    parser.add_argument('--accum_factor', type=int, default=1, help='Accumulation value for frame representation (1-255)')
    parser.add_argument('--offset', type=float, default=0, help='Offset for frame representation (in µs)')

    # Parse arguments
    args = parser.parse_args()

    # Run specified tool
    run_tool(args)

if __name__ == "__main__":
    dvs_parser()