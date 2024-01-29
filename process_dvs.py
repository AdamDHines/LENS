'''
This script contains a series of tools for processing DVS data and creating temporal representative frames.

The following tools are available:

    - extract_rosbag: From a .bag file, extracts the DVS event data into a .zip file with [timestamp x y polarity]
    - simple_rep: Creates a very rudimentary plot of the DVS data over a timebin
TODO - decay_rep: Creates a plot of the DVS data over a timebin with input decay for improved temporal resolution
    - video_create: From a series of generated DVS frames, reconstructs into an .avi video file
TODO - event_stats: Calculates a variety of statistics from the event data, including event rate, event density, and event entropy

See https://github.com/AdamDHines/VPRTempoNeuro for more details. 

REQUIRED: The output from a DAVIS camera, other sensors/output are not supported.
'''

# Imports
import sys
import argparse

from vprtemponeuro.tools.dvstools import ExtractRosbag, SimpleRep, DecayRep, CreateVideo

def run_tool(args):
    # Check if a tool was specified
    if args.tool == '':
        print("No tool specified.")
        sys.exit()
    # Extract rosbag
    if args.tool == 'extract_rosbag':
        extract = ExtractRosbag(args)
        extract.run_extract()
    # Simple representation
    elif args.tool == 'simple_rep':
        simple = SimpleRep(args)
        simple.simple_representation()
    # Decay representation
    elif args.tool == 'decay_rep':
        decay = DecayRep(args)
        decay.decay_representation()
    # Video creation
    elif args.tool == 'create_video':
        video = CreateVideo(args)
        video.create_video_from_frames()

def dvs_parser():
    # Define the base parameter parser (configurable by the user)
    parser = argparse.ArgumentParser(description="Args for configuration DVS processing")

    # Define which tool to implement
    parser.add_argument('--tool', type=str, default='', help='Tool to implement')

    # Define the input files (this can a rosbag, .txt, or .parquet file)
    parser.add_argument('--input_file', type=str, default='', help='Input file')
    parser.add_argument('--hot_pixels', type=str, default='', help='Hot pixels file')

    # Define the output file name (default is the input file name)
    parser.add_argument('--output_name', type=str, default='', help='Output name')

    # Define the dataset folder, used as the output folder as well (default is relative path to ./dataset)
    parser.add_argument('--dataset_folder', type=str, default='./vprtemponeuro/dataset', help='Dataset folder')

    # Define the arguments for the frame representations
    parser.add_argument('--timebin', type=float, default=30, help='Timebin for frame representation (in fps)')
    parser.add_argument('--decay_factor', type=float, default=0.95, help='Decay factor for frame representation (0-1)')
    parser.add_argument('--accum_factor', type=int, default=1, help='Accumulation value for frame representation (1-255)')
    parser.add_argument('--offset', type=float, default=0, help='Offset for frame representation (in µs)')

    # Parse arguments
    args = parser.parse_args()

    # Run specified tool
    run_tool(args)

if __name__ == "__main__":
    dvs_parser()