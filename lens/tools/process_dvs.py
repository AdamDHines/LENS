'''
This script contains a series of tools for processing DVS data and creating temporal representative frames.

The following tools are available:

    - extract_rosbag: From a .bag file, extracts the DVS event data into a .zip file with [timestamp x y polarity]
    - simple_rep: Creates a very rudimentary plot of the DVS data over a timebin
    - decay_rep: Creates a plot of the DVS data over a timebin with input decay for improved temporal resolution
    - video_create: From a series of generated DVS frames, reconstructs into an .avi video file
TODO- event_stats: Calculates a variety of statistics from the event data, including event rate, event density, and event entropy

See https://github.com/AdamDHines/LENS for more details. 

REQUIRED: The output from a DAVIS camera, other sensors/output are not supported.
'''

# Imports
import sys
import argparse

from lens.tools.dvstools import ExtractRosbag, FrameRep, CreateVideo

def run_tool(args):
    #args.tool = 'simple_rep'
    #args.frame_limit = True
    #args.reference = True
    # Check if a tool was specified
    if args.tool == '':
        print("No tool specified.")
        sys.exit()
    # Extract rosbag
    if args.tool == 'extract_rosbag':
        extract = ExtractRosbag(args)
        extract.run_extract()
    # Simple representation
    elif args.tool == 'simple_rep' or args.tool == 'decay_rep' or args.tool == 'event_profile':
        rep = FrameRep(args)
        for _ in rep.event_data():
            print('Processed')
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
    parser.add_argument('--input_file', type=str, default='sunset1', help='Input file')
    parser.add_argument('--hot_pixels', type=str, default='sunset1_hot_pixels', help='Hot pixels file')

    # Define the output file name (default is the input file name)
    parser.add_argument('--output_name', type=str, default='sunset1_profile', help='Output name')

    # Define the dataset folder, used as the output folder as well (default is relative path to ./dataset)
    parser.add_argument('--dataset_folder', type=str, default='./lens/dataset/brisbane_event/davis', help='Dataset folder')

    # Define the arguments for the frame representations
    parser.add_argument('--timebin', type=float, default=1, 
                            help='Timebin for frame representation (in fps)')
    parser.add_argument('--decay_factor', type=float, default=5, 
                            help='Decay factor for frame representation (0-1)')
    parser.add_argument('--accum_factor', type=float, default=1, 
                            help='Accumulation value for frame representation (1-255)')
    parser.add_argument('--offset', type=float, default=1587452582.35, 
                            help='Offset for frame representation (in µs)')
    parser.add_argument('--frames_max', type=int, default=900, 
                            help='Number of frames to generate (default is 900 for 30s at 30fps)')
    parser.add_argument('--frame_limit', action='store_true', 
                            help='Flag to limit frames (True) or just create all frames (False)')
    parser.add_argument('--pixels', type=int, default=25,
                    help="Number of places to use for training and/or inferencing")
    parser.add_argument('--reference', action='store_true', 
                            help='Flag to limit frames (True) or just create all frames (False)')

    # Parse arguments
    args = parser.parse_args()

    # Run specified tool
    run_tool(args)

if __name__ == "__main__":
    dvs_parser()