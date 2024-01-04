#MIT License

#Copyright (c) 2023 Adam Hines, Peter G Stratton, Michael Milford, Tobias Fischer

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

'''
Imports
'''
import argparse
import sys

from vprtemponeuro.src.event_stats import EventStats

from vprtemponeuro.VPRTempoNeuro import VPRTempoNeuro, run_inference
from vprtemponeuro.VPRTempoRaster import VPRTempoRaster, run_inference_raster
from vprtemponeuro.VPRTempoTrain import VPRTempoTrain, generate_model_name, train_new_model, check_pretrained_model

def initialize_and_run_model(args):
    # If user wants to train a new network
    if args.train_new_model:
        # Initialize the model
        model = VPRTempoTrain(args)
        eventModel = EventStats(model,event_type="max",max_pixels=args.pixels) # Initialize EventStats model
        eventModel.main() # Run the event statistics
        # Generate the model name
        model_name = generate_model_name(model)
        # Check if the model has been trained before
        check_pretrained_model(model_name)
        # Train the model
        train_new_model(model, model_name)
    else: # Run the inference network
        # Set the quantization configuration
        if args.raster:
            # Initialize the quantized model
            model = VPRTempoRaster(args)
            # Generate the model name
            model_name = generate_model_name(model)
            # Run the quantized inference model
            run_inference_raster(model, model_name)
        else:
            # Initialize the model
            model = VPRTempoNeuro(args)
            # Generate the model name
            model_name = generate_model_name(model)
            # Run the inference model
            run_inference(model, model_name)

def parse_network(use_raster=False, train_new_model=False):
    '''
    Define the base parameter parser (configurable by the user)
    '''
    parser = argparse.ArgumentParser(description="Args for base configuration file")

    # Define the dataset arguments
    parser.add_argument('--dataset', type=str, default='event',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--data_dir', type=str, default='./vprtemponeuro/dataset/',
                            help="Directory where dataset files are stored")
    parser.add_argument('--num_places', type=int, default=25,
                            help="Number of places to use for training and/or inferencing")
    parser.add_argument('--num_modules', type=int, default=1,
                            help="Number of expert modules to use split images into")
    parser.add_argument('--database_dirs', nargs='+', default=['database_filtered'],
                            help="Directories to use for training")
    parser.add_argument('--query_dir', nargs='+', default=['query_filtered'],
                            help="Directories to use for testing")
    parser.add_argument('--pixels', type=int, default=121,
                        help="Number of places to use for training and/or inferencing")

    # Define training parameters
    parser.add_argument('--filter', type=int, default=5,
                            help="Images to skip for training and/or inferencing")
    parser.add_argument('--epoch', type=int, default=4,
                            help="Number of epochs to train the model")

    # Define image transformation parameters
    parser.add_argument('--dims', nargs='+', type=int, default=[11,11],
                            help="Dimensions to resize the image to")

    # Define the network functionality
    parser.add_argument('--train_new_model', action='store_true',
                            help="Flag to run the training or inferencing model")
    parser.add_argument('--raster', action='store_true',
                            help="Run the raster version of VPRTempo, for non-neuromorphic chip inferencing")
    
    # On-chip specific parameters
    parser.add_argument('--power_monitor', action='store_true',
                            help="Whether or not to use the power monitor")
    parser.add_argument('--raster_device', type=str, default='cpu',
                            help="When using raster analysis, use CPU or GPU")
    
    # If the function is called with specific arguments, override sys.argv
    if use_raster or train_new_model:
        sys.argv = ['']
        if use_raster:
            sys.argv.append('--raster')
        if train_new_model:
            sys.argv.append('--train_new_model')

    # Output base configuration
    args = parser.parse_args()

    # Run the network with the desired settings
    initialize_and_run_model(args)

if __name__ == "__main__":
    # User input to determine if using quantized network or to train new model 
    parse_network(
                use_raster=False, 
                train_new_model=False
                )