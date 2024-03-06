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

import wandb
import pprint
import argparse

import numpy as np

from vprtemponeuro.src.event_stats import EventStats
from vprtemponeuro.VPRTempo import VPRTempo, run_inference_norm
from vprtemponeuro.VPRTempoNeuro import VPRTempoNeuro, run_inference
from vprtemponeuro.VPRTempoTrain import VPRTempoTrain, train_new_model
from vprtemponeuro.VPRTempoRaster import VPRTempoRaster, run_inference_raster

def generate_model_name(model):
    """
    Generate the model name based on its parameters.
    """
    model_name = (''.join(model.reference)+"_"+
            "VPRTempo_" +
            "IN"+str(model.input)+"_" +
            "FN"+str(model.feature)+"_" + 
            "DB"+str(model.reference_places) +
            ".pth")
    return model_name

def initialize_and_run_model(args):
    args.norm = True
    #args.train_new_model = True 
    #args.raster = True
    # If user wants to train a new network
    if args.train_new_model:
        # Initialize the model
        model = VPRTempoTrain(args)

        if args.event_stats:
            eventModel = EventStats(
                                    model,
                                    event_type=args.event_type,
                                    max_pixels=args.pixels,
                                    total_patches=args.patches
                                    ) 
            eventModel.main() # Run the event statistics
        
        # Generate the model name
        model_name = generate_model_name(model)

        # Train the model
        train_new_model(model, model_name)
    elif args.wandb:
        # log into weights & biases
        wandb.login()
        
        # define the method and parameters for grid search
        sweep_config = {'method':'random'}
        metric = {'name':'R@1', 'goal':'maximize'}
        
        sweep_config['metric'] = metric
        
        parameters_dict = {    
                        'fire_l': {'values': list(np.linspace(0.1, 0.49, 16))},
                        'fire_h': {'values': list(np.linspace(0.5, 1.0, 16))},
                        'thr_h': {'values': list(np.linspace(0.1, 0.5, 8))},

        }
        
        sweep_config['parameters'] = parameters_dict
        pprint.pprint(sweep_config)
    
        # start sweep controller
        sweep_id = wandb.sweep(sweep_config, project="vprtemponeuro-dvs_100pixel")

        # Initialize w&b search
        def wandsearch(config=None):
            with wandb.init(config=config):
                # Initialize config
                config = wandb.config

                # Set arguments for the sweeps
                args.fire_l = config.fire_l
                args.fire_h = config.fire_h
                args.thr_h = config.thr_h

                if args.event_stats:
                    model = VPRTempoTrain(args)
                    eventModel = EventStats(
                            model,
                            event_type="max",
                            max_pixels=args.pixels,
                            total_patches=args.patches
                            ) 
                    eventModel.main() # Run the event statistics
                
                # Initialize the training model
                args.train_new_model = True
                R_all = []
                model = VPRTempoTrain(args)
                model_name = generate_model_name(model)
                train_new_model(model, model_name)
                
                # Initialize the inference model
                model = VPRTempo(args)
                # Generate the model name
                model_name = generate_model_name(model)
                # Run the quantized inference model
                R_all = run_inference_norm(model, model_name)

                # x = [1,5,10,15,20,25]
                # AUC= np.trapz(R_all, x)
                wandb.log({"R@1" : R_all[0]})
                
        wandb.agent(sweep_id,wandsearch)
    else: # Run the inference network
        # Set the quantization configuration
        if args.raster:
            # Initialize the quantized model
            model = VPRTempoRaster(args)
            # Generate the model name
            model_name = generate_model_name(model)
            # Run the quantized inference model
            run_inference_raster(model, model_name)
        elif args.norm:
            # Initialize the model
            model = VPRTempo(args)
            # Generate the model name
            model_name = generate_model_name(model)
            # Run the inference model
            run_inference_norm(model, model_name)
        else:
            # Initialize the model
            model = VPRTempoNeuro(args)
            # Generate the model name
            model_name = generate_model_name(model)
            # Run the inference model
            run_inference(model, model_name)

def parse_network():
    '''
    Define the base parameter parser (configurable by the user)
    '''
    parser = argparse.ArgumentParser(description="Args for base configuration file")

    # Define the dataset arguments
    parser.add_argument('--dataset', type=str, default='brisbane_event',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--camera', type=str, default='davis',
                            help="Camera to use for training and/or inferencing")
    parser.add_argument('--reference', type=str, default='sunset1_49',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--query', type=str, default='sunset2_49',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--data_dir', type=str, default='./vprtemponeuro/dataset/',
                            help="Directory where dataset files are stored")
    parser.add_argument('--reference_places', type=int, default=724,
                            help="Number of places to use for training and/or inferencing")
    parser.add_argument('--query_places', type=int, default=641,
                            help="Number of places to use for training and/or inferencing")
    parser.add_argument('--pixels', type=int, default=49,
                        help="Number of places to use for training and/or inferencing")

    # Define training parameters
    parser.add_argument('--filter', type=int, default=1,
                            help="Images to skip for training and/or inferencing")
    parser.add_argument('--epoch', type=int, default=10,
                            help="Number of epochs to train the model")
    
    # Hyperparameters
    parser.add_argument('--thr_l', type=float, default=0,
                        help="Low threshold value")
    parser.add_argument('--thr_h', type=float, default=0.2,
                        help="High threshold value")
    parser.add_argument('--fire_l', type=float, default=0.4,
                        help="Low threshold value")
    parser.add_argument('--fire_h', type=float, default=0.6,
                        help="High threshold value")
    parser.add_argument('--ip_rate', type=float, default=0.25,
                        help="ITP learning rate")
    parser.add_argument('--stdp_rate', type=float, default=0.25,
                        help="STDP learning rate")
    
    # Connection probabilities
    parser.add_argument('--f_exc', type=float, default=1.0,
                        help="Feature layer excitatory connection")
    parser.add_argument('--f_inh', type=float, default=1.0,
                        help="Feature layer inhibitory connection")
    parser.add_argument('--o_exc', type=float, default=1.0,
                        help="Output layer excitatory connection")
    parser.add_argument('--o_inh', type=float, default=1.0,
                        help="Output layer inhibitory connection")
    
    # Define image transformation parameters
    parser.add_argument('--dims', nargs='+', type=int, default=[7,7],
                            help="Dimensions to resize the image to")
    parser.add_argument('--patches', type=int, default=15,
                            help="Number of patches")
    parser.add_argument('--repeats', type=int, default=9,
                        help="Number of repeats in the temporal code")

    # Define the network functionality
    parser.add_argument('--train_new_model', action='store_true',
                            help="Flag to run the training or inferencing model")
    parser.add_argument('--raster', action='store_true',
                            help="Run the raster version of VPRTempo, for non-neuromorphic chip inferencing")
    parser.add_argument('--sim_mat', action='store_true',
                            help="Plot a similarity matrix")
    parser.add_argument('--event_stats', action='store_true',
                            help="Runs the event stats")
    parser.add_argument('--event_type', type=str, default='random',
                            help="When using raster analysis, use CPU or GPU")
    
    # On-chip specific parameters
    parser.add_argument('--power_monitor', action='store_true',
                            help="Whether or not to use the power monitor")
    parser.add_argument('--raster_device', type=str, default='cpu',
                            help="When using raster analysis, use CPU or GPU")
    parser.add_argument('--norm', action='store_true',
                            help="Run the regular VPRTempo")
    parser.add_argument('--reference_annotation', action='store_true', 
                            help='Flag to limit frames (True) or just create all frames (False)')
    
    # Run weights and biases
    parser.add_argument('--wandb', action='store_true',
                            help="Run weights and biases")
    
    # Output base configuration
    args = parser.parse_args()

    # Run the network with the desired settings
    initialize_and_run_model(args)

if __name__ == "__main__":
    parse_network()