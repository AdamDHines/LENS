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
import wandb
import pprint

import numpy as np

from vprtemponeuro.src.event_stats import EventStats

from vprtemponeuro.VPRTempoNeuro import VPRTempoNeuro, run_inference
from vprtemponeuro.VPRTempoRaster import VPRTempoRaster, run_inference_raster
from vprtemponeuro.VPRTempoTrain import VPRTempoTrain, generate_model_name, train_new_model, check_pretrained_model
from vprtemponeuro.VPRTempo import VPRTempo, run_inference_norm

def initialize_and_run_model(args):
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
        # Check if the model has been trained before
        #check_pretrained_model(model_name)
        # Train the model
        train_new_model(model, model_name)
    elif args.wandb:
        # log into weights & biases
        wandb.login()
        
        # define the method and parameters for grid search
        sweep_config = {'method':'random'}
        metric = {'name':'AUC', 'goal':'maximize'}
        
        sweep_config['metric'] = metric
        
        parameters_dict = {
                        'fire_l': {'values':[0.03, 0.05, 0.07]},
                        'fire_h': {'values':[0.25, 0.3, 0.35]},
                        'thr_l': {'values':[0.1, 0.15, 0.2]},
                        'thr_h': {'values':[0.2, 0.25, 0.3]},
                        'ip_rate': {'values':[0.001,0.01,0.1]},
                        'stdp_rate': {'values':[0.0001,0.001,0.01]},
                        'f_exc': {'values':[0.1,0.15,0.25]},
                        'f_inh': {'values':[0.25,0.5,0.75]},
                        'epoch': {'values':[10,25,50]}
        }
        
        sweep_config['parameters'] = parameters_dict
        pprint.pprint(sweep_config)
    
        # start sweep controller
        sweep_id = wandb.sweep(sweep_config, project="vprtemponeuro-dvs_RANDOM")

        # Initialize w&b search
        def wandsearch(config=None):
            with wandb.init(config=config):
                # Initialize config
                config = wandb.config

                # Set arguments for the sweeps
                #args.epoch = config.epochs
                #args.patches = config.patches
                #args.repeat = config.repeat

                args.fire_l = config.fire_l
                args.fire_h = config.fire_h
                args.thr_l = config.thr_l
                args.thr_h = config.thr_h
                args.ip_rate = config.ip_rate
                args.stdp_rate = config.stdp_rate
                args.f_exc = config.f_exc
                args.f_inh = config.f_inh
                args.epoch = config.epoch

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

                x = [1,5,10,15,20,25]
                AUC= np.trapz(R_all, x)
                wandb.log({"AUC" : AUC})
                
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

def parse_network(use_raster=False, train_new_model=False, norm=False, wandb=False, event_stats=False):
    '''
    Define the base parameter parser (configurable by the user)
    '''
    parser = argparse.ArgumentParser(description="Args for base configuration file")

    # Define the dataset arguments
    parser.add_argument('--dataset', type=str, default='brisbane_event',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--data_dir', type=str, default='./vprtemponeuro/dataset/Brisbane-Event',
                            help="Directory where dataset files are stored")
    parser.add_argument('--database_places', type=int, default=259,
                            help="Number of places to use for training and/or inferencing")
    parser.add_argument('--query_places', type=int, default=259,
                            help="Number of places to use for training and/or inferencing")
    parser.add_argument('--num_modules', type=int, default=1,
                            help="Number of expert modules to use split images into")
    parser.add_argument('--database_dirs', nargs='+', default=['database_filtered'],
                            help="Directories to use for training")
    parser.add_argument('--query_dir', nargs='+', default=['query_filtered'],
                            help="Directories to use for testing")
    parser.add_argument('--pixels', type=int, default=784,
                        help="Number of places to use for training and/or inferencing")

    # Define training parameters
    parser.add_argument('--filter', type=int, default=1,
                            help="Images to skip for training and/or inferencing")
    parser.add_argument('--epoch', type=int, default=4,
                            help="Number of epochs to train the model")
    
    # Hyperparameters
    parser.add_argument('--thr_l', type=int, default=0,
                        help="Low threshold value")
    parser.add_argument('--thr_h', type=int, default=0.5,
                        help="High threshold value")
    parser.add_argument('--fire_l', type=int, default=0.2,
                        help="Low threshold value")
    parser.add_argument('--fire_h', type=int, default=0.9,
                        help="High threshold value")
    parser.add_argument('--ip_rate', type=int, default=0.25,
                        help="ITP learning rate")
    parser.add_argument('--stdp_rate', type=int, default=0.001,
                        help="STDP learning rate")
    
    # Connection probabilities
    parser.add_argument('--f_exc', type=int, default=0.1,
                        help="Feature layer excitatory connection")
    parser.add_argument('--f_inh', type=int, default=0.25,
                        help="Feature layer inhibitory connection")
    parser.add_argument('--o_exc', type=int, default=1,
                        help="Output layer excitatory connection")
    parser.add_argument('--o_inh', type=int, default=1,
                        help="Output layer inhibitory connection")
    
    # Define image transformation parameters
    parser.add_argument('--dims', nargs='+', type=int, default=[56,56],
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
    
    # Run weights and biases
    parser.add_argument('--wandb', action='store_true',
                            help="Run weights and biases")
    
    # If the function is called with specific arguments, override sys.argv
    #if use_raster or train_new_model or norm or wandb or event_stats:
    #    sys.argv = ['']
    #    if use_raster:
    #        sys.argv.append('--raster')
    #    if train_new_model:
    #        sys.argv.append('--train_new_model')
    #    if norm:
    #        sys.argv.append('--norm')
    #    if wandb:
    #        sys.argv.append('--wandb')
    #    if event_stats:
    #        sys.argv.append('--event_stats')

    # Output base configuration
    args = parser.parse_args()

    # Run the network with the desired settings
    initialize_and_run_model(args)

if __name__ == "__main__":
    # User input to determine if using quantized network or to train new model 
    parse_network(
                use_raster=False, 
                train_new_model=False,
                norm=False,
                wandb = False,
                event_stats = False
                )