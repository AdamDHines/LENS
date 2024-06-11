#MIT License

#Copyright (c) 2024 Adam Hines, Michael Milford, Tobias Fischer

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

from lens.run_model import LENS, run_inference
from lens.train_model import LENS_Trainer, train_model
from lens.collect_data import LENS_Collector, run_collector

def generate_model_name(model):
    """
    Generate the model name based on its parameters.
    """

    model_name = (''.join(model.reference)+"_"+
            "LENS_" +
            "IN"+str(model.input)+"_" +
            "FN"+str(model.feature)+"_" + 
            "DB"+str(model.reference_places) +
            ".pth")
    return model_name

def initialize_and_run_model(args):
    """
    Initialize the model and run the desired functionality.
    """

    # args.train_new_model = True
    if args.train_model: # If user wants to train a new network
        # Initialize the model
        model = LENS_Trainer(args)
        # Generate the model name
        model_name = generate_model_name(model)
        # Train the model
        train_model(model, model_name)
    
    elif args.wandb: # Run a weights and biases sweep    
        # Log into weights & biases
        wandb.login()
        
        # Define the method and parameters for grid search
        sweep_config = {'method':'random'}
        metric = {'name':'AUC', 'goal':'maximize'}
        sweep_config['metric'] = metric
        
        # Define the parameters for the search
        parameters_dict = {
                        }
        sweep_config['parameters'] = parameters_dict
        pprint.pprint(sweep_config)
    
        # Start sweep controller
        sweep_id = wandb.sweep(sweep_config, project="")

        # Initialize w&b search
        def wandsearch(config=None):
            with wandb.init(config=config):
                # Initialize config
                config = wandb.config

                # Set arguments for the sweep


                # Initialize the training model
                args.train_new_model = True
                model = VPRTempoTrain(args)
                model_name = generate_model_name(model)

                train_new_model(model, model_name)
                
                # Initialize the inference model
                model = VPRTempo(args)

                # Run the inference model
                R_all = run_inference_norm(model, model_name)
                
                # Evaluate the model
                x = [1,5,10,15,20,25]
                AUC= np.trapz(R_all, x)
                wandb.log({"AUC" : AUC})
                print("AUC: ", AUC)

        # Start the agent with the sweeps
        wandb.agent(sweep_id,wandsearch)

    else: # Run the inference network
        if args.raster: # Runs the sinabs model on CPU/GPU hardware
            # Initialize the quantized model
            model = VPRTempoRaster(args)
            # Generate the model name
            model_name = generate_model_name(model)
            # Run the quantized inference model
            run_inference_raster(model, model_name)
        elif args.norm: # Runs base VPRTempo
            # Initialize the model
            model = VPRTempo(args)
            # Generate the model name
            model_name = generate_model_name(model)
            # Run the inference model
            run_inference_norm(model, model_name)
        else:
            # Initialize the model
            model = LENS(args) # Runs the DynapCNN on-chip model
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
    parser.add_argument('--dataset', type=str, default='qcr',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--camera', type=str, default='speck',
                            help="Camera to use for training and/or inferencing")
    parser.add_argument('--reference', type=str, default='trolley-ref',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--query', type=str, default='trolley-qry',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--data_dir', type=str, default='./vprtemponeuro/dataset/',
                            help="Directory where dataset files are stored")
    parser.add_argument('--reference_places', type=int, default=78,
                            help="Number of places to use for training and/or inferencing")
    parser.add_argument('--query_places', type=int, default=90,
                            help="Number of places to use for training and/or inferencing")
    parser.add_argument('--sequence_length', type=int, default=3,
                        help="Length of the sequence matcher")
    parser.add_argument('--feature_multiplier', type=float, default=2.0,
                        help="Size multiplier for the feature/hidden layer")

    # Define training parameters
    parser.add_argument('--filter', type=int, default=1,
                            help="Images to skip for training and/or inferencing")
    parser.add_argument('--epoch_feat', type=int, default=128,
                            help="Number of epochs to train the model")
    parser.add_argument('--epoch_out', type=int, default=128,
                            help="Number of epochs to train the model")
    
    # Hyperparameters - feature layer
    parser.add_argument('--thr_l_feat', type=float, default=0,
                        help="Low threshold value")
    parser.add_argument('--thr_h_feat', type=float, default=0.75,
                            help="High threshold value")
    parser.add_argument('--fire_l_feat', type=float, default=0.4,
                            help="Low threshold value")
    parser.add_argument('--fire_h_feat', type=float, default=0.6,
                            help="High threshold value")
    parser.add_argument('--const_input_l', type=float, default=0.0,
                            help="Low constant input value"),
    parser.add_argument('--const_input_h', type=float, default=0.0,
                            help="High constant input value"),
    parser.add_argument('--ip_rate_feat', type=float, default=0.02,
                            help="ITP learning rate")
    parser.add_argument('--stdp_rate_feat', type=float, default=0.01,
                            help="STDP learning rate")
    
   # Hyperparameters - output layer 
    parser.add_argument('--thr_l_out', type=float, default=0,
                        help="Low threshold value")
    parser.add_argument('--thr_h_out', type=float, default=0.5,
                            help="High threshold value")
    parser.add_argument('--fire_l_out', type=float, default=0.5,
                            help="Low threshold value")
    parser.add_argument('--fire_h_out', type=float, default=0.5,
                            help="High threshold value")
    parser.add_argument('--ip_rate_out', type=float, default=0.02,
                            help="ITP learning rate")
    parser.add_argument('--stdp_rate_out', type=float, default=0.01,
                            help="STDP learning rate")

    # Connection probabilities
    parser.add_argument('--f_exc', type=float, default=0.1,
                        help="Feature layer excitatory connection")
    parser.add_argument('--f_inh', type=float, default=0.5,
                        help="Feature layer inhibitory connection")
    parser.add_argument('--o_exc', type=float, default=1.0,
                        help="Output layer excitatory connection")
    parser.add_argument('--o_inh', type=float, default=1.0,
                        help="Output layer inhibitory connection")
    
    # Define image transformation parameters
    parser.add_argument('--dims', nargs='+', type=int, default=[10,10],
                            help="Dimensions to resize the image to")
    parser.add_argument('--convolve_events', action='store_true',
                            help="Decide to convolve the events or not")

    # Define the network functionality
    parser.add_argument('--train_model', action='store_true',
                            help="Flag to run the training or inferencing model")
    parser.add_argument('--raster', action='store_true',
                            help="Run the raster version of VPRTempo, for non-neuromorphic chip inferencing")
    parser.add_argument('--sim_mat', action='store_true',
                            help="Plot a similarity matrix")
    parser.add_argument('--PR_curve', action='store_true',
                            help="Plot a precision recall curve")
    parser.add_argument('--matching', action='store_true',
                            help="Perform matching to GT, if available")
    
    # On-chip specific parameters
    parser.add_argument('--power_monitor', action='store_true',
                            help="Whether or not to use the power monitor")
    parser.add_argument('--raster_device', type=str, default='cpu',
                            help="When using raster analysis, use CPU or GPU")
    parser.add_argument('--norm', action='store_true',
                            help="Run the regular VPRTempo")
    parser.add_argument('--reference_annotation', action='store_true', 
                            help='Flag to limit frames (True) or just create all frames (False)')
    parser.add_argument('--event_driven', action='store_true', 
                            help='Define the source of the input data to be from the speck event sensor')
    parser.add_argument('--simulated_speck', action='store_true', 
                            help='Run time based simulation on the Speck2fDevKit')
    # Run weights and biases
    parser.add_argument('--wandb', action='store_true',
                            help="Run weights and biases")
    
    # Output base configuration
    args = parser.parse_args()

    # Run the network with the desired settings
    initialize_and_run_model(args)

if __name__ == "__main__":
    parse_network()