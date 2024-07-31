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

import argparse


def generate_model_name(model):
    """
    Generate the model name based on its parameters.
    """
    # Define model name based on the parameters
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
    if args.train_model: # If user wants to train a new network
        from lens.train_model import LENS_Trainer, train_model
        # Initialize the model
        model = LENS_Trainer(args)
        # Generate the model name
        model_name = generate_model_name(model)
        # Train the model
        train_model(model, model_name)
    elif args.collect_data:  # If user wants to collect data to train new model
        from lens.collect_data import LENS_Collector, run_collector
        # Initialize the model
        model = LENS_Collector(args)
        # Collect the data
        run_collector(model)
    elif args.event_driven:
        from lens.run_speck import LENSSpeck, run_speck
        # Initialize the model
        model = LENSSpeck(args)
        # Generate the model name
        model_name = generate_model_name(model)
        # Run the model on the Speck2fDevKit
        run_speck(model, model_name)
    else: # Run the inference network
        from lens.run_model import LENS, run_inference
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
    parser.add_argument('--data_name', type=str, default='experiment001',
                            help="Define dataset same for data collection")
    parser.add_argument('--reference', type=str, default='indoor-reference',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--query', type=str, default='indoor-query',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--data_dir', type=str, default='./lens/dataset/',
                            help="Directory where dataset files are stored")
    parser.add_argument('--reference_places', type=int, default=75,
                            help="Number of places to use for training and/or inferencing")
    parser.add_argument('--query_places', type=int, default=75,
                            help="Number of places to use for training and/or inferencing")
    parser.add_argument('--sequence_length', type=int, default=4,
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
    parser.add_argument('--dims', type=int, default=10,
                            help="Dimensions to resize the image to")
    parser.add_argument('--roi_dim', type=int, default=80,
                            help="Dimensions to resize the image to")
    
    # Define the network functionality
    parser.add_argument('--train_model', action='store_true',
                            help="Flag to run the training or inferencing model")
    parser.add_argument('--sim_mat', action='store_true',
                            help="Plot a similarity matrix")
    parser.add_argument('--PR_curve', action='store_true',
                            help="Plot a precision recall curve")
    parser.add_argument('--matching', action='store_true',
                            help="Perform matching to GT, if available")
    parser.add_argument('--timebin', type=int, default=1000,
                        help="dt for spike collection window and time based simulation")
    
    # On-chip specific parameters
    parser.add_argument('--event_driven', action='store_true', 
                            help='Run the online inferencing model on Speck2fDevKit')
    parser.add_argument('--simulated_speck', action='store_true', 
                            help='Run time based simulation on the Speck2fDevKit')
    parser.add_argument('--collect_data', action='store_true', 
                            help='Collect images from SPECK to train new model')
    parser.add_argument('--headless', action='store_true',
                            help="Runs the Speck2fDevKit in headless mode")
    parser.add_argument('--save_input', action='store_true',
                            help="Collects and saves the input spikes as NumPy arrays")
    
    # Output base configuration
    args = parser.parse_args()

    # Run the network with the desired settings
    initialize_and_run_model(args)

if __name__ == "__main__":
    parse_network()
