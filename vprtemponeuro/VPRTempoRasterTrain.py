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

import os
import gc
import torch

import numpy as np
import torch.nn as nn
import sinabs.layers as sl
import vprtemponeuro.src.blitnet as bn
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader
from sinabs.from_torch import from_model
from vprtemponeuro.src.loggers import model_logger
from vprtemponeuro.src.dataset import CustomImageDataset, ProcessImage

class VPRTempoRasterTrain(nn.Module):
    def __init__(self, args):
        super(VPRTempoRasterTrain, self).__init__()

        # Set the arguments
        self.args = args
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

        # Set the dataset file
        if self.reference_annotation:
            self.dataset_file = os.path.join(self.data_dir, self.reference + '_reference.csv')
        else:
            self.dataset_file = os.path.join(self.data_dir, self.reference + '.csv')

        # Set the reference image folder
        self.reference_dir = os.path.join(self.data_dir, self.dataset, self.camera, self.reference)

        # Configure the model logger and get the device
        self.device = model_logger(self)  

        # Layer dict to keep track of layer names and their order
        self.layer_dict = {}
        self.layer_counter = 0

        # Define layer architecture
        self.input = int(args.dims[0]*args.dims[1])
        self.feature = int(self.input*args.feature_multiplier)
        self.output = int(args.reference_places)

        """
        Define trainable layers here
        """
        self.add_layer(
            'feature_layer',
            dims=[self.input, self.feature],
            thr_range=[self.thr_l_feat, self.thr_h_feat],
            fire_rate=[self.fire_l_feat, self.fire_h_feat],
            ip_rate=self.ip_rate_feat,
            stdp_rate=self.stdp_rate_feat,
            p=[self.f_exc, self.f_inh],
            device=self.device
        )
        self.add_layer(
            'output_layer',
            dims=[self.feature, self.output],
            thr_range=[self.thr_l_out, self.thr_h_out],
            fire_rate=[self.fire_l_out, self.fire_h_out],
            ip_rate=self.ip_rate_out,
            stdp_rate=self.stdp_rate_out,
            p=[self.o_exc, self.o_inh],
            spk_force=True,
            device=self.device
        )

        self.relu_outputs = None
        self.hook = None
        
    def add_layer(self, name, **kwargs):
        """
        Dynamically add a layer with given name and keyword arguments.
        
        :param name: Name of the layer to be added
        :type name: str
        :param kwargs: Hyperparameters for the layer
        """
        # Check for layer name duplicates
        if name in self.layer_dict:
            raise ValueError(f"Layer with name {name} already exists.")
        
        # Add a new SNNLayer with provided kwargs
        setattr(self, name, bn.SNNLayer(**kwargs))
        
        # Add layer name and index to the layer_dict
        self.layer_dict[name] = self.layer_counter
        self.layer_counter += 1                           
        
    def model_logger(self):
        """
        Log the model configuration to the console.
        """
        model_logger(self)

    def _anneal_learning_rate(self, layer, mod, itp, stdp):
        """
        Anneal the learning rate for the current layer.
        """
        if np.mod(mod, 10) == 0: # Modify learning rate every 100 timesteps
            pt = pow(float(self.T - mod) / self.T, 2)
            layer.eta_ip = torch.mul(itp, pt) # Anneal intrinsic threshold plasticity learning rate
            layer.eta_stdp = torch.mul(stdp, pt) # Anneal STDP learning rate
        return layer

    def train_model(self, model, train_loader, layer, feature=False):
        """
        Train a layer of the network model.

        :param train_loader: Training data loader
        :param layer: Layer to train
        :param prev_layers: Previous layers to pass data through
        """

        # Define the inferencing model
        # ReLU layer required for the sinabs model, this becomes the spiking layer

        # if feature:
        #     self.train_sequential = nn.Sequential(
        #         self.feature_layer.w,
        #         nn.ReLU()
        #     )
            
        #     # Set up the sinabs model
        #     input_shape = (1, 1, self.dims[0] * self.dims[1])
        #     self.sinabs_model = from_model(
        #                             self.train_sequential, 
        #                             input_shape=input_shape,
        #                             batch_size=1,
        #                             add_spiking_output=False,
        #                             ) 
        # else:
        self.train_sequential = nn.Sequential(
            self.feature_layer.w,
            nn.ReLU(),
            self.output_layer.w
        )
        
        # Set up the sinabs model
        input_shape = (1, 1, self.dims[0] * self.dims[1])
        self.sinabs_model = from_model(
                                self.train_sequential, 
                                input_shape=input_shape,
                                batch_size=1,
                                add_spiking_output=True,
                                )       
            # model.relu_output = None
        
        if feature:
            self.epoch = self.args.epoch_feat
            prev_layer = None
        else:
            prev_layer = getattr(self, 'feature_layer')
            self.epoch = self.args.epoch_out
        # Set the total timestep count
        self.T = int((self.reference_places) * self.epoch)
        # Initialize the tqdm progress bar
        pbar = tqdm(total=int(self.T),
                    desc="Training ",
                    position=0)
                    
        # Initialize the learning rates for each layer (used for annealment)
        init_itp = layer.eta_ip.detach()
        init_stdp = layer.eta_stdp.detach()
        mod = 0  # Used to determine the learning rate annealment, resets at each epoch
        # Run training for the specified number of epochs
        #with torch.no_grad():
        for _ in range(self.epoch):
            # Run training for the specified number of timesteps
            for spikes, labels, _, spikes_og in train_loader:
                spikes, labels, spikes_og = spikes.to(self.device), labels.to(self.device), spikes_og.to(self.device)
                pre_spike = spikes.detach() # Previous layer spikes for STDP
                pre_spike = pre_spike.sum(dim=0).squeeze()/255
                spikes = sl.FlattenTime()(spikes)
                self.sinabs_model.reset_states()
                idx = labels / self.filter # Set output index for spike forcing
                # Pass through previous layers if they exist
                # if prev_layers:
                #     with torch.no_grad():
                #         for prev_layer_name in prev_layers:
                #             prev_layer = getattr(self, prev_layer_name) # Get the previous layer object
                #             spikes = self.forward(spikes, prev_layer) # Pass spikes through the previous layer
                #             spikes = bn.clamp_spikes(spikes, prev_layer) # Clamp spikes [0, 0.9]
                # else:
                #     prev_layer = None
                # Get the output spikes from the current layer
                #pre_spike = spikes.detach() # Previous layer spikes for STDP
                #pre_spike = pre_spike.sum(dim=0).squeeze()/255
                spikes = self.forward(self.sinabs_model, spikes,feature) # Current layer spikes
                spikes_noclp = spikes.detach().to(self.device) # Used for inhibitory homeostasis
                spikes = spikes.squeeze(0).to(self.device)
                spikes = bn.clamp_spikes(spikes, layer) # Clamp spikes [0, 0.9]
                # Calculate STDP
                layer = bn.calc_stdp(spikes_og,spikes,spikes_noclp,layer, idx, prev_layer=prev_layer)
                #print(torch.mean(layer.w.weight))
                # Adjust learning rates
                layer = self._anneal_learning_rate(layer, mod, init_itp, init_stdp)
                # Update the annealing mod & progress bar 
                mod += 1
                pbar.update(1)

        # Close the tqdm progress bar
        pbar.close()
        # Free up memory
        if self.device == "cuda:0":
            torch.cuda.empty_cache()
            gc.collect()

    def get_relu_output(self, module, input, output):
        self.relu_outputs = output.detach().cpu()

    def forward(self, model, spikes, feature=False):
        """
        Compute the forward pass of the model.
    
        Parameters:
        - spikes (Tensor): Input spikes.
    
        Returns:
        - Tensor: Output after processing.
        """
        # Function to be called by the hook

        if feature:
            if self.hook is None:  # Ensure the hook is registered only once
                relu_layer = model.spiking_model[1]
                self.hook = relu_layer.register_forward_hook(self.get_relu_output)
            spikes = self.sinabs_model(spikes)
            spikes = self.relu_outputs.detach()
            self.relu_outputs = None
        else:
            spikes = self.sinabs_model(spikes)
        output = spikes.detach().sum(dim=0).squeeze()/255

        return output 
    
    def save_model(self, model_out):    
        """
        Save the trained model to models output folder.
        """
        torch.save(self.state_dict(), model_out) 
            
def generate_model_name(model):
    """
    Generate the model name based on its parameters.
    """
    return ("VPRTempo" +
            str(model.input) +
            str(model.feature) +
            str(model.output) +
            str(model.num_modules) +
            '.pth')

def check_pretrained_model(model_name):
    """
    Check if a pre-trained model exists and prompt the user to retrain if desired.
    """
    if os.path.exists(os.path.join('./vprtemponeuro/models', model_name)):
        prompt = "A network with these parameters exists, re-train network? (y/n):\n"
        retrain = input(prompt).strip().lower()
        return retrain == 'n'
    return False

def train_new_model_raster(model, model_name):
    """
    Train a new model.

    :param model: Model to train
    :param model_name: Name of the model to save after training
    :param qconfig: Quantization configuration
    """
    # Initialize the image transforms and datasets
    image_transform = transforms.Compose([
                                        ProcessImage()
                                            ])
    train_dataset =  CustomImageDataset(annotations_file=model.dataset_file, 
                                      img_dir=model.reference_dir,
                                      transform=image_transform,
                                      skip=model.filter,
                                      max_samples=model.reference_places,
                                      test=False,
                                      is_raster=True)
    # Initialize the data loader
    train_loader = DataLoader(train_dataset, 
                              batch_size=1, 
                              shuffle=False,
                              num_workers=1,
                              persistent_workers=False)
    # Set the model to training mode and move to device
    model.train()
    # Keep track of trained layers to pass data through them
    trained_layers = [] 
    # Training each layer
    feature = True
    for layer_name, _ in sorted(model.layer_dict.items(), key=lambda item: item[1]):
        print(f"Training layer: {layer_name}")
        # Retrieve the layer object
        layer = getattr(model, layer_name)
        # Train the layer
        model.train_model(model, train_loader, layer, feature=feature)
        # After training the current layer, add it to the list of trained layers
        trained_layers.append(layer_name)
        feature = False
    # Convert the model to a quantized model
    model.eval()
    # Save the model
    model.save_model(os.path.join('./vprtemponeuro/models', model_name))    