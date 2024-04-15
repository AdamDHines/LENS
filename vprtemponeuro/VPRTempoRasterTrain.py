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

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch.nn.functional as F
import torch.optim as optim

class MaximalOutputLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MaximalOutputLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs, target_index):
        """
        outputs: The raw scores from the output layer of the model (shape: [batch_size, num_classes]).
        target_index: The index of the correct class for each example in the batch (shape: [batch_size]).
        """
        # Get the scores of the correct class
        target_index = target_index.to(dtype=torch.int64)
        correct_scores = outputs.gather(1, target_index.unsqueeze(1)).squeeze()

        # Apply the margin ranking loss
        loss = 0
        for i in range(outputs.size(1)):
            if i != target_index:
                # Calculate the margin loss for each incorrect class
                incorrect_scores = outputs[:, i]
                # Maximize the difference between the correct class score and each incorrect class score
                loss += F.relu(self.margin - (correct_scores - incorrect_scores)).mean()

        return loss

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

        """
        Define trainable layers here
        """

        self.add_layer(
            'output_layer',
            dims=[24, 118],
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

        # define a CNN model
        self.train_sequential = nn.Sequential(
            # 2 x 128 x 128
            # Core 0
            nn.Conv2d(1, 8, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),  # 8, 64, 64
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 8,32,32
            # """Core 1"""
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 16, 32, 32
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 16, 16, 16
            # """Core 2"""
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 16, 16
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 16, 16, 16
            nn.Flatten(),
            nn.Dropout(0.75),
            nn.Linear(8 * 8 * 8, 113, bias=False)
        )


        for cnnlayer in self.train_sequential.modules():
            if isinstance(cnnlayer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(cnnlayer.weight.data)
        self.train_sequential.to(self.device)
        # Set the total timestep count
        self.T = int((self.reference_places) * self.epoch_feat)
        # Initialize the tqdm progress bar
        pbar = tqdm(total=int(self.T),
                    desc="Training ",
                    position=0)
        lr = 0.1
        batch_size = 4
        optimizer = SGD(params=self.train_sequential.parameters(), lr=lr)
        criterion = CrossEntropyLoss()
        # Initialize the learning rates for each layer (used for annealment)
        init_itp = layer.eta_stdp.detach() * 2
        init_stdp = layer.eta_stdp.detach()
        mod = 0  # Used to determine the learning rate annealment, resets at each epoch
        lossav = []
        k = 6
        # Run training for the specified number of epochs
        for _ in range(self.epoch_feat):
            # Run training for the specified number of timesteps
            for spikes, labels, gps, _ in train_loader:
                spikes, labels = spikes.to(self.device), labels.to(self.device)
                spikes = spikes.to(torch.float32)
                # spikes = torch.squeeze(spikes,0)
                idx = labels / self.filter # Set output index for spike forcing

                # spikes = torch.reshape(spikes,(128,128))
                # spikes = spikes.unsqueeze(0)
                # spikes = spikes.unsqueeze(0)
                optimizer.zero_grad()
                output = self.forward(spikes) # Current layer spikes
                loss = criterion(output, idx.long())
                # backward
                loss.backward()
                optimizer.step()
                # spikes_noclp = output_spikes.detach() # Used for inhibitory homeostasis
                # output_spikes = bn.clamp_spikes(output_spikes, self.output_layer) # Clamp spikes [0, 0.9]
                
                # # Calculate STDP
                # layer = bn.calc_stdp(feature_spikes,output_spikes,spikes_noclp,self.output_layer, idx)
                # # Adjust learning rates
                # layer = self._anneal_learning_rate(layer, mod, init_itp, init_stdp)
                # Update the annealing mod & progress bar 
                mod += 1
                pbar.update(1)
                if mod % 100 == 0:
                    pbar.set_description(f"Training Loss: {round(np.mean(lossav), 4)}")
                    lossav = []
                else:
                    lossav.append(loss.item())

        # Close the tqdm progress bar
        pbar.close()

        # Free up memory
        if self.device == "cuda:0":
            torch.cuda.empty_cache()
            gc.collect()


    def forward(self,spikes):
        """
        Compute the forward pass of the model.
    
        Parameters:
        - spikes (Tensor): Input spikes.
    
        Returns:
        - Tensor: Output after processing.
        """
        # Function to be called by the hook

        spikes = self.train_sequential(spikes)

        return spikes 
    
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
                                      is_raster=False)
    # Initialize the data loader
    train_loader = DataLoader(train_dataset, 
                              batch_size=1, 
                              shuffle=True,
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