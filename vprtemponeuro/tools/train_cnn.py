'''
Imports
'''
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The model is defined as blew.

class DVSConv(nn.Module):
    def __init__(self):
        super(DVSConv, self).__init__()
        # Define the architecture up until the desired feature extraction layer
        self.seq =  nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # Output: [8, x/2, y/2]
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # Output: [16, x/4, y/4]
            nn.Dropout2d(0.5),
            nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # Output: [8, x/8, y/8]
            nn.Flatten(),  # Assuming we need to flatten for a dense layer that's not shown here
        )
        
        # Assuming an additional dense layer or specifying the dimension before the decoder starts
        
        # Hypothetical sizes for illustration purposes
        encoded_channels = 128  # The number of channels in the encoded representation
        final_channels = 3      # The number of channels in the output (e.g., for RGB images)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoded_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upscale
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Further upscale
            nn.ReLU(),
            # Adjust the kernel_size, stride, and padding as needed to match the spatial dimensions precisely.
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Continue upscaling
            nn.ReLU(),
            # Final layer to match the original image's channel depth and size. Adjust as necessary.
            nn.ConvTranspose2d(16, final_channels, kernel_size=2, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid()  # Assuming the original images are normalized between [0, 1]
        )

    def forward(self, x):
        encoded = self.seq(x)
        # Here, you'd need to reshape the encoded output if it was flattened and you're using dense layers in between
        decoded = self.decoder(encoded.view(-1, 8, 21, 15))  # Adjust x and y according to the actual dimensions
        return decoded
    
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load image
        image = Image.open(img_path)
        
        # Convert image to numpy array
        image_np = np.array(image)
        
        # Keep only the Red and Blue channels (assuming the order is RGB)
        # This results in a shape of (H, W, 2)
        image_np = image_np[:, :, [0, 2]]
        
        # Convert back to PIL Image to use torchvision transforms
        image = Image.fromarray(image_np)
        
        if self.transform:
            image = self.transform(image)

        return image
    
# Assuming you have a dataset `CustomDataset` for your images
transform = transforms.Compose([
    transforms.Resize((346, 240)),  # Resize the image to 346x240
    transforms.ToTensor(),  # Convert the image to PyTorch tensor
])

dataset = CustomDataset('/home/adam/repo/VPRTempoNeuro/vprtemponeuro/dataset/brisbane_event/davis/sunset1_simple', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = DVSConv().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 60

# Example training loop adaptation
for epoch in range(num_epochs):
    for data in dataloader:
        inputs = data.to(device)
        optimizer.zero_grad()
        reconstructed = model(inputs)
        loss = criterion(reconstructed, inputs)  # Reconstruction loss
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# To save:
torch.save(model.state_dict(), 'model_path.pth')