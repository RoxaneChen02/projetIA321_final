import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms as T
from collections import OrderedDict
import os

### Code inspired by TP on VAE of the Representation Learning Course M2 DataScience at IPP 

class VAE(nn.Module):
    """
    A simple convolutional autoencoder. Processes RGB images in tensor form and reconstructs
    3 channel RGB image tensors from a 32 dimensional latent space.  
    """
    def __init__(self, latent_size=32):
        super().__init__()
        self.latent_size = latent_size
        self.device = None
        
        # encoder
        self.enc_conv1 = nn.Conv2d(3,32,kernel_size=4,stride=2, padding=0)
        self.enc_conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2, padding=0)
        self.enc_conv3 = nn.Conv2d(64,128,kernel_size=4,stride=2, padding=0)
        self.enc_conv4 = nn.Conv2d(128,256,kernel_size=4,stride=2, padding=0)
        
        # z
        self.mu = nn.Linear(1024, latent_size)
        self.logvar = nn.Linear(1024, latent_size)
        
        # decoder
        self.de_fc = nn.Linear(latent_size, 1024)
        self.dec_conv1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2, padding=0)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, padding=0)
        
    
    def set_device(self, device):
        self.device = device
        
    def forward(self, x):
        
        shape = x.shape
        mu, logvar = self.encode(x)
        z= self.latent(mu, logvar)
        
        out = self.decode(z).reshape(shape)
        
        return out, mu, logvar   
        
    def encode(self, x):
        batch_size = x.shape[0]
        out = F.relu(self.enc_conv1(x))
        out = F.relu(self.enc_conv2(out))
        out = F.relu(self.enc_conv3(out))
        out = F.relu(self.enc_conv4(out))
        out = torch.flatten(out, start_dim=1)
        mu = self.mu(out)
        logvar = self.logvar(out)
        
        return mu, logvar
        
    def decode(self, z):
        
        out = self.de_fc(z)
        out = out.view(-1, 1024, 1, 1)
        out = F.relu(self.dec_conv1(out))
        out = F.relu(self.dec_conv2(out))
        out = F.relu(self.dec_conv3(out))
        out = torch.sigmoid(self.dec_conv4(out))
        
        return out
        
        
    def latent(self, mu, logvar):
        sigma = torch.exp(0.5*logvar)
        eps = torch.randn_like(logvar).to(self.device)
        z = mu + eps*sigma
        return z
    
    def obs_to_z(self, x):
        mu, logvar = self.encode(x)
        z = self.latent(mu, logvar)
        return z

    def sample(self, z):
        out = self.decode(z)
        return out
    
    def vae_loss(self, out, y, mu, logvar):
        batch_size = out.shape[0]
    
    
        BCE = F.mse_loss(out, y, reduction="sum")

        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        BCE /= batch_size
        KL /= batch_size
        
        return BCE + KL, BCE, KL

    def get_latent_size(self):
        
        return self.latent_size

    def save(self,path="./model/vae_model"):
        os.makedirs("model", exist_ok=True)
        torch.save(self.state_dict(), path)

        


    