import torch
from vae.vae import VAE
from vae.CarRacingDataset import CarRacingDataset
from matplotlib import pyplot as plt
from torchvision import transforms
from torch import nn
import os
import torchvision
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def train(model, dataset, epochs, learning_rate, batch_size, device, verbose=1):
    """Trains the VAE using the given dataset.

    #Inputs :
    - dataset (torch.utils.data.Dataset object): CarRacingDataset containing data to train 
    - epochs : number of epoch to train the vae for
    - learning_rate
    - batch_size
    - device 
    - verbose
        
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        with tqdm(dataloader, unit="batch") as tepoch:
            for y in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                # Make 'prediciton' and calculate loss
                y = y.to(device)
                out, mu, logvar = model(y)
                loss,_,_ =  vae.vae_loss( out, y, mu, logvar)
                epoch_losses.append(loss.detach())
                    
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        
        losses.append(torch.stack(epoch_losses).mean())
        
        if verbose == 1:
            print("Epoch {}/{} : Loss {}".format(epoch+1, epochs, torch.stack(epoch_losses).mean()))
            
    # plot losses 
    x = list(range(epochs))
    y = [t.cpu().numpy() for t in losses]

    plt.figure()
    plt.plot(x, y)
    plt.savefig("{}.png".format("loss_plot"))

def print_result(model, dataset, indicies, device, to_file=True, filename="reconstruction_examples"):
    """Encodes and decodes n random images from a dataset
        
    params:
        dataset (object): A torch.utils.data.Dataset object containing the dataset to encode
                              and decode randomly images from
        n (int): How many random images to encode and decode.
    """
    ims = []
    for i in indicies:
        im= dataset[i]
        ims.append(im)
    ims = torch.stack(ims).to(device)
    output,_,_ = model.forward(ims)

    combined = torch.cat((ims, output))
    grid = torchvision.utils.make_grid(combined, len(indicies))

    if to_file:
        torchvision.utils.save_image(grid, "{}.png".format(filename))
    else:
        plt.figure()
        plt.imshow(grid.cpu().detach().numpy().transpose(1, 2, 0))
        plt.show()

if __name__ == "__main__" :
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Work on ', device)
    
    dataset = CarRacingDataset(transform = transforms.Compose([
                                    transforms.ToPILImage() , 
                                    transforms.Resize((64, 64)),
                                    transforms.ToTensor()
                                ]))
    
    
    dataset.load("data/dataset")
    
    vae = VAE(latent_size=32)
    vae.set_device(device)
    vae.to(device)
    
    print("Start Training. ")
    
    train(vae, dataset, epochs = 10, batch_size= 64, learning_rate =0.001, device= device )
    
    print("Save model")
    
    vae.save()
    
    print("Save Reconstruction examples")
    
    print_result(vae, dataset, indicies = np.random.randint(0, len(dataset) - 1, 10), device= device )