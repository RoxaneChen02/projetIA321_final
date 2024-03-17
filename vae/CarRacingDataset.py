
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.envs.box2d import CarRacing
from matplotlib import pyplot as plt



class CarRacingDataset(Dataset):
    """Class to create a dataset from a gymnasium CarRacing-v2 environment"""
    
    def __init__(self, transform=transforms.ToTensor()):
        
        self.transform = transform
        self.dataset = []
        

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, index):
        """Returns the data at the given index"""
        return self.transform(self.dataset[index])

    
    def generate_action(self, prev_action):
        
        """
        Generates random actions in the gymnasium CarRacing-v2 .
        The actions are biased towards acceleration to induce exploration of the environment.  
        Inspired by https://github.com/timoklein/car_racer/tree/master but adapted for newer version of gymnasium 
        ## Input:  
        
        - prev_action (ndarray): Array with 3 elements representing the previous action.     
        
        ## Output:  
        
        - action (ndarray): Array with 3 elements representing the new sampled action.
        """
        
        if np.random.randint(3) % 3:
            return prev_action

        index = np.random.randn(3)
        # Favor acceleration over the others:
        index[1] = np.abs(index[1])
        index = np.argmax(index)
        mask = np.zeros(3)
        mask[index] = 1

        action = np.random.randn(3)
        action = np.tanh(action)
        action[1] = (action[1] + 1) / 2
        action[2] = (action[2] + 1) / 2

        return action*mask
    
    def collect(self, num_episodes= 80, timestep_per_episode = 170):
        
        """
        Collect dataset by simulating several episode and saving the observation during several timesteps
        
        ## Input:
        - num_episodes : number of episode to simulate
        - timestep_per_episode
        
        """
        
        env = CarRacing(render_mode="rgb_array")
        obs_data = []
        
        #sample first random action 
        action = env.action_space.sample()
        
        for _ in range(num_episodes):
            observation = env.reset()
            
            # Make the Car start at random positions in the race-track
            position = np.random.randint(len(env.track))
            env.car = Car(env.world, *env.track[position][1:4])

            for i in range(timestep_per_episode):
                env.render()
                action = self.generate_action(action)
                observation, _, done, _,_ = env.step(action)
                if i > 20: # we don't save the first few frame
                    obs_data.append(observation)
                
        env.close()
        self.dataset = np.array(obs_data)


    def save(self, filepath='data/dataset'):
        """Saves the dataset to file"""
        os.makedirs("data", exist_ok=True)
        if len(self) == 0:
            print("No data in this dataset, cannot save.")
            return
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.dataset, f)


    def load(self, filepath='data/dataset'):
        """Load a dataset from file"""
        
        if not os.path.isfile(filepath):
            print("File not found. Check path or collect dataset")
            return

        with open(filepath, 'rb') as f:
            self.dataset = pickle.load(f)
    
    
    def print_random_data(self, to_file = False):
        """
        Print random image from dataset
        """
        ims = []
        indicies = np.random.randint(0, len(self) - 1, 10)
        for i in indicies:
            im= dataset[i]
            ims.append(im)
        ims = torch.stack(ims)
        grid = torchvision.utils.make_grid(ims, len(indicies))
        
        if to_file==True:
            torchvision.utils.save_image(grid, "./{}.png".format('example'))
        else :
            plt.figure()
            plt.imshow(grid.cpu().detach().numpy().transpose(1, 2, 0))
            plt.show()
            
            
if __name__ == "__main__":
    
    dataset = CarRacingDataset()
    print("Start Collecting for 80 episodes")
    dataset.collect()
    print("Collected {} sample".format(len(dataset)))
    dataset.save()
    dataset.load()
    dataset.print_random_data(to_file = True)