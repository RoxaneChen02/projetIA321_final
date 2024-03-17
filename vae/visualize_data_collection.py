
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
import time

def generate_action(prev_action):
        
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

env = CarRacing(render_mode="human")
obs_data = []
        
#sample first random action 
action = env.action_space.sample()
        
for _ in range(10):
    observation = env.reset()
            
    # Make the Car start at random positions in the race-track
    position = np.random.randint(len(env.track))
    env.car = Car(env.world, *env.track[position][1:4])

    for i in range(150):
                action = generate_action(action)
                observation, _, _, _,_ = env.step(action)
                env.render()