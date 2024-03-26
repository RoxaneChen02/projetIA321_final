import os
import gymnasium as gym
import numpy as np
from torchvision import transforms
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import argparse
import torch
from vae.vae import VAE
from VaeWrapper import VaeWrapper
from datetime import datetime
import itertools
# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the PPO algorithm")
parser.add_argument("--folder", type=str, default="runs", help="Folder name")
args = parser.parse_args()

 
hyperparameters_grid = {
    "learning_rate": [1e-5],  
    "batch_size": [64],  
    "gradient_steps": [1],
    "tau": [0.001,0.01,0.1],
    "buffer_size": [10000,100000,1000000]
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load VAE
print('Load Vae..')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_size=32)
vae.set_device(device)
vae.to(device)
vae.load("./model/vae_model")
    
# Wrap environment
print('Wrap environment in VaeWrapper')

env = gym.make('CarRacing-v2', render_mode='rgb_array', continuous = True)
env =  VaeWrapper(env, vae, device)
    
print("Training model...")

# define save folder
folder_path = args.folder


n_actions = env.action_space.shape[-1]

action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            theta=float(0.6) * np.ones(n_actions),
            sigma=float(0.2) * np.ones(n_actions)
            )

for hyperparameters in itertools.product(*hyperparameters_grid.values()):
    # Unpack hyperparameters
    learning_rate, batch_size, gradient_steps, tau, buffer_size = hyperparameters
    folder_path = os.path.join(args.folder, f"lr_{learning_rate}_bs_{batch_size}_gs_{gradient_steps}_bf_{buffer_size}_tau_{tau}")
    print(folder_path)
    
    os.makedirs(folder_path, exist_ok=True)
    
    eval_env = Monitor( env)
    # eval every 500 timestep, 5 times, save best model

    eval_callback = EvalCallback(eval_env, best_model_save_path=folder_path+"/best_model/",
                             log_path=folder_path+"/eval_log/", eval_freq=1000,
                             deterministic=True, render=False)
    
    model = DDPG("MlpPolicy",
                    env,
                    verbose=1,
                    learning_rate = learning_rate,
                    batch_size=batch_size,
                    gamma=0.9,
                    tau=tau,
                    action_noise=action_noise,
                    buffer_size =buffer_size,
                    gradient_steps =gradient_steps,
                
                    tensorboard_log=folder_path+"/tensorboard/")

    model.learn(total_timesteps=50000, log_interval=1, tb_log_name=f"ddpg", callback=eval_callback, progress_bar=True)