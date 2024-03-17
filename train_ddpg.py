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

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the PPO algorithm")
parser.add_argument("--folder", type=str, default="logs", help="Folder name")
args = parser.parse_args()


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
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print("Directory created successfully.")
else:
    print("Directory already exists.")
    
eval_env = Monitor( env)

# eval every 500 timestep, 5 times, save best model

eval_callback = EvalCallback(eval_env, best_model_save_path=folder_path+"/best_model/",
                             log_path=folder_path+"/eval_log/", eval_freq=500,
                             deterministic=True, render=False)

n_actions = env.action_space.shape[-1]

action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            theta=float(0.6) * np.ones(n_actions),
            sigma=float(0.2) * np.ones(n_actions)
            )

model = DDPG("MlpPolicy",
                    env,
                    verbose=1,
                    learning_rate = args.learning_rate,
                    batch_size=64,
                    gamma=0.9,
                    action_noise=action_noise,
                    buffer_size =10000,
                    gradient_steps =3000,
                
                    tensorboard_log=folder_path+"/tensorboard/")

model.learn(total_timesteps=100000, log_interval=1, tb_log_name=f"ppo_lr_{args.learning_rate}", callback=eval_callback, progress_bar=True)
