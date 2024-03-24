
import torch
from stable_baselines3 import DDPG
import gymnasium as gym 
from vae.vae import VAE
from VaeWrapper import VaeWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""


# Load VAE
print('Load Vae..')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_size=32)
vae.set_device(device)
vae.to(device)
vae.load("./model/vae_model")
    
# Wrap environment
print('Wrap environment in VaeWrapper')

env = gym.make('CarRacing-v2', render_mode='human', continuous = True)
env =  VaeWrapper(env, vae, device)

model = DDPG.load("runs/exemple_ddpg_vae/best_model/best_model.zip")

terminated = False
truncated = False
obs, info  = env.reset()
total_reward = 0
step = 0
while not terminated and step < 1000:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward+=reward
    step+=1
    env.render()
    
print("The total reaward is:", total_reward)