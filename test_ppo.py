
import torch
from stable_baselines3 import PPO
import gymnasium as gym 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""

env = gym.make('CarRacing-v2', render_mode='human', continuous = True)

model = PPO.load("runs/exemple_ppo_cnn/best_model/best_model.zip")

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