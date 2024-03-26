from stable_baselines3 import DQN, PPO
import torch.nn as nn
# from gymnasium.envs.box2d.car_racing import CarRacing
from carracing_env import CarRacing
from typing import Callable
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback


lidar_angles = [270, 280, 290, 300, 310, 320, 330, 340, 350, 355, 0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]

env = CarRacing(lidar=True, verbose=1, render_mode="human", continuous=True, lidar_angles=lidar_angles)
model = PPO.load("model/best_lidar_model.zip", env=env, print_system_info=True)
# model = DQN("MlpPolicy", env)
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    # time.sleep(0.33)
    # print(_states)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()