from stable_baselines3 import DQN, PPO
import torch.nn as nn
from carracing_env import CarRacing
from typing import Callable
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse

parser = argparse.ArgumentParser(description="Train or test the car racing environment with lidar")
parser.add_argument("--train", action="store_true", help="Train the model")
parser.add_argument("--test", action="store_true", help="Test the model")

# Add the hyperparameters arguments
parser.add_argument("--batch_size", type=int, default=128, help="The batch size for the PPO algorithm")
parser.add_argument("--n_steps", type=int, default=512, help="The number of steps for the PPO algorithm")
parser.add_argument("--sde_sample_freq", type=int, default=4, help="The sample frequency for the SDE")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="The learning rate for the PPO algorithm")
parser.add_argument("--linear_schedule", action="store_true", help="Use a linear learning rate schedule")
parser.add_argument("--use_sde", action="store_true", help="Use SDE for the PPO algorithm")
parser.add_argument("--clip_range", type=float, default=0.2, help="The clip range for the PPO algorithm")
parser.add_argument("--total_timesteps", type=int, default=100_000, help="The total number of timesteps for the PPO algorithm")
parser.add_argument("--run_id", type=int, default=0, help="The run id for the experiment")
args = parser.parse_args()

train = False
lidar_angles = [270, 280, 290, 300, 310, 320, 330, 340, 350, 355, 0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]

checkpoint_callback = CheckpointCallback(
  save_freq=10_000,
  save_path="logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


env = CarRacing(lidar=True, verbose=1, continuous=True, lidar_angles=lidar_angles)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_car_racing_lidar_tensorboard/", batch_size=args.batch_size, n_steps=args.n_steps, gamma=0.99, gae_lambda=0.95, n_epochs=10, ent_coef=0.0, sde_sample_freq=args.sde_sample_freq, max_grad_norm=0.5, vf_coef=0.5, learning_rate=args.learning_rate, use_sde=args.use_sde, clip_range=args.clip_range, policy_kwargs=dict(log_std_init=-2,
                    ortho_init=False,
                    activation_fn=nn.GELU,
                    net_arch=dict(pi=[256], vf=[256]),
                    ))
# Write the hyperparams to the tensorboard and in the model file name
model.learn(total_timesteps=args.total_timesteps, log_interval=1, tb_log_name="ppo_car_racing_lidar_{}".format(args))
model.save("models/ppo_car_racing_lidar_{}.zip".format(args))