from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecTransposeImage,
)
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn
import os

folder_path = './REPORT_PPO_CNN_RLZOO_HP/Iteration5'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print("Directory created successfully.")
else:
    print("Directory already exists.")
    

n_envs = 5

eval_env = VecTransposeImage(make_vec_env('CarRacing-v2', n_envs))

# eval every 500 timestep, 5 times, save best model
eval_callback = EvalCallback(eval_env, best_model_save_path=folder_path+"/best_model/",
                             log_path=folder_path+"/eval_log/", eval_freq=max(1000 // n_envs, 1),
                             deterministic=True, render=False)


env = make_vec_env('CarRacing-v2', n_envs)

model = PPO("CnnPolicy", env,
            seed = 42, 
            batch_size= 128,
            n_steps= 512,
            gamma= 0.99,
            gae_lambda= 0.95,
            n_epochs= 10,
            ent_coef= 0.0,
            sde_sample_freq= 4,
            max_grad_norm= 0.5,
            vf_coef= 0.5,
            learning_rate= 1e-4,
            use_sde= True,
            clip_range= 0.2,
            policy_kwargs= dict(log_std_init=-2, ortho_init=False,activation_fn=nn.GELU,net_arch=dict(pi=[256], vf=[256]),),
            verbose=1, tensorboard_log=folder_path+"/tensorboard/")

try:
  model.learn(total_timesteps=100000, log_interval=1, tb_log_name="ppo_rlzoohp", callback=eval_callback, progress_bar=True)

except KeyboardInterrupt:
  # this allows to save the model when interrupting training
  pass
finally:
  # Release resources
  try:
      assert model.env is not None
      model.env.close()
  except EOFError:
      pass

model.save(folder_path+"/ppo_cnnpolicy_rlzoohp.zip")