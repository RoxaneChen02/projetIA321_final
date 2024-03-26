
import os
import sys
from time import time

from torch import nn as nn

import torch
sys.path.append("..")  

import gymnasium as gym 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecEnv,
    VecTransposeImage,
)
from stable_baselines3.common.callbacks import EvalCallback

from typing import Optional, Dict, Any

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elasped time ?
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

class PPOTrainer :
    
    def __init__(self, device) -> None:
        self.n_jobs = 4
        self.n_trials = 50
        self.log_folder= "./"
        self.log_dir= "./"
        self.pruner= "default"
        self.env_name= 'CarRacing-v2'
        self.verbose=True
        self.n_timesteps=100000
        self.deterministic_eval= True
        self.optimization_log_path='./optimization_log'
        self.n_eval_episodes = 10
        self._hyperparams= {}
        self.device = device
        self.n_evaluations = 10
        pass 
    
    def sample_ppo_params(self,trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sampler for PPO hyperparams.

        :param trial:
        :return:
        """
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
        n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
        n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
        gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        vf_coef = trial.suggest_float("vf_coef", 0, 1)
        net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
        ortho_init = False
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

        if batch_size > n_steps:
            batch_size = n_steps
        net_arch = {
            "tiny": dict(pi=[64], vf=[64]),
            "small": dict(pi=[64, 64], vf=[64, 64]),
            "medium": dict(pi=[256, 256], vf=[256, 256]),
        }[net_arch_type]

        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]

        return {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "clip_range": clip_range,
            "n_epochs": n_epochs,
            "gae_lambda": gae_lambda,
            "max_grad_norm": max_grad_norm,
            "vf_coef": vf_coef,
            # "sde_sample_freq": sde_sample_freq,
            "policy_kwargs": dict(
                # log_std_init=log_std_init,
                net_arch=net_arch,
                activation_fn=activation_fn,
                ortho_init=ortho_init,
            ),
        }
    
    def objective(self, trial: optuna.Trial) -> float:
        
        kwargs = self._hyperparams.copy()
        
        sampled_hyperparams = self.sample_ppo_params(trial)
        
        kwargs.update(sampled_hyperparams)
        
        n_envs = 4

        env =  make_vec_env('CarRacing-v2', n_envs, monitor_dir=self.log_dir)
        #env = CarRacing(lidar=False, verbose=0)
        env = VecTransposeImage(env)
        
        model = PPO(
            env=env,
            policy = 'CnnPolicy',
            tensorboard_log=None,
            verbose=1,
            device=self.device,
            **kwargs,
        )
        
        eval_env =  make_vec_env('CarRacing-v2', n_envs, monitor_dir=self.log_dir)
        
        eval_env  = VecTransposeImage(eval_env)

        optuna_eval_freq = int(self.n_timesteps / self.n_evaluations)
        
        # Account for parallel envs
        optuna_eval_freq = max(optuna_eval_freq // n_envs, 1)

        path = None
        if self.optimization_log_path is not None:
            path = os.path.join(self.optimization_log_path, f"trial_{trial.number!s}")
            
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=path,
            log_path=path,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=optuna_eval_freq,
            deterministic=self.deterministic_eval,
        )
        
        model.learn(self.n_timesteps, callback=eval_callback) 

        assert model.env is not None
        model.env.close()
        eval_env.close()
            
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        return reward

    def hyperparameters_optimization(self, verbose=True) -> None:
            if verbose == 1:
                print("Optimizing hyperparameters")

            study = optuna.create_study()
        
            study.optimize(self.objective, n_jobs=self.n_jobs, n_trials=self.n_trials)

            print("Number of finished trials: ", len(study.trials))

            print("Best trial:")
            trial = study.best_trial

            print("Value: ", trial.value)

            print("Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

            report_name = (
                f"report_{self.env_name}_{self.n_trials}-trials-{self.n_timesteps}"
                f"_{int(time.time())}"
            )

            log_path = os.path.join(self.log_folder, "PPO", report_name)

            if self.verbose:
                print(f"Writing report to {log_path}")

            # Write report
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            study.trials_dataframe().to_csv(f"{log_path}.csv")

            # Plot optimization result
            try:
                fig1 = plot_optimization_history(study)
                fig2 = plot_param_importances(study)

                fig1.show()
                fig2.show()
            except (ValueError, ImportError, RuntimeError):
                pass
        
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = PPOTrainer(device)
    trainer.hyperparameters_optimization()