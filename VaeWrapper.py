from gymnasium import ObservationWrapper
from gymnasium.spaces.box import Box
import sys
from torchvision import transforms
import gymnasium as gym
from vae.vae import VAE
import torch
import numpy as np

class VaeWrapper(ObservationWrapper):
    """
    Observation Wrapper for Gymnasium environment to change observation VAE output
    """
    def __init__(self, env, vae, device):
        super().__init__(env)
        self.observation_space = Box(
            low=sys.float_info.min, high=sys.float_info.max, shape=(vae.latent_size+2,), dtype=float
        )
        self.vae = vae
        self.device= device

    def observation(self, obs):
        """
        # Input
        - obs : default observation from car racing environment (96, 96, 3) array
        # Output
        - obs : latent space output from vae (latent_size+2,) array
        """
        transform = transforms.Compose([
                                    transforms.ToPILImage() , 
                                    transforms.Resize((64, 64)),
                                    transforms.ToTensor()
                                ])
        z = self.vae.obs_to_z(transform(obs).unsqueeze(0).to(self.device)).cpu().detach().numpy().squeeze(0)
        
        speed_norm = np.sqrt(
            np.square(self.env.unwrapped.car.hull.linearVelocity[0])
            + np.square(self.env.unwrapped.car.hull.linearVelocity[1])
        )
        z = np.append(z, speed_norm)
        z = np.append(z, self.env.unwrapped.car.hull.angularVelocity)
        
        return z

# Test Wrapper
if __name__ == "__main__":
    # Load VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(latent_size=32)
    vae.set_device(device)
    vae.to(device)
    vae.load("./model/vae_model")
    
    # Wrap environment
    env = gym.make('CarRacing-v2', render_mode='rgb_array')
    env =  VaeWrapper(env, vae, device)
    env.reset()
    # Test Environement
    for i in range(1):
        action =  env.action_space.sample()
        observation, _, _, _,_ = env.step(action)
        print(observation.shape)
        env.render()