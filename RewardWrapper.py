"""Wrapper for transforming the reward."""
from typing import Callable
import gymnasium as gym
import numpy as np
import pygame

class TransformReward(gym.RewardWrapper):

    def __init__(self, env: gym.Env):
        """Initialize the :class:`TransformReward` wrapper.
            New reward = the speed of the car and if the car is not on track we give a penalty of -1000.

        Args:
            env: The environment to apply the wrapper
        """
        gym.RewardWrapper.__init__(self, env)
    
    def check_ontrack(self,image):
        # Define the colors representing grass and track
        grass_colors = [229,203]  

        # Define the region around the car to inspect (adjust as needed)
        region_size = 50
        x, y = 300,100
        region = image[y - region_size:y + region_size, x - region_size:x + region_size]

        # Check colors in the region
        grass_pixels = np.array([pixel[1] in grass_colors for row in region for pixel in row])

        # Determine if the car is on grass or track based on the number of grass pixels
        grass_count = sum(grass_pixels)

        if grass_count > 3000:
            return False
        return True

    def reward(self, reward):
        
        speed_norm = np.sqrt(
            np.square(self.env.unwrapped.car.hull.linearVelocity[0])
            + np.square(self.env.unwrapped.car.hull.linearVelocity[1])
        )
        self.env.unwrapped.surf = pygame.transform.flip(self.env.unwrapped.surf, False, True)
        scaled_screen = pygame.transform.smoothscale(self.env.unwrapped.surf , (600,400))
        image =  np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )
        
        if self.check_ontrack(image)==False:
            speed_norm = -1000
        return speed_norm
    


# Test Wrapper
if __name__ == "__main__":
    
    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = gym.make("CarRacing-v2", render_mode='human')
    env = TransformReward(env)
    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
