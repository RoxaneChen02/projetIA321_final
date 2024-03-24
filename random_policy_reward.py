import gymnasium as gym
import numpy as np
env = gym.make("CarRacing-v2", render_mode="rgb_array")

rewards = []

for j in range(10):
    observation = env.reset()
    ep_reward = 0
    done = False
    step = 0
    while not done and step < 1000:
        action = env.action_space.sample()
        observation, reward, done,_,_ = env.step(action)
        env.render()
        ep_reward +=reward
        step += 1
    print(ep_reward)
    rewards.append(ep_reward)

env.close()

print("Mean rewards for random policy (over 10 simulations):", np.mean(rewards))
                