# Car Racing Environment 

For this project we use exclusively the gymnasium CarRacing-v2.


# VAE

To generate a dataset : 

```
python3 vae/CarRacingDataset.py
```

It will collect data from 80 episodes with 150 timesteps during each episodes. It can take several minutes it's normal.

To launch the training:

```
python3 train_vae.py
```

# Test agent

To test our best agent trained with DDPG and a VAE : 

``` 
python3 test_ddpg.py
```

Same to test PPO + CNN:

```
python3 test_ppo.py
```



## Project structure: 
    
    .
    ├── model    
    |   ├── vae_model # Trained Vae model
    |
    ├── vae   # Vae module
    |   ├── CarRacingDataset.py    # Contains code to collect training data and the CarRacingDataset class
    |   ├── vae.py    # Contains the Vae definition
    |   ├── visualize_data_collection.py # Contains code to see what happens during the datacollection
    |
    ├── runs    # Example tensorboard training log
    |   ├──exemple_ddpg_vae 
    |   ├──exemple_ppo_cnn
    |
    ├── videos 
    |   ├── best_ddpg.mp4 #exemple of test with our best ddpg+vae trained agent
    |
    ├── grid_train_ddpg.py # Train ddpg with different hyperparameters
    ├── plot_eval_ddpg.ipynb # notebook to plot result from multiple training sessions
    ├── random_policy_reward.py # Estimate the average reward of the random policy
    ├── RewardWrapper.py # Wrapper to change the reward to just the speed of the car (not used)
    ├── test_ddpg.py # Test DDPG model
    ├── train_ppo_cnn.py # Code to train a simple PPO model with CNN policy
    ├── train_ddpg_vae.py # Contains code to train DDPG model with VAE wrapper  
    ├── train_vae.py # Contains code to train vae
    ├── VaeWrapper.py # Gymnasium observation wrapper    
    └── README.md     # This file
