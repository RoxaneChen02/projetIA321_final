# Car Racing Environment 

For this project we use exclusively the gymnasium CarRacing-v2.

To play the game manually:

``
python3 gymnasium/envs/box2d/car_racing.py
``

# VAE

To generate a dataset : 

``
python3 vae/CarRacingDataset.py
``

It will collect data from 80 episodes with 150 timesteps during each episodes. It can take several minutes it's normal.

To launch the training:

``
python3 train_vae.py
``


## Project structure: 
    
    .
    ├── model    # Folder for autoencoder and racing agent
    |
    ├── vae   # Vae module
    |   ├── CarRacingDataset.py    # Contains code to collect training data and the CarRacingDataset class
    |   ├── vae.py    # Contains the Vae definition
    |   ├── visualize_data_collection.py # Contains code to see what happens during the datacollection
    |
    ├── runs    # Example tensorboard training log 
    |
    ├── train_vae.py # Contains code to train vae 
    └── README.md     # This file
