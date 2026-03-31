# ARCTIC (Activity Reconstruction in Closed Loop)
Code for preprint: Zhou, S., Badman, R.P., Arlt, C., Rajan, K.\*, Harvey, C.D.\* Data-derived agents reveal dynamical reservoirs in mouse cortex for adaptive behavior. bioRxiv (2026) https://www.biorxiv.org/content/10.64898/2026.03.03.709365v1

Combine data-constrained RNNs with task-performing agents to study how neural dynamics support behavior. The agent interacts with an environment in closed loop, so that its neural activity drives actions, actions shape future sensory input, which in turn influences subsequent neural activity. An agent has a densely connected RNN in which each unit is trained to reproduce the activity of a corresponding real neuron, while the network as a whole is optimized to match animal behavior. 

## Structure of Repository
`src/` contains core implementations of the RNN-based model and the optimization algorithm.

`scripts/Ymaze_simulation/` contains code customized for the Y-maze experiment. For example, `Environment.py` defines the structure of the Y-maze, how position is updated according to the agent's velocity, and what observations are observable to the agent. Another key file is `Model_utils.py` which includes standardized interfaces bewteen the environment and the model, so that the model can be easily applied to various environments without the need to change the source code; only this file need to be rewritten. This directory also contains helper functions to load data and collect results specific to the Y-maze experiment. 

`notebooks/` contains demonstrations of the method.

## Demonstrations included
`Activity_reconstruction_in_closed_loop.ipynb` is the main demo. It shows how to train and evaluate a data-derived agent using ARCTIC, and contains key results of the chaotic dynamics.

`Compare_to_fitting_trial_average.ipynb` shows that fitting the model to trial-averaged data instead of individual trials results in classic point-attractor or continuous-attractor dynamics, underscoring that it is essential to consider the trial-to-trial variation to reveal the underlying dynamics.

`Adaptation_in_obstacle_environment_RL.ipynb` contains code to further train the data-derived agents through reward-based learning, in environments beyond those originally used to collect the data for training via ARCTIC. This allows us to examine the behavioral capacity of the data-derived neural dynamics. This notebook is only for the purpose of demonstrating the code, as the actually training loop takes too long (~2 hours per epoch) and thus is more optimal to be run as a bash job.  

## Data
For demonstration purpose, example data of one mouse session and processed results can be found at: . This session was used to generate the plots in the above notebooks. Additional data are available upon request to the corresponding authors.  

## Software requirements
This repository was developed with `Python=3.7.12`, `numpy=1.16.5` and `cupy=8.4.0`. 

The code works most efficiently on a GPU; the run time indicated in the notebooks are results on an rtx6000. 