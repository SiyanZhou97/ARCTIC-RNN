import copy
try:
    import cupy as np
    a=np.array([0])
except:
    import numpy as np

# Standardized interface bewteen env and model
# For customized env, need to re-implement the following functions:

class Ymaze_inputs():
    """For any env, the model is only allowed to access the env through this class, 
    which provides a standardized interface to observe and update the env..
    """
    def __init__(self,maze):
        self.maze=copy.deepcopy(maze)

    def sample(self,n=None):
        input, termination = self.maze.observations_from_env()
        env_states = self.maze.beh_readout()
        return input,termination,env_states

    def update(self,beh):
        self.maze.beh_update(*beh)

def Ymaze_align(behavior_target,observation):
    """ For online training purpose, at each time step, the target data is determined by this alignment function (goes into Model.train())
    For example, in the Ymaze, the alignment is according to the forward progress in the Ymaze.
    """
    target_trajectory=behavior_target[3, :]
    query=observation[3]
    t_target = np.argmin(np.abs(target_trajectory - query)) + 1
    if t_target >= target_trajectory.shape[0]:  # if run out of training time steps
        t_target = target_trajectory.shape[0]-1
    return t_target