"""
This code re-implement the output layer of the RNN agent with the torch library, which allows us to use autograd for further RL training. 
The RNN class defines the recurrent neural network that generates the neural activity, 
while the Actor and Critic classes define the policy and value networks for RL. 
The Actor network takes the RNN activity as input and outputs the action (velocity), 
while the Critic network takes the state (cue and current position) and action as input and outputs a value estimate.
The weights of the Actor network are initialized to the output layer of the data-derived RNN agent).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

if torch.cuda.is_available(): 
    dev = "cuda:0" 
    import cupy as cp 
else: 
    dev = "cpu" 
    import numpy as cp  
device = torch.device(dev) 

code_dir = '/n/data2/hms/neurobio/harvey/siyan/arctic/'
os.sys.path.insert(0,code_dir)
from src.Model import RnnModel as rnn 
from src.Tools import phi,reverse_phi
try:
    from RL_Env_utils_torch import create_cosine_bumps_torch, position_expansion_torch
except:
    from .RL_Env_utils_torch import create_cosine_bumps_torch, position_expansion_torch


#define the actor and critic networks
class RNN():
    def __init__(self,dtData=0.186, dt=0.001, tau=0.1, N=500,
                 phi='modifiedtanh',J_neu=None,cur_noise=0):
        #model settings
        self.tau = tau
        self.dtData = dtData
        self.dt = dt
        self.phi = phi
        self.cur_noise=cur_noise
        #model structure
        self.N = N
        self.J_neu=J_neu
        
    def initialize(self,neural_init):
        self.r=neural_init
        self.x=reverse_phi(self.r, self.phi)

    def step(self, input):
        # forward pass
        u_vec = cp.hstack((phi(self.x, self.phi), input))
        u = cp.dot(self.J_neu, u_vec)  # rec+input+feedback
        self.x = self.x + self.dt / self.tau * (u - self.x + cp.random.randn(self.N) * self.cur_noise)
        self.r = phi(self.x, self.phi)
        return self.r


class Actor(nn.Module):
    def __init__(self,N=500,J_beh=None):
        super(Actor, self).__init__()
        #forward velocity (weight fixed)
        self.output1 = nn.Linear(N,1)
        with torch.no_grad():
            self.output1.weight.copy_(torch.from_numpy(J_beh[0])) #initialize output weight to J_beh
            self.output1.bias.copy_(torch.zeros_like(self.output1.bias)) #fixed bias to 0
        self.output1.weight.requires_grad = False
        self.output1.bias.requires_grad = False
        
        #lateral velocity (weight trainable)
        self.output2 = nn.Linear(N,1)
        with torch.no_grad():
            self.output2.weight.copy_(torch.from_numpy(J_beh[1])) #initialize output weight to J_beh
            self.output2.bias.copy_(torch.zeros_like(self.output2.bias)) #fixed bias to 0
        self.output2.bias.requires_grad = False

        #yaw velocity (weight fixed)
        self.output3 = nn.Linear(N,1)
        with torch.no_grad():
            self.output3.weight.copy_(torch.from_numpy(J_beh[2])) #initialize output weight to J_beh
            self.output3.bias.copy_(torch.zeros_like(self.output3.bias)) #fixed bias to 0
        self.output3.weight.requires_grad = False
        self.output3.bias.requires_grad = False
        
    def forward(self,r):
        r=torch.FloatTensor(r).to(device)
        beh1=self.output1(r)
        beh2=self.output2(r)
        beh3=self.output3(r)
        return torch.cat((beh1,beh2,beh3),dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim=10+10+2):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, state,action,choice):
        action=position_expansion_torch(action, 10, -1, 1,device)
        fullstate=torch.cat((state,action,choice),dim=1).to(device)
        return self.fc(fullstate)
