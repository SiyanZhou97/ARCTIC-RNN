import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import copy

if torch.cuda.is_available(): 
    dev = "cuda:0" 
    import cupy as cp 
else: 
    dev = "cpu" 
    import numpy as cp  
device = torch.device(dev) 

code_dir = '/n/data2/hms/neurobio/harvey/siyan/arctic/'
os.sys.path.insert(0,code_dir)
from scripts.Ymaze_simulation.LoadData import generator

def evaluation_loop(rnn, actor, critic, env, param,
                    cues, activity, behavior,frame_trial,eval_trial_idxes):
    td_t=int(0.186/0.0093)
    gamma=0.9
    R_eval = cp.full((activity.shape[0], 0), np.nan)
    beh_output_eval = cp.full((behavior.shape[0], 0), np.nan)
    frame_trial_eval = []
    value_eval=[]
    delta=[]
    rewards=[]
    for trial_index in eval_trial_idxes:
        cue=int(cues[trial_index])
        choice=cp.array([cue,1-cue])
        choice=torch.FloatTensor(choice).to(device)
        beh_init=cp.array(behavior[:, frame_trial == trial_index][:,0])
        neu_init=cp.array(activity[:, frame_trial == trial_index][:,0])
        R_eval = cp.hstack((R_eval, neu_init[:,cp.newaxis]))
        beh_output_eval = cp.hstack((beh_output_eval, beh_init[:,cp.newaxis]))
        frame_trial_eval.append(trial_index)
        value_eval.append(np.nan)
        delta.append([np.nan]*2)
        rewards.append(0)
        rnn.initialize(neu_init)
        env.maze_init(cue,*beh_init)
        i=0
        reward=0
        intermediate_reward=0
        intermediate_reward2=0
        while(1):
            input, termination, gameover = env.observations_from_env()
            obs_pos= env.observations_position()
            obs_pos=torch.FloatTensor(obs_pos).to(device)
            r=rnn.step(input)
            vel=actor(r[cp.newaxis,:])[0]
            env.beh_update(*cp.array(vel.detach()))
            beh = env.beh_readout()
            i+=1
            reward_old=copy.deepcopy(reward)
            if i % td_t==0 or termination or gameover:
                #reward of getting over the first obstacle
                if env.posF>=181 and intermediate_reward==0:
                    reward=param[0]
                    intermediate_reward=1
                elif intermediate_reward==1:
                    reward=0
                #reward of entering the correct arm
                if env.posF>=220 and intermediate_reward2==0 and\
                        ((cue==0 and input[-1]>0) or (cue==1 and input[-1]<0)):
                    reward=param[1]
                    intermediate_reward2=1
                elif env.posF>=220 and intermediate_reward2==0 and\
                        ((cue==0 and input[-1]<0) or (cue==1 and input[-1]>0)):
                    reward=param[3]
                    intermediate_reward2=1
                elif intermediate_reward2==1:
                    reward=0
                #reward of trial outcome
                if termination: #hit end of maze
                    if (cue==0 and input[-1]>0) or (cue==1 and input[-1]<0):
                        reward=param[2]
                    else:
                        reward=param[3]
                if gameover: #hit obstacle
                    reward=param[3]
                R_eval = cp.hstack((R_eval, r[:,np.newaxis]))
                beh_output_eval = cp.hstack((beh_output_eval, beh[:,np.newaxis]))
                frame_trial_eval.append(trial_index)
                rewards.append(reward)
                if critic is not None:
                    vel=vel.detach()
                    value=critic(obs_pos.unsqueeze(0),vel[1:2].unsqueeze(0),choice.unsqueeze(0))
                    value_eval.append(value)
                    if i/td_t>=2:
                        delta.append((reward+gamma*value-value_old).detach().cpu().numpy()[0])
                    value_old=copy.deepcopy(value.detach())
            if termination or i/td_t>200 or gameover:
                break
    return R_eval,beh_output_eval,frame_trial_eval,value_eval,delta,rewards


class ReplayBuffer:
    def __init__(self):
        self.buffer = []
    
    def add(self, r_old,obs_pos,vel,obs_pos_old,vel_old,choice,reward):
        self.buffer.append((r_old,obs_pos,vel,obs_pos_old,vel_old,
                            choice,reward))
    
    def sample(self):
        batch = self.buffer
        r_old,obs_pos,vel,obs_pos_old,vel_old,choice,reward = zip(*batch)
        r_old=cp.array(r_old)
        obs_pos=torch.stack(obs_pos)
        vel=torch.stack(vel)
        obs_pos_old=torch.stack(obs_pos_old)
        vel_old=torch.stack(vel_old)
        choice=torch.stack(choice)
        reward=torch.FloatTensor(reward).unsqueeze(1)
        return (r_old,obs_pos,vel,obs_pos_old,vel_old,choice,reward)


def training_loop_batch(rnn, actor, critic, actor_optimizer, critic_optimizer, env, param, 
                        cues, activity, behavior,frame_trial,train_trial_idxes,n_epoch=10,train_actor=False):
    gamma=0.9
    td_t=int(0.186/0.0093)
    train_steps=0
    train_generator = generator(train_trial_idxes)
    while True:
        replay_buffer = ReplayBuffer()
        trial_index = next(train_generator)
        cue=int(cues[trial_index])
        choice=cp.array([cue,1-cue])
        choice=torch.FloatTensor(choice).to(device)
        beh_init=cp.array(behavior[:, frame_trial == trial_index][:,0])
        neu_init=cp.array(activity[:, frame_trial == trial_index][:,0])
        rnn.initialize(neu_init)
        env.maze_init(cue,*beh_init)
        i=0
        reward=0
        intermediate_reward=0
        intermediate_reward2=0
        while(1):
            #evaluate gradient at t-1 and update weights at t
            input, termination,gameover = env.observations_from_env()
            obs_pos = env.observations_position()
            obs_pos=torch.FloatTensor(obs_pos).to(device)
            #forward step
            r=rnn.step(input)
            vel=actor(r[cp.newaxis,:])[0]
            #update env
            env.beh_update(*cp.array(vel.detach()))
            i+=1
            if i % td_t==0 or termination or gameover:
                #reward of getting over the first obstacle
                if env.posF>=181 and intermediate_reward==0:
                    reward=param[0]
                    intermediate_reward=1
                elif intermediate_reward==1:
                    reward=0
                #reward of entering the correct arm
                if env.posF>=220 and intermediate_reward2==0 and\
                        ((cue==0 and input[-1]>0) or (cue==1 and input[-1]<0)):
                    reward=param[1]
                    intermediate_reward2=1
                elif env.posF>=220 and intermediate_reward2==0 and\
                        ((cue==0 and input[-1]<0) or (cue==1 and input[-1]>0)):
                    reward=param[3]
                    intermediate_reward2=1
                elif intermediate_reward2==1:
                    reward=0
                #reward of trial outcome
                if termination: #hit end of maze
                    if (cue==0 and input[-1]>0) or (cue==1 and input[-1]<0):
                        reward=param[2]
                    else:
                        reward=param[3]
                if gameover: #hit obstacle
                    reward=param[3]
                if i/td_t>=2:
                    replay_buffer.add(r_old,obs_pos.detach(),vel.detach(),
                                      obs_pos_old.detach(),vel_old.detach(),choice,reward)
                r_old=copy.deepcopy(r)
                vel_old=copy.deepcopy(vel.detach())
                obs_pos_old=copy.deepcopy(obs_pos)
            if termination or i/td_t>200 or gameover:
                break
        train_steps+=1
        if termination or gameover:
            #train all datapoint from a trial as a batch, which is more efficient and stable than training at each time step         
            r_old,obs_pos,vel,obs_pos_old,vel_old,choice,reward=replay_buffer.sample()
            #update critic
            value=critic(obs_pos,vel[:,1:2],choice)
            value=value.detach()
            value_old=critic(obs_pos_old,vel_old[:,1:2],choice)
            delta=reward.to(device)+gamma*value-value_old
            critic_loss=(delta**2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            #update actor
            if train_actor==True:
                actor_loss = -critic(obs_pos_old, actor(r_old)[:,1:2],choice).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
        if train_steps % train_trial_idxes.shape[0] == 0:
            idx_epoch = int(train_steps / train_trial_idxes.shape[0])
            if idx_epoch == n_epoch:
                break