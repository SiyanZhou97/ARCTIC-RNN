import os
import pickle
import random
import numpy as np
import math
import copy
import sys
import h5py as h5


def load_charlotte_delay(data_dir,filename):
    # 1.import original data
    root_dir = data_dir + filename
    with h5.File(root_dir + 'data.h5', 'r') as f:
        neuron_activity = f['neuron_activity']
        activity = np.concatenate(
            (neuron_activity['RSC'], neuron_activity['PPC'], neuron_activity['M2'], neuron_activity['V1']))
        beh = f['behavior']
        delay = f['delay'][0]
        trial_type = beh[0, :]
        x_vel = beh[1, :]
        y_vel = beh[2, :]
        rule = beh[3, :]
        posL = beh[4, :]
        posF = beh[5, :]
        licks = beh[6, :]
        ITI = beh[7, :]
        reward = beh[8, :]
        dt = beh[9, :]
        pitch = beh[10, :]
        roll = beh[11, :]
        yaw = beh[12, :]
        frame_trial = beh[13, :]
    
    # 1/3 select valid frames
    num_trial = int(np.max(frame_trial[~np.isnan(frame_trial)]))
    frame_trial = frame_trial - 1  # make it count from zero
    # disgard the few starting frames in each trial when the mouse is not actively running forward
    for i in range(num_trial):
        fv = y_vel[frame_trial == i]/ 100
        s = np.where(fv > 0.1)[0][0]
        s = max(0, s - 5)
        frame_trial[np.where(frame_trial == i)[0][:s]] = -1

    # 2/3 binary trial labels: cue, choice, rew/correctness
    start_trials=np.zeros(num_trial)
    for i in range(num_trial):
        start_trials[i] = np.where(frame_trial == i)[0][0]
    start_trials = start_trials.astype('int')
    trial_types = trial_type[start_trials]
    cues = np.array([1 if trial_types[i] in [1, 3, 5, 7] else 0 for i in range(num_trial)])
    visguides = np.array([1 if trial_types[i] in [1, 2, 3, 4] else 0 for i in range(num_trial)])
    cor_chos = np.array([1 if trial_types[i] in [1, 4, 5, 8] else 0 for i in range(num_trial)])
    rews_frame_trial = frame_trial[reward == 1].astype('int')
    rews = np.zeros(num_trial)
    rews[rews_frame_trial] = 1
    rews = rews.astype('int')
    chos = np.abs(cor_chos - (1 - rews))
    binary_labels=np.vstack([cues,chos,rews,visguides])

    # 3/3 behaviors: velF,velL,velY,posF,posL
    # want only during-maze frames
    frame_trial[ITI != 0] = -1
    #some trials have the first ITI frame incorrectly labeled as maze period
    for i in range(num_trial):
        if posF[frame_trial == i][-1]<200:
            #frame_trial[frame_trial == i][-1]=-1
            frame_trial[np.where(frame_trial==i)[0][-1]]=-1
    with np.errstate(invalid='ignore'):
        posF[posF > 233] = 233 # clip forward position
        posF[np.logical_and(posF < 5, posF > 0)] = 5
    for i in range(num_trial):
        posF[start_trials[i]] = min(9.9,np.nanmin(posF[frame_trial == i]))
    posF[ITI != 0] = -100
    posF[np.isnan(posF)] = -100
    behavior = np.zeros((5, posF.shape[0]))
    behavior[0, :] = y_vel / 100 # forward velocity
    behavior[1, :] = x_vel / 100 # lateral velocity
    behavior[2, :] = yaw - np.nanmean(yaw)
    behavior[3, :] = posF
    behavior[4, :] = posL

    return delay, binary_labels, frame_trial, activity, behavior


def generator(data, shuffle=True):
    """
    Args:
        data (list):
            Data for generator.
            Specific in this case, it is the indexes of selected trials.
        shuffle (bool, optional): Whether trial sequence is shuffled in each epoch.

    """
    data_lng = len(data)
    index_list = [*range(data_lng)]
    # If shuffle is set to true, we traverse the list in a random way
    if shuffle:
        random.shuffle(index_list)  # Inplace shuffle of the list
    index = 0  # Start with the first element
    while True:
        # Wrap the index each time that we reach the end of the list
        if index >= data_lng:
            index = 0
            # Shuffle the index_list if shuffle is true
            if shuffle:
                random.shuffle(index_list)  # re-shuffle the order
        data_i = data[index_list[index]]
        index += 1
        yield data_i
