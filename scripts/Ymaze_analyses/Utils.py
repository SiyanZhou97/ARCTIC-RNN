import numpy as np
try:
    import cupy as cp
    a=cp.array([0])
except:
    import numpy as cp

###binning
def binning_maze(original, min_posF, max_posF, n_bin, idx_trials, maze_position, trialnum):
    # Bin individual trial according to the forward position. Each bin takes the averaged of all time steps in a trial that fall in the bin.

    original=cp.array(original)
    idx_trials=cp.array(idx_trials)
    maze_position=cp.array(maze_position)
    trialnum=cp.array(trialnum)
    
    bin_spacing = (max_posF - min_posF) / n_bin
    bin_half_width = bin_spacing / 2
    maze_position[cp.isnan(maze_position)] = -100  # Set to a number that we can ignore

    # n_trails by n_neuron by n_bin
    beh_binned = cp.zeros((idx_trials.shape[0], original.shape[0], n_bin))

    # for each trial
    for (i, idx_trial) in enumerate(idx_trials):
        # note that idx_trials should start from 1 if trialnum starts from 1
        maze_position_i = maze_position[trialnum == idx_trial]
        beh_i = original[:, trialnum == idx_trial].T
        beh_binned[i, :, :n_bin] = cp.array([cp.mean(
            beh_i[cp.logical_and(maze_position_i >= bin_center - bin_half_width,
                                 maze_position_i < bin_center + bin_half_width)], axis=0)
            for bin_center in cp.arange(min_posF + bin_half_width,
                                        max_posF, bin_spacing)]).T
        
    result = np.zeros(beh_binned.shape)
    try:
        beh_binned=cp.asnumpy(beh_binned)
    except:
        pass
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j, :] = np.interp(np.arange(0, result.shape[2]),
                                        np.arange(0, result.shape[2])[~np.isnan(beh_binned[i, j, :])],
                                        beh_binned[i, j, :][~np.isnan(beh_binned[i, j, :])])
    return result


def bin_1d(original,va,num=None,size=1):
    """
    bin the neural data w.r.t. one beh variable and return the binned average
    
    Args:
        original (ndarray of shape (n_neuron, n_step)): neural data
        va (list): A 4-element list [beh (ndarray of shape (n_step)), start, end, nbins]
                   specifying the behavior variable and the bins.
        size: width of each bin
    
    Returns:
        binned (ndarray of shape (n_neuron,va[3]+1-size))
    """
    if num is None:
        num=size*10
    n_neuron=original.shape[0]
    binned=np.full((n_neuron,va[3]+1-size),np.nan)
    a=va[0]
    behavior_bins_a=np.linspace(va[1],va[2],va[3]+1)
    for i in range(va[3]+1-size):
        t_range=np.logical_and(a>=behavior_bins_a[i],a<behavior_bins_a[i+size])
        if np.sum(t_range)<num: #if too few data points in the bin
            continue
        binned[:,i]=np.average(original[:,t_range],axis=1)
    return binned


def bin_2d(original,va,vb,count=False,num=None,size=1):
    """
    bin the neural data w.r.t. a 2d mesh of two beh variables and return the binned average
    
    Args:
        original (ndarray of shape (n_neuron, n_step)): neural data
        va (list): A 4-element list [beh (ndarray of shape (n_step)), start, end, nbins]
                   specifying the first behavior variable and the bins.
        vb (list): A 4-element list [beh (ndarray of shape (n_step)), start, end, nbins]
                   specifying the second behavior variable and the bins.
        count (optional): if you want to count the number of samples fall in each bin
        size (optional, default 1): width of each variable bin
        
    Returns:
        binned (ndarray of shape (n_neuron,nbins_a+1-size,nbins_b+1-size))
        count_bin (optional if count==True, ndarray of shape (nbins_a,nbins_b)): count of samples fall in each bin 
    """
    if num is None:
        num=size**2*3
    n_neuron=original.shape[0]
    binned=np.full((n_neuron,va[3]+1-size,vb[3]+1-size),np.nan)
    if count is True:
        count_bin=np.full((va[3],vb[3]),np.nan)
    else:
        count_bin=None
    a=va[0]
    b=vb[0]
    behavior_bins_a=np.linspace(va[1],va[2],va[3]+1)
    behavior_bins_b=np.linspace(vb[1],vb[2],vb[3]+1)
    for i in range(va[3]+1-size):
        for j in range(vb[3]+1-size):
            t_range=np.logical_and(np.logical_and(a>=behavior_bins_a[i],a<behavior_bins_a[i+size]),
                                           np.logical_and(b>=behavior_bins_b[j],b<behavior_bins_b[j+size]))
            if count is True:
                count_bin[i,j]=np.sum(t_range)
            if np.sum(t_range)<num:
                continue
            binned[:,i,j]=np.average(original[:,t_range],axis=1)
    return binned,count_bin


def align2peak_2d(binned,peak_a,peak_b,count_bin=None):
    """
    Aligned a 2d grid to its peak x and y coordinate
    
    Args:
        binned (ndarray of shape (n_neuron,nbins_a,nbins_b)): 2d binned activity of n neurons.
        peak_a (ndarray of shape (n_neuron)): peak coordinate on the first axis
        peak_b (ndarray of shape (n_neuron)): peak coordinate on the second axis
        count_bin (ndarray of shape (nbins_a,nbins_b)): count of samples fall in each bin
    
    Returns:
        aligned (ndarray of shape (n_neuron,41,41)): aligned neural activity
        count_align (optional, ndarray of shape (n_neuron,41,41)): count of samples that fall in each aligned bin
    """
    aligned=np.full((binned.shape[0],41,41),np.nan)
    if count_bin is None:
        count_align=None
    else:
        count_align=np.full((binned.shape[0],41,41),np.nan)
    bins_a=binned.shape[1]
    bins_b=binned.shape[2]
    for idx in range(binned.shape[0]):
        start_a=int(peak_a[idx]-20)
        end_a=int(peak_a[idx]+21)
        start_b=int(peak_b[idx]-20)
        end_b=int(peak_b[idx]+21)
        aligned[idx,max(0,-start_a):41-max(0,end_a-bins_a),max(0,-start_b):41-max(0,end_b-bins_b)]=\
            binned[idx,max(0,start_a):end_a,:][:,max(0,start_b):end_b]
        if count_bin is not None:
            if count_bin.ndim==2:
                count_align[idx,max(0,-start_a):41-max(0,end_a-bins_a),max(0,-start_b):41-max(0,end_b-bins_b)]=\
                    count_bin[max(0,start_a):end_a,:][:,max(0,start_b):end_b]
            elif count_bin.ndim==3:
                count_align[idx,max(0,-start_a):41-max(0,end_a-bins_a),max(0,-start_b):41-max(0,end_b-bins_b)]=\
                    count_bin[idx,max(0,start_a):end_a,:][:,max(0,start_b):end_b]
    return aligned,count_align

def model_correctness(beh_model,frame_trial_eval,cor_trial_idxes,choices):  
    #return the idxes of correctly performed trials
    beh_m_end=[beh_model[4,np.where(frame_trial_eval==i)[0][-1]] for i in cor_trial_idxes]
    posF_end=[beh_model[3,np.where(frame_trial_eval==i)[0][-1]] for i in cor_trial_idxes]
    chos_m=(np.array(beh_m_end)>0).astype('int')
    correct_m=((choices-chos_m)==0)
    trial_length=np.array([np.sum(frame_trial_eval==i) for i in cor_trial_idxes])
    correct_m=np.logical_and(correct_m,trial_length<200)
    correct_m[np.array(posF_end)<200]=0 
    return correct_m

def model_correctness_RL(beh_model,frame_trial_eval,cor_trial_idxes,choices):  
    #return the idxes of correctly performed trials
    beh_m_end=[beh_model[4,np.where(frame_trial_eval==i)[0][-1]] for i in cor_trial_idxes]
    posF_end=np.array([beh_model[3,np.where(frame_trial_eval==i)[0][-1]] for i in cor_trial_idxes])
    chos_m=(np.array(beh_m_end)>0).astype('int')
    correct_m=((choices-chos_m)==0)
    trial_length=np.array([np.sum(frame_trial_eval==i) for i in cor_trial_idxes])
    correct_m=np.logical_and(correct_m,trial_length<200)
    # The following number is changed to 230 becuase the RL simulation is desigend to collect the last timestep (dt=0.0093s) before trial end,
    # while in data-fitting simulation (the above function) that last timestep is usually not collected as there is not a corresponding data frame (dtData=0.186s)
    correct_m[posF_end<=230]=0  
    return correct_m

def stack(a,axis):
    if axis==0:
        raise Exception("cannot stack on the first axis")
    if a.ndim==1:
        raise Exception("dim of array is 1")
    return np.squeeze(np.vstack([np.take(a,[i],axis=axis) for i in range(a.shape[axis])]),axis=axis)

def simple_beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    
    Cite from https://stackoverflow.com/a/71498646
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x
