import numpy as np
try:
    import cupy as cp
    a=cp.array([0])
except:
    import numpy as cp

try:
    from LoadData import generator
    from Model_utils import Ymaze_inputs, Ymaze_align
except:
    from .LoadData import generator
    from .Model_utils import Ymaze_inputs, Ymaze_align
from tqdm import tqdm


def evaluation_closed_loop(net,env,
                    cues, activity, behavior,frame_trial,
                    eval_trial_idxes,
                    cur_noise):
    R_eval = cp.full((activity.shape[0], 0), cp.nan)
    beh_eval = cp.full((behavior.shape[0], 0), cp.nan)
    frame_trial_eval = []
    for trial_index in eval_trial_idxes:
        net.set_target(neuron_target=activity[:, frame_trial == trial_index],
                       neuron_target_type='r',
                       behavior_target=behavior[:3, frame_trial == trial_index],
                       true_env_states=behavior[:, frame_trial == trial_index])
        env.maze_init(cues[trial_index],*behavior[:, frame_trial == trial_index][:,0])
        input_generator=Ymaze_inputs(env)
        R,_,_,beh,_ = net.run(cur_noise=cur_noise,closed_loop=True,input_generator=input_generator)
        R_eval = cp.hstack((R_eval, R))
        beh_eval = cp.hstack((beh_eval, beh))
        frame_trial_eval += [trial_index] * (R.shape[1])
    return R_eval, beh_eval, frame_trial_eval


def perturbation_closed_loop(net,env,
                    cues, R_model, beh_model,frame_trial_eval,
                    legit_trial_idxes,
                    pos, norm,cd):
    R_perturb = cp.full((R_model.shape[0], 0), cp.nan)
    beh_output_perturb = cp.full((beh_model.shape[0], 0), cp.nan)
    frame_trial_perturb = []
    for trial_index in legit_trial_idxes:
        pos_trial=beh_model[3,frame_trial_eval==trial_index]
        R_init=R_model[:,frame_trial_eval==trial_index][:,pos_trial>=pos][:,0][:,np.newaxis]
        R_init=cp.array(R_init)
        beh_init=beh_model[:,frame_trial_eval==trial_index][:,pos_trial>=pos][:,0][:,np.newaxis]
        beh_init=cp.array(beh_init)
        pos1=pos_trial[pos_trial>=pos][0]
        pos1_idx=int(pos1/5-1)
        net.set_target(neuron_target=R_init+cd[:,pos1_idx][:,cp.newaxis]*norm*((int(cues[trial_index])-0.5)*2),
                       neuron_target_type='r',
                       behavior_target=beh_init[:3],
                       true_env_states=beh_init)
        env.maze_init(cues[trial_index],*beh_init[:,0])
        input_generator=Ymaze_inputs(env)
        R,_,_,beh,_ = net.run(cur_noise=0,closed_loop=True,input_generator=input_generator)
        R_perturb = cp.hstack((R_perturb,
                        cp.hstack((cp.array(R_model[:,frame_trial_eval==trial_index][:,pos_trial<pos]),
                                R))))
        beh_output_perturb = cp.hstack((beh_output_perturb,
                        cp.hstack((cp.array(beh_model[:,frame_trial_eval==trial_index][:,pos_trial<pos]),
                                        beh))))
        frame_trial_perturb += [trial_index] * (R.shape[1]+np.sum(pos_trial<pos))  
    return R_perturb, beh_output_perturb, frame_trial_perturb


def CLV_closed_loop(net,env,
                    cues, activity, behavior,frame_trial,
                    eval_trial_idxes,
                    K,max_length):
    ratios_all = np.empty((0,K))
    Qs_all = np.empty((0,activity.shape[0],K))
    for trial_index in eval_trial_idxes:
        net.set_target(neuron_target=activity[:, frame_trial == trial_index],
                       neuron_target_type='r',
                       behavior_target=behavior[:3, frame_trial == trial_index],
                       true_env_states=behavior[:, frame_trial == trial_index])
        env.maze_init(cues[trial_index],*behavior[:, frame_trial == trial_index][:,0])
        input_generator=Ymaze_inputs(env)
        ratios,Qs = net.run_CLV(K=K,norm=0.01,renorm_period=1,closed_loop=True,input_generator=input_generator,
                                CLV=False,max_length=max_length)
        ratios_all = np.vstack((ratios_all,np.full((1,K),cp.nan),cp.asnumpy(ratios)))
        Qs_all = np.vstack((Qs_all, cp.asnumpy(Qs)))
    return ratios_all, Qs_all

def train_closed_loop(net,env,optim_inputs,optim_outputs,
                    cues, activity, behavior,frame_trial,
                    train_trial_idxes,
                    cur_noise=0.1,n_epoch=30):
    net.initialize_params()
    #train loop
    train_steps = 0
    train_generator = generator(train_trial_idxes)
    pbar = tqdm(total=n_epoch)
    while True:
        trial_index = next(train_generator)
        net.set_target(neuron_target=activity[:, frame_trial == trial_index],
                       neuron_target_type='r',
                       behavior_target=behavior[:3, frame_trial == trial_index],
                       true_env_states=behavior[:, frame_trial == trial_index])
        env.maze_init(cues[trial_index],*behavior[:, frame_trial == trial_index][:,0])
        input_generator=Ymaze_inputs(env)
        net.train(optim_inputs,optim_outputs, 1, cur_noise,
                  closed_loop=True,input_generator=input_generator,target_align=Ymaze_align)
        train_steps += 1

        # evaluate model performance and record learning curve
        if train_steps % train_trial_idxes.shape[0] == 0:
            pbar.update(1)
            idx_epoch = int(train_steps / train_trial_idxes.shape[0])
            if idx_epoch == n_epoch:
                return net.J_all,net.J_out

