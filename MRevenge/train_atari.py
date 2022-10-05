import numpy as np
import torch
import os
import math
import gym
import sys
import random
import time
import json
import copy
from tqdm import tqdm
from collections import deque

import multiprocessing

from utils import parse_args, set_seed_everywhere, ReplayBuffer
from algs.FQI import FQI
from algs.FQI_huber import FQI_huber
from envs.atari_envs import AtariEnvironment

os.system('python experience_replay_setup.py build_ext --inplace')

from experience_replay import PrioritizedExperienceReplay



os.environ["OMP_NUM_THREADS"] = "1"


def eval(args, env, agent):
    print("eval:")
    obs = env.reset()
    while True:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action[0])
        obs = next_obs

        if done:
            break
    
    return env.rall, np.mean(env.recent_rlist), len(info.get('episode', {}).get('visited_rooms', {}))

def main(args):
    
    set_seed_everywhere(args.seed)

    env = AtariEnvironment(args.env_name, args.is_render, sticky_action=args.sticky_action, p=args.action_prob,
                        life_done=args.life_done,seed=args.seed)


    test_env = AtariEnvironment(args.env_name, args.is_render, sticky_action=args.sticky_action, p=args.action_prob,
                        life_done=args.life_done,seed=args.seed)


    from soco_device import DeviceCheck

    dc = DeviceCheck()
    device_name, device_ids = dc.get_device(n_gpu=1)

    if len(device_ids) == 1:
        device_name = '{}:{}'.format(device_name, device_ids[0])
        device = torch.device(device_name)
    print(device)

    if not os.path.exists(args.temp_path):
        os.makedirs(args.temp_path)


    # create buffer for each timestep, maybe should do the same for offline
    buffer =  PrioritizedExperienceReplay(max_frame_num = args.num_timesteps,
                                     num_stacked_frames = 4,
                                     batch_size = args.batch_size,
                                     prio_coeff = args.prio_coeff,
                                     is_schedule = args.is_schedule,
                                     epsilon = args.replay_epsilon,
                                     restore_path = None)
    eps_schedule = [[0.25, 0.1, 250000],
        [0.1, 0.01, 5000000],
        [0.01, 0.001, 15000000]]


    agent = FQI(env.env.action_space.n,
                    device,
                    num_update = args.fqi_num_update,
                    lr=args.learning_rate,
                    beta=args.beta,  
                    gamma=args.gamma,                     
                    temp_path = args.temp_path,
                    eps_schedule=eps_schedule,
                    lr_schedule=args.lr_schedule,
                    seperate_buffer=args.seperate_buffer,
                    ratio_ann=args.ratio_ann,
                    dueling=args.dueling,
                    huber=args.huber)

    buffer_path = args.offline_buffer_path

    offline_buffer = PrioritizedExperienceReplay(restore_path=buffer_path)


    obs = env.reset()
    buffer.add_experience(0, 0, obs[-1,...], 1, False, scale=False)

    for t in range(int(args.num_trainsteps)):
        
        action = agent.act(obs, t)
        next_obs, reward, done, info = env.step(action[0])
        buffer.add_experience(action[0], reward, next_obs[-1,...], 1, done, scale=False)
        obs = next_obs
        if done:
            for i in range(4):
                buffer.add_experience(0, 0, obs[i,...], 1, False, scale=False)

        if t % args.update_freq == 0:
            on_loss, off_loss, grad_norm, last_lr = agent.update(buffer, offline_buffer, t)
        
        if t % args.soft_update_freq == 0:
            agent.soft_update_target()
            average_on_loss = np.mean(on_loss)
            average_off_loss = np.mean(off_loss)
            average_norm = np.mean(grad_norm)

            ret, r_recent, num_rooms = eval(args, test_env, agent)
            wandb.log({ "Timestep": t,
                        "Episode": env.episode,
                        "Epi Step": env.steps,
                        "Test Reward:": ret,
                        "Train Recent Reward": np.mean(env.recent_rlist),
                        "Test Recent Reward": r_recent,
                        "epsilon": agent._get_current_epsilon(t),
                        "Visited rooms": num_rooms,
                        "on loss": average_on_loss,
                        "off loss": average_off_loss,
                        "average norm": average_norm,
                        "lr": last_lr
                        })
        if t % 100000 == 0:
            agent.save_q(t)

        
if __name__ == '__main__':

    args = parse_args()

    import wandb

    # comment this out if don't use wandb
    os.environ['WANDB_MODE'] = 'offline'

    with wandb.init(
            project= args.offline_buffer_path[13:],
            job_type="ratio_search",
            config=vars(args),
            name=args.exp_name):
        main(args)
