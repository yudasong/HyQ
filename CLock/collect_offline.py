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
import pickle

from collections import deque

from utils import set_seed_everywhere, ReplayBuffer
(3)


os.environ["OMP_NUM_THREADS"] = "1"


def evaluate_policy(env, epsilon, args):
    returns = np.zeros((args.num_envs, 1))

    obs = env.reset()
    for h in range(args.horizon):
        action = eps_greedy_actions(env, args, epsilon)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        # print(reward)
        returns += reward

    return np.mean(returns)


def eps_greedy_actions(env, args, epsilon=-1, random_step=True):
    """
    return a list of epsilon greedy actions
    """
    num_envs = env.num_envs
    h = env.h
    if epsilon == -1:
        epsilon = 1/env.horizon
    action = np.array([env.opt_a[h] if state == 0 else env.opt_b[h]
                      for state in env.get_state()])
    random_actions = np.random.randint(
        low=0, high=env.action_space.n, size=env.num_envs)
    ber = np.random.binomial(1, 1 - epsilon, env.num_envs)
    action = np.where(ber, action, random_actions)
    if random_step and h == args.horizon//2:
        action = random_actions
    return action


def collect_offline_buffer(args, env, num_episodes, option="epsilon", verbose=False, buffers=None):
    """
    collect offline replay buffer with an epsilon greedy policy
    Args:
        - :param: `args` (parsed argument): the main arguments
        - :param: 'num_episodes': the number of episodes to collect
        - :param: 'option': this method supports two type of datasets, one is 
        option 'epsilon' and the other one is 'mixed', more details in the paper
        - :param: 'verbose': if set to True, print out fraction of episodes that
            reach the end
        - :param: 'buffers': a list of H buffers to start with

    Return:
        - :param: 'buffers': a list of ReplayBuffer of the number of horizon
    """

    set_seed_everywhere(args.seed)

    device = torch.device("cpu")

    num_runs = int(num_episodes / env.horizon / env.num_envs)

    # num_reaches keep track of the number of episodes that make to the end
    num_reaches = 0
    if buffers == None:
        buffers = []
        for _ in range(args.horizon):
            buffers.append(
                ReplayBuffer(env.observation_space.shape,
                             env.action_space.n,
                             int(num_episodes / args.horizon)*3+1,
                             args.batch_size,
                             device)
            )

    if args.dense:
        args.alpha = args.horizon / 50
    else:
        args.alpha = args.horizon / 5

    if args.variable_latent:
        returns = deque(maxlen=50)
    else:
        returns = deque(maxlen=5)

    collection_time = time.time()
    horizon_episodes = 0
    for n in tqdm(range(num_runs)):
        for h in range(args.horizon):
            t = 0
            obs = env.reset()
            while t < h:
                if option == 'epsilon':
                    action = eps_greedy_actions(
                        env, args, -1, random_step=True)
                else:
                    action = eps_greedy_actions(
                        env, args, 0, random_step=False)
                next_obs, _, _, _ = env.step(action)
                obs = next_obs
                t += 1
            # action = np.random.randint(0, num_actions, args.num_envs)
            if option == 'epsilon':
                action = eps_greedy_actions(env, args, -1, random_step=True)
            else:
                action = eps_greedy_actions(env, args, 1, random_step=False)
            next_obs, reward, done, _ = env.step(action)
            buffers[h].add_batch(obs, action, reward, next_obs, args.num_envs)

            if h == args.horizon - 1:
                count = env.get_counts()

        num_reaches += count[:2].sum()
        # print(counts[-1,:2])
    print(f'num_reaches: {num_reaches}')
    print(f'horizon episodes: {horizon_episodes}')
    collection_time = time.time() - collection_time
    if verbose:
        print(
            f"fraction of episodes reach the end: {num_reaches/num_episodes * env.horizon}")
        print(f'it took {collection_time}s')
    return buffers
