import argparse
import torch
import numpy as np

import random
import os

from envs.Lock_batch import LockBatch
from envs.Dlock import DiabolicalLockMaze

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', default="atari", type=str)

    parser.add_argument('--env_name', default="MontezumaRevengeNoFrameskip-v4", type=str)

    parser.add_argument('--temp_path', default="temp", type=str)
    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--offline_buffer_path', default="offline_data/5000_old", type=str)

    parser.add_argument('--load', default=False, type=bool)
    parser.add_argument('--life_done', default=False, type=bool)
    parser.add_argument('--is_render', default=False, type=bool)

    parser.add_argument('--seed', default=12, type=int) 
    parser.add_argument('--num_warm_start', default=0, type=int)
    parser.add_argument('--num_timesteps', default=2**20, type=int)
    parser.add_argument('--num_trainsteps', default=15000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--num_offline', default=500000, type=int)
    parser.add_argument('--num_offline_warm', default=1000, type=int)

    parser.add_argument('--is_schedule', default=[0.6, 1.0, 1500000], type=list)
    parser.add_argument('--lr_schedule', default=[[0.0001, 0.0000625, 100000],
                                                [0.0000625, 0.00001, 15000000]], type=list)
    parser.add_argument('--replay_epsilon', default=0.001, type=float)
    parser.add_argument('--prio_coeff', default=0.4, type=float)

    parser.add_argument('--seperate_buffer', default=False, type=bool)
    parser.add_argument('--ratio_ann', default=False, type=bool)
    parser.add_argument('--dueling', default=False, type=bool)
    parser.add_argument('--huber', default=False, type=bool)

    parser.add_argument('--update_freq', default=4, type=int)
    parser.add_argument('--soft_update_freq', default=10000, type=int)
    parser.add_argument('--add_offline_freq', default=100, type=int)
    
    #environment
    parser.add_argument('--sticky_action', default=True, type=bool)
    parser.add_argument('--action_prob', default=0.25, type=float)

    #rep
    parser.add_argument('--fqi_num_update', default=1, type=int)
    parser.add_argument('--learning_rate', default=0.0000625, type=float)
    parser.add_argument('--beta', default=0.9, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)

    #eval
    parser.add_argument('--num_eval', default=20, type=int)

    args = parser.parse_args()
    return args


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    #if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    random.seed(seed)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )



class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def global_grad_norm_(parameters, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm



class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, num_actions, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.num_actions = num_actions

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape[1:]), dtype=np.uint8)
        self.actions = np.empty((capacity, num_actions), dtype=np.uint8)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        aoh = np.zeros(self.num_actions, dtype=np.uint8)
        aoh[action] = 1
        np.copyto(self.actions[self.idx], aoh)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs[-1,...])
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_batch(self, obs, action, reward, next_obs, done, size):
        np.copyto(self.obses[self.idx:self.idx+size], obs)
        aoh = np.zeros((size,self.num_actions), dtype=np.uint8)
        aoh[np.arange(size), action] = 1
        np.copyto(self.actions[self.idx:self.idx+size], aoh)
        np.copyto(self.rewards[self.idx:self.idx+size], reward)
        np.copyto(self.next_obses[self.idx:self.idx+size], next_obs)
        np.copyto(self.dones[self.idx:self.idx+size], done)

        self.idx = (self.idx + size) % self.capacity
        self.full = self.full or self.idx == 0

    def add_from_buffer(self, buf, batch_size = 1):
        obs, action, reward, next_obs, dones = buf.sample(batch_size = batch_size)
        np.copyto(self.obses[self.idx: self.idx + batch_size], obs.cpu().data.numpy())
        np.copyto(self.actions[self.idx: self.idx + batch_size], action.cpu().data.numpy())
        np.copyto(self.rewards[self.idx: self.idx + batch_size], reward.cpu().data.numpy())
        np.copyto(self.next_obses[self.idx: self.idx + batch_size], next_obs.cpu().data.numpy()[:,-1,...])
        np.copyto(self.dones[self.idx: self.idx + batch_size], dones.cpu().data.numpy())

        self.idx = (self.idx + batch_size) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size= batch_size 
            )

        obses = torch.as_tensor(self.obses[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.obses[idxs], device=self.device)
        next_obses[:,:-1,...] = next_obses[:,1:,...]
        next_obses[:,-1,...] = torch.as_tensor(self.next_obses[idxs], device=self.device)
        dones = torch.as_tensor(self.dones[idxs], device=self.device)
        
        return obses, actions, rewards, next_obses, dones

        
    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        print(chucks)
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.dones[start:end] = payload[4]
            self.idx = end
