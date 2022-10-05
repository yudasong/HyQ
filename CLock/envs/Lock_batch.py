import numpy as np
import gym
from gym.spaces import Discrete, Box
import scipy.linalg
import math


'''
fast sampling. credit: https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix/34190035
'''


def sample(prob_matrix, items, n):

    cdf = np.cumsum(prob_matrix, axis=1)
    # random numbers are expensive, so we'll get all of them at once
    ridx = np.random.random(size=n)
    # the one loop we can't avoid, made as simple as possible
    idx = np.zeros(n, dtype=int)
    for i, r in enumerate(ridx):
        idx[i] = np.searchsorted(cdf[i], r)
    # fancy indexing all at once is faster than indexing in a loop
    return items[idx]


class LockBatch(gym.Env):
    """A (stochastic) combination lock environment.

    Can configure the length, dimension, and switching probability via env_config"""

    def __init__(self, env_config={}):
        self.initialized = False

    def init(self, horizon=100, action_dim=10, p_switch=0.5, p_anti_r=0.5, anti_r=0.1, noise=0.1, num_envs=10, temperature=1,
             variable_latent=False, dense=False):
        self.initialized = True
        self.max_reward = 1
        self.horizon = horizon
        self.state_dim = 3
        self.action_dim = action_dim
        self.action_space = Discrete(self.action_dim)

        self.reward_range = (0.0, 1.0)

        self.observation_dim = 2 ** int(math.ceil(np.log2(self.horizon+4)))

        self.observation_space = Box(low=0.0, high=1.0, shape=(
            self.observation_dim,), dtype=np.float)

        self.p_switch = p_switch
        self.p_anti_r = p_anti_r
        self.anti_r = anti_r
        self.noise = noise
        self.rotation = scipy.linalg.hadamard(self.observation_space.shape[0])

        self.num_envs = num_envs
        self.tau = temperature

        self.variable_latent = variable_latent
        self.dense = dense

        self.optimal_reward = 1
        if dense:
            self.step_reward = 0.1

        self.all_latents = np.arange(self.state_dim)

        self.opt_a = np.random.randint(
            low=0, high=self.action_space.n, size=self.horizon)
        self.opt_b = np.random.randint(
            low=0, high=self.action_space.n, size=self.horizon)

        print("[LOCK] Initializing Combination Lock Environment")
        print("[LOCK] A sequence: ", end="")
        print([z for z in self.opt_a])
        print("[LOCK] B sequence: ", end="")
        print([z for z in self.opt_b])

    def step(self, action):
        if self.h == self.horizon:
            raise Exception("[LOCK] Exceeded horizon")

        r = np.zeros((self.num_envs, 1))
        next_state = np.zeros(self.num_envs, dtype=np.int)
        ber = np.random.binomial(1, self.p_switch, self.num_envs)
        ber_r = np.random.binomial(1, self.p_anti_r, self.num_envs)
        # First check for end of episode
        if self.h == self.horizon-1:
            # Done with episode, need to compute reward
            for e in range(self.num_envs):
                if self.state[e] == 0 and action[e] == self.opt_a[self.h]:
                    r[e] = self.optimal_reward
                    if ber[e]:
                        next_state[e] = 1
                    else:
                        next_state[e] = 0
                elif self.state[e] == 1 and action[e] == self.opt_b[self.h]:
                    r[e] = self.optimal_reward
                    if ber[e]:
                        next_state[e] = 1
                    else:
                        next_state[e] = 0
                else:
                    if self.state[e] != 2 and ber_r[e]:
                        if not self.dense:
                            r[e] = self.anti_r
                    next_state[e] = 2
            self.h += 1
            self.state = next_state
            obs = self.make_obs(self.state)

            if self.variable_latent:
                self.sample_latent(obs)

            return obs, r, True, {}

        # Decode current state
        for e in range(self.num_envs):
            if self.state[e] == 0:
                # In state A
                if action[e] == self.opt_a[self.h]:
                    if self.dense:
                        r[e] = self.step_reward
                    if ber[e]:
                        next_state[e] = 1
                    else:
                        next_state[e] = 0
                else:
                    if ber_r[e]:
                        if not self.dense:
                            r[e] = self.anti_r
                    next_state[e] = 2
            elif self.state[e] == 1:
                # In state B
                if action[e] == self.opt_b[self.h]:
                    if self.dense:
                        r[e] = self.step_reward
                    if ber[e]:
                        next_state[e] = 0
                    else:
                        next_state[e] = 1
                else:
                    if ber_r[e] and not self.dense:
                        r[e] = self.anti_r
                    next_state[e] = 2
            else:
                # In state C
                next_state[e] = 2

        self.h += 1
        self.state = next_state
        obs = self.make_obs(self.state)

        if self.variable_latent:
            self.sample_latent(obs)

        return obs, r, False, {}

    def get_state(self):
        return self.state

    def get_counts(self):
        counts = np.zeros(3, dtype=np.int)
        for i in range(self.num_envs):
            counts[self.state[i]] += 1

        return counts

    def make_obs(self, s):

        gaussian = np.zeros((self.num_envs, self.observation_space.shape[0]))
        gaussian[:, :(self.horizon+self.state_dim)] = np.random.normal(0,
                                                                       self.noise, [self.num_envs, self.horizon+self.state_dim])
        gaussian[np.arange(self.num_envs), s] += 1
        gaussian[:, self.state_dim+self.h] += 1

        self.latents = gaussian[:, :3]

        x = np.matmul(self.rotation, gaussian.T).T

        return x

    def sample_latent(self, obs):

        latent_exp = np.exp(self.latents / self.tau)

        softmax = latent_exp / latent_exp.sum(axis=-1, keepdims=True)
        self.state = sample(softmax, self.all_latents, self.num_envs)

    def generate_obs(self, s, h):

        gaussian = np.zeros((self.num_envs, self.observation_space.shape[0]))
        gaussian[:, :(self.horizon+self.state_dim)] = np.random.normal(0,
                                                                       self.noise, [self.num_envs, self.horizon+self.state_dim])
        gaussian[:, s] += 1
        gaussian[:, self.state_dim+h] += 1

        x = np.matmul(self.rotation, gaussian.T).T

        return x

    def trim_observation(self, o, h):
        return (o)

    def reset(self, bad=False):
        if not self.initialized:
            raise Exception("Environment not initialized")
        self.h = 0

        self.state = np.random.binomial(1, self.p_switch, self.num_envs)

        if bad:
            self.state = np.zeros(self.num_envs, dtype=np.int)
            for i in range(self.num_envs):
                if np.random.rand() > 0.1:
                    self.state[i] = 2
                else:
                    self.state[i] = np.random.binomial(1, self.p_switch, 1)[0]
            self.h = 1

        obs = self.make_obs(self.state)

        return (obs)

    def render(self, mode='human'):
        if self.state == 0:
            print("A%d" % (self.h))
        if self.state == 1:
            print("B%d" % (self.h))
        if self.state == 2:
            print("C%d" % (self.h))
