import numpy as np
import torch
import os
import time
from tqdm import tqdm
from collections import deque

from utils import make_batch_env, parse_args, set_seed_everywhere, ReplayBuffer
from collect_offline import collect_offline_buffer
from algs.FQI import FQI


def make_agents(env, device, args):

    agents = []
    for h in range(args.horizon):
        agents.append(
            FQI(env.observation_space.shape[0],
                env.state_dim,
                env.action_dim,
                args.horizon,
                args.alpha,
                device,
                num_update=args.fqi_num_update,
                h=h,
                lr=args.learning_rate,
                beta=args.beta,
                batch_size=args.batch_size,
                tau=args.temperature,
                optimizer=args.optimizer,
                softmax=args.softmax,
                temp_path=args.temp_path)
        )
    return agents


def evaluate(env, agents, args):
    returns = np.zeros((args.num_eval, 1))

    obs = env.reset()
    for h in range(args.horizon):
        action = agents[h].act_batch(obs)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        returns += reward

    return np.mean(returns)


def fqi_update(agents, buffer, h, horizon, queue):

    target_agent = agents[h+1] if h < horizon-1 else None

    loss, grad_norm = agents[h].update(buffer, target_agent)

    # save weights to load later because multiproc does deep copy
    agents[h].save_q()

    if queue is not None:
        queue.put([h, loss, grad_norm[-1]])
    else:
        return loss[-1], grad_norm[-1]


def main(args):

    set_seed_everywhere(args.seed)

    # create environment, LockBatch
    env, eval_env = make_batch_env(args)

    num_actions = env.action_space.n

    device = torch.device("cpu")

    if not os.path.exists(args.temp_path):
        os.makedirs(args.temp_path)

    num_runs = int(args.num_episodes / args.horizon / args.num_envs)

    # create buffer for each timestep, maybe should do the same for offline
    buffers = []
    for _ in range(args.horizon):
        buffers.append(
            ReplayBuffer(env.observation_space.shape,
                         env.action_space.n,
                         int(args.num_episodes / args.horizon) *
                         4 + args.num_warm_start * args.num_envs,
                         args.batch_size,
                         device)
        )

    # create FQI policies, also one for each timestep
    agents = make_agents(env, device, args)

    if args.variable_latent:
        returns = deque(maxlen=50)
    else:
        returns = deque(maxlen=5)

    inference_start_time = time.time()

    epsilon = 1 / args.horizon

    offline_buffers = collect_offline_buffer(args, env, num_episodes=args.num_episodes, option=args.offline_dataset,
                                             verbose=True)
    # offline_buffers = collect_uniform_random_buffer_lock(args, env, args.num_episodes)
    logged_steps = 0

    # warm start from offline buffer
    for h in range(args.horizon):
        buffers[h].add_from_buffer(offline_buffers[h], args.warmup_size)

    for n in tqdm(range(num_runs)):

        counts = np.zeros((args.horizon, 3), dtype=np.int)

        for h in range(args.horizon):
            t = 0
            obs = env.reset()
            # roll in with current policy first
            while t < h:
                action = agents[t].act_batch(obs)
                next_obs, _, _, _ = env.step(action)
                obs = next_obs
                t += 1

            # then take a random action
            action = np.random.randint(0, num_actions, args.num_envs)
            next_obs, reward, done, _ = env.step(action)
            buffers[h].add_batch(obs, action, reward, next_obs, args.num_envs)

            buffers[h].add_from_buffer(offline_buffers[h], args.num_envs)

            count = env.get_counts()
            counts[h] = counts[h] + count

        # if n % 10 == 0:
        #     for agent in agents:
        #         agent.q_function.apply(weight_init)

        if n % args.update_frequency == 0:

            inference_time = time.time() - inference_start_time

            # assert args.horizon % args.num_threads == 0
            start_time = time.time()
            num_multi_runs = int(args.horizon / args.num_threads)

            loss_list = []
            norm_list = []

            # no multi-processing now
            for h in range(env.horizon-1, -1, -1):
                loss, grad_norm = fqi_update(
                    agents, buffers[h], h, args.horizon, None)
                loss_list.append(loss)
                norm_list.append(grad_norm)

            fqi_time = time.time() - start_time

            start_time = time.time()

            eval_return = evaluate(eval_env, agents, args)

            returns.append(eval_return)

            average_loss = np.mean(loss_list)
            average_norm = np.mean(norm_list)

            eval_time = time.time() - start_time

            # get how far we reached in the good states
            reached = 0
            for h in range(args.horizon):
                if counts[h, :2].sum() < 5:
                    reached = h
                    break

            wandb.log({"fqi_time": fqi_time,
                       "eval": np.mean(list(returns)) if args.variable_latent else eval_return,
                       "episode:": n * args.num_envs,
                       "reached": reached,
                       "state 0": counts[-1, 0],
                       "state 1": counts[-1, 1],
                       "episode:": n * args.num_envs * args.horizon,
                       "sampling time": inference_time,
                       "eval time": eval_time,
                       "average loss": average_loss,
                       "average norm": average_norm})

            np.save("{}/counts".format(args.temp_path), counts)

            logged_steps += 1
            # print(f'logging {logged_steps} step')

            inference_start_time = time.time()

            if np.mean(list(returns)) == 1 and not args.variable_latent and not args.dense:
                break


if __name__ == '__main__':

    args = parse_args()

    import wandb

    # comment this out if don't use wandb
    # os.environ['WANDB_MODE'] = 'offline'

    with wandb.init(
            project="fqi",
            job_type="ratio_search",
            config=vars(args),
            name=args.exp_name):
        main(args)
