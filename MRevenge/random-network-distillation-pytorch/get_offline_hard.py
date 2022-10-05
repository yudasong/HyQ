# compile cython modules
import os
from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe

from tensorboardX import SummaryWriter

import numpy as np
import pickle
os.system('python experience_replay_setup.py build_ext --inplace')

from experience_replay import PrioritizedExperienceReplay


def main():

    print({section: dict(config[section]) for section in config.sections()})
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'mario':
        env = BinarySpaceToDiscreteSpaceEnv(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    print(env.observation_space.shape)

    env.close()

    is_render = False
    model_path = 'ckpt/{}_{}.model'.format(env_id, str(ckpt))
    predictor_path = 'ckpt/{}_{}.pred'.format(env_id, str(ckpt))
    target_path = 'ckpt/{}_{}.target'.format(env_id, str(ckpt))

    use_cuda = False
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = 1

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])

    sticky_action = True
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    agent = RNDAgent

    if default_config['EnvType'] == 'atari':
        env_type = AtariSingleEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    else:
        raise NotImplementedError

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net
    )

    import os
    buffer_path = "../offline_data/hard"
    os.makedirs(buffer_path, exist_ok=True)


    env = env_type(env_id, is_render, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done)

    buffer = PrioritizedExperienceReplay()
    
    states = np.zeros([1, 4, 84, 84], dtype=np.uint8)
    s = env.reset()

    steps = 0
    rall = 0
    rd = False

    ckpt_list = ["expert", None]

    num_steps = 50000
    num_samples = num_steps * len(ckpt_list)

    intrinsic_reward_list = []

    buffer.add_experience(0, 0, s[-1,...], 1, False)
    
    for ckpt in ckpt_list:
        model_path = 'models/{}.model'.format(env_id)
        predictor_path = 'models/{}.pred'.format(env_id)
        target_path = 'models/{}.target'.format(env_id)
        
        if ckpt is not None:
            print('Loading Pre-trained model....')
            if use_cuda:
                agent.model.load_state_dict(torch.load(model_path))
                agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
                agent.rnd.target.load_state_dict(torch.load(target_path))
            else:
                agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
                agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
            print('End load...')

        else:
            num_steps = 200000

        for t in range(num_steps):
            steps += 1

            if ckpt is not None:
                actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)
                action = actions[0]
            
            else:
                action = np.random.randint(0, output_size, [1])[0]

            next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
            
            s, r, rd, lr = env.step(action)
            rall += r

            buffer.add_experience(action, r, s[-1,...], 1, rd)
            if rd:
                for i in range(4):
                    buffer.add_experience(0, 0, s[i,...], 1, False)

            next_states = s.reshape([1, 4, 84, 84])
            next_obs = s[3, :, :].reshape([1, 1, 84, 84])

            # total reward = int reward + ext Reward
            intrinsic_reward = agent.compute_intrinsic_reward(next_obs)
            intrinsic_reward_list.append(intrinsic_reward)
            states = next_states[:, :, :, :]


    buffer.save_internal_state(buffer_path)

if __name__ == '__main__':
    main()
