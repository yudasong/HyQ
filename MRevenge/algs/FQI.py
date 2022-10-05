from tkinter import X
import numpy as np
from zmq import device
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import utils


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class QFunction(nn.Module):
    def __init__(self,  action_dim, device, dueling=False):
        super().__init__()

        self.device = device
        self.action_dim = action_dim

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(
                7 * 7 * 64,
                256),
            nn.ReLU(),
            nn.Linear(
                256,
                448),
            nn.ReLU()
        )

        self.dueling = dueling

        if dueling:
            self.value_head = nn.Linear(448, self.action_dim)
            self.value_extra_layer = nn.Sequential(
                nn.Linear(448, 448),
                nn.ReLU()
            )

            self.adv_head = nn.Linear(448, self.action_dim)
            self.adv_extra_layer = nn.Sequential(
                nn.Linear(448, 448),
                nn.ReLU()
            )

        else:
            self.head = nn.Linear(448, self.action_dim)
            self.extra_layer = nn.Sequential(
                nn.Linear(448, 448),
                nn.ReLU()
            )

        self.apply(weight_init)

        

    def forward(self, obs):
        x = self.feature(obs)
        if self.dueling:
            v = self.value_head(self.value_extra_layer(x) + x)
            adv = self.adv_head(self.adv_extra_layer(x) + x)
            v = v + (adv - torch.mean(adv,-1,keepdim=True))
        else:
            v = self.head(self.extra_layer(x) + x)
        return v

    def q(self, obs, action):
        assert obs.size(0) == action.size(0)
        v = self(obs)
        q = torch.sum(torch.multiply(v,action),-1)

        return q

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


class FQI(object): 

    def __init__(
        self,
        action_dim,
        device,
        num_update=30,
        lr=1e-4,
        beta=0.9,
        gamma = 0.99,
        tau=0.05,
        temp_path = "temp",
        eps_schedule = [[0.25, 0.1, 50000],
                [0.1, 0.01, 1000000],
                [0.01, 0.001, 15000000]],
        double = True,
        lr_schedule = [[0.0001, 0.0000625, 100000],
                [0.0000625, 0.00001, 15000000]],
        offline_schedule = [[0.2, 0.2, 5000000],
                [0.2, 0.01, 15000000]],
        seperate_buffer = False,
        ratio_ann = False,
        dueling=False,
        huber=False
    ):

        self.action_dim = action_dim
        self.device = device

        self.q_function = QFunction(action_dim, device, dueling=dueling).to(device)
        self.target_q1 = QFunction(action_dim, device, dueling=dueling).to(device)
        self.target_q1.load_state_dict(self.q_function.state_dict())

        self.num_update = num_update

        self.temp_path = temp_path

        self.gamma = gamma
        self.tau = tau

        self.double = double
        self.dueling = dueling

        self.seperate_buffer = seperate_buffer
        self.fix_ratio = not ratio_ann

        self.eps_lag = 0
        self.lr_lag = 0
        self.ratio_lag = 0

        self.offline_schedule = np.array(offline_schedule)

        self.eps_schedule = np.array(eps_schedule)

        self.q_optimizer = torch.optim.Adam(
                self.q_function.parameters(), lr=1, betas=(beta, 0.999), weight_decay=1e-5
            )
        self.lr_schedule = np.array(lr_schedule)

        lambda_q = lambda t: self._get_current_lr(t)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.q_optimizer, lr_lambda=lambda_q)

        self.on_loss = 0
        self.off_loss = 0

        if huber:
            self.loss_func = nn.HuberLoss(reduction='none')
        else:
            self.loss_func = nn.MSELoss(reduction='none')

            
    def _get_current_epsilon(self, t):
        if t > self.eps_schedule[0, 2] and self.eps_schedule.shape[0] > 1:
            self.eps_schedule = np.delete(self.eps_schedule, 0, 0)
            self.eps_lag = t
        max_eps, min_eps, eps_steps = self.eps_schedule[0]
        epsilon = max_eps - min(1, (t - self.eps_lag) / (eps_steps - self.eps_lag)) * (max_eps - min_eps)
        return epsilon

    def _get_current_lr(self, t):
        if t > self.lr_schedule[0, 2] and self.lr_schedule.shape[0] > 1:
            self.lr_schedule = np.delete(self.lr_schedule, 0, 0)
            self.lr_lag = t
        max_lr, min_lr, lr_steps = self.lr_schedule[0]
        lr = max_lr - min(1, (t - self.lr_lag) / (lr_steps - self.lr_lag)) * (max_lr - min_lr)
        return lr

    def _get_ratio(self, t):
        if t > self.offline_schedule[0, 2] and self.offline_schedule.shape[0] > 1:
            self.offline_schedule = np.delete(self.offline_schedule, 0, 0)
            self.ratio_lag = t
        max_lr, min_lr, lr_steps = self.offline_schedule[0]
        ratio = max_lr - min(1, (t - self.ratio_lag) / (lr_steps - self.ratio_lag)) * (max_lr - min_lr)
        return ratio



    def act(self, obs, t=None):
        if t is not None:
            eps = self._get_current_epsilon(t)
            if np.random.rand() < eps:
                action = np.random.randint(0, self.action_dim, [1])
                return action
        
        with torch.no_grad():
            obs = torch.FloatTensor(obs / 255.).to(self.device)
            obs = obs.unsqueeze(0)
            Qs = self.q_function(obs)
            action = torch.argmax(Qs, dim=1)

        return action.cpu().data.numpy().flatten()

    def compute_loss(self, buffer):
        for t in range(self.num_update):      
            obses, actions, rewards, next_obses, dones, weights = buffer.get_mini_batch()
            obses = torch.as_tensor(obses, device=self.device)
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.int64)
            rewards = torch.as_tensor(rewards, device=self.device)
            next_obses = torch.as_tensor(next_obses, device=self.device)
            dones = torch.as_tensor(dones, device=self.device)
            weights = torch.as_tensor(weights, device=self.device)
            
            obses = obses / 255.
            next_obses = next_obses / 255.
            rewards = torch.sign(rewards) * torch.log(1 + torch.abs(rewards))
        
            
            with torch.no_grad():
                if self.double:
                    Qp = self.q_function(next_obses)
                    actionp = torch.argmax(Qp, dim=1)
                    Q1 = self.target_q1(next_obses).gather(1, actionp.unsqueeze(1)).squeeze()
                    target_Q = rewards + (1-dones) * self.gamma * Q1
                else:
                    Q1 = self.target_q1(next_obses)
                    target_Q = rewards + (1-dones) * self.gamma * torch.max(Q1, dim=1)[0].reshape(-1,1)
                
            Qs = self.q_function(obses).gather(1, actions.unsqueeze(1))
            Qs = Qs.reshape(target_Q.size())

            loss = self.loss_func(Qs, target_Q)
            mean_loss = torch.mean(loss * weights)
            
            buffer.update_mini_batch_priorities(loss.cpu().data.numpy())

        return mean_loss

    def update(self, buffer, offline_buffer, t):

        grad_list = []

        if self.fix_ratio:
            offline_ratio = 0.2
        else:
            offline_ratio = self._get_ratio(t)

        if not self.seperate_buffer:
            mean_on_loss = self.compute_loss(buffer)
            mean_off_loss = self.compute_loss(offline_buffer)
            loss = mean_on_loss + mean_off_loss * offline_ratio

            self.on_loss = mean_on_loss.item()
            self.off_loss = mean_off_loss.item()
        else:
            if np.random.rand() < offline_ratio:
                loss = self.compute_loss(offline_buffer)
                self.off_loss = loss.item()
            else:
                loss = self.compute_loss(buffer)
                self.on_loss = loss.item()
      
        self.q_optimizer.zero_grad()
        loss.backward()

        for param in self.q_function.parameters():
            grad_list.append(torch.norm(param.grad).item())

        torch.nn.utils.clip_grad_norm_(self.q_function.parameters(), 10)

        self.q_optimizer.step()
        self.scheduler.step()


        return self.on_loss, self.off_loss, grad_list, self.scheduler.get_last_lr()[0]

    def soft_update_target(self):
        self.target_q1.load_state_dict(self.q_function.state_dict())

    def save_q(self, t):
        self.q_function.save("{}/Q_{}.pth".format(self.temp_path,str(t)))

    def load_q(self, t):
        self.q_function.load("{}/Q_{}.pth".format(self.temp_path,str(t)))