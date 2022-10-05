import numpy as np
from zmq import device
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def kron(a, b):
    siz1 = torch.Size(torch.tensor(a.shape[-1:]) * torch.tensor(b.shape[-1:]))
    res = a.unsqueeze(-1) * b.unsqueeze(-2)
    siz0 = res.shape[:-2]
    return res.reshape(siz0 + siz1)

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
    def __init__(self, obs_dim, action_dim, device, state_dim=3, tau=1, softmax="vanilla"):
        super().__init__()

        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.tau = tau
        self.softmax = softmax

        self.encoder = nn.Linear(obs_dim, state_dim, bias=False)
        self.head = nn.Linear(state_dim * action_dim, 1, bias=True)

        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        state_encoding = self.encoder(obs)
        if self.softmax == "gumble":
            state_encoding = F.gumbel_softmax(state_encoding, tau=self.tau, hard=False)
        elif self.softmax == 'vanilla': 
            state_encoding = F.softmax(state_encoding / self.tau, dim=-1)
        phi = kron(action, state_encoding)
        q_values = self.head(phi)
        return q_values


    def encode_state(self, obs):
        # print(obs)
        obs = torch.FloatTensor(obs).to(self.device)
        state_encoding = self.encoder(obs)
        if self.softmax == "gumble":
            state_encoding = F.gumbel_softmax(state_encoding, tau=self.tau, hard=False)
        elif self.softmax == 'vanilla': 
            state_encoding = F.softmax(state_encoding / self.tau, dim=-1)
        
        return state_encoding

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


class FQI(object): 

    def __init__(
        self,
        obs_dim,
        state_dim,
        action_dim,
        horizon,
        alpha,
        device,
        num_update=30,
        h = 0,
        lr=1e-2,
        beta=0.9,
        batch_size = 128,
        tau = 1,
        optimizer = "Adam",
        softmax = "vanilla",
        temp_path = "temp"
    ):

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon

        self.feature_dim = state_dim * action_dim

        self.device = device
        self.lamb = 0.1
        #tau is the parameter for soft update the target network
        self.tau = 0.1

        self.q_function = QFunction(obs_dim, action_dim, device, tau=tau, softmax=softmax).to(device)
        self.target_q = None
        self.alpha = alpha

        self.num_update = num_update
        self.h = h

        self.temp_path = temp_path

        if optimizer == "Adam":
            self.q_optimizer = torch.optim.Adam(
                self.q_function.parameters(), lr=lr, betas=(beta, 0.999)
            )

        else:
            self.q_optimizer = torch.optim.SGD(
                self.q_function.parameters(), lr=lr, momentum=0.99
            )
            
    def q_values(self, obs):
        Qs = torch.zeros((len(obs),self.action_dim)).to(self.device)
        for a in range(self.action_dim):
            actions = torch.zeros((len(obs),self.action_dim)).to(self.device)
            actions[:,a] = 1
            #print(self.q_function(obs,actions).shape)
            #print(Qs[:,a].shape)
            Qs[:,a] = self.q_function(obs,actions).squeeze()
        return Qs

    def act_batch(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():   
            Qs = self.q_values(obs)
        action = torch.argmax(Qs, dim=1)

        return action.cpu().data.numpy().flatten()


    def act(self, obs, h):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            Qs = self.q_values(obs)
            action = torch.argmax(Qs, dim=1)

        return action.cpu().data.numpy().flatten()

    def update(self, buffer, target, accelerator = None):

        # self.optimizer = torch.optim.Adam(
        #         self.q_function.parameters(), lr=1e-2, betas=(0.9, 0.999)
        #     )

        #warmstart with the encoder
        if target is not None:
            self.q_function.encoder.weight.data.copy_(
                target.q_function.encoder.weight.data
            )

        loss_list = []
        grad_list = []
        if target is not None and accelerator is not None:
            target  = accelerator.prepare(target)
        if accelerator is not None:
            self.q_function = accelerator.prepare(self.q_function)
        for t in range(self.num_update-1):      
            obses, actions, rewards, next_obses = buffer.sample()
            # if t == self.num_update - 2:
            #     obses, actions, rewards, next_obses = buffer.sample(5000)
        #     if t > self.num_update//2 :
        #         self.q_function.encoder.requires_grad = False
        #     else:
        #         self.q_function.encoder.requires_grad = True

            obses = obses.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_obses = next_obses.to(self.device)
        
            # learn one step reward at the last timestep
            if self.h == self.horizon - 1:
                target_Q = rewards
            
            #compute target Q 
            else:
                with torch.no_grad():
                    Q_prime = torch.max(target.q_values(next_obses),dim=1)[0].unsqueeze(-1)
                target_Q = rewards + Q_prime
            # #update the exactly solved w
            # feature = kron(actions, self.q_function.encode_state(obses.cpu()))
            # feature = feature.reshape(feature.size(0),self.state_dim * self.action_dim)
            # target_Q = target_Q.reshape(feature.size(0),1)
            # Sigma = torch.matmul(feature.T, feature) + self.lamb * torch.eye(self.state_dim * self.action_dim).to(self.device)
            # W = torch.matmul(torch.inverse(Sigma), torch.sum(torch.mul(feature.unsqueeze(-1),target_Q.unsqueeze(-2)),0))
            # self.q_function.head.weight.data = W.T

            Qs = self.q_function(obses, actions)
            Qs = Qs.reshape(target_Q.size())
            # if self.h == 9 and t == self.num_update-2:
            #     print('======> h= 8')
            #     print(f'target Qs:{target_Q}')
            #     print(f'learned Qs: {Qs}')
            loss = F.mse_loss(Qs, target_Q)

            self.q_optimizer.zero_grad()
            if accelerator is None:
                loss.backward()
            else:
                accelerator.backward(loss)

            for param in self.q_function.parameters():
                grad_list.append(torch.norm(param.grad).item())
            self.q_optimizer.step()

            # if t == self.num_update - 2:
            #     break

            #update the exactly solved w
            # with torch.no_grad():
            #     feature = kron(actions, self.q_function.encode_state(obses))
            # feature = feature.reshape(feature.size(0),self.state_dim * self.action_dim)
            # target_Q = target_Q.reshape(feature.size(0),1)
            # Sigma = torch.matmul(feature.T, feature) + self.lamb * torch.eye(self.state_dim * self.action_dim).to(self.device)
            # W = torch.matmul(torch.inverse(Sigma), torch.sum(torch.mul(feature.unsqueeze(-1),target_Q.unsqueeze(-2)),0))
            # self.q_function.head.weight.data = W.T
            loss_list.append(loss.item()) 


        return loss_list , grad_list

    def save_q(self):
        self.q_function.save("{}/Q_{}.pth".format(self.temp_path,str(self.h)))

    def load_q(self):
        self.q_function.load("{}/Q_{}.pth".format(self.temp_path,str(self.h)))










