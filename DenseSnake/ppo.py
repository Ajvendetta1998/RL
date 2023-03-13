import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, activation):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activation = activation
    
    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        value = self.fc3(x)
        return value
    
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, activation, hidden_dim):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim,activation)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.MseLoss = nn.MSELoss()
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.activation = activation

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1))
        action_mean, std = self.actor(state)
        print(f"action_mean shape: {action_mean.shape}, std shape: {std.shape}")
        dist = Normal(action_mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        print(action)
        print(action_logprob)
        return action.item(), action_logprob.item()

    def update(self, state, action, reward, next_state, done):
        state = torch.Tensor(state.reshape(1, -1))
        next_state = torch.Tensor(next_state.reshape(1, -1))
        action = torch.Tensor([action])
        reward = torch.Tensor([reward])
        done = torch.Tensor([done])

        # Compute critic loss
        target = reward + self.gamma * (1 - done) * self.critic(next_state)
        delta = target - self.critic(state)
        critic_loss = delta**2

        # Optimize the critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Compute actor loss
        action_mean, std = self.actor(state)
        dist = Normal(action_mean, std)
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()

        advantage = delta.detach()
        advantage_logprob = advantage * action_logprob
        ratio = torch.exp(action_logprob - advantage_logprob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
        actor_loss = -torch.min(surr1, surr2) - 0.01*entropy.mean()

        # Optimize the actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

    def save(self, filename):
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'actor_optimizer_state_dict': self.optimizer_actor.state_dict(),
                    'critic_optimizer_state_dict': self.optimizer_critic.state_dict()}, filename)


    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_mean = torch.tanh(self.fc3(x))
        std = torch.ones_like(action_mean)
        return action_mean, std