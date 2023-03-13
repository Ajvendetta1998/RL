import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Rollout():
    def __init__(self):
        self.states = []
        self.actions = []
        self.values = []
        self.log_probs = []
        self.rewards = []
    
    def add(self, state, action, value, log_prob, reward):
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.values = []
        self.log_probs = []
        self.rewards = []
        
    def compute_advantages(self, next_value, discount_factor, gae_lambda):
        returns = []
        advs = []
        gae = 0
        
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + discount_factor * next_value - self.values[i]
            gae = delta + discount_factor * gae_lambda * gae
            adv = gae + self.values[i]
            returns.insert(0, gae + self.values[i])
            advs.insert(0, adv)
            next_value = self.values[i]
        
        self.advantages = torch.tensor(advs, dtype=torch.float32).unsqueeze(1).to(device)
        self.returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(device)

# Define neural network for policy and value function estimation
class PolicyValueNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(PolicyValueNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

# Define PPO agent class
class PPOAgent():
    def __init__(self, input_size, hidden_size, num_actions, lr, gamma, eps_clip):
        self.policy_value_net = PolicyValueNet(input_size, hidden_size, num_actions).to(device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        action_probs, _ = self.policy_value_net(state)
        m = Categorical(action_probs)
        action = m.sample()
        return action.item()

    def update(self, rollout):
        states = torch.FloatTensor(rollout.states).to(device)
        actions = torch.LongTensor(rollout.actions).to(device)
        old_action_log_probs = torch.FloatTensor(rollout.action_log_probs).to(device)
        returns = torch.FloatTensor(rollout.returns).to(device)
        advantages = torch.FloatTensor(rollout.advantages).to(device)

        for _ in range(10): # perform 10 epochs of updates on the collected rollouts
            # Evaluate current policy
            action_probs, state_values = self.policy_value_net(states)
            dist = Categorical(action_probs)
            action_log_probs = dist.log_prob(actions)

            # Compute actor and critic losses
            ratios = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), returns)

            # Compute total loss and perform gradient update
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), 0.5)
            self.optimizer.step()

    def train(self, env, num_episodes):
        for i in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            rollout = Rollout()

            while not done:
                # Select action and take a step in environment
                action = self.select_action(state)
                next_state, reward, done = env.step(action)
                episode_reward += reward

                # Compute advantage and add transition to rollout buffer
                value = self.policy_value_net(torch.FloatTensor(state).to(device))[1]
                next_value = self.policy_value_net(torch.FloatTensor(next_state).to(device))[1]
                advantage = reward + (1 - done) * self.discount_factor * next_value.detach() - value.detach()
                rollout.add_transition(state, action, advantage, reward)

                # Update state for next iteration
                state = next_state

            # Compute returns and advantages for rollout buffer
            rollout.compute_returns(self.discount_factor)
            rollout.compute_advantages(self.policy_value_net, self.discount_factor, self.gae_lambda)

            # Train policy and value networks using rollout buffer
            self.policy_value_net.train_on_rollout(rollout, self.optimizer, self.clip_param, self.ppo_epochs, self.mini_batch_size)

            # Print episode results
            print(f"Episode {i+1}: Reward = {episode_reward:.2f}")



import pygame 
import random

from copy import deepcopy
import numpy as np 
# Font for displaying score
font = pygame.font.Font(None, 30)

# FPS
fps = 6
# Snake block size
block_size = 25


# Set display width and height
width = 500 
height = 500

pygame.init()   


# Create display surface
screen = pygame.display.set_mode((width, height))

# Set title for the display window
pygame.display.set_caption("Snake Game CNN")

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
grey = (100,100,100)
green = (0, 255, 0)
dark_green = (0, 100, 0)
# Set clock to control FPS
clock = pygame.time.Clock()
def display_score(score,gen,s,maxscore,episode_len):
    # Display current score
    text = font.render("Gen:" + str(gen) + " Len:" + str(score)+ " Scr: " + str(s) + " MaxScr: "+str(maxscore) + " EpLen: "+str(episode_len)+ " AvgLe: "+str(np.average(lens)), True, grey)
    screen.blit(text, [0,0])

def draw_snake(snake_list):
    # Draw the snake
    for block in snake_list[:-1]:
        pygame.draw.rect(screen, green, [block[0], block[1], block_size, block_size])
        pygame.draw.rect(screen, black, [block[0], block[1], block_size, block_size], 1)
    pygame.draw.rect(screen, dark_green, [snake_list[-1][0], snake_list[-1][1], block_size, block_size])
    pygame.draw.rect(screen, black, [snake_list[-1][0], snake_list[-1][1], block_size, block_size],1)

def generate_food(snake_list):
    # Generate food for the snake where there is no snake
    food_x, food_y = None, None
    
    while food_x is None or food_y is None or [food_x, food_y] in snake_list:
        food_x = round(random.randrange(0, width - block_size) / block_size) * block_size
        food_y = round(random.randrange(0, height - block_size) / block_size) * block_size
    return food_x, food_y


def state(snake_list,apple):

    #create an input vector starting from the apple, head .. the rest of the tail 
    input = np.zeros(input_size)
    input[0],input[1] = apple[0]/width,apple[1]/height
    for u in range(len(snake_list)):
        if(2*u+2>=input_size):
            break
        input[2*u+2],input[2*u+3]= s[len(snake_list)-1-u][0]/width,s[len(snake_list)-1-u][1]/height
    return input