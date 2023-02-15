import random
from collections import deque
import numpy as np

class DQL:
    def __init__(self, model, actions, discount_factor=0.95, exploration_rate=0.4, memory_size=100000, batch_size=20, decay_rate=0.995, base_exploration_rate = 0.1):
        self.model = model
        self.actions = actions
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.base_exploration_rate = base_exploration_rate
    
    def get_action(self, state):
        if np.random.rand() < self.base_exploration_rate + self.exploration_rate:
            # Choose a random action
            action = np.random.choice(range(len(self.actions)))
        else:
            # Choose the best action according to the model
            q_values = self.model.predict(state,verbose = 0)
            action = np.argmax(q_values)

        return action
    
    def add_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            # Not enough memories to train the model
            return
        # Randomly sample memories from the replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in batch:
            states.append(state[0])
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state[0])
            dones.append(done)
        states = np.array(states)
        next_states = np.array(next_states)

        # Decrease the exploration rate
        self.exploration_rate *= self.decay_rate

        
        # Calculate the target Q-values
        next_q_values = self.model.predict(next_states,verbose = 0)
        target_q_values = np.zeros((self.batch_size,len(self.actions)))

        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]

            else:
                target_q_values[i][actions[i]] = rewards[i] + self.discount_factor * max(next_q_values[i])
        # Train the model with the target Q-values
        self.model.fit(states, target_q_values, batch_size=self.batch_size, verbose=0)

