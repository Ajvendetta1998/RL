import random
from collections import deque
import numpy as np
from itertools import chain
class DQL:
    def __init__(self, model, actions, discount_factor=0.95, exploration_rate=0.3, memory_size=100000, batch_size=100, decay_rate=0.995, base_exploration_rate = 0.1):
        #NN
        self.model = model

        self.actions = actions
        #gamma
        self.discount_factor = discount_factor
        #for epsilon-greedy
        self.exploration_rate = exploration_rate
        #buffer
        self.memory = deque(maxlen=memory_size)

        self.batch_size = batch_size
        #diminish exploration 
        self.decay_rate = decay_rate
        self.base_exploration_rate = base_exploration_rate
        #buffer for evaluation
        self.evalmemory = deque(maxlen = memory_size)

    def get_action(self, state, direction,length):
        if np.random.rand() < self.base_exploration_rate + self.exploration_rate:
            # Choose a random action
            l = list(range(0,len(self.actions)))

            l.remove(direction)
            action = l[np.random.choice(len(self.actions)-1)]
        else:
            # Choose the best action according to the model
            q_values = self.model.predict(state,verbose = 0)
            sorted = q_values.argsort()
            action = sorted[0][0]
            if(action == direction and length>1):
                action= sorted[0][1]


        return action
    
    def add_memory(self, state, action, reward, next_state, done):
        x = np.random.rand()

        if(x<0.9):
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.evalmemory.append((state, action, reward, next_state, done))

    
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

        #A VERIFIER :
        # Calculate the target Q-values
        #qw(st+1,a[0],a[1],a[2].. )
        next_q_values = self.model.predict(next_states,verbose = 0)

        target_q_values = np.zeros((self.batch_size,len(self.actions)))

        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]

            else:
                target_q_values[i][actions[i]] = rewards[i] + self.discount_factor * max(next_q_values[i])

        # Train the model with the target Q-values
        # we want for qw(St,a) to become target_q[a]
        self.model.fit(states, target_q_values, verbose=0)

    def evaluate(self):
        if len(self.evalmemory) < self.batch_size:
            # Not enough memories to train the model
            return

        batch = random.sample(self.evalmemory, self.batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in batch:
            states.append(state[0])
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state[0])
            dones.append(done)
        states = np.array(states)
        next_states = np.array(next_states)
        # Calculate the target Q-values
        next_q_values = self.model.predict(next_states,verbose = 0)
        target_q_values = np.zeros((len(batch),len(self.actions)))

        for i in range(len(batch)):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]

            else:
                target_q_values[i][actions[i]] = rewards[i] + self.discount_factor * max(next_q_values[i])
        j = np.random.randint(len(target_q_values))
        #print(target_q_values[j],self.model.predict(np.array([states[j]]))[0])
        #print(np.linalg.norm(target_q_values[j]-self.model.predict(np.array([states[j]]))[0]))
        #print(abs(np.argmax(target_q_values[j])- np.argmax(self.model.predict(np.array([states[j]]),verbose = 0)[0])))
        self.model.evaluate(states, target_q_values ,verbose = 0 )