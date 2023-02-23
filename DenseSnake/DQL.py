import random
from collections import deque
import numpy as np

class DQL:
    def __init__(self, model, actions, discount_factor=0.5, exploration_rate=0.9, memory_size=1000000, batch_size=500,base_decay_rate = 0.99995, decay_rate=0.95, base_exploration_rate = 0.1,validation_batch_size = 100):
        #NN
        self.model = model
        self.actions = actions
        #gamma
        self.discount_factor = discount_factor
        #for epsilon-greedy
        self.exploration_rate = exploration_rate
        #buffer
        self.memory = deque(maxlen=memory_size)
        self.evalmemory  = deque(maxlen = memory_size)
        self.batch_size = batch_size
        #diminish exploration 
        self.base_decay_rate = base_decay_rate
        self.decay_rate = decay_rate
        self.validation_batch_size = validation_batch_size
        self.base_exploration_rate = base_exploration_rate

    def get_action(self, state, direction, snake_list, block_size, width, height):
        action = np.random.randint(len(self.actions))
        possible_moves = list(range(0, len(self.actions)))
        # eliminate all impossible moves
        acts = list(self.actions)
        poss_copy = possible_moves.copy()
        for p in poss_copy:
            (u, v) = (snake_list[-1][0] + acts[p][1] * block_size, snake_list[-1][1] + acts[p][0] * block_size)
            if (u < 0 or u >= width or v < 0 or v >= height or [u, v] in snake_list):
                possible_moves.remove(p)
        if (len(possible_moves) > 0):
            if np.random.rand() < self.base_exploration_rate + self.exploration_rate:
                # Choose a random action
                action = possible_moves[np.random.randint(len(possible_moves))]
            else:
                # Choose the best action according to the model
                q_values = self.model.predict(np.array([state]), verbose=0)

                sorted = q_values[0].argsort()[::-1]
                for s in sorted:
                    if (s in possible_moves):
                        action = s
                        break

        return action

    def add_memory(self, state, action, reward, next_state, done,episode_reward):
        x = np.random.rand()

        if(x<0.9):
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.evalmemory.append((state, action, reward, next_state, done))
        episode_reward=episode_reward*self.discount_factor+reward
        return episode_reward
    def train(self,batch_size):
        if len(self.memory) < batch_size:
            # Not enough memories to train the model
            return
        # Randomly sample memories from the replay buffer
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        states = np.array(states)
        next_states = np.array(next_states)

        # Decrease the exploration rate
        self.exploration_rate *= self.decay_rate
        self.base_exploration_rate*= self.base_decay_rate

        # Calculate the target Q-values
        #qw(st+1,a[0],a[1],a[2].. )
        next_q_values = self.model.predict(next_states,verbose = 0)

        target_q_values = np.zeros((batch_size,len(self.actions)))

        for i in range(batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]

            else:
                target_q_values[i][actions[i]] = rewards[i] + self.discount_factor * max(next_q_values[i])

        # Train the model with the target Q-values
        # we want for qw(St,a) to become target_q[a]
        self.model.fit(states,target_q_values,epochs=1, verbose = 0)


    def evaluate(self):

        batch = random.sample(self.evalmemory, min(len(self.evalmemory),self.validation_batch_size))
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
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

        print(target_q_values[j],self.model.predict(np.array([states[j]]))[0])

        self.model.evaluate(states, target_q_values  )

        print("Exploration rate " , self.exploration_rate)