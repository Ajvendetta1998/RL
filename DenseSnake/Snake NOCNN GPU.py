
import pygame 
import random
from copy import deepcopy
import numpy as np 
import os 
import sys 
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf


class DQL:
    def __init__(self, model, criterion, optimizer, actions, discount_factor=0.6, exploration_rate=0.9, memory_size=100000, batch_size=500,base_decay_rate = 0.995, decay_rate=0.98565207, base_exploration_rate = 0.1,validation_batch_size = 100 ,epochs = 1,device = 'cpu'):
        #NN
        self.model = model
        self.actions = actions
        self.epochs = epochs
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
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
        action = direction
        possible_moves = list(range(0, len(self.actions)))

        # eliminate all impossible moves
        acts = list(self.actions)
        poss_copy = possible_moves.copy()
        for p in poss_copy:
            (u, v) = (snake_list[-1][0] + acts[p][1] * block_size, snake_list[-1][1] + acts[p][0] * block_size)
            if (u < 0 or u >= width or v < 0 or v >= height or [u, v] in snake_list):
                possible_moves.remove(p)

        if len(possible_moves) > 0:
            if np.random.rand() < self.base_exploration_rate + self.exploration_rate:
                # Choose a random action
                action = possible_moves[np.random.randint(len(possible_moves))]
            else:
                # Choose the best action according to the model
                with torch.no_grad():

                    q_values = self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
                    q_values = q_values.squeeze(0)

 
                q_sorted = np.array([q_values.cpu()[i] for i in possible_moves])
                gamma = 5
                soft_maxed = np.exp(gamma *q_sorted)/sum(np.exp(gamma * q_sorted))

                action = possible_moves[np.random.choice(len(soft_maxed),p = soft_maxed)]
                #action = possible_moves[q_sorted.argmax()]

        return action


    def add_memory(self, state, action, reward, next_state, done,episode_reward):
        x = np.random.rand()

        if(x<0.9):
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.evalmemory.append((state, action, reward, next_state, done))
        episode_reward=episode_reward*self.discount_factor+reward
        return episode_reward
    
    def train(self, batch_size):
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

        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        # Decrease the exploration rate
        self.exploration_rate *= self.decay_rate
        self.base_exploration_rate *= self.base_decay_rate

        target_q_values = np.zeros((batch_size,len(self.actions)))

        # Calculate the target Q-values
        with torch.no_grad():
            next_q_values = self.model(next_states).to(self.device)
            for i in range(batch_size):
                if dones[i]:
                    target_q_values[i][actions[i]] = rewards[i]

                else:
                    target_q_values[i][actions[i]] = rewards[i] + self.discount_factor * max(next_q_values[i])
        target_q_values = torch.tensor(target_q_values).float().to(self.device)

            # Train the model for a specified number of epochs
        for epoch in range(self.epochs):
            predicted_q_values = self.model(states)
            loss = nn.functional.mse_loss(predicted_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Snake block size
block_size = 25


# Set display width and height
width = 500 
height = 500

#heatmap = np.zeros((height // block_size, width // block_size))
#plt.imshow(heatmap, cmap='hot', interpolation='nearest')
#plt.show(block=False)

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

penalty_names  = ['accessible_points_proportion','penalty_distance','penalty_touch_self','penalty_distance*gass_reward','reward_eat','penalty_wall','penalty_danger','compacity','episode_len_penalty']
c = np.array([0.4,0.5,0.3,0.1,0.6,0.3,0.2,0.1,0.4])
#c = np.array([0.5525170689977776,0.9582674747339271,0.025280188852446105,0.5910711871018972,0.820824845042075,0.11647511348340267,0.4174262886175807,0.18321977917311702,0.604671417871147])
c = np.array([0.3869715852256136,0.5254869870433494,0.2203822793872801,0.40087986200533904,0.3399549149174315,0.29913072453369155,0.5875165725424327,0.3464123490760858,0.4896371909472169])
c = np.array([0.36457211116822963,0.7930600985182027,0.45402896962721,0.35221863030539163,0.3303118328610477,0.35520454891348063,0.2902639320811671,0.5488430988037623,0.4264797443049767])
c = np.array([0.28294775819658663,0.8728298782729212,0.5167730024282512,0.38563350087969495,0.4814133212546019,0.2419615008911828,0.3914063628345444,0.6103051401118678,0.29985089780131485])
c = np.array([0.4863077196643149,0.8176132430371873,0.560514500321722,0.2359458810412274,0.8130391965064794,0.5729984548003503,0.02492140605359113,0.5531607672522686,0.4327075132071214])
c = np.array([0.5963010289868701,0.608367565016974,0.44956136139029124,0.5828203469830129,0.657816743464842,0.2014442632543414,0.19307960898165127,0.5540849999385403,0.03983106446551208])
c = np.array([0.5963010289868701,0.608367565016974,0.44956136139029124,0.5828203469830129,0.657816743464842,0.2014442632543414,0.19307960898165127,0.5540849999385403,0.03983106446551208])
c= np.array([0.1863077196643149,0.8176132430371873,0.560514500321722,0.2359458810412274,0.5130391965064794,0.4729984548003503,0.02492140605359113,0.1531607672522686,0.4327075132071214])
# Font for displaying score
font = pygame.font.Font(None, 30)

# FPS
fps = 6

def game_over():
    # Display Game Over message
    text = font.render("Game Over!", True, red)
    screen.blit(text, [width/2 - text.get_width()/2, height/2 - text.get_height()/2])
    
    '''pygame.display.update()
    pygame.time.wait(3000)
    pygame.quit()
    sys.exit()'''
lens = []
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


#dictionary of possible actions 
actions = {"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}

#(x,y) apple + snake body (which has at most width*height parts) 
input_size = 2*width*height//block_size**2+2



import torch
import torch.nn as nn
import torch.optim as optim
device = 'cuda'
def initNNmodel():
    # Define the input shape
    input_shape = (3, height//block_size, width//block_size)

    # Create a Sequential model
    model = nn.Sequential(
        nn.Linear(input_size, 4096),
        nn.ReLU(),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, len(actions))
    )

    # Send the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Print the model summary
    print(model)

    return model, criterion, optimizer
filename = 'NOCNNGPU.pt'
def save_model(model, optimizer, criterion, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
    }, filename)

def load_model(model, optimizer, criterion, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion.load_state_dict(checkpoint['criterion_state_dict'])


#returns a vector that has the state of snake and apple ( the NN input vector ) 

def state(snake_list,apple):
    s = np.array(snake_list)
    #mini0 = int(min(s[:,0].min(),apple[0]))
    #mini1 = int(min(s[:,1].min(),apple[1]))
    #maxi0 = int(max(s[:,0].max(),apple[0]))
    #maxi1 = int(max(s[:,1].max(),apple[1]))

    #centralize the game
    #apple[0]-=(mini0+maxi0-width)//2
    #apple[1] -= (mini1+maxi1-height)//2
    #s[:,0] -=(mini0+maxi0-width)//2
    #s[:,1]-= (mini1+maxi1-height)//2
    #create an input vector starting from the apple, head .. the rest of the tail 
    input = np.zeros(input_size)
    input[0],input[1] = apple[0]/width,apple[1]/height
    for u in range(len(snake_list)):
        if(2*u+2>=input_size):
            break
        input[2*u+2],input[2*u+3]= s[len(snake_list)-1-u][0]/width,s[len(snake_list)-1-u][1]/height
    return input

def normalized_distance(u,v,food_x,food_y):
    return np.sqrt((((u-food_x)/width)**2+((v-food_y)/height)**2)/2)

def inBounds(u,v):
    if(u>=0 and v>=0):
        if(u<width and v<height):
            return True
    return False

def gaussian_aroundone(x,alpha):
    return(np.exp(-alpha*(x-1)**2))
def danger_distance(direction, snake_list):
    dis =0  
    acts = list(actions.values())
    (u,v) = (snake_list[-1][0],snake_list[-1][1])
    while (inBounds(u,v)):
        u +=block_size*acts[direction][1]
        v+=block_size*acts[direction][0]
        if([u,v] in snake_list[1:]):
            return (-1+1.0*dis*block_size/max(width,height))
        dis+=1
    return(0)

#reward function for each state and action
def reward(action, snake_list,episode_length):
    copy = deepcopy(snake_list)
    p = copy[-1]
    a = list(actions.values())
    (u,v)=(a[action][1]*block_size+p[0],a[action][0]*block_size+p[1])
    penalty_touch_self = 0 
    if [u,v] in snake_list:
        penalty_touch_self=-1 # return a negative reward if the snake collides with itself
    copy.append([u,v])
    del copy[0]
    
    global food_x, food_y
    # reward the agent for getting closer to the food
    reward_distance = 1-normalized_distance(u,v,food_x,food_y)
    #if too far then the reward is very close to 0 
    gass_reward =gaussian_aroundone(reward_distance,20)
    # reward the agent for eating the food
    reward_eat = 1 if u == food_x and v == food_y else 0
    # penalize the agent for moving away from the food
    penalty_distance = -1 if normalized_distance(u,v,food_x,food_y) > normalized_distance(p[0], p[1], food_x, food_y) else 0.5
    # penalize he agent for hitting a wall
    penalty_wall = -1 if not (inBounds(u,v)) else 0.2
    #penalize the agent for getting closer to danger
    penalty_danger = danger_distance(action,snake_list)
    #print(penalty_danger)
    compacity_value = 1.0/compacity(snake_list)
    #accessible points 
    accessible_points_proportion = find_accessible_points(snake_list)-1
    episode_length_penalty = -episode_length/(width*height//block_size**2)
    penalties = np.array([accessible_points_proportion,penalty_distance,penalty_touch_self,penalty_distance*gass_reward,reward_eat,penalty_wall,penalty_danger,compacity_value,episode_length_penalty])
  


    total_reward = penalties@c/c.sum() 

   # total_reward = accessible_points_proportion*gass_reward
    #total_reward = penalty_distance + 10*reward_eat
    #print(total_reward, penalty_danger)
    #if(inBounds(u,v)):
     #  heatmap[u//block_size,v//block_size] = total_reward

    #print([q+" = "+str(p) + " , " for p,q in zip(penalties,penalty_names)])
    return total_reward

def compacity(snake_list):
    snake_list = np.array(snake_list)
    min_x = snake_list[:,0].min()
    min_y = snake_list[:,1].min()
    max_x = snake_list[:,0].max()
    max_y = snake_list[:,1].max()
    return((max_y-min_y+block_size)*(max_x-min_x+block_size)/(len(snake_list)*block_size**2))   


#if all cells are accessible 
def find_accessible_points(snake_list):
    accessible_points= np.zeros((height//block_size,width//block_size))
    head_position = snake_list[-1]
    explore = [head_position]

    while(len(explore)>0):
        p = explore.pop()
        accessible_points[p[1]//block_size,p[0]//block_size]=1
        for m in actions.values():
            (u,v)=(m[1]*block_size+p[0],m[0]*block_size+p[1])
            if(inBounds(u,v)):
                if(accessible_points[v//block_size,u//block_size]==0):
                    if(not [u,v] in snake_list):
                        explore.append((u,v))

    return((np.sum(accessible_points)+len(snake_list)-1)*block_size**2/(height*width))

model, criterion, optimizer = initNNmodel()
if(os.path.exists(filename)):

    load_model(model,optimizer,criterion,filename)
    print("model loaded")
else:
    print("made new model")
    save_model(model,optimizer,criterion,filename)

max_exploration_episodes = 100
max_exploration_rate = 0.9
min_exploration_rate = 0.05
decay_rate = (min_exploration_rate/max_exploration_rate)**(1.0/max_exploration_episodes)
#initialize deepQ
dql = DQL(model, criterion, optimizer,actions.values(),decay_rate = decay_rate ,exploration_rate= max_exploration_rate ,device = device,epochs=5)
# Initialize pygame

food_x, food_y = generate_food([])   

def main(gen,length,maxlen):
    episode_reward = 0
    # Initial snake position and food
    snake_x = (width//block_size)//2     * block_size   
    snake_y = (height//block_size)//2     * block_size   

    snake_list = [[snake_x,snake_y]]
    
    global food_x,food_y
    #food_x, food_y = generate_food([])   
    episode_length =0 
    # Initial snake direction and length
    direction = "right"
    snake_length =length
    
    # Update the heatmap
    #plt.clf()  # clear the previous heatmap
    #plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    #plt.pause(0.001)  # show the updated heatmap for a short duration
    prev_direction = "right"
    #direction of the snake [0,1,2,3] each corresponding to one of up down left right
    a=0
    # = True if snake hits wall or itself
    done = False
    # number of collected fruit
    score = 0
    acts = list(actions.values())
    # Game loop
    check = 0 
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # Start a new game
                main()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


        St1 = state(snake_list,[food_x,food_y]) 
        a= dql.get_action(St1,a,snake_list,block_size,width,height)
        r = reward(a,snake_list,episode_length-check)

        #move snake
        snake_x+=acts[a][1]*block_size
        snake_y+=acts[a][0]*block_size
        # Add new block of snake to the list
        snake_list.append([snake_x, snake_y])
        # Keep the length of snake same as snake_length
        if len(snake_list) > snake_length:
            del snake_list[0]

        # Check if snake hits the boundaries
        if snake_x >= width or snake_x < 0 or snake_y >= height or snake_y < 0:
            done = True
        # Check if snake hits itself
        for block in snake_list[:-1]:
            if block[0] == snake_x and block[1] == snake_y:
                done = True
                break

        # Fill the screen with white color
        screen.fill(white)

        # Display food
        pygame.draw.rect(screen, red, [food_x, food_y, block_size, block_size])
        pygame.draw.rect(screen, green, [food_x + block_size/3, food_y, block_size/3, block_size/3])

        # Draw the snake
        draw_snake(snake_list)

        St2 = state(snake_list,[food_x,food_y])
        #add to Buffer
        episode_reward= dql.add_memory(St1,a,r,St2,done,episode_reward)

        # Display score and other metrics
        display_score(snake_length-1,gen,score,maxlen,episode_length-check,)

        # Update the display
        pygame.display.update()

        # Check if snake hits the food
        if snake_x == food_x and snake_y == food_y:
            lens.append(episode_length-check)
            food_x, food_y = generate_food(snake_list)
            snake_length += 1
            score+=1
            check = episode_length
        episode_length+=1
        '''if((episode_length-check)%((width*height//block_size**2)) ==0 ):
            dql.train(episode_length-check)'''
        if((episode_length-check)%(3*(width*height//block_size**2)) ==0 ):
  
            #dql.exploration_rate/= dql.decay_rate
            done = True
        #if snake has hit something quit
        if(done):
            return [snake_length,episode_length,score,episode_reward]
        
        # Set the FPS
        #clock.tick(fps)
        #dql.train()


max_m = 0
test = 0
#number of episodes
num_episodes =10000000
#maximum score reached
m =0 
#initial max length for the snake at birth
max_length = 1
#maximum allowed length for a snake 
max_max_length = width*height//block_size**2
#max_length = width*height//block_size**2//20

#generation of episodes 
for i in range(num_episodes):

    #do a generation and see the outcome
    a= main(i,np.random.randint(1,max_length+1),m)
    #update maximum score 
    m = max(a[2],m)
    #generate a new food position every 20 generations
    if(i%2 ==0):
        food_x, food_y = generate_food([])   
    if(i%40==0):
        save_model(model,optimizer,criterion,filename)
    #increase maximum birth length every 1000 generation 
    
        #dql.exploration_rate = 0.9
    print("episode reward : ", a[-1], "Exploration Rate : ", dql.exploration_rate+dql.base_exploration_rate)
    #train the DQL 
    dql.train(a[1])
    #dql.evaluate()
