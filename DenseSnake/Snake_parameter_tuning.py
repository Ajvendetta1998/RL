
from DQL import DQL 
import pygame 
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from copy import deepcopy
import numpy as np 
import os 
import keras 
import sys 
from copy import deepcopy
import matplotlib.pyplot as plt
# Snake block size
block_size = 25


# Set display width and height
width = 500 
height = 500
c = np.array([0.4,0.5,0.3,0.1,0.6,0.3,0.2,0.1,0.4])



lens = []


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


def initNNmodel():
    # Define the input shape
    input_shape = (3, height//block_size, width//block_size)

    # Create a Sequential model
    model = Sequential()

    # Add a 2D convolutional layer with 32 filters, a kernel size of 3x3, and relu activation
    model.add(Conv2D(30, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))

    # Add a flatten layer to convert the 2D output to a 1D vector
    model.add(Flatten())
    model.add(Dense(512 , activation = 'ReLU'))
    model.add(Dense(256 , activation = 'ReLU'))
    # Add the output layer with one unit and sigmoid activation
    model.add(Dense(len(actions), activation='linear'))

    # Compile the model with binary cross-entropy loss and adam optimizer
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # Print the model summary
    model.summary()
    return(model)

def state(snake_list,apple):
    layer_head = np.zeros((height//block_size,width//block_size))
    layer_tail = np.zeros((height//block_size,width//block_size))
    layer_apple = np.zeros((height//block_size,width//block_size))
    layer_head[snake_list[-1][0]//block_size-1, snake_list[-1][1]//block_size-1] =1 
    layer_apple[apple[0]//block_size-1, apple[1]//block_size-1] =1 
    for s in snake_list[:-1]:
        layer_tail[s[0]//block_size-1, s[1]//block_size-1] =1 
    input = np.zeros( (3,height//block_size, width//block_size))
    input[0,:,:] = layer_head
    input[1,:,:] = layer_tail
    input[2,:,:] = layer_apple

    return(input)

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
def reward(action, snake_list,episode_length,c):
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
    penalty_wall = -1 if not (inBounds(u,v)) else 0
    #penalize the agent for getting closer to danger
    penalty_danger = danger_distance(action,snake_list)
    #print(penalty_danger)
    compacity_value = 1/compacity(snake_list)
    #accessible points 
    accessible_points_proportion = find_accessible_points(snake_list)-1
    episode_length_penalty = -episode_length/(width*height//block_size**2)
    penalties = np.array([accessible_points_proportion,penalty_distance,penalty_touch_self,penalty_distance*gass_reward,reward_eat,penalty_wall,penalty_danger,compacity_value,episode_length_penalty])
  
    penalty_names  = ['accessible_points_proportion','penalty_distance','penalty_touch_self','penalty_distance*gass_reward','reward_eat','penalty_wall','penalty_danger','compacity','episode_len_penalty']


    total_reward = penalties@c/c.sum() 

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

#initialize NN
def load_model():
    filename = str(width)+" " + str(height)+" CNNDeepQ.h5"
    if(os.path.exists("./DenseSnake/"+filename)):
        print("model already exists ")
        model = keras.models.load_model("./DenseSnake/"+filename)

    else: 

        model = initNNmodel()
    return model 

# Initialize pygame

food_x, food_y = generate_food([])   

def main(length,c,dql):
    episode_reward = 0
    # Initial snake position and food
    snake_x = (width//block_size)//2     * block_size   
    snake_y = (height//block_size)//2     * block_size   

    snake_list = [[snake_x,snake_y]]
    
    global food_x,food_y
    #food_x, food_y = generate_food([])   
    episode_length =0 
    # Initial snake direction and length

    snake_length =length
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


        St1 = state(snake_list,[food_x,food_y]) 
        a= dql.get_action(St1,a,snake_list,block_size,width,height)
        r = reward(a,snake_list,episode_length-check,c)

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

        St2 = state(snake_list,[food_x,food_y])
        #add to Buffer
        episode_reward= dql.add_memory(St1,a,r,St2,done,episode_reward)

        # Check if snake hits the food
        if snake_x == food_x and snake_y == food_y:
            lens.append(episode_length-check)
            food_x, food_y = generate_food(snake_list)
            snake_length += 1
            score+=1
            check = episode_length
        episode_length+=1
#
        if((episode_length-check)%((width*height//block_size**2)) ==0 ):
            done = True
        #if snake has hit something quit
        if(done):
            return [snake_length,episode_length,score,episode_reward]
        
max_m = 0
max_avg =0 
if(os.path.exists('tuning_c_avg.txt')):
    with open('tuning_c_avg.txt', 'r') as f:

        last_line = f.readlines()[-1]
        m,max_avg_str,cs = last_line.split(" ")
        c=  np.array( [float(num) for num in cs.split(",")])
        max_m = float(m)
        max_avg = float(max_avg_str)
best_c = deepcopy(c)
print(best_c)
print(max_avg)

#number of episodes
num_episodes =25
#maximum score reached
m =0 
#initial max length for the snake at birth
max_length = 1
#maximum allowed length for a snake 
max_max_length = width*height//block_size**2
exploration_rate = 0.9
min_exploration_rate = 0.05
decay_rate = (min_exploration_rate/exploration_rate)**(1.0/num_episodes)
print(decay_rate)
def run_new_model(c,num_episodes,thread):
    m=0
    avg = 0 
    global max_length,max_m,max_avg
    file_name = "./DenseSnake/"+",".join([str(k) for k in c])+str(width)+" " + str(height)+" DeepQCNN.h5"
    model = initNNmodel()
    dql = DQL(model,actions.values(),decay_rate = decay_rate)
    for i in range(num_episodes):
        print(thread," ",i,m,avg, max_m)
        #do a generation and see the outcome
        a= main(np.random.randint(1,max_length+1),c,dql)
        #update maximum score 
        m = max(a[2],m)
        avg = (1.0*i*avg+a[2])/(i+1)
        #generate a new food position every 20 generations
        if(i%2 ==0):
            food_x, food_y = generate_food([])   
        #increase maximum birth length every 1000 generation 
       # if((i+1)%20==0):
      #      model.save(file_name)
#                max_length+=1
        dql.train(a[1]//4)

    del model
    del dql
    if(avg>max_avg):# or (m==max_m and avg>max_avg)):

        best_c = deepcopy(c)
        with open("tuning_c_avg.txt", "a") as f:
            f.write("\n")  # add a newline character before writing the list
            f.write(str(m)+" ")
            f.write(str(avg)+ " ")
            f.write(",".join([str(k) for k in best_c])) 
        max_m = m
        max_avg = avg
    
number_of_threads = 10

from threading import Thread 
run = 0 
borne = 0.5

while(borne>0.05):
    borne = 0.5*np.exp(-run/4)
    print(borne)
    threads = []
    for i in range(number_of_threads):
            # Create a thread for this combination of parameter values and start it
        c= np.array([np.random.uniform(np.clip(x-borne,0,1),np.clip(x+borne,0,1)) for x in best_c])
        t = Thread(target=run_new_model, args=(c, num_episodes,i))
        t.start()
        
        # Add the thread to the list
        threads.append(t)
    # Wait for all the threads to finish
    for t in threads:
        t.join()
    run+=1
os.system("shutdown /s /t 1")


