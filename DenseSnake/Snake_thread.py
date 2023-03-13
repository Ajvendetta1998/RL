
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
import matplotlib.pyplot as plt

import threading
import time

# Snake block size
block_size = 10


# Set display width and height
width = 200 
height = 200

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

# Set clock to control FPS
clock = pygame.time.Clock()

# Font for displaying score
font = pygame.font.Font(None, 30)

# FPS
fps = 2

def game_over():
    # Display Game Over message
    text = font.render("Game Over!", True, red)
    screen.blit(text, [width/2 - text.get_width()/2, height/2 - text.get_height()/2])
    
    '''pygame.display.update()
    pygame.time.wait(3000)
    pygame.quit()
    sys.exit()'''

def display_score(score,gen,s,maxscore):
    # Display current score
    text = font.render("Gen: " + str(gen) + " Length : " + str(score)+ " Score: " + str(s) + " Max_score : "+str(maxscore), True, black)
    screen.blit(text, [0,0])

def draw_snake(snake_list):
    # Draw the snake
    for block in snake_list:
        pygame.draw.rect(screen, black, [block[0], block[1], block_size, block_size])

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

    # create a CNN
    model = Sequential()
    model.add(Conv2D(80, kernel_size = 3 , input_shape=(3,width//block_size, height//block_size), activation='ReLU'))
    model.add(Flatten())
    model.add(Dense(1024 , activation = 'ReLU'))
    model.add(Dense(512 , activation = 'ReLU'))
    model.add(Dense(256 , activation = 'ReLU'))
    model.add(Dense(128,activation = 'ReLU'))
    model.add(Dense(len(actions), activation='linear'))

    # Compile the model using mean squared error loss and the Adam optimizer
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

#returns a vector that has the state of snake and apple ( the NN input vector ) 
'''def state(snake_list,apple):
    s = np.array(snake_list)

    #create an input vector starting from the apple, head .. the rest of the tail 
    input = np.zeros(input_size)
    input[0],input[1] = apple[0]/width,apple[1]/height
    for u in range(len(snake_list)):
        if(2*u+2>=input_size):
            break
        input[2*u+2],input[2*u+3]= s[len(snake_list)-1-u][0]/width,s[len(snake_list)-1-u][1]/height
    input=input.reshape(1,input_size)
    return input'''
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
    penalty_distance = -2 if normalized_distance(u,v,food_x,food_y) > normalized_distance(p[0], p[1], food_x, food_y) else 1
    # penalize he agent for hitting a wall
    penalty_wall = -1 if not (inBounds(u,v)) else 0
    #penalize the agent for getting closer to danger
    penalty_danger = danger_distance(action,snake_list)
    #print(penalty_danger)
    compacity_value = 1/compacity(snake_list)
    #accessible points 
    accessible_points_proportion = find_accessible_points(snake_list)
    episode_length_penalty = -episode_length/(width*height//block_size**2+2)/5
    penalties = np.array([accessible_points_proportion,penalty_distance,penalty_touch_self,penalty_distance*gass_reward,reward_eat,penalty_wall,penalty_danger,compacity_value,episode_length_penalty])
    penalty_names  = ['accessible_points_proportion','penalty_distance','penalty_touch_self','penalty_distance*gass_reward','reward_eat','penalty_wall','penalty_danger','compacity','episode_len_penalty']
    c = np.array([5,1,3,0,1,3,0,0,0])

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

#initialize NN
filename = str(width)+" " + str(height)+" CNNDeepQ.h5"
if(os.path.exists("./"+filename)):
    print("model already exists ")
    model = keras.models.load_model("./"+filename)

else: 

    model = initNNmodel()

#initialize deepQ
dql = DQL(model,actions.values())
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

        # Draw the snake
        draw_snake(snake_list)

        St2 = state(snake_list,[food_x,food_y])
        #add to Buffer
        episode_reward= dql.add_memory(St1,a,r,St2,done,episode_reward)

        # Display score and other metrics
        display_score(snake_length-1,gen,score,maxlen)

        # Update the display
        pygame.display.update()

        # Check if snake hits the food
        if snake_x == food_x and snake_y == food_y:
            food_x, food_y = generate_food(snake_list)
            snake_length += 1
            score+=1
            check = episode_length
        episode_length+=1
#
       # if(episode_length%1000 ==0):
       #     dql.train(episode_length)
        if((episode_length-check)%((width*height//block_size**2)) ==0 ):
  
            #dql.exploration_rate/= dql.decay_rate
            done = True
        #if snake has hit something quit
        if(done):
            return [snake_length,episode_length,score,episode_reward]
        
        # Set the FPS
        #clock.tick(fps)
        #dql.train()

#number of episodes
num_episodes =10000000
#maximum score reached
m =0 
#initial max length for the snake at birth
max_length = 1
#maximum allowed length for a snake 
max_max_length = width*height//block_size**2
#max_length = width*height//block_size**2//20

def episode(i, m):
    global max_length
    a= main(i,np.random.randint(1,max_length+1),m)
    #update maximum score 
    m = max(a[2],m)
    #generate a new food position every 20 generations
    global food_x, food_y
    if(i%20 ==0):
        food_x, food_y = generate_food([])   
    #increase maximum birth length every 1000 generation 
    if((i+1)%20==0):
        model.save(str(width)+" " + str(height)+" DeepQ.h5")
        max_length+=1
        #dql.exploration_rate = 0.9
    print("episode reward : ", a[-1])
    #train the DQL 
    dql.train(a[1])
    dql.evaluate()

start = time.perf_counter()
#generation of episodes 
for i in range(1):
    episode(i, m)
    ##do a generation and see the outcome
    #a= main(i,np.random.randint(1,max_length+1),m)
    ##update maximum score 
    #m = max(a[2],m)
    ##generate a new food position every 20 generations
    #if(i%20 ==0):
    #    food_x, food_y = generate_food([])   
    ##increase maximum birth length every 1000 generation 
    #if((i+1)%20==0):
    #    model.save(str(width)+" " + str(height)+" DeepQ.h5")
    #    max_length+=1
    #    #dql.exploration_rate = 0.9
    #print("episode reward : ", a[-1])
    ##train the DQL 
    #dql.train(a[1])
    #dql.evaluate()
finish = time.perf_counter()
print(f'Normal finished in {round(finish-start, 2)} second(s)')


start = time.perf_counter()
threads = []
#generation of episodes 
for i in range(50):
    t = threading.Thread(target=episode, args=(i,m, ))
    t.start()
    threads.append(t)
for thread in threads:
    thread.join()
finish = time.perf_counter()
print(f'Thread finished in {round(finish-start, 2)} second(s)')