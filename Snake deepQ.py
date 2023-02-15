import pygame
import random
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.preprocessing.image import ImageDataGenerator
from copy import deepcopy
from DQL import DQL 
import tensorflow as tf 


# Snake block size
block_size = 10

# Set display width and height
width = 100
height = 100

#dictionary of possible actions 
actions = {"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
#the snake + apple 
input_size = 2*width*height//block_size**2+2
#input_size = 4
def initNN():
    num_classes = 4
    model = Sequential()
    model.add(Dense(256, input_shape=(input_size,), activation='sigmoid'))
    model.add(Dense(128,activation = 'ReLU'))
    model.add(Flatten())
    model.add(Dense(len(actions), activation='linear'))

    # Compile the model using categorical crossentropy loss and the Adam optimizer
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


#returns a vector that has the state of snake and apple
def state(snake_list,apple):
    s = np.array(snake_list)

    #mini0 = int(min(s[:,0].min(),apple[0])-10)
    #mini1 = int(min(s[:,1].min(),apple[1])-10)
    #apple[0]-=mini0
    #apple[1] -= mini1
    #s[:,0] -= mini0
    #s[:,1] -= mini1
    input = np.zeros((width*height//block_size**2*2+2))
    input = np.zeros(input_size)
    input[0],input[1] = apple[0]/width,apple[1]/height
    for u in range(len(snake_list)):
        if(2*u+2>=input_size):
            break
        input[2*u+2],input[2*u+3]= s[len(snake_list)-1-u][0]/width,s[len(snake_list)-1-u][1]/height
    input=input.reshape(1,input_size)


    return input

def distance2(u,v,food_x,food_y):
    return np.sqrt((((u-food_x)/width)**2+((v-food_y)/height)**2)/2)
def distance(u,v,food_x,food_y):
    return (np.abs(u-food_x)+np.abs(v-food_y))/(height+width)

#reward function for each state and action
def reward(action, snake_list):
    copy = deepcopy(snake_list)
    p = copy[-1]
    a = list(actions.values())
    (u,v)=(a[action][1]*block_size+p[0],a[action][0]*block_size+p[1])
    if inBounds(u, v):
        if [u,v] in snake_list:
            return -1 # return a negative reward if the snake collides with itself
        copy.append([u,v])
        del copy[0]
        
        global food_x, food_y
        # reward the agent for getting closer to the food
        reward_distance = (1 - distance2(u,v,food_x,food_y)) 
        # reward the agent for eating the food
        reward_eat = 10 if u == food_x and v == food_y else 0
        # penalize the agent for moving away from the food
        penalty_distance = -1 if distance(u,v,food_x,food_y) > distance(p[0], p[1], food_x, food_y) else 0
        # penalize the agent for hitting a wall
        penalty_wall = -1 if (u == 0 or v == 0 or u == width-block_size or v == height-block_size) else 0
        # combine all rewards and penalties
        total_reward = reward_distance*2 + reward_eat + penalty_distance + penalty_wall
        return total_reward
    else:
        return -1 # return a negative reward if the snake goes out of bounds


#initialize NN
model = initNN()
#initialize deepQ
dql = DQL(model,actions.values())
# Initialize pygame
pygame.init()   



# Create display surface
screen = pygame.display.set_mode((width, height))

# Set title for the display window
pygame.display.set_caption("Snake Game")

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

# Set clock to control FPS
clock = pygame.time.Clock()



# Font for displaying score
font = pygame.font.Font(None, 30)

# FPS
fps = 15

def game_over():
    # Display Game Over message
    text = font.render("Game Over!", True, red)
    screen.blit(text, [width/2 - text.get_width()/2, height/2 - text.get_height()/2])
    
    '''pygame.display.update()
    pygame.time.wait(3000)
    pygame.quit()
    sys.exit()'''

def display_score(score,gen):
    # Display current score
    text = font.render("Gen: " + str(gen) + " Score: " + str(score), True, black)
    screen.blit(text, [0,0])

def draw_snake(snake_list):
    # Draw the snake
    for block in snake_list:
        pygame.draw.rect(screen, black, [block[0], block[1], block_size, block_size])

def generate_food(snake_list):
    # Generate food for the snake where there is no snake
    food_x, food_y = None, None
    
    while food_x is None or food_y is None or (food_x, food_y) in snake_list:
        food_x = round(random.randrange(0, width - block_size) / 10.0) * 10.0
        food_y = round(random.randrange(0, height - block_size) / 10.0) * 10.0
    return food_x, food_y

accessible_points = np.ones((height//block_size,width//block_size))

def inBounds(u,v):
    if(u>=0 and v>=0):
        if(u<width and v<height):
            return True
    return False
#if all cells are accessible 
def find_accessible_points(snake_list):

    global accessible_points 
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

#if snake is just a line
def compacity(snake_list):
    snake_list = np.array(snake_list)
    min_x = snake_list[:,0].min()
    min_y = snake_list[:,1].min()
    max_x = snake_list[:,0].max()
    max_y = snake_list[:,1].max()
    return((max_y-min_y+block_size)*(max_x-min_x+block_size)/(len(snake_list)*block_size**2))   
food_x, food_y = generate_food([])   

def main(gen,length):

    # Initial snake position and food
    snake_x = width//2        
    snake_y = height//2
    snake_list = [[snake_x,snake_y]]
    global food_x,food_y

    episode_length =0 
    # Initial snake direction and length
    direction = "right"
    snake_length =length
    

    prev_direction = "right"
    a=0
    done = False
    acts = list(actions.values())
    # Game loop
    while True:


        S_t = state(snake_list,[food_x,food_y])
        prev_a = a 
        b = dql.get_action(S_t)
        
        if(not(abs(a-prev_a)==1 and ((prev_a<2 and a<2) or (prev_a>1 and a>1)))):
            a =b
        
        r = reward(a,snake_list)
        snake_x+=acts[a][1]*block_size
        snake_y+=acts[a][0]*block_size
       
        # Check if snake hits the boundaries
        if snake_x >= width or snake_x < 0 or snake_y >= height or snake_y < 0:
            done = True
            #game_over()

        # Add new block of snake to the list
        snake_list.append([snake_x, snake_y])

        # Keep the length of snake same as snake_length
        if len(snake_list) > snake_length:
            del snake_list[0]


        # Check if snake hits itself
        for block in snake_list[:-1]:
            if block[0] == snake_x and block[1] == snake_y:
                done = True
        # Fill the screen with white color
        screen.fill(white)

        # Display food
        pygame.draw.rect(screen, red, [food_x, food_y, block_size, block_size])


        # Draw the snake
        draw_snake(snake_list)

        S_t2 = state(snake_list,[food_x,food_y])
        dql.add_memory(S_t,a,r,S_t2,done)
        dql.train()
        if(done):
            return [snake_length,episode_length]

        # Display score
        #display_score(snake_length-1,gen)

        # Update the display
        pygame.display.update()
        # Check if snake hits the food
        if snake_x == food_x and snake_y == food_y:
            food_x, food_y = generate_food(snake_list)
            snake_length += 1
        episode_length+=1
        # Set the FPS
        clock.tick(fps)
#main()
num_episodes =1000000
m =0 
max_length = 5 
for i in range(num_episodes):
    a= main(i,np.random.randint(1,max_length+1))
    m = max(a[0],m)
    if((i+1)%5==0):
        max_length+=1
    dql.evaluate()
    #dql.train()
    #print("Gen " + str(i) + " Score : " + str(a[0]) + " Episode length : "+str(a[1]) + " max length "+str(m)+ "Exploration rate "+ str(dql.exploration_rate))

