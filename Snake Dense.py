import pygame
import random
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.preprocessing.image import ImageDataGenerator
from copy import deepcopy
from DQL import DQL 
import sys

# Snake block size
block_size = 10

# Set display width and height
width = 150
height = 150


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

def display_score(score,gen,s):
    # Display current score
    text = font.render("Gen: " + str(gen) + " Length : " + str(score)+ " Score: " + str(s), True, black)
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

#dictionary of possible actions 
actions = {"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}

#(x,y) apple + snake body (which has at most width*height parts) 
input_size = 2*width*height//block_size**2+2

#input_size = 10
def initNN():

    #create a dense NN 
    model = Sequential()
    model.add(Dense(256, input_shape=(input_size,), activation='tanh'))
    model.add(Dense(128,activation = 'sigmoid'))
    model.add(Dense(64,activation = 'ReLU'))
    model.add(Dense(len(actions), activation='linear'))


    # Compile the model using categorical crossentropy loss and the Adam optimizer
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

#returns a vector that has the state of snake and apple
def state(snake_list,apple):

    s = np.array(snake_list)
    #find the position of the bounding box of the game 
    mini0 = int(min(s[:,0].min(),apple[0]))
    mini1 = int(min(s[:,1].min(),apple[1]))
    maxi0 = int(max(s[:,0].max(),apple[0]))
    maxi1 = int(max(s[:,1].max(),apple[1]))

    #centralize the game
    apple[0]-=(mini0+maxi0-width)//2
    apple[1] -= (mini1+maxi1-height)//2
    s[:,0] -=(mini0+maxi0-width)//2
    s[:,1]-= (mini1+maxi1-height)//2

    #create an input vector starting from the apple, head .. the rest of the tail 
    input = np.zeros((width*height//block_size**2*2+2))
    input = np.zeros(input_size)
    input[0],input[1] = apple[0]/width,apple[1]/height
    for u in range(len(snake_list)):
        if(2*u+2>=input_size):
            break
        input[2*u+2],input[2*u+3]= s[len(snake_list)-1-u][0]/width,s[len(snake_list)-1-u][1]/height
    input=input.reshape(1,input_size)
    return input

def normalized_distance(u,v,food_x,food_y):
    return np.sqrt((((u-food_x)/width)**2+((v-food_y)/height)**2)/2)

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

#reward function for each state and action
def reward(action, snake_list):
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
    reward_distance = np.exp(-normalized_distance(u,v,food_x,food_y))
    # reward the agent for eating the food
    reward_eat = 1 if u == food_x and v == food_y else 0
    # penalize the agent for moving away from the food
    penalty_distance = -1 if normalized_distance(u,v,food_x,food_y) > normalized_distance(p[0], p[1], food_x, food_y) else 0
    # penalize the agent for hitting a wall
    penalty_wall = -1 if (u == 0 or v == 0 or u == width-block_size or v == height-block_size) else 0
    # combine all rewards and penalties
    c = np.ones(5)
    c[0] = 5
    c[1] = 10
    c[2]= 4
    total_reward =(c[0]* reward_distance + c[1]*reward_eat + c[2]*penalty_distance + c[3]*penalty_wall + c[4]*penalty_touch_self)/c.sum()

    #print(total_reward)
    return total_reward

#initialize NN
model = initNN()
#initialize deepQ
dql = DQL(model,actions.values())
# Initialize pygame

food_x, food_y = generate_food([])   

def main(gen,length):

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
    

    prev_direction = "right"
    #direction of the snake [0,1,2,3] each corresponding to one of up down left right
    a=0
    # = True if snake hits wall or itself
    done = False
    # number of collected fruit
    score = 0
    acts = list(actions.values())
    # Game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # Start a new game
                main()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


        St1 = state(snake_list,[food_x,food_y]) 
        a = dql.get_action(St1,a,snake_length)
        r = reward(a,snake_list)

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
        dql.add_memory(St1,a,r,St2,done)

        # Display score and other metrics
        #display_score(snake_length-1,gen,score)

        # Update the display
        pygame.display.update()

        # Check if snake hits the food
        if snake_x == food_x and snake_y == food_y:
            food_x, food_y = generate_food(snake_list)
            snake_length += 1
            score+=1
        episode_length+=1

        #if snake has hit something quit
        if(done):
            return [snake_length,episode_length,score]
        
        # Set the FPS
        #clock.tick(fps)


#number of episodes
num_episodes =10000000
#maximum score reached
m =0 
#initial max length for the snake at birth
max_length = 10
#maximum allowed length for a snake 
max_max_length = width*height//block_size**2



#generation des generations
for i in range(num_episodes):
    #do a generation and see the outcome
    a= main(i,np.random.randint(1,max_length))
    #update maximum score 
    m = max(a[2],m)
    #generate a new food position every 20 generations
    if(i%20 ==0):
        food_x, food_y = generate_food([])   
    #increase maximum birth length every 1000 generation 
    if((i+1)%1000==0):
        model.save(str(width)+" " + str(height)+" DeepQ.h5")
        max_length+=1

    #train the DQL 
    dql.train()
    #evaluate the DQL 
    dql.evaluate()

    #print("Gen " + str(i) + " Score : " + str(a[2]) + " Episode length : "+str(a[1]) + " max score "+str(m))

