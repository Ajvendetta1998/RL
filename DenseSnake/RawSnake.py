
import pygame 
import random
from copy import deepcopy
import numpy as np 
import os 
import sys 
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import deque
from ppo import PPO, Actor, Critic


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

penalty_names  = ['accessible_points_proportion','penalty_distance','penalty_touch_self','penalty_distance*gass_reward','reward_eat','penalty_wall','penalty_danger','compacity','episode_len_penalty']
c= np.array([0.1863077196643149,0.8176132430371873,0.2359458810412274,0.5130391965064794,0.02492140605359113,0.1531607672522686,0.4327075132071214])

# Font for displaying score
font = pygame.font.Font(None, 30)

# FPS
fps = 6

def game_over():
    # Display Game Over message
    text = font.render("Game Over!", True, red)
    screen.blit(text, [width/2 - text.get_width()/2, height/2 - text.get_height()/2])
    
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def state(snake_list,apple):
    s = np.array(snake_list)
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
def reward(snake_list,episode_length,action):
    u,v = snake_list[-1][0],snake_list[-1][1]
    if [u,v] in snake_list[:-1] or not inBounds(u,v):
        #penalty_touch_self=-1 # return a negative reward if the snake collides with itself
        return -10
    else:
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
        #penalty_wall = -1 if not (inBounds(u,v)) else 0.2
        #penalize the agent for getting closer to danger
        penalty_danger = danger_distance(action,snake_list)
        #print(penalty_danger)
        compacity_value = 1.0/compacity(snake_list)
        #accessible points 
        accessible_points_proportion = find_accessible_points(snake_list)-1
        episode_length_penalty = -episode_length/(width*height//block_size**2)
        penalties = np.array([accessible_points_proportion,penalty_distance,penalty_distance*gass_reward,reward_eat,penalty_danger,compacity_value,episode_length_penalty])

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





# Initialize pygame

food_x, food_y = generate_food([])   
episode_length = 0 

ppo = PPO(input_size,len(actions.values()),0.0001,0.001,0.99,10,0.2,nn.ReLU,256)
def main():

    # Initial snake position and food
    snake_x = (width//block_size)//2     * block_size   
    snake_y = (height//block_size)//2     * block_size   

    snake_list = [[snake_x,snake_y]]
    
    global food_x,food_y

    acts = list(actions.values())
    check = 0
    done = False
    # Game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # Start a new game
                main()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


        s = state(snake_list,[food_x,food_y])
        a, _ = ppo.select_action(s)


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




        # Update the display
        pygame.display.update()

        # Check if snake hits the food
        if snake_x == food_x and snake_y == food_y:
            food_x, food_y = generate_food(snake_list)
            check = episode_length
            snake_length += 1
        episode_length+=1
        next_s = state(snake_list,[food_x,food_y])
        r = reward(next_s,episode_length-check,a)
        ppo.update(s,a,r,next_s,done)
        #if snake has hit something quit
        if(done):
            return
        
        # Set the FPS
        clock.tick(fps)



main()