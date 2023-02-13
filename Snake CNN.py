import pygame
import sys
import random
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, LSTM
from keras.preprocessing.image import ImageDataGenerator
from copy import deepcopy
from keras.utils import plot_model
import imageio
from DQL import DQL 
import tensorflow as tf 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Set display width and height
width = 500
height = 500
def snapShot(frame):
    pygame.image.save(pygame.display.get_surface(), "frame"+str(frame)+".bmp")


def initNN():
    num_classes = 4


    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(5, (10,10), activation='relu', input_shape=(height,3* width, 3)),
        tf.keras.layers.MaxPooling2D((10,10)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    print(model.summary())
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    return model

actions = {"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}


def state():
    # Load the image
    img = np.concatenate((imageio.imread('frame0.bmp'), imageio.imread('frame1.bmp'), imageio.imread('frame2.bmp')), axis=1)
    # Preprocess the data
    # Convert the data type to float32 and scale the pixel values
    print(img.shape)
    img = img.astype('float32')
    img /= 255
    img = img.reshape(1,height,3*width,3)
    # Preprocess the data
    img = tf.keras.applications.resnet.preprocess_input(img)
    return img

def reward(action,snake_list):
    copy = deepcopy(snake_list)
    p = copy[-1]
    a = list(actions.values())
    (u,v)=(a[action][1]*block_size+p[0],a[action][0]*block_size+p[1])
    if(inBounds(u,v)):
        if([u,v] in snake_list):
            return(-np.inf)
        copy.append([u,v])
        del copy[0]
        global food_x,food_y
        c,acc,dis= compacity(copy),find_accessible_points(copy),((abs(u-food_x)+abs(v-food_y))/(height+width))
        return(1/c*acc*(1-dis))
    return (-np.inf)

model = initNN()

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

# Snake block size
block_size = 10

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

def display_score(score):
    # Display current score
    text = font.render("Score: " + str(score), True, black)
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
def main():


    # Initial snake position and food
    captured_frame =0
    snake_x = 150        
    snake_y = 150
    snake_list = [[snake_x,snake_y]]
    global food_x,food_y


    # Initial snake direction and length
    direction = "right"
    snake_length = 10
    

    prev_direction = "right"
    a=0

    acts = list(actions.values())
    # Game loop
    while True:
        useNN = os.path.isfile("frame0.bmp") and os.path.isfile("frame1.bmp") and os.path.isfile("frame2.bmp")
        if(useNN):
            S_t = state()
            a = dql.get_action(S_t)

            r = reward(a,snake_list)
        snake_x+=acts[a][1]*block_size
        snake_y+=acts[a][0]*block_size
        '''
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # Start a new game
                main()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Change direction based on user key press
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN and direction != "up":
                    direction = "down"
                    a=1
                if event.key == pygame.K_UP and direction != "down":
                    direction = "up"
                    a=0
                if event.key == pygame.K_LEFT and direction != "right":
                    direction = "left"
                    a=2
                if event.key == pygame.K_RIGHT and direction != "left":
                    direction = "right"
                    a=3

        print(reward(a,snake_list))
        # Move the snake
        if (snake_length ==1 and direction=="left") or (direction == "left" and not prev_direction == "right"):
            snake_x -= block_size
        if (snake_length ==1 and direction=="right") or (direction == "right" and not prev_direction == "left"):
            snake_x += block_size
        if (snake_length ==1 and direction=="up") or (direction == "up" and not prev_direction == "down"):
            snake_y -= block_size
        if (snake_length ==1 and direction=="down") or (direction == "down" and not prev_direction == "up"):
            snake_y += block_size
        prev_direction = direction
        '''
        # Check if snake hits the boundaries
        if snake_x >= width or snake_x < 0 or snake_y >= height or snake_y < 0:
            return snake_length
            #game_over()

        # Add new block of snake to the list
        snake_list.append([snake_x, snake_y])

        # Keep the length of snake same as snake_length
        if len(snake_list) > snake_length:
            del snake_list[0]


        # Check if snake hits itself
        for block in snake_list[:-1]:
            if block[0] == snake_x and block[1] == snake_y:
                return snake_length
        # Fill the screen with white color
        screen.fill(white)

        # Display food
        pygame.draw.rect(screen, red, [food_x, food_y, block_size, block_size])


        # Draw the snake
        draw_snake(snake_list)
        snapShot(captured_frame%3)
        captured_frame+=1
        if(useNN):
            S_t2 = state()
            dql.add_memory(S_t,a,r,S_t2,False)
            dql.train()

        # Display score
        display_score(snake_length-1)

        # Update the display
        pygame.display.update()
        # Check if snake hits the food
        if snake_x == food_x and snake_y == food_y:
            food_x, food_y = generate_food(snake_list)
            snake_length += 1

        # Set the FPS
        #clock.tick(fps)
#main()
num_episodes =1000
for i in range(num_episodes):
    main()
