import pygame
import sys
import random

# Initialize pygame
pygame.init()   

# Set display width and height
width = 500
height = 500

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
    pygame.display.update()
    pygame.time.wait(3000)
    pygame.quit()
    sys.exit()

def display_score(score):
    # Display current score
    text = font.render("Score: " + str(score), True, white)
    screen.blit(text, [0,0])

def draw_snake(snake_list):
    # Draw the snake
    for block in snake_list:
        pygame.draw.rect(screen, black, [block[0], block[1], block_size, block_size])

def generate_food():
    # Generate food for the snake
    food_x = round(random.randrange(0, width - block_size) / 10.0) * 10.0
    food_y = round(random.randrange(0, height - block_size) / 10.0) * 10.0
    return food_x, food_y

def main():
    # Initial snake position and food
    snake_x = 150
    snake_y = 150
    food_x, food_y = generate_food()

    # Initial snake direction and length
    direction = "right"
    snake_length = 1
    snake_list = []
    eaten = False
    prev_direction = "right"
    # Game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # Start a new game
                main()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Change direction based on user key press
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != "down":
                    direction = "up"
                if event.key == pygame.K_DOWN and direction != "up":
                    direction = "down"
                if event.key == pygame.K_LEFT and direction != "right":
                    direction = "left"
                if event.key == pygame.K_RIGHT and direction != "left":
                    direction = "right"

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
        # Check if snake hits the boundaries
        if snake_x >= width or snake_x < 0 or snake_y >= height or snake_y < 0:
            game_over()

        # Add new block of snake to the list
        snake_list.append([snake_x, snake_y])

        # Keep the length of snake same as snake_length
        if len(snake_list) > snake_length:
            del snake_list[0]

        # Check if snake hits itself
        for block in snake_list[:-1]:
            if block[0] == snake_x and block[1] == snake_y:
                game_over()

        # Fill the screen with white color
        screen.fill(white)

        # Display food
        pygame.draw.rect(screen, red, [food_x, food_y, block_size, block_size])


        # Draw the snake
        draw_snake(snake_list)
        if(eaten):
            pygame.image.save(pygame.display.get_surface(), "screenshot.bmp")
        # Display score
        display_score(snake_length-1)

        # Update the display
        pygame.display.update()
        eaten=False
        # Check if snake hits the food
        if snake_x == food_x and snake_y == food_y:
            food_x, food_y = generate_food()
            eaten = True
            snake_length += 1

        # Set the FPS
        clock.tick(fps)
main()