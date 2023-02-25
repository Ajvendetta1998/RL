Snake Game using Deep Q-Learning (DQL)
    This script is an implementation of a snake game using Deep Q-Learning (DQL). The game is displayed using the Pygame library.

Dependencies
    DQL
    Pygame
    Keras
    Numpy
    Matplotlib
    
Usage
    Make sure all the dependencies are installed.
    Run the script.

Overview
    DQL.py contains the Deep Q-Learning algorithm implementation.
    The snake game is displayed using Pygame.
    The neural network used in this implementation is a Convolutional Neural Network (CNN).
    The CNN has 3 convolutional layers and 3 fully connected layers.
    The initNNmodel() function initializes the CNN with the specified layers.
    The state(snake_list,apple) function creates an input vector that represents the current state of the game.
    The game_over() function displays a "Game Over" message on the screen when the game ends.
    The display_score(score,gen,s,maxscore,episode_len) function displays the current score, generation, length, score, max score, and episode length on the screen.
    The draw_snake(snake_list) function draws the snake on the screen.
    The generate_food(snake_list) function generates a new food for the snake to eat.
    The actions dictionary contains the possible actions that the snake can take.
    The width and height variables specify the size of the display window.
    The fps variable specifies the number of frames per second.
    The block_size variable specifies the size of the snake blocks.
    The font variable specifies the font used to display the score.
    The clock variable is used to control the game's frame rate.
    The lens list keeps track of the length of each episode for computing the average episode length.