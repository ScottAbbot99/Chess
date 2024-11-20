import numpy as np
import ChessEngine
from ChessAI import player_move, random_move, greedy_algorithm, minimax
import pygame as p
from constants import HEIGHT, WIDTH, NUM_ROWS, NUM_COLS, SQUARE_SIZE, MAX_FPS, IMAGES, MOVE_LOG_HEIGHT, MOVE_LOG_WIDTH
import time
import copy

# initialize pygame
p.init()

# CREDIT "Eddie Sharick (Eddie)" ON YOUTUBE FOR THE PYGAME CODE

global colors
colors = [p.Color("white"), p.Color("light blue")]

def load_images():
    pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load(f"images/{piece}.png"), (SQUARE_SIZE, SQUARE_SIZE))


def draw_board(screen):
    # white squares are odd when summing row number + column number
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            color = colors[(row + col) % 2]
            p.draw.rect(screen, color, p.Rect(col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def draw_pieces(screen, board):
    # loop over every square and get the piece at that square
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            piece = board[row][col]

            # draw piece if there is one
            if piece != "--":
                screen.blit(IMAGES[piece], p.Rect(col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_move_log(screen, game_state, font):
    move_log_rectangle = p.Rect(WIDTH, 0, MOVE_LOG_WIDTH, MOVE_LOG_HEIGHT)
    p.draw.rect(screen, p.Color("gray"), move_log_rectangle)
    move_log = game_state.move_log
    move_texts = move_log
    initial_padding = 5
    row_padding = {0:initial_padding, 1:initial_padding+80} # white moves on the left and black moves on the right
    height_padding = initial_padding
    for i in range(len(move_texts)): 
        text = move_texts[i][0].chess_notation()
        text_object = font.render(text, True, p.Color('black'))
        text_location = move_log_rectangle.move(row_padding[i%2], height_padding)
        screen.blit(text_object, text_location)

        # increment height every other turn
        if i%2 == 1:
            height_padding += text_object.get_height() + 5


def animate_move(move, screen, clock, game_state):
    dr = move.to_row - move.from_row
    dc = move.to_col - move.from_col
    frames_per_square = 10 # frames to move one square
    frame_count = (abs(dr) + abs(dc)) * frames_per_square

    for frame in range(frame_count + 1):
        # parametrize
        t = frame / frame_count
        cur_row, cur_col = (move.from_row + dr * t, move.from_col + dc * t)

        draw_board(screen)
        draw_pieces(screen, game_state.board)

        # erase the piece moved from its ending square
        color = colors[(move.to_row + move.to_col) % 2]
        end_square = p.Rect(move.to_col * SQUARE_SIZE, move.to_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        p.draw.rect(screen, color, end_square)

        # draw captured piece onto rectangle
        if game_state.captured_piece != "--":
            if move.en_passant:
                en_passant_row = 3 if move.piece[0] == 'w' else 5
                end_square = p.Rect(move.to_col * SQUARE_SIZE, en_passant_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            screen.blit(IMAGES[game_state.captured_piece], end_square)
        
        # draw moving pieces
        screen.blit(IMAGES[move.piece], p.Rect(cur_col * SQUARE_SIZE, cur_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
        p.display.flip()
        clock.tick(60)


def main():
    screen = p.display.set_mode((WIDTH + MOVE_LOG_WIDTH, HEIGHT)) # initialize pygame window
    clock = p.time.Clock()
    screen.fill(p.Color("gray"))
    game_state = ChessEngine.GameState() # get the game state
    move_log_font = p.font.SysFont("Arial", 12, False, False)
    load_images()

    running = True 

    click = dict.fromkeys(["first_square", "first_piece"]) # dictionary to store square and piece clicked

    players = {True:"person", False:"person"} # True represents the white player and False represents the black player

    while running:
        
        
        for event in p.event.get():


            # stop running if the user closes the window
            if event.type == p.QUIT:
                running = False

            # key handlers
            elif event.type == p.KEYDOWN:
                if event.key == p.K_u: # undo when "u" is pressed
                    game_state.undo_move()

            # not a key handler and not exiting (so making a move)
            else:

                # determine if it's a person's turn
                if players[game_state.white_turn] == "person":
                    move = player_move(event, game_state, SQUARE_SIZE, click, screen)
                    
                    # if a move is made
                    if move != None:
                        animate_move(move, screen, clock, game_state)

                # AI turn
                elif players[game_state.white_turn] == "AI":
                    #start_time = time.time()
                    move = random_move(game_state.get_all_moves())
                    game_state.apply_move(move)
                    
                    animate_move(move, screen, clock, game_state)
                    #print("--- %s seconds ---", (time.time() - start_time))

                # greedy move
                elif players[game_state.white_turn] == "greedy_algorithm":

                    move = greedy_algorithm(game_state)
                    game_state.apply_move(move)

                    animate_move(move, screen, clock, game_state)
                
                # minimax
                elif players[game_state.white_turn] == "minimax":
                    game_state_copy = copy.deepcopy(game_state)
                    turn = 1 if game_state.white_turn else -1
                    move, value = minimax(game_state_copy, True, turn)
                    print(value)
                    game_state.apply_move(move)

                    animate_move(move, screen, clock, game_state)

        #start_time = time.time()
        draw_board(screen) # draw the board
        draw_pieces(screen, game_state.board) # draw the pieces on the board
        draw_move_log(screen, game_state, move_log_font) # draw move log


        clock.tick(MAX_FPS)

        p.display.flip()
        #print("--- %s seconds ---", (time.time() - start_time))



if __name__ == "__main__":
    main()