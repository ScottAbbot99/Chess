
import numpy as np
import pygame as p
import ChessEngine, ChessAI

# initialize pygame
p.init()

# definitions
HEIGHT, WIDTH = 512, 512 # board image size
NUM_ROWS = NUM_COLS = 8 # board is 8x8
SQUARE_SIZE = WIDTH / NUM_ROWS # size of each square is 512/8 x 512/8
MAX_FPS = 15 
IMAGES = {} # dictionary to store images



def load_images():
    pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load(f"images/{piece}.png"), (SQUARE_SIZE, SQUARE_SIZE))


def draw_board(screen):
    colors = [p.Color("white"), p.Color("light blue")]

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


def main():
    screen = p.display.set_mode((WIDTH, HEIGHT)) # initialize pygame window
    clock = p.time.Clock()
    screen.fill(p.Color("gray"))
    game_state = ChessEngine.GameState() # get the game state
    load_images()

    running = True 

    click = dict.fromkeys(["first_square", "first_piece"]) # dictionary to store square and piece clicked

    while running:
        for event in p.event.get():


            # stop running if the user closes the window
            if event.type == p.QUIT:
                running = False


            # mouse click
            elif event.type == p.MOUSEBUTTONDOWN:

                # no moves can be made if it's stalemate or checkmate
                if game_state.stalemate or game_state.checkmate:
                    continue

                mouse_x, mouse_y = p.mouse.get_pos()

                # get the square where the mouse is located
                row = int(mouse_y // SQUARE_SIZE)
                col = int(mouse_x // SQUARE_SIZE)

                cur_piece = game_state.board[row][col] # get the current piece selected

                turn = 'w' if game_state.white_turn else 'b'

                # First selection
                # if the click dictionary has values of "None", then store these new values
                if click["first_square"] == None:
                    # first ensure the first selection is actually a piece
                    if cur_piece != "--":

                        # ensure proper color
                        if cur_piece[0] == turn:
                            click["first_square"] = (row, col)
                            click["first_piece"] = cur_piece
                    
                # Second selection or Reselection of first piece
                # otherwise a piece has already been selected and should be moved
                else:

                    # generate all possible moves
                    all_moves = game_state.get_all_moves()

                    # ensure it's not stalemate or checkmate
                    if len(all_moves) == 0:
                        if game_state.checks():
                            game_state.checkmate = True
                            print("CHECKMATE")
                        else:
                            game_state.stalemate = True
                            print("STALEMATE")
                        
                        continue
                    
                    # ensure the move is valid and store if it is
                    move_valid = False

                    attempted_move = ChessEngine.Move(click["first_piece"], click["first_square"][0], click["first_square"][1], row, col)

                    # if the move is legal, apply the move
                    for move in all_moves:
                        if attempted_move == move:
                            
                            # note that the move is valid
                            move_valid = True
                            # apply the move
                            game_state.apply_move(move)

                            # reset click dictionaryu
                            click["first_square"] = None
                            click["first_piece"] = None


                    # if the move isn't valid, then set the current selection as the second selection or set to none
                    if not move_valid:
                        # new piece being selected
                        if cur_piece != "--":
                            click["first_square"] = (row, col)
                            click["first_piece"] = cur_piece
                        # empty square selected so no piece is selected
                        else:
                            click["first_square"] = None
                            click["first_piece"] = None   
                    
            # key handlers
            elif event.type == p.KEYDOWN:
                if event.key == p.K_u: # undo when "u" is pressed
                    game_state.undo_move()
                

        
        draw_board(screen) # draw the board
        draw_pieces(screen, game_state.board) # draw the pieces on the board
        clock.tick(MAX_FPS)

        p.display.flip()










if __name__ == "__main__":
    main()