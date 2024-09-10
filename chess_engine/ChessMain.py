
import numpy as np
import pygame as p
import ChessEngine

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

    click = dict.fromkeys(["square", "piece"]) # dictionary to store square and piece clicked

    while running:
        for event in p.event.get():


            # stop running if the user closes the window
            if event.type == p.QUIT:
                running = False


            # mouse click
            elif event.type == p.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = p.mouse.get_pos()

                # get the square where the mouse is located
                row = int(mouse_y // SQUARE_SIZE)
                col = int(mouse_x // SQUARE_SIZE)

                cur_piece = game_state.board[row][col] # get the current piece selected

                # if the click dictionary has values of "None", then store these new values
                if (click["square"] == None):
                    # first ensure the first selection is actually a piece
                    if (cur_piece != "--"):

                        # ensure proper color
                        if game_state.white_turn and (cur_piece[0] == 'w'):
                            click["square"] = (row, col)
                            click["piece"] = cur_piece
                        
                        elif (not game_state.white_turn) and (cur_piece[0] == 'b'):
                            click["square"] = (row, col)
                            click["piece"] = cur_piece
                
                # otherwise a piece has already been selected and should be moved
                else:
                    
                    # ensure the move is valid
                    is_valid_move = make_valid_move(game_state, click["piece"], click["square"], tuple([row, col]))

                    if is_valid_move == True:    
                        game_state.board[row][col] = click["piece"] # move piece to new square
                        prev_row = click["square"][0]
                        prev_col = click["square"][1]
                        game_state.board[prev_row][prev_col] = '--' # set previous square to a blank square

                        # change turns
                        game_state.white_turn = not game_state.white_turn

                    # reset click dictionary
                    click["square"] = None
                    click["piece"] = None
                

        
        draw_board(screen) # draw the board
        draw_pieces(screen, game_state.board) # draw the pieces on the board
        clock.tick(MAX_FPS)

        p.display.flip()



def make_valid_move(game_state, selected_piece , from_square, to_square):
    """
    determine if selected_piece is able to move from from_square to to_square given game_state

    game_state: Class GameState()
    selected_piece: string
    from_square: tuple (row, col)
    to_square: tuple (row, col)
    """

    is_valid_move = True # initialize valid move to true

    # extract the piece type and color
    piece_color = selected_piece[0]
    piece_type = selected_piece[1]


    # if there is the same color piece on the to_square as piece_color, then the move is automatically invalid 
    # aside from castling!
    # this also prevents a piece from moving to the same square it's already at!
    to_piece_color = game_state.board[to_square[0]][to_square[1]][0]
    if (to_piece_color == piece_color) and (piece_type != 'K'):
        return False
    


    # determine if the move is valid based on the given piece and color
    match piece_type:


        case "P":
            is_valid_move = is_valid_pawn_move(game_state, piece_color, from_square, to_square)


        case "R":
            is_valid_move = is_valid_rook_move(game_state, from_square, to_square)


        case "N":
            is_valid_move = is_valid_knight_move(from_square, to_square)


        case "B":
            is_valid_move = is_valid_bishop_move(game_state, from_square, to_square)


        case "Q":
            is_valid_move = is_valid_queen_move(game_state, from_square, to_square)


        case "K":
            is_valid_move = is_valid_king_move(game_state, piece_color, from_square, to_square)
    


    #TODO: ensure king is not under attack / checkmate / stalemate

    return is_valid_move


def is_valid_pawn_move(game_state, piece_color, from_square, to_square):
    """
    Pawns can move forward 1 or 2 squares and capture diagonally 1 square
    En passant
    """

    from_row = from_square[0]
    from_col = from_square[1]
    to_row = to_square[0]
    to_col = to_square[1]

    col_diff = to_col - from_col
    row_diff = to_row - from_row

    # pawns can never move horizontally
    if row_diff == 0:
        return False
    

    # pawn can move one square forward, or one square diagonally
    elif np.abs(row_diff) == 1:

        # pawn can only move forward if there is an empty square
        if (col_diff == 0) and (game_state.board[to_row][to_col] == "--"):

            # lastly ensure pawn doesn't move backwards
            return no_backward_pawn(piece_color, row_diff)
            
        # capturing
        elif (np.abs(col_diff) == 1) and (game_state.board[to_row][to_col] != "--"):

            # lastly ensure pawn doesn't move backwards
            return no_backward_pawn(piece_color, row_diff)
        
        # Ensure the game state allows for en passant and that you're not capturing your own piece
        elif (game_state.en_passant[piece_color]["able"]) and (to_col == game_state.en_passant[piece_color]["column"]) and (game_state.board[from_row][to_col][0] != piece_color):
            if no_backward_pawn(piece_color, row_diff):
                game_state.board[from_row][to_col] = "--"
                set_en_passant(game_state, piece_color, False, None)

                return True
        
        else:
            return False

    
    # pawn can move two squares forward if on its home square
    elif np.abs(row_diff) == 2:

        # white pieces
        if (piece_color == "w") and (from_row == 6) and (col_diff == 0) and (game_state.board[to_row][to_col] == "--"):

            # set game_state to allow for en passant
            can_en_passant(game_state, 'w', to_row, to_col)

            return True
        
        # black pieces
        elif (piece_color == "b") and (from_row == 1) and (col_diff == 0) and (game_state.board[to_row][to_col] == "--"):

            # set game_state to allow for en passant
            can_en_passant(game_state, 'b', to_row, to_col)

            return True
    


def is_valid_rook_move(game_state, from_square, to_square):
    """
    A rook can either move across the row or column
    """

    # TODO: adjust castling if rook moves
        
    # split up from_square and to_square into rows and columns
    from_row = from_square[0]
    from_col = from_square[1]
    to_row = to_square[0]
    to_col = to_square[1]

    # along same row
    if from_row == to_row:

        # account for rook moving left or right
        if to_col < from_col:
            shift = -1 # move left
        else:
            shift = 1 # move right

        # ensure there is no piece along the path
        for col in range(from_col + shift , to_col, shift):
            cur_piece = game_state.board[from_row][col]
            if cur_piece != "--":
                return False
            
        return True


    # along same column
    elif from_col == to_col:

        # account for rook moving up or down
        if to_row < from_row:
            shift = -1 # move down 
        else:
            shift = 1 # move up
        
        # ensure there is no piece along the path
        for row in range(from_row + shift, to_row, shift):
            cur_piece = game_state.board[row][from_col]
            if cur_piece != "--":
                return False
        
        return True
    

    # not along same row or column
    else:
        return False


def is_valid_knight_move(from_square, to_square):
    from_row = from_square[0]
    from_col = from_square[1]
    to_row = to_square[0]
    to_col = to_square[1]

    # A knight moves 1 square along the row/column and 2 along the column/row
    row_diff = np.abs(from_row - to_row)
    col_diff = np.abs(from_col - to_col)
    if (row_diff == 1 and col_diff == 2) or (row_diff == 2 and col_diff == 1):
        return True
    else:
        return False


def is_valid_bishop_move(game_state, from_square, to_square):
    from_row = from_square[0]
    from_col = from_square[1]
    to_row = to_square[0]
    to_col = to_square[1]

    row_diff = np.abs(to_row - from_row)
    col_diff = np.abs(to_col - from_col)

    # A bishop moves diagonally
    if (row_diff != col_diff):
        return False
    

    else:

        # determine which direction bishop is moving along (row and col)
        row_shift = (to_row - from_row) / row_diff
        col_shift = (to_col - from_col) / col_diff
    
        # ensure there are no pieces in between
        i = 1 # iterable
        while i < row_diff:
            cur_row = int(from_row + row_shift*i)
            cur_col = int(from_col + col_shift*i)
            if game_state.board[cur_row][cur_col] != "--":
                return False
            i += 1
        
        return True


def is_valid_queen_move(game_state, from_square, to_square):
    # Queen is just a rookie combined with a bishop
    if (is_valid_rook_move(game_state, from_square, to_square) or is_valid_bishop_move(game_state, from_square, to_square)):
        return True


def is_valid_king_move(game_state, piece_color, from_square, to_square):
    from_row = from_square[0]
    from_col = from_square[1]
    to_row = to_square[0]
    to_col = to_square[1]

    # king can move in any direction 1 square
    row_diff = np.abs(to_row - from_row)
    col_diff = np.abs(to_col - from_col)

    # castling
    if (row_diff == 0) and (col_diff > 1) and can_castle(game_state, piece_color, from_row, from_col, to_row, to_col):
        # TODO: need to arrange pieces carefully
        pass

    elif (row_diff > 1 ) or (col_diff > 1):
        return False
    
    return True

    #TODO: castling



def can_en_passant(game_state, piece_color, to_row, to_col):
    """
    En Passant is only possible if there's an opposite colored pawn in the same row as the pawn moving 2 squares
    """
    left_piece, right_piece = get_left_right_piece(game_state, to_row, to_col)

    # distinction between white and black pawns
    if piece_color == 'w':
        if (left_piece == "bP") or (right_piece == "bP"):
            set_en_passant(game_state, 'b', True, to_col) 
    else:
        if (left_piece == "wP") or (right_piece == "wP"):
            set_en_passant(game_state, 'w', True, to_col)


def get_left_right_piece(game_state, to_row, to_col):
    """
    get pieces to the left and right of pawn that just advanced 2 squares
    """

    # initialize left and right piece
    left_piece = "--"
    right_piece = "--"

    # get piece to either side of pawn
    if to_col > 0:
        left_piece = game_state.board[to_row][to_col - 1]
    if to_col < 7:
        right_piece = game_state.board[to_row][to_col + 1]
    
    return left_piece, right_piece

def set_en_passant(game_state, piece_color, able, column):
    game_state.en_passant[piece_color]["able"] = able
    game_state.en_passant[piece_color]["column"] = column

def no_backward_pawn(piece_color, row_diff):
    """
    Ensures pawns don't move backwards
    """
    if (piece_color == 'b') and (row_diff > 0):
        return True
    elif (piece_color == 'w') and (row_diff < 0):
        return True
    else:
        return False


def can_castle(game_state, piece_color, from_row, from_col, to_row, to_col):
    """
    Determines if King can castle
    """
    # determine which direction we're castling to
    if to_col > from_col:
        shift = 1
    else:
        shift = -1

    # if a king has moved already then they cannot castle
    if game_state.king_rook_moved[piece_color]["king"]:
        return False
    
    # if a rook has moved then the king cannot castle on that side
    if game_state.king_rook_moved[piece_color][str(to_col)]:
        return False

    # king cannot castle out of or through check
    for i in range(4, 4 + 3*shift, shift): # check every square 2 away from the starting king square along the same row
        # determine if there is a threatening piece at each square 
        if attacking_piece(game_state, piece_color, from_row, i):
            print(i)
            print("attacking piece")
            return False

    # king cannot castle over pieces
    for i in range(4 + shift, to_col, shift):
        if game_state.board[from_row][i] != "--":
            return False
        

def attacking_piece(game_state, piece_color, row, col):
    """
    Determines if a piece is attacking a square (for castling / check / checkmate)
    """

    # Ensure there is no opposite colored rook or queen along the column
    for i in range(0, NUM_ROWS):
        # skip over king
        if i == row:
            continue

        cur_piece = get_piece(game_state, i, col)

        # determine if the current piece is a threat
        threat = piece_threatening(piece_color, cur_piece, ['R', 'Q'])
        if threat:
            return True
    
        
    # row
    for i in range(0, NUM_COLS):
        # skip over king
        if i == col:
            continue

        cur_piece = get_piece(game_state, row, i)

        # determine if the current piece is a threat
        threat = piece_threatening(piece_color, cur_piece, ['R', 'Q'])
        if threat:
            return True


    # Bishop / Queen / Pawn
    # determine which directions are worth exploring
    up_right = [True]
    up_left = [True]
    bottom_right = [True]
    bottom_left = [True]
    for i in range(1, NUM_ROWS):
        pawn = False

        # a pawn can threaten this square
        if i == 1:
            pawn = True

        # explore each direction

        if up_right[0] == True:
            up_right_square = [row - i, col + i]

            # determine if a piece is threatening and what piece
            output, cur_piece = explore_diag(game_state, piece_color, up_right, up_right_square[0], up_right_square[1], pawn)

            if output == None:
                continue

            # a piece is threatening
            if output == True:
                return True

            # piece not threatening so stop exploring that direction
            elif (not output) and (cur_piece != "--") and (cur_piece[0] != piece_color):
                up_right[0] = False

        
        if up_left[0] == True:
            up_left_square = [row - i, col - i]

            # determine if a piece is threatening and what piece
            output, cur_piece = explore_diag(game_state, piece_color, up_left, up_left_square[0], up_left_square[1], pawn)

            if output == None:
                continue

            # a piece is threatening
            if output == True:
                print("bishop")
                return True

            # piece not threatening so stop exploring that direction
            elif (not output) and (cur_piece != "--") and (cur_piece[0] != piece_color):
                up_left[0] = False
        
        if bottom_right[0] == True:
            bottom_right_square = [row + i, col + i]

            # determine if a piece is threatening and what piece
            output, cur_piece = explore_diag(game_state, piece_color, bottom_right, bottom_right_square[0], bottom_right_square[1], False)

            if output == None:
                continue

            # a piece is threatening
            if output == True:
                return True
            
            # piece not threatening so stop exploring that direction
            elif (not output) and (cur_piece != "--") and (cur_piece[0] != piece_color):
                bottom_right[0] = False

        if bottom_left[0] == True:
            bottom_left_square = [row + i, col - i]

            # determine if a piece is threatening and what piece
            output, cur_piece = explore_diag(game_state, piece_color, bottom_left, bottom_left_square[0], bottom_left_square[1], False)

            if output == None:
                continue

            # a piece is threatening
            if output == True:
                return True
            
            # piece not threatening so stop exploring that direction
            elif (not output) and (cur_piece != "--") and (cur_piece[0] != piece_color):
                bottom_left[0] = False
            

            

    # knight
    # get all locations where a knight could be threatening
    possible_knights = get_knight_squares(row, col)

    # consider each square and determine if there is a threatening knight
    for square in possible_knights:
        cur_piece = game_state.board[square[0]][square[1]]
        piece_threatening(piece_color, cur_piece, 'K')
        if piece_threatening:
            return True
                

    # otherwise there are no threatening pieces for the given square
    return False



def get_piece(game_state, row, col):
    """
    Outputs the piece at the given square
    """

    return game_state.board[row][col]


def explore_diag(game_state, piece_color, direction, row, col, pawn):
    """
    outputs True if square is threatened and updates direction if necessary
    "pawn" is a boolean representing if a pawn can threaten
    """

    # out of bounds so set direction to False
    if (row < 0) or (row > 7) or (col < 0) or (col > 7):
        direction[0] = False
        return None, None

    # otherwise get the piece at the current square
    cur_piece = get_piece(game_state, row, col)

    # bishop / queen /pawn
    if pawn:
        return piece_threatening(piece_color, cur_piece, ['B', 'Q', 'P']), cur_piece
    
    # bishop / queen
    else: 
        return piece_threatening(piece_color, cur_piece, ['B', 'Q']), cur_piece

def piece_threatening(piece_color, cur_piece, pieces_to_consider):
    """
    Considers whether cur_piece is a threat
    piece_color: color of our king piece
    pieces_to_consider: pieces that can be of threat on the given square
    """
    # same color
    if piece_color == cur_piece[0]:
        return False
    
    # in the potential threatening pieces
    elif cur_piece[1] in pieces_to_consider:
        return True 

    else:
        return False

def get_knight_squares(row, col):
    """
    outputs a list of tuples indicating potential knight squares
    """

    # All possible moves a knight can make
    potential_moves = [
        (row + 2, col + 1),
        (row + 2, col - 1),
        (row - 2, col + 1),
        (row - 2, col - 1),
        (row + 1, col + 2),
        (row + 1, col - 2),
        (row - 1, col + 2),
        (row - 1, col - 2)
    ]

    # filter out moves that are off the board
    valid_moves = [(r, c) for r, c in  potential_moves if 0 <= r < 8 and 0 <= c < 8]

    return valid_moves

if __name__ == "__main__":
    main()