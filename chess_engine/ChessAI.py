import random
import pygame as p
import ChessEngine
from constants import HEIGHT, WIDTH, NUM_ROWS, NUM_COLS, SQUARE_SIZE, MAX_FPS, IMAGES
import copy
import torch
from eval_model import preprocess_fen

def highlight_squares(screen, game_state, moves, square_selected):
    if square_selected != ():
        row, col = square_selected

        # ensure the piece being selected can actually be moved
        if game_state.board[row][col][0] == ('w' if game_state.white_turn else 'b'):
            
            s = p.Surface((SQUARE_SIZE, SQUARE_SIZE))
            s.set_alpha(255) # opaqueness

            # highlight the selected square
            s.fill(p.Color('yellow'))
            screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))

            # highlight moves
            s.fill(p.Color('green'))
            for move in moves:
                if (move.from_row == row) and (move.from_col == col):
                    screen.blit(s, (move.to_row * SQUARE_SIZE, move.to_col * SQUARE_SIZE))


def player_move(event, game_state, square_size, click, screen):
    """
    Player controlled moves
    
    click: dictionary to store player clicks
    """

    # mouse click
    if event.type == p.MOUSEBUTTONDOWN:

        # no moves can be made if it's stalemate or checkmate
        if game_state.stalemate or game_state.checkmate:
            return

        mouse_x, mouse_y = p.mouse.get_pos()

        # get the square where the mouse is located
        row = int(mouse_y // square_size)
        col = int(mouse_x // square_size)
        
        # if the user clicked the move log
        if col >= 8:
            click["first_square"] = None
            click["first_piece"] = None
            return

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

            all_moves = game_state.get_all_moves()

            # ensure it's not stalemate or checkmate
            if len(all_moves) == 0:
                if game_state.checks():
                    game_state.checkmate = True
                    print("CHECKMATE")
                else:
                    game_state.stalemate = True
                    print("STALEMATE")
                
                return
            
            # ensure the move is valid and store if it is
            move_valid = False

            attempted_move = ChessEngine.Move(click["first_piece"], click["first_square"][0], click["first_square"][1], row, col)

            # if the move is legal, apply the move
            for move in all_moves:
                if move == attempted_move:
                    
                    # note that the move is valid
                    move_valid = True
                    # apply the move
                    game_state.apply_move(move)

                    # reset click dictionary
                    click["first_square"] = None
                    click["first_piece"] = None

                    return move


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




def random_move(moves):
    """
    Selects a random move
    """

    random_move = moves[random.randint(0, len(moves) - 1)]

    return random_move




def greedy_algorithm(game_state):
    """
    Greedy approach
    depth: turns to look ahead
    """

    # the valuation flips depending on who's turn it is
    turn = 1 if game_state.white_turn else -1

    valid_moves = game_state.get_all_moves()

    best_value = float("-inf") # initialize the state value to infinity
    best_move = None

    # loop through each move and evaluate it
    for move in valid_moves:
        
        # create a copy to not affect the original
        game_state_copy = copy.deepcopy(game_state)

        # apply the given move
        game_state_copy.apply_move(move)

        
        # evaluate the current state
        value = evaluate_state(game_state_copy.board, turn, game_state_copy.checkmate, game_state_copy.stalemate)
        # if this new move generates a better state than previous
        if value > best_value:
            # update best_value and best_move
            best_value = value
            best_move = move


    # return the best move
    return best_move

def evaluate_state(game_state, turn_factor, checkmate = False, stalemate = False):
    """
    Evaluates a given board state
    turn_factor: 1 if it's white's turn and -1 if it's black's turn
    checkmate: True if it's checkmate
    stalemate: True if it's stalemate
    """
    
    state = game_state.board

    # if it's checkmate or stalemate, then return those values
    checkmate_value = 10000
    stalemate_value = 0
    if checkmate:
        return checkmate_value
    if stalemate:
        return stalemate_value

    # piece values
    piece_values = {'P':1, 'N':3, 'B':3.3, 'R':5, 'Q':9, 'K':0}

    # white and black board values
    board_values = {'w':0, 'b':0}

    # store how many threats are still on the board
    threats = {'w':0, 'b':0}

    centered_squares = [2, 3, 4, 5]

    # loop through each square
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):

            cur_square = state[row][col]

            if cur_square == "--":
                continue

            # incentive centered squares
            if (row in centered_squares) and (col in centered_squares):
                coef = 1.2
            else:
                coef = 1

            # add the square value
            board_values[cur_square[0]] += coef*piece_values[cur_square[1]]

            # store a count of all non-pawn pieces
            if cur_square[1] != 'P':
                threats[cur_square[0]] += 1
                # encourage development
                development(row, col, cur_square, board_values)


            # more value for pawns up the board
            if cur_square[1] == 'P':
                board_values[cur_square[0]] += pawn_advancement(row, cur_square[0])


    # value king safety and checks
    king_safety(game_state, threats, board_values)
    

    return (board_values['w'] - board_values['b']) * turn_factor


def minimax(game_state, maximizing_player, turn, depth = 4, alpha = float('-inf'), beta = float('inf'), counter = {"iterations":0}):
    """
    Minimax approach
    maximizing_player: True if we're trying to maximize the move and False if we're trying to minimize it
    turn: keeps track of whose turn it is
    depth: turns to look ahead
    alpha, beta: alpha-beta pruning values
    """
    # deubgging purposes
    counter["iterations"] += 1

    # stop looking forward once the max depth is reached or the game ends, and evaluate the state
    if (depth == 0) or (game_state.checkmate) or (game_state.stalemate):
        return None, evaluate_state(game_state, turn, game_state.checkmate, game_state.stalemate)

    valid_moves = game_state.get_all_moves()
    #random.shuffle(valid_moves)

    # maximizing player's turn
    if maximizing_player:
        max_value = float("-inf") # initialize the state value to infinity
        best_move = None # there no optimal move yet

        # loop through each move and evaluate it
        for move in valid_moves:
            
            # create a copy to not affect the original
            #game_state_copy = copy.deepcopy(game_state)

            # apply the given move
            game_state.apply_move(move)

            # apply minimax algorithm to this new state
            _, value = minimax(game_state, False, turn, depth - 1, alpha, beta, counter)


            game_state.undo_move()
            
            # if this new move generates a better state than previous
            if value > max_value:
                # update best_value and best_move
                max_value = value
                best_move = move


            # pruning
            alpha = max(alpha, value)
            if beta <= alpha:
                break

        
        # debugging purposes
        """
        num_pieces = game_state.count_pieces()
        if num_pieces > 32:
            print(num_pieces, "counter: ", counter["iterations"])
        """

        return best_move, max_value

    # minimizing player's turn
    else:
        min_value = float("inf")
        best_move = None

        for move in valid_moves:

            # create a copy to not affect the original
            #game_state_copy = copy.deepcopy(game_state)

            # apply the given move
            game_state.apply_move(move)

            # apply minimax algorithm to this new state
            _, value = minimax(game_state, True, turn, depth - 1, alpha, beta, counter)

            
            game_state.undo_move()

            # if this new move generates a better state than previous
            if value < min_value:
                # update best_value and best_move
                min_value = value
                best_move = move


            # pruning
            beta = min(beta, value)
            if beta <= alpha:
                break
        
        """
        num_pieces = game_state.count_pieces()
        if num_pieces > 32:
            print(num_pieces, "counter: ", counter["iterations"])
        """
        
        return best_move, min_value



def pawn_advancement(row, color):
    """
    Associates value with pawns further up the board
    """

    values = [0, 0, .1, .15, .2, .4, .75]

    # determine how far up the board the pawn is
    adjusted_index = 7-row if color == 'w' else row

    # return the value of the pawn advancement
    return values[adjusted_index]


def king_safety(game_state, threats, board_values):
    """
    Evaluates how safe each king is
    threats: count of non-pawn pieces
    """

    # get the location of each king
    king_locations = game_state.king_locations
    white_king = king_locations['white']
    black_king = king_locations['black']

    
    # white
    if threats['b'] > 3:
        # encourage castling
        if (white_king[1] == 2) or (white_king[1] == 6):
            board_values['w'] += .3
    
    if threats['w'] > 3:
        # encourage castling
        if (black_king[1] == 2) or (black_king[1] == 6):
            board_values['b'] += .3

    
    # slightly encourage checks
    if game_state.checking != None:
        # black is checking white
        if game_state.white_turn:
            board_values['b'] += .3
        # white is checking black
        else:
            board_values['w'] += .3



def development(row, col, cur_square, board_values):
    """
    Calculate value from development
    """

    piece_color = cur_square[0]

    # ensure the piece isn't on it's home square and is developed towards the center of the board



def board_to_fen(game_state):
    """
    Converts a move to fen format
    """

    board = game_state.board

    pieces = []

    # loop over the board
    for row in range(NUM_ROWS):

        num_empty_squares = 0

        # get each piece and add to pieces list in fen format
        for col in range(NUM_COLS):
            piece = board[row][col]

            if piece != "--":

                # append number of previous empty squares if there are any
                if num_empty_squares > 0:
                    pieces.append(str(num_empty_squares))
                    
                    # reset num_empty_squares
                    num_empty_squares = 0

                # append lower case piece is it's black and upper case if white
                pieces.append(piece[1].lower() if piece[0] == 'b' else piece[1])
            
            # empty squares are stored as number representing the number of continuous empty squares
            else:
                num_empty_squares += 1
        
        # a row that ends in an empty square needs to be accounted for
        if num_empty_squares > 0:
            pieces.append(str(num_empty_squares))

        # append row separator unless it's the last
        if row != 7:
            pieces.append("/")

    
    # convert pieces portion of fen to string
    piece_string = "".join(pieces)

    # turn string
    if game_state.white_turn:
        turn_string = "w"
    else:
        turn_string = "b"

    # castling list
    castling_list = []
    if game_state.castling_moves['w']['H']:
        castling_list.append("K")
    if game_state.castling_moves['w']['A']:
        castling_list.append("Q")
    if game_state.castling_moves['b']['H']:
        castling_list.append("k")
    if game_state.castling_moves['b']['A']:
        castling_list.append("q")

    # castling string
    if len(castling_list) > 0:
        castling_string = "".join(castling_list)
    else:
        castling_string = "-"
    
    # en passant string
    if game_state.en_passant[turn_string]["able"]:
        en_passant_string = str(game_state.en_passant[turn_string]["column"])
    else:
        en_passant_string = "-"
    

    fen = piece_string + " " + turn_string + " " + castling_string + " " + en_passant_string

    return fen


def evaluate_state_stockfish(game_state, model):
    """
    Evaluates a given state based on our trained model based on stockfish evaluations
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # obtain FEN from board state
    fen = board_to_fen(game_state)

    # preprocess values
    board, turn, castling_rights, en_passant = preprocess_fen(fen)

    # Convert board to tensor
    board = torch.tensor(board, dtype=torch.float32).to(device)

    # Combine scalar features into a single tensor
    scalar_features = torch.tensor(
        [turn, castling_rights[0], castling_rights[1], castling_rights[2], castling_rights[3], en_passant], 
        dtype=torch.float32
    ).to(device)

    # shape is (1, 64) and (1, 6) as opposed to (64,) and (6,) above
    board = board.unsqueeze(0)
    scalar_features = scalar_features.unsqueeze(0)
    # predict evaluation
    with torch.no_grad():
        prediction = model(board, scalar_features)

    evaluation = prediction.item() # get scalar value from tensor

    return evaluation


def minimax_stockfish(game_state, maximizing_player, turn, model, depth = 4, alpha = float('-inf'), beta = float('inf'), counter = {"iterations":0}):
    """
    Minimax approach
    maximizing_player: True if we're trying to maximize the move and False if we're trying to minimize it
    turn: keeps track of whose turn it is
    depth: turns to look ahead
    alpha, beta: alpha-beta pruning values
    """
    # deubgging purposes
    counter["iterations"] += 1

    # stop looking forward once the max depth is reached or the game ends, and evaluate the state
    if (depth == 0) or (game_state.checkmate) or (game_state.stalemate):
        return None, evaluate_state_stockfish(game_state, model)

    valid_moves = game_state.get_all_moves()
    #random.shuffle(valid_moves)

    # maximizing player's turn
    if maximizing_player:
        max_value = float("-inf") # initialize the state value to infinity
        best_move = None # there no optimal move yet

        # loop through each move and evaluate it
        for move in valid_moves:
            
            # create a copy to not affect the original
            #game_state_copy = copy.deepcopy(game_state)

            # apply the given move
            game_state.apply_move(move)

            # apply minimax algorithm to this new state
            _, value = minimax_stockfish(game_state, False, turn, model, depth - 1, alpha, beta, counter)


            game_state.undo_move()
            
            # if this new move generates a better state than previous
            if value > max_value:
                # update best_value and best_move
                max_value = value
                best_move = move


            # pruning
            alpha = max(alpha, value)
            if beta <= alpha:
                break

        
        # debugging purposes
        """
        num_pieces = game_state.count_pieces()
        if num_pieces > 32:
            print(num_pieces, "counter: ", counter["iterations"])
        """

        return best_move, max_value

    # minimizing player's turn
    else:
        min_value = float("inf")
        best_move = None

        for move in valid_moves:

            # create a copy to not affect the original
            #game_state_copy = copy.deepcopy(game_state)

            # apply the given move
            game_state.apply_move(move)

            # apply minimax algorithm to this new state
            _, value = minimax_stockfish(game_state, True, turn, model, depth - 1, alpha, beta, counter)

            
            game_state.undo_move()

            # if this new move generates a better state than previous
            if value < min_value:
                # update best_value and best_move
                min_value = value
                best_move = move


            # pruning
            beta = min(beta, value)
            if beta <= alpha:
                break
        
        """
        num_pieces = game_state.count_pieces()
        if num_pieces > 32:
            print(num_pieces, "counter: ", counter["iterations"])
        """
        
        return best_move, min_value




def minimax_combined(game_state, maximizing_player, turn, model, depth = 4, alpha = float('-inf'), beta = float('inf'), counter = {"iterations":0}):
    """
    Minimax approach
    maximizing_player: True if we're trying to maximize the move and False if we're trying to minimize it
    turn: keeps track of whose turn it is
    depth: turns to look ahead
    alpha, beta: alpha-beta pruning values
    """
    # deubgging purposes
    counter["iterations"] += 1

    # stop looking forward once the max depth is reached or the game ends, and evaluate the state
    if (depth == 0) or (game_state.checkmate) or (game_state.stalemate):
        return None, 0.05*evaluate_state_stockfish(game_state, model) + 0.95*evaluate_state(game_state, turn, game_state.checkmate, game_state.stalemate)

    valid_moves = game_state.get_all_moves()
    #random.shuffle(valid_moves)

    # maximizing player's turn
    if maximizing_player:
        max_value = float("-inf") # initialize the state value to infinity
        best_move = None # there no optimal move yet

        # loop through each move and evaluate it
        for move in valid_moves:
            
            # create a copy to not affect the original
            #game_state_copy = copy.deepcopy(game_state)

            # apply the given move
            game_state.apply_move(move)

            # apply minimax algorithm to this new state
            _, value = minimax_stockfish(game_state, False, turn, model, depth - 1, alpha, beta, counter)


            game_state.undo_move()
            
            # if this new move generates a better state than previous
            if value > max_value:
                # update best_value and best_move
                max_value = value
                best_move = move


            # pruning
            alpha = max(alpha, value)
            if beta <= alpha:
                break

        
        # debugging purposes
        """
        num_pieces = game_state.count_pieces()
        if num_pieces > 32:
            print(num_pieces, "counter: ", counter["iterations"])
        """

        return best_move, max_value

    # minimizing player's turn
    else:
        min_value = float("inf")
        best_move = None

        for move in valid_moves:

            # create a copy to not affect the original
            #game_state_copy = copy.deepcopy(game_state)

            # apply the given move
            game_state.apply_move(move)

            # apply minimax algorithm to this new state
            _, value = minimax_stockfish(game_state, True, turn, model, depth - 1, alpha, beta, counter)

            
            game_state.undo_move()

            # if this new move generates a better state than previous
            if value < min_value:
                # update best_value and best_move
                min_value = value
                best_move = move


            # pruning
            beta = min(beta, value)
            if beta <= alpha:
                break
        
        """
        num_pieces = game_state.count_pieces()
        if num_pieces > 32:
            print(num_pieces, "counter: ", counter["iterations"])
        """
        
        return best_move, min_value


"""
Improvements:
transposition table for moves
move ordering to do optimal moves first (captures, checks, etc) to optimize alpha-beta pruning
store moves that protect and attack
count number of recursive function calls (to show benefit of something like alpha-beta pruning)
openings database
piece value masks

"""