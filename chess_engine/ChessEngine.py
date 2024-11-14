"""
Stores board state, valid moves and move log
"""

import numpy as np

# definitions
HEIGHT, WIDTH = 512, 512 # board image size
NUM_ROWS = NUM_COLS = 8 # board is 8x8
SQUARE_SIZE = WIDTH / NUM_ROWS # size of each square is 512/8 x 512/8
MAX_FPS = 15 
IMAGES = {} # dictionary to store images

class GameState():
    def __init__(self):

        """
        8x8 board with the following: 
        '--' = empty square
        otherwise, the first character represents the color and the second represents the piece
        """

        self.board = [
            ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR'],
            ['bP', 'bP', 'bP', 'bP', 'bP', 'bP', 'bP', 'bP'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['wP', 'wP', 'wP', 'wP', 'wP', 'wP', 'wP', 'wP'],
            ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR']
        ]

        # dictionary to indicate if en passant is possible
        # "w" indicates if white can en passant
        self.en_passant = {'w': {"able": False, "column": None}, 'b': {"able": False, "column": None}}

        self.white_turn = True # stores who's turn it is

        # determine if king or rooks have moved yet
        # "0" represents queen rook and "7" represents king rook
        self.king_rook_moved = {'w': {"king": False, "0": False, "7": False}, 
                                'b': {"king": False, "0": False, "7": False}} 

        self.move_log = []

        self.piece_functions = {"P":self.all_pawn_moves, "R":self.all_rook_moves, "N":self.all_knight_moves, "B":self.all_bishop_moves,
                           "Q":self.all_queen_moves, "K":self.all_king_moves}

        # quick access to white/black king locations
        self.king_locations = {"white":(7,4), "black":(0,4)}

        # determine if white/black kings have moved (for castling)
        self.king_moved = {'w':(False, None), 'b':(False, None)}
        # determine if the rooks have moved (for castling)
        self.rooks_moved = {'w':{'A':(False, None), 'H':(False, None)}, 'b':{'A':(False, None), 'H':(False, None)}}


        # stores the location of pinned pieces and checking pieces
        self.pinned = []
        self.checking = []

        # boolean representing if it's stalemate or checkmate
        self.stalemate = False
        self.checkmate = False

        # move counter
        self.move_counter = 0
    
    def get_all_moves(self):
        """
        gets all possible moves
        returns all the moves for the given turn as a list of Class Move
        """

        # list to store moves
        moves = [] 

        # king details
        king_location = self.get_king_location(self.white_turn)
        king_piece = self.get_friendly(self.white_turn) + 'K'
        king_checked = self.checks()

        # double check means the king must move
        if (king_checked and len(self.checking) == 2):
            self.all_king_moves(king_piece, king_location[0], king_location[1], moves)
            return moves

        # loop through every square and gather all the possible moves
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):

                # ensure it's the proper turn and there's a piece there
                if (self.white_turn and self.board[row][col][0] == 'w') or (not self.white_turn and self.board[row][col][0] == 'b'):

                    # determine what piece is on the given square
                    piece = self.board[row][col]

                    # get and use the correct piece function
                    piece_function = self.piece_functions[piece[1]]
                    piece_function(piece, row, col, moves)

        # king is singly checked
        if king_checked:

            # tuple representing the location of the piece checking the king
            checking_location = self.checking[0] 

            # obtain a vector from the king to the checking piece
            checking_vector = (checking_location[0] - king_location[0], checking_location[1] - king_location[1])
            checking_row_magnitude = np.abs(checking_vector[0])
            checking_col_magnitude = np.abs(checking_vector[1])

            # store vector directions
            checking_direction_row = 0 if checking_row_magnitude == 0 else ((checking_location[0] - king_location[0]) / checking_row_magnitude)
            checking_direction_col = 0 if checking_col_magnitude == 0 else ((checking_location[1] - king_location[1]) / checking_col_magnitude)

            # boolean variables to determine if a piece can block the check
            # pieces 1 square away from the king cannot be blocked
            piece_next_to_king = (checking_row_magnitude == 1 and checking_col_magnitude == 1) or (checking_row_magnitude == 1 and checking_col_magnitude == 0) or (checking_row_magnitude == 0 and checking_col_magnitude == 1)
            knight_checking = ((checking_row_magnitude == 1) and (checking_col_magnitude == 2)) or ((checking_row_magnitude == 2) and (checking_col_magnitude == 1))

            # the king must move or the checking piece must be captured
            if piece_next_to_king or knight_checking:
                # loop through the moves in reverse and remove moves that won't resolve the check
                for i in range(len(moves) - 1, -1, -1):
                    cur_move = moves[i]

                    # if the current move does not capture the checking piece and it's not a king move, then remove it from the moves list
                    if (not self.verify_move(cur_move, checking_location[0], checking_location[1])) and (cur_move.piece != king_piece):
                        moves.pop(i)

            # otherwise we can also block
            else:
                # loop through the moves in reverse and remove moves that won't resolve the check
                for i in range(len(moves) - 1, -1, -1):
                    cur_move = moves[i]

                    # king moves are acceptable
                    if cur_move.piece == king_piece:
                        continue

                    # a move that captures the checking piece is acceptable
                    elif self.verify_move(cur_move, checking_location[0], checking_location[1]):
                        continue

                    # a move that blocks the check is acceptable
                    elif self.blocking_move(cur_move, king_location, checking_row_magnitude, checking_col_magnitude, checking_direction_row, checking_direction_col):
                        continue

                    # remove piece otherwise. Also, a capturing piece is dealt with in blocking_move
                    else:
                        moves.pop(i)


        # account for pins
        if len(self.pinned) > 0:

            # loop through each move and ensure a pinned piece isn't moving illegally
            for i in range(len(moves) - 1, -1, -1):
                cur_move = moves[i]
                cur_row = cur_move.from_row
                cur_col = cur_move.from_col
                new_row = cur_move.to_row
                new_col = cur_move.to_col

                pinned_row = self.pinned[0][0]
                pinned_col = self.pinned[0][1]

                # only consider the pinned piece
                if (cur_row == pinned_row) and (cur_col == pinned_col):

                    # ensure this piece stays in line with the king

                    # slope from where the piece currently is
                    piece_king_row_diff = cur_row - king_location[0]
                    piece_king_col_diff = cur_col - king_location[1]
                    piece_to_king_slope = float("inf") if (piece_king_col_diff == 0) else (piece_king_row_diff / piece_king_col_diff)

                    # slope from where the piece is moving
                    new_row_diff = new_row - king_location[0]
                    new_col_diff = new_col - king_location[1]
                    new_slope = float("inf") if (new_col_diff == 0) else (new_row_diff / new_col_diff)

                    # remove the move if the two slopes are different
                    if (new_slope != piece_to_king_slope):
                        moves.pop(i)


        return moves

    def all_pawn_moves(self, piece, row, col, moves):
        """
        Pawns can move forward 1 or 2 squares and capture diagonally 1 square
        En passant
        """
        piece_color = piece[0]

        # TODO Consolidate the below (one set of code for both black and white)

        # white pawns
        if piece_color == 'w':

            # pawns can move forward 1 assuming it's an empty square
            if self.board[row - 1][col] == "--":
                moves.append(Move(piece, row, col, row - 1, col))

                # move forward 2 if on it's home square
                if row == 6 and self.board[row - 2][col] == "--":
                    moves.append(Move(piece, row, col, row - 2, col, pawn_advanced_2 = True))

            
            # capturing
            if (col + 1 < 8) and self.board[row - 1][col + 1][0] == 'b':
                moves.append(Move(piece, row, col, row - 1, col + 1))
            if (col - 1 >= 0) and self.board[row - 1][col - 1][0] == 'b':
                moves.append(Move(piece, row, col, row - 1, col - 1))

            
            # en passant
            self.move_en_passant(piece, row, col, moves)


        # black pawns
        else:

            # pawns can move forward 1 assuming it's an empty square
            if self.board[row + 1][col] == "--":
                moves.append(Move(piece, row, col, row + 1, col))

                # move forward 2 if on it's home square
                if row == 1 and self.board[row + 2][col] == "--":
                    moves.append(Move(piece, row, col, row + 2, col, pawn_advanced_2 = True))
                

            # capturing
            if (col + 1 < 8) and self.board[row + 1][col + 1][0] == 'w':
                moves.append(Move(piece, row, col, row + 1, col + 1))
            if (col - 1 >= 0) and self.board[row + 1][col - 1][0] == 'w':
                moves.append(Move(piece, row, col, row + 1, col - 1))
        
            # en passant
            self.move_en_passant(piece, row, col, moves)


    def all_rook_moves(self, piece, row, col, moves):
        """
        A rook can move along the rows or column
        """

        # vector representing moving along each direction
        rook_vector = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        self.explore_moves(piece, row, col, rook_vector, moves)


    def all_knight_moves(self, piece, row, col, moves):
        """
        A knight can move 2 squares along row/col and 1 square along col/row
        """

        piece_color = piece[0]

        # knight directions
        knight_vector = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

        for vector in knight_vector:
            # get the new square
            potential_square = (row + vector[0], col + vector[1])
            # ensure the square is on the board
            if (0 <= potential_square[0] < 8) and (0 <= potential_square[1] < 8):
                # ensure the square doesn't have a friendly piece
                if piece_color != self.board[potential_square[0]][potential_square[1]][0]:
                    moves.append(Move(piece, row, col, potential_square[0], potential_square[1]))


    def all_bishop_moves(self, piece, row, col, moves):
        """
        A bishop can move along the diagonals
        """

        # bishop directions
        bishop_vector = [(1, 1), (-1, 1), (-1, -1), (1, -1)]

        self.explore_moves(piece, row, col, bishop_vector, moves)


    def all_queen_moves(self, piece, row, col, moves):
        """
        A Queen can move along any direction
        """

        # Queen is just a rooke combined with a bishop
        self.all_rook_moves(piece, row, col, moves)
        self.all_bishop_moves(piece, row, col, moves)
        


    def all_king_moves(self, piece, row, col, moves):
        """
        A King can move a single square in any direction
        """

        piece_color = piece[0]

        king_vector = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]

        # enemy piece color
        enemy_color = self.get_enemy(self.white_turn)
        
        # go along each direction and determine if there's a friendly piece there
        for vector in king_vector:
            cur_position = (row + vector[0], col + vector[1])
            # ensure the new position is in bounds
            if not (0 <= cur_position[0] < 8) or not (0 <= cur_position[1] < 8):
                continue

            # no friendly piece means the king can move there so long as the square is not under attack
            if self.board[cur_position[0]][cur_position[1]][0] != piece_color:
                if not self.square_under_attack(cur_position, piece_color, enemy_color, False):
                    moves.append(Move(piece, row, col, cur_position[0], cur_position[1]))

        
        self.castling(piece, row, col, moves)
    
    def apply_move(self, move):
        """
        Takes a Move object as an input and makes the given move
        """

        # store piece on "captured" square
        captured_piece = self.board[move.to_row][move.to_col]

        # first update the square where the piece is moving to
        self.board[move.to_row][move.to_col] = move.piece

        # remove the piece from it's original square
        self.board[move.from_row][move.from_col] = "--"

        # update en passant dictionary if a pawn advanced two squares
        if move.pawn_advanced_2:
            enemy_color = self.get_enemy(self.white_turn)
            self.en_passant[enemy_color] = {"able":True, "column":move.to_col}

        # handle en passant
        if move.en_passant:
            self.board[move.from_row][move.to_col] = "--"


        # account for castling if a rook/king moves
        turn = self.get_friendly(self.white_turn)
        # if the king is moving and the king hasn't already moved, then ensure the king can no longer castle
        # also store the move at which the king first move (for undoing moves later)
        if move.piece[1] == 'K':
            if not self.king_moved[turn][0]:
                self.king_moved[turn] = (True, self.move_counter)

        # similar for rooks
        if move.piece[1] == 'R':
            # left rooks are (7, 0) / (0, 0) for w/b, and right rooks are (7, 7) / (0, 7) for w/b
            rook_types = {'w':[(7,0), (7,7)], 'b':[(0,0), (0,7)]}
            if (move.from_row, move.from_col) in rook_types[turn]:
                if (move.from_col == 0) and not self.rooks_moved[turn]['A']:
                    self.rooks_moved[turn]['A'] = True
                if (move.from_col == 7) and not self.rooks_moved[turn]['H']:
                    self.rooks_moved[turn]['H'] = True
            
        # handle castling
        if move.castling[0]:

            # just need to move the rook to the proper square

            # king side castle
            if move.castling[1] == 'H':
                self.board[move.from_row][5] = move.piece[0] + 'R'
                self.board[move.from_row][7] = "--"
            # queen side castle
            else:
                self.board[move.from_row][3] = move.piece[0] + 'R'
                self.board[move.from_row][0] = "--"
            
        
        # pawn promotion
        if move.piece[1] == 'P':
            if (move.to_row == 0) and (move.piece[0] == 'w'):
                # white pawn promoted
                self.board[0][move.to_col] = move.piece[0] + 'Q'
            elif (move.to_row == 7) and (move.piece[0] == 'b'):
                # black pawn promoted
                self.board[7][move.to_col] = move.piece[0] + 'Q'


        # updates move_log
        self.move_log.append((move, captured_piece))

        
        # update king location if the king moves
        if move.piece[1] == 'K':
            if self.white_turn:
                self.king_locations["white"] = (move.to_row, move.to_col)
            else:
                self.king_locations["black"] = (move.to_row, move.to_col)

        # swap turns
        self.white_turn = not self.white_turn


        # reset self.pinned and self.checking
        self.pinned = []
        self.checking = []

        # update move counter
        self.move_counter = len(self.move_log)

    def undo_move(self):
        """
        Undo moves based on the move log
        """

        move_log = self.move_log
        move_count = self.move_counter

        current_move_idx = move_count - 1

        # reset self.pinned and self.checking
        self.pinned = []
        self.checking = []

        # can't undo a move if no move has been made yet
        if current_move_idx < 0:
            return

        # change turn back
        self.white_turn = not self.white_turn

        
        # gather the move in consideration
        current_move = move_log[current_move_idx][0]
        captured_piece = move_log[current_move_idx][1]


        # reverse the move
        self.board[current_move.from_row][current_move.from_col] = current_move.piece
        self.board[current_move.to_row][current_move.to_col] = captured_piece

        # en passant reversal
        if current_move.pawn_advanced_2:
            enemy_color = self.get_enemy(self.white_turn)
            self.en_passant[enemy_color] = {"able":False, "column":None}
        if current_move.en_passant:
            self.board[current_move.from_row][current_move.to_col] = enemy_color + 'P'


        # castling reversal
        turn = self.get_friendly(self.white_turn)
        # first undo whether the king has moved
        if current_move.piece[1] == 'K':
            # determine if we're undoing the move when the king first moved
            if self.king_moved[turn][0] and current_move_idx == self.king_moved[turn][1]:
                self.king_moved[turn] = (False, None)
        
        # rook flags
        if current_move.piece[1] == 'R':
            # left rooks are (7, 0) / (0, 0) for w/b, and right rooks are (7, 7) / (0, 7) for w/b
            rook_types = {'w':[(7,0), (7,7)], 'b':[(0,0), (0,7)]}
            if (current_move.from_row, current_move.from_col) in rook_types[turn]:
                if (current_move.from_col == 0) and self.rooks_moved[turn]['A'] and self.rooks_moved[turn]['A'][1] == current_move_idx:
                    self.rooks_moved[turn]['A'] = False
                if (current_move.from_col == 7) and self.rooks_moved[turn]['H'] and self.rooks_moved[turn]['H'][1] == current_move_idx:
                    self.rooks_moved[turn]['H'] = False

        # undo rook movements
        if current_move.castling[0]:
            # king side
            if current_move.castling[1] == 'H':
                self.board[current_move.from_row][7] = current_move.piece[0] + 'R'
                self.board[current_move.from_row][5] = "--"
            # queen side
            else:
                self.board[current_move.from_row][0] = current_move.piece[0] + 'R'
                self.board[current_move.from_row][3] = "--"

        # undo pawn promotion
        if current_move.piece[1] == 'P':
            if (current_move.to_row == 0) and (current_move.piece[0] == 'w'):
                self.board[0][current_move.to_col] = captured_piece
            elif (current_move.to_row == 7) and (current_move.piece[0] == 'b'):
                self.board[7][current_move.to_col] = captured_piece

        #  revert king location if the king moves
        if current_move.piece[1] == 'K':
            if self.white_turn:
                self.king_locations["white"] = (current_move.from_row, current_move.from_col)
            else:
                self.king_locations["black"] = (current_move.from_row, current_move.from_col)


        # revert stalemate / checkmate
        if self.checkmate:
            self.checkmate = False
        elif self.stalemate:
            self.stalemate = False


        # update the move counter and move log
        self.move_log.pop()
        self.move_counter = len(self.move_log)


    def checks(self):
        """
        Outputs True if the king is in check
        """

        enemy_piece_color = self.get_enemy(self.white_turn)
        friendly_piece_color = self.get_friendly(self.white_turn)
        king_location = self.get_king_location(self.white_turn)
        king_square = True

        in_check = self.square_under_attack(king_location, friendly_piece_color, enemy_piece_color, king_square)

        return in_check


    def square_under_attack(self, square, friendly_piece_color, enemy_piece_color, square_is_king):
        """
        Outputs True if a square is under attack
        square_is_king is true only if the square being evluated is a king. This ensures pins/checks are only added if the king square is evaluated
        Also uncoveres pins / checks
        """
            
        # expand outward from the given square and determine if an enemy piece is attacking that square
        pawn_attacking = self.pawn_attacking(square, friendly_piece_color, enemy_piece_color, square_is_king)
        knight_attacking = self.knight_attacking(square, enemy_piece_color, square_is_king)
        bishop_attacking = self.bishiop_attacking(square, friendly_piece_color, enemy_piece_color, square_is_king)
        rook_attacking = self.rook_attacking(square, friendly_piece_color, enemy_piece_color, square_is_king)
        queen_attacking = self.queen_attacking(square, friendly_piece_color, enemy_piece_color, square_is_king)
        king_attacking = self.king_attacking(square, enemy_piece_color)

        # return true if a piece is attacking the square
        if pawn_attacking or knight_attacking or rook_attacking or bishop_attacking or queen_attacking or king_attacking:
            return True

        return False


    def castling(self, piece, row, col, moves):
        """
        Determines if the king can castle
        """

        piece_color = piece[0]

        # ensure the king has not moved
        if self.king_moved[piece_color][0]:
            return

        # store the legal castling moves
        self.castling_squares(piece, row, col, moves)
    
    def castling_squares(self, piece, row, col, moves):
        """
        evaluates the castling squares
        """

        friendly_color = piece[0]
        enemy_color = self.get_enemy(self.white_turn)

        # ensure the king isn't in check
        if self.square_under_attack((row, col), friendly_color, enemy_color, False):
            return
        

        # evaluate the two squares on either side of the king

        # A-side (queen side)
        if not self.square_under_attack((row, col-1), friendly_color, enemy_color, False) \
           and not self.square_under_attack((row, col-2), friendly_color, enemy_color, False) \
           and not self.rooks_moved[friendly_color]['A'][0] \
           and self.squares_open(row, 'A'):
            # store the castling move
            moves.append(Move(piece, row, col, row, col-2, castling=[True, 'A']))
        
        # H-side (king side)
        if not self.square_under_attack((row, col+1), friendly_color, enemy_color, False) \
           and not self.square_under_attack((row, col+2), friendly_color, enemy_color, False) \
           and not self.rooks_moved[friendly_color]['H'][0] \
           and self.squares_open(row, 'H'):
            # store the castling move
            moves.append(Move(piece, row, col, row, col+2, castling=[True, 'H']))

    def squares_open(self, row, side):
        """
        Determines if the squares from the king to the rook are open
        side: represents either 'A' or 'H' for which side the king can castle to
        """

        squares_open = True

        if side == 'A':
            for i in range(1, 4):
                if self.board[row][i] != "--":
                    squares_open = False
        
        elif side == 'H':
            for i in range(5, 7):
                if self.board[row][i] != "--":
                    squares_open = False
        
        return squares_open


    def pawn_attacking(self, square, friendly_piece_color, enemy_piece_color, square_is_king):
        """
        Outputs True if a pawn is attacking the square
        square_is_king is true if square represents a king square. In this case we must record all pins and checks
        """

        # vectors where pawns could be for each color
        attacking_vector = {'w':[(-1, 1), (-1, -1)], 'b':[(1, 1), (1, -1)]}
        
        # pawns attack 1 square diagonally
        # obtain both possible squares where a pawn could be attacking from
        square1 = (square[0] + attacking_vector[friendly_piece_color][0][0], square[1] + attacking_vector[friendly_piece_color][0][1])
        square2 = (square[0] + attacking_vector[friendly_piece_color][1][0], square[1] + attacking_vector[friendly_piece_color][1][1])

        if (self.board[square1[0]][square1[1]] == enemy_piece_color + 'P'):
            if square_is_king:
                self.checking.append(square1)
            return True

        if (self.board[square2[0]][square2[1]] == enemy_piece_color + 'P'):
            if square_is_king:
                self.checking.append(square2)
            return True
            
        return False

    def knight_attacking(self, square, enemy_piece_color, square_is_king):
        """
        Outputs True if a knight is attacking the square
        square_is_king is true if square represents a king square. In this case we must record all pins and checks
        """

        # knight directions
        knight_vector = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

        for vector in knight_vector:
            cur_square = (square[0] + vector[0], square[1] + vector[1])
            # ensure the square is on the board
            if (0 <= cur_square[0] < 8) and (0 <= cur_square[1] < 8):
                if (self.board[cur_square[0]][cur_square[1]][0] == enemy_piece_color + 'N'):
                    if square_is_king:
                        self.checking.append(cur_square)
                    return True
        
        return False

    def bishiop_attacking(self, square, friendly_piece_color, enemy_piece_color, square_is_king, attacker_is_queen = False):
        """
        Outputs True if a bishop is attacking the square
        square_is_king is true if square represents a king square. In this case we must record all pins and checks
        """

        # bishop directions
        bishop_vector = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        
        if attacker_is_queen:
            enemy_piece = enemy_piece_color + 'Q'
        else:
            enemy_piece = enemy_piece_color + 'B'

        # adds pins/checks and determines if the square is under attack
        square_attacked = self.explore_pins_checks(square, friendly_piece_color, enemy_piece, bishop_vector, square_is_king)
        
        return square_attacked

    def rook_attacking(self, square, friendly_piece_color, enemy_piece_color, square_is_king, attacker_is_queen = False):
        """
        Outputs True if a rook is attacking the square
        square_is_king is true if square represents a king square. In this case we must record all pins and checks
        """

        # rook directions
        rook_vector = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        if attacker_is_queen:
            enemy_piece = enemy_piece_color + 'Q'
        else:
            enemy_piece = enemy_piece_color + 'R'

        # adds pins/checks and determines if the square is under attack
        square_attacked = self.explore_pins_checks(square, friendly_piece_color, enemy_piece, rook_vector, square_is_king)

        return square_attacked

    def queen_attacking(self, square, friendly_piece_color, enemy_piece_color, square_is_king):
        """
        Outputs True if a queen is attacking the square
        square_is_king is true if square represents a king square. In this case we must record all pins and checks
        """

        # queen is a bishop combined with a rook
        if self.rook_attacking(square, friendly_piece_color, enemy_piece_color, square_is_king, attacker_is_queen=True) \
            or self.bishiop_attacking(square, friendly_piece_color, enemy_piece_color, square_is_king, attacker_is_queen=True):
            return True

        return False

    def king_attacking(self, square, enemy_piece_color):
        """
        Outputs True if a king is attacking the square
        """

        king_vector = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]

        # loop along each direction and determine if square is attacked
        for vector in king_vector:
            cur_position = (square[0] + vector[0], square[1] + vector[1])

            # ensure the current square is on the board
            if (0 <= cur_position[0] < 8) and (0 <= cur_position[1] < 8):
                # determine if an enemy king is on the current square
                if self.board[cur_position[0]][cur_position[1]] == enemy_piece_color + 'K':
                    return True
        
        return False

    
    def explore_moves(self, piece, row, col, vector_set, moves):
        """
        Garners all moves available according to the given vector set
        """

        piece_color = piece[0]

        # enemy piece color
        enemy_piece_color = self.get_enemy(self.white_turn)

        # for each vector, add that vector until a piece is reached or the end of the board
        for vector in vector_set:
            cur_position = (row + vector[0], col + vector[1])

            while (0 <= cur_position[0] < 8) and (0 <= cur_position[1] < 8):

                # ensure there is no friendly piece on the square
                if self.board[cur_position[0]][cur_position[1]][0] == piece_color:
                    break
                
                # add move but stop looping if it's an emeny piece
                if self.board[cur_position[0]][cur_position[1]][0] == enemy_piece_color:
                    moves.append(Move(piece, row, col, cur_position[0], cur_position[1]))
                    break

                # otherwise it's an empty square
                moves.append(Move(piece, row, col, cur_position[0], cur_position[1]))

                # update cur_position
                cur_position = (cur_position[0] + vector[0], cur_position[1] + vector[1])


    def explore_pins_checks(self, square, friendly_piece_color, enemy_piece, vector_set, square_is_king):
        """
        Explores outwards along each vector in the given vector set looking for pins and checks
        """
        # boolean representing if the given square is under attack
        square_attacked = False

        for vector in vector_set:
            cur_square = (square[0] + vector[0], square[1] + vector[1])

            potential_pin = []
            potential_check = []

            while (0 <= cur_square[0] < 8) and (0 <= cur_square[1] < 8):

                # if the piece is a friendly piece, then add it as a potential pin
                if (self.board[cur_square[0]][cur_square[1]][0] == friendly_piece_color):
                    potential_pin.append(cur_square)
                    cur_square = (cur_square[0] + vector[0], cur_square[1] + vector[1])

                # if it's an enemy piece, then add it as a potential checking piece
                elif (self.board[cur_square[0]][cur_square[1]] == enemy_piece):
                    potential_check.append(cur_square)
                    # no reason to check more squares
                    break
                
                # otherwise move to the next square
                else:
                    cur_square = (cur_square[0] + vector[0], cur_square[1] + vector[1])

            # if the length of potential_pin > 1 then this means there are no pins nor are there any checks along that direction
            if len(potential_pin) > 1: 
                continue

            elif len(potential_check) == 1:
                # potential check and one potential pin means the piece IS pinned
                if len(potential_pin) == 1:
                    if square_is_king:
                        self.pinned.append(potential_pin[0])
                # otherwise there is a piece attacking the square
                else:
                    if square_is_king:
                        self.checking.append(potential_check[0])
                    square_attacked = True
        
        return square_attacked


    def move_en_passant(self, piece, row, col, moves):
        """
        make en passant move
        """

        if piece[0] == 'w':

            # pawn needs to be on the 3rd row
            if (row == 3):
                en_passant = self.en_passant["w"]

                # an enemy pawn must have just advanced two squares
                if en_passant["able"]:
                    
                    # the given pawn must be directly next to the pawn that advanced two
                    col_diff = col - en_passant["column"]
                    if np.abs(col_diff) == 1:
                        
                        # the move is legal
                        moves.append(Move(piece, row, col, row - 1, en_passant["column"], True))
        
        if piece[0] == 'b':

            # pawn needs to be on the 4th row
            if (row == 4):
                en_passant = self.en_passant["b"]

                # an enemy pawn must have just advanced two squares
                if en_passant["able"]:
                    
                    # the given pawn must be directly next to the pawn that advanced two
                    col_diff = col - en_passant["column"]
                    if np.abs(col_diff) == 1:
                        
                        # the move is legal
                        moves.append(Move(piece, row, col, row + 1, en_passant["column"], True))




    def get_enemy(self, turn):
        """
        Turn is true if it's white's turn and false if not
        Outputs the color of the enemy pieces
        """

        if turn:
            return 'b'
        return 'w'

    def get_friendly(self, turn):
        """
        Turn is true if it's white's turn and false if not
        Outputs the color of the friendly pieces
        """

        if turn:
            return 'w'
        return 'b'

    def get_king_location(self, turn):
        if turn:
            return self.king_locations["white"]
        return self.king_locations["black"]
    
    
    def verify_move(self, Move, to_row, to_col):
        """
        Checks if the Move object goes to the desired row and col
        """

        move_row = Move.to_row
        move_col = Move.to_col

        if (move_row == to_row) and (move_col == to_col):

            # if the capturing piece is a king then we must ensure the checking piece isn't defended
            if Move.piece[1] == 'K':
                return True
        
        return False

    def blocking_move(self, Move, king_location, checking_row_magnitude, checking_col_magnitude, checking_direction_row, checking_direction_col):
        """
        Determines if a move blocks a check on the king
        """

        move_row = Move.to_row
        move_col = Move.to_col

        # calculate a vector from the king to where the piece is moving
        piece_vector_row = move_row - king_location[0]
        piece_vector_col = move_col - king_location[1]
        piece_row_magnitude = np.abs(piece_vector_row)
        piece_col_magnitude = np.abs(piece_vector_col)

        # determine the slope of the checking piece and the "blocking" piece
        checking_slope = float("inf") if checking_direction_col == 0 else (checking_direction_row / checking_direction_col)
        blocking_slope = float("inf") if piece_vector_col == 0 else (piece_vector_row / piece_vector_col)


        # to determine if the piece is blocking the check, it has to be inline with the checking_vector and between the king and checking piece

        # if the slopes of the blocking piece and the checking piece do not match, then the piece cannot be blocking
        if (checking_slope != blocking_slope):
            return False

        # otherwise we must ensure the piece is between the king and checking piece
        elif (piece_row_magnitude > checking_row_magnitude) or (piece_col_magnitude > checking_col_magnitude):
            return False

        
        return True




class Move():

    def __init__(self, piece, from_row, from_col, to_row, to_col, en_passant = False, pawn_advanced_2 = False, castling = [False, None]):
        """
        en_passant: true if move is en passant
        pawn_advanced_2: True if pawn just advanced two squares
        castling: list indicating if a move is a castling move and which side
        """

        # values
        self.piece = piece
        self.from_row = from_row
        self.from_col = from_col
        self.to_row = to_row
        self.to_col = to_col

        # store if whether the move is en passant
        self.en_passant = en_passant
        self.pawn_advanced_2 = pawn_advanced_2

        # store if the move is a castling move and which side
        self.castling = castling

    def __eq__(self, other):
        """
        Checks if two Move objects are the same by comparing their attributes.
        """
        if isinstance(other, Move):
            return (self.piece == other.piece and
                    self.from_row == other.from_row and
                    self.from_col == other.from_col and
                    self.to_row == other.to_row and
                    self.to_col == other.to_col)
        return False

    def __str__(self):
        """
        Return a string representation of Move
        """

        return f"{self.piece} moves from ({self.from_row}, {self.from_col}) to ({self.to_row}, {self.to_col})"


"""
Notes:

make chess notation (use dictionaries to convert from (0, 0) to a8)
Bitboard
undo move
move log
pawn promotion

Improving speed:
while exploring along bishop/rook directions, can also check for pawns/king 
self.king_location should just have 'w' rather than 'white'

after undoing a move, the engine will need to recalculate possible moves which is inefficient
"""