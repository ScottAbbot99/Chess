"""
Stores board state, valid moves and move log
"""

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

        # dictionary to indicate of en passant is possible
        self.en_passant = {'w': {"able": False, "column": None}, 'b': {"able": False, "column": None}}

        self.white_turn = True # stores who's turn it is

        # determine if king or rooks have moved yet
        # "0" represents queen rook and "7" represents king rook
        self.king_rook_moved = {'w': {"king": False, "0": False, "7": False}, 
                                'b': {"king": False, "0": False, "7": False}} 

        self.move_log = []