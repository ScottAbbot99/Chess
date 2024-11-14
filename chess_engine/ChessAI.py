import random


def random_move(moves):
    """
    Selects a random move
    """

    random_move = moves[random.randint(0, len(moves) - 1)]

    return random_move