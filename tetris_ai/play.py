from collections import deque
from enum import Enum


class PieceType(Enum):
    I = 0  # light blue
    J = 1  # dark blue
    L = 2  # orange
    O = 3  # yellow
    S = 4  # green
    Z = 5  # red
    T = 6  # magenta


class Game:
    def __init__(self):
        self.hold = None
        self.next_pieces = deque()
