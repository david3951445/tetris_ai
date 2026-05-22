from collections import deque
from dataclasses import dataclass
from enum import Enum

from environment.piece import Piece
from environment.board import Board


class PieceType(Enum):
    I = 0  # light blue
    J = 1  # dark blue
    L = 2  # orange
    O = 3  # yellow
    S = 4  # green
    Z = 5  # red
    T = 6  # magenta


@dataclass(frozen=True, slots=True)
class Action:
    def __init__(self, piece: Piece, name: str):
        self.piece = piece
        self.name = name


class GameState:
    def __init__(self):
        self.current: Piece
        self.hold: Piece
        self.next_pieces = deque[Piece]()
        self.board = Board()


class Player:
    def __init__(self):
        pass

    def decide(game_state: GameState) -> Action:
        # 1. no action
        # 2. place the current piece
        # 3. hold current piece
        pass


class Game:
    def __init__(self, player: Player):  # Player is not defined
        self.state = GameState()
        self.player = player

    def start(self):
        while not self.state.board.is_game_over():
            action = self.player.decide(self.state)

            if action.name is "place":
                cells = action.piece.cells
                self.state.board.place(cells)
                continue

            if action.name is "hold":
                temp = self.state.hold
                self.state.hold = action.piece
                temp.transform(Piece.INITIAL_POS, Piece.INITIAL_ROTATION)
                cells = temp.cells
                self.state.board.place(cells)
                continue
            
            # otherwise no action
            
            # show game state by ui.py
