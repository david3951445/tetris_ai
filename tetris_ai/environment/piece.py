import random
from typing import List, Tuple

# Each piece: list of rotation states, each state is list of (row, col) offsets
TETROMINOES = {
    "I": [
        [(0,0),(0,1),(0,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0)],
    ],
    "O": [
        [(0,0),(0,1),(1,0),(1,1)],
    ],
    "T": [
        [(0,1),(1,0),(1,1),(1,2)],
        [(0,0),(1,0),(2,0),(1,1)],
        [(1,0),(1,1),(1,2),(0,1)],
        [(0,1),(1,1),(2,1),(1,0)],
    ],
    "S": [
        [(0,1),(0,2),(1,0),(1,1)],
        [(0,0),(1,0),(1,1),(2,1)],
    ],
    "Z": [
        [(0,0),(0,1),(1,1),(1,2)],
        [(0,1),(1,0),(1,1),(2,0)],
    ],
    "J": [
        [(0,0),(1,0),(1,1),(1,2)],
        [(0,0),(0,1),(1,0),(2,0)],
        [(1,0),(1,1),(1,2),(2,2)],
        [(0,1),(1,1),(2,0),(2,1)],
    ],
    "L": [
        [(0,2),(1,0),(1,1),(1,2)],
        [(0,0),(1,0),(2,0),(2,1)],
        [(1,0),(1,1),(1,2),(2,0)],
        [(0,0),(0,1),(1,1),(2,1)],
    ],
}

PIECE_NAMES = list(TETROMINOES.keys())


class Piece:
    """Represents a single Tetromino with its type, rotation state, and position."""

    def __init__(self, name: str = None):
        self.name: str = name or random.choice(PIECE_NAMES)
        self.rotations: List[List[Tuple[int, int]]] = TETROMINOES[self.name]
        self.rotation_index: int = 0
        self.row: int = 0
        self.col: int = 0

    @property
    def cells(self) -> List[Tuple[int, int]]:
        """Current rotation's cell offsets."""
        return self.rotations[self.rotation_index]

    @property
    def num_rotations(self) -> int:
        return len(self.rotations)

    def get_cells_at(self, row: int, col: int, rotation: int) -> List[Tuple[int, int]]:
        """Return absolute (row, col) positions for a given placement."""
        return [(row + dr, col + dc) for dr, dc in self.rotations[rotation]]

    def rotate(self) -> None:
        self.rotation_index = (self.rotation_index + 1) % self.num_rotations

    @staticmethod
    def random() -> "Piece":
        return Piece(random.choice(PIECE_NAMES))
