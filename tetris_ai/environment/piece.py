import random
from collections.abc import Generator

from .coord import Coord

# Each piece: list of rotation states, each state is list of Coord offsets
TETROMINOES: dict[str, list[list[Coord]]] = {
    "I": [
        [Coord(0,0), Coord(0,1), Coord(0,2), Coord(0,3)],
        [Coord(0,0), Coord(1,0), Coord(2,0), Coord(3,0)],
    ],
    "O": [
        [Coord(0,0), Coord(0,1), Coord(1,0), Coord(1,1)],
    ],
    "T": [
        [Coord(0,1), Coord(1,0), Coord(1,1), Coord(1,2)],
        [Coord(0,0), Coord(1,0), Coord(2,0), Coord(1,1)],
        [Coord(1,0), Coord(1,1), Coord(1,2), Coord(0,1)],
        [Coord(0,1), Coord(1,1), Coord(2,1), Coord(1,0)],
    ],
    "S": [
        [Coord(0,1), Coord(0,2), Coord(1,0), Coord(1,1)],
        [Coord(0,0), Coord(1,0), Coord(1,1), Coord(2,1)],
    ],
    "Z": [
        [Coord(0,0), Coord(0,1), Coord(1,1), Coord(1,2)],
        [Coord(0,1), Coord(1,0), Coord(1,1), Coord(2,0)],
    ],
    "J": [
        [Coord(0,0), Coord(1,0), Coord(1,1), Coord(1,2)],
        [Coord(0,0), Coord(0,1), Coord(1,0), Coord(2,0)],
        [Coord(1,0), Coord(1,1), Coord(1,2), Coord(2,2)],
        [Coord(0,1), Coord(1,1), Coord(2,0), Coord(2,1)],
    ],
    "L": [
        [Coord(0,2), Coord(1,0), Coord(1,1), Coord(1,2)],
        [Coord(0,0), Coord(1,0), Coord(2,0), Coord(2,1)],
        [Coord(1,0), Coord(1,1), Coord(1,2), Coord(2,0)],
        [Coord(0,0), Coord(0,1), Coord(1,1), Coord(2,1)],
    ],
}

PIECE_NAMES = list(TETROMINOES.keys())


class Piece:
    """Represents a single Tetromino with its type, rotation state, and position."""

    def __init__(self, name: str = None):
        self.name: str = name or random.choice(PIECE_NAMES)
        self.rotations: list[list[Coord]] = TETROMINOES[self.name]
        self.rotation_index: int = 0
        self.pos: Coord = Coord(0, 0)  # top-left anchor on the board

    @property
    def cells(self) -> list[Coord]:
        """Current rotation's cell offsets."""
        return self.rotations[self.rotation_index]

    @property
    def num_rotations(self) -> int:
        return len(self.rotations)

    def get_cells_at(self, row: int, col: int, rotation: int) -> list[Coord]:
        """Return absolute Coord positions for a given placement."""
        origin = Coord(row, col)
        return [origin + offset for offset in self.rotations[rotation]]

    def rotate(self) -> None:
        self.rotation_index = (self.rotation_index + 1) % self.num_rotations

    @staticmethod
    def bag() -> Generator["Piece", None, None]:
        """
        7-bag randomiser: yields each of the 7 pieces in shuffled order,
        then refills. Call repeatedly to get the next piece.

        Usage:
            gen = Piece.bag()
            piece = next(gen)
        """
        while True:
            bag = PIECE_NAMES[:]
            random.shuffle(bag)
            for name in bag:
                yield Piece(name)