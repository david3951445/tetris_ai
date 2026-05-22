from dataclasses import dataclass
import random
from collections.abc import Generator

from .coord import Coord

# Each piece: list of rotation states, each state is list of Coord offsets
TETROMINOES: dict[str, list[list[Coord]]] = {
    "I": [
        [Coord(0, 0), Coord(0, 1), Coord(0, 2), Coord(0, 3)],
        [Coord(0, 0), Coord(1, 0), Coord(2, 0), Coord(3, 0)],
    ],
    "O": [
        [Coord(0, 0), Coord(0, 1), Coord(1, 0), Coord(1, 1)],
    ],
    "T": [
        [Coord(0, 1), Coord(1, 0), Coord(1, 1), Coord(1, 2)],
        [Coord(0, 0), Coord(1, 0), Coord(2, 0), Coord(1, 1)],
        [Coord(1, 0), Coord(1, 1), Coord(1, 2), Coord(0, 1)],
        [Coord(0, 1), Coord(1, 1), Coord(2, 1), Coord(1, 0)],
    ],
    "S": [
        [Coord(0, 1), Coord(0, 2), Coord(1, 0), Coord(1, 1)],
        [Coord(0, 0), Coord(1, 0), Coord(1, 1), Coord(2, 1)],
    ],
    "Z": [
        [Coord(0, 0), Coord(0, 1), Coord(1, 1), Coord(1, 2)],
        [Coord(0, 1), Coord(1, 0), Coord(1, 1), Coord(2, 0)],
    ],
    "J": [
        [Coord(0, 0), Coord(1, 0), Coord(1, 1), Coord(1, 2)],
        [Coord(0, 0), Coord(0, 1), Coord(1, 0), Coord(2, 0)],
        [Coord(1, 0), Coord(1, 1), Coord(1, 2), Coord(2, 2)],
        [Coord(0, 1), Coord(1, 1), Coord(2, 0), Coord(2, 1)],
    ],
    "L": [
        [Coord(0, 2), Coord(1, 0), Coord(1, 1), Coord(1, 2)],
        [Coord(0, 0), Coord(1, 0), Coord(2, 0), Coord(2, 1)],
        [Coord(1, 0), Coord(1, 1), Coord(1, 2), Coord(2, 0)],
        [Coord(0, 0), Coord(0, 1), Coord(1, 1), Coord(2, 1)],
    ],
}

PIECE_NAMES = list(TETROMINOES.keys())


@dataclass(frozen=True, slots=True)
class Transform:
    pos: Coord
    rotation: int


class Piece:
    """Represents a single Tetromino with its type, rotation state, and position."""

    INITIAL_POS = Coord(0, 0)
    INITIAL_ROTATION = 0

    def __init__(self, name: str = None):
        self.name: str = name
        self.rotations: list[list[Coord]] = TETROMINOES[self.name]
        self.rotation_index: int = Piece.INITIAL_POS
        self.pos: Coord = Piece.INITIAL_ROTATION

    @property
    def cells(self) -> list[Coord]:
        """Current rotation's cells."""
        return self.pos + self.rotations[self.rotation_index]

    @property
    def num_rotations(self) -> int:
        return len(self.rotations)

    def get_cells_at(self, row: int, col: int, rotation: int) -> list[Coord]:
        """Return absolute Coord positions for a given placement."""
        origin = Coord(row, col)
        return [origin + offset for offset in self.rotations[rotation]]

    def transform(self, pos: Coord, rotation: int):
        self.pos = pos
        self.rotation_index = rotation

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
