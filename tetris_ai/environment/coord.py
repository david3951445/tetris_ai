from __future__ import annotations
from typing import Iterator


class Coord:
    """Immutable (row, col) grid coordinate."""

    __slots__ = ("row", "col")

    def __init__(self, row: int, col: int) -> None:
        self.row = row
        self.col = col

    # Arithmetic
    def __add__(self, other: Coord) -> Coord:
        return Coord(self.row + other.row, self.col + other.col)

    def __sub__(self, other: Coord) -> Coord:
        return Coord(self.row - other.row, self.col - other.col)

    # Unpacking support: for r, c in cells
    def __iter__(self) -> Iterator[int]:
        yield self.row
        yield self.col

    # Equality & hashing (safe to use in sets/dicts)
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Coord):
            return self.row == other.row and self.col == other.col
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.row, self.col))

    def __repr__(self) -> str:
        return f"Coord({self.row}, {self.col})"