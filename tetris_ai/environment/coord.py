from dataclasses import dataclass
from collections.abc import Iterator


@dataclass(frozen=True, slots=True)
class Coord:
    row: int
    col: int

    def __add__(self, other: "Coord") -> "Coord":
        return Coord(self.row + other.row, self.col + other.col)

    def __sub__(self, other: "Coord") -> "Coord":
        return Coord(self.row - other.row, self.col - other.col)

    def __iter__(self) -> Iterator[int]:
        yield self.row
        yield self.col
