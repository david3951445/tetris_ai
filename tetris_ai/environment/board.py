import numpy as np
from typing import List, Tuple

from .piece import Piece


class Board:
    """
    Manages the Tetris grid and all board-level logic:
    collision detection, line clearing, and state feature extraction.
    """

    def __init__(self, rows: int = 20, cols: int = 10):
        self.rows = rows
        self.cols = cols
        self.grid: np.ndarray = np.zeros((rows, cols), dtype=np.int8)

    def reset(self) -> None:
        self.grid[:] = 0

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def is_valid(self, cells: List[Tuple[int, int]]) -> bool:
        """Return True if all cells are within bounds and unoccupied."""
        for r, c in cells:
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                return False
            if self.grid[r, c]:
                return False
        return True

    def place(self, cells: List[Tuple[int, int]]) -> None:
        """Stamp piece cells onto the grid."""
        for r, c in cells:
            self.grid[r, c] = 1

    def drop_row(self, piece: Piece, col: int, rotation: int) -> int:
        """
        Find the lowest valid row for a piece at a given column and rotation.
        Returns the final row index.
        """
        row = 0
        while True:
            cells = piece.get_cells_at(row + 1, col, rotation)
            if not self.is_valid(cells):
                return row
            row += 1

    # ------------------------------------------------------------------
    # Line clearing
    # ------------------------------------------------------------------

    def clear_lines(self) -> int:
        """Remove full rows, shift everything down. Returns count cleared."""
        full_rows = [r for r in range(self.rows) if self.grid[r].all()]
        if not full_rows:
            return 0
        keep = [r for r in range(self.rows) if r not in full_rows]
        new_grid = np.zeros_like(self.grid)
        new_grid[self.rows - len(keep):] = self.grid[keep]
        self.grid = new_grid
        return len(full_rows)

    # ------------------------------------------------------------------
    # State features (used as neural network input)
    # ------------------------------------------------------------------

    def get_heights(self) -> List[int]:
        """Column heights (number of filled cells from bottom)."""
        heights = []
        for c in range(self.cols):
            col = self.grid[:, c]
            filled = np.where(col)[0]
            heights.append(self.rows - filled[0] if len(filled) else 0)
        return heights

    def count_holes(self) -> int:
        """Cells below the top filled cell in each column that are empty."""
        holes = 0
        for c in range(self.cols):
            col = self.grid[:, c]
            filled = np.where(col)[0]
            if len(filled):
                holes += int(col[filled[0]:].size - col[filled[0]:].sum())
        return holes

    def get_bumpiness(self, heights: List[int] = None) -> int:
        """Sum of absolute differences between adjacent column heights."""
        h = heights or self.get_heights()
        return sum(abs(h[i] - h[i+1]) for i in range(len(h) - 1))

    def get_state_features(self, lines_cleared: int = 0) -> List[float]:
        """
        4 scalar features fed to the DQN:
          [total_height, holes, bumpiness, lines_cleared]
        """
        heights = self.get_heights()
        return [
            float(sum(heights)),
            float(self.count_holes()),
            float(self.get_bumpiness(heights)),
            float(lines_cleared),
        ]

    def is_game_over(self) -> bool:
        """True if any cell in the top two rows is filled."""
        return bool(self.grid[:2].any())
