from collections.abc import Generator
import numpy as np

from .board import Board
from .piece import Piece


class TetrisEnv:
    """
    High-level Tetris environment.

    Each step the agent picks a (column, rotation) pair.
    The environment drops the piece, clears lines, and returns
    (next_states, reward, done).

    next_states: list of feature vectors for every legal placement
                 of the *next* piece — the agent picks among these.
    """

    def __init__(self, row_count: int = 20, col_count: int = 10):
        self.board = Board(row_count, col_count)
        self._bag: Generator = Piece.bag()  # shared 7-bag across the game
        self.current_piece: Piece = None
        self.next_piece: Piece = None
        self.score: int = 0
        self.total_lines: int = 0

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self) -> list[dict]:
        self.board.reset()
        self.score = 0
        self.total_lines = 0
        self._bag = Piece.bag()  # fresh bag on each episode
        self.current_piece = next(self._bag)
        self.next_piece = next(self._bag)
        return self._get_legal_states(self.current_piece)

    def step(self, action: tuple[int, int]) -> tuple[list[dict], float, bool]:
        """
        action: (col, rotation) chosen by the agent.
        Returns (next_legal_states, reward, done).
        """
        col, rotation = action
        row = self.board.drop_row(self.current_piece, col, rotation)
        cells = self.current_piece.get_cells_at(row, col, rotation)
        self.board.place(cells)

        lines = self.board.clear_lines()
        self.total_lines += lines
        reward = self._compute_reward(lines)
        self.score += reward

        done = self.board.is_game_over()

        self.current_piece = self.next_piece
        self.next_piece = next(self._bag)
        next_states = self._get_legal_states(self.current_piece)

        return next_states, reward, done

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_legal_states(self, piece: Piece) -> list[dict]:
        """
        Enumerate all (col, rotation) placements for the piece.
        For each, return the board state features *after* that placement
        so the agent can compare them.
        """
        states = []
        for rotation in range(piece.num_rotations):
            for col in range(self.board.col_count):
                row = self.board.drop_row(piece, col, rotation)
                cells = piece.get_cells_at(row, col, rotation)
                if not self.board.is_valid(cells):
                    continue

                # Temporarily place the piece to get resulting features
                self.board.place(cells)
                lines = sum(1 for r in range(self.board.row_count) if self.board.grid[r].all())
                features = self.board.get_state_features(lines)
                # Undo placement
                for r, c in cells:
                    self.board.grid[r, c] = 0

                states.append({"features": features, "action": (col, rotation)})

        return states

    def _compute_reward(self, lines_cleared: int) -> float:
        heights = self.board.get_heights()
        return (
            lines_cleared**2 * 10
            - self.board.count_holes() * 1.5
            - self.board.get_bumpiness(heights) * 0.5
            - sum(heights) * 0.1
        )
