from collections import deque
from typing import List


class Logger:
    """Tracks per-episode metrics and prints rolling summaries."""

    def __init__(self, window: int = 100):
        self.window = window
        self.scores: deque = deque(maxlen=window)
        self.lines: deque = deque(maxlen=window)
        self.history = []

    def record(self, episode: int, score: float, lines: int, epsilon: float, losses: List[float]) -> None:
        self.scores.append(score)
        self.lines.append(lines)
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        self.history.append(
            {"episode": episode, "score": score, "lines": lines, "epsilon": epsilon, "loss": avg_loss}
        )

    def print_summary(self, episode: int) -> None:
        avg_score = sum(self.scores) / len(self.scores)
        avg_lines = sum(self.lines) / len(self.lines)
        last = self.history[-1]
        print(
            f"Ep {episode:5d} | "
            f"score {avg_score:8.1f} | "
            f"lines {avg_lines:5.1f} | "
            f"ε {last['epsilon']:.3f} | "
            f"loss {last['loss']:.4f}"
        )
