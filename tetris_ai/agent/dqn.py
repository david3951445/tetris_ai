import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Simple fully-connected Q-network.
    Input:  4 board features  [total_height, holes, bumpiness, lines_cleared]
    Output: scalar Q-value for that state

    The agent evaluates every legal next-state and picks the one with
    the highest Q-value (no discrete action head needed).
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → (batch, 1)"""
        return self.net(x)
