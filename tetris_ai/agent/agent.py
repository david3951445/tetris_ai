import random

import torch
import torch.nn as nn
import torch.optim as optim

from .dqn import DQN
from .replay_buffer import ReplayBuffer

class TetrisAgent:
    """
    DQN agent for Tetris.

    Decision strategy:
      - Enumerate all legal next-states (provided by TetrisEnv).
      - Score each with the Q-network.
      - With probability epsilon, pick randomly (explore).
      - Otherwise, pick the highest-scoring state (exploit).

    Training:
      - Sample a batch from ReplayBuffer.
      - Compute TD target: r + γ * max Q(s', ·)  (0 if done).
      - Minimise MSE between prediction and target (online net vs target net).
      - Soft-copy online → target every `target_update_freq` steps.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 2000,
        batch_size: int = 512,
        buffer_capacity: int = 10_000,
        target_update_freq: int = 100,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.steps = 0

        self.online_net = DQN(input_dim, hidden_dim).to(self.device)
        self.target_net = DQN(input_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self, legal_states: list[dict]
    ) -> tuple[tuple[int, int], list[float]]:
        """
        legal_states: list of {"features": [...], "action": (col, rot)}
        Returns (action, chosen_state_features).
        """
        if random.random() < self.epsilon:
            chosen = random.choice(legal_states)
        else:
            features = torch.tensor(
                [s["features"] for s in legal_states],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                q_values = self.online_net(features).squeeze(1)
            chosen = legal_states[q_values.argmax().item()]

        return chosen["action"], chosen["features"]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def store(
        self, state: list[float], next_state: list[float], reward: float, done: bool
    ) -> None:
        self.replay_buffer.push(state, next_state, reward, done)

    def train_step(self) -> float | None:
        """Run one gradient update. Returns loss or None if buffer too small."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, next_states, rewards, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # Current Q estimates
        q_pred = self.online_net(states)

        # TD target
        with torch.no_grad():
            q_next = self.target_net(next_states)
        q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_epsilon()
        self._maybe_sync_target()
        self.steps += 1

        return loss.item()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_epsilon(self) -> None:
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
        )

    def _maybe_sync_target(self) -> None:
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())
