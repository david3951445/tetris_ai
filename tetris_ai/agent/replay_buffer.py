import random
from collections import deque
from typing import Deque

import torch

# One transition in RL
Experience = tuple[
    list[float],  # state
    list[float],  # next_state
    float,  # reward
    bool,  # done
]


class ReplayBuffer:
    """
    Fixed-size replay buffer for DQN.

    Stores transitions:
        (state, next_state, reward, done)

    Purpose:
        - Break correlation between sequential samples
        - Enable mini-batch training
        - Improve sample efficiency
    """

    def __init__(self, capacity: int = 10_000) -> None:
        # deque with maxlen acts as circular buffer
        self.buffer: Deque[Experience] = deque(maxlen=capacity)

    # ------------------------------------------------------------
    # Store experience
    # ------------------------------------------------------------
    def push(
        self,
        state: list[float],
        next_state: list[float],
        reward: float,
        done: bool,
    ) -> None:
        """
        Add one transition into replay buffer.
        Oldest data will be automatically removed if full.
        """
        self.buffer.append((state, next_state, reward, done))

    # ------------------------------------------------------------
    # Sample mini-batch
    # ------------------------------------------------------------
    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch and convert to tensors.

        Returns:
            states:       [B, state_dim]
            next_states:  [B, state_dim]
            rewards:      [B, 1]
            dones:        [B, 1]
        """

        batch = random.sample(self.buffer, batch_size)

        # unzip transitions → group by field
        states, next_states, rewards, dones = zip(*batch)

        # convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return states_tensor, next_states_tensor, rewards_tensor, dones_tensor

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------
    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return len(self.buffer)
