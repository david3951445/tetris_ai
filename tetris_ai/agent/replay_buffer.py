import random
from collections import deque
from typing import List, Tuple

import torch


Experience = Tuple[
    List[float],   # state features
    List[float],   # next state features
    float,         # reward
    bool,          # done
]


class ReplayBuffer:
    """
    Fixed-size circular buffer storing (state, next_state, reward, done) tuples.
    Samples random mini-batches for DQN training.
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: List[float], next_state: List[float], reward: float, done: bool) -> None:
        self.buffer.append((state, next_state, reward, done))

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, next_states, rewards, dones = zip(*batch)
        return (
            torch.tensor(states,      dtype=torch.float32, device=device),
            torch.tensor(next_states, dtype=torch.float32, device=device),
            torch.tensor(rewards,     dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(dones,       dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)
