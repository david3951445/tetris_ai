import random
from collections import deque

import torch


Experience = tuple[
    list[float],   # state features
    list[float],   # next state features
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

    def push(self, state: list[float], next_state: list[float], reward: float, done: bool) -> None:
        self.buffer.append((state, next_state, reward, done))

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
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
