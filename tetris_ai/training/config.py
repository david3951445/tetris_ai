from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    board_rows: int = 20
    board_cols: int = 10

    # Agent
    input_dim: int = 4
    hidden_dim: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 2000      # steps until epsilon reaches minimum
    batch_size: int = 512
    buffer_capacity: int = 10_000
    target_update_freq: int = 100

    # Training
    num_episodes: int = 5000
    save_every: int = 500
    log_every: int = 50
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"
