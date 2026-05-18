from dataclasses import dataclass


@dataclass
class Config:
    """Central hyperparameter store. Pass a modified instance to Trainer to experiment."""

    # --- Environment ---
    board_row_count: int = 20  # standard Tetris height
    board_col_count: int = 10  # standard Tetris width

    # --- Network ---
    input_dim: int = 4  # number of state features fed to DQN
    hidden_dim: int = 64  # neurons per hidden layer

    # --- Optimiser ---
    lr: float = 1e-3  # Adam learning rate
    gamma: float = 0.99  # discount factor; high = value future rewards more

    # --- Exploration (ε-greedy) ---
    epsilon_start: float = 1.0  # fully random at the start
    epsilon_end: float = 0.01  # minimum randomness after decay
    epsilon_decay: int = 2000  # steps to decay from start → end (linear)

    # --- Replay buffer ---
    batch_size: int = 512  # samples per gradient update
    buffer_capacity: int = 10_000  # max experiences stored; old ones are discarded

    # --- Target network ---
    target_update_freq: int = 100  # sync online → target net every N steps

    # --- Training loop ---
    num_episodes: int = 5000  # total training episodes
    save_every: int = 500  # save checkpoint every N episodes
    log_every: int = 50  # print summary every N episodes
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"  # "cuda" to use GPU
