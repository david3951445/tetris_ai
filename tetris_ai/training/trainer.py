import os

from environment.tetris_env import TetrisEnv
from agent.agent import TetrisAgent
from training.config import Config
from utils.logger import Logger


class Trainer:
    """
    Orchestrates the training loop:
      for each episode:
        reset env → get legal states
        loop: agent picks action → env steps → store experience → train
      log, checkpoint.
    """

    def __init__(self, config: Config):
        self.config = config
        self.environment = TetrisEnv(config.board_row_count, config.board_col_count)
        self.agent = TetrisAgent(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
            batch_size=config.batch_size,
            buffer_capacity=config.buffer_capacity,
            target_update_freq=config.target_update_freq,
            device=config.device,
        )
        self.logger = Logger()
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def run(self) -> None:
        for episode in range(1, self.config.num_episodes + 1):
            score, lines, losses = self._run_episode()

            self.logger.record(episode, score, lines, self.agent.epsilon, losses)

            if episode % self.config.log_every == 0:
                self.logger.print_summary(episode)

            if episode % self.config.save_every == 0:
                path = os.path.join(self.config.checkpoint_dir, f"ep{episode}.pt")
                self.agent.save(path)
                print(f"  ✓ Saved checkpoint: {path}")

    def _run_episode(self):
        legal_states = self.environment.reset()
        total_score = 0.0
        total_lines = 0
        losses: list[float] = []

        while True:
            action, state_features = self.agent.select_action(legal_states)
            next_legal_states, reward, done = self.environment.step(action)

            # Best next state features for TD target
            if next_legal_states and not done:
                import torch

                feats = torch.tensor(
                    [s["features"] for s in next_legal_states],
                    dtype=torch.float32,
                    device=self.agent.device,
                )
                with __import__("torch").no_grad():
                    best_idx = self.agent.online_net(feats).argmax().item()
                next_features = next_legal_states[best_idx]["features"]
            else:
                next_features = [0.0] * self.config.input_dim

            self.agent.store(state_features, next_features, reward, done)
            loss = self.agent.train_step()
            if loss is not None:
                losses.append(loss)

            total_score += reward
            total_lines += self.environment.total_lines
            legal_states = next_legal_states

            if done:
                break

        return total_score, self.environment.total_lines, losses
