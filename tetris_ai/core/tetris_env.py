import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .tetris_model import TetrisModel


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(self, render_mode=None, rows=20, cols=10):
        super().__init__()
        self.model = TetrisModel(rows=rows, cols=cols)

        # 定義 action space: 0=left, 1=right, 2=rotate, 3=drop
        self.action_space = spaces.Discrete(4)

        # Observation: 棋盤 (rows x cols)，用 0/1 矩陣表示
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(rows, cols), dtype=np.int8
        )

        self.render_mode = render_mode
        self.viewer = None  # 用於 human render

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.model.reset()
        info = {}
        return obs, info

    def step(self, action):
        obs, reward, done = self.model.step(action)
        info = {}
        return obs, reward, done, False, info  # gymnasium: (obs, reward, terminated, truncated, info)

    def render(self):
        if self.render_mode == "ansi":
            return str(self.model.get_state())
        elif self.render_mode == "human":
            # lazy import，避免 pygame 成為強依賴
            from tetris_ai.view.pygame_view import PygameView
            if self.viewer is None:
                self.viewer = PygameView(self.model)
            self.viewer.draw()

    def close(self):
        if self.viewer:
            self.viewer = None
