import numpy as np

class TetrisModel:
    def __init__(self, rows=20, cols=10):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        # TODO: 初始化方塊

    def reset(self):
        self.board[:] = 0
        # TODO: 重設方塊
        return self.get_state()

    def step(self, action):
        """action: 0=left, 1=right, 2=rotate, 3=drop"""
        # TODO: 移動邏輯
        reward = 0
        done = False
        return self.get_state(), reward, done

    def get_state(self):
        """Return numpy board for RL"""
        return self.board.copy()
