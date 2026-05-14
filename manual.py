import gymnasium as gym
from tetris_ai.core.tetris_env import TetrisEnv

env = TetrisEnv(render_mode="human")

obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # 隨機動作
    obs, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
