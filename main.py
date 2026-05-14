import pygame
from tetris_ai.core.tetris_model import TetrisModel
from tetris_ai.view.pygame_view import PygameView

def main():
    model = TetrisModel()
    view = PygameView(model)

    clock = pygame.time.Clock()
    running = True

    while running:
        # 處理事件（手動玩或 RL 控制）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # TODO: 這裡可以呼叫 model.step(action)
        view.draw()
        clock.tick(10)  # 控制遊戲速度

    pygame.quit()

if __name__ == "__main__":
    main()
