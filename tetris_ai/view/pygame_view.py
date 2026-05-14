import pygame

CELL_SIZE = 30

class PygameView:
    def __init__(self, model):
        self.model = model
        pygame.init()
        self.screen = pygame.display.set_mode(
            (model.cols * CELL_SIZE, model.rows * CELL_SIZE)
        )

    def draw(self):
        self.screen.fill((0, 0, 0))
        board = self.model.get_state()

        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if board[r, c] != 0:
                    pygame.draw.rect(
                        self.screen,
                        (0, 200, 200),
                        pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    )
        pygame.display.flip()
