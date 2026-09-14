import random
import pygame

GRID = [
    list("ABCDEF"),
    list("GHIJKL"),
    list("MNOPQR"),
    list("STUVWX"),
    list("YZ0123"),
    list("456789"),
]


class FlashScheduler:
    def __init__(self, n_rows=6, n_cols=6):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self._sequence = []

    def _reshuffle(self):
        seq = [(i, True) for i in range(self.n_rows)] + [(i, False) for i in range(self.n_cols)]
        random.shuffle(seq)
        self._sequence = seq

    def next_flash(self):
        if not self._sequence:
            self._reshuffle()
        return self._sequence.pop()


class SpellerMatrix:
    def __init__(self, screen, cell_size=80, origin=(50, 100)):
        self.screen = screen
        self.cell_size = cell_size
        self.origin = origin
        self.font = pygame.font.SysFont(None, 40)

    def draw(self, flash_row=None, flash_col=None):
        ox, oy = self.origin
        for r, row in enumerate(GRID):
            for c, char in enumerate(row):
                x, y = ox + c * self.cell_size, oy + r * self.cell_size
                flashing = flash_row == r or flash_col == c
                color = (255, 255, 255) if flashing else (60, 60, 70)
                pygame.draw.rect(self.screen, color, (x, y, self.cell_size - 4, self.cell_size - 4))
                text = self.font.render(char, True, (0, 0, 0) if flashing else (200, 200, 200))
                self.screen.blit(text, (x + 25, y + 20))
