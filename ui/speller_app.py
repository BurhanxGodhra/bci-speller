import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame

from ui.matrix import SpellerMatrix, FlashScheduler
from ui.ssvep_targets import SSVEPTarget
from llm.predictor import WordPredictor

REFRESH_HZ = 60
SSVEP_FREQS = [12.0, 15.0, 20.0, 30.0]  # safe divisors of 60Hz, confirmed in Phase 1


def run():
    pygame.init()
    screen = pygame.display.set_mode((900, 750))
    pygame.display.set_caption("BCI Speller")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    matrix = SpellerMatrix(screen)
    scheduler = FlashScheduler()
    predictor = WordPredictor()

    targets = [
        SSVEPTarget("SPACE", SSVEP_FREQS[0], (650, 100, 150, 60), REFRESH_HZ),
        SSVEPTarget("BACK", SSVEP_FREQS[1], (650, 180, 150, 60), REFRESH_HZ),
        SSVEPTarget("SUGGEST1", SSVEP_FREQS[2], (650, 260, 150, 60), REFRESH_HZ),
        SSVEPTarget("SUGGEST2", SSVEP_FREQS[3], (650, 340, 150, 60), REFRESH_HZ),
    ]

    text_buffer = ""
    suggestions = []
    flash_row = flash_col = None
    flash_timer = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    text_buffer = text_buffer[:-1]
                elif event.key == pygame.K_SPACE:
                    text_buffer += " "
                    suggestions = predictor.predict_next(text_buffer)
                elif event.unicode.isalnum():
                    text_buffer += event.unicode.upper()
                    suggestions = predictor.predict_next(text_buffer)
                elif event.key == pygame.K_1 and suggestions:
                    text_buffer += suggestions[0] + " "
                    suggestions = predictor.predict_next(text_buffer)

        flash_timer += clock.get_time()
        if flash_timer >= 150:
            flash_timer = 0
            idx, is_row = scheduler.next_flash()
            flash_row, flash_col = (idx, None) if is_row else (None, idx)

        screen.fill((20, 20, 25))
        matrix.draw(flash_row, flash_col)
        for t in targets:
            t.tick()
            t.draw(screen, font)

        screen.blit(font.render(text_buffer, True, (255, 255, 255)), (50, 50))
        sugg_text = "  ".join(f"[{i + 1}] {w}" for i, w in enumerate(suggestions))
        screen.blit(font.render(sugg_text, True, (150, 220, 150)), (50, 650))

        pygame.display.flip()
        clock.tick(REFRESH_HZ)

    pygame.quit()


if __name__ == "__main__":
    run()
