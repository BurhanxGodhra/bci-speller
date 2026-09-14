import pygame


class SSVEPTarget:
    def __init__(self, label, freq_hz, rect, refresh_hz):
        self.label = label
        self.freq_hz = freq_hz
        self.rect = rect
        self.period_frames = max(1, round(refresh_hz / (2 * freq_hz)))
        self._frame_count = 0
        self._on = True

    def tick(self):
        self._frame_count += 1
        if self._frame_count >= self.period_frames:
            self._frame_count = 0
            self._on = not self._on

    def draw(self, screen, font):
        color = (255, 255, 255) if self._on else (30, 30, 30)
        pygame.draw.rect(screen, color, self.rect)
        text_color = (0, 0, 0) if self._on else (150, 150, 150)
        text = font.render(f"{self.label} ({self.freq_hz}Hz)", True, text_color)
        screen.blit(text, (self.rect[0] + 10, self.rect[1] + self.rect[3] // 2 - 10))
