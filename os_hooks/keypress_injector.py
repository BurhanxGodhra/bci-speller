import pyautogui

pyautogui.PAUSE = 0.01


class KeypressInjector:
    def __init__(self, char_delay=0.02):
        self.char_delay = char_delay

    def type_char(self, char: str):
        pyautogui.write(char, interval=self.char_delay)

    def backspace(self):
        pyautogui.press("backspace")
