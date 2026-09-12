import random
import threading
import time
from pathlib import Path

import yaml

from calibration.lsl_markers import MarkerOutlet
from calibration.recorder import EEGRecorder

GRID = [list("ABCDEF"), list("GHIJKL"), list("MNOPQR"),
        list("STUVWX"), list("YZ0123"), list("456789")]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "hardware" / "config" / "config.yaml"


def find_target_position(letter):
    for r, row in enumerate(GRID):
        for c, ch in enumerate(row):
            if ch == letter:
                return r, c
    raise ValueError(f"'{letter}' not in grid")


def flash_schedule(n_rows=6, n_cols=6):
    seq = [("row", i) for i in range(n_rows)] + [("col", i) for i in range(n_cols)]
    random.shuffle(seq)
    return seq


class CalibrationRunner:
    """Runs flash scheduling + marker pushing + EEG recording in a background
    thread. The Streamlit page polls `active_row`/`active_col`/`is_done` across
    reruns rather than calling any UI code from this thread directly."""

    def __init__(self, target_letter, config_path=None, n_repetitions=10,
                 flash_ms=100, isi_ms=75):
        self.target_letter = target_letter
        self.config_path = config_path or _DEFAULT_CONFIG
        self.n_repetitions = n_repetitions
        self.flash_ms = flash_ms
        self.isi_ms = isi_ms

        self.active_row = None
        self.active_col = None
        self.is_done = False
        self.error = None
        self.result = None  # (eeg, eeg_ts, events, sample_rate)

        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            config = yaml.safe_load(open(self.config_path))
            target_row, target_col = find_target_position(self.target_letter)

            marker_outlet = MarkerOutlet()
            recorder = EEGRecorder(config)
            recorder.start()
            time.sleep(1.0)

            events = []
            for _ in range(self.n_repetitions):
                for kind, idx in flash_schedule():
                    row = idx if kind == "row" else None
                    col = idx if kind == "col" else None
                    is_target = row == target_row or col == target_col

                    t_marker = marker_outlet.push("flash")
                    self.active_row, self.active_col = row, col
                    events.append((t_marker, row, col, is_target))
                    time.sleep(self.flash_ms / 1000.0)

                    self.active_row, self.active_col = None, None
                    time.sleep(self.isi_ms / 1000.0)

            time.sleep(0.5)
            eeg, eeg_ts = recorder.stop()
            self.result = (eeg, eeg_ts, events, recorder.sample_rate)
        except Exception as e:
            self.error = str(e)
        finally:
            self.is_done = True
