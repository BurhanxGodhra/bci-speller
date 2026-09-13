import threading
import time
from pathlib import Path

import yaml

from calibration.lsl_markers import MarkerOutlet
from calibration.recorder import EEGRecorder

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "hardware" / "config" / "config.yaml"

SSVEP_TARGETS = [
    {"label": "SPACE", "freq_hz": 12.0},
    {"label": "BACK", "freq_hz": 15.0},
    {"label": "SUGGEST1", "freq_hz": 20.0},
    {"label": "SUGGEST2", "freq_hz": 30.0},
]


class SSVEPCalibrationRunner:
    """Runs a sequence of attend-to-this-target trials in a background thread,
    pushing an LSL marker at each trial's start/end. Unlike P300, this doesn't
    train anything — CCA/FBCCA are zero-shot. This validates whether the
    zero-shot classifier can actually detect this person's steady-state
    response on this headset, per target frequency."""

    def __init__(self, config_path=None, trial_seconds=4.0, n_repeats=1):
        self.config_path = config_path or _DEFAULT_CONFIG
        self.trial_seconds = trial_seconds
        self.n_repeats = n_repeats

        self.current_target_idx = None
        self.trial_index = 0
        self.total_trials = len(SSVEP_TARGETS) * n_repeats
        self.trial_start_time = None
        self.is_done = False
        self.error = None
        self.trials = None  # list of (freq_hz, t_start, t_end)
        self.eeg = None
        self.eeg_ts = None
        self.sample_rate = None

        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            config = yaml.safe_load(open(self.config_path))
            marker_outlet = MarkerOutlet(name="SSVEPCalibrationMarkers")
            recorder = EEGRecorder(config)
            recorder.start()
            time.sleep(1.0)

            trials = []
            order = list(range(len(SSVEP_TARGETS))) * self.n_repeats
            for i, idx in enumerate(order):
                self.trial_index = i
                self.current_target_idx = idx
                self.trial_start_time = time.time()
                freq = SSVEP_TARGETS[idx]["freq_hz"]
                t_start = marker_outlet.push(f"attend_{freq}")
                time.sleep(self.trial_seconds)
                t_end = marker_outlet.push("trial_end")
                trials.append((freq, t_start, t_end))
                self.current_target_idx = None
                time.sleep(1.0)

            time.sleep(0.5)
            eeg, eeg_ts = recorder.stop()
            self.eeg, self.eeg_ts, self.sample_rate = eeg, eeg_ts, recorder.sample_rate
            self.trials = trials
        except Exception as e:
            self.error = str(e)
        finally:
            self.is_done = True
