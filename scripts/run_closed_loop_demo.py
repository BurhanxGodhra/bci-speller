"""
Offline-replay -> online decoder -> OS-output demo.

Classifies pre-recorded Nakanishi2015 SSVEP trials one at a time (paced with
time.sleep to simulate trial timing) and types the predicted letter via
pyautogui. This is NOT a real-time closed loop -- there's no live headset
feeding the classifier. It demonstrates the decoder-to-OS-output path and
measures ITR on known-label data; a genuine closed loop would need a live
LSL stream and a causal (not filtfilt-based) decoder -- see
docs/ARCHITECTURE.md and the Roadmap in README.md.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import string
import time

from paradigms.ssvep.epoching import load_ssvep_epochs
from paradigms.ssvep.fbcca import FBCCAClassifier
from os_hooks.keypress_injector import KeypressInjector
from os_hooks.itr import itr_bits_per_min

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Nakanishi2015")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--sfreq", type=float, default=256.0)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--trial-seconds", type=float, default=4.0)
    args = parser.parse_args()

    X, y, freqs, meta = load_ssvep_epochs(args.dataset, args.subjects)
    letters = string.ascii_uppercase[:len(freqs)]
    classifier = FBCCAClassifier(freqs, args.sfreq)

    print("Click into a text editor now — typing starts in 5 seconds...")
    time.sleep(5)

    injector = KeypressInjector()
    correct = 0
    n = min(args.n_trials, len(X))

    for i in range(n):
        pred_idx = classifier.predict(X[i][None, ...])[0]
        injector.type_char(letters[pred_idx])
        correct += int(pred_idx == y[i])
        time.sleep(args.trial_seconds)

    accuracy = correct / n
    itr = itr_bits_per_min(len(freqs), accuracy, args.trial_seconds)

    print(f"\nSelections: {n}, Accuracy: {accuracy:.2%}")
    print(f"ITR: {itr:.2f} bits/min")
