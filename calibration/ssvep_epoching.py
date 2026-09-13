import numpy as np


def extract_ssvep_trials(eeg, eeg_ts, trials, settle_seconds=0.5):
    """trials: list of (freq_hz, t_start, t_end). Discards the first
    `settle_seconds` of each trial to let the steady-state response build up.
    Returns (segments, freqs) — segments may differ slightly in length."""
    segments, freqs = [], []
    for freq, t_start, t_end in trials:
        start_idx = np.searchsorted(eeg_ts, t_start + settle_seconds)
        end_idx = np.searchsorted(eeg_ts, t_end)
        if end_idx <= start_idx:
            continue
        segments.append(eeg[:, start_idx:end_idx])
        freqs.append(freq)
    return segments, freqs
