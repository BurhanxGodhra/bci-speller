import numpy as np


def extract_epochs(eeg, eeg_ts, events, tmin=0.0, tmax=0.8, sample_rate=250):
    """events: list of (t_marker, row, col, is_target)"""
    n_samples = int((tmax - tmin) * sample_rate)
    X, y = [], []

    for t_marker, row, col, is_target in events:
        start_idx = np.searchsorted(eeg_ts, t_marker + tmin)
        end_idx = start_idx + n_samples
        if end_idx > eeg.shape[1]:
            continue
        X.append(eeg[:, start_idx:end_idx])
        y.append(int(is_target))

    return np.array(X), np.array(y)
