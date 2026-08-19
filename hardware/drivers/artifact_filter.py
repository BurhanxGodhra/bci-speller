import numpy as np


class ArtifactRejector:
    def __init__(self, amplitude_uv=100e-6, ptp_uv=150e-6, window_size=50, variance_uv2=400e-12):
        self.amplitude_uv = amplitude_uv
        self.ptp_uv = ptp_uv
        self.window_size = window_size
        self.variance_uv2 = variance_uv2

    def is_clean(self, chunk: np.ndarray) -> bool:
        if np.any(np.abs(chunk) > self.amplitude_uv):
            return False
        if np.any(chunk.max(axis=1) - chunk.min(axis=1) > self.ptp_uv):
            return False
        if chunk.shape[1] >= self.window_size:
            windows = np.lib.stride_tricks.sliding_window_view(chunk, self.window_size, axis=1)
            if np.any(windows.var(axis=2) > self.variance_uv2):
                return False
        return True
