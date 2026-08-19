"""
Harmonizes an incoming stream's native sample rate to the project's target_sample_rate
(config.yaml), so paradigm code never has to branch on which headset is connected.

Uses polyphase resampling (scipy.signal.resample_poly) rather than naive interpolation —
naive/linear resampling smears high-frequency content, which is exactly the content SSVEP
detection (Phase 4) depends on. Polyphase filtering applies a proper anti-aliasing
low-pass before decimating/interpolating, preserving frequency-domain integrity.
"""

from __future__ import annotations
import logging
from math import gcd

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


def compute_resample_ratio(native_hz: float, target_hz: float) -> tuple[int, int]:
    """Returns (up, down) integers for resample_poly, reduced to lowest terms."""
    # scipy wants integer up/down factors; round native_hz since some devices report
    # non-integer nominal rates (e.g. 256.0 is fine, but float LSL metadata can be messy).
    native_int = round(native_hz * 100)
    target_int = round(target_hz * 100)
    divisor = gcd(native_int, target_int)
    down = native_int // divisor
    up = target_int // divisor
    return up, down


def harmonize_sample_rate(
    samples: np.ndarray,
    native_hz: float,
    target_hz: float,
) -> np.ndarray:
    """
    samples: shape (n_channels, n_timepoints) at native_hz.
    Returns: shape (n_channels, n_timepoints_resampled) at target_hz.

    No-op (returns input unchanged) if native_hz == target_hz — avoids unnecessary
    filtering/precision loss when a device already matches the target rate.
    """
    if abs(native_hz - target_hz) < 1e-6:
        return samples

    up, down = compute_resample_ratio(native_hz, target_hz)
    logger.debug(
        f"Resampling {native_hz}Hz -> {target_hz}Hz via resample_poly(up={up}, down={down})"
    )
    return resample_poly(samples, up, down, axis=1)
