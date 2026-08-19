"""
Standalone mock EEG streamer. Publishes an LSL outlet exactly like a real headset would,
so the rest of the pipeline (HAL discovery, channel mapping, paradigms) can be developed
and tested with zero hardware attached.

Two modes:
  --source synthetic   : generates plausible EEG-like noise + injected alpha rhythm
  --source dataset      : replays a local .npy/.fif recording in real time (Phase 3/4 will
                           point this at BCI Competition III / Wang2016 files fetched via MOABB)

Run:
    python hardware/playback.py --source synthetic
"""

from __future__ import annotations
import argparse
import time
import logging

import numpy as np
import pylsl

logger = logging.getLogger(__name__)

CHANNELS = ["O1", "Oz", "O2", "Cz", "Pz"]
SAMPLE_RATE = 250  # Hz — must match config.yaml devices.mock_playback.native_sample_rate


def make_synthetic_chunk(n_samples: int, n_channels: int, t0: float, fs: float) -> np.ndarray:
    """Plausible-looking EEG: 1/f-ish pink-noise approximation + a mild 10Hz alpha rhythm."""
    t = t0 + np.arange(n_samples) / fs
    alpha = 8e-6 * np.sin(2 * np.pi * 10.0 * t)  # ~10Hz alpha, few microvolts
    noise = np.random.default_rng().normal(0, 5e-6, size=(n_channels, n_samples))
    return noise + alpha[np.newaxis, :]


def stream_synthetic(chunk_size: int = 25):
    info = pylsl.StreamInfo(
        name="MockPlayback",
        type="EEG",
        channel_count=len(CHANNELS),
        nominal_srate=SAMPLE_RATE,
        channel_format="float32",
        source_id="mock-playback-001",
    )
    chans = info.desc().append_child("channels")
    for label in CHANNELS:
        ch = chans.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")

    outlet = pylsl.StreamOutlet(info)
    logger.info(
        f"Publishing mock LSL stream 'MockPlayback' "
        f"({len(CHANNELS)}ch @ {SAMPLE_RATE}Hz). Ctrl+C to stop."
    )

    t = 0.0
    dt_chunk = chunk_size / SAMPLE_RATE
    try:
        while True:
            chunk = make_synthetic_chunk(chunk_size, len(CHANNELS), t, SAMPLE_RATE)
            outlet.push_chunk(chunk.T.tolist())  # LSL wants (n_timepoints, n_channels)
            t += dt_chunk
            time.sleep(dt_chunk)  # pace to real-time
    except KeyboardInterrupt:
        logger.info("Playback stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "dataset"], default="synthetic")
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()

    if args.source == "dataset":
        raise NotImplementedError(
            "Dataset playback wires into MOABB-fetched recordings"
        )

    stream_synthetic(chunk_size=args.chunk_size)
