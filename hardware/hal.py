"""
Hardware Abstraction Layer — the single entry point paradigm/UI code should import.

Usage:
    from hardware.hal import HAL
    hal = HAL.from_config("hardware/config/config.yaml")
    for chunk in hal.stream():
        # chunk: np.ndarray, shape (len(required_channels), n_new_samples), at target_sample_rate
        ...
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from hardware.drivers.lsl_discovery import discover_eeg_stream, get_channel_labels
from hardware.drivers.channel_mapper import build_channel_mapping, ChannelMapping
from hardware.drivers.resampler import harmonize_sample_rate

logger = logging.getLogger(__name__)


@dataclass
class HAL:
    config: dict
    mapping: ChannelMapping
    inlet: object
    native_sample_rate: float
    target_sample_rate: float

    @classmethod
    def from_config(cls, config_path: str | Path) -> "HAL":
        config = yaml.safe_load(Path(config_path).read_text())

        active_device = config["active_device"]
        device_cfg = config["devices"][active_device]
        required_channels = config["required_channels"]
        target_sample_rate = config["target_sample_rate"]
        timeout = config.get("lsl_discovery_timeout_sec", 5.0)
        stream_type = config.get("lsl_stream_type", "EEG")

        logger.info(f"HAL initializing with active_device='{active_device}'")

        discovered = discover_eeg_stream(
            stream_type=stream_type,
            name_hint=device_cfg["lsl_name_hint"],
            timeout_sec=timeout,
        )

        # Prefer channel labels reported by the live stream; fall back to config.yaml's
        # declared channel_order if the source didn't populate LSL metadata.
        live_labels = get_channel_labels(discovered)
        device_channel_order = live_labels if live_labels else device_cfg["channel_order"]

        mapping = build_channel_mapping(
            required_channels=required_channels,
            device_channel_order=device_channel_order,
            device_channel_fallback=device_cfg.get("channel_fallback", {}),
        )

        if mapping.fallback_used:
            logger.warning(f"Active fallback substitutions in use: {mapping.fallback_used}")

        return cls(
            config=config,
            mapping=mapping,
            inlet=discovered.inlet,
            native_sample_rate=discovered.native_sample_rate or device_cfg["native_sample_rate"],
            target_sample_rate=target_sample_rate,
        )

    def pull_chunk(self, timeout: float = 1.0, max_samples: int = 1024) -> np.ndarray | None:
        """
        Pull whatever new samples are available, map to required channels, and harmonize
        sample rate. Returns None if no new data arrived within timeout.
        """
        samples, timestamps = self.inlet.pull_chunk(timeout=timeout, max_samples=max_samples)
        if not samples:
            return None

        raw = np.array(samples).T  # LSL gives (n_timepoints, n_channels); we want (n_channels, n_timepoints)
        mapped = self.mapping.apply(raw)
        harmonized = harmonize_sample_rate(mapped, self.native_sample_rate, self.target_sample_rate)
        return harmonized

    def stream(self, timeout: float = 1.0, max_samples: int = 1024):
        """Generator yielding harmonized chunks indefinitely. Ctrl+C to stop."""
        while True:
            chunk = self.pull_chunk(timeout=timeout, max_samples=max_samples)
            if chunk is not None:
                yield chunk
