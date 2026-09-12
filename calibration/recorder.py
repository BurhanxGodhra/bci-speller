import threading
import numpy as np

from hardware.drivers.lsl_discovery import discover_eeg_stream, get_channel_labels
from hardware.drivers.channel_mapper import build_channel_mapping


class EEGRecorder:
    """Continuously records raw (unresampled) mapped channels + LSL timestamps
    in a background thread, for later epoch extraction against marker timestamps."""

    def __init__(self, config: dict):
        self.config = config
        self._stop = threading.Event()
        self._thread = None
        self.chunks = []
        self.timestamps = []
        self.mapping = None
        self.sample_rate = None

    def start(self):
        device_cfg = self.config["devices"][self.config["active_device"]]
        discovered = discover_eeg_stream(
            stream_type=self.config.get("lsl_stream_type", "EEG"),
            name_hint=device_cfg["lsl_name_hint"],
            timeout_sec=self.config.get("lsl_discovery_timeout_sec", 5.0),
        )
        live_labels = get_channel_labels(discovered)
        device_channel_order = live_labels if live_labels else device_cfg["channel_order"]
        self.mapping = build_channel_mapping(
            self.config["required_channels"], device_channel_order, device_cfg.get("channel_fallback", {})
        )
        self.sample_rate = discovered.native_sample_rate or device_cfg["native_sample_rate"]
        self.inlet = discovered.inlet

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            samples, timestamps = self.inlet.pull_chunk(timeout=0.5, max_samples=256)
            if samples:
                self.chunks.append(self.mapping.apply(np.array(samples).T))
                self.timestamps.extend(timestamps)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self.chunks:
            return np.concatenate(self.chunks, axis=1), np.array(self.timestamps)
        return np.zeros((len(self.config["required_channels"]), 0)), np.array([])
