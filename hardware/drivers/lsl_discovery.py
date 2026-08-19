"""
LSL stream auto-discovery.

Wraps pylsl.resolve_byprop so paradigm/UI code never needs to know about stream names,
ports, or host IPs — it just asks the HAL for "the EEG stream" and gets an inlet.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

import pylsl

logger = logging.getLogger(__name__)


class NoStreamFoundError(RuntimeError):
    """Raised when no matching LSL stream is found within the discovery timeout."""


@dataclass
class DiscoveredStream:
    inlet: pylsl.StreamInlet
    name: str
    native_sample_rate: float
    channel_count: int
    source_id: str


def discover_eeg_stream(
    stream_type: str = "EEG",
    name_hint: str | None = None,
    timeout_sec: float = 5.0,
) -> DiscoveredStream:
    """
    Discover an active LSL EEG stream.

    Primary strategy: resolve_byprop('type', stream_type) — this is LSL's canonical
    device-agnostic discovery mechanism; any correctly-configured EEG source (OpenBCI's
    LSL bridge, muse-lsl, BrainFlow's LSL outlet, or our own mock_playback) announces
    itself this way regardless of host/port.

    Fallback: if multiple streams of the right type are found (e.g. you're running the
    mock playback AND a real headset simultaneously), we disambiguate using name_hint
    (a substring match against the declared device name in config.yaml).

    Raises NoStreamFoundError if nothing is found within timeout_sec.
    """
    logger.info(f"Resolving LSL streams of type='{stream_type}' (timeout={timeout_sec}s)...")
    streams = pylsl.resolve_byprop("type", stream_type, timeout=timeout_sec)

    if not streams:
        raise NoStreamFoundError(
            f"No LSL stream of type '{stream_type}' found within {timeout_sec}s. "
            f"Is your headset's LSL bridge running? For development without hardware, "
            f"start hardware/playback.py first."
        )

    chosen = streams[0]
    if len(streams) > 1:
        logger.warning(f"Found {len(streams)} matching streams — disambiguating.")
        if name_hint:
            matches = [s for s in streams if name_hint.lower() in s.name().lower()]
            if matches:
                chosen = matches[0]
                logger.info(f"Selected stream '{chosen.name()}' via name_hint='{name_hint}'.")
            else:
                logger.warning(
                    f"No stream matched name_hint='{name_hint}'; defaulting to first "
                    f"result '{chosen.name()}'. Check config.yaml `active_device`."
                )
        else:
            logger.warning(f"No name_hint given; defaulting to first result '{chosen.name()}'.")

    inlet = pylsl.StreamInlet(chosen, max_buflen=360, recover=True)
    info = inlet.info()

    result = DiscoveredStream(
        inlet=inlet,
        name=info.name(),
        native_sample_rate=info.nominal_srate(),
        channel_count=info.channel_count(),
        source_id=info.source_id(),
    )
    logger.info(
        f"Connected to stream '{result.name}' "
        f"({result.channel_count}ch @ {result.native_sample_rate}Hz, source_id={result.source_id})"
    )
    return result


def get_channel_labels(discovered: DiscoveredStream) -> list[str]:
    """Extract channel labels from stream metadata, if the source populated them."""
    info = discovered.inlet.info()
    labels = []
    ch = info.desc().child("channels").child("channel")
    while not ch.empty():
        label = ch.child_value("label")
        labels.append(label if label else f"ch{len(labels)}")
        ch = ch.next_sibling()

    if not labels:
        logger.warning(
            "Stream did not populate channel labels in metadata. "
            "Falling back to config.yaml `channel_order` for this device."
        )
    return labels
