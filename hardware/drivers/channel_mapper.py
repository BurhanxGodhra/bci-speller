"""
Reconciles a device's actual channel labels against the 10-20 positions our paradigms
need (config.yaml `required_channels`), using each device's declared fallback map for
electrodes it doesn't physically have.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChannelMapping:
    # Ordered list of required channel names (e.g. ["O1","Oz","O2","Cz","Pz"])
    required_channels: list[str]
    # For each required channel, the index into the RAW incoming data array to pull from
    source_indices: list[int]
    # Which required channels were satisfied via fallback substitution rather than an exact match
    fallback_used: dict[str, str]

    def apply(self, raw_samples: np.ndarray) -> np.ndarray:
        """
        raw_samples: shape (n_source_channels, n_timepoints) as delivered by the device.
        Returns: shape (len(required_channels), n_timepoints), reordered/subsetted.
        """
        return raw_samples[self.source_indices, :]


def build_channel_mapping(
    required_channels: list[str],
    device_channel_order: list[str],
    device_channel_fallback: dict[str, str],
) -> ChannelMapping:
    """
    required_channels: what the paradigms need, e.g. ["O1","Oz","O2","Cz","Pz"]
    device_channel_order: actual electrode order this device streams, e.g. Muse's
        ["TP9","AF7","AF8","TP10"]
    device_channel_fallback: config.yaml's declared substitutions for channels this
        device doesn't physically have, e.g. {"O1": "TP9", ...}
    """
    source_indices = []
    fallback_used = {}
    label_to_index = {label: i for i, label in enumerate(device_channel_order)}

    for req in required_channels:
        if req in label_to_index:
            source_indices.append(label_to_index[req])
            continue

        if req in device_channel_fallback:
            substitute = device_channel_fallback[req]
            if substitute not in label_to_index:
                raise ValueError(
                    f"config.yaml fallback for '{req}' points to '{substitute}', which is "
                    f"not in this device's channel_order {device_channel_order}. Fix config.yaml."
                )
            source_indices.append(label_to_index[substitute])
            fallback_used[req] = substitute
            logger.warning(
                f"Channel '{req}' not physically present on this device — "
                f"substituting nearest available electrode '{substitute}'. "
                f"Expect degraded SNR for analyses relying on '{req}'."
            )
            continue

        raise ValueError(
            f"Required channel '{req}' has no exact match and no fallback declared "
            f"in config.yaml for this device. Available channels: {device_channel_order}"
        )

    return ChannelMapping(
        required_channels=required_channels,
        source_indices=source_indices,
        fallback_used=fallback_used,
    )
