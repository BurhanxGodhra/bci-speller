from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = {"BNCI2014009"}


@dataclass
class P300Data:
    X: np.ndarray
    y: np.ndarray
    channel_names: list[str]
    sample_rate: float
    subject_ids: list[int]


def load_p300_epochs(dataset_name: str = "BNCI2014009", subject_ids: list[int] = [1],
                      channels: list[str] | None = None) -> P300Data:
    """
    If `channels` is given, restricts to whichever of those channel names actually
    exist in this dataset (logs a warning for any that don't) — used to align a
    pretrained "prior" model to the same channel subset a calibration wizard uses.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {SUPPORTED_DATASETS}")

    from moabb.datasets import BNCI2014009
    from moabb.paradigms import P300

    dataset = {"BNCI2014009": BNCI2014009}[dataset_name]()
    paradigm = P300()

    logger.info(f"Loading {dataset_name} for subjects {subject_ids} via MOABB...")

    if channels:
        epochs_obj, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject_ids, return_epochs=True)
        available = [ch for ch in channels if ch in epochs_obj.ch_names]
        missing = set(channels) - set(available)
        if missing:
            logger.warning(f"Dataset '{dataset_name}' missing channels {missing} — using {available}")
        if not available:
            raise ValueError(f"None of {channels} exist in {dataset_name} (has: {epochs_obj.ch_names})")
        epochs_obj.pick(available)
        X = epochs_obj.get_data()
        channel_names = available
        sample_rate = epochs_obj.info["sfreq"]
    else:
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject_ids)
        channel_names = [f"ch{i}" for i in range(X.shape[1])]
        sample_rate = getattr(paradigm, "resample", None) or 250.0

    y = (labels == "Target").astype(int)
    logger.info(f"Loaded {X.shape[0]} epochs, {X.shape[1]} channels. "
                f"{y.sum()} Target / {(y == 0).sum()} NonTarget")

    return P300Data(X=X, y=y, channel_names=channel_names, sample_rate=sample_rate, subject_ids=subject_ids)
