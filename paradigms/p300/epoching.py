"""
Loads P300 speller data via MOABB and returns clean (X, y, meta) arrays.

MOABB's P300 paradigm handles dataset-specific epoch windowing, baseline correction,
and resampling for us, so this module stays dataset-agnostic — swap `dataset_name` for
any other MOABB P300 dataset without touching downstream xDAWN/LDA code.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Registry of supported MOABB P300 datasets. BNCI2014009 is the closest well-supported
# MOABB equivalent to the classic BCI Competition III Dataset II P300 speller paradigm
# (row/column flashing matrix). Add more here as needed (e.g. "bi2014a", "EPFLP300").
SUPPORTED_DATASETS = {"BNCI2014009"}


@dataclass
class P300Data:
    X: np.ndarray        # shape (n_epochs, n_channels, n_times)
    y: np.ndarray         # shape (n_epochs,) — 1 = Target, 0 = NonTarget
    channel_names: list[str]
    sample_rate: float
    subject_ids: list[int]


def load_p300_epochs(dataset_name: str = "BNCI2014009", subject_ids: list[int] = [1]) -> P300Data:
    """
    Downloads (on first call — cached afterward by MOABB in ~/mne_data) and epochs a
    MOABB P300 dataset for the given subjects.

    NOTE: first run requires internet access; MOABB caches datasets locally after that.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported: {SUPPORTED_DATASETS}. "
            f"MOABB has other P300 datasets (bi2014a, EPFLP300, etc) — add them to "
            f"SUPPORTED_DATASETS if you want to use one."
        )

    from moabb.datasets import BNCI2014009
    from moabb.paradigms import P300

    dataset_cls = {"BNCI2014009": BNCI2014009}[dataset_name]
    dataset = dataset_cls()
    paradigm = P300()

    logger.info(f"Loading {dataset_name} for subjects {subject_ids} via MOABB "
                f"(downloads on first run, cached after)...")
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject_ids)

    y = (labels == "Target").astype(int)
    logger.info(
        f"Loaded {X.shape[0]} epochs, {X.shape[1]} channels, {X.shape[2]} timepoints. "
        f"Class balance: {y.sum()} Target / {(y == 0).sum()} NonTarget "
        f"({100 * y.mean():.1f}% Target — P300 speller data is normally very imbalanced, "
        f"this is expected)."
    )

    channel_names = list(paradigm.match_all([dataset], channels=None)[1]) if False else []
    # Channel names aren't always cleanly exposed via the high-level paradigm API across
    # MOABB versions — fall back to generic labels if unavailable, callers that need real
    # 10-20 labels should cross-reference the HAL's channel_order in hardware/config.yaml.
    if not channel_names:
        channel_names = [f"ch{i}" for i in range(X.shape[1])]

    sample_rate = getattr(paradigm, "resample", None) or 250.0

    return P300Data(
        X=X, y=y, channel_names=channel_names, sample_rate=sample_rate, subject_ids=subject_ids
    )
