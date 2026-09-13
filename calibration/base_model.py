from pathlib import Path
import joblib
import numpy as np

from paradigms.p300.epoching import load_p300_epochs
from paradigms.p300.classifier import build_xdawn_lda_pipeline

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "calibration" / "cache"
_CACHE_DIR.mkdir(exist_ok=True)


def build_or_load_prior_model(hal_channels, dataset_name="BNCI2014009", subject_ids=[1], n_filters=4):
    """Trains (or loads cached) a P300 prior model restricted to whichever of
    `hal_channels` actually exist in the public dataset. Returns a dict with the
    fitted pipeline, its own score distribution (for z-scoring at blend time),
    and the actual channel subset used."""
    cache_key = f"prior_{dataset_name}_{'-'.join(sorted(hal_channels))}.joblib"
    cache_path = _CACHE_DIR / cache_key
    if cache_path.exists():
        return joblib.load(cache_path)

    data = load_p300_epochs(dataset_name=dataset_name, subject_ids=subject_ids, channels=hal_channels)
    pipeline = build_xdawn_lda_pipeline(n_filters=n_filters)
    pipeline.fit(data.X, data.y)

    scores = pipeline.decision_function(data.X)
    bundle = {
        "pipeline": pipeline,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores) + 1e-8),
        "channels": data.channel_names,
        "n_times": int(data.X.shape[2]),
    }
    joblib.dump(bundle, cache_path)
    return bundle
