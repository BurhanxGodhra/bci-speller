"""
xDAWN + shrinkage-LDA classification pipeline, and stratified cross-validated evaluation.

Shrinkage LDA (Ledoit-Wolf, via solver='lsqr', shrinkage='auto') is used instead of plain
LDA because P300 speller datasets have relatively few epochs per subject relative to
feature dimensionality after xDAWN flattening — plain LDA's covariance estimate would
overfit in that regime; shrinkage regularizes it toward a well-conditioned target matrix.
"""

from __future__ import annotations
import logging

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from paradigms.p300.xdawn import XdawnVectorizer

logger = logging.getLogger(__name__)


def build_xdawn_lda_pipeline(n_filters: int = 4):
    return make_pipeline(
        XdawnVectorizer(n_filters=n_filters),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def evaluate_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    n_filters: int = 4,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Stratified K-fold cross-validation (stratified because P300 data is heavily
    class-imbalanced — plain K-fold could give folds with too few Target examples).
    Returns mean/std accuracy and ROC-AUC across folds. AUC is the more meaningful metric
    here given the imbalance; accuracy alone can look deceptively high.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, aucs = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        pipeline = build_xdawn_lda_pipeline(n_filters=n_filters)
        pipeline.fit(X[train_idx], y[train_idx])

        y_pred = pipeline.predict(X[test_idx])
        y_score = pipeline.decision_function(X[test_idx])

        acc = accuracy_score(y[test_idx], y_pred)
        auc = roc_auc_score(y[test_idx], y_score)
        accs.append(acc)
        aucs.append(auc)
        logger.info(f"Fold {fold_idx}: accuracy={acc:.3f}, AUC={auc:.3f}")

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n_folds": n_splits,
    }
