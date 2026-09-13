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


def evaluate_pipeline(X, y, n_filters=4, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, aucs = [], []
    for train_idx, test_idx in skf.split(X, y):
        pipeline = build_xdawn_lda_pipeline(n_filters=n_filters)
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred = pipeline.predict(X[test_idx])
        y_score = pipeline.decision_function(X[test_idx])
        accs.append(accuracy_score(y[test_idx], y_pred))
        aucs.append(roc_auc_score(y[test_idx], y_score))
    return {
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "n_folds": n_splits,
    }
