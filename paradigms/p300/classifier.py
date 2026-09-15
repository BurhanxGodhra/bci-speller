from __future__ import annotations
import logging

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix

from paradigms.p300.xdawn import XdawnVectorizer

logger = logging.getLogger(__name__)


def build_xdawn_lda_pipeline(n_filters: int = 4):
    return make_pipeline(
        XdawnVectorizer(n_filters=n_filters),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def evaluate_pipeline(X, y, n_filters=4, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, aucs, bal_accs = [], [], []
    cm_total = np.zeros((2, 2), dtype=int)

    for train_idx, test_idx in skf.split(X, y):
        pipeline = build_xdawn_lda_pipeline(n_filters=n_filters)
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred = pipeline.predict(X[test_idx])
        y_score = pipeline.decision_function(X[test_idx])

        accs.append(accuracy_score(y[test_idx], y_pred))
        aucs.append(roc_auc_score(y[test_idx], y_score))
        bal_accs.append(balanced_accuracy_score(y[test_idx], y_pred))
        cm_total += confusion_matrix(y[test_idx], y_pred, labels=[0, 1])

    return {
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "balanced_accuracy_mean": float(np.mean(bal_accs)), "balanced_accuracy_std": float(np.std(bal_accs)),
        "confusion_matrix": cm_total.tolist(),  # summed across folds: [[TN, FP], [FN, TP]]
        "n_folds": n_splits,
    }
