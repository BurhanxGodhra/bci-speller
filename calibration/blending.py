import numpy as np
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from paradigms.p300.classifier import build_xdawn_lda_pipeline


def personal_weight(n_target_epochs, ramp_to=15):
    """Weight given to the personal model, ramping 0 -> 1 as target-class
    calibration epochs accumulate. Below ramp_to, the prior model fills the gap."""
    return min(1.0, n_target_epochs / ramp_to)


def _resample_to(X, target_n_times):
    if X.shape[-1] == target_n_times:
        return X
    return resample(X, target_n_times, axis=-1)


class BlendedClassifier:
    def __init__(self, prior_bundle, channel_indices):
        self.prior_pipeline = prior_bundle["pipeline"]
        self.prior_mean = prior_bundle["score_mean"]
        self.prior_std = prior_bundle["score_std"]
        self.prior_n_times = prior_bundle["n_times"]
        self.channel_indices = channel_indices
        self.personal_pipeline = None
        self.personal_mean = 0.0
        self.personal_std = 1.0
        self.weight = 0.0

    def fit_personal(self, X, y, n_filters=3):
        self.personal_pipeline = build_xdawn_lda_pipeline(n_filters=n_filters)
        self.personal_pipeline.fit(X, y)
        scores = self.personal_pipeline.decision_function(X)
        self.personal_mean, self.personal_std = float(np.mean(scores)), float(np.std(scores) + 1e-8)
        self.weight = personal_weight(int(y.sum()))
        return self

    def decision_function(self, X):
        prior_input = _resample_to(X[:, self.channel_indices, :], self.prior_n_times)
        prior_scores = self.prior_pipeline.decision_function(prior_input)
        prior_z = prior_scores / self.prior_std

        personal_scores = self.personal_pipeline.decision_function(X)
        personal_z = personal_scores / self.personal_std

        return self.weight * personal_z + (1 - self.weight) * prior_z

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def evaluate_blended(X, y, prior_bundle, channel_indices, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    personal_accs, personal_aucs, blended_accs, blended_aucs, weights = [], [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        blended = BlendedClassifier(prior_bundle, channel_indices).fit_personal(X[train_idx], y[train_idx])

        p_pred = blended.personal_pipeline.predict(X[test_idx])
        p_score = blended.personal_pipeline.decision_function(X[test_idx])
        personal_accs.append(accuracy_score(y[test_idx], p_pred))
        personal_aucs.append(roc_auc_score(y[test_idx], p_score))

        b_pred = blended.predict(X[test_idx])
        b_score = blended.decision_function(X[test_idx])
        blended_accs.append(accuracy_score(y[test_idx], b_pred))
        blended_aucs.append(roc_auc_score(y[test_idx], b_score))
        weights.append(blended.weight)

    return {
        "personal_accuracy": float(np.mean(personal_accs)), "personal_auc": float(np.mean(personal_aucs)),
        "blended_accuracy": float(np.mean(blended_accs)), "blended_auc": float(np.mean(blended_aucs)),
        "weight_used": float(np.mean(weights)),
    }
