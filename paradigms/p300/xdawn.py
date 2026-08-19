"""
xDAWN spatial filtering (Rivet et al., 2009).

xDAWN finds spatial filters that maximize the signal-to-signal-plus-noise ratio (SSNR)
specifically for the evoked-response class (Target flashes), unlike PCA/ICA which
maximize variance/independence without regard to which class carries the signal we
actually care about. This concentrates the P300 into a small number of virtual channels,
dramatically improving classifier SNR versus feeding in raw electrode channels.
"""

from __future__ import annotations

from pyriemann.estimation import Xdawn
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class XdawnVectorizer(BaseEstimator, TransformerMixin):
    """
    Wraps pyriemann's Xdawn spatial filter + flattens the filtered epochs into feature
    vectors, so it can slot directly into an sklearn Pipeline ahead of LDA.
    """

    def __init__(self, n_filters: int = 4, estimator: str = "lwf"):
        self.n_filters = n_filters
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.xdawn_ = Xdawn(nfilter=self.n_filters, estimator=self.estimator)
        self.xdawn_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        filtered = self.xdawn_.transform(X)  # shape (n_epochs, n_filters*n_classes, n_times)
        return filtered.reshape(filtered.shape[0], -1)  # flatten to (n_epochs, n_features)
