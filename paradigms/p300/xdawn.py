from __future__ import annotations

from pyriemann.estimation import Xdawn
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class XdawnVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_filters: int = 4, estimator: str = "lwf"):
        self.n_filters = n_filters
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.xdawn_ = Xdawn(nfilter=self.n_filters, estimator=self.estimator)
        self.xdawn_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        filtered = self.xdawn_.transform(X)
        return filtered.reshape(filtered.shape[0], -1)
