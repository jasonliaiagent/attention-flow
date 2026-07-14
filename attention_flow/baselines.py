"""Graph-blind baselines. The GNN only matters if it beats these.

- zero:        predict 0 (shocks are near mean-zero; MSE(0) = target variance)
- persistence: next week = last week
- ridge:       closed-form ridge regression on each node's last 14 daily lags,
               fit on train windows pooled across nodes and themes
"""

from __future__ import annotations

import numpy as np

RIDGE_LAGS = 14


def predict_zero(X: np.ndarray) -> np.ndarray:
    return np.zeros(X.shape[:2], dtype=np.float32)


def predict_persistence(X: np.ndarray) -> np.ndarray:
    return X[:, :, -7:].mean(axis=2)


class RidgeBaseline:
    def __init__(self, lam: float = 10.0):
        self.lam = lam
        self.w: np.ndarray | None = None

    @staticmethod
    def _features(X: np.ndarray) -> np.ndarray:
        return X[:, :, -RIDGE_LAGS:].reshape(-1, RIDGE_LAGS)

    def fit(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        f = self._features(X)[mask.reshape(-1)]
        t = y.reshape(-1)[mask.reshape(-1)]
        a = f.T @ f + self.lam * np.eye(RIDGE_LAGS)
        self.w = np.linalg.solve(a, f.T @ t)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._features(X) @ self.w).reshape(X.shape[:2]).astype(np.float32)
