"""Logistic regression in pure NumPy.

Weights are fully exposed so federated averaging is a simple vector
operation. Exact SHAP values for a linear model are computed in closed
form: phi_j(x) = w_j * (x_j - E[x_j])  (in log-odds space), which avoids
a heavy dependency on the shap library while remaining exact.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticModel:
    def __init__(self, n_features: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, n_features)
        self.b = 0.0

    # ---- parameter vector interface (for FL) ----
    def get_params(self) -> np.ndarray:
        return np.concatenate([self.w, [self.b]])

    def set_params(self, theta: np.ndarray) -> None:
        self.w = theta[:-1].copy()
        self.b = float(theta[-1])

    # ---- training ----
    def fit_epochs(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 5,
        lr: float = 0.1, batch_size: int = 32, l2: float = 1e-3,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        n = len(y)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                Xb, yb = X[idx], y[idx]
                p = sigmoid(Xb @ self.w + self.b)
                err = p - yb
                gw = Xb.T @ err / len(idx) + l2 * self.w
                gb = err.mean()
                self.w -= lr * gw
                self.b -= lr * gb

    # ---- inference ----
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, thr: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= thr).astype(int)

    # ---- explainability ----
    def shap_values(self, X: np.ndarray, background_mean: np.ndarray) -> np.ndarray:
        """Exact SHAP values (log-odds space) for each sample and feature."""
        return self.w * (X - background_mean)

    def shap_signature(self, X_ref: np.ndarray, background_mean: np.ndarray) -> np.ndarray:
        """Client 'explanation signature': signed global attribution per
        feature on a shared reference set — the mean SHAP value among samples
        with above-background feature values, which for a linear model equals
        w_j * sd_j(ref). Signed (not |.|) so that a label-flipped model,
        whose attributions all point the wrong way, is detectably anomalous.
        L2-normalized; clients report only this vector to the server."""
        phi = self.w * X_ref.std(axis=0)
        norm = np.linalg.norm(phi) + 1e-12
        return phi / norm


def evaluate(model: LogisticModel, X: np.ndarray, y: np.ndarray) -> dict:
    from .metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_curve,
    )
    p = model.predict_proba(X)
    yhat = (p >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        "confusion": confusion_matrix(y, yhat),
        "roc": (fpr, tpr),
    }
