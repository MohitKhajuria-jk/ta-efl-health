"""Binary logistic regression trained with genuine mini-batch SGD.

Why not ``sklearn.linear_model.SGDClassifier``? Its ``partial_fit`` performs
per-sample updates, so a "batch size" setting would be cosmetic. This estimator
implements true mini-batch gradient descent on the L2-regularised log-loss and
follows the scikit-learn estimator API (``fit``, ``partial_fit``,
``predict_proba``, ``decision_function``, ``coef_``, ``intercept_``,
``classes_``), so it works with scikit-learn metrics and with
``shap.LinearExplainer``.

Federated parameter representation: ``[coef (d,), intercept (1,)]``. Because
the model is linear in its parameters and every client shares the same
preprocessing and feature order, a (weighted) average or coordinate-wise median
of these vectors is a well-defined model of the same family.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_loss(y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if sample_weight is None:
        return float(np.mean(losses))
    return float(np.sum(losses * sample_weight) / np.sum(sample_weight))


class MiniBatchLogisticRegression(ClassifierMixin, BaseEstimator):
    """L2-regularised logistic regression optimised by mini-batch SGD."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        epochs: int = 1,
        l2: float = 1e-4,
        class_weight: str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2 = l2
        self.class_weight = class_weight
        self.random_state = random_state

    # ------------------------------------------------------------------ params
    def init_params(self, n_features: int) -> "MiniBatchLogisticRegression":
        self.coef_ = np.zeros((1, n_features), dtype=np.float64)
        self.intercept_ = np.zeros(1, dtype=np.float64)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = n_features
        return self

    def get_params_arrays(self) -> list[np.ndarray]:
        check_is_fitted(self, ["coef_", "intercept_"])
        return [self.coef_.ravel().copy(), self.intercept_.copy()]

    def set_params_arrays(self, arrays: list[np.ndarray]) -> "MiniBatchLogisticRegression":
        if len(arrays) != 2:
            raise ValueError(f"Expected [coef, intercept], got {len(arrays)} arrays")
        coef = np.asarray(arrays[0], dtype=np.float64).ravel()
        intercept = np.asarray(arrays[1], dtype=np.float64).ravel()
        if intercept.shape != (1,):
            raise ValueError(f"Intercept must have shape (1,), got {intercept.shape}")
        if hasattr(self, "n_features_in_") and coef.shape[0] != self.n_features_in_:
            raise ValueError(
                f"Coefficient length {coef.shape[0]} != model features {self.n_features_in_}")
        self.coef_ = coef.reshape(1, -1).copy()
        self.intercept_ = intercept.copy()
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = coef.shape[0]
        return self

    # ---------------------------------------------------------------- training
    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.ones_like(y, dtype=np.float64)
        if self.class_weight != "balanced":
            raise ValueError("class_weight must be None or 'balanced'")
        n = len(y)
        w = np.ones(n, dtype=np.float64)
        for c in (0, 1):
            nc = int(np.sum(y == c))
            if nc > 0:
                w[y == c] = n / (2.0 * nc)
        return w

    def partial_fit(self, X: np.ndarray, y: np.ndarray, epochs: int | None = None,
                    rng: np.random.Generator | None = None) -> "MiniBatchLogisticRegression":
        """Run ``epochs`` passes of mini-batch SGD starting from the current parameters."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D with one row per label")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values; preprocess (impute) first")
        if not np.all(np.isin(y, (0.0, 1.0))):
            raise ValueError("Labels must be 0/1")
        if not hasattr(self, "coef_"):
            self.init_params(X.shape[1])
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, model expects {self.n_features_in_}")
        if self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("batch_size must be >= 1 and learning_rate > 0")
        rng = rng or np.random.default_rng(self.random_state)
        sw = self._sample_weights(y.astype(int))
        w = self.coef_.ravel().copy()
        b = float(self.intercept_[0])
        n = len(y)
        for _ in range(int(epochs if epochs is not None else self.epochs)):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                xb, yb, swb = X[idx], y[idx], sw[idx]
                p = _sigmoid(xb @ w + b)
                err = (p - yb) * swb
                m = len(idx)
                grad_w = xb.T @ err / m + self.l2 * w
                grad_b = float(np.sum(err) / m)
                w -= self.learning_rate * grad_w
                b -= self.learning_rate * grad_b
        if not (np.all(np.isfinite(w)) and np.isfinite(b)):
            raise FloatingPointError("Training diverged (non-finite parameters); "
                                     "reduce the learning rate")
        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MiniBatchLogisticRegression":
        X = np.asarray(X, dtype=np.float64)
        self.init_params(X.shape[1])
        return self.partial_fit(X, y)

    # --------------------------------------------------------------- inference
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Log-odds of the positive class."""
        check_is_fitted(self, ["coef_", "intercept_"])
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_.ravel() + self.intercept_[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return log_loss(np.asarray(y), self.predict_proba(X)[:, 1])
