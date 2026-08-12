"""Classification metrics.

Uses scikit-learn when it is installed (the normal case) and falls back to
equivalent NumPy implementations otherwise, so the simulator and the test
suite run in minimal environments.
"""
from __future__ import annotations

import numpy as np

try:                                                     # pragma: no cover
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_curve,
    )
    from sklearn.model_selection import train_test_split as _sk_split
    HAS_SKLEARN = True
except Exception:                                        # pragma: no cover
    HAS_SKLEARN = False

    def accuracy_score(y, yhat):
        return float(np.mean(np.asarray(y) == np.asarray(yhat)))

    def _counts(y, yhat):
        y, yhat = np.asarray(y), np.asarray(yhat)
        tp = int(np.sum((y == 1) & (yhat == 1)))
        fp = int(np.sum((y == 0) & (yhat == 1)))
        fn = int(np.sum((y == 1) & (yhat == 0)))
        tn = int(np.sum((y == 0) & (yhat == 0)))
        return tn, fp, fn, tp

    def precision_score(y, yhat, zero_division=0):
        _, fp, _, tp = _counts(y, yhat)
        return tp / (tp + fp) if tp + fp else float(zero_division)

    def recall_score(y, yhat, zero_division=0):
        _, _, fn, tp = _counts(y, yhat)
        return tp / (tp + fn) if tp + fn else float(zero_division)

    def f1_score(y, yhat, zero_division=0):
        p = precision_score(y, yhat, zero_division)
        r = recall_score(y, yhat, zero_division)
        return 2 * p * r / (p + r) if p + r else float(zero_division)

    def confusion_matrix(y, yhat):
        tn, fp, fn, tp = _counts(y, yhat)
        return np.array([[tn, fp], [fn, tp]])

    def roc_curve(y, score):
        y = np.asarray(y)
        order = np.argsort(-np.asarray(score))
        y = y[order]
        tps = np.cumsum(y == 1)
        fps = np.cumsum(y == 0)
        P, N = max(1, int((y == 1).sum())), max(1, int((y == 0).sum()))
        tpr = np.concatenate([[0.0], tps / P])
        fpr = np.concatenate([[0.0], fps / N])
        return fpr, tpr, None

    def roc_auc_score(y, score):
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            return float("nan")
        ranks = np.empty(len(score), dtype=float)
        order = np.argsort(np.asarray(score))
        sorted_scores = np.asarray(score)[order]
        r, i = np.empty(len(score)), 0
        while i < len(sorted_scores):          # average ranks over ties
            j = i
            while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
                j += 1
            r[i:j + 1] = 0.5 * (i + j) + 1
            i = j + 1
        ranks[order] = r
        P, N = int((y == 1).sum()), int((y == 0).sum())
        return float((ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N))

    def _sk_split(X, y, test_size=0.25, stratify=None, random_state=0):
        rng = np.random.default_rng(random_state)
        idx_test = []
        groups = [np.where(y == c)[0] for c in np.unique(y)] if stratify is not None \
            else [np.arange(len(y))]
        for g in groups:
            g = g.copy()
            rng.shuffle(g)
            idx_test.extend(g[:int(round(test_size * len(g)))].tolist())
        mask = np.zeros(len(y), dtype=bool)
        mask[idx_test] = True
        return X[~mask], X[mask], y[~mask], y[mask]


def train_test_split(X, y, test_size=0.25, stratify=None, random_state=0):
    return _sk_split(X, y, test_size=test_size, stratify=stratify,
                     random_state=random_state)
