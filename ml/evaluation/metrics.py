"""Evaluation metrics computed from genuine predictions on a named split."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.models.logistic import log_loss


def classification_metrics(y_true: np.ndarray, p_positive: np.ndarray, threshold: float,
                           split: str, max_roc_points: int = 200) -> dict[str, Any]:
    """Return metrics + ROC curve + confusion matrix as JSON-serialisable values.

    ROC-AUC is undefined when ``y_true`` has a single class; it is then reported
    as ``None`` instead of a made-up number.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p_positive, dtype=float)
    if len(y_true) != len(p) or len(y_true) == 0:
        raise ValueError("y_true and probabilities must be non-empty and equal length")
    if not np.all(np.isfinite(p)):
        raise ValueError("Probabilities contain non-finite values")
    y_pred = (p >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) == 2
    out: dict[str, Any] = {
        "split": split,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int(len(y_true) - y_true.sum()),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": log_loss(y_true, p),
        "roc_auc": float(roc_auc_score(y_true, p)) if two_classes else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "brier": float(brier_score_loss(y_true, p, pos_label=1)),
        **calibration_summary(y_true, p),
    }
    if two_classes:
        fpr, tpr, _ = roc_curve(y_true, p)
        if len(fpr) > max_roc_points:  # thin for storage; keeps the endpoints
            idx = np.unique(np.linspace(0, len(fpr) - 1, max_roc_points).round().astype(int))
            fpr, tpr = fpr[idx], tpr[idx]
        out["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    else:
        out["roc_curve"] = None
    return out


def auc_or_none(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2 or not np.all(np.isfinite(scores)):
        return None
    return float(roc_auc_score(y_true, scores))


def calibration_summary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Expected calibration error (equal-width bins) and the reliability-diagram bins.

    Used to check, rather than assume, whether model probabilities are calibrated.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_p, frac_pos = float(p[mask].mean()), float(y_true[mask].mean())
        ece += n / len(p) * abs(mean_p - frac_pos)
        bins.append({"lower": float(edges[b]), "upper": float(edges[b + 1]), "n": n,
                     "mean_predicted": mean_p, "observed_rate": frac_pos})
    return {"ece": float(ece), "calibration_bins": bins}
