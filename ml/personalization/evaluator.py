"""Global versus personalised (locally calibrated) evaluation on identical test rows.

All metrics reuse ``ml.evaluation.metrics.classification_metrics`` (the same definitions
used for every historical experiment) with the model's fixed decision threshold.

Edge cases are explicit, never imputed:
* ROC-AUC is ``None`` when a test partition contains one class (``auc_note`` says why).
* A hospital without a valid calibrator gets ``personalized = None`` and a status/reason;
  its global metrics are still reported, and it is excluded from personalised aggregates.
"""
from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from ml.evaluation.metrics import classification_metrics
from ml.personalization.calibration import HospitalCalibrator, stable_sigmoid

METRIC_KEYS = ("roc_auc", "f1", "precision", "recall", "accuracy", "brier", "log_loss", "ece")
LOWER_IS_BETTER = {"brier", "log_loss", "ece"}


def metrics(y: np.ndarray, p: np.ndarray, threshold: float, split: str) -> dict[str, Any]:
    m = classification_metrics(np.asarray(y).astype(int), np.asarray(p, dtype=float), threshold, split)
    out = {k: m[k] for k in METRIC_KEYS}
    out.update(n=m["n"], positives=m["positives"], negatives=m["negatives"],
               prevalence=m["prevalence"], threshold=m["threshold"],
               confusion_matrix=m["confusion_matrix"], mean_predicted=float(np.mean(p)))
    if m["roc_auc"] is None:
        out["auc_note"] = "ROC-AUC undefined: this test partition contains a single class"
    return out


def deltas(personal: dict[str, Any], global_: dict[str, Any]) -> dict[str, float | None]:
    return {k: (None if personal[k] is None or global_[k] is None else float(personal[k] - global_[k]))
            for k in METRIC_KEYS}


def compare_hospital(hospital_id: str, y_test: np.ndarray, z_test: np.ndarray, threshold: float,
                     calibrator: HospitalCalibrator | None, status: str, reason: str = "") -> dict[str, Any]:
    """Evaluate the frozen global model and (if available) its calibrated version."""
    y_test = np.asarray(y_test).astype(int)
    z_test = np.asarray(z_test, dtype=float)
    p_global = stable_sigmoid(z_test)
    out: dict[str, Any] = {"hospital_id": hospital_id, "status": status, "reason": reason,
                           "global": metrics(y_test, p_global, threshold, f"{hospital_id} local test")}
    if calibrator is not None:
        if calibrator.hospital_id != hospital_id:
            raise ValueError(f"Calibrator for {calibrator.hospital_id} applied to {hospital_id}")
        p_pers = calibrator.predict_proba(z_test)
        out["personalized"] = metrics(y_test, p_pers, threshold, f"{hospital_id} local test")
        out["delta"] = deltas(out["personalized"], out["global"])
        out["predictions"] = {"y": y_test.tolist(), "p_global": p_global.tolist(),
                              "p_personalized": p_pers.tolist()}
    else:
        out["personalized"], out["delta"] = None, None
        out["predictions"] = {"y": y_test.tolist(), "p_global": p_global.tolist(),
                              "p_personalized": None}
    return out


def _mean_sd(vals: list[float | None]) -> dict[str, Any]:
    v = [float(x) for x in vals if x is not None]
    if not v:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": statistics.fmean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "n": len(v)}


def aggregate(hospitals: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Pooled and across-hospital summaries.

    ``matched`` compares global and personalised predictions on exactly the same rows:
    the local test rows of hospitals that received a calibrator.
    """
    done = [h for h in hospitals if h["personalized"] is not None]
    out: dict[str, Any] = {"n_hospitals": len(hospitals), "n_personalized": len(done),
                           "not_personalized": [{"hospital_id": h["hospital_id"], "reason": h["reason"]}
                                                for h in hospitals if h["personalized"] is None]}
    y_all = np.concatenate([h["predictions"]["y"] for h in hospitals]).astype(int)
    pg_all = np.concatenate([h["predictions"]["p_global"] for h in hospitals])
    out["pooled_global_all_hospitals"] = metrics(y_all, pg_all, threshold, "pooled local test (all hospitals)")
    if done:
        y = np.concatenate([h["predictions"]["y"] for h in done]).astype(int)
        pg = np.concatenate([h["predictions"]["p_global"] for h in done])
        pp = np.concatenate([h["predictions"]["p_personalized"] for h in done])
        g = metrics(y, pg, threshold, "pooled local test (personalised hospitals)")
        p = metrics(y, pp, threshold, "pooled local test (personalised hospitals)")
        out["matched"] = {"global": g, "personalized": p, "delta": deltas(p, g),
                          "hospitals": [h["hospital_id"] for h in done]}
        out["across_hospitals"] = {
            k: {"global": _mean_sd([h["global"][k] for h in done]),
                "personalized": _mean_sd([h["personalized"][k] for h in done]),
                "delta": _mean_sd([h["delta"][k] for h in done])} for k in METRIC_KEYS}
    else:
        out["matched"], out["across_hospitals"] = None, None
    return out
