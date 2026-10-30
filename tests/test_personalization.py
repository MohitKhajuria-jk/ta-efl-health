"""Unit tests for hospital-specific probability calibration (ml/personalization)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)

from ml.personalization.calibration import (CalibrationConfig, CalibrationError, HospitalCalibrator,
                                            InsufficientCalibrationData, fit_calibrator, logit,
                                            stable_sigmoid)
from ml.personalization.evaluator import aggregate, compare_hospital, metrics
from ml.personalization.storage import PersonalizationStorageError, PersonalizationStore


def synthetic(n=400, a=0.5, b=-1.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2, n)
    y = (rng.uniform(size=n) < stable_sigmoid(a * z + b)).astype(int)
    return z, y


# 1. fitting -----------------------------------------------------------------
def test_fit_matches_unpenalised_logistic_regression_when_penalty_negligible():
    z, y = synthetic()
    cal = fit_calibrator("hospital_1", z, y, CalibrationConfig(regularization=1e-8))
    ref = LogisticRegression(C=1e12, tol=1e-12, max_iter=10000).fit(z[:, None], y)
    assert cal.a == pytest.approx(ref.coef_[0, 0], abs=1e-6)
    assert cal.b == pytest.approx(ref.intercept_[0], abs=1e-6)
    assert cal.converged and cal.calibration_log_loss_after < cal.calibration_log_loss_before


def test_penalty_shrinks_towards_identity_and_fades_with_sample_size():
    z, y = synthetic(n=60)
    weak = fit_calibrator("h", z, y, CalibrationConfig(regularization=1e-8))
    strong = fit_calibrator("h", z, y, CalibrationConfig(regularization=50.0))
    assert abs(strong.a - 1) + abs(strong.b) < abs(weak.a - 1) + abs(weak.b)


def test_well_calibrated_scores_give_near_identity():
    z, y = synthetic(n=20000, a=1.0, b=0.0, seed=3)
    cal = fit_calibrator("h", z, y)
    assert cal.a == pytest.approx(1.0, abs=0.05) and cal.b == pytest.approx(0.0, abs=0.05)


# 2. transformation ----------------------------------------------------------
def test_probability_transformation_and_adjustment():
    z, y = synthetic()
    cal = fit_calibrator("h", z, y)
    zt = np.array([-3.0, 0.0, 2.5])
    np.testing.assert_allclose(cal.predict_proba(zt), 1 / (1 + np.exp(-(cal.a * zt + cal.b))))
    np.testing.assert_allclose(cal.adjustment(zt), cal.transform_logits(zt) - zt)
    order = np.argsort(zt)
    assert np.all(np.diff(cal.predict_proba(zt)[order]) > 0)  # a > 0 keeps the ranking


# 3. numerical stability -----------------------------------------------------
def test_numerical_stability_extremes():
    big = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    p = stable_sigmoid(big)
    assert np.all(np.isfinite(p)) and p[0] == 0.0 and p[-1] == 1.0 and p[2] == 0.5
    assert np.all(np.isfinite(logit(np.array([0.0, 1e-300, 0.5, 1.0]))))
    zs = np.r_[np.linspace(-50, -1, 15), np.linspace(1, 50, 15)]  # perfectly separable
    ys = np.r_[np.zeros(15), np.ones(15)].astype(int)
    cal = fit_calibrator("h", zs, ys)
    assert np.isfinite(cal.a) and np.isfinite(cal.b) and cal.a > 0
    assert np.all(np.isfinite(cal.predict_proba(big)))


# 4. invalid inputs ----------------------------------------------------------
@pytest.mark.parametrize("z,y,msg", [
    ([0.1, np.nan] * 15, [0, 1] * 15, "NaN"),
    ([0.1] * 30, [0, 1] * 14, "Length mismatch"),
    ([0.1] * 30, [0, 2] * 15, "0 or 1"),
    ([[0.1]] * 30, [0, 1] * 15, "one-dimensional"),
    ([0.1] * 30, ["a", "b"] * 15, "numeric"),
])
def test_invalid_inputs_raise_clear_errors(z, y, msg):
    with pytest.raises(CalibrationError, match=msg):
        fit_calibrator("h", z, y)


@pytest.mark.parametrize("bad", [dict(regularization=0), dict(regularization=float("nan")),
                                 dict(min_samples=4, min_per_class=5), dict(min_samples=1)])
def test_invalid_config(bad):
    with pytest.raises(ValueError):
        CalibrationConfig(**bad)


def test_calibrator_rejects_invalid_parameters_and_logits():
    z, y = synthetic()
    d = fit_calibrator("h", z, y).to_dict()
    for change in ({"a": -0.5}, {"a": float("inf")}, {"converged": False}, {"format_version": 99}):
        with pytest.raises(CalibrationError):
            HospitalCalibrator.from_dict({**d, **change})
    with pytest.raises(CalibrationError):
        HospitalCalibrator.from_dict({k: v for k, v in d.items() if k != "b"})
    with pytest.raises(CalibrationError, match="NaN"):
        fit_calibrator("h", z, y).predict_proba([np.nan])


# 5./6. insufficient and single-class data ------------------------------------
def test_insufficient_samples():
    z, y = synthetic(n=19)
    with pytest.raises(InsufficientCalibrationData, match="at least 20"):
        fit_calibrator("h", z, y)
    z = np.linspace(-2, 2, 40)
    y = np.r_[np.ones(4), np.zeros(36)].astype(int)
    with pytest.raises(InsufficientCalibrationData, match="4 positive and 36 negative"):
        fit_calibrator("h", z, y)
    with pytest.raises(InsufficientCalibrationData):
        fit_calibrator("h", [], [])


def test_single_class_calibration_data():
    with pytest.raises(InsufficientCalibrationData, match="single class"):
        fit_calibrator("h", np.linspace(-1, 1, 50), np.zeros(50, dtype=int))


def test_reversing_slope_is_rejected_not_silently_used():
    z = np.linspace(-3, 3, 200)
    y = (z < 0).astype(int)  # global ranking is backwards at this hospital
    with pytest.raises(CalibrationError, match="non-positive slope"):
        fit_calibrator("h", z, y, CalibrationConfig(regularization=1e-6))


# 7. isolation ----------------------------------------------------------------
def test_hospital_calibrators_are_isolated():
    za, ya = synthetic(seed=1, b=-1.5)
    zb, yb = synthetic(seed=2, b=1.0)
    a1 = fit_calibrator("hospital_1", za, ya)
    b1 = fit_calibrator("hospital_2", zb, yb)
    b2 = fit_calibrator("hospital_2", zb[::-1] * 1.0, yb[::-1])  # same rows, other order
    a2 = fit_calibrator("hospital_1", za, ya)  # refit A after B changed
    assert a1 == a2 and (a1.a, a1.b) != (b1.a, b1.b)
    assert b1.a == pytest.approx(b2.a) and b1.b == pytest.approx(b2.b)
    with pytest.raises(ValueError, match="applied to hospital_1"):
        compare_hospital("hospital_1", ya, za, 0.5, b1, "personalized")


# 8. storage --------------------------------------------------------------------
def _payload(run_id, cal):
    from ml.datasets.registry import DatasetRegistry
    return {"format_version": 1, "calibrator": cal.to_dict(), "provenance": {
        "personalization_run_id": run_id, "hospital_id": cal.hospital_id, "global_experiment_id": "g1",
        "global_artifact_sha256": "abc", "dataset_key": "heart",
        "schema_hash": DatasetRegistry.get("heart").schema_hash(), "pipeline_version": 2}}


def test_save_and_load_calibrators_with_validation(tmp_path):
    z, y = synthetic()
    cal = fit_calibrator("hospital_1", z, y)
    store = PersonalizationStore(tmp_path)
    rec = {"id": "pers-test-1", "kind": "personalization", "created": "2026-01-01"}
    store.save_run(rec, {"hospital_1": _payload("pers-test-1", cal)})
    loaded, prov = store.load_calibrator("pers-test-1", "hospital_1", expected_global_id="g1",
                                         expected_artifact_sha256="abc")
    assert loaded == cal and prov["global_experiment_id"] == "g1"
    assert [r["id"] for r in store.list_records()] == ["pers-test-1"]
    with pytest.raises(PersonalizationStorageError, match="different global model"):
        store.load_calibrator("pers-test-1", "hospital_1", expected_global_id="other")
    with pytest.raises(PersonalizationStorageError, match="changed since calibration"):
        store.load_calibrator("pers-test-1", "hospital_1", expected_artifact_sha256="xyz")
    with pytest.raises(PersonalizationStorageError, match="no calibrator"):
        store.load_calibrator("pers-test-1", "hospital_2")
    with pytest.raises(FileExistsError):  # an existing run is never overwritten
        store.save_run(dict(rec), {"hospital_1": _payload("pers-test-1", cal)})
    path = tmp_path / "personalization" / "pers-test-1" / "calibrators" / "hospital_1.json"
    data = json.loads(path.read_text())
    data["calibrator"]["a"] = 5.0
    path.write_text(json.dumps(data))
    with pytest.raises(PersonalizationStorageError, match="modified"):
        store.load_calibrator("pers-test-1", "hospital_1")


@pytest.mark.parametrize("bad", ["../x", "a/b", "", "x" * 200])
def test_storage_rejects_unsafe_identifiers(tmp_path, bad):
    with pytest.raises(PersonalizationStorageError):
        PersonalizationStore(tmp_path).load_record(bad)


# 9. reproducibility ----------------------------------------------------------
def test_fitting_is_deterministic():
    z, y = synthetic()
    assert fit_calibrator("h", z, y).to_dict() == fit_calibrator("h", z, y).to_dict()


# 12. metrics -------------------------------------------------------------------
def test_metrics_match_scikit_learn_and_manual_ece():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 300)
    p = np.clip(rng.uniform(size=300) * 0.6 + 0.4 * y, 1e-6, 1 - 1e-6)
    m = metrics(y, p, 0.5, "t")
    pred = (p >= 0.5).astype(int)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["precision"] == pytest.approx(precision_score(y, pred))
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["brier"] == pytest.approx(brier_score_loss(y, p))
    assert m["log_loss"] == pytest.approx(log_loss(y, p), rel=1e-6)
    bins = np.clip(np.digitize(p, np.linspace(0, 1, 11)[1:-1]), 0, 9)
    ece = sum((bins == b).sum() / len(p) * abs(p[bins == b].mean() - y[bins == b].mean())
              for b in range(10) if (bins == b).any())
    assert m["ece"] == pytest.approx(ece)


def test_single_class_test_partition_reports_undefined_auc():
    m = metrics(np.zeros(30, dtype=int), np.linspace(0.1, 0.4, 30), 0.5, "t")
    assert m["roc_auc"] is None and "single class" in m["auc_note"]


def test_within_hospital_auc_is_unchanged_and_aggregate_uses_identical_rows():
    z, y = synthetic(n=300, a=0.6, b=-0.8, seed=9)
    cal = fit_calibrator("hospital_1", z[:150], y[:150])
    h1 = compare_hospital("hospital_1", y[150:], z[150:], 0.5, cal, "personalized")
    assert h1["personalized"]["roc_auc"] == pytest.approx(h1["global"]["roc_auc"], abs=1e-12)
    assert h1["personalized"]["brier"] < h1["global"]["brier"]
    h2 = compare_hospital("hospital_2", y[:40], z[:40], 0.5, None, "insufficient_calibration_data", "few rows")
    agg = aggregate([h1, h2], 0.5)
    assert agg["n_personalized"] == 1 and agg["not_personalized"][0]["hospital_id"] == "hospital_2"
    assert agg["matched"]["global"]["n"] == agg["matched"]["personalized"]["n"] == 150
    assert agg["pooled_global_all_hospitals"]["n"] == 190
    assert h2["personalized"] is None and h2["delta"] is None
