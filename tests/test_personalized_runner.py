"""Integration tests: personalisation workflow on a genuine Flower-trained global model.

All writes go to temporary storage. The global model and experiment records are only read.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.store import ExperimentStore
from federated.personalized_runner import PersonalizationConfig, PersonalizationError, run_personalization
from ml.datasets.loader import load_dataset, make_splits
from ml.models.artifact import load_artifact
from ml.personalization.calibration import CalibrationConfig, fit_calibrator
from ml.personalization.storage import PersonalizationStore
from ml.prediction.service import predict
from ml.preprocessing.pipeline import transform


def tree_digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def storage(tmp_path, bc_experiment):
    """A private copy of the fixture's storage (experiment record + model artifact)."""
    dst = tmp_path / "storage"
    shutil.copytree(bc_experiment["storage"], dst)
    return dst


def cfg(rec, **kw):
    base = dict(global_experiment_id=rec["id"], num_hospitals=2, partition="iid")
    base.update(kw)
    return PersonalizationConfig(**base)


def test_runner_personalises_saves_and_leaves_global_model_untouched(storage, bc_experiment):
    rec = bc_experiment["record"]
    before_models = tree_digest(storage / "models")
    before_records = tree_digest(storage / "experiments")
    art_dir = storage / rec["artifact_dir"]
    artifact = load_artifact(art_dir, "breast_cancer")
    probe = {f: v for f, v in load_dataset("breast_cancer", Path("data")).X.iloc[0].items()}
    pred_before = predict(artifact, probe)

    out = run_personalization(cfg(rec), Path("data"), storage)

    assert tree_digest(storage / "models") == before_models          # 10. immutability
    assert tree_digest(storage / "experiments") == before_records    # history untouched
    assert out["integrity"]["global_model_unchanged"]
    assert out["integrity"]["global_artifact_sha256_before"] == out["integrity"]["global_artifact_sha256_after"]
    pred_after = predict(load_artifact(art_dir, "breast_cancer"), probe)  # 14. prediction unchanged
    assert pred_after.probability_positive == pred_before.probability_positive
    assert out["aggregate"]["n_personalized"] == 2
    store = PersonalizationStore(storage)
    assert store.load_record(out["id"])["id"] == out["id"]
    for h in out["hospitals"]:
        cal, prov = store.load_calibrator(out["id"], h["hospital_id"], expected_global_id=rec["id"],
                                          expected_artifact_sha256=out["integrity"]["global_artifact_sha256_before"]["model.joblib"])
        assert cal.to_dict() == h["calibrator"] and prov["hospital_id"] == h["hospital_id"]
        assert h["personalized"]["roc_auc"] == pytest.approx(h["global"]["roc_auc"], abs=1e-12)


def test_calibration_and_test_rows_are_disjoint_and_held_out(storage, bc_experiment):  # 11.
    rec = bc_experiment["record"]
    out = run_personalization(cfg(rec, num_hospitals=3, partition="dirichlet", dirichlet_alpha=5.0,
                                  min_hospital_rows=10), Path("data"), storage, save=False)
    data = load_dataset("breast_cancer", Path("data"))
    sc = rec["split_config"]
    sp = make_splits(data, sc["seed"], sc["test_fraction"], sc["validation_fraction"])
    key = lambda X: set(pd.util.hash_pandas_object(X.round(10), index=False))
    seen_train = key(sp.X_train) | key(sp.X_val)
    all_cal, all_test = set(), set()
    for h in out["hospitals"]:
        cal, test = set(h["rows"]["calibration_indices"]), set(h["rows"]["test_indices"])
        assert not cal & test
        assert not (cal | test) & (all_cal | all_test)  # hospitals are disjoint too
        all_cal |= cal
        all_test |= test
        rows = sp.X_test.iloc[sorted(cal | test)]
        assert not key(rows) & seen_train  # never a training/validation row
    assert all_cal | all_test == set(range(len(sp.y_test)))
    assert out["data_protocol"]["server_validation_split_used"] is False


def test_each_calibrator_uses_only_its_own_calibration_rows(storage, bc_experiment):  # 7.
    rec = bc_experiment["record"]
    out = run_personalization(cfg(rec), Path("data"), storage, save=False)
    art = load_artifact(storage / rec["artifact_dir"], "breast_cancer")
    data = load_dataset("breast_cancer", Path("data"))
    sc = rec["split_config"]
    sp = make_splits(data, sc["seed"], sc["test_fraction"], sc["validation_fraction"])
    z = art.model.decision_function(transform(art.preprocessor, data.spec, sp.X_test))
    for h in out["hospitals"]:
        idx = h["rows"]["calibration_indices"]
        own = fit_calibrator(h["hospital_id"], z[idx], sp.y_test[idx], CalibrationConfig())
        assert own.to_dict() == h["calibrator"]
    a, b = out["hospitals"]
    assert (a["calibrator"]["a"], a["calibrator"]["b"]) != (b["calibrator"]["a"], b["calibrator"]["b"])


def test_runner_is_reproducible(storage, bc_experiment):  # 9.
    rec = bc_experiment["record"]
    r1 = run_personalization(cfg(rec), Path("data"), storage, save=False)
    r2 = run_personalization(cfg(rec), Path("data"), storage, save=False)
    strip = lambda r: [{k: v for k, v in h.items()} for h in r["hospitals"]]
    assert strip(r1) == strip(r2)
    assert r1["integrity"]["partition_sha256"] == r2["integrity"]["partition_sha256"]
    assert r1["id"] != r2["id"]  # new identifier every run


def test_insufficient_data_is_reported_never_claimed_as_personalised(storage, bc_experiment):  # 5./6.
    rec = bc_experiment["record"]
    out = run_personalization(cfg(rec, calibration=CalibrationConfig(min_samples=500, min_per_class=5)),
                              Path("data"), storage)
    assert out["aggregate"]["n_personalized"] == 0 and out["aggregate"]["matched"] is None
    for h in out["hospitals"]:
        assert h["status"] == "insufficient_calibration_data" and "at least 500" in h["reason"]
        assert h["personalized"] is None and h["calibrator"] is None
    assert out["calibrator_files"] == {}


def test_runner_rejects_unknown_or_unusable_global_models(storage, bc_experiment):
    with pytest.raises(PersonalizationError, match="not found"):
        run_personalization(PersonalizationConfig("does-not-exist"), Path("data"), storage)
    legacy = dict(bc_experiment["record"], id="legacy-run", pipeline_version=1)
    ExperimentStore(storage).save(legacy)
    with pytest.raises(PersonalizationError, match="current-pipeline"):
        run_personalization(PersonalizationConfig("legacy-run"), Path("data"), storage, save=False)
    with pytest.raises(PersonalizationError, match="hospitals"):
        run_personalization(cfg(bc_experiment["record"], num_hospitals=1), Path("data"), storage)


# 13. personalised prediction API (and 14. the existing prediction endpoint) ---------------
@pytest.fixture
def web_personalized(tmp_path, bc_experiment):
    from fastapi.testclient import TestClient
    from api.main import create_app
    from app.configuration.settings import AppSettings, SettingsManager
    storage = tmp_path / "storage"
    rec = bc_experiment["record"]
    shutil.copytree(bc_experiment["storage"] / rec["artifact_dir"], storage / rec["artifact_dir"])
    ExperimentStore(storage).save(rec)
    run = run_personalization(cfg(rec), Path("data"), storage)
    manager = SettingsManager(tmp_path / "settings.json")
    manager.save(AppSettings(storage_dir=str(storage), data_dir=str(Path("data").resolve())))
    with TestClient(create_app(manager.path), base_url="http://127.0.0.1") as client:
        r = client.post("/api/auth/register", json={"username": "webtester", "password": "test-only-password"})
        client.headers["X-CSRF-Token"] = r.json()["csrf"]
        yield client, rec, run, storage


def _values():
    row = load_dataset("breast_cancer", Path("data")).X.iloc[3]
    return {k: float(v) for k, v in row.items()}


def test_personalisation_api_lists_runs_and_predicts(web_personalized):
    client, rec, run, _ = web_personalized
    listed = client.get("/api/personalization/experiments", params={"global_experiment_id": rec["id"]}).json()
    assert [r["id"] for r in listed] == [run["id"]] and listed[0]["n_personalized"] == 2
    detail = client.get(f"/api/personalization/experiments/{run['id']}").json()
    assert "predictions" not in detail["hospitals"][0] and detail["integrity"]["global_model_unchanged"]
    body = {"experiment_id": rec["id"], "personalization_id": run["id"], "hospital_id": "hospital_1",
            "values": _values(), "explain": True}
    out = client.post("/api/personalization/predictions", json=body)
    assert out.status_code == 200, out.text
    out = out.json()
    h = next(x for x in run["hospitals"] if x["hospital_id"] == "hospital_1")["calibrator"]
    z = out["global"]["log_odds"]
    assert out["personalized"]["probability_positive"] == pytest.approx(1 / (1 + np.exp(-(h["a"] * z + h["b"]))))
    assert out["calibration"]["log_odds_adjustment"] == pytest.approx((h["a"] - 1) * z + h["b"])
    plain = client.post("/api/predictions", json={"experiment_id": rec["id"], "values": _values(), "explain": True}).json()
    assert plain["probability_positive"] == out["global"]["probability_positive"]  # existing endpoint unchanged
    assert plain["explanation"] == out["explanation"]  # SHAP is the global model's, unmodified
    assert "global model" in out["explanation_scope"]


def test_personalisation_api_refuses_mismatches(web_personalized):
    client, rec, run, storage = web_personalized
    base = {"experiment_id": rec["id"], "personalization_id": run["id"], "hospital_id": "hospital_1",
            "values": _values(), "explain": False}
    assert client.post("/api/personalization/predictions", json={**base, "hospital_id": "hospital_9"}).status_code == 409
    assert client.post("/api/personalization/predictions", json={**base, "personalization_id": "pers-missing"}).status_code == 404
    assert client.post("/api/personalization/predictions", json={**base, "experiment_id": "other-model"}).status_code == 409
    assert client.post("/api/personalization/predictions", json={**base, "hospital_id": "../x"}).status_code == 422
    assert client.get("/api/personalization/reports/..%2Fsecret").status_code in (400, 404)
    # if the global model file changes, the calibrator is refused (provenance check)
    (storage / rec["artifact_dir"] / "model.joblib").open("ab").write(b"\0")
    assert client.post("/api/personalization/predictions", json=base).status_code == 409
    client.headers.pop("X-CSRF-Token")
    assert client.post("/api/personalization/predictions", json=base).status_code == 403
