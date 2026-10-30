"""Leakage tests: preprocessing and training must depend on training rows only.

Integration tests use the genuine bundled breast cancer data; the heart fixture is a
synthetic schema fixture used only to exercise missing-value handling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import federated.server.runner as runner_mod
from federated.clients.attacks import AttackConfig
from federated.clients.hospital_client import HospitalClient
from federated.server.runner import ExperimentConfig, run_experiment
from federated.simulation.partition import partition_indices
from ml.datasets.loader import load_dataset, make_splits
from ml.datasets.registry import BREAST_CANCER, HEART
from ml.models.artifact import load_artifact
from ml.preprocessing.pipeline import (
    SchemaPreprocessor,
    combine_statistics,
    fit_preprocessor,
    local_statistics,
    transform,
)


def _sklearn_reference(spec, X):
    cats = [list(spec.feature(c).categories) for c in spec.categorical_features]
    ct = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="mean", keep_empty_features=True)),
                          ("s", StandardScaler())]), spec.numeric_features),
        *([("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                             ("o", OneHotEncoder(categories=cats, sparse_output=False))]),
            spec.categorical_features)] if cats else []),
    ])
    return ct.fit(X[spec.feature_names])


@pytest.mark.parametrize("which", ["heart_fixture", "breast_cancer"])
def test_preprocessor_matches_scikit_learn_fit_on_training_rows(which, heart_fixture_dir):
    if which == "heart_fixture":
        d = load_dataset("heart", heart_fixture_dir)
    else:
        d = load_dataset("breast_cancer", Path("data"))
    s = make_splits(d, seed=4)
    ours = fit_preprocessor(d.spec, s.X_train)
    ref = _sklearn_reference(d.spec, s.X_train)
    for X in (s.X_train, s.X_val, s.X_test):
        np.testing.assert_allclose(transform(ours, d.spec, X),
                                   ref.transform(X[d.spec.feature_names]), atol=1e-9)


def test_federated_statistics_equal_pooled_fit(heart_fixture_dir):
    d = load_dataset("heart", heart_fixture_dir)
    s = make_splits(d, seed=2)
    parts = np.array_split(np.arange(len(s.X_train)), 4)
    stats = [local_statistics(HEART, s.X_train.iloc[p]) for p in parts]
    fed = SchemaPreprocessor("heart").fit_statistics(combine_statistics(HEART, stats), "fed")
    pooled = fit_preprocessor(HEART, s.X_train)
    np.testing.assert_allclose(fed.means_, pooled.means_, rtol=1e-12)
    np.testing.assert_allclose(fed.scales_, pooled.scales_, rtol=1e-10)
    assert fed.modes_ == pooled.modes_


def test_validation_and_test_rows_do_not_change_fitted_preprocessor():
    d = load_dataset("breast_cancer", Path("data"))
    s = make_splits(d, seed=0)
    a = fit_preprocessor(BREAST_CANCER, s.X_train)
    b = fit_preprocessor(BREAST_CANCER, s.X_val)
    assert not np.allclose(a.means_, b.means_)  # fitting on other rows WOULD differ
    s.X_val.iloc[:, :] = s.X_val.to_numpy() * 100.0
    s.X_test.iloc[:, :] = s.X_test.to_numpy() * 100.0
    c = fit_preprocessor(BREAST_CANCER, s.X_train)
    np.testing.assert_array_equal(a.means_, c.means_)


def _client_training_rows(cfg: ExperimentConfig) -> pd.DataFrame:
    """Rebuild exactly the local training rows each simulated client holds."""
    d = load_dataset(cfg.dataset_key, Path("data"))
    s = make_splits(d, cfg.seed, cfg.test_fraction, cfg.val_fraction)
    parts = partition_indices(s.y_train, cfg.num_clients, cfg.partition, cfg.seed,
                              cfg.dirichlet_alpha)
    rows = []
    for k, idx in enumerate(parts, start=1):
        c = HospitalClient(f"client_{k}", k, d.spec, s.X_train.iloc[idx], s.y_train[idx],
                           cfg.seed, AttackConfig())
        rows.append(c._raw_train)
    return pd.concat(rows, ignore_index=True)


def test_runner_fits_preprocessing_on_client_training_rows_only(bc_experiment):
    rec = bc_experiment["record"]
    cfg = bc_experiment["config"]
    pp = rec["preprocessing"]
    assert pp["validation_and_test_used_for_fitting"] is False
    assert pp["n_fit_rows"] == sum(c["num_train"] for c in rec["clients"])
    art = load_artifact(bc_experiment["storage"] / rec["artifact_dir"], "breast_cancer")
    expected = fit_preprocessor(BREAST_CANCER, _client_training_rows(cfg))
    np.testing.assert_allclose(art.preprocessor.means_, expected.means_, rtol=1e-12)
    np.testing.assert_allclose(art.preprocessor.scales_, expected.scales_, rtol=1e-10)


def _corrupting_splits(corrupt_val: bool):
    original = runner_mod.make_splits

    def wrapper(*args, **kwargs):
        s = original(*args, **kwargs)
        s.X_test.iloc[:, :] = s.X_test.to_numpy() * 50.0 + 7.0
        if corrupt_val:
            s.X_val.iloc[:, :] = s.X_val.to_numpy() * 50.0 + 7.0
        return s
    return wrapper


def _final_params(tmp, cfg):
    rec = run_experiment(cfg, Path("data"), tmp)
    art = load_artifact(tmp / rec["artifact_dir"], cfg.dataset_key)
    return art.model.coef_.copy(), art.model.intercept_.copy()


@pytest.mark.parametrize("strategy,corrupt_val", [("fedavg", True), ("median", True),
                                                  ("trust", False)])
def test_trained_model_is_invariant_to_held_out_rows(tmp_path, monkeypatch, strategy, corrupt_val):
    """Corrupting test rows (and, for strategies that never read it, validation rows)
    must leave the trained model bit-for-bit unchanged."""
    cfg = ExperimentConfig(dataset_key="breast_cancer", num_clients=3, num_rounds=3,
                           strategy=strategy, malicious_clients=[], attack_type="none", seed=11,
                           centralized_baseline=False)
    coef_a, b_a = _final_params(tmp_path / "a", cfg)
    monkeypatch.setattr(runner_mod, "make_splits", _corrupting_splits(corrupt_val))
    coef_b, b_b = _final_params(tmp_path / "b", cfg)
    np.testing.assert_array_equal(coef_a, coef_b)
    np.testing.assert_array_equal(b_a, b_b)


def test_old_pipeline_artifacts_and_records_are_rejected(bc_experiment, tmp_path):
    import json
    import shutil

    from experiments.store import ExperimentStore, is_current
    from ml.models.artifact import ArtifactError

    rec = bc_experiment["record"]
    src = bc_experiment["storage"] / rec["artifact_dir"]
    old = tmp_path / "old_model"
    shutil.copytree(src, old)
    meta = json.loads((old / "model_meta.json").read_text())
    meta.pop("pipeline_version")
    (old / "model_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ArtifactError, match="earlier preprocessing pipeline"):
        load_artifact(old, "breast_cancer")
    stale = dict(rec, id="stale-1")
    stale.pop("pipeline_version")
    assert not is_current(stale) and is_current(rec)
    store = ExperimentStore(tmp_path)
    store.save(stale)
    assert store.latest("breast_cancer", with_model=False) is None  # stale is never "latest"
