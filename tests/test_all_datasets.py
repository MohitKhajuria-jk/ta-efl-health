"""Dataset loading/validation for all five datasets.

* ``*_schema_*`` tests use SYNTHETIC schema fixtures to unit-test each loader rule.
* ``test_genuine_dataset_*`` tests use the REAL files and run whenever the file is
  present in ``data/`` (breast cancer is bundled; others are skipped with a reason).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.datasets.loader import DatasetError, load_dataset, make_splits, validate_and_clean
from ml.datasets.registry import DatasetRegistry
from tests.conftest import make_schema_fixture

CSV_KEYS = ["heart", "diabetes", "framingham", "stroke"]
DATA = Path("data")


def _fixture(tmp_path, key, **kw):
    spec = DatasetRegistry.get(key)
    make_schema_fixture(key, tmp_path / spec.file_name, **kw)
    return tmp_path, spec


@pytest.mark.parametrize("key", CSV_KEYS)
def test_schema_fixture_loads_with_correct_target_and_types(tmp_path, key):
    d, spec = _fixture(tmp_path, key)
    data = load_dataset(key, d)
    assert list(data.X.columns) == spec.feature_names
    assert set(np.unique(data.y)) == {0, 1}
    for f in spec.features:
        if f.kind in ("numeric", "binary"):
            assert data.X[f.name].dtype.kind == "f"
        else:
            assert set(data.X[f.name].dropna()) <= set(f.categories)


@pytest.mark.parametrize("key", CSV_KEYS)
def test_schema_fixture_missing_column_and_bad_target_rejected(tmp_path, key):
    d, spec = _fixture(tmp_path, key)
    raw = pd.read_csv(d / spec.file_name)
    with pytest.raises(DatasetError, match="Missing columns"):
        validate_and_clean(spec, raw.drop(columns=[spec.feature_names[0]]))
    bad = raw.copy()
    bad.loc[0, spec.target] = 2
    with pytest.raises(DatasetError, match="0/1"):
        validate_and_clean(spec, bad)


@pytest.mark.parametrize("key", CSV_KEYS)
def test_schema_fixture_splits_reproducible(tmp_path, key):
    d, _ = _fixture(tmp_path, key)
    data = load_dataset(key, d)
    a, b = make_splits(data, seed=5), make_splits(data, seed=5)
    pd.testing.assert_frame_equal(a.X_test, b.X_test)
    assert not make_splits(data, seed=6).X_test.equals(a.X_test)


def test_diabetes_zero_codes_become_missing(tmp_path):
    d, spec = _fixture(tmp_path, "diabetes")
    raw = pd.read_csv(d / spec.file_name)
    raw.loc[:4, "Insulin"] = 0
    raw.loc[:1, "Glucose"] = 0
    data = validate_and_clean(spec, raw)
    assert data.report.missing_codes_converted == {"Insulin": 5, "Glucose": 2}
    assert data.X["Pregnancies"].min() >= 0  # zero pregnancies is a real value, not missing


def test_stroke_rules(tmp_path):
    d, spec = _fixture(tmp_path, "stroke")
    raw = pd.read_csv(d / spec.file_name)
    assert "id" in raw.columns
    raw["bmi"] = raw["bmi"].astype(object)
    raw.loc[:2, "bmi"] = "N/A"
    raw.loc[3, "gender"] = "Other"
    raw.to_csv(d / spec.file_name, index=False)
    data = load_dataset("stroke", d)
    assert "id" not in data.X.columns
    assert data.X["bmi"].isna().sum() == 3
    assert "Other" in set(data.X["gender"])
    raw.loc[4, "smoking_status"] = "sometimes"
    with pytest.raises(DatasetError, match="smoking_status"):
        validate_and_clean(spec, raw)


def test_framingham_missing_values_kept_as_missing(tmp_path):
    d, spec = _fixture(tmp_path, "framingham")
    raw = pd.read_csv(d / spec.file_name)
    raw.loc[:9, "glucose"] = np.nan
    data = validate_and_clean(spec, raw)
    assert data.report.missing_per_feature["glucose"] == 10


def test_duplicates_and_imbalance_are_reported(tmp_path):
    d, spec = _fixture(tmp_path, "heart", n=200)
    raw = pd.read_csv(d / spec.file_name)
    raw[spec.target] = 0
    raw.loc[:9, spec.target] = 1  # 5 % positive
    raw = pd.concat([raw, raw.iloc[:3]], ignore_index=True)
    data = validate_and_clean(spec, raw)
    assert data.report.n_duplicates == 3
    assert any("imbalance" in w for w in data.report.warnings)


def test_too_few_records_rejected(tmp_path):
    d, spec = _fixture(tmp_path, "heart", n=30)
    with pytest.raises(DatasetError, match="at least 50"):
        load_dataset("heart", d)


# --------------------------------------------------------------- genuine files
@pytest.mark.parametrize("key", DatasetRegistry.keys())
def test_genuine_dataset_matches_documented_counts(key):
    spec = DatasetRegistry.get(key)
    if not DatasetRegistry(DATA).availability(key).available:
        pytest.skip(f"genuine {spec.file_name} not present in data/ (see data/README.md)")
    r = load_dataset(key, DATA).report
    assert r.n_rows_raw == spec.expected_rows
    assert r.positives == spec.expected_positives


@pytest.mark.parametrize("key", DatasetRegistry.keys())
def test_genuine_dataset_federated_predict_explain(tmp_path, key):
    """Short genuine experiment -> load artifact -> predict a test row -> SHAP."""
    from federated.server.runner import ExperimentConfig, run_experiment
    from ml.explainability.shap_service import explain_prediction
    from ml.models.artifact import load_artifact
    from ml.prediction.service import predict

    spec = DatasetRegistry.get(key)
    if not DatasetRegistry(DATA).availability(key).available:
        pytest.skip(f"genuine {spec.file_name} not present in data/ (see data/README.md)")
    cfg = ExperimentConfig(dataset_key=key, num_rounds=2, malicious_clients=[],
                           attack_type="none", centralized_baseline=False)
    rec = run_experiment(cfg, DATA, tmp_path)
    assert rec["status"] == "completed" and rec["test_metrics"]["n"] == rec["splits"]["test"]
    art = load_artifact(tmp_path / rec["artifact_dir"], key)
    s = make_splits(load_dataset(key, DATA), cfg.seed, cfg.test_fraction, cfg.val_fraction)
    row = {k: (None if pd.isna(v) else v) for k, v in s.X_test.iloc[0].to_dict().items()}
    if any(v is None for v in row.values()):  # prediction requires complete input
        row = {k: (s.X_train[k].mode()[0] if v is None else v) for k, v in row.items()}
    pr = predict(art, row)
    exp = explain_prediction(art, pr)
    assert exp.base_value + sum(c.shap_value for c in exp.contributions) == pytest.approx(
        pr.decision_value if exp.explained_class == 1 else -pr.decision_value, abs=1e-6)


def test_dataset_summary_states(tmp_path):
    from ml.datasets.loader import dataset_summary
    s = dataset_summary("stroke", tmp_path)
    assert s["status"] == "missing" and not s["available"]
    assert "healthcare-dataset-stroke-data.csv" in s["message"]
    bc = dataset_summary("breast_cancer", tmp_path)
    assert bc["status"] == "valid" and bc["rows"] == 569 and bc["positives"] == 212
    d, spec = _fixture(tmp_path, "heart")
    bad = pd.read_csv(d / spec.file_name).drop(columns=["Sex"])
    bad.to_csv(d / spec.file_name, index=False)
    h = dataset_summary("heart", d)
    assert h["status"] == "invalid" and "Sex" in h["error"]


def test_import_validates_before_copying(tmp_path):
    import zipfile

    from ml.datasets.importer import import_dataset_file

    src_dir, spec = _fixture(tmp_path / "downloads", "stroke")
    data_dir = tmp_path / "data"
    res = import_dataset_file("stroke", src_dir / spec.file_name, data_dir)
    assert res.path == data_dir / spec.file_name and res.path.is_file()
    assert not res.matches_documented_counts  # synthetic fixture is not the genuine file
    with pytest.raises(DatasetError, match="already exists"):
        import_dataset_file("stroke", src_dir / spec.file_name, data_dir)

    # Kaggle-style zip
    zpath = tmp_path / "archive.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(src_dir / spec.file_name, spec.file_name)
    assert import_dataset_file("stroke", zpath, data_dir, overwrite=True).rows == res.rows

    # wrong file for the dataset is rejected and NOT copied
    d2, hspec = _fixture(tmp_path / "other", "heart")
    with pytest.raises(DatasetError, match="Missing columns"):
        import_dataset_file("framingham", d2 / hspec.file_name, data_dir)
    assert not (data_dir / "framingham.csv").exists()
