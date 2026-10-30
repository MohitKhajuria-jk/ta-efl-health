from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.datasets.loader import DatasetError, load_dataset, make_splits, validate_and_clean
from ml.datasets.registry import ALL_SPECS, DatasetRegistry, HEART
from ml.preprocessing.pipeline import fit_preprocessor, transform, transformed_columns


def test_registry_has_five_distinct_datasets():
    keys = DatasetRegistry.keys()
    assert keys == ["heart", "diabetes", "breast_cancer", "framingham", "stroke"]
    hashes = {s.schema_hash() for s in ALL_SPECS}
    assert len(hashes) == 5
    with pytest.raises(KeyError):
        DatasetRegistry.get("unknown")


def test_missing_dataset_is_reported_not_substituted(tmp_path):
    reg = DatasetRegistry(tmp_path)
    av = reg.availability("heart")
    assert not av.available and "heart.csv" in av.message
    with pytest.raises(DatasetError):
        load_dataset("heart", tmp_path)


def test_breast_cancer_is_genuine_bundled_data():
    d = load_dataset("breast_cancer", Path("data"))
    assert d.report.n_rows == 569
    assert d.X.shape == (569, 30)
    assert d.report.positives == 212  # 212 malignant cases in WDBC
    assert list(d.X.columns) == d.spec.feature_names


def test_heart_schema_loading_and_missing_codes(heart_fixture_dir):
    d = load_dataset("heart", heart_fixture_dir)
    assert d.report.missing_codes_converted["Cholesterol"] >= 10
    assert d.X["Cholesterol"].isna().sum() >= 10
    assert set(np.unique(d.y)) == {0, 1}
    assert any("918" in w for w in d.report.warnings)  # fixture is not the genuine file


def test_schema_mismatch_rejected(heart_fixture_dir):
    df = pd.read_csv(heart_fixture_dir / "heart.csv").drop(columns=["ST_Slope"])
    with pytest.raises(DatasetError, match="ST_Slope"):
        validate_and_clean(HEART, df)
    df2 = pd.read_csv(heart_fixture_dir / "heart.csv")
    df2.loc[0, "ChestPainType"] = "XYZ"
    with pytest.raises(DatasetError, match="ChestPainType"):
        validate_and_clean(HEART, df2)
    df3 = pd.read_csv(heart_fixture_dir / "heart.csv")
    df3.loc[0, "HeartDisease"] = 3
    with pytest.raises(DatasetError, match="0/1"):
        validate_and_clean(HEART, df3)


def test_splits_are_disjoint_and_stratified():
    d = load_dataset("breast_cancer", Path("data"))
    s = make_splits(d, seed=1)
    sizes = s.sizes()
    assert sum(sizes.values()) == 569
    assert abs(s.y_test.mean() - d.y.mean()) < 0.03
    rows = lambda X: {tuple(r) for r in X.round(8).to_numpy()}  # noqa: E731
    assert not rows(s.X_train) & rows(s.X_test)
    assert not rows(s.X_val) & rows(s.X_test)


def test_preprocessing_is_consistent_between_batch_and_single_row(heart_fixture_dir):
    d = load_dataset("heart", heart_fixture_dir)
    s = make_splits(d, seed=3)
    pre = fit_preprocessor(HEART, s.X_train)  # training rows only
    full = transform(pre, HEART, s.X_test)
    single = transform(pre, HEART, s.X_test.iloc[[5]])
    np.testing.assert_allclose(full[5], single[0])
    assert full.shape[1] == len(transformed_columns(HEART)) == 6 + 2 + 4 + 3 + 2 + 3
    assert np.all(np.isfinite(full))  # missing cholesterol was imputed


def test_preprocessor_columns_follow_schema_vocabulary():
    names = [c.name for c in transformed_columns(HEART)]
    assert names[:6] == ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
    assert "ChestPainType=ASY" in names and "ST_Slope=Down" in names
