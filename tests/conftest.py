"""Shared fixtures.

Real-data tests use the genuine Wisconsin Diagnostic Breast Cancer data bundled
with scikit-learn. Tests of the CSV loader for user-supplied datasets build
small *schema fixtures* in a temporary directory; those rows are synthetic,
exist only inside the test run and are never presented as patient data.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

logging.getLogger("flwr").setLevel(logging.WARNING)


def make_heart_schema_fixture(path: Path, n: int = 120, seed: int = 0) -> Path:
    """Write a synthetic CSV with the heart.csv schema (for loader/validation tests only)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Age": rng.integers(30, 78, n),
        "Sex": rng.choice(["M", "F"], n),
        "ChestPainType": rng.choice(["TA", "ATA", "NAP", "ASY"], n),
        "RestingBP": rng.integers(95, 180, n),
        "Cholesterol": rng.integers(150, 350, n),
        "FastingBS": rng.integers(0, 2, n),
        "RestingECG": rng.choice(["Normal", "ST", "LVH"], n),
        "MaxHR": rng.integers(90, 200, n),
        "ExerciseAngina": rng.choice(["N", "Y"], n),
        "Oldpeak": np.round(rng.uniform(-1, 4, n), 1),
        "ST_Slope": rng.choice(["Up", "Flat", "Down"], n),
    })
    logit = 0.04 * (df["Age"] - 54) + 0.8 * (df["ExerciseAngina"] == "Y") + 0.5 * df["Oldpeak"] - 0.5
    df["HeartDisease"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.loc[:9, "Cholesterol"] = 0  # documented missing-value code
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def heart_fixture_dir(tmp_path: Path) -> Path:
    make_heart_schema_fixture(tmp_path / "data" / "heart.csv")
    return tmp_path / "data"


@pytest.fixture(scope="session")
def bc_experiment(tmp_path_factory):
    """One genuine Flower experiment on the bundled breast cancer data (shared by tests)."""
    from federated.server.runner import ExperimentConfig, run_experiment

    storage = tmp_path_factory.mktemp("storage")
    cfg = ExperimentConfig(dataset_key="breast_cancer", num_clients=4, num_rounds=6,
                           local_epochs=3, strategy="trust", malicious_clients=[4],
                           attack_type="sign_flip", seed=7)
    events: list[dict] = []
    record = run_experiment(cfg, Path("data"), storage, on_event=events.append)
    return {"record": record, "storage": storage, "events": events, "config": cfg}


def make_schema_fixture(spec_key: str, path: Path, n: int = 150, seed: int = 0) -> Path:
    """Synthetic rows following a dataset's schema (unit tests of loader rules only)."""
    from ml.datasets.registry import DatasetRegistry
    spec = DatasetRegistry.get(spec_key)
    rng = np.random.default_rng(seed)
    cols = {}
    for f in spec.features:
        if f.kind == "categorical":
            cols[f.name] = rng.choice(list(f.categories), n)
        elif f.kind == "binary":
            cols[f.name] = rng.integers(0, 2, n)
        else:
            lo = f.min_value if f.min_value is not None else 0.0
            hi = f.max_value if f.max_value is not None else lo + 100.0
            v = rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), n)
            cols[f.name] = np.round(v) if f.integer else np.round(v, 3)
    df = pd.DataFrame(cols)
    df[spec.target] = (rng.uniform(size=n) < 0.4).astype(int)
    for extra in spec.drop_columns:
        df.insert(0, extra, np.arange(n))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
