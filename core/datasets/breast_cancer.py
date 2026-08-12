"""Breast Cancer Wisconsin (Diagnostic) — malignant vs benign tumour.

Uses the ten *mean* cell-nucleus measurements, which keeps the SHAP
explanations readable while preserving the discriminative signal. The data
ships with scikit-learn, so this dataset never needs a download.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DatasetSpec, FeatureSpec, logits_to_labels

_COLS = [
    ("mean_radius", "Mean Radius", 6.9, 28.2, 14.1, 2),
    ("mean_texture", "Mean Texture", 9.7, 39.3, 19.3, 2),
    ("mean_perimeter", "Mean Perimeter", 43.8, 188.5, 92.0, 1),
    ("mean_area", "Mean Area", 143.5, 2501.0, 654.9, 1),
    ("mean_smoothness", "Mean Smoothness", 0.05, 0.17, 0.096, 4),
    ("mean_compactness", "Mean Compactness", 0.019, 0.35, 0.104, 4),
    ("mean_concavity", "Mean Concavity", 0.0, 0.43, 0.089, 4),
    ("mean_concave_points", "Mean Concave Points", 0.0, 0.21, 0.049, 4),
    ("mean_symmetry", "Mean Symmetry", 0.11, 0.31, 0.181, 4),
    ("mean_fractal_dimension", "Mean Fractal Dimension", 0.05, 0.098, 0.063, 5),
]
FEATURES = [FeatureSpec(k, lab, lo, hi, d, dec) for k, lab, lo, hi, d, dec in _COLS]
_SK_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
]


def builtin() -> "tuple[pd.DataFrame, str] | None":
    """The real WDBC data ships with scikit-learn — no download required."""
    df = _from_sklearn()
    if df is None:
        return None
    return df, "Real (scikit-learn load_breast_cancer)"


def _from_sklearn() -> pd.DataFrame | None:
    try:
        from sklearn.datasets import load_breast_cancer
    except Exception:                                    # pragma: no cover
        return None
    d = load_breast_cancer(as_frame=True).frame
    df = d[_SK_NAMES].copy()
    df.columns = [f.key for f in FEATURES]
    # sklearn encodes 0 = malignant; we want 1 = malignant (the risk class).
    df["target"] = 1 - d["target"].to_numpy()
    return df


def synth(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Correlated simulation used only when scikit-learn is unavailable."""
    rng = np.random.default_rng(seed)
    radius = rng.normal(14.1, 3.5, n).clip(6.9, 28.2)
    texture = rng.normal(19.3, 4.3, n).clip(9.7, 39.3)
    perimeter = (6.5 * radius + rng.normal(0, 3, n)).clip(43.8, 188.5)
    area = (3.14 * radius ** 2 + rng.normal(0, 60, n)).clip(143.5, 2501)
    smooth = rng.normal(0.096, 0.014, n).clip(0.05, 0.163)
    compact = (rng.gamma(3, 0.03, n) + 0.004 * (radius - 14)).clip(0.019, 0.345)
    concavity = (compact * rng.normal(0.9, 0.25, n)).clip(0, 0.427)
    cpoints = (concavity * 0.55 + rng.normal(0, 0.012, n)).clip(0, 0.201)
    symmetry = rng.normal(0.181, 0.027, n).clip(0.106, 0.304)
    fractal = rng.normal(0.0628, 0.007, n).clip(0.05, 0.097)

    logit = (
        0.42 * (radius - 14.1) + 0.11 * (texture - 19.3)
        + 22.0 * (cpoints - 0.049) + 9.0 * (concavity - 0.089)
        + 0.0022 * (area - 655) + 18.0 * (compact - 0.104)
        + 6.0 * (smooth - 0.096)
    )
    y = logits_to_labels(logit, rng, prevalence=0.37, sharpness=1.1)
    return pd.DataFrame({
        "mean_radius": radius.round(2), "mean_texture": texture.round(2),
        "mean_perimeter": perimeter.round(1), "mean_area": area.round(1),
        "mean_smoothness": smooth.round(4), "mean_compactness": compact.round(4),
        "mean_concavity": concavity.round(4), "mean_concave_points": cpoints.round(4),
        "mean_symmetry": symmetry.round(4),
        "mean_fractal_dimension": fractal.round(5), "target": y,
    })


def adapt(df: pd.DataFrame) -> pd.DataFrame:
    ren = {s: f.key for s, f in zip(_SK_NAMES, FEATURES)}
    ren.update({s.replace(" ", "_"): f.key for s, f in zip(_SK_NAMES, FEATURES)})
    df = df.rename(columns=ren)
    if "diagnosis" in df.columns:
        df["target"] = (df["diagnosis"].astype(str).str.upper().str[0] == "M").astype(int)
    return df


SPEC = DatasetSpec(
    key="breast_cancer", name="Breast Cancer Wisconsin",
    task="Malignant vs benign tumour",
    target="target", positive_label="Malignant",
    features=FEATURES,
    csv_names=["breast_cancer.csv", "wdbc.csv", "data_wdbc.csv"],
    synth=synth, csv_adapter=adapt, builtin=builtin, n_synth=569,
    citation="Breast Cancer Wisconsin (Diagnostic), UCI / sklearn",
)
