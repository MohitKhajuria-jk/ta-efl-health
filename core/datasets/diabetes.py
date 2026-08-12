"""Pima Indians Diabetes — onset of diabetes within five years."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DatasetSpec, FeatureSpec, logits_to_labels

FEATURES = [
    FeatureSpec("pregnancies", "Pregnancies", 0, 17, 3),
    FeatureSpec("glucose", "Plasma Glucose (mg/dl)", 44, 200, 120),
    FeatureSpec("blood_pressure", "Diastolic BP (mmHg)", 24, 122, 72),
    FeatureSpec("skin_thickness", "Triceps Skinfold (mm)", 7, 99, 23),
    FeatureSpec("insulin", "Serum Insulin (mu U/ml)", 14, 846, 90),
    FeatureSpec("bmi", "Body Mass Index", 18, 68, 32, decimals=1),
    FeatureSpec("pedigree", "Diabetes Pedigree", 0.05, 2.5, 0.47, decimals=3),
    FeatureSpec("age", "Age", 21, 81, 33),
]

_KAGGLE = {
    "Pregnancies": "pregnancies", "Glucose": "glucose",
    "BloodPressure": "blood_pressure", "SkinThickness": "skin_thickness",
    "Insulin": "insulin", "BMI": "bmi",
    "DiabetesPedigreeFunction": "pedigree", "Age": "age", "Outcome": "target",
}


def adapt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _KAGGLE.items() if k in df.columns})
    # In the original file, physiologically impossible zeros mean "missing".
    for c in ("glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"):
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
            df[c] = df[c].fillna(df[c].median())
    return df


def synth(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.gamma(5.0, 6.6, n).clip(21, 81) + 0.0
    pregnancies = rng.poisson(3.0 + 0.03 * (age - 33), n).clip(0, 17)
    bmi = rng.normal(32, 7, n).clip(18, 67)
    glucose = (rng.normal(120, 30, n) + 0.35 * (bmi - 32) + 0.3 * (age - 33)).clip(44, 199)
    blood_pressure = (rng.normal(69, 12, n) + 0.25 * (bmi - 32)).clip(24, 122)
    skin_thickness = (rng.normal(20, 10, n) + 0.55 * (bmi - 32)).clip(7, 99)
    insulin = (rng.gamma(2.0, 45, n) + 0.9 * (glucose - 120)).clip(14, 846)
    pedigree = rng.gamma(1.8, 0.26, n).clip(0.05, 2.42)

    logit = (
        0.035 * (glucose - 120) + 0.075 * (bmi - 32) + 0.028 * (age - 33)
        + 0.85 * (pedigree - 0.47) + 0.055 * pregnancies
        + 0.004 * (insulin - 90) + 0.006 * (blood_pressure - 69)
    )
    y = logits_to_labels(logit, rng, prevalence=0.35, sharpness=1.5)
    return pd.DataFrame({
        "pregnancies": pregnancies, "glucose": glucose.round(0),
        "blood_pressure": blood_pressure.round(0),
        "skin_thickness": skin_thickness.round(0), "insulin": insulin.round(0),
        "bmi": bmi.round(1), "pedigree": pedigree.round(3),
        "age": age.round(0), "target": y,
    })


SPEC = DatasetSpec(
    key="diabetes", name="Pima Diabetes",
    task="Five-year diabetes onset",
    target="target", positive_label="Diabetes onset",
    features=FEATURES,
    csv_names=["diabetes.csv", "pima_diabetes.csv", "pima-indians-diabetes.csv"],
    synth=synth, csv_adapter=adapt, n_synth=1500,
    citation="Pima Indians Diabetes Database (NIDDK / Kaggle)",
)
