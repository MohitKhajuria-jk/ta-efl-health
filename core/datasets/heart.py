"""UCI / Kaggle Heart Disease (Cleveland) — coronary artery disease presence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DatasetSpec, FeatureSpec, logits_to_labels

FEATURES = [
    FeatureSpec("age", "Age", 29, 77, 54),
    FeatureSpec("sex", "Sex", 0, 1, 1, choices={0: "Female", 1: "Male"}),
    FeatureSpec("cp", "Chest Pain Type", 0, 3, 0,
                choices={0: "Typical angina", 1: "Atypical angina",
                         2: "Non-anginal", 3: "Asymptomatic"}),
    FeatureSpec("trestbps", "Resting BP (mmHg)", 90, 200, 130),
    FeatureSpec("chol", "Cholesterol (mg/dl)", 120, 570, 246),
    FeatureSpec("fbs", "Fasting Blood Sugar > 120", 0, 1, 0,
                choices={0: "No", 1: "Yes"}),
    FeatureSpec("restecg", "Resting ECG", 0, 2, 1,
                choices={0: "Normal", 1: "ST-T wave abnormality",
                         2: "Left ventricular hypertrophy"}),
    FeatureSpec("thalach", "Max Heart Rate", 70, 205, 150),
    FeatureSpec("exang", "Exercise Induced Angina", 0, 1, 0,
                choices={0: "No", 1: "Yes"}),
    FeatureSpec("oldpeak", "ST Depression", 0, 6.5, 1.0, decimals=1),
    FeatureSpec("slope", "Slope of ST", 0, 2, 1,
                choices={0: "Upsloping", 1: "Flat", 2: "Downsloping"}),
    FeatureSpec("ca", "Number of Vessels", 0, 4, 0),   # a count, stays numeric
    FeatureSpec("thal", "Thalassemia", 0, 3, 2,
                choices={0: "Not measured", 1: "Fixed defect",
                         2: "Normal", 3: "Reversible defect"}),
]


def synth(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(54, 9, n).clip(29, 77)
    sex = rng.binomial(1, 0.68, n)
    cp = rng.integers(0, 4, n)
    trestbps = rng.normal(131, 17, n).clip(94, 200)
    chol = rng.normal(246, 51, n).clip(126, 564)
    fbs = rng.binomial(1, 0.15, n)
    restecg = rng.integers(0, 3, n)
    thalach = rng.normal(149, 22, n).clip(71, 202)
    exang = rng.binomial(1, 0.33, n)
    oldpeak = rng.gamma(1.2, 0.9, n).clip(0, 6.2)
    slope = rng.integers(0, 3, n)
    ca = rng.integers(0, 4, n)
    thal = rng.integers(0, 4, n)

    logit = (
        0.035 * (age - 54) + 0.55 * sex - 0.45 * (cp == 0)
        + 0.012 * (trestbps - 131) + 0.004 * (chol - 246)
        - 0.022 * (thalach - 149) + 0.95 * exang + 0.62 * oldpeak
        + 0.75 * (ca > 0) + 0.5 * (thal == 2)
    )
    y = logits_to_labels(logit, rng, prevalence=0.54)
    return pd.DataFrame({
        "age": age.round(0), "sex": sex, "cp": cp, "trestbps": trestbps.round(0),
        "chol": chol.round(0), "fbs": fbs, "restecg": restecg,
        "thalach": thalach.round(0), "exang": exang, "oldpeak": oldpeak.round(1),
        "slope": slope, "ca": ca, "thal": thal, "target": y,
    })


SPEC = DatasetSpec(
    key="heart", name="Heart Disease",
    task="Coronary artery disease presence",
    target="target", positive_label="Heart disease present",
    features=FEATURES,
    csv_names=["heart.csv", "heart_disease.csv", "dataset.csv"],
    synth=synth, n_synth=1500,
    citation="UCI Cleveland Heart Disease / Kaggle heart.csv",
)
