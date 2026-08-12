"""Framingham Heart Study — ten-year risk of coronary heart disease."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DatasetSpec, FeatureSpec, logits_to_labels

FEATURES = [
    FeatureSpec("male", "Sex", 0, 1, 1, choices={0: "Female", 1: "Male"}),
    FeatureSpec("age", "Age", 32, 70, 49),
    FeatureSpec("current_smoker", "Current Smoker", 0, 1, 0,
                choices={0: "No", 1: "Yes"}),
    FeatureSpec("cigs_per_day", "Cigarettes / Day", 0, 70, 0),
    FeatureSpec("bp_meds", "On BP Medication", 0, 1, 0, choices={0: "No", 1: "Yes"}),
    FeatureSpec("prevalent_stroke", "Prior Stroke", 0, 1, 0,
                choices={0: "No", 1: "Yes"}),
    FeatureSpec("prevalent_hyp", "Hypertensive", 0, 1, 0, choices={0: "No", 1: "Yes"}),
    FeatureSpec("diabetes", "Diabetic", 0, 1, 0, choices={0: "No", 1: "Yes"}),
    FeatureSpec("tot_chol", "Total Cholesterol", 110, 600, 237),
    FeatureSpec("sys_bp", "Systolic BP", 83, 295, 132, decimals=1),
    FeatureSpec("dia_bp", "Diastolic BP", 48, 143, 83, decimals=1),
    FeatureSpec("bmi", "Body Mass Index", 15, 57, 26, decimals=1),
    FeatureSpec("heart_rate", "Heart Rate", 44, 143, 76),
    FeatureSpec("glucose", "Glucose", 40, 394, 82),
]

_KAGGLE = {
    "male": "male", "Sex_male": "male", "age": "age",
    "currentSmoker": "current_smoker", "cigsPerDay": "cigs_per_day",
    "BPMeds": "bp_meds", "prevalentStroke": "prevalent_stroke",
    "prevalentHyp": "prevalent_hyp", "diabetes": "diabetes",
    "totChol": "tot_chol", "sysBP": "sys_bp", "diaBP": "dia_bp",
    "BMI": "bmi", "heartRate": "heart_rate", "glucose": "glucose",
    "TenYearCHD": "target",
}


def adapt(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in _KAGGLE.items() if k in df.columns})


def synth(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    male = rng.binomial(1, 0.44, n)
    age = rng.normal(49.6, 8.6, n).clip(32, 70)
    smoker = rng.binomial(1, 0.49, n)
    cigs = (smoker * rng.gamma(2.2, 8.5, n)).clip(0, 70)
    bmi = rng.normal(25.8, 4.1, n).clip(15.5, 56.8)
    hyp = rng.binomial(1, 1 / (1 + np.exp(-(0.06 * (age - 49) + 0.12 * (bmi - 26) - 1.0))))
    bp_meds = rng.binomial(1, 0.03 + 0.09 * hyp)
    stroke = rng.binomial(1, 0.004 + 0.012 * (age > 60))
    diab = rng.binomial(1, 0.02 + 0.04 * (bmi > 30))
    tot_chol = rng.normal(236, 44, n).clip(113, 600)
    sys_bp = (rng.normal(126, 18, n) + 18 * hyp + 0.35 * (age - 49)).clip(83.5, 295)
    dia_bp = (0.55 * sys_bp + rng.normal(12, 8, n)).clip(48, 142)
    hr = rng.normal(75.9, 12, n).clip(44, 143)
    glucose = (rng.normal(80, 15, n) + 45 * diab).clip(40, 394)

    logit = (
        0.062 * (age - 49) + 0.52 * male + 0.019 * cigs + 0.30 * smoker
        + 0.017 * (sys_bp - 126) + 0.0035 * (tot_chol - 236)
        + 0.75 * diab + 0.45 * hyp + 0.9 * stroke + 0.30 * bp_meds
        + 0.011 * (glucose - 82) + 0.02 * (bmi - 26)
    )
    y = logits_to_labels(logit, rng, prevalence=0.152, sharpness=1.3)
    return pd.DataFrame({
        "male": male, "age": age.round(0), "current_smoker": smoker,
        "cigs_per_day": cigs.round(0), "bp_meds": bp_meds,
        "prevalent_stroke": stroke, "prevalent_hyp": hyp, "diabetes": diab,
        "tot_chol": tot_chol.round(0), "sys_bp": sys_bp.round(1),
        "dia_bp": dia_bp.round(1), "bmi": bmi.round(2),
        "heart_rate": hr.round(0), "glucose": glucose.round(0), "target": y,
    })


SPEC = DatasetSpec(
    key="framingham", name="Framingham CHD",
    task="Ten-year coronary heart disease risk",
    target="target", positive_label="CHD within 10 years",
    features=FEATURES,
    csv_names=["framingham.csv", "framingham_heart_study.csv", "cvd.csv"],
    synth=synth, csv_adapter=adapt, n_synth=2000,
    citation="Framingham Heart Study (Kaggle framingham.csv)",
)
