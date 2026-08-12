"""Healthcare Stroke Prediction — occurrence of a stroke event.

Highly imbalanced (~5% positives), which makes it a useful stress test for
the trust engine: honest hospitals genuinely disagree more here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DatasetSpec, FeatureSpec, logits_to_labels

FEATURES = [
    FeatureSpec("gender", "Gender", 0, 1, 1, choices={0: "Female", 1: "Male"}),
    FeatureSpec("age", "Age", 1, 82, 45),
    FeatureSpec("hypertension", "Hypertension", 0, 1, 0, choices={0: "No", 1: "Yes"}),
    FeatureSpec("heart_disease", "Heart Disease", 0, 1, 0, choices={0: "No", 1: "Yes"}),
    FeatureSpec("ever_married", "Ever Married", 0, 1, 1, choices={0: "No", 1: "Yes"}),
    FeatureSpec("work_type", "Work Type", 0, 4, 2,
                choices={0: "Never worked", 1: "Children", 2: "Private",
                         3: "Self-employed", 4: "Government"}),
    FeatureSpec("residence_urban", "Urban Residence", 0, 1, 1,
                choices={0: "Rural", 1: "Urban"}),
    FeatureSpec("avg_glucose_level", "Avg Glucose Level", 55, 272, 106, decimals=1),
    FeatureSpec("bmi", "Body Mass Index", 12, 98, 28, decimals=1),
    FeatureSpec("smoking_status", "Smoking Status", 0, 3, 1,
                choices={0: "Never smoked", 1: "Unknown", 2: "Formerly smoked",
                         3: "Smokes"}),
]

_WORK = {"Never_worked": 0, "children": 1, "Private": 2,
         "Self-employed": 3, "Govt_job": 4}
_SMOKE = {"never smoked": 0, "Unknown": 1, "formerly smoked": 2, "smokes": 3}


def adapt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "gender" in df.columns and df["gender"].dtype == object:
        df["gender"] = (df["gender"].astype(str).str.lower() == "male").astype(int)
    if "ever_married" in df.columns and df["ever_married"].dtype == object:
        df["ever_married"] = (df["ever_married"].astype(str).str.lower() == "yes").astype(int)
    if "work_type" in df.columns and df["work_type"].dtype == object:
        df["work_type"] = df["work_type"].map(_WORK).fillna(2)
    if "Residence_type" in df.columns:
        df["residence_urban"] = (
            df["Residence_type"].astype(str).str.lower() == "urban").astype(int)
    if "smoking_status" in df.columns and df["smoking_status"].dtype == object:
        df["smoking_status"] = df["smoking_status"].map(_SMOKE).fillna(1)
    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    if "stroke" in df.columns:
        df["target"] = df["stroke"]
    return df


def synth(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.binomial(1, 0.41, n)
    age = rng.uniform(1, 82, n)
    hyp = rng.binomial(1, 1 / (1 + np.exp(-(0.05 * (age - 55) - 1.6))))
    hd = rng.binomial(1, 1 / (1 + np.exp(-(0.06 * (age - 60) - 2.0))))
    married = rng.binomial(1, np.clip((age - 15) / 30, 0.02, 0.9))
    work = np.where(age < 16, 1, rng.choice([0, 2, 3, 4], n, p=[.02, .64, .18, .16]))
    urban = rng.binomial(1, 0.51, n)
    glucose = (rng.gamma(6.5, 14, n) + 0.25 * (age - 45)).clip(55, 272)
    bmi = (rng.normal(28.9, 7.7, n) + 0.05 * (age - 45)).clip(12, 97)
    smoke = np.where(age < 18, 0, rng.choice([0, 1, 2, 3], n, p=[.42, .25, .18, .15]))

    logit = (
        0.075 * (age - 45) + 1.05 * hyp + 1.15 * hd + 0.006 * (glucose - 106)
        + 0.012 * (bmi - 28) + 0.22 * (smoke >= 2) + 0.15 * gender
        + 0.18 * married
    )
    y = logits_to_labels(logit, rng, prevalence=0.10, sharpness=1.2)
    return pd.DataFrame({
        "gender": gender, "age": age.round(0), "hypertension": hyp,
        "heart_disease": hd, "ever_married": married, "work_type": work,
        "residence_urban": urban, "avg_glucose_level": glucose.round(2),
        "bmi": bmi.round(1), "smoking_status": smoke, "target": y,
    })


SPEC = DatasetSpec(
    key="stroke", name="Stroke Prediction",
    task="Stroke event risk (highly imbalanced)",
    target="target", positive_label="Stroke",
    features=FEATURES,
    csv_names=["stroke.csv", "healthcare-dataset-stroke-data.csv"],
    synth=synth, csv_adapter=adapt, n_synth=2500,
    citation="Healthcare Stroke Prediction dataset (Kaggle)",
)
