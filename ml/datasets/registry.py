"""Dataset registry: the five supported healthcare datasets.

Only the Wisconsin Diagnostic Breast Cancer data is bundled (it ships inside
scikit-learn). The other four must be supplied by the user as the genuine CSV
files from their documented sources; the application never substitutes another
dataset when one is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml.datasets.schema import DatasetSpec, FeatureSpec

# --------------------------------------------------------------------------- #
# 1. Heart disease - Kaggle "Heart Failure Prediction Dataset" (fedesoriano)
# --------------------------------------------------------------------------- #
HEART = DatasetSpec(
    key="heart",
    name="Heart Failure Prediction",
    description="Heart disease (918 patients, 11 clinical features; combined Cleveland, "
    "Hungarian, Switzerland, Long Beach VA and Stalog cohorts).",
    source="Kaggle: fedesoriano, 'Heart Failure Prediction Dataset' (2021), file heart.csv",
    source_url="https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction",
    file_name="heart.csv",
    target="HeartDisease",
    positive_label="Heart disease",
    negative_label="No heart disease",
    expected_rows=918,
    expected_positives=508,
    licence="Check the licence shown on the Kaggle page before redistribution; cite fedesoriano (2021).",
    features=(
        FeatureSpec("Age", "Age", "numeric", 18, 100, integer=True, unit="years"),
        FeatureSpec("Sex", "Sex", "categorical", categories=("M", "F"),
                    category_labels=("Male", "Female")),
        FeatureSpec("ChestPainType", "Chest Pain Type", "categorical",
                    categories=("TA", "ATA", "NAP", "ASY"),
                    category_labels=("Typical Angina", "Atypical Angina",
                                     "Non-Anginal Pain", "Asymptomatic")),
        FeatureSpec("RestingBP", "Resting Blood Pressure", "numeric", 60, 250, integer=True,
                    unit="mm Hg", missing_codes=(0,),
                    help="A value of 0 in the source file is a recording error and is treated as missing."),
        FeatureSpec("Cholesterol", "Cholesterol", "numeric", 80, 700, integer=True,
                    unit="mg/dl", missing_codes=(0,),
                    help="172 source records store 0 for 'not measured'; treated as missing and imputed."),
        FeatureSpec("FastingBS", "Fasting Blood Sugar > 120 mg/dl", "binary", 0, 1, integer=True,
                    category_labels=("No", "Yes")),
        FeatureSpec("RestingECG", "Resting ECG", "categorical",
                    categories=("Normal", "ST", "LVH"),
                    category_labels=("Normal", "ST-T wave abnormality",
                                     "Left ventricular hypertrophy")),
        FeatureSpec("MaxHR", "Max Heart Rate", "numeric", 60, 220, integer=True, unit="bpm"),
        FeatureSpec("ExerciseAngina", "Exercise-Induced Angina", "categorical",
                    categories=("N", "Y"), category_labels=("No", "Yes")),
        FeatureSpec("Oldpeak", "ST Depression (Oldpeak)", "numeric", -3.0, 7.0, unit="mm"),
        FeatureSpec("ST_Slope", "ST Slope", "categorical", categories=("Up", "Flat", "Down"),
                    category_labels=("Upsloping", "Flat", "Downsloping")),
    ),
    notes=("Zeros in RestingBP and Cholesterol are documented missing-value codes.",),
)

# --------------------------------------------------------------------------- #
# 2. Diabetes - Pima Indians Diabetes Database (NIDDK via UCI / Kaggle)
# --------------------------------------------------------------------------- #
DIABETES = DatasetSpec(
    key="diabetes",
    name="Pima Indians Diabetes",
    description="Onset of diabetes within 5 years in Pima Indian women aged 21+ (768 records).",
    source="Kaggle: UCI Machine Learning, 'Pima Indians Diabetes Database', file diabetes.csv",
    source_url="https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
    file_name="diabetes.csv",
    target="Outcome",
    positive_label="Diabetes",
    negative_label="No diabetes",
    expected_rows=768,
    expected_positives=268,
    licence="Originally NIDDK, donated to UCI (1990). Check the licence shown on the source page; cite Smith et al. (1988).",
    features=(
        FeatureSpec("Pregnancies", "Pregnancies", "numeric", 0, 20, integer=True),
        FeatureSpec("Glucose", "Plasma Glucose (2h OGTT)", "numeric", 40, 300, unit="mg/dl",
                    missing_codes=(0,)),
        FeatureSpec("BloodPressure", "Diastolic Blood Pressure", "numeric", 20, 140,
                    unit="mm Hg", missing_codes=(0,)),
        FeatureSpec("SkinThickness", "Triceps Skin Fold", "numeric", 5, 100, unit="mm",
                    missing_codes=(0,)),
        FeatureSpec("Insulin", "2-Hour Serum Insulin", "numeric", 10, 900, unit="mu U/ml",
                    missing_codes=(0,)),
        FeatureSpec("BMI", "BMI", "numeric", 12, 70, unit="kg/m^2", missing_codes=(0,)),
        FeatureSpec("DiabetesPedigreeFunction", "Diabetes Pedigree Function", "numeric",
                    0.0, 3.0),
        FeatureSpec("Age", "Age", "numeric", 21, 100, integer=True, unit="years"),
    ),
    notes=("Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are "
           "physiologically impossible and treated as missing.",),
)

# --------------------------------------------------------------------------- #
# 3. Breast cancer - Wisconsin Diagnostic (bundled with scikit-learn)
# --------------------------------------------------------------------------- #
_BC_BASE = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
            "concavity", "concave points", "symmetry", "fractal dimension"]
# sklearn column names: "mean radius", "radius error" (standard error), "worst radius"
_BC_FEATURES = tuple(
    [FeatureSpec(f"mean {b}", f"Mean {b.title()}", "numeric", 0.0, None) for b in _BC_BASE]
    + [FeatureSpec(f"{b} error", f"{b.title()} (SE)", "numeric", 0.0, None) for b in _BC_BASE]
    + [FeatureSpec(f"worst {b}", f"Worst {b.title()}", "numeric", 0.0, None) for b in _BC_BASE]
)

BREAST_CANCER = DatasetSpec(
    key="breast_cancer",
    name="Breast Cancer (Wisconsin Diagnostic)",
    description="Malignant vs benign breast mass from digitised FNA images (569 records, "
    "30 nuclear features).",
    source="UCI Breast Cancer Wisconsin (Diagnostic), Wolberg et al. 1995; "
    "bundled as sklearn.datasets.load_breast_cancer",
    source_url="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    file_name=None,
    target="malignant",
    positive_label="Malignant",
    negative_label="Benign",
    expected_rows=569,
    expected_positives=212,
    licence="UCI Machine Learning Repository, CC BY 4.0. Cite Wolberg et al. (1995).",
    features=_BC_FEATURES,
    notes=("scikit-learn encodes 0=malignant, 1=benign; this application recodes the target "
           "so that 1 = malignant (the risk class).",),
)

# --------------------------------------------------------------------------- #
# 4. Framingham - 10-year coronary heart disease risk
# --------------------------------------------------------------------------- #
FRAMINGHAM = DatasetSpec(
    key="framingham",
    name="Framingham CHD Risk",
    description="10-year coronary heart disease (TenYearCHD) from the Framingham Heart "
    "Study teaching extract (~4,240 records).",
    source="Framingham Heart Study teaching dataset as distributed on Kaggle, file framingham.csv",
    source_url="https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset",
    file_name="framingham.csv",
    target="TenYearCHD",
    positive_label="CHD within 10 years",
    negative_label="No CHD within 10 years",
    expected_rows=4240,
    expected_positives=644,
    licence="Teaching extract of the NHLBI Framingham Heart Study; check the Kaggle page terms before redistribution.",
    features=(
        FeatureSpec("male", "Sex", "binary", 0, 1, integer=True, category_labels=("Female", "Male")),
        FeatureSpec("age", "Age", "numeric", 20, 100, integer=True, unit="years"),
        FeatureSpec("education", "Education Level", "numeric", 1, 4, integer=True),
        FeatureSpec("currentSmoker", "Current Smoker", "binary", 0, 1, integer=True,
                    category_labels=("No", "Yes")),
        FeatureSpec("cigsPerDay", "Cigarettes per Day", "numeric", 0, 80, integer=True),
        FeatureSpec("BPMeds", "On BP Medication", "binary", 0, 1, integer=True,
                    category_labels=("No", "Yes")),
        FeatureSpec("prevalentStroke", "Prior Stroke", "binary", 0, 1, integer=True,
                    category_labels=("No", "Yes")),
        FeatureSpec("prevalentHyp", "Hypertensive", "binary", 0, 1, integer=True,
                    category_labels=("No", "Yes")),
        FeatureSpec("diabetes", "Diabetes", "binary", 0, 1, integer=True,
                    category_labels=("No", "Yes")),
        FeatureSpec("totChol", "Total Cholesterol", "numeric", 100, 700, unit="mg/dl"),
        FeatureSpec("sysBP", "Systolic BP", "numeric", 70, 300, unit="mm Hg"),
        FeatureSpec("diaBP", "Diastolic BP", "numeric", 40, 160, unit="mm Hg"),
        FeatureSpec("BMI", "BMI", "numeric", 12, 70, unit="kg/m^2"),
        FeatureSpec("heartRate", "Heart Rate", "numeric", 35, 160, unit="bpm"),
        FeatureSpec("glucose", "Glucose", "numeric", 40, 400, unit="mg/dl"),
    ),
)

# --------------------------------------------------------------------------- #
# 5. Stroke - Kaggle "Stroke Prediction Dataset" (fedesoriano)
# --------------------------------------------------------------------------- #
STROKE = DatasetSpec(
    key="stroke",
    name="Stroke Prediction",
    description="Stroke occurrence (5,110 records, strongly imbalanced: ~4.9% positive).",
    source="Kaggle: fedesoriano, 'Stroke Prediction Dataset', file "
    "healthcare-dataset-stroke-data.csv",
    source_url="https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset",
    file_name="healthcare-dataset-stroke-data.csv",
    target="stroke",
    positive_label="Stroke",
    negative_label="No stroke",
    expected_rows=5110,
    expected_positives=249,
    licence="The Kaggle page describes the source as confidential; use for research/education only and do not redistribute.",
    drop_columns=("id",),
    features=(
        FeatureSpec("gender", "Gender", "categorical", categories=("Male", "Female", "Other"),
                    category_labels=("Male", "Female", "Other")),
        FeatureSpec("age", "Age", "numeric", 0, 100, unit="years"),
        FeatureSpec("hypertension", "Hypertension", "binary", 0, 1, integer=True,
                    category_labels=("No", "Yes")),
        FeatureSpec("heart_disease", "Heart Disease", "binary", 0, 1, integer=True,
                    category_labels=("No", "Yes")),
        FeatureSpec("ever_married", "Ever Married", "categorical", categories=("No", "Yes"),
                    category_labels=("No", "Yes")),
        FeatureSpec("work_type", "Work Type", "categorical",
                    categories=("children", "Govt_job", "Never_worked", "Private", "Self-employed"),
                    category_labels=("Children", "Government job", "Never worked", "Private",
                                     "Self-employed")),
        FeatureSpec("Residence_type", "Residence Type", "categorical",
                    categories=("Urban", "Rural"), category_labels=("Urban", "Rural")),
        FeatureSpec("avg_glucose_level", "Average Glucose Level", "numeric", 40, 350,
                    unit="mg/dl"),
        FeatureSpec("bmi", "BMI", "numeric", 10, 100, unit="kg/m^2"),
        FeatureSpec("smoking_status", "Smoking Status", "categorical",
                    categories=("formerly smoked", "never smoked", "smokes", "Unknown"),
                    category_labels=("Formerly smoked", "Never smoked", "Smokes", "Unknown")),
    ),
    notes=("'N/A' BMI values are treated as missing. 'Unknown' smoking status is a "
           "documented category in the source and is kept as its own category.",),
)

ALL_SPECS: tuple[DatasetSpec, ...] = (HEART, DIABETES, BREAST_CANCER, FRAMINGHAM, STROKE)


@dataclass(frozen=True)
class DatasetAvailability:
    spec: DatasetSpec
    available: bool
    path: Path | None
    message: str


class DatasetRegistry:
    """Lookup of dataset specs and whether their genuine data is present."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    @staticmethod
    def keys() -> list[str]:
        return [s.key for s in ALL_SPECS]

    @staticmethod
    def get(key: str) -> DatasetSpec:
        for s in ALL_SPECS:
            if s.key == key:
                return s
        raise KeyError(f"Unknown dataset '{key}'. Supported: {', '.join(DatasetRegistry.keys())}")

    @staticmethod
    def by_name(name: str) -> DatasetSpec:
        for s in ALL_SPECS:
            if s.name == name:
                return s
        raise KeyError(name)

    def availability(self, key: str) -> DatasetAvailability:
        spec = self.get(key)
        if spec.file_name is None:
            return DatasetAvailability(spec, True, None, "Bundled with scikit-learn")
        path = self.data_dir / spec.file_name
        if path.is_file():
            return DatasetAvailability(spec, True, path, f"Found {path.name}")
        return DatasetAvailability(
            spec, False, None,
            f"Not available: place the genuine '{spec.file_name}' from {spec.source_url} "
            f"in {self.data_dir}",
        )

    def all_availability(self) -> list[DatasetAvailability]:
        return [self.availability(k) for k in self.keys()]
