"""Hospital-specific probability calibration on top of a frozen global federated model.

This package adds *local probability calibration* (two parameters per hospital):

    P_personalized = sigmoid(a_h * z_global + b_h)

It does not retrain, fine-tune or otherwise modify the global model, and it does not
learn hospital-specific features. See docs/PERSONALIZATION.md.
"""
from ml.personalization.calibration import (
    CALIBRATOR_FORMAT_VERSION,
    CalibrationConfig,
    CalibrationError,
    HospitalCalibrator,
    InsufficientCalibrationData,
    fit_calibrator,
    logit,
    stable_sigmoid,
)

__all__ = [
    "CALIBRATOR_FORMAT_VERSION", "CalibrationConfig", "CalibrationError", "HospitalCalibrator",
    "InsufficientCalibrationData", "fit_calibrator", "logit", "stable_sigmoid",
]
