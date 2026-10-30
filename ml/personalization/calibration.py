"""Hospital-specific logistic probability calibration.

For hospital ``h`` with global-model log-odds ``z`` (``z = w . x + c`` of the frozen
federated logistic regression) the calibrated probability is

    P_h(y = 1 | x) = sigmoid(a_h * z + b_h)

``(a_h, b_h)`` are fitted on that hospital's *local calibration rows only* by
minimising the penalised negative log-likelihood

    J(a, b) = sum_i [ log(1 + exp(s_i)) - y_i * s_i ]  +  (lambda / 2) * ((a - 1)^2 + b^2),
    s_i = a * z_i + b.

The penalty shrinks the calibrator towards the identity map (a = 1, b = 0), i.e. towards
the unchanged global model, with a fixed strength ``lambda`` chosen before any experiment
(it is never tuned on test data). Because the penalty is on the *sum* of the log-loss
terms, its relative influence decreases as the number of local calibration rows grows.
With ``lambda > 0`` the objective is strictly convex, so the Newton iterations below
converge to the unique minimiser; perfectly separable calibration data therefore cannot
produce infinite parameters.

Requirements enforced before fitting (no silent fallback to the global model):
* finite log-odds, 0/1 labels, equal lengths;
* at least ``min_samples`` rows and ``min_per_class`` rows of each class;
* the fitted slope ``a`` must be strictly positive (otherwise the calibrator would
  reverse the global model's risk ranking, and it is rejected).

Because ``a > 0`` the transformation is strictly increasing: it changes probabilities,
log loss, Brier score, ECE and threshold-based metrics, but **never the ROC-AUC computed
within one hospital**.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

CALIBRATOR_FORMAT_VERSION = 1
METHOD = "hospital-specific logistic probability calibration (penalised towards identity)"


class CalibrationError(ValueError):
    """The calibrator cannot be fitted or applied; the reason is in the message."""


class InsufficientCalibrationData(CalibrationError):
    """Too few local calibration rows, or too few rows of one class."""


@dataclass(frozen=True)
class CalibrationConfig:
    min_samples: int = 20
    min_per_class: int = 5
    regularization: float = 1.0  # lambda (> 0) in the penalty above
    max_iter: int = 100
    tol: float = 1e-10

    def __post_init__(self) -> None:
        if int(self.min_samples) != self.min_samples or self.min_samples < 2:
            raise ValueError("min_samples must be an integer >= 2")
        if int(self.min_per_class) != self.min_per_class or self.min_per_class < 1:
            raise ValueError("min_per_class must be an integer >= 1")
        if 2 * self.min_per_class > self.min_samples:
            raise ValueError("min_samples must be at least 2 * min_per_class")
        if not np.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("regularization must be a finite value > 0")
        if self.max_iter < 1 or not (0 < self.tol < 1e-3):
            raise ValueError("max_iter must be >= 1 and tol in (0, 1e-3)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CalibrationConfig":
        return CalibrationConfig(**{k: d[k] for k in CalibrationConfig.__dataclass_fields__ if k in d})


# --------------------------------------------------------------------- numerics
def stable_sigmoid(z: np.ndarray | float) -> np.ndarray:
    """Overflow-free logistic function."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p: np.ndarray | float, eps: float = 1e-15) -> np.ndarray:
    """Log-odds of a probability, clipped to [eps, 1 - eps] for finiteness."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def _nll(z: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    s = a * z + b
    return float(np.sum(np.logaddexp(0.0, s) - y * s))


def validate_calibration_inputs(z: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.float64)
    y_raw = np.asarray(y)
    if z.ndim != 1 or y_raw.ndim != 1:
        raise CalibrationError("Calibration inputs must be one-dimensional arrays")
    if len(z) != len(y_raw):
        raise CalibrationError(f"Length mismatch: {len(z)} log-odds but {len(y_raw)} labels")
    if len(z) == 0:
        raise InsufficientCalibrationData("No local calibration rows were provided")
    if not np.all(np.isfinite(z)):
        raise CalibrationError("Global-model log-odds contain NaN or infinite values")
    try:
        y_num = y_raw.astype(np.float64)
    except (TypeError, ValueError):
        raise CalibrationError("Labels must be numeric 0/1 values") from None
    if not np.all(np.isin(y_num, (0.0, 1.0))):
        raise CalibrationError("Labels must be 0 or 1")
    return z, y_num.astype(np.int64)


def check_sample_requirements(y: np.ndarray, config: CalibrationConfig) -> None:
    n, pos = int(len(y)), int(np.sum(y))
    neg = n - pos
    if pos == 0 or neg == 0:
        raise InsufficientCalibrationData(
            f"Local calibration data has a single class ({pos} positive, {neg} negative); "
            "a calibrator cannot be fitted")
    if n < config.min_samples:
        raise InsufficientCalibrationData(
            f"Only {n} local calibration rows; at least {config.min_samples} are required")
    if min(pos, neg) < config.min_per_class:
        raise InsufficientCalibrationData(
            f"Local calibration data has {pos} positive and {neg} negative rows; at least "
            f"{config.min_per_class} of each class are required")


# --------------------------------------------------------------------- calibrator
@dataclass(frozen=True)
class HospitalCalibrator:
    hospital_id: str
    a: float
    b: float
    n_samples: int
    n_positive: int
    n_negative: int
    regularization: float
    iterations: int
    converged: bool
    calibration_log_loss_before: float  # mean log loss of the global model on calibration rows
    calibration_log_loss_after: float
    format_version: int = CALIBRATOR_FORMAT_VERSION
    method: str = METHOD

    def __post_init__(self) -> None:
        if not self.hospital_id or not isinstance(self.hospital_id, str):
            raise CalibrationError("Calibrator needs a hospital identifier")
        if not (np.isfinite(self.a) and np.isfinite(self.b)):
            raise CalibrationError("Calibrator parameters must be finite")
        if self.a <= 0:
            raise CalibrationError("Calibrator slope must be positive")
        if not self.converged:
            raise CalibrationError("Calibrator did not converge")

    def _z(self, z: Any) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if not np.all(np.isfinite(z)):
            raise CalibrationError("Global-model log-odds contain NaN or infinite values")
        return z

    def transform_logits(self, z: Any) -> np.ndarray:
        return self.a * self._z(z) + self.b

    def predict_proba(self, z: Any) -> np.ndarray:
        """Personalised probability of the positive class from global log-odds."""
        return stable_sigmoid(self.transform_logits(z))

    def adjustment(self, z: Any) -> np.ndarray:
        """Change in log-odds applied by this hospital's calibrator: (a - 1) z + b."""
        return (self.a - 1.0) * self._z(z) + self.b

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "HospitalCalibrator":
        if int(d.get("format_version", -1)) != CALIBRATOR_FORMAT_VERSION:
            raise CalibrationError("Unsupported calibrator format version")
        try:
            return HospitalCalibrator(
                hospital_id=str(d["hospital_id"]), a=float(d["a"]), b=float(d["b"]),
                n_samples=int(d["n_samples"]), n_positive=int(d["n_positive"]),
                n_negative=int(d["n_negative"]), regularization=float(d["regularization"]),
                iterations=int(d["iterations"]), converged=bool(d["converged"]),
                calibration_log_loss_before=float(d["calibration_log_loss_before"]),
                calibration_log_loss_after=float(d["calibration_log_loss_after"]),
                format_version=int(d["format_version"]), method=str(d.get("method", METHOD)))
        except CalibrationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"Calibrator data is incomplete or invalid ({type(exc).__name__}: {exc})") from None


def fit_calibrator(hospital_id: str, z: Any, y: Any,
                   config: CalibrationConfig | None = None) -> HospitalCalibrator:
    """Fit (a_h, b_h) on one hospital's calibration rows. Deterministic (no randomness)."""
    config = config or CalibrationConfig()
    z, y = validate_calibration_inputs(z, y)
    check_sample_requirements(y, config)
    lam = float(config.regularization)
    theta = np.array([1.0, 0.0])  # start at the identity map (= global model)

    def objective(t: np.ndarray) -> float:
        return _nll(z, y, t[0], t[1]) + 0.5 * lam * ((t[0] - 1.0) ** 2 + t[1] ** 2)

    current = objective(theta)
    converged, it = False, 0
    for it in range(1, config.max_iter + 1):
        p = stable_sigmoid(theta[0] * z + theta[1])
        r = p - y
        grad = np.array([np.dot(r, z) + lam * (theta[0] - 1.0), r.sum() + lam * theta[1]])
        w = p * (1.0 - p)
        hess = np.array([[np.dot(w, z * z) + lam, np.dot(w, z)],
                         [np.dot(w, z), w.sum() + lam]])
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while True:  # backtracking line search: guarantees monotone decrease
            cand = theta - t * step
            val = objective(cand)
            if val <= current - 1e-4 * t * float(np.dot(grad, step)) or t < 1e-12:
                break
            t *= 0.5
        theta, previous, current = cand, current, val
        if np.max(np.abs(grad)) <= config.tol * max(1, len(y)) or abs(previous - current) <= 1e-14 * max(1.0, abs(current)):
            converged = True
            break
    a, b = float(theta[0]), float(theta[1])
    if not converged:
        raise CalibrationError(f"Calibration for {hospital_id} did not converge in {config.max_iter} iterations")
    if not (np.isfinite(a) and np.isfinite(b)):
        raise CalibrationError(f"Calibration for {hospital_id} produced non-finite parameters")
    if a <= 0:
        raise CalibrationError(
            f"Calibration for {hospital_id} produced a non-positive slope (a = {a:.4f}); it would "
            "reverse the global model's risk ranking and is rejected")
    n = len(y)
    return HospitalCalibrator(
        hospital_id=hospital_id, a=a, b=b, n_samples=n, n_positive=int(y.sum()),
        n_negative=int(n - y.sum()), regularization=lam, iterations=it, converged=True,
        calibration_log_loss_before=_nll(z, y, 1.0, 0.0) / n,
        calibration_log_loss_after=_nll(z, y, a, b) / n)
