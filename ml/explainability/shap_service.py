"""SHAP explanations for individual predictions and global feature importance.

Computed with ``shap.LinearExplainer`` on the trained model in the model's
*log-odds* output space, with the server-validation split (transformed by the
saved preprocessor) as the background distribution (independent masker).

Additivity: base_value + sum(SHAP) == model log-odds for the explained instance;
this is checked before an explanation is returned.

One-hot columns of a categorical feature are summed into one contribution for
that feature. For a linear model with an independent masker each column's SHAP
value is w_j * (x_j - E[x_j]), so the group sum is exact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ml.models.artifact import ModelArtifact
from ml.prediction.service import PredictionResult
from ml.preprocessing.pipeline import transformed_columns


class ExplanationError(RuntimeError):
    pass


@dataclass
class FeatureContribution:
    feature: str
    label: str
    value_display: str
    shap_value: float  # log-odds contribution toward the explained class
    reference_display: str  # what the contribution is measured against


@dataclass
class LocalExplanation:
    explained_class: int
    explained_label: str
    base_value: float  # expected log-odds over the background
    output_value: float  # log-odds for this instance (for the explained class)
    contributions: list[FeatureContribution]  # sorted by |shap|, descending
    summary: str
    statements: list[str]
    caveats: list[str]
    n_background: int
    output_scale: str = "log-odds"

    @property
    def base_probability(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.base_value))

    @property
    def output_probability(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.output_value))


@dataclass
class GlobalImportance:
    features: list[tuple[str, float]]  # (label, mean |SHAP|) descending
    split: str
    n: int
    caveats: list[str] = field(default_factory=list)


def _explainer(artifact: ModelArtifact):
    try:
        import shap  # noqa: WPS433 - optional heavy import
    except Exception as exc:  # noqa: BLE001
        raise ExplanationError(f"SHAP is not available: {exc}") from exc
    bg = np.asarray(artifact.background, dtype=np.float64)
    if bg.ndim != 2 or len(bg) == 0:
        raise ExplanationError("Model artifact has no background data for SHAP")
    try:
        masker = shap.maskers.Independent(bg, max_samples=len(bg))
        return shap.LinearExplainer(artifact.model, masker)
    except Exception as exc:  # noqa: BLE001
        raise ExplanationError(f"Could not build SHAP explainer: {exc}") from exc


def _shap_matrix(artifact: ModelArtifact, X: np.ndarray) -> tuple[np.ndarray, float]:
    explainer = _explainer(artifact)
    try:
        res = explainer(np.atleast_2d(X))
    except Exception as exc:  # noqa: BLE001
        raise ExplanationError(f"SHAP computation failed: {exc}") from exc
    values = np.asarray(res.values, dtype=np.float64)
    base = np.asarray(res.base_values, dtype=np.float64).ravel()
    if values.ndim == 3:  # (n, d, classes) - take the positive class
        values = values[..., -1]
        base = np.asarray(res.base_values)[..., -1].ravel()
    if values.shape[1] != X.shape[-1] or not np.all(np.isfinite(values)):
        raise ExplanationError("SHAP returned values of unexpected shape or non-finite values")
    return values, float(base[0])


def _group(artifact: ModelArtifact, values: np.ndarray) -> dict[str, float]:
    cols = transformed_columns(artifact.spec)
    grouped: dict[str, float] = {f: 0.0 for f in artifact.feature_names}
    for col, v in zip(cols, values):
        grouped[col.feature] += float(v)
    return grouped


def _numeric_reference(artifact: ModelArtifact) -> dict[str, float]:
    """Background mean of each numeric feature on its original scale."""
    try:
        return artifact.preprocessor.numeric_reference(artifact.background)
    except Exception:  # noqa: BLE001 - reference text is optional
        return {}


def explain_prediction(artifact: ModelArtifact, prediction: PredictionResult,
                       explained_class: int | None = None, top_k: int = 3) -> LocalExplanation:
    if prediction.experiment_id != artifact.experiment_id:
        raise ExplanationError("Prediction and model artifact do not match")
    spec = artifact.spec
    cls = prediction.predicted_class if explained_class is None else int(explained_class)
    if cls not in (0, 1):
        raise ExplanationError("Explained class must be 0 or 1")
    x = prediction.x_transformed.reshape(1, -1)
    values, base = _shap_matrix(artifact, x)
    phi = values[0]
    fx = float(artifact.model.decision_function(x)[0])
    if abs(base + phi.sum() - fx) > 1e-6 * max(1.0, abs(fx)):
        raise ExplanationError(
            f"SHAP additivity check failed (base {base:.6f} + sum {phi.sum():.6f} != "
            f"output {fx:.6f}); explanation withheld")
    sign = 1.0 if cls == 1 else -1.0
    grouped = {k: sign * v for k, v in _group(artifact, phi).items()}
    base_c, out_c = sign * base, sign * fx
    refs = _numeric_reference(artifact)
    label = artifact.class_labels[cls]

    contribs = []
    for name, v in grouped.items():
        f = spec.feature(name)
        raw = prediction.inputs.get(name)
        if f.kind == "categorical":
            ref = "relative to the category mix of the validation patients"
        elif name in refs:
            ref = f"relative to the validation average of {refs[name]:.3g} {f.unit}".rstrip()
        else:
            ref = "relative to the validation average"
        contribs.append(FeatureContribution(name, f.label, f.display_value(raw), v, ref))
    contribs.sort(key=lambda c: abs(c.shap_value), reverse=True)

    up = [c for c in contribs if c.shap_value > 1e-9][:top_k]
    down = [c for c in contribs if c.shap_value < -1e-9][:top_k]

    def fmt(cs: list[FeatureContribution]) -> str:
        return ", ".join(f"{c.label} = {c.value_display} ({c.shap_value:+.2f})" for c in cs)

    summary = (
        f"Explaining the model's output for '{label}'. Its log-odds for this patient is "
        f"{out_c:+.2f} (model probability {1 / (1 + math.exp(-out_c)):.2f}), versus a baseline "
        f"of {base_c:+.2f} (probability {1 / (1 + math.exp(-base_c)):.2f}) - the average model "
        f"output over {len(artifact.background)} server-validation records. The feature "
        f"contributions below add up exactly to the difference of {out_c - base_c:+.2f}."
    )
    statements: list[str] = []
    if up:
        statements.append(f"Increased the model output for '{label}': {fmt(up)}.")
    if down:
        statements.append(f"Decreased the model output for '{label}': {fmt(down)}.")
    if contribs and abs(contribs[0].shap_value) > 1e-9:
        top = contribs[0]
        direction = "toward" if top.shap_value > 0 else "away from"
        statements.append(
            f"Largest single contribution: {top.label} ({top.value_display}), "
            f"{abs(top.shap_value):.2f} log-odds {direction} '{label}' {top.reference_display}.")
    if not up and not down:
        statements.append("All features contributed approximately zero; the output equals the baseline.")
    caveats = [
        "Values are SHAP contributions in log-odds units. They are not percentage-point "
        "changes in probability.",
        "SHAP describes how this model uses its inputs. It does not show medical causation "
        "and is not a diagnosis.",
        "This is an explanation of one prediction (local), not global feature importance.",
    ]
    return LocalExplanation(cls, label, base_c, out_c, contribs, summary, statements, caveats,
                            len(artifact.background))


def global_importance(artifact: ModelArtifact) -> GlobalImportance:
    values, _ = _shap_matrix(artifact, artifact.background)
    cols = transformed_columns(artifact.spec)
    per_feature: dict[str, np.ndarray] = {}
    for j, col in enumerate(cols):
        per_feature.setdefault(col.feature, np.zeros(len(values)))
        per_feature[col.feature] = per_feature[col.feature] + values[:, j]
    imp = [(artifact.spec.feature(f).label, float(np.mean(np.abs(v))))
           for f, v in per_feature.items()]
    imp.sort(key=lambda t: t[1], reverse=True)
    return GlobalImportance(imp, "server validation", len(values), [
        "Mean absolute SHAP value (log-odds) over the server-validation split; "
        "it summarises the model across patients and does not describe any individual."])


def explanation_to_dict(e: LocalExplanation) -> dict[str, Any]:
    return {
        "explained_class": e.explained_class, "explained_label": e.explained_label,
        "base_value": e.base_value, "output_value": e.output_value,
        "output_scale": e.output_scale, "summary": e.summary, "statements": e.statements,
        "caveats": e.caveats,
        "contributions": [{"feature": c.feature, "label": c.label, "value": c.value_display,
                           "shap": c.shap_value} for c in e.contributions],
    }


def feature_statements(e: LocalExplanation, k: int = 5) -> list[tuple[int, str]]:
    """Per-feature sentences for the strongest contributions: (direction, text).

    direction is +1 when the feature increased the model output for the explained
    class, -1 when it decreased it. Text is generated from the actual values.
    """
    out: list[tuple[int, str]] = []
    for c in e.contributions[:k]:
        if abs(c.shap_value) < 1e-9:
            continue
        verb = "increased" if c.shap_value > 0 else "decreased"
        size = abs(c.shap_value)
        strength = "strongly " if size >= 0.5 else ("slightly " if size < 0.1 else "")
        out.append((1 if c.shap_value > 0 else -1,
                    f"{c.label} ({c.value_display}) {strength}{verb} the model's output for "
                    f"'{e.explained_label}' by {size:.2f} log-odds {c.reference_display}."))
    return out
