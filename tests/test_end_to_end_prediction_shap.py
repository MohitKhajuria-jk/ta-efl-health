"""End-to-end: genuine Flower experiment -> load artifact -> predict -> SHAP explanation."""
from pathlib import Path

import numpy as np
import pytest

from ml.datasets.loader import load_dataset, make_splits
from ml.explainability.shap_service import (
    ExplanationError,
    explain_prediction,
    explanation_to_dict,
    global_importance,
)
from ml.models.artifact import ArtifactError, load_artifact
from ml.prediction.service import InputValidationError, predict
from ml.preprocessing.pipeline import transformed_columns


@pytest.fixture(scope="module")
def artifact(bc_experiment):
    rec = bc_experiment["record"]
    return load_artifact(bc_experiment["storage"] / rec["artifact_dir"], "breast_cancer")


@pytest.fixture(scope="module")
def test_row(bc_experiment):
    cfg = bc_experiment["config"]
    d = load_dataset("breast_cancer", Path("data"))
    s = make_splits(d, cfg.seed, cfg.test_fraction, cfg.val_fraction)  # same split as training
    return s.X_test.iloc[0].to_dict(), int(s.y_test[0])


def test_artifact_bundles_everything(artifact, bc_experiment):
    assert artifact.dataset_key == "breast_cancer"
    assert artifact.threshold == 0.5
    assert artifact.experiment_id == bc_experiment["record"]["id"]
    assert artifact.background.shape[1] == len(artifact.transformed_names)
    assert artifact.metrics["roc_auc"] == bc_experiment["record"]["test_metrics"]["roc_auc"]


def test_artifact_rejected_for_other_dataset(bc_experiment):
    rec = bc_experiment["record"]
    with pytest.raises(ArtifactError, match="never used"):
        load_artifact(bc_experiment["storage"] / rec["artifact_dir"], "heart")


def test_prediction_matches_model(artifact, test_row):
    row, _ = test_row
    pr = predict(artifact, row)
    x = pr.x_transformed.reshape(1, -1)
    assert pr.probability_positive == pytest.approx(artifact.model.predict_proba(x)[0, 1])
    assert pr.predicted_class == int(pr.probability_positive >= artifact.threshold)
    assert pr.predicted_label in ("Malignant", "Benign")


def test_invalid_inputs_rejected(artifact, test_row):
    row, _ = test_row
    bad = dict(row)
    bad["mean radius"] = -4
    with pytest.raises(InputValidationError) as e:
        predict(artifact, bad)
    assert "mean radius" in e.value.errors
    incomplete = dict(row)
    incomplete.pop("worst area")
    with pytest.raises(InputValidationError):
        predict(artifact, incomplete)
    with pytest.raises(InputValidationError):
        predict(artifact, {**row, "mean texture": "abc"})


def test_shap_explanation_is_additive_and_matches_linear_formula(artifact, test_row):
    row, _ = test_row
    pr = predict(artifact, row)
    exp = explain_prediction(artifact, pr, explained_class=1)
    total = sum(c.shap_value for c in exp.contributions)
    assert exp.base_value + total == pytest.approx(pr.decision_value, abs=1e-6)
    assert exp.output_probability == pytest.approx(pr.probability_positive, abs=1e-9)
    # independent-masker linear SHAP: phi_j = w_j (x_j - mean_bg_j), grouped by feature
    w = artifact.model.coef_.ravel()
    phi = w * (pr.x_transformed - artifact.background.mean(axis=0))
    cols = transformed_columns(artifact.spec)
    by_feature = {}
    for c, v in zip(cols, phi):
        by_feature[c.feature] = by_feature.get(c.feature, 0.0) + v
    for c in exp.contributions:
        assert c.shap_value == pytest.approx(by_feature[c.feature], abs=1e-8)


def test_explanation_text_generated_from_values(artifact, test_row):
    row, _ = test_row
    pr = predict(artifact, row)
    exp = explain_prediction(artifact, pr)
    top = exp.contributions[0]
    assert any(top.label in s for s in exp.statements)
    assert f"{top.shap_value:+.2f}" in " ".join(exp.statements)
    assert "log-odds" in exp.summary and any("causation" in c for c in exp.caveats)
    d = explanation_to_dict(exp)
    assert len(d["contributions"]) == 30


def test_explaining_other_class_negates(artifact, test_row):
    row, _ = test_row
    pr = predict(artifact, row)
    e1 = explain_prediction(artifact, pr, explained_class=1)
    e0 = explain_prediction(artifact, pr, explained_class=0)
    m1 = {c.feature: c.shap_value for c in e1.contributions}
    for c in e0.contributions:
        assert c.shap_value == pytest.approx(-m1[c.feature])


def test_explanation_rejects_mismatched_prediction(artifact, test_row):
    row, _ = test_row
    pr = predict(artifact, row)
    pr.experiment_id = "other"
    with pytest.raises(ExplanationError):
        explain_prediction(artifact, pr)


def test_shap_failure_is_reported_not_fabricated(artifact, test_row, monkeypatch):
    import ml.explainability.shap_service as svc
    row, _ = test_row
    pr = predict(artifact, row)

    def boom(*_a, **_k):
        raise ExplanationError("SHAP computation failed: simulated")
    monkeypatch.setattr(svc, "_shap_matrix", boom)
    with pytest.raises(ExplanationError):
        svc.explain_prediction(artifact, pr)
    assert 0.0 <= pr.probability_positive <= 1.0  # prediction is untouched


def test_global_importance(artifact):
    g = global_importance(artifact)
    assert g.n == len(artifact.background)
    vals = [v for _, v in g.features]
    assert vals == sorted(vals, reverse=True) and all(v >= 0 for v in vals)
    assert np.isfinite(vals).all()


def test_feature_statements_follow_contribution_signs(artifact, test_row):
    from ml.explainability.shap_service import feature_statements
    row, _ = test_row
    exp = explain_prediction(artifact, predict(artifact, row))
    st = feature_statements(exp, k=5)
    assert 0 < len(st) <= 5
    for (direction, text), c in zip(st, [c for c in exp.contributions if abs(c.shap_value) >= 1e-9]):
        assert direction == (1 if c.shap_value > 0 else -1)
        assert c.label in text and f"{abs(c.shap_value):.2f}" in text
