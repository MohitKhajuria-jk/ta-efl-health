from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from ml.datasets.loader import load_dataset, make_splits
from ml.evaluation.metrics import classification_metrics
from ml.models.logistic import MiniBatchLogisticRegression
from ml.preprocessing.pipeline import fit_preprocessor, transform


@pytest.fixture(scope="module")
def bc_arrays():
    d = load_dataset("breast_cancer", Path("data"))
    s = make_splits(d, seed=0)
    pre = fit_preprocessor(d.spec, s.X_train)  # training rows only
    return (transform(pre, d.spec, s.X_train), s.y_train,
            transform(pre, d.spec, s.X_test), s.y_test)


def test_training_reduces_loss_and_generalises(bc_arrays):
    Xtr, ytr, Xte, yte = bc_arrays
    m = MiniBatchLogisticRegression(learning_rate=0.05, batch_size=32, epochs=1, random_state=0)
    m.init_params(Xtr.shape[1])
    before = m.loss(Xtr, ytr)
    m.partial_fit(Xtr, ytr, epochs=20)
    assert m.loss(Xtr, ytr) < before
    met = classification_metrics(yte, m.predict_proba(Xte)[:, 1], 0.5, "test")
    assert met["roc_auc"] > 0.95  # genuine held-out evaluation


def test_probabilities_valid_and_consistent(bc_arrays):
    Xtr, ytr, Xte, _ = bc_arrays
    m = MiniBatchLogisticRegression(epochs=5, random_state=1).fit(Xtr, ytr)
    p = m.predict_proba(Xte)
    assert p.shape == (len(Xte), 2)
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(p[:, 1], 1 / (1 + np.exp(-m.decision_function(Xte))))
    assert set(m.predict(Xte)) <= {0, 1}


def test_param_roundtrip_and_validation(bc_arrays):
    Xtr, ytr, Xte, _ = bc_arrays
    m = MiniBatchLogisticRegression(epochs=2, random_state=2).fit(Xtr, ytr)
    params = m.get_params_arrays()
    m2 = MiniBatchLogisticRegression().init_params(Xtr.shape[1]).set_params_arrays(params)
    np.testing.assert_allclose(m.predict_proba(Xte), m2.predict_proba(Xte))
    with pytest.raises(ValueError):
        m2.set_params_arrays([np.zeros(3), np.zeros(1)])
    with pytest.raises(ValueError):
        m2.set_params_arrays([params[0]])
    clone(m)  # scikit-learn compatible


def test_training_is_reproducible(bc_arrays):
    Xtr, ytr, _, _ = bc_arrays
    a = MiniBatchLogisticRegression(epochs=3, random_state=5).fit(Xtr, ytr).get_params_arrays()
    b = MiniBatchLogisticRegression(epochs=3, random_state=5).fit(Xtr, ytr).get_params_arrays()
    np.testing.assert_array_equal(a[0], b[0])


def test_non_finite_inputs_and_divergence_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        MiniBatchLogisticRegression().fit(np.array([[np.nan, 1.0], [0.0, 1.0]]), np.array([0, 1]))
    # lr * l2 = 3 makes every step multiply w by about -2: genuine numerical divergence
    X = np.array([[1.0, 0.5], [-1.0, -0.5]])
    with np.errstate(all="ignore"), pytest.raises(FloatingPointError, match="diverged"):
        MiniBatchLogisticRegression(learning_rate=1.0, l2=3.0, epochs=1500).fit(X, np.array([1, 0]))


def test_metrics_match_hand_computation():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.8, 0.9])
    m = classification_metrics(y, p, 0.5, "unit")
    # predictions: 0 1 0 1 1 -> TP=2 FP=1 FN=1 TN=1
    assert m["confusion_matrix"] == [[1, 1], [1, 2]]
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(5 / 6)
    single = classification_metrics(np.array([1, 1]), np.array([0.2, 0.9]), 0.5, "unit")
    assert single["roc_auc"] is None and single["roc_curve"] is None
