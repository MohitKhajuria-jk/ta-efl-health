import threading

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from federated.server.transport import TrafficLog
from federated.strategies.strategies import (
    MonitorContext,
    MonitoredFedAvg,
    MonitoredFedMedian,
    TrustAwareFedAvg,
)
from federated.strategies.trust import (
    TrustConfig,
    TrustManager,
    cosine,
    performance_signal,
    similarity_signals,
    trust_weights,
    validate_update,
)


# ------------------------------------------------------------------- formulas
def test_trust_weight_formula_and_normalisation():
    w = trust_weights({"a": 0.8, "b": 0.4, "c": 0.9}, {"a": 100, "b": 50, "c": 0},
                      eligible={"a", "b", "c"}, threshold=0.3)
    assert w["c"] == 0.0  # zero examples
    assert w["a"] == pytest.approx(80 / (80 + 20))
    assert w["b"] == pytest.approx(20 / (80 + 20))
    assert sum(w.values()) == pytest.approx(1.0)


def test_trust_threshold_eligibility_and_zero_total():
    w = trust_weights({"a": 0.2, "b": 0.6}, {"a": 10, "b": 10}, {"a", "b"}, 0.3)
    assert w == {"a": 0.0, "b": 1.0}
    w = trust_weights({"a": 0.9, "b": 0.6}, {"a": 10, "b": 10}, {"b"}, 0.3)  # a invalid
    assert w["a"] == 0.0 and w["b"] == 1.0
    assert trust_weights({"a": 0.1}, {"a": 10}, {"a"}, 0.3) is None
    assert trust_weights({}, {}, set(), 0.3) is None


def test_ema_and_instant_score():
    tm = TrustManager(TrustConfig(alpha=0.7, initial_trust=0.5, performance_weight=0.6), ["c1"])
    r = tm.instant_score(perf=1.0, sim=0.5)
    assert r == pytest.approx(0.6 + 0.4 * 0.5)
    assert tm.update("c1", r) == pytest.approx(0.7 * 0.5 + 0.3 * 0.8)


def test_signals_definitions():
    assert performance_signal(0.5) == 0.0 and performance_signal(1.0) == 1.0
    assert performance_signal(0.3) == 0.0 and performance_signal(None) == 0.0
    assert performance_signal(0.75) == pytest.approx(0.5)
    assert cosine(np.zeros(3), np.ones(3)) == 0.0
    sims, raw = similarity_signals({"a": np.array([1.0, 1.0]), "b": np.array([1.0, 0.9]),
                                    "c": np.array([-1.0, -1.0])})
    assert raw["c"] < 0 and sims["c"] == 0.0 and sims["a"] > 0.9


def test_update_validation():
    shapes = [(3,), (1,)]
    assert validate_update([np.zeros(3), np.zeros(1)], shapes) is None
    assert "shape" in validate_update([np.zeros(4), np.zeros(1)], shapes)
    assert "NaN" in validate_update([np.array([0, np.nan, 0]), np.zeros(1)], shapes)
    assert "expected 2" in validate_update([np.zeros(3)], shapes)


@pytest.mark.parametrize("bad", [dict(alpha=1.5), dict(alpha=1.0), dict(exclusion_threshold=-0.1)])
def test_invalid_trust_config(bad):
    with pytest.raises(ValueError):
        TrustConfig(**bad)


# ------------------------------------------------------------- strategies
class _Proxy:
    def __init__(self, cid):
        self.cid = cid


def _ctx(trust_cfg=None, ids=("c1", "c2", "c3", "c4")):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    events = []
    ctx = MonitorContext(X, y, [(3,), (1,)], TrustManager(trust_cfg or TrustConfig(), list(ids)),
                         TrafficLog(), threading.Event(), events.append)
    return ctx, events


def _res(params, n):
    return FitRes(status=Status(Code.OK, "ok"), parameters=ndarrays_to_parameters(params),
                  num_examples=n, metrics={"train_loss": 0.5})


def _setup(strategy, glob):
    strategy._global = [np.asarray(g, dtype=float) for g in glob]


def test_fedavg_uses_flower_weighted_average():
    ctx, _ = _ctx()
    s = MonitoredFedAvg(ctx, inplace=False)
    _setup(s, [np.zeros(3), np.zeros(1)])
    results = [(_Proxy("c1"), _res([np.array([1.0, 0, 0]), np.array([0.0])], 10)),
               (_Proxy("c2"), _res([np.array([3.0, 0, 0]), np.array([1.0])], 30))]
    params, _ = s.aggregate_fit(1, results, [])
    coef, b = parameters_to_ndarrays(params)
    assert coef[0] == pytest.approx((10 * 1 + 30 * 3) / 40)
    assert b[0] == pytest.approx(30 / 40)
    rec = s.history[-1]["clients"]
    assert {r["cid"]: r["weight"] for r in rec} == pytest.approx({"c1": 0.25, "c2": 0.75})


def test_median_uses_flower_fedmedian():
    ctx, _ = _ctx()
    s = MonitoredFedMedian(ctx)
    _setup(s, [np.zeros(3), np.zeros(1)])
    vals = [1.0, 2.0, 100.0]
    results = [(_Proxy(f"c{i+1}"), _res([np.array([v, v, v]), np.array([v])], 10))
               for i, v in enumerate(vals)]
    params, _ = s.aggregate_fit(1, results, [])
    coef, b = parameters_to_ndarrays(params)
    np.testing.assert_allclose(coef, [2.0, 2.0, 2.0])
    assert all(r["weight"] is None for r in s.history[-1]["clients"])


def test_trust_aware_downweights_and_rejects_invalid():
    ctx, events = _ctx(TrustConfig(alpha=0.5, initial_trust=0.5, performance_weight=0.5,
                                   exclusion_threshold=0.3))
    s = TrustAwareFedAvg(ctx)
    _setup(s, [np.zeros(3), np.zeros(1)])
    good = [np.array([1.0, 0.0, 0.0]), np.array([0.0])]  # aligned with the label rule
    good2 = [np.array([0.9, 0.1, 0.0]), np.array([0.0])]
    flipped = [np.array([-5.0, 0.0, 0.0]), np.array([0.0])]  # reversed update
    nan = [np.array([np.nan, 0, 0]), np.array([0.0])]
    results = [(_Proxy("c1"), _res(good, 10)), (_Proxy("c2"), _res(flipped, 10)),
               (_Proxy("c3"), _res(nan, 10)), (_Proxy("c4"), _res(good2, 10))]
    params, _ = s.aggregate_fit(1, results, [])
    rec = {r["cid"]: r for r in s.history[-1]["clients"]}
    assert not rec["c3"]["valid"] and rec["c3"]["weight"] == 0.0
    assert rec["c1"]["trust"] > rec["c2"]["trust"]
    assert rec["c1"]["weight"] > rec["c2"]["weight"]
    assert sum(r["weight"] for r in rec.values()) == pytest.approx(1.0)
    coef, _ = parameters_to_ndarrays(params)
    expected = rec["c1"]["weight"] * 1.0 + rec["c2"]["weight"] * -5.0 + rec["c4"]["weight"] * 0.9
    assert coef[0] == pytest.approx(expected)
    assert any(e["type"] == "round_aggregated" for e in events)


def test_trust_aware_keeps_model_when_no_client_trusted():
    ctx, events = _ctx(TrustConfig(alpha=0.5, initial_trust=0.0, exclusion_threshold=0.9))
    s = TrustAwareFedAvg(ctx)
    _setup(s, [np.zeros(3), np.zeros(1)])
    res = [(_Proxy("c1"), _res([np.array([1.0, 0, 0]), np.array([0.0])], 10))]
    params, _ = s.aggregate_fit(1, res, [])
    assert params is None  # Flower keeps the previous global parameters
    assert any("global model kept" in e.get("message", "") for e in events)


def test_trust_aware_equals_fedavg_when_trust_equal():
    cfg = TrustConfig(alpha=0.0 + 1e-9, initial_trust=1.0, performance_weight=1.0,
                      exclusion_threshold=0.0)
    ctx, _ = _ctx(cfg)
    s = TrustAwareFedAvg(ctx)
    _setup(s, [np.zeros(3), np.zeros(1)])
    # identical models -> identical performance -> identical trust -> weights n_k/N
    p = [np.array([1.0, 0.2, 0.0]), np.array([0.1])]
    params, _ = s.aggregate_fit(1, [(_Proxy("c1"), _res(p, 10)), (_Proxy("c2"), _res(p, 30))], [])
    w = {r["cid"]: r["weight"] for r in s.history[-1]["clients"]}
    assert w == pytest.approx({"c1": 0.25, "c2": 0.75})


def test_relative_performance_signal():
    from federated.strategies.trust import reference_auc
    ref = reference_auc([0.70, 0.72, 0.74, 0.55, None], "relative")
    assert ref == pytest.approx(0.71)
    assert performance_signal(0.72, ref) == 1.0  # at/above the typical client
    assert performance_signal(0.605, ref) == pytest.approx(0.5)
    assert performance_signal(0.45, ref) == 0.0
    assert reference_auc([0.7, 0.8], "absolute") is None
    assert performance_signal(0.75, reference_auc([0.48, 0.49], "relative")) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        TrustConfig(performance_mode="bogus")
