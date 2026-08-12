"""Smoke tests: run with `python -m pytest tests` or `python tests/test_smoke.py`."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.datasets import DATASETS, get_spec                      # noqa: E402
from core.fl import HAS_FLWR, build, parse_attacks                # noqa: E402
from core.fl.strategy import STRATEGIES                           # noqa: E402
from core.session import Session                                  # noqa: E402


def test_every_dataset_loads():
    for key, spec in DATASETS.items():
        df, src = spec.load()
        assert len(df) > 100, key
        assert spec.target in df.columns
        assert set(np.unique(df[spec.target])) == {0, 1}, key
        assert all(f in df.columns for f in spec.feature_names), key
        assert src


def test_partitions_are_deterministic():
    a = build("diabetes", 5, 0.5, 42)
    b = build("diabetes", 5, 0.5, 42)
    for sa, sb in zip(a.shards, b.shards):
        assert np.array_equal(sa, sb)
    assert sum(len(s) for s in a.shards) == len(a.y_train)


def test_parse_attacks_roundtrip():
    assert parse_attacks("2:label_flip,4:noise") == {2: "label_flip", 4: "noise"}
    assert parse_attacks("") == {}


def test_all_strategies_train():
    for name in STRATEGIES:
        s = Session(n_clients=4, strategy=name, n_rounds=5, dataset="heart")
        s.run()
        assert len(s.server.records) == 5
        assert 0.0 <= s.server.records[-1].metrics["auc"] <= 1.0


def test_trust_penalizes_attackers():
    s = Session(n_clients=6, attacks={2: "label_flip", 4: "noise"},
                strategy="trust", n_rounds=15, dataset="heart")
    s.run()
    t = s.server.trust_engine.trust
    honest = [t[c.cid] for c in s.clients if c.attack == "none"]
    attackers = [t[c.cid] for c in s.clients if c.attack != "none"]
    assert np.mean(honest) > np.mean(attackers) + 0.2


def test_trust_beats_fedavg_under_attack():
    kw = dict(n_clients=6, attacks={2: "label_flip", 4: "noise"},
              n_rounds=20, dataset="diabetes")
    trust = Session(strategy="trust", **kw); trust.run()
    fedavg = Session(strategy="fedavg", **kw); fedavg.run()
    assert (trust.server.records[-1].metrics["auc"]
            >= fedavg.server.records[-1].metrics["auc"])


def test_prediction_pipeline():
    spec = get_spec("stroke")
    s = Session(n_clients=4, strategy="trust", n_rounds=5, dataset="stroke")
    s.run()
    raw = np.array([f.default for f in spec.features])
    x = s.standardize_row(raw)
    p = float(s.model.predict_proba(x[None, :])[0])
    assert 0.0 <= p <= 1.0
    phi = s.model.shap_values(x[None, :], s.bg_mean)[0]
    assert phi.shape == (len(s.feature_names),)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed "
          f"(flwr {'installed' if HAS_FLWR else 'not installed — local driver'})")
