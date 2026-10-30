from pathlib import Path

import numpy as np
import pytest
from flwr.common import Code, FitIns, GetParametersIns, ndarrays_to_parameters, parameters_to_ndarrays

from federated.clients.attacks import AttackConfig, corrupt_update, poison_labels
from federated.clients.hospital_client import PREPROCESSOR_KEY, HospitalClient
from federated.server.runner import ExperimentConfig, ExperimentError, run_experiment
from federated.server.transport import InProcessClientProxy, TrafficLog
from federated.simulation.partition import partition_indices
from ml.datasets.loader import load_dataset, make_splits
from ml.preprocessing.pipeline import fit_preprocessor


@pytest.fixture(scope="module")
def bc_setup():
    d = load_dataset("breast_cancer", Path("data"))
    s = make_splits(d, seed=0)
    pre = fit_preprocessor(d.spec, s.X_train)  # training rows only
    return d.spec, s, pre


def _client(bc_setup, attack=None, idx=slice(0, 120)):
    spec, s, pre = bc_setup
    return HospitalClient("client_1", 1, spec, s.X_train.iloc[idx], s.y_train[idx], 0, attack)


def _cfg(bc_setup, **kw):
    return {PREPROCESSOR_KEY: bc_setup[2].to_bytes(), **kw}


def test_partitions_disjoint_and_complete():
    y = np.array([0] * 70 + [1] * 30)
    for mode in ("iid", "dirichlet"):
        parts = partition_indices(y, 4, mode, seed=1, dirichlet_alpha=1.0, min_size=10)
        allidx = np.concatenate(parts)
        assert len(allidx) == len(set(allidx)) == 100
        assert min(len(p) for p in parts) >= 10
    with pytest.raises(ValueError):
        partition_indices(y, 20, "iid", seed=1)


def test_client_fit_and_evaluate_return_flower_types(bc_setup):
    c = _client(bc_setup)
    d = c.n_features
    params, n, metrics = c.fit([np.zeros(d), np.zeros(1)],
                               _cfg(bc_setup, learning_rate=0.05, batch_size=16, local_epochs=2,
                                    server_round=1))
    assert [p.shape for p in params] == [(d,), (1,)]
    assert isinstance(n, int) and n == c.num_train
    assert np.any(params[0] != 0)
    loss, n_eval, m = c.evaluate(params, _cfg(bc_setup))
    assert isinstance(loss, float) and n_eval == c.num_eval and 0 <= m["accuracy"] <= 1


def test_flower_message_roundtrip_through_proxy(bc_setup):
    c = _client(bc_setup)
    traffic = TrafficLog()
    proxy = InProcessClientProxy("client_1", c.to_client(), traffic)
    d = c.n_features
    ins = FitIns(ndarrays_to_parameters([np.zeros(d), np.zeros(1)]),
                 _cfg(bc_setup, learning_rate=0.05, batch_size=32, local_epochs=1, server_round=3))
    res = proxy.fit(ins, timeout=None, group_id=3)
    assert res.status.code == Code.OK and res.num_examples == c.num_train
    assert len(parameters_to_ndarrays(res.parameters)) == 2
    assert traffic.bytes_to_clients[3] > 0 and traffic.bytes_from_clients[3] > 0
    gp = proxy.get_parameters(GetParametersIns({}), None, 0)
    assert len(parameters_to_ndarrays(gp.parameters)) == 2


def test_client_failure_is_recorded_with_id(bc_setup):
    c = _client(bc_setup)
    traffic = TrafficLog()
    proxy = InProcessClientProxy("client_1", c.to_client(), traffic)
    bad = FitIns(ndarrays_to_parameters([np.zeros(3), np.zeros(1)]), _cfg(bc_setup, server_round=2))
    with pytest.raises(Exception):
        proxy.fit(bad, None, 2)
    assert traffic.failures[2][0][0] == "client_1"


def test_attack_transformations():
    rng = np.random.default_rng(0)
    g = [np.zeros(3), np.zeros(1)]
    honest = [np.array([1.0, 2.0, 3.0]), np.array([0.5])]
    flip = corrupt_update(g, honest, AttackConfig("sign_flip", scale=2.0), rng)
    np.testing.assert_allclose(flip[0], [-2, -4, -6])
    scaled = corrupt_update(g, honest, AttackConfig("scaling", scale=10.0), rng)
    np.testing.assert_allclose(scaled[0], [10, 20, 30])
    noisy = corrupt_update(g, honest, AttackConfig("gaussian_noise", noise_std=1.0), rng)
    assert not np.allclose(noisy[0], honest[0])
    assert list(poison_labels(np.array([0, 1, 1]), AttackConfig("label_flip"))) == [1, 0, 0]
    with pytest.raises(ValueError):
        AttackConfig("teleport")  # type: ignore[arg-type]


def test_label_flip_client_moves_opposite_to_honest(bc_setup):
    d = bc_setup[0]
    honest = _client(bc_setup)
    bad = _client(bc_setup, AttackConfig("label_flip"))
    zeros = [np.zeros(honest.n_features), np.zeros(1)]
    cfg = _cfg(bc_setup, learning_rate=0.05, batch_size=16, local_epochs=3, server_round=1)
    h, _, _ = honest.fit(zeros, cfg)
    b, _, _ = bad.fit(zeros, cfg)
    assert np.dot(h[0], b[0]) < 0


def test_config_validation():
    with pytest.raises(ExperimentError):
        ExperimentConfig(num_clients=1).validate()
    with pytest.raises(ExperimentError):
        ExperimentConfig(malicious_clients=[9]).validate()
    with pytest.raises(ExperimentError):
        ExperimentConfig(malicious_clients=[1], attack_type="none").validate()
    with pytest.raises(ExperimentError):
        ExperimentConfig(strategy="krum").validate()


def test_missing_dataset_aborts_experiment(tmp_path):
    with pytest.raises(ExperimentError, match="heart.csv"):
        run_experiment(ExperimentConfig(dataset_key="heart", num_rounds=1), tmp_path, tmp_path)


def test_genuine_flower_experiment_record(bc_experiment):
    rec = bc_experiment["record"]
    cfg = bc_experiment["config"]
    assert rec["status"] == "completed" and rec["rounds_completed"] == cfg.num_rounds
    assert rec["framework"]["flower"]
    assert len(rec["rounds"]) == cfg.num_rounds
    # every round: each client reported, weights sum to 1 (trust-aware)
    for rnd in rec["rounds"]:
        assert len(rnd["clients"]) == cfg.num_clients
        assert sum(c["weight"] for c in rnd["clients"]) == pytest.approx(1.0)
        assert rnd["bytes_from_clients"] > 0
    assert rec["test_metrics"]["split"] == "test"
    assert rec["test_metrics"]["n"] == rec["splits"]["test"]
    assert rec["test_metrics"]["roc_auc"] > 0.9
    # Flower's own log output was observed during the run
    assert any(e["type"] == "flower_log" for e in bc_experiment["events"])
    # the configured sign-flip attacker ends with lower trust than every honest client
    final = rec["final_trust"]
    assert all(final["client_4"] < final[c] for c in ("client_1", "client_2", "client_3"))
    assert rec["detection"]["configured_attackers"] == ["client_4"]


def test_stop_before_first_round_saves_nothing(tmp_path):
    import threading
    ev = threading.Event()
    ev.set()
    with pytest.raises(ExperimentError, match="before any round"):
        run_experiment(ExperimentConfig(dataset_key="breast_cancer", num_rounds=3), Path("data"),
                       tmp_path, stop_event=ev)


def test_stop_midway_keeps_partial_model(tmp_path):
    import threading
    ev = threading.Event()

    def on_event(e):
        if e["type"] == "round_aggregated" and e["round"] == 2:
            ev.set()
    rec = run_experiment(ExperimentConfig(dataset_key="breast_cancer", num_rounds=10,
                                          malicious_clients=[], attack_type="none"),
                         Path("data"), tmp_path, on_event=on_event, stop_event=ev)
    assert rec["status"] == "stopped" and rec["rounds_completed"] == 2
    assert rec["test_metrics"]["roc_auc"] is not None


@pytest.mark.parametrize("strategy", ["fedavg", "median"])
def test_baseline_strategies_run(tmp_path, strategy):
    rec = run_experiment(ExperimentConfig(dataset_key="breast_cancer", num_rounds=3, strategy=strategy,
                                          malicious_clients=[], attack_type="none"),
                         Path("data"), tmp_path)
    assert rec["status"] == "completed"
    assert rec["rounds"][0]["trust_used_for_weighting"] is False


@pytest.mark.parametrize("strategy", ["trust", "fedavg", "median"])
def test_malformed_updates_are_rejected_every_round(tmp_path, strategy):
    rec = run_experiment(ExperimentConfig(dataset_key="breast_cancer", num_rounds=3, strategy=strategy,
                                          malicious_clients=[2], attack_type="invalid_update"),
                         Path("data"), tmp_path)
    assert rec["status"] == "completed"
    for rnd in rec["rounds"]:
        c2 = next(c for c in rnd["clients"] if c["cid"] == "client_2")
        assert not c2["valid"] and "NaN" in c2["reason"] and c2["excluded"]
        if strategy != "median":
            assert c2["weight"] == 0.0
            assert sum(c["weight"] for c in rnd["clients"]) == pytest.approx(1.0)
    assert rec["test_metrics"]["roc_auc"] is not None  # global model stayed finite


def test_crashing_client_is_recorded_and_training_continues(tmp_path, monkeypatch):
    from federated.clients import hospital_client as hc

    original = hc.HospitalClient.fit

    def flaky(self, parameters, config):
        if self.client_id == "client_3" and int(config["server_round"]) == 2:
            raise RuntimeError("simulated hospital outage")
        return original(self, parameters, config)

    monkeypatch.setattr(hc.HospitalClient, "fit", flaky)
    rec = run_experiment(ExperimentConfig(dataset_key="breast_cancer", num_rounds=3,
                                          malicious_clients=[], attack_type="none"),
                         Path("data"), tmp_path)
    assert rec["status"] == "completed" and rec["rounds_completed"] == 3
    r2 = rec["rounds"][1]
    assert r2["failures"] == [{"cid": "client_3", "error": "RuntimeError: simulated hospital outage"}]
    assert "client_3" not in {c["cid"] for c in r2["clients"]}
    assert sum(c["weight"] for c in r2["clients"]) == pytest.approx(1.0)
    assert "client_3" in r2["non_participating_trust"]  # trust carried, not invented


def test_fedavg_with_no_valid_update_keeps_previous_model():
    import threading

    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

    from federated.strategies.strategies import MonitorContext, MonitoredFedAvg
    from federated.strategies.trust import TrustConfig, TrustManager

    X = np.random.default_rng(0).normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    ctx = MonitorContext(X, y, [(3,), (1,)], TrustManager(TrustConfig(), ["c1"]), TrafficLog(),
                         threading.Event(), lambda e: None)
    s = MonitoredFedAvg(ctx, inplace=False)
    s._global = [np.zeros(3), np.zeros(1)]

    class P:
        cid = "c1"
    bad = FitRes(Status(Code.OK, ""), ndarrays_to_parameters([np.full(3, np.inf), np.zeros(1)]), 10, {})
    params, _ = s.aggregate_fit(1, [(P(), bad)], [])
    assert params is None  # Flower keeps the previous global parameters
