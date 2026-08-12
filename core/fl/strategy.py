"""Flower aggregation strategies for TA-EFL.

* :class:`TrustWeightedFedAvg` — this project's method. Server-side trust
  scoring from (a) the client's update direction versus the cohort median,
  (b) the *extrapolated attribution signature* of the submitted weights, and
  (c) an EMA over history; aggregation weights ``p_k ∝ n_k · T_k^γ`` with a
  hard drop below ``τ``.
* :class:`PlainFedAvg` — classic sample-size weighted averaging.
* :class:`CoordinateMedian` — coordinate-wise median (robust baseline).

All three implement the standard ``flwr.server.strategy.Strategy`` interface
and are therefore usable with Flower's own simulation/deployment runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..model import LogisticModel, evaluate as evaluate_model
from ..trust import TrustEngine
from .compat import (
    Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Status,
    Strategy, ndarrays_to_parameters, parameters_to_ndarrays,
)
from .task import parameters_to_theta, shap_signature, theta_to_parameters


@dataclass
class RoundRecord:
    """One federated round, as consumed by the GUI and the experiment scripts."""
    rnd: int
    metrics: dict
    trust: np.ndarray | None = None
    weights: np.ndarray | None = None
    components: list | None = None
    client_ids: list[int] = field(default_factory=list)


class TAEFLStrategy(Strategy):
    """Common scaffolding: centralized evaluation, round records, callbacks."""

    name = "base"

    def __init__(self, data, n_clients: int, epochs: int = 3, lr: float = 0.15,
                 seed: int = 1, on_round_end: Callable[[RoundRecord], None] | None = None):
        self.data = data
        self.n_clients = n_clients
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.on_round_end = on_round_end
        self.records: list[RoundRecord] = []
        self.model = LogisticModel(data.n_features, seed=seed)
        self._pending: dict = {}

    # ---- Flower Strategy API --------------------------------------------
    def initialize_parameters(self, client_manager=None) -> Parameters:
        return theta_to_parameters(self.model.get_params())

    def configure_fit(self, server_round: int, parameters, client_manager):
        config = {"server_round": server_round, "local_epochs": self.epochs,
                  "lr": self.lr}
        clients = _sample_all(client_manager, self.n_clients)
        return [(c, FitIns(parameters, config)) for c in clients]

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        return []                      # evaluation is centralized on the server

    def aggregate_evaluate(self, server_round: int, results, failures):
        if not results:
            return None, {}
        n = sum(r.num_examples for _, r in results)
        loss = sum(r.loss * r.num_examples for _, r in results) / max(n, 1)
        return float(loss), {}

    def aggregate_fit(self, server_round: int, results, failures):
        thetas, sizes, cids = [], [], []
        for _proxy, res in results:
            thetas.append(parameters_to_theta(res.parameters))
            sizes.append(int(res.num_examples))
            cids.append(int(res.metrics.get("cid", len(cids))))
        order = np.argsort(cids)                    # deterministic client order
        thetas = [thetas[i] for i in order]
        sizes = [sizes[i] for i in order]
        cids = [cids[i] for i in order]
        if not thetas:
            return None, {}

        theta_g = self.model.get_params()
        new_theta, extra = self.aggregate_thetas(server_round, theta_g,
                                                 thetas, sizes, cids)
        self.model.set_params(new_theta)
        self._pending = dict(extra, client_ids=cids)
        return theta_to_parameters(new_theta), {"strategy": self.name}

    def evaluate(self, server_round: int, parameters):
        """Centralized evaluation on the held-out test set (and bookkeeping)."""
        if server_round == 0:
            return None
        theta = parameters_to_theta(parameters)
        self.model.set_params(theta)
        metrics = evaluate_model(self.model, self.data.X_test, self.data.y_test)
        rec = RoundRecord(
            rnd=server_round, metrics=metrics,
            trust=self._pending.get("trust"),
            weights=self._pending.get("weights"),
            components=self._pending.get("components"),
            client_ids=self._pending.get("client_ids", []),
        )
        self.records.append(rec)
        if self.on_round_end:
            self.on_round_end(rec)
        return 1.0 - float(metrics["accuracy"]), {
            "accuracy": float(metrics["accuracy"]), "auc": float(metrics["auc"]),
            "f1": float(metrics["f1"]),
        }

    # ---- to be provided by each strategy --------------------------------
    def aggregate_thetas(self, rnd, theta_g, thetas, sizes, cids):
        raise NotImplementedError


class PlainFedAvg(TAEFLStrategy):
    """FedAvg: average weighted by the number of local training samples."""
    name = "fedavg"

    def aggregate_thetas(self, rnd, theta_g, thetas, sizes, cids):
        ns = np.array(sizes, dtype=float)
        w = ns / ns.sum()
        return np.sum([wi * t for wi, t in zip(w, thetas)], axis=0), {"weights": w}


class CoordinateMedian(TAEFLStrategy):
    """Coordinate-wise median — a classic Byzantine-robust baseline."""
    name = "median"

    def aggregate_thetas(self, rnd, theta_g, thetas, sizes, cids):
        return np.median(np.stack(thetas), axis=0), {}


class TrustWeightedFedAvg(TAEFLStrategy):
    """Trust-aware explainable aggregation (the TA-EFL contribution)."""
    name = "trust"

    def __init__(self, *args, kappa: float = 4.0, trust_kwargs: dict | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.kappa = kappa
        self.trust_engine = TrustEngine(self.n_clients, **(trust_kwargs or {}))

    def aggregate_thetas(self, rnd, theta_g, thetas, sizes, cids):
        # Extrapolated attribution signature, computed SERVER-side: amplify
        # each client's update direction by kappa, then take the signed SHAP
        # signature of the resulting model on the shared public reference set.
        # A label-flipper's extrapolated model attributes risk with inverted
        # signs, and this cannot be spoofed since it derives from the weights
        # the client actually submitted.
        sigs = [
            shap_signature(theta_g + self.kappa * (t - theta_g),
                           self.data.X_ref, self.data.bg_mean)
            for t in thetas
        ]
        trust = self.trust_engine.score_round(theta_g, thetas, sigs).copy()
        weights = self.trust_engine.aggregation_weights(sizes)
        new_theta = np.sum([w * t for w, t in zip(weights, thetas)], axis=0)
        return new_theta, {"trust": trust, "weights": weights,
                           "components": self.trust_engine.last_components}


STRATEGIES = {
    "trust": TrustWeightedFedAvg,
    "fedavg": PlainFedAvg,
    "median": CoordinateMedian,
}


def make_strategy(name: str, data, n_clients: int, **kwargs) -> TAEFLStrategy:
    name = (name or "trust").lower()
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGIES)}")
    cls = STRATEGIES[name]
    if cls is not TrustWeightedFedAvg:
        kwargs.pop("kappa", None)
        kwargs.pop("trust_kwargs", None)
    return cls(data, n_clients, **kwargs)


def _sample_all(client_manager, n_clients: int):
    """Select every connected client (hospitals are few and always available)."""
    if client_manager is None:
        return []
    if hasattr(client_manager, "wait_for"):
        client_manager.wait_for(num_clients=n_clients, timeout=120)
    clients = client_manager.sample(num_clients=n_clients,
                                    min_num_clients=n_clients)
    return list(clients)
