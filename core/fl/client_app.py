"""Flower ClientApp — one SuperNode per simulated hospital.

Each client trains the shared logistic risk model on its own Dirichlet shard
and returns the updated parameters. Malicious hospitals corrupt either their
labels (label flipping) or the update they submit (Gaussian noise, scaled
model poisoning). Clients never send explanations: the server derives the
attribution signature from the submitted weights, so it cannot be spoofed.
"""
from __future__ import annotations

import numpy as np

from .compat import ClientApp, NumPyClient
from .data import build, parse_attacks
from .task import (
    ATTACK_LABEL_FLIP, ATTACK_NONE, apply_attack, local_train,
    ndarrays_to_theta, theta_to_ndarrays,
)
from ..model import LogisticModel


class HospitalClient(NumPyClient):
    """A single hospital participating in federated training."""

    def __init__(self, cid: int, name: str, X: np.ndarray, y: np.ndarray,
                 attack: str = ATTACK_NONE, seed: int = 0):
        self.cid = cid
        self.name = name
        self.X = X
        self.y = y.copy()
        self.attack = attack or ATTACK_NONE
        self.seed = seed
        if self.attack == ATTACK_LABEL_FLIP:
            self.y = 1 - self.y                 # poisoned labels

    # ---- Flower NumPyClient API -----------------------------------------
    @property
    def n_samples(self) -> int:
        return len(self.y)

    def get_parameters(self, config=None):
        model = LogisticModel(self.X.shape[1], seed=self.seed)
        return theta_to_ndarrays(model.get_params())

    def fit(self, parameters, config=None):
        config = config or {}
        rnd = int(config.get("server_round", 0))
        epochs = int(config.get("local_epochs", 3))
        lr = float(config.get("lr", 0.15))

        theta_g = ndarrays_to_theta(parameters)
        theta = local_train(theta_g, self.X, self.y, epochs, lr,
                            seed=self.seed * 1000 + rnd)
        theta = apply_attack(theta, theta_g, self.attack,
                             seed=self.seed * 7919 + rnd)
        metrics = {"cid": self.cid, "name": self.name, "attack": self.attack}
        return theta_to_ndarrays(theta), self.n_samples, metrics

    def evaluate(self, parameters, config=None):
        theta = ndarrays_to_theta(parameters)
        model = LogisticModel(len(theta) - 1)
        model.set_params(theta)
        p = np.clip(model.predict_proba(self.X), 1e-7, 1 - 1e-7)
        loss = float(-np.mean(self.y * np.log(p) + (1 - self.y) * np.log(1 - p)))
        acc = float(np.mean((p >= 0.5).astype(int) == self.y))
        return loss, self.n_samples, {"accuracy": acc, "cid": self.cid}


def make_client(pid: int, dataset: str = "heart", num_partitions: int = 5,
                alpha: float = 0.5, seed: int = 42,
                attacks: dict[int, str] | str | None = None) -> HospitalClient:
    """Build the hospital client that owns partition ``pid``."""
    data = build(dataset, num_partitions, alpha, seed)
    X, y = data.partition(pid)
    atk = parse_attacks(attacks).get(pid, ATTACK_NONE)
    return HospitalClient(pid, data.hospital_name(pid), X, y, atk, seed + pid)


def client_fn(context):
    """Flower entry point: ``context`` carries node and run configuration."""
    node_cfg = dict(getattr(context, "node_config", {}) or {})
    run_cfg = dict(getattr(context, "run_config", {}) or {})
    pid = int(node_cfg.get("partition-id", node_cfg.get("partition_id", 0)))
    num_partitions = int(node_cfg.get("num-partitions",
                                      run_cfg.get("num-clients", 5)))
    client = make_client(
        pid,
        dataset=str(run_cfg.get("dataset", "heart")),
        num_partitions=num_partitions,
        alpha=float(run_cfg.get("dirichlet-alpha", 0.5)),
        seed=int(run_cfg.get("seed", 42)),
        attacks=run_cfg.get("attacks", ""),
    )
    return client.to_client()


app = ClientApp(client_fn=client_fn)
