"""Runner that ties the Flower client/server apps into one object.

Two execution backends:

``local``  (default) — :class:`core.fl.sim.InProcessDriver` steps the Flower
           strategy one round at a time. Deterministic, dependency-free and
           what the GUI uses so training can be watched and stopped.
``flower`` — Flower's own simulation runtime (``flwr.simulation.run_simulation``)
           with real SuperNodes/SuperLink. Selected with ``backend="flower"``
           or ``TAEFL_BACKEND=flower``; falls back to ``local`` (with a
           warning) if ``flwr`` is not installed.
"""
from __future__ import annotations

import os

import numpy as np

from .client_app import make_client
from .compat import HAS_FLWR
from .data import build, format_attacks, parse_attacks
from .server_app import build_components
from .sim import InProcessDriver
from .strategy import RoundRecord, make_strategy

DEFAULT_BACKEND = os.environ.get("TAEFL_BACKEND", "local").lower()


class FederatedRunner:
    """Server-side facade over a Flower strategy and its hospital clients."""

    def __init__(self, dataset: str = "heart", n_clients: int = 5,
                 dirichlet_alpha: float = 0.5, attacks: dict[int, str] | None = None,
                 strategy: str = "trust", epochs: int = 3, lr: float = 0.15,
                 seed: int = 42, kappa: float = 4.0,
                 trust_kwargs: dict | None = None, backend: str | None = None):
        self.dataset = dataset
        self.n_clients = n_clients
        self.dirichlet_alpha = dirichlet_alpha
        self.epochs = epochs
        self.lr = lr
        self.attacks = parse_attacks(attacks)
        self.strategy_name = strategy
        self.seed = seed
        self.backend = (backend or DEFAULT_BACKEND).lower()
        if self.backend == "flower" and not HAS_FLWR:
            print("[TA-EFL] flwr not installed — falling back to the local "
                  "in-process Flower driver (pip install flwr[simulation]).")
            self.backend = "local"

        self.data = build(dataset, n_clients, dirichlet_alpha, seed)
        self.clients = [
            make_client(pid, dataset, n_clients, dirichlet_alpha, seed,
                        self.attacks)
            for pid in range(n_clients)
        ]
        self.strategy = make_strategy(
            strategy, self.data, n_clients, epochs=epochs, lr=lr, seed=1,
            kappa=kappa, trust_kwargs=trust_kwargs or {},
        )
        self.driver = InProcessDriver(self.strategy, self.clients)

    # ---- state exposed to the GUI / experiments --------------------------
    @property
    def records(self) -> list[RoundRecord]:
        return self.strategy.records

    @property
    def model(self):
        return self.strategy.model

    @property
    def trust_engine(self):
        return getattr(self.strategy, "trust_engine", None)

    @property
    def kappa(self) -> float:
        return getattr(self.strategy, "kappa", float("nan"))

    @kappa.setter
    def kappa(self, value: float):
        if hasattr(self.strategy, "kappa"):
            self.strategy.kappa = float(value)

    # ---- execution -------------------------------------------------------
    def run_round(self, rnd: int) -> RoundRecord:
        return self.driver.run_round(rnd)

    def run(self, n_rounds: int, callback=None) -> list[RoundRecord]:
        if self.backend == "flower":
            try:
                return self._run_flower(n_rounds, callback)
            except Exception as exc:                     # pragma: no cover
                print(f"[TA-EFL] Flower runtime unavailable ({exc}); "
                      "using the local driver instead.")
                self.backend = "local"
        return self.driver.run(n_rounds, callback)

    # ---- Flower simulation runtime ---------------------------------------
    def _run_flower(self, n_rounds: int, callback=None):
        from flwr.client import ClientApp
        from flwr.server import ServerApp
        from flwr.simulation import run_simulation

        from .client_app import client_fn

        run_config = {
            "dataset": self.dataset,
            "num-clients": self.n_clients,
            "num-server-rounds": n_rounds,
            "dirichlet-alpha": self.dirichlet_alpha,
            "seed": self.seed,
            "strategy": self.strategy_name,
            "attacks": format_attacks(self.attacks),
            "kappa": self.kappa,
            "local-epochs": self.epochs,
            "lr": self.lr,
            "trust-kwargs": getattr(self.strategy, "trust_engine", None)
            and {"alpha": self.strategy.trust_engine.alpha,
                 "beta": self.strategy.trust_engine.beta,
                 "gamma": self.strategy.trust_engine.gamma,
                 "tau": self.strategy.trust_engine.tau} or {},
        }

        def server_fn(context):
            comps = build_components(run_config, on_round_end=callback)
            self.strategy = comps.strategy      # keep records/trust reachable
            return comps

        run_simulation(
            server_app=ServerApp(server_fn=server_fn),
            client_app=ClientApp(client_fn=client_fn),
            num_supernodes=self.n_clients,
            backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
        )
        return self.strategy.records


# Backwards-compatible alias: the old hand-rolled server class.
FLServer = FederatedRunner
