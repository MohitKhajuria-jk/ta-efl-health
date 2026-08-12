"""Flower ServerApp for TA-EFL.

Usable both from Python (:mod:`core.fl.runner`) and from the Flower CLI
(``flwr run .``) once a ``pyproject.toml`` with a ``[tool.flwr.app]`` section
points at ``core.fl.server_app:app`` and ``core.fl.client_app:app``.
"""
from __future__ import annotations

from .compat import ServerApp, ServerAppComponents, ServerConfig
from .data import build
from .strategy import make_strategy


def build_components(run_config: dict, on_round_end=None) -> ServerAppComponents:
    cfg = dict(run_config or {})
    n_clients = int(cfg.get("num-clients", 5))
    data = build(
        dataset=str(cfg.get("dataset", "heart")),
        num_partitions=n_clients,
        alpha=float(cfg.get("dirichlet-alpha", 0.5)),
        seed=int(cfg.get("seed", 42)),
    )
    strategy = make_strategy(
        str(cfg.get("strategy", "trust")), data, n_clients,
        epochs=int(cfg.get("local-epochs", 3)),
        lr=float(cfg.get("lr", 0.15)),
        seed=int(cfg.get("model-seed", 1)),
        kappa=float(cfg.get("kappa", 4.0)),
        trust_kwargs=cfg.get("trust-kwargs") or {},
        on_round_end=on_round_end,
    )
    config = ServerConfig(num_rounds=int(cfg.get("num-server-rounds", 30)))
    return ServerAppComponents(strategy=strategy, config=config)


def server_fn(context):
    return build_components(dict(getattr(context, "run_config", {}) or {}))


app = ServerApp(server_fn=server_fn)
