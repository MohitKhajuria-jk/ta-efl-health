"""Deterministic in-process driver for Flower ClientApps and Strategies.

Flower's own simulation runtime (``flwr.simulation.run_simulation``) spawns
Ray actors and owns the round loop, which is ideal for headless runs but
cannot be stepped one round at a time — something the desktop GUI needs for
live progress and a responsive Stop button.

This module executes the *same* ``HospitalClient`` and ``Strategy`` objects
through the same ``configure_fit -> fit -> aggregate_fit -> evaluate``
sequence, synchronously and one round per call. It is a driver, not a
re-implementation of Flower: no transport, scheduling or fault tolerance.
"""
from __future__ import annotations

import numpy as np

from .compat import Code, FitRes, Status, ndarrays_to_parameters


class LocalClientProxy:
    """Stands in for a Flower ``ClientProxy`` around an in-process client."""

    def __init__(self, cid: int, client):
        self.cid = str(cid)
        self.client = client

    @property
    def properties(self):
        return {"cid": self.cid}

    def fit(self, ins, timeout=None, group_id=None) -> FitRes:
        from .compat import parameters_to_ndarrays
        ndarrays, n, metrics = self.client.fit(
            parameters_to_ndarrays(ins.parameters), dict(ins.config))
        return FitRes(status=Status(code=Code.OK, message="OK"),
                      parameters=ndarrays_to_parameters(ndarrays),
                      num_examples=n, metrics=metrics)

    def evaluate(self, ins, timeout=None, group_id=None):
        from .compat import EvaluateRes, parameters_to_ndarrays
        loss, n, metrics = self.client.evaluate(
            parameters_to_ndarrays(ins.parameters), dict(ins.config))
        return EvaluateRes(status=Status(code=Code.OK, message="OK"),
                           loss=float(loss), num_examples=n, metrics=metrics)


class LocalClientManager:
    """Minimal ``ClientManager``: every hospital is always available."""

    def __init__(self, proxies: list[LocalClientProxy]):
        self.proxies = list(proxies)

    def num_available(self) -> int:
        return len(self.proxies)

    def wait_for(self, num_clients: int, timeout: int = 0) -> bool:
        return len(self.proxies) >= num_clients

    def sample(self, num_clients: int, min_num_clients: int | None = None,
               criterion=None):
        return self.proxies[:num_clients]

    def all(self):
        return {p.cid: p for p in self.proxies}


class InProcessDriver:
    """Steps a Flower strategy through federated rounds, synchronously."""

    def __init__(self, strategy, clients):
        self.strategy = strategy
        self.clients = list(clients)
        self.proxies = [LocalClientProxy(c.cid, c) for c in self.clients]
        self.client_manager = LocalClientManager(self.proxies)
        self.parameters = strategy.initialize_parameters(self.client_manager)

    def run_round(self, rnd: int):
        instructions = self.strategy.configure_fit(
            rnd, self.parameters, self.client_manager)
        results, failures = [], []
        for proxy, ins in instructions:
            try:
                results.append((proxy, proxy.fit(ins)))
            except Exception as exc:                     # pragma: no cover
                failures.append(exc)
        params, _metrics = self.strategy.aggregate_fit(rnd, results, failures)
        if params is not None:
            self.parameters = params
        self.strategy.evaluate(rnd, self.parameters)
        return self.strategy.records[-1]

    def run(self, n_rounds: int, callback=None):
        for r in range(1, n_rounds + 1):
            rec = self.run_round(r)
            if callback:
                callback(rec)
        return self.strategy.records
