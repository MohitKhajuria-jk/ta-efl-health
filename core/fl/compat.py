"""Flower compatibility layer.

The federated code in this package is written against the Flower (``flwr``)
APIs: ``NumPyClient``/``ClientApp`` on the client side and a custom
``Strategy`` on the server side. Flower is the intended runtime and is listed
in requirements.txt.

If ``flwr`` is not installed, this module provides drop-in equivalents of the
handful of ``flwr.common`` message types the project uses, so the exact same
client and strategy code can still be executed by the lightweight in-process
driver in :mod:`core.fl.sim`. Nothing here re-implements Flower's transport,
scheduling or fault tolerance — it only keeps the object model available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np

try:                                                       # pragma: no cover
    from flwr.common import (
        Code, Config, EvaluateIns, EvaluateRes, FitIns, FitRes, Metrics,
        NDArrays, Parameters, Scalar, Status, ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.client import ClientApp, NumPyClient
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import Strategy
    HAS_FLWR = True
    FLWR_IMPORT_ERROR = None
except Exception as _exc:                                  # pragma: no cover
    HAS_FLWR = False
    FLWR_IMPORT_ERROR = _exc

    NDArrays = list
    Scalar = Any
    Metrics = dict
    Config = dict

    class Code:
        OK = "OK"

    @dataclass
    class Status:
        code: Any = Code.OK
        message: str = ""

    @dataclass
    class Parameters:
        tensors: list[bytes]
        tensor_type: str = "numpy.ndarray"

    @dataclass
    class FitIns:
        parameters: Parameters
        config: dict = field(default_factory=dict)

    @dataclass
    class FitRes:
        status: Status
        parameters: Parameters
        num_examples: int
        metrics: dict = field(default_factory=dict)

    @dataclass
    class EvaluateIns:
        parameters: Parameters
        config: dict = field(default_factory=dict)

    @dataclass
    class EvaluateRes:
        status: Status
        loss: float
        num_examples: int
        metrics: dict = field(default_factory=dict)

    def ndarrays_to_parameters(ndarrays) -> Parameters:
        tensors = []
        for arr in ndarrays:
            buf = BytesIO()
            np.save(buf, np.asarray(arr), allow_pickle=False)
            tensors.append(buf.getvalue())
        return Parameters(tensors=tensors, tensor_type="numpy.ndarray")

    def parameters_to_ndarrays(parameters: Parameters):
        return [np.load(BytesIO(t), allow_pickle=False) for t in parameters.tensors]

    class NumPyClient:                       # minimal stand-in
        def get_parameters(self, config): raise NotImplementedError
        def fit(self, parameters, config): raise NotImplementedError
        def evaluate(self, parameters, config): raise NotImplementedError
        def to_client(self): return self

    class ClientApp:
        def __init__(self, client_fn=None, **kwargs):
            self.client_fn = client_fn

    class ServerApp:
        def __init__(self, server_fn=None, **kwargs):
            self.server_fn = server_fn

    @dataclass
    class ServerConfig:
        num_rounds: int = 1

    @dataclass
    class ServerAppComponents:
        strategy: Any = None
        config: ServerConfig = field(default_factory=ServerConfig)

    class ClientProxy:                        # noqa: D401 - placeholder
        cid: str = ""

    class Strategy:
        """Interface mirror of ``flwr.server.strategy.Strategy``."""
        def initialize_parameters(self, client_manager): raise NotImplementedError
        def configure_fit(self, server_round, parameters, client_manager): raise NotImplementedError
        def aggregate_fit(self, server_round, results, failures): raise NotImplementedError
        def configure_evaluate(self, server_round, parameters, client_manager): raise NotImplementedError
        def aggregate_evaluate(self, server_round, results, failures): raise NotImplementedError
        def evaluate(self, server_round, parameters): raise NotImplementedError


__all__ = [
    "HAS_FLWR", "FLWR_IMPORT_ERROR", "Code", "Status", "Parameters", "FitIns",
    "FitRes", "EvaluateIns", "EvaluateRes", "NDArrays", "Scalar", "Metrics",
    "Config", "ndarrays_to_parameters", "parameters_to_ndarrays", "NumPyClient",
    "ClientApp", "ServerApp", "ServerAppComponents", "ServerConfig",
    "ClientProxy", "Strategy",
]
