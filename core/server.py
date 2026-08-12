"""Backwards-compatible entry point for the federated server.

The hand-rolled round loop that used to live here has been replaced by the
Flower-based implementation in :mod:`core.fl`. ``FLServer`` is now an alias
of :class:`core.fl.runner.FederatedRunner`.
"""
from __future__ import annotations

from .fl.runner import FederatedRunner, FLServer          # noqa: F401
from .fl.strategy import RoundRecord                      # noqa: F401

__all__ = ["FederatedRunner", "FLServer", "RoundRecord"]
