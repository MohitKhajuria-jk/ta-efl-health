"""Backwards-compatible client shim.

Hospital clients are now Flower ``NumPyClient`` instances; see
:mod:`core.fl.client_app`.
"""
from __future__ import annotations

from .fl.client_app import HospitalClient, HospitalClient as Client, make_client
from .fl.task import (                                     # noqa: F401
    ATTACK_LABEL_FLIP, ATTACK_NOISE, ATTACK_NONE, ATTACK_SCALE, ATTACKS,
)

__all__ = [
    "HospitalClient", "Client", "make_client", "ATTACKS", "ATTACK_NONE",
    "ATTACK_LABEL_FLIP", "ATTACK_NOISE", "ATTACK_SCALE",
]
