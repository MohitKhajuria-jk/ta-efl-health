"""Flower-based federated learning layer for TA-EFL."""
from .compat import HAS_FLWR
from .data import FederatedData, build, format_attacks, parse_attacks
from .runner import FederatedRunner, FLServer
from .strategy import (
    CoordinateMedian, PlainFedAvg, RoundRecord, STRATEGIES, TrustWeightedFedAvg,
    make_strategy,
)
from .task import (
    ATTACK_LABEL_FLIP, ATTACK_NOISE, ATTACK_NONE, ATTACK_SCALE, ATTACKS,
)

__all__ = [
    "HAS_FLWR", "FederatedData", "build", "parse_attacks", "format_attacks",
    "FederatedRunner", "FLServer", "RoundRecord", "STRATEGIES", "make_strategy",
    "TrustWeightedFedAvg", "PlainFedAvg", "CoordinateMedian",
    "ATTACKS", "ATTACK_NONE", "ATTACK_LABEL_FLIP", "ATTACK_NOISE", "ATTACK_SCALE",
]
