"""Controlled malicious-client behaviours for robustness experiments ONLY.

An attack is configured for specific simulated clients by the experiment
harness. The server/strategy never receives this configuration; whether a
client is *detected* as suspicious is decided only from what it sends.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

AttackType = Literal["none", "sign_flip", "scaling", "gaussian_noise", "label_flip",
                     "invalid_update"]

ATTACK_DESCRIPTIONS: dict[str, str] = {
    "none": "No attack.",
    "sign_flip": "Sends w_g - s*(w_local - w_g): the local update reversed and scaled by s.",
    "scaling": "Sends w_g + s*(w_local - w_g): the honest update boosted by factor s.",
    "gaussian_noise": "Sends w_local + N(0, sigma^2) noise on every parameter.",
    "label_flip": "Trains on flipped labels (1 - y) and sends the resulting model (data poisoning).",
    "invalid_update": "Sends parameters containing NaN (malformed update; tests input validation).",
}


@dataclass(frozen=True)
class AttackConfig:
    type: AttackType = "none"
    scale: float = 3.0  # s for sign_flip / scaling
    noise_std: float = 1.0  # sigma for gaussian_noise

    def __post_init__(self) -> None:
        if self.type not in ATTACK_DESCRIPTIONS:
            raise ValueError(f"Unknown attack type '{self.type}'")
        if self.scale <= 0 or self.noise_std <= 0:
            raise ValueError("Attack scale and noise_std must be > 0")

    def to_dict(self) -> dict:
        return asdict(self)


def poison_labels(y: np.ndarray, attack: AttackConfig) -> np.ndarray:
    if attack.type == "label_flip":
        return 1 - np.asarray(y).astype(int)
    return y


def corrupt_update(global_params: list[np.ndarray], local_params: list[np.ndarray],
                   attack: AttackConfig, rng: np.random.Generator) -> list[np.ndarray]:
    """Transform the honest local parameters into what a malicious client sends."""
    if attack.type in ("none", "label_flip"):
        return [p.copy() for p in local_params]
    if attack.type == "sign_flip":
        return [g - attack.scale * (lp - g) for g, lp in zip(global_params, local_params)]
    if attack.type == "scaling":
        return [g + attack.scale * (lp - g) for g, lp in zip(global_params, local_params)]
    if attack.type == "gaussian_noise":
        return [lp + rng.normal(0.0, attack.noise_std, size=lp.shape) for lp in local_params]
    if attack.type == "invalid_update":
        bad = [lp.copy() for lp in local_params]
        bad[0][0] = np.nan
        return bad
    raise ValueError(attack.type)  # pragma: no cover
