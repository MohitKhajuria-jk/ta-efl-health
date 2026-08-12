"""Shared model plumbing between the Flower client and server apps."""
from __future__ import annotations

import numpy as np

from ..model import LogisticModel
from .compat import ndarrays_to_parameters, parameters_to_ndarrays

ATTACK_NONE = "none"
ATTACK_LABEL_FLIP = "label_flip"
ATTACK_NOISE = "noise"
ATTACK_SCALE = "scale"
ATTACKS = (ATTACK_NONE, ATTACK_LABEL_FLIP, ATTACK_NOISE, ATTACK_SCALE)


# ---- parameter vector <-> Flower NDArrays --------------------------------
def theta_to_ndarrays(theta: np.ndarray) -> list[np.ndarray]:
    """Split the flat parameter vector into Flower's list-of-arrays format."""
    return [np.asarray(theta[:-1], dtype=np.float64),
            np.asarray([theta[-1]], dtype=np.float64)]


def ndarrays_to_theta(ndarrays) -> np.ndarray:
    w, b = ndarrays[0], np.atleast_1d(ndarrays[1])
    return np.concatenate([np.asarray(w, dtype=np.float64),
                           np.asarray(b, dtype=np.float64)])


def theta_to_parameters(theta: np.ndarray):
    return ndarrays_to_parameters(theta_to_ndarrays(theta))


def parameters_to_theta(parameters) -> np.ndarray:
    return ndarrays_to_theta(parameters_to_ndarrays(parameters))


# ---- local training ------------------------------------------------------
def local_train(theta_g: np.ndarray, X: np.ndarray, y: np.ndarray,
                epochs: int, lr: float, seed: int) -> np.ndarray:
    model = LogisticModel(len(theta_g) - 1)
    model.set_params(theta_g)
    model.fit_epochs(X, y, epochs=epochs, lr=lr, seed=seed)
    return model.get_params()


def apply_attack(theta: np.ndarray, theta_g: np.ndarray, attack: str,
                 seed: int, noise_sigma: float = 1.0,
                 scale_factor: float = 8.0) -> np.ndarray:
    """Post-training corruption of the update a malicious hospital submits."""
    rng = np.random.default_rng(seed)
    if attack == ATTACK_NOISE:
        return theta + rng.normal(0, noise_sigma, size=theta.shape)
    if attack == ATTACK_SCALE:
        return theta_g + scale_factor * (theta - theta_g)
    return theta


def shap_signature(theta: np.ndarray, X_ref: np.ndarray,
                   bg_mean: np.ndarray) -> np.ndarray:
    model = LogisticModel(len(theta) - 1)
    model.set_params(theta)
    return model.shap_signature(X_ref, bg_mean)
