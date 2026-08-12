"""Deterministic federated data construction.

Everything here is a pure function of ``(dataset, num_partitions, alpha,
seed)``, so a Flower SuperNode running in a separate Ray worker rebuilds
exactly the same shard the server assumes it has — no shared state, no
pickling of patient data.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..datasets import get_spec
from ..dataset import standardize
from ..metrics import train_test_split
from ..partition import dirichlet_partition

HOSPITALS = [
    "City Hospital", "Green Valley Hospital", "Life Care Hospital",
    "Sunrise Hospital", "Hope Medical Center", "Riverside Clinic",
    "Mountain View Hospital", "Central Medical", "Lakeside Hospital",
    "Unity Health",
]


@dataclass
class FederatedData:
    spec: object
    source: str
    feature_names: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_ref: np.ndarray          # shared public reference set (SHAP signatures)
    bg_mean: np.ndarray
    shards: list[np.ndarray]
    mu: np.ndarray
    sd: np.ndarray

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def partition(self, pid: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self.shards[pid]
        return self.X_train[idx], self.y_train[idx]

    def hospital_name(self, pid: int) -> str:
        return HOSPITALS[pid % len(HOSPITALS)]


@lru_cache(maxsize=32)
def build(dataset: str = "heart", num_partitions: int = 5, alpha: float = 0.5,
          seed: int = 42, test_size: float = 0.25,
          ref_size: int = 200) -> FederatedData:
    spec = get_spec(dataset)
    df, source = spec.load()
    feature_names = [c for c in spec.feature_names if c in df.columns]
    X = df[feature_names].to_numpy(dtype=float)
    y = df[spec.target].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    mu, sd, (X_train, X_test) = standardize(X_train, X_test)

    rng = np.random.default_rng(seed)
    ref_idx = rng.choice(len(X_test), size=min(ref_size, len(X_test)),
                         replace=False)
    X_ref = X_test[ref_idx]

    min_size = max(10, min(30, len(X_train) // (num_partitions * 4)))
    shards = dirichlet_partition(y_train, num_partitions, alpha, seed=seed,
                                 min_size=min_size)
    return FederatedData(
        spec=spec, source=source, feature_names=feature_names,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        X_ref=X_ref, bg_mean=X_ref.mean(axis=0), shards=shards, mu=mu, sd=sd,
    )


def parse_attacks(spec: str | dict | None) -> dict[int, str]:
    """``"2:label_flip,4:noise"`` (Flower run-config friendly) -> dict."""
    if not spec:
        return {}
    if isinstance(spec, dict):
        return {int(k): v for k, v in spec.items()}
    out = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        cid, _, atk = part.partition(":")
        out[int(cid)] = atk or "none"
    return out


def format_attacks(attacks: dict[int, str]) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(attacks.items())
                    if v and v != "none")
