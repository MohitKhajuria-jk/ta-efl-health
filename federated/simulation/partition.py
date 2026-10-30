"""Split the pooled training data into disjoint hospital (client) partitions."""
from __future__ import annotations

from typing import Literal

import numpy as np

PartitionMode = Literal["iid", "dirichlet"]


def partition_indices(y: np.ndarray, num_clients: int, mode: PartitionMode, seed: int,
                      dirichlet_alpha: float = 0.5, min_size: int = 10) -> list[np.ndarray]:
    """Return one index array per client; indices are disjoint and cover ``y``.

    * ``iid`` - stratified round-robin assignment (similar size and class mix).
    * ``dirichlet`` - label-skewed non-IID split: for each class, client shares are
      drawn from Dirichlet(alpha). Resampled until every client has ``min_size``
      samples (raises if that is impossible after 200 attempts).
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    if num_clients < 2:
        raise ValueError("At least 2 clients are required")
    if n < num_clients * min_size:
        raise ValueError(f"{n} training samples cannot give {num_clients} clients "
                         f"at least {min_size} samples each")
    rng = np.random.default_rng(seed)
    if mode == "iid":
        parts: list[list[int]] = [[] for _ in range(num_clients)]
        offset = 0
        for c in np.unique(y):
            idx = rng.permutation(np.flatnonzero(y == c))
            for j, i in enumerate(idx):
                parts[(j + offset) % num_clients].append(int(i))
            offset += len(idx)
        return [np.sort(np.array(p, dtype=int)) for p in parts]
    if mode == "dirichlet":
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0")
        for _ in range(200):
            parts = [[] for _ in range(num_clients)]
            for c in np.unique(y):
                idx = rng.permutation(np.flatnonzero(y == c))
                shares = rng.dirichlet(np.full(num_clients, dirichlet_alpha))
                cuts = (np.cumsum(shares)[:-1] * len(idx)).astype(int)
                for k, chunk in enumerate(np.split(idx, cuts)):
                    parts[k].extend(int(i) for i in chunk)
            if min(len(p) for p in parts) >= min_size:
                return [np.sort(np.array(p, dtype=int)) for p in parts]
        raise ValueError("Could not draw a Dirichlet partition with the minimum client size; "
                         "increase alpha or reduce the number of clients")
    raise ValueError(f"Unknown partition mode '{mode}'")
