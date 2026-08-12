"""Non-IID data partitioning across simulated hospital clients."""
from __future__ import annotations

import numpy as np


def dirichlet_partition(
    y: np.ndarray, n_clients: int, alpha: float = 0.5, seed: int = 0,
    min_size: int = 30,
) -> list[np.ndarray]:
    """Split indices into n_clients non-IID shards via a Dirichlet distribution.

    Lower alpha => more heterogeneous label distributions per client
    (simulates hospitals with very different patient populations).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    classes = np.unique(y)
    while True:
        idx_per_client: list[list[int]] = [[] for _ in range(n_clients)]
        for c in classes:
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for k, shard in enumerate(np.split(idx_c, cuts)):
                idx_per_client[k].extend(shard.tolist())
        sizes = [len(s) for s in idx_per_client]
        if min(sizes) >= min_size:
            break
    return [np.array(sorted(s)) for s in idx_per_client]
