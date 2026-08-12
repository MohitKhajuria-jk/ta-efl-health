"""Build a complete federated session: dataset -> partitions -> Flower run."""
from __future__ import annotations

import numpy as np

from .datasets import DEFAULT_DATASET, get_spec
from .fl.data import HOSPITALS as DEFAULT_HOSPITALS      # noqa: F401 (compat)
from .fl.runner import FederatedRunner


class Session:
    """One experiment: a dataset, N hospitals, and a federated strategy."""

    def __init__(
        self, n_clients: int = 5, dirichlet_alpha: float = 0.5,
        attacks: dict[int, str] | None = None, strategy: str = "trust",
        n_rounds: int = 30, seed: int = 42, trust_kwargs: dict | None = None,
        dataset: str = DEFAULT_DATASET, kappa: float = 4.0,
        epochs: int = 3, lr: float = 0.15, backend: str | None = None,
    ):
        self.n_rounds = n_rounds
        self.dataset = get_spec(dataset).key
        self.spec = get_spec(dataset)

        self.server = FederatedRunner(
            dataset=self.dataset, n_clients=n_clients,
            dirichlet_alpha=dirichlet_alpha, attacks=attacks,
            strategy=strategy, epochs=epochs, lr=lr, seed=seed, kappa=kappa,
            trust_kwargs=trust_kwargs or {}, backend=backend,
        )
        data = self.server.data
        self.data = data
        self.source = data.source
        self.feature_names = data.feature_names
        self.mu, self.sd = data.mu, data.sd
        self.X_test, self.y_test = data.X_test, data.y_test
        self.bg_mean = data.bg_mean
        self.clients = self.server.clients

    # ---- execution --------------------------------------------------------
    def run(self, callback=None):
        return self.server.run(self.n_rounds, callback=callback)

    # ---- helpers ----------------------------------------------------------
    @property
    def model(self):
        return self.server.model

    @property
    def records(self):
        return self.server.records

    def standardize_row(self, raw: np.ndarray) -> np.ndarray:
        """Raw feature vector -> standardized space used by the model."""
        return (np.asarray(raw, dtype=float) - self.mu) / self.sd

    def destandardize_row(self, x: np.ndarray) -> np.ndarray:
        """Standardized vector -> original clinical units (inverse of above)."""
        return np.asarray(x, dtype=float) * self.sd + self.mu

    def feature_labels(self) -> dict[str, str]:
        return self.spec.labels

    def describe(self) -> str:
        return (f"{self.spec.name} — {self.spec.task} | source: {self.source} | "
                f"{len(self.feature_names)} features, "
                f"{len(self.data.X_train) + len(self.X_test)} patients")
