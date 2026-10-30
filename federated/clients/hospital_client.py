"""Flower client representing one simulated hospital.

Raw patient rows never leave this object. Communication with the server:

* ``get_properties({"request": "preprocessing_statistics"})`` -> aggregate statistics of
  the LOCAL TRAINING rows only (counts, means, sums of squared deviations, category
  counts). No individual row is sent.
* ``fit`` / ``evaluate`` receive the global ``[coef, intercept]`` plus a config containing
  the fitted preprocessing parameters (JSON bytes). The client transforms its own rows
  locally, trains, and returns parameters, its example count and aggregate metrics.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from sklearn.metrics import accuracy_score

from federated.clients.attacks import AttackConfig, corrupt_update, poison_labels
from ml.datasets.schema import DatasetSpec
from ml.evaluation.metrics import auc_or_none
from ml.models.logistic import MiniBatchLogisticRegression, log_loss
from ml.preprocessing.pipeline import SchemaPreprocessor, local_statistics, transformed_columns

STATS_REQUEST = "preprocessing_statistics"
PREPROCESSOR_KEY = "preprocessor"


def encode_statistics(stats: dict) -> dict[str, Scalar]:
    """Flower Properties must be scalars: ship the aggregate statistics as JSON bytes."""
    return {"statistics": json.dumps(stats, sort_keys=True).encode("utf-8")}


def decode_statistics(props: dict[str, Scalar]) -> dict:
    raw = props.get("statistics")
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("Client did not return preprocessing statistics")
    return json.loads(bytes(raw).decode("utf-8"))


class HospitalClient(NumPyClient):
    def __init__(
        self,
        client_id: str,
        client_index: int,
        spec: DatasetSpec,
        X_local: pd.DataFrame,
        y_local: np.ndarray,
        seed: int,
        attack: AttackConfig | None = None,
        local_eval_fraction: float = 0.2,
    ) -> None:
        self.client_id = client_id
        self.client_index = client_index
        self.seed = seed
        self.spec = spec
        self.attack = attack or AttackConfig()
        X_local = X_local.reset_index(drop=True)
        y = np.asarray(y_local).astype(int)
        if len(y) < 2:
            raise ValueError(f"{client_id} needs at least 2 local records")
        rng = np.random.default_rng([seed, client_index, 0])
        order = rng.permutation(len(y))
        n_eval = max(1, int(round(local_eval_fraction * len(y))))
        eval_idx, train_idx = order[:n_eval], order[n_eval:]
        self._raw_train, self._y_train = X_local.iloc[train_idx].reset_index(drop=True), y[train_idx]
        self._raw_eval, self._y_eval = X_local.iloc[eval_idx].reset_index(drop=True), y[eval_idx]
        self.n_features = len(transformed_columns(spec))
        self._pre_hash: str | None = None
        self._X_train: np.ndarray | None = None
        self._X_eval: np.ndarray | None = None

    # Only aggregate, non-identifying information is exposed.
    @property
    def num_train(self) -> int:
        return int(len(self._y_train))

    @property
    def num_eval(self) -> int:
        return int(len(self._y_eval))

    @property
    def class_counts(self) -> dict[str, int]:
        y = np.concatenate([self._y_train, self._y_eval])
        return {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())}

    # ------------------------------------------------------------- Flower API
    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        if config.get("request") == STATS_REQUEST:
            return encode_statistics(local_statistics(self.spec, self._raw_train))
        return {"client_id": self.client_id, "num_train": self.num_train}

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return [np.zeros(self.n_features), np.zeros(1)]

    def _ensure_transformed(self, config: dict[str, Scalar]) -> None:
        raw = config.get(PREPROCESSOR_KEY)
        if not isinstance(raw, (bytes, bytearray)):
            raise ValueError("Server did not send preprocessing parameters")
        digest = hashlib.sha256(bytes(raw)).hexdigest()
        if digest == self._pre_hash:
            return
        pre = SchemaPreprocessor.from_bytes(bytes(raw))
        if pre.dataset_key != self.spec.key:
            raise ValueError("Preprocessing parameters belong to another dataset")
        self._X_train = pre.transform(self._raw_train)
        self._X_eval = pre.transform(self._raw_eval)
        self._pre_hash = digest

    def _model(self, parameters: NDArrays, config: dict[str, Scalar]) -> MiniBatchLogisticRegression:
        cw = str(config.get("class_weight", "none"))
        model = MiniBatchLogisticRegression(
            learning_rate=float(config.get("learning_rate", 0.01)),
            batch_size=int(config.get("batch_size", 32)),
            epochs=int(config.get("local_epochs", 1)),
            l2=float(config.get("l2", 1e-4)),
            class_weight=None if cw == "none" else cw,
        )
        model.init_params(self.n_features)
        model.set_params_arrays([np.asarray(p) for p in parameters])
        return model

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]):
        self._ensure_transformed(config)
        server_round = int(config.get("server_round", 0))
        global_params = [np.asarray(p, dtype=np.float64).copy() for p in parameters]
        model = self._model(parameters, config)
        y_train = poison_labels(self._y_train, self.attack)
        rng = np.random.default_rng([self.seed, self.client_index, server_round])
        model.partial_fit(self._X_train, y_train, rng=rng)
        honest = model.get_params_arrays()
        sent = corrupt_update(global_params, honest, self.attack, rng)
        train_loss = log_loss(y_train, model.predict_proba(self._X_train)[:, 1])
        metrics: dict[str, Scalar] = {
            "client_id": self.client_id,
            "train_loss": float(train_loss),  # self-reported; NOT used for trust
        }
        return sent, int(len(self._y_train)), metrics

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]):
        self._ensure_transformed(config)
        model = self._model(parameters, config)
        p = model.predict_proba(self._X_eval)[:, 1]
        loss = log_loss(self._y_eval, p)
        acc = accuracy_score(self._y_eval, (p >= 0.5).astype(int))
        metrics: dict[str, Scalar] = {"client_id": self.client_id, "accuracy": float(acc)}
        auc = auc_or_none(self._y_eval, p)
        if auc is not None:
            metrics["auc"] = auc
        return float(loss), int(len(self._y_eval)), metrics
