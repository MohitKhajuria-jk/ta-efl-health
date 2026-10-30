"""Flower strategies used by TA-EFL Health.

* ``MonitoredFedAvg``    - Flower's own ``FedAvg.aggregate_fit`` (weights n_k / N).
* ``MonitoredFedMedian`` - Flower's own ``FedMedian.aggregate_fit``
                           (coordinate-wise median of [coef, intercept]).
* ``TrustAwareFedAvg``   - TA-EFL: weighted average with omega_k = T_k n_k / sum T_j n_j
                           (see ``federated/strategies/trust.py`` for the policy).

All three share the same monitoring layer: malformed updates (wrong shapes,
NaN/Inf) are rejected before aggregation, and trust signals are measured for
every client every round so trust can be compared across strategies. Only
``TrustAwareFedAvg`` *uses* trust for weighting; the others report it as
"monitored only".
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedMedian

from federated.server.transport import TrafficLog
from federated.strategies.trust import (
    TrustManager,
    flatten,
    performance_signal,
    reference_auc,
    similarity_signals,
    trust_weights,
    validate_update,
)
from ml.evaluation.metrics import auc_or_none

STRATEGY_LABELS = {
    "fedavg": "FedAvg (Flower)",
    "median": "Coordinate-wise Median (Flower FedMedian)",
    "trust": "Trust-Aware (TA-EFL)",
}


def _cid_key(cid: str) -> tuple:
    head, _, num = cid.rpartition("_")
    return (head, int(num)) if num.isdigit() else (cid, 0)


class ExperimentStopped(Exception):
    """Raised between rounds when the user requests a stop."""


@dataclass
class ClientRoundRecord:
    cid: str
    num_examples: int
    valid: bool
    reason: str
    val_auc: float | None
    performance: float
    similarity: float
    cosine: float | None
    instant: float
    trust: float
    weight: float | None  # None for median (no per-client weights)
    excluded: bool
    train_loss_reported: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MonitorContext:
    X_val: np.ndarray  # transformed server-validation features
    y_val: np.ndarray
    expected_shapes: list[tuple[int, ...]]
    trust: TrustManager
    traffic: TrafficLog
    stop_event: threading.Event
    emit: Callable[[dict[str, Any]], None]

    def val_auc(self, params: list[np.ndarray]) -> float | None:
        coef, intercept = params
        scores = self.X_val @ np.asarray(coef).ravel() + float(np.asarray(intercept).ravel()[0])
        return auc_or_none(self.y_val, scores)


def weighted_train_loss(metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
    pairs = [(n, float(m["train_loss"])) for n, m in metrics if "train_loss" in m]
    total = sum(n for n, _ in pairs)
    return {"train_loss_reported": sum(n * v for n, v in pairs) / total} if total else {}


def weighted_eval(metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
    pairs = [(n, float(m["accuracy"])) for n, m in metrics if "accuracy" in m]
    total = sum(n for n, _ in pairs)
    return {"client_accuracy": sum(n * v for n, v in pairs) / total} if total else {}


class _MonitoredMixin:
    strategy_key = "base"
    uses_trust = False

    def _setup(self, ctx: MonitorContext) -> None:
        self._ctx = ctx
        self._global: list[np.ndarray] | None = None
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------- configure
    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager):
        if self._ctx.stop_event.is_set():
            raise ExperimentStopped(f"Stopped by user before round {server_round}")
        self._global = parameters_to_ndarrays(parameters)
        instructions = super().configure_fit(server_round, parameters, client_manager)
        self._ctx.emit({"type": "round_start", "round": server_round,
                        "clients": [proxy.cid for proxy, _ in instructions]})
        return instructions

    # ------------------------------------------------------------- aggregate
    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]],
                      failures: list) -> tuple[Parameters | None, dict[str, Scalar]]:
        ctx = self._ctx
        ctx.emit({"type": "aggregating", "round": server_round})
        assert self._global is not None
        # Flower's thread pool returns results in completion order; sort by client id so
        # floating-point aggregation (and therefore every run) is bit-for-bit reproducible.
        results = sorted(results, key=lambda r: _cid_key(r[0].cid))
        g_flat = flatten(self._global)
        records: dict[str, ClientRoundRecord] = {}
        valid: list[tuple[ClientProxy, FitRes]] = []
        arrays: dict[str, list[np.ndarray]] = {}
        for proxy, res in results:
            params = parameters_to_ndarrays(res.parameters)
            reason = validate_update(params, ctx.expected_shapes)
            loss = res.metrics.get("train_loss") if res.metrics else None
            records[proxy.cid] = ClientRoundRecord(
                cid=proxy.cid, num_examples=int(res.num_examples), valid=reason is None,
                reason=reason or "", val_auc=None, performance=0.0, similarity=0.0,
                cosine=None, instant=0.0, trust=0.0, weight=0.0, excluded=reason is not None,
                train_loss_reported=float(loss) if loss is not None else None)
            if reason is None and res.num_examples > 0:
                valid.append((proxy, res))
                arrays[proxy.cid] = params
            elif reason is None:
                records[proxy.cid].valid = False
                records[proxy.cid].excluded = True
                records[proxy.cid].reason = "reported 0 training examples"

        # ---- trust signals (measured for every strategy) ---------------------
        deltas = {cid: flatten(p) - g_flat for cid, p in arrays.items()}
        sims, cos_raw = similarity_signals(deltas)
        for cid in arrays:
            records[cid].val_auc = ctx.val_auc(arrays[cid])
        ref_auc = reference_auc([records[c].val_auc for c in arrays], ctx.trust.config.performance_mode)
        for cid, rec in records.items():
            if cid in arrays:
                rec.performance = performance_signal(rec.val_auc, ref_auc)
                rec.similarity = sims[cid]
                rec.cosine = cos_raw[cid]
                rec.instant = ctx.trust.instant_score(rec.performance, rec.similarity)
            else:
                rec.instant = 0.0  # invalid update
            rec.trust = ctx.trust.update(cid, rec.instant)

        params_agg, metrics = self._aggregate_valid(server_round, valid, failures, records)

        failed = ctx.traffic.failures.get(server_round, [])
        participants = set(records)
        round_info = {
            "type": "round_aggregated",
            "round": server_round,
            "strategy": self.strategy_key,
            "trust_used_for_weighting": self.uses_trust,
            "reference_auc": ref_auc,
            "clients": [r.to_dict() for r in records.values()],
            "non_participating_trust": {cid: t for cid, t in ctx.trust.trust.items()
                                        if cid not in participants},
            "failures": [{"cid": c, "error": e} for c, e in failed],
            "n_failures": len(failures),
            "model_updated": params_agg is not None,
            "bytes_to_clients": ctx.traffic.bytes_to_clients.get(server_round, 0),
            "bytes_from_clients": ctx.traffic.bytes_from_clients.get(server_round, 0),
        }
        self.history.append(round_info)
        ctx.emit(round_info)
        return params_agg, metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self._ctx.emit({"type": "federated_evaluate", "round": server_round,
                        "loss": loss, "metrics": dict(metrics),
                        "clients": {p.cid: float(r.loss) for p, r in results}})
        return loss, metrics

    def _aggregate_valid(self, server_round, valid, failures, records):
        params, metrics = super().aggregate_fit(server_round, valid, failures)
        total = sum(res.num_examples for _, res in valid)
        for rec in records.values():
            rec.weight = (rec.num_examples / total) if (rec.valid and total) else 0.0
        return params, metrics


class MonitoredFedAvg(_MonitoredMixin, FedAvg):
    strategy_key = "fedavg"

    def __init__(self, ctx: MonitorContext, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._setup(ctx)


class MonitoredFedMedian(_MonitoredMixin, FedMedian):
    strategy_key = "median"

    def __init__(self, ctx: MonitorContext, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._setup(ctx)

    def _aggregate_valid(self, server_round, valid, failures, records):
        params, metrics = FedMedian.aggregate_fit(self, server_round, valid, failures)
        for rec in records.values():
            rec.weight = None  # coordinate-wise median: no per-client weight exists
        return params, metrics


class TrustAwareFedAvg(_MonitoredMixin, FedAvg):
    strategy_key = "trust"
    uses_trust = True

    def __init__(self, ctx: MonitorContext, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._setup(ctx)

    def _aggregate_valid(self, server_round, valid, failures, records):
        if not valid:
            return None, {}
        cfg = self._ctx.trust.config
        eligible = {proxy.cid for proxy, _ in valid}
        trust_now = {cid: rec.trust for cid, rec in records.items()}
        n = {cid: rec.num_examples for cid, rec in records.items()}
        weights = trust_weights(trust_now, n, eligible, cfg.exclusion_threshold)
        if weights is None:
            for rec in records.values():
                rec.weight = 0.0
                rec.excluded = True
            self._ctx.emit({"type": "log", "round": server_round,
                            "message": "No client met the trust threshold; global model kept."})
            return None, {}
        aggregated: list[np.ndarray] | None = None
        for proxy, res in valid:
            w = weights[proxy.cid]
            if w == 0.0:
                continue
            arrs = parameters_to_ndarrays(res.parameters)
            if aggregated is None:
                aggregated = [w * a for a in arrs]
            else:
                aggregated = [acc + w * a for acc, a in zip(aggregated, arrs)]
        for cid, rec in records.items():
            rec.weight = weights.get(cid, 0.0)
            rec.excluded = rec.weight == 0.0
        metrics: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            metrics = self.fit_metrics_aggregation_fn([(r.num_examples, r.metrics) for _, r in valid])
        assert aggregated is not None
        return ndarrays_to_parameters(aggregated), metrics


def build_strategy(key: str, ctx: MonitorContext, initial: list[np.ndarray], num_clients: int,
                   fraction_fit: float, fit_config: Callable[[int], dict[str, Scalar]],
                   evaluate_fn: Callable) -> _MonitoredMixin:
    min_fit = max(1, int(np.ceil(fraction_fit * num_clients)))
    common = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
        initial_parameters=ndarrays_to_parameters(initial),
        fit_metrics_aggregation_fn=weighted_train_loss,
        evaluate_metrics_aggregation_fn=weighted_eval,
        accept_failures=True,
    )
    classes = {"fedavg": MonitoredFedAvg, "median": MonitoredFedMedian, "trust": TrustAwareFedAvg}
    if key not in classes:
        raise ValueError(f"Unknown aggregation strategy '{key}'")
    if key == "fedavg":
        common["inplace"] = False
    return classes[key](ctx, **common)
