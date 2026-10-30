"""End-to-end federated experiment: data -> Flower Server.fit -> evaluation -> artifact.

Communication model (documented for the README):

* Each simulated hospital holds a disjoint partition of the training split and
  keeps it inside its ``HospitalClient`` object. Raw rows are never sent.
* The coordinator combines aggregate statistics from client training rows only
  and shares the fitted transform (mean imputation, scaling and schema categories).
  Server validation and test rows are transformed only, never used for fitting.
* Every round, Flower's ``Server`` sends ``FitIns`` (global ``[coef, intercept]``
  plus training config) and receives ``FitRes`` (updated parameters, number of
  local training examples and self-reported training loss). ``EvaluateIns`` /
  ``EvaluateRes`` carry parameters and aggregate local evaluation loss/accuracy.
* Messages are encoded/decoded with Flower's protobuf serde by
  ``InProcessClientProxy``. No encryption, secure aggregation or differential
  privacy is implemented; parameters of a linear model can leak information
  about local data.
"""
from __future__ import annotations

import logging
import secrets
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from federated.clients.attacks import ATTACK_DESCRIPTIONS, AttackConfig
from federated.clients.hospital_client import (
    PREPROCESSOR_KEY,
    STATS_REQUEST,
    HospitalClient,
    decode_statistics,
)
from federated.server.transport import InProcessClientProxy, TrafficLog
from federated.simulation.partition import partition_indices
from federated.strategies.strategies import (
    STRATEGY_LABELS,
    ExperimentStopped,
    MonitorContext,
    build_strategy,
)
from federated.strategies.trust import TrustConfig, TrustManager
from ml.datasets.loader import load_dataset, make_splits
from ml.evaluation.metrics import classification_metrics
from ml.models.artifact import ModelArtifact, save_artifact
from ml.models.logistic import MiniBatchLogisticRegression
from ml.preprocessing.pipeline import (
    PIPELINE_VERSION,
    SchemaPreprocessor,
    combine_statistics,
    fit_preprocessor,
    transform,
    transformed_columns,
)

EventCallback = Callable[[dict[str, Any]], None]


class ExperimentError(Exception):
    pass


class ExperimentCancelled(ExperimentError):
    """Cooperative stop before an artifact can be saved."""


@dataclass
class ExperimentConfig:
    dataset_key: str = "heart"
    num_clients: int = 5
    num_rounds: int = 30
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 1
    l2: float = 1e-4
    class_weight: str = "none"  # "none" | "balanced"
    strategy: str = "trust"  # "trust" | "fedavg" | "median"
    trust_alpha: float = 0.7
    trust_initial: float = 0.5
    trust_performance_weight: float = 0.6
    trust_exclusion_threshold: float = 0.3
    trust_performance_mode: str = "relative"
    partition: str = "iid"  # "iid" | "dirichlet"
    dirichlet_alpha: float = 0.5
    fraction_fit: float = 1.0
    malicious_clients: list[int] = field(default_factory=list)  # 1-based client numbers
    attack_type: str = "none"
    attack_scale: float = 3.0
    attack_noise_std: float = 1.0
    seed: int = 42
    test_fraction: float = 0.20
    val_fraction: float = 0.15
    decision_threshold: float = 0.5
    centralized_baseline: bool = True
    label: str = ""

    def validate(self) -> None:
        errors = []
        if not 2 <= self.num_clients <= 20:
            errors.append("Number of clients must be between 2 and 20")
        if not 1 <= self.num_rounds <= 500:
            errors.append("Number of rounds must be between 1 and 500")
        if not 1e-5 <= self.learning_rate <= 1.0:
            errors.append("Learning rate must be in [1e-5, 1]")
        if not 1 <= self.batch_size <= 4096:
            errors.append("Batch size must be in [1, 4096]")
        if not 1 <= self.local_epochs <= 50:
            errors.append("Local epochs must be in [1, 50]")
        if not 0 <= self.l2 <= 1:
            errors.append("L2 penalty must be in [0, 1]")
        if self.class_weight not in ("none", "balanced"):
            errors.append("Class weight must be 'none' or 'balanced'")
        if self.strategy not in STRATEGY_LABELS:
            errors.append(f"Strategy must be one of {list(STRATEGY_LABELS)}")
        if self.partition not in ("iid", "dirichlet"):
            errors.append("Partition must be 'iid' or 'dirichlet'")
        if not 0.1 <= self.fraction_fit <= 1.0:
            errors.append("Fraction of clients per round must be in [0.1, 1]")
        if any(not 1 <= c <= self.num_clients for c in self.malicious_clients):
            errors.append("Malicious client numbers must refer to existing clients")
        if len(set(self.malicious_clients)) >= self.num_clients:
            errors.append("At least one client must be honest")
        if self.malicious_clients and self.attack_type == "none":
            errors.append("Choose an attack type for the malicious clients")
        if not 0.05 <= self.decision_threshold <= 0.95:
            errors.append("Decision threshold must be in [0.05, 0.95]")
        try:
            self.trust_config()
            AttackConfig(self.attack_type, self.attack_scale, self.attack_noise_std)
        except ValueError as exc:
            errors.append(str(exc))
        if errors:
            raise ExperimentError("; ".join(errors))

    def trust_config(self) -> TrustConfig:
        return TrustConfig(alpha=self.trust_alpha, initial_trust=self.trust_initial,
                           performance_weight=self.trust_performance_weight,
                           exclusion_threshold=self.trust_exclusion_threshold,
                           performance_mode=self.trust_performance_mode)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ExperimentConfig":
        known = {k: v for k, v in d.items() if k in ExperimentConfig.__dataclass_fields__}
        return ExperimentConfig(**known)


def new_experiment_id(cfg: ExperimentConfig) -> str:
    return f"{cfg.dataset_key}-{cfg.strategy}-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


class _FlowerLogForwarder(logging.Handler):
    def __init__(self, emit: EventCallback) -> None:
        super().__init__(level=logging.INFO)
        self._emit = emit

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = record.getMessage().strip()
            if msg:
                self._emit({"type": "flower_log", "message": msg})
        except Exception:  # noqa: BLE001 - logging must never break training
            pass


def run_experiment(cfg: ExperimentConfig, data_dir: Path, storage_dir: Path,
                   on_event: EventCallback | None = None,
                   stop_event: threading.Event | None = None) -> dict[str, Any]:
    """Run one genuine Flower experiment and persist its record + model artifact."""
    from flwr import __version__ as flwr_version  # imported lazily for clear errors
    from flwr.common import GetPropertiesIns
    from flwr.server import Server, SimpleClientManager

    cfg.validate()
    stop_event = stop_event or threading.Event()
    exp_id = new_experiment_id(cfg)
    started = time.time()
    record: dict[str, Any] = {
        "id": exp_id,
        "pipeline_version": PIPELINE_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "config": asdict(cfg),
        "framework": {"flower": flwr_version,
                      "transport": "Flower ClientProxy, protobuf-serialised in-process messages",
                      "model": "Logistic regression (mini-batch SGD), params [coef, intercept]"},
        "strategy_label": STRATEGY_LABELS[cfg.strategy],
        "attack_description": ATTACK_DESCRIPTIONS[cfg.attack_type],
        "trust_formula": cfg.trust_config().formula_text(),
        "rounds": [],
        "validation_history": [],
        "federated_eval_history": [],
        "log": [],
    }

    def emit(event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "round_aggregated":
            record["rounds"].append(event)
        elif etype == "evaluate":
            record["validation_history"].append(event)
        elif etype == "federated_evaluate":
            record["federated_eval_history"].append(event)
        if etype in ("log", "round_start", "round_aggregated", "evaluate", "status"):
            record["log"].append(_log_line(event))
        if on_event is not None:
            on_event(event)

    flwr_logger = logging.getLogger("flwr")
    forwarder = _FlowerLogForwarder(emit)
    flwr_logger.addHandler(forwarder)
    try:
        # ---------------------------------------------------------- data
        emit({"type": "experiment_created", "record_id": exp_id})
        emit({"type": "status", "message": f"Loading dataset: {_dataset_name(cfg.dataset_key)}"})
        data = load_dataset(cfg.dataset_key, data_dir)
        spec = data.spec
        splits = make_splits(data, cfg.seed, cfg.test_fraction, cfg.val_fraction)
        record["dataset"] = {
            "key": spec.key, "name": spec.name, "source": spec.source,
            "source_url": spec.source_url, "file": spec.file_name or "bundled (scikit-learn)",
            "file_sha256": data.report.file_sha256,
            "rows": data.report.n_rows, "positives": data.report.positives,
            "duplicates": data.report.n_duplicates,
            "warnings": data.report.warnings,
            "missing_codes_converted": data.report.missing_codes_converted,
        }
        record["splits"] = splits.sizes()
        record["split_config"] = {"seed": cfg.seed, "test_fraction": cfg.test_fraction,
                                  "validation_fraction": cfg.val_fraction, "stratified": True,
                                  "client_local_eval_fraction": 0.2}

        # ---------------------------------------------------------- clients
        parts = partition_indices(splits.y_train, cfg.num_clients, cfg.partition,  # type: ignore[arg-type]
                                  cfg.seed, cfg.dirichlet_alpha)
        attack = AttackConfig(cfg.attack_type, cfg.attack_scale, cfg.attack_noise_std)  # type: ignore[arg-type]
        traffic = TrafficLog(on_event=emit)
        manager = SimpleClientManager()
        client_info = []
        client_ids = []
        for k, idx in enumerate(parts, start=1):
            cid = f"client_{k}"
            client_ids.append(cid)
            is_attacker = k in cfg.malicious_clients
            hc = HospitalClient(cid, k, spec, splits.X_train.iloc[idx], splits.y_train[idx],
                                cfg.seed, attack if is_attacker else AttackConfig())
            manager.register(InProcessClientProxy(cid, hc.to_client(), traffic))
            client_info.append({"cid": cid, "name": f"Client {k}", "num_records": int(len(idx)),
                                "num_train": hc.num_train, "num_eval": hc.num_eval,
                                **hc.class_counts, "configured_attacker": is_attacker})
        record["clients"] = client_info

        # ---------------------------------------------------------- preprocessing
        # Fitted ONLY from the clients' local training rows, via aggregate statistics
        # requested through Flower (validation/test rows are never used for fitting).
        emit({"type": "status", "message": "Requesting preprocessing statistics from clients"})
        stats_parts = []
        for cid, proxy in manager.all().items():
            res = proxy.get_properties(GetPropertiesIns(config={"request": STATS_REQUEST}),
                                       timeout=None, group_id=0)
            stats_parts.append(decode_statistics(res.properties))
        pre = SchemaPreprocessor(spec.key).fit_statistics(
            combine_statistics(spec, stats_parts),
            fitted_on="federated statistics of client training rows")
        pre_bytes = pre.to_bytes()
        X_val_t = transform(pre, spec, splits.X_val)
        X_test_t = transform(pre, spec, splits.X_test)
        record["preprocessing"] = {
            "pipeline_version": PIPELINE_VERSION,
            "fitted_on": pre.fitted_on_,
            "n_fit_rows": pre.n_fit_rows_,
            "numeric": "mean imputation, then standardisation",
            "categorical": "most-frequent imputation, then one-hot with schema vocabulary",
            "validation_and_test_used_for_fitting": False,
            "parameters": pre.to_params(),
        }

        n_features = len(transformed_columns(spec))
        initial = [np.zeros(n_features), np.zeros(1)]
        trust = TrustManager(cfg.trust_config(), client_ids)
        ctx = MonitorContext(X_val_t, splits.y_val, [(n_features,), (1,)], trust, traffic,
                             stop_event, emit)

        def fit_config(server_round: int) -> dict[str, Any]:
            return {"server_round": server_round, "learning_rate": cfg.learning_rate,
                    "batch_size": cfg.batch_size, "local_epochs": cfg.local_epochs,
                    "l2": cfg.l2, "class_weight": cfg.class_weight,
                    PREPROCESSOR_KEY: pre_bytes}

        def evaluate_fn(server_round: int, params, _config):
            emit({"type": "evaluating", "round": server_round})
            m = classification_metrics(splits.y_val, _proba(params, X_val_t),
                                       cfg.decision_threshold, "validation", max_roc_points=2)
            emit({"type": "evaluate", "round": server_round, "split": "validation",
                  "auc": m["roc_auc"], "accuracy": m["accuracy"], "loss": m["log_loss"],
                  "f1": m["f1"]})
            out = {"val_accuracy": m["accuracy"], "val_f1": m["f1"]}
            if m["roc_auc"] is not None:
                out["val_auc"] = m["roc_auc"]
            return m["log_loss"], out

        strategy = build_strategy(cfg.strategy, ctx, initial, cfg.num_clients,
                                  cfg.fraction_fit, fit_config, evaluate_fn)
        server = Server(client_manager=manager, strategy=strategy)
        emit({"type": "status", "message": f"Starting Flower server: {cfg.num_rounds} rounds, "
                                           f"{cfg.num_clients} clients, {STRATEGY_LABELS[cfg.strategy]}"})
        status = "completed"
        try:
            server.fit(num_rounds=cfg.num_rounds, timeout=None)
        except ExperimentStopped as stop:
            status = "stopped"
            emit({"type": "status", "message": str(stop)})
        final_params = _params_from(server.parameters)
        rounds_done = len(record["rounds"])
        if rounds_done == 0:
            raise ExperimentCancelled("Experiment stopped before any round completed; nothing saved")

        # ---------------------------------------------------------- evaluation
        model = MiniBatchLogisticRegression(cfg.learning_rate, cfg.batch_size, cfg.local_epochs,
                                            cfg.l2).init_params(n_features)
        model.set_params_arrays(final_params)
        test_metrics = classification_metrics(splits.y_test, model.predict_proba(X_test_t)[:, 1],
                                              cfg.decision_threshold, "test")
        val_metrics = classification_metrics(splits.y_val, model.predict_proba(X_val_t)[:, 1],
                                             cfg.decision_threshold, "validation")
        record.update(status=status, rounds_completed=rounds_done,
                      test_metrics=test_metrics, final_validation_metrics=val_metrics)
        if cfg.centralized_baseline:
            base_pre = fit_preprocessor(spec, splits.X_train)  # pooled TRAINING rows only
            X_train_t = transform(base_pre, spec, splits.X_train)
            X_test_base = transform(base_pre, spec, splits.X_test)
            base = LogisticRegression(max_iter=5000, C=1.0 / max(cfg.l2 * len(X_train_t), 1e-6))
            base.fit(X_train_t, splits.y_train)
            record["centralized_baseline"] = {
                "description": "Reference only: scikit-learn LogisticRegression trained on the "
                               "pooled client data (not federated, violates data locality).",
                "test_metrics": classification_metrics(
                    splits.y_test, base.predict_proba(X_test_base)[:, 1], cfg.decision_threshold,
                    "test"),
            }
        record["detection"] = _detection_summary(record, cfg)
        record["final_trust"] = dict(trust.trust)

        # ---------------------------------------------------------- artifact
        art_dir = storage_dir / "models" / exp_id
        artifact = ModelArtifact(
            dataset_key=spec.key, schema_hash=spec.schema_hash(),
            feature_names=spec.feature_names,
            transformed_names=[c.name for c in transformed_columns(spec)],
            target=spec.target, class_labels={0: spec.negative_label, 1: spec.positive_label},
            threshold=cfg.decision_threshold,
            threshold_policy="fixed (configured before training; not tuned on test data)",
            model=model, preprocessor=pre, background=X_val_t, experiment_id=exp_id,
            model_description=f"Federated logistic regression - {STRATEGY_LABELS[cfg.strategy]}, "
                              f"{rounds_done} rounds, {cfg.num_clients} clients",
            metrics={k: test_metrics[k] for k in ("roc_auc", "accuracy", "precision", "recall", "f1")},
        )
        save_artifact(artifact, art_dir)
        record["artifact_dir"] = str(art_dir.relative_to(storage_dir))
        record["elapsed_seconds"] = round(time.time() - started, 2)
        emit({"type": "status", "message": f"Experiment {status}. Test ROC-AUC = "
                                           f"{_fmt(test_metrics['roc_auc'])}"})
        _save_record(record, storage_dir)
        emit({"type": "finished", "record_id": exp_id, "status": status})
        return record
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        record["elapsed_seconds"] = round(time.time() - started, 2)
        if record["rounds"]:
            _save_record(record, storage_dir)
        if isinstance(exc, ExperimentError):
            raise
        raise ExperimentError(record["error"]) from exc
    finally:
        flwr_logger.removeHandler(forwarder)


# ----------------------------------------------------------------- helpers
def _params_from(parameters) -> list[np.ndarray]:
    from flwr.common import parameters_to_ndarrays
    return [np.asarray(a, dtype=np.float64) for a in parameters_to_ndarrays(parameters)]


def _proba(params, X: np.ndarray) -> np.ndarray:
    coef, intercept = params
    z = X @ np.asarray(coef).ravel() + float(np.asarray(intercept).ravel()[0])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _client_name(cid: str) -> str:
    head, _, num = cid.rpartition("_")
    return f"Client {num}" if head == "client" and num.isdigit() else cid


def _dataset_name(key: str) -> str:
    from ml.datasets.registry import DatasetRegistry
    try:
        return DatasetRegistry.get(key).name
    except KeyError:
        return key


def _log_line(event: dict[str, Any]) -> str:
    t = event.get("type")
    r = event.get("round")
    if t == "round_start":
        return f"[Round {r}] Global model sent to {len(event['clients'])} clients."
    if t == "round_aggregated":
        n_ok = sum(1 for c in event["clients"] if c["valid"])
        excl = [_client_name(c["cid"]) for c in event["clients"] if c["excluded"]]
        s = f"[Round {r}] {n_ok} client updates received; trust scores updated; aggregated"
        if not event["model_updated"]:
            s += " (no update applied)"
        if excl:
            s += f"; zero weight (rejected or below trust threshold): {', '.join(excl)}"
        return s + "."
    if t == "evaluate":
        return (f"[Round {r}] Global model validation AUC = {_fmt(event['auc'])}, "
                f"accuracy = {event['accuracy']:.3f}")
    return str(event.get("message", ""))


def _detection_summary(record: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    attackers = {f"client_{k}" for k in cfg.malicious_clients}
    per_round = []
    for rnd in record["rounds"]:
        flagged = {c["cid"] for c in rnd["clients"] if c["excluded"]}
        per_round.append({"round": rnd["round"], "flagged": sorted(flagged),
                          "true_positives": len(flagged & attackers),
                          "false_positives": len(flagged - attackers)})
    last_flagged = set(per_round[-1]["flagged"]) if per_round else set()
    return {
        "configured_attackers": sorted(attackers),
        "flagged_last_round": sorted(last_flagged),
        "trust_used_for_weighting": cfg.strategy == "trust",
        "per_round": per_round,
        "note": ("'Flagged' means trust below the exclusion threshold or an invalid update in a "
                 "simulated attack. It is not evidence of real-world attack detection."),
    }


def _save_record(record: dict[str, Any], storage_dir: Path) -> Path:
    from experiments.store import ExperimentStore
    return ExperimentStore(storage_dir).save(record)
