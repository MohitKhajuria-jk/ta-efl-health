"""Sensitivity analysis: run several genuine experiments varying one factor."""
from __future__ import annotations

import copy
import json
from datetime import datetime
import threading
from pathlib import Path
from typing import Any, Callable

from experiments.store import atomic_write_text
from federated.server.runner import ExperimentConfig, run_experiment

SWEEPS = {
    "strategy_vs_attackers": "Each aggregation strategy × number of configured attackers (0, 1, 2)",
    "trust_alpha": "Trust-aware aggregation with trust alpha 0.3 / 0.5 / 0.7 / 0.9",
    "num_clients": "Trust-aware aggregation with 3 / 5 / 8 clients",
}


def plan_sweep(kind: str, base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    runs: list[tuple[str, ExperimentConfig]] = []

    def variant(**kw: Any) -> ExperimentConfig:
        c = copy.deepcopy(base)
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    attack = base.attack_type if base.attack_type != "none" else "sign_flip"
    if kind == "strategy_vs_attackers":
        for strat in ("fedavg", "median", "trust"):
            for n_att in (0, 1, 2):
                if n_att >= base.num_clients:
                    continue
                mal = list(range(base.num_clients - n_att + 1, base.num_clients + 1))
                names = {"fedavg": "FedAvg", "median": "Median", "trust": "Trust-aware"}
                runs.append((f"{names[strat]}, {n_att} attacker{'s' if n_att != 1 else ''}",
                             variant(strategy=strat, malicious_clients=mal,
                                     attack_type=attack if n_att else "none",
                                     label=f"sweep:{kind}")))
    elif kind == "trust_alpha":
        for a in (0.3, 0.5, 0.7, 0.9):
            runs.append((f"Trust alpha {a}", variant(strategy="trust", trust_alpha=a,
                                               label=f"sweep:{kind}")))
    elif kind == "num_clients":
        for n in (3, 5, 8):
            mal = [m for m in base.malicious_clients if m <= n][: n - 1]
            runs.append((f"{n} clients", variant(strategy="trust", num_clients=n,
                                                 malicious_clients=mal,
                                                 attack_type=base.attack_type if mal else "none",
                                                 label=f"sweep:{kind}")))
    else:
        raise ValueError(f"Unknown sweep '{kind}'")
    return runs


def run_sweep(kind: str, base: ExperimentConfig, data_dir: Path, storage_dir: Path,
              on_progress: Callable[[int, int, str], None] | None = None,
              stop_event: threading.Event | None = None) -> list[dict[str, Any]]:
    stop_event = stop_event or threading.Event()
    rows = []
    runs = plan_sweep(kind, base)
    for i, (name, cfg) in enumerate(runs, start=1):
        if stop_event.is_set():
            break
        if on_progress:
            on_progress(i, len(runs), name)
        rec = run_experiment(cfg, data_dir, storage_dir, stop_event=stop_event)
        tm = rec.get("test_metrics", {})
        rows.append({"run": name, "experiment_id": rec["id"], "status": rec["status"],
                     "strategy": cfg.strategy, "attackers": len(cfg.malicious_clients),
                     "roc_auc": tm.get("roc_auc"), "accuracy": tm.get("accuracy"),
                     "f1": tm.get("f1")})
    if rows:
        save_sweep(kind, rows, storage_dir)
    return rows


def save_sweep(kind: str, rows: list[dict[str, Any]], storage_dir: Path) -> Path:
    path = Path(storage_dir) / "sweeps" / f"{kind}-{datetime.now():%Y%m%d-%H%M%S}.json"
    atomic_write_text(path, json.dumps({"kind": kind, "description": SWEEPS[kind],
                                        "created": datetime.now().isoformat(timespec="seconds"),
                                        "rows": rows}, indent=1))
    return path


def list_sweeps(storage_dir: Path) -> list[dict[str, Any]]:
    d = Path(storage_dir) / "sweeps"
    out = []
    for p in sorted(d.glob("*.json"), reverse=True) if d.is_dir() else []:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out
