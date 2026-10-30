"""Headless entry point: run a genuine Flower experiment from the command line.

Examples (PowerShell):
    python run_experiment.py --dataset breast_cancer --rounds 30 --strategy trust
    python run_experiment.py --dataset heart --malicious 4 --attack sign_flip
    python run_experiment.py --sweep strategy_vs_attackers --dataset heart
    python run_experiment.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.configuration.settings import SettingsManager
from app.logging.setup import configure_logging
from experiments.store import ExperimentStore
from experiments.sweeps import SWEEPS, run_sweep
from federated.server.runner import ExperimentError, run_experiment
from ml.datasets.registry import DatasetRegistry


def main(argv: list[str] | None = None) -> int:
    settings = SettingsManager().load()
    p = argparse.ArgumentParser(description="TA-EFL Health federated experiment runner")
    p.add_argument("--dataset", choices=DatasetRegistry.keys(), default=settings.default_dataset)
    p.add_argument("--clients", type=int, default=settings.num_clients)
    p.add_argument("--rounds", type=int, default=settings.num_rounds)
    p.add_argument("--strategy", choices=["trust", "fedavg", "median"], default=None)
    p.add_argument("--lr", type=float, default=settings.learning_rate)
    p.add_argument("--batch-size", type=int, default=settings.batch_size)
    p.add_argument("--local-epochs", type=int, default=settings.local_epochs)
    p.add_argument("--partition", choices=["iid", "dirichlet"], default=settings.partition)
    p.add_argument("--malicious", type=int, nargs="*", default=None,
                   help="1-based client numbers acting as attackers (empty = none)")
    p.add_argument("--attack", choices=["none", "sign_flip", "scaling", "gaussian_noise",
                                        "label_flip"], default=None)
    p.add_argument("--seed", type=int, default=settings.seed)
    p.add_argument("--sweep", choices=list(SWEEPS), default=None)
    p.add_argument("--list", action="store_true", help="list stored experiments and exit")
    a = p.parse_args(argv)
    configure_logging(settings.storage_path, settings.log_level)

    if a.list:
        for r in ExperimentStore(settings.storage_path).list():
            auc = (r.get("test_metrics") or {}).get("roc_auc")
            print(f"{r['id']:55s} {r['status']:10s} test AUC="
                  f"{'n/a' if auc is None else f'{auc:.3f}'}")
        return 0

    cfg = settings.experiment_config(
        dataset_key=a.dataset, num_clients=a.clients, num_rounds=a.rounds,
        learning_rate=a.lr, batch_size=a.batch_size, local_epochs=a.local_epochs,
        partition=a.partition, seed=a.seed)
    if a.strategy:
        cfg.strategy = a.strategy
    if a.malicious is not None:
        cfg.malicious_clients = a.malicious
        if not a.malicious:
            cfg.attack_type = "none"
    if a.attack:
        cfg.attack_type = a.attack
    if not cfg.malicious_clients:
        cfg.attack_type = "none"
    try:
        if a.sweep:
            rows = run_sweep(a.sweep, cfg, settings.data_path, settings.storage_path,
                             on_progress=lambda i, n, name: print(f"[{i}/{n}] {name}"))
            print(json.dumps(rows, indent=1))
            return 0
        rec = run_experiment(cfg, settings.data_path, settings.storage_path)
    except ExperimentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    tm = rec["test_metrics"]
    print(f"\nExperiment {rec['id']} {rec['status']} ({rec['rounds_completed']} rounds)")
    print(f"Test (n={tm['n']}): ROC-AUC={tm['roc_auc']}, accuracy={tm['accuracy']:.3f}, "
          f"precision={tm['precision']:.3f}, recall={tm['recall']:.3f}, f1={tm['f1']:.3f}")
    print(f"Final trust: {json.dumps({k: round(v, 3) for k, v in rec['final_trust'].items()})}")
    print(f"Model artifact: {Path(settings.storage_path) / rec['artifact_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
