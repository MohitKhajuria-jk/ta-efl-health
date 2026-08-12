"""Headless experiment: trust-aware FL vs FedAvg vs median under attack.

Usage:
    python run_experiment.py                        # heart dataset
    python run_experiment.py --dataset diabetes     # any registered dataset
    python run_experiment.py --dataset all          # every dataset in turn
    python run_experiment.py --backend flower       # Flower simulation runtime

Saves plots to results/ and prints a summary table.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.datasets import DATASETS, DEFAULT_DATASET, get_spec
from core.session import Session

N_CLIENTS = 6
ROUNDS = 30
ATTACKS = {2: "label_flip", 4: "noise"}   # hospitals 2 and 4 are malicious


def run(strategy: str, dataset: str, backend: str | None = None,
        rounds: int = ROUNDS, n_clients: int = N_CLIENTS) -> Session:
    s = Session(n_clients=n_clients, dirichlet_alpha=0.5, attacks=ATTACKS,
                strategy=strategy, n_rounds=rounds, seed=42, dataset=dataset,
                backend=backend)
    s.run()
    return s


def run_dataset(dataset: str, backend: str | None, rounds: int, n_clients: int):
    sessions = {name: run(name, dataset, backend, rounds, n_clients)
                for name in ("trust", "fedavg", "median")}
    ref = sessions["trust"]
    spec = get_spec(dataset)

    print(f"\n=== {spec.name} — {spec.task} ===")
    print(f"Data source: {ref.source} | engine: Flower ({ref.server.backend})")
    print(f"Clients: {n_clients} | malicious: {ATTACKS}\n")
    print(f"{'strategy':<10}{'final AUC':>11}{'final ACC':>11}{'final F1':>10}")
    for name, s in sessions.items():
        m = s.server.records[-1].metrics
        print(f"{name:<10}{m['auc']:>11.3f}{m['accuracy']:>11.3f}{m['f1']:>10.3f}")

    final_trust = ref.server.trust_engine.trust
    print("\nFinal trust scores:")
    for c in ref.clients:
        tag = f"  <-- ATTACKER ({c.attack})" if c.attack != "none" else ""
        print(f"  [{c.cid}] {c.name:<24} {final_trust[c.cid]:.3f}{tag}")

    tag = f"_{dataset}" if dataset != DEFAULT_DATASET else ""
    plt.figure(figsize=(7, 4.5))
    for name, s in sessions.items():
        aucs = [r.metrics["auc"] for r in s.server.records]
        plt.plot(range(1, len(aucs) + 1), aucs, label=name)
    plt.xlabel("Round"); plt.ylabel("Test AUC")
    plt.title(f"{spec.name}: global AUC under attack "
              f"({len(ATTACKS)}/{n_clients} malicious)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"results/auc_comparison{tag}.png", dpi=150)

    hist = np.stack(ref.server.trust_engine.history)
    plt.figure(figsize=(7, 4.5))
    for c in ref.clients:
        style = "--" if c.attack != "none" else "-"
        plt.plot(hist[:, c.cid], style, label=c.name +
                 (f" ({c.attack})" if c.attack != "none" else ""))
    plt.xlabel("Round"); plt.ylabel("Trust score")
    plt.title(f"{spec.name}: trust score trajectories")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"results/trust_trajectories{tag}.png", dpi=150)
    plt.close("all")
    return sessions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help=f"one of {sorted(DATASETS)} or 'all'")
    ap.add_argument("--backend", default=None, choices=[None, "local", "flower"],
                    help="execution backend (default: local Flower driver)")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--clients", type=int, default=N_CLIENTS)
    args = ap.parse_args()

    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for key in keys:
        run_dataset(key, args.backend, args.rounds, args.clients)
    print("\nPlots saved to results/")


if __name__ == "__main__":
    main()
