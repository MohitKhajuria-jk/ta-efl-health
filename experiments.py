"""Experiment suite for the paper.

E1: Attack-fraction sweep  — trust vs fedavg vs median, 0..3 of 6 malicious.
E2: Ablation               — full trust vs update-only (alpha=1) vs
                             explanation-only (alpha=0).
E3: Detection quality      — can attackers be separated from honest clients
                             by trust score alone? (precision/recall of
                             flagging T < tau, plus separation margin)

All experiments averaged over multiple seeds. Outputs: results/*.png + CSV.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.datasets import DEFAULT_DATASET
from core.session import Session

DATASET = os.environ.get("TAEFL_DATASET", DEFAULT_DATASET)
SEEDS = [42, 7, 123]
N_CLIENTS = 6
ROUNDS = 30

ATTACK_SETS = {
    0: {},
    1: {2: "label_flip"},
    2: {2: "label_flip", 4: "noise"},
    3: {1: "scale", 2: "label_flip", 4: "noise"},
}


def run_once(strategy, attacks, seed, trust_kwargs=None):
    s = Session(n_clients=N_CLIENTS, dirichlet_alpha=0.5, attacks=attacks,
                strategy=strategy, n_rounds=ROUNDS, seed=seed,
                trust_kwargs=trust_kwargs or {}, dataset=DATASET)
    s.run()
    return s


def final_auc(s):
    return s.server.records[-1].metrics["auc"]


# ------------------------------------------------------------------ E1
def e1_attack_sweep():
    print("\n=== E1: attack-fraction sweep (mean AUC ± std over seeds) ===")
    strategies = ["trust", "fedavg", "median"]
    results = {st: {k: [] for k in ATTACK_SETS} for st in strategies}
    for st in strategies:
        for k, atk in ATTACK_SETS.items():
            for seed in SEEDS:
                results[st][k].append(final_auc(run_once(st, atk, seed)))
    header = f"{'#malicious':<12}" + "".join(f"{st:>18}" for st in strategies)
    print(header)
    rows = []
    for k in ATTACK_SETS:
        row = [k]
        line = f"{k}/{N_CLIENTS:<10}"
        for st in strategies:
            v = results[st][k]
            line += f"{np.mean(v):>10.3f} ±{np.std(v):.3f}"
            row += [np.mean(v), np.std(v)]
        print(line)
        rows.append(row)
    with open("results/e1_attack_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_malicious"] + [f"{st}_{x}" for st in strategies
                                      for x in ("mean", "std")])
        w.writerows(rows)

    plt.figure(figsize=(7, 4.5))
    xs = list(ATTACK_SETS.keys())
    for st, marker in zip(strategies, "osd"):
        means = [np.mean(results[st][k]) for k in xs]
        stds = [np.std(results[st][k]) for k in xs]
        plt.errorbar(xs, means, yerr=stds, marker=marker, capsize=3,
                     label={"trust": "Trust-aware (ours)",
                            "fedavg": "FedAvg",
                            "median": "Coord. median"}[st])
    plt.xlabel(f"Malicious clients (of {N_CLIENTS})")
    plt.ylabel("Final test AUC")
    plt.title("Robustness to increasing attack fraction")
    plt.xticks(xs); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("results/e1_attack_sweep.png", dpi=150)
    plt.close()


# ------------------------------------------------------------------ E2
def e2_ablation():
    print("\n=== E2: ablation of trust components (2/6 malicious) ===")
    variants = {
        "full (α=0.5)": {"alpha": 0.5},
        "update-only (α=1)": {"alpha": 1.0},
        "explanation-only (α=0)": {"alpha": 0.0},
    }
    atk = ATTACK_SETS[2]
    print(f"{'variant':<24}{'AUC':>14}{'atk trust':>12}{'honest trust':>14}")
    rows = []
    for name, kw in variants.items():
        aucs, atk_t, hon_t = [], [], []
        for seed in SEEDS:
            s = run_once("trust", atk, seed, trust_kwargs=kw)
            aucs.append(final_auc(s))
            t = s.server.trust_engine.trust
            mal = [c.cid for c in s.clients if c.attack != "none"]
            hon = [c.cid for c in s.clients if c.attack == "none"]
            atk_t.append(np.mean(t[mal])); hon_t.append(np.mean(t[hon]))
        print(f"{name:<24}{np.mean(aucs):>8.3f} ±{np.std(aucs):.3f}"
              f"{np.mean(atk_t):>12.3f}{np.mean(hon_t):>14.3f}")
        rows.append([name, np.mean(aucs), np.std(aucs),
                     np.mean(atk_t), np.mean(hon_t)])
    with open("results/e2_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "auc_mean", "auc_std",
                    "attacker_trust", "honest_trust"])
        w.writerows(rows)

    labels = [r[0] for r in rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(x - 0.2, [r[3] for r in rows], 0.4, label="Attacker trust",
            color="#e0483e")
    plt.bar(x + 0.2, [r[4] for r in rows], 0.4, label="Honest trust",
            color="#1a56db")
    plt.xticks(x, labels, fontsize=9)
    plt.ylabel("Mean final trust score")
    plt.title("Ablation: separation between attackers and honest clients")
    plt.legend(); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig("results/e2_ablation.png", dpi=150)
    plt.close()


# ------------------------------------------------------------------ E3
def e3_detection():
    print("\n=== E3: detection quality (flag if T < tau=0.25) ===")
    tp = fp = fn = tn = 0
    margins = []
    for k, atk in ATTACK_SETS.items():
        if k == 0:
            continue
        for seed in SEEDS:
            s = run_once("trust", atk, seed)
            t = s.server.trust_engine.trust
            tau = s.server.trust_engine.tau
            for c in s.clients:
                flagged = t[c.cid] < tau
                mal = c.attack != "none"
                tp += flagged and mal
                fp += flagged and not mal
                fn += (not flagged) and mal
                tn += (not flagged) and not mal
            mal_idx = [c.cid for c in s.clients if c.attack != "none"]
            hon_idx = [c.cid for c in s.clients if c.attack == "none"]
            margins.append(min(t[hon_idx]) - max(t[mal_idx]))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    print(f"precision={prec:.3f}  recall={rec:.3f}")
    print(f"mean separation margin (min honest - max attacker) = "
          f"{np.mean(margins):.3f}")
    print(f"ranking success (margin>0): "
          f"{np.mean([m > 0 for m in margins])*100:.0f}% of runs")
    with open("results/e3_detection.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["precision", "recall", "mean_margin", "ranking_success"])
        w.writerow([prec, rec, np.mean(margins),
                    np.mean([m > 0 for m in margins])])


if __name__ == "__main__":
    e1_attack_sweep()
    e2_ablation()
    e3_detection()
    print("\nAll results saved to results/")
