"""Full experimental study for the TA-EFL paper.

E1  Strategy comparison under increasing malicious fraction
E2  Trust-component ablation (update-only vs explanation-only vs both)
E3  Extrapolation factor kappa sweep (detection margin for label flipping)
E4  Non-IID heterogeneity sweep with NO attack (false-positive check)
E5  Cross-dataset generalization (all five clinical datasets)

Usage:
    python run_study.py                      # E1-E4 on heart, then E5
    python run_study.py --dataset diabetes   # E1-E4 on another dataset
    python run_study.py --only e5            # just the cross-dataset study
    python run_study.py --backend flower     # run through Flower's runtime

Outputs: results/study_*.png and results/study_results.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.datasets import DATASETS, DEFAULT_DATASET, get_spec
from core.session import Session

N, ROUNDS = 6, 30
ATTACK_SETS = {
    "0/6": {},
    "1/6": {2: "label_flip"},
    "2/6": {2: "label_flip", 4: "noise"},
    "3/6": {1: "scale", 2: "label_flip", 4: "noise"},
}
SEEDS = [42, 43, 44]
LABELS = {"trust": "TA-EFL (ours)", "fedavg": "FedAvg", "median": "Coord. median"}
rows: list[list] = []

OUTDIR = "results"          # set by --label, so runs never overwrite each other
ALPHA_OVERRIDE = None       # set by --alpha; None = use the project default


def _out(name: str) -> str:
    os.makedirs(OUTDIR, exist_ok=True)
    return os.path.join(OUTDIR, name)


def run(strategy, attacks, seed, alpha_nid=0.5, trust_kwargs=None, kappa=4.0,
        dataset=DEFAULT_DATASET, backend=None, rounds=ROUNDS):
    tk = dict(trust_kwargs or {})
    # --alpha pins the trust weighting for experiments that do not set it
    # themselves (E1/E3/E4/E5); E2 and E6 always specify their own alpha.
    if ALPHA_OVERRIDE is not None and "alpha" not in tk:
        tk["alpha"] = ALPHA_OVERRIDE
    s = Session(n_clients=N, dirichlet_alpha=alpha_nid, attacks=attacks,
                strategy=strategy, n_rounds=rounds, seed=seed,
                trust_kwargs=tk, kappa=kappa, dataset=dataset,
                backend=backend)
    s.run()
    return s


def detection_margin(s):
    """min honest trust - max attacker trust (positive = clean separation)."""
    t = s.server.trust_engine.trust
    hon = [t[c.cid] for c in s.clients if c.attack == "none"]
    att = [t[c.cid] for c in s.clients if c.attack != "none"]
    if not att:
        return None
    return min(hon) - max(att)


# ---------------- E1: robustness vs malicious fraction -------------------
def e1_robustness(dataset, backend):
    print("E1: strategy comparison vs malicious fraction")
    e1 = {}
    for frac, attacks in ATTACK_SETS.items():
        for strat in ("trust", "fedavg", "median"):
            aucs = []
            for seed in SEEDS:
                s = run(strat, attacks, seed, dataset=dataset, backend=backend)
                auc = s.server.records[-1].metrics["auc"]
                aucs.append(auc)
                rows.append(["E1", strat, frac, seed, f"{auc:.4f}", dataset])
            e1[(frac, strat)] = (np.mean(aucs), np.std(aucs))
            print(f"  {frac} {strat:<8} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")

    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(ATTACK_SETS))
    for i, strat in enumerate(("trust", "fedavg", "median")):
        means = [e1[(f, strat)][0] for f in ATTACK_SETS]
        stds = [e1[(f, strat)][1] for f in ATTACK_SETS]
        plt.errorbar(x + (i - 1) * 0.05, means, yerr=stds, marker="o",
                     capsize=3, label=LABELS[strat])
    plt.xticks(x, ATTACK_SETS.keys())
    plt.xlabel("Malicious clients"); plt.ylabel("Final test AUC")
    plt.title(f"{get_spec(dataset).name}: robustness to attack fraction "
              "(mean ± sd, 3 seeds)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(_out(f"study_e1_robustness{_tag(dataset)}.png"), dpi=150)
    plt.close()


# ---------------- E2: trust-component ablation ---------------------------
def e2_ablation(dataset, backend):
    print("E2: trust-component ablation (2/6 attackers)")
    abls = {"both (α=0.5)": {"alpha": 0.5}, "update-only (α=1)": {"alpha": 1.0},
            "explanation-only (α=0)": {"alpha": 0.0}}
    e2 = {}
    for name, tk in abls.items():
        aucs, margins = [], []
        for seed in SEEDS:
            s = run("trust", ATTACK_SETS["2/6"], seed, trust_kwargs=tk,
                    dataset=dataset, backend=backend)
            aucs.append(s.server.records[-1].metrics["auc"])
            margins.append(detection_margin(s))
            rows.append(["E2", name, "2/6", seed, f"{aucs[-1]:.4f}",
                         dataset, f"{margins[-1]:.4f}"])
        e2[name] = (np.mean(aucs), np.mean(margins))
        print(f"  {name:<24} AUC={np.mean(aucs):.3f}  margin={np.mean(margins):.3f}")

    plt.figure(figsize=(7, 4))
    names = list(e2)
    plt.bar(np.arange(3) - 0.18, [e2[n][0] for n in names], 0.36, label="Final AUC")
    plt.bar(np.arange(3) + 0.18, [e2[n][1] for n in names], 0.36,
            label="Detection margin")
    plt.xticks(range(3), names, fontsize=9)
    plt.title(f"{get_spec(dataset).name}: trust-signal ablation (2/6 malicious)")
    plt.legend(); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(_out(f"study_e2_ablation{_tag(dataset)}.png"), dpi=150)
    plt.close()


# ---------------- E3: kappa sweep ----------------------------------------
def e3_kappa(dataset, backend):
    print("E3: kappa sweep (1 label flipper)")
    kappas = [1, 2, 4, 8, 16]
    e3 = {}
    for kp in kappas:
        margins = []
        for seed in SEEDS:
            s = run("trust", {2: "label_flip"}, seed, kappa=float(kp),
                    dataset=dataset, backend=backend)
            margins.append(detection_margin(s))
            rows.append(["E3", f"kappa={kp}", "1/6", seed,
                         f"{margins[-1]:.4f}", dataset])
        e3[kp] = np.mean(margins)
        print(f"  kappa={kp:<3} detection margin={np.mean(margins):.3f}")

    plt.figure(figsize=(6.5, 4))
    plt.plot(kappas, [e3[k] for k in kappas], marker="o", color="#1a56db")
    plt.axhline(0, color="red", lw=0.8, ls=":")
    plt.xlabel("Extrapolation factor κ"); plt.ylabel("Detection margin")
    plt.title(f"{get_spec(dataset).name}: signature extrapolation vs label flipping")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(_out(f"study_e3_kappa{_tag(dataset)}.png"), dpi=150)
    plt.close()


# ---------------- E4: non-IID false positives ----------------------------
def e4_noniid(dataset, backend):
    print("E4: heterogeneity false-positive check (no attack)")
    e4 = {}
    for a in (0.1, 0.5, 1.0, 5.0):
        min_trusts = []
        for seed in SEEDS:
            s = run("trust", {}, seed, alpha_nid=a, dataset=dataset,
                    backend=backend)
            min_trusts.append(s.server.trust_engine.trust.min())
            rows.append(["E4", f"alpha={a}", "0/6", seed,
                         f"{min_trusts[-1]:.4f}", dataset])
        e4[a] = np.mean(min_trusts)
        print(f"  Dirichlet alpha={a:<4} min honest trust={np.mean(min_trusts):.3f}")

    plt.figure(figsize=(6.5, 4))
    alphas = list(e4)
    plt.bar([str(a) for a in alphas], [e4[a] for a in alphas], color="#1a9e5c")
    plt.axhline(0.25, color="red", lw=0.8, ls=":", label="exclusion threshold τ")
    plt.xlabel("Dirichlet α (lower = more non-IID)")
    plt.ylabel("Min honest-client trust")
    plt.title(f"{get_spec(dataset).name}: honest clients under heterogeneity")
    plt.legend(); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(_out(f"study_e4_noniid{_tag(dataset)}.png"), dpi=150)
    plt.close()


# ---------------- E6: sensitivity to the trust weighting α ----------------
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def e6_alpha(dataset, backend):
    """Sweep α over the same setup as E2: 2/6 attackers, same seeds,
    same partitions, same κ, everything except α held fixed."""
    print("E6: trust-weighting α sensitivity (2/6 attackers)")
    e6 = {}
    for a in ALPHAS:
        aucs, margins = [], []
        for seed in SEEDS:
            s = run("trust", ATTACK_SETS["2/6"], seed,
                    trust_kwargs={"alpha": a}, dataset=dataset,
                    backend=backend)
            aucs.append(s.server.records[-1].metrics["auc"])
            margins.append(detection_margin(s))
            rows.append(["E6", f"alpha={a}", "2/6", seed, f"{aucs[-1]:.4f}",
                         dataset, f"{margins[-1]:.4f}"])
        e6[a] = (np.mean(aucs), np.std(aucs), np.mean(margins), np.std(margins))
        print(f"  α={a:<5} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
              f"margin={np.mean(margins):+.3f}±{np.std(margins):.3f}")

    best_auc = max(ALPHAS, key=lambda a: e6[a][0])
    best_margin = max(ALPHAS, key=lambda a: e6[a][2])
    print(f"  best AUC: α={best_auc} | best detection margin: α={best_margin}"
          + ("  (consistent)" if best_auc == best_margin else "  (they differ)"))

    # Per-seed results in their own file so E1–E5 output is untouched.
    with open(_out(f"study_e6_results{_tag(dataset)}.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "config", "attack_frac", "seed", "auc",
                    "dataset", "detection_margin"])
        for r in rows:
            if r[0] == "E6":
                w.writerow(r)
        w.writerow([])
        w.writerow(["summary", "alpha", "auc_mean", "auc_std",
                    "margin_mean", "margin_std", f"seeds={SEEDS}"])
        for a in ALPHAS:
            m = e6[a]
            w.writerow(["E6", a, f"{m[0]:.4f}", f"{m[1]:.4f}",
                        f"{m[2]:.4f}", f"{m[3]:.4f}", ""])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax1.errorbar(ALPHAS, [e6[a][0] for a in ALPHAS],
                 yerr=[e6[a][1] for a in ALPHAS], marker="o", capsize=3,
                 color="#1a56db")
    ax1.set_xlabel("α  (0 = explanation only, 1 = update only)")
    ax1.set_ylabel("Final test AUC")
    ax1.set_title("α vs AUC")
    ax1.grid(alpha=0.3)
    ax2.errorbar(ALPHAS, [e6[a][2] for a in ALPHAS],
                 yerr=[e6[a][3] for a in ALPHAS], marker="o", capsize=3,
                 color="#1a9e5c")
    ax2.axhline(0, color="red", lw=0.8, ls=":", label="no separation")
    ax2.set_xlabel("α  (0 = explanation only, 1 = update only)")
    ax2.set_ylabel("Detection margin")
    ax2.set_title("α vs detection margin")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle(f"E6: {get_spec(dataset).name} — trust weighting sensitivity "
                 f"(2/6 malicious, {len(SEEDS)} seeds)")
    fig.tight_layout()
    fig.savefig(_out(f"study_e6_alpha{_tag(dataset)}.png"), dpi=150)
    plt.close(fig)
    return e6


# ---------------- E5: cross-dataset generalization ------------------------
def e5_datasets(backend, seeds=None, rounds=ROUNDS):
    print("E5: cross-dataset generalization (2/6 attackers)")
    seeds = seeds or SEEDS
    keys = list(DATASETS)
    res = {}
    for key in keys:
        for strat in ("trust", "fedavg", "median"):
            aucs, margins = [], []
            for seed in seeds:
                s = run(strat, ATTACK_SETS["2/6"], seed, dataset=key,
                        backend=backend, rounds=rounds)
                aucs.append(s.server.records[-1].metrics["auc"])
                if strat == "trust":
                    margins.append(detection_margin(s))
                rows.append(["E5", strat, "2/6", seed, f"{aucs[-1]:.4f}", key])
            res[(key, strat)] = (float(np.mean(aucs)), float(np.std(aucs)))
            if strat == "trust":
                res[(key, "margin")] = (float(np.mean(margins)), 0.0)
        print(f"  {get_spec(key).name:<26} "
              + "  ".join(f"{s}={res[(key, s)][0]:.3f}"
                          for s in ("trust", "fedavg", "median"))
              + f"  margin={res[(key, 'margin')][0]:+.3f}")

    x = np.arange(len(keys))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for i, strat in enumerate(("trust", "fedavg", "median")):
        ax1.bar(x + (i - 1) * 0.27, [res[(k, strat)][0] for k in keys], 0.26,
                yerr=[res[(k, strat)][1] for k in keys], capsize=3,
                label=LABELS[strat])
    ax1.set_xticks(x)
    ax1.set_xticklabels([get_spec(k).name for k in keys], fontsize=8, rotation=15)
    ax1.set_ylabel("Final test AUC"); ax1.set_ylim(0.5, 1.0)
    ax1.set_title("Robustness across clinical datasets (2/6 malicious)")
    ax1.legend(); ax1.grid(alpha=0.3, axis="y")

    margins = [res[(k, "margin")][0] for k in keys]
    ax2.bar(x, margins, 0.5, color=["#1a9e5c" if m > 0 else "#e0483e"
                                    for m in margins])
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([get_spec(k).name for k in keys], fontsize=8, rotation=15)
    ax2.set_ylabel("Detection margin")
    ax2.set_title("Honest/attacker trust separation")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(_out("study_e5_datasets.png"), dpi=150)
    plt.close(fig)


def _tag(dataset: str) -> str:
    return "" if dataset == DEFAULT_DATASET else f"_{dataset}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help=f"dataset for E1-E4 ({sorted(DATASETS)})")
    ap.add_argument("--backend", default=None, choices=[None, "local", "flower"])
    ap.add_argument("--only", default="all",
                    help="all, or a comma-separated list: e1,e3,e5")
    ap.add_argument("--label", default=None,
                    help="write results to results/<label>/ instead of results/")
    ap.add_argument("--alpha", type=float, default=None,
                    help="pin the trust weighting alpha (default: project default)")
    args = ap.parse_args()
    global OUTDIR, ALPHA_OVERRIDE
    only = args.only.lower()
    sel = {e.strip() for e in only.split(",")}
    if args.label:
        OUTDIR = os.path.join("results", args.label)
    ALPHA_OVERRIDE = args.alpha
    ds, be = args.dataset, args.backend
    from core.trust import TrustEngine
    print(f"trust weighting alpha = "
          f"{ALPHA_OVERRIDE if ALPHA_OVERRIDE is not None else TrustEngine(1).alpha}"
          f"  |  output dir = {OUTDIR}\n")

    if "all" in sel or "e1" in sel:
        e1_robustness(ds, be)
    if "all" in sel or "e2" in sel:
        e2_ablation(ds, be)
    if "all" in sel or "e3" in sel:
        e3_kappa(ds, be)
    if "all" in sel or "e4" in sel:
        e4_noniid(ds, be)
    if "all" in sel or "e5" in sel:
        e5_datasets(be)
    if "all" in sel or "e6" in sel:
        e6_alpha(ds, be)

    # A partial run writes its own file: a single-experiment run must never
    # clobber the combined E1-E5 results produced by a full run.
    out = (_out("study_results.csv") if "all" in sel
           else _out("study_" + "_".join(sorted(sel)) + "_rows.csv"))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "config", "attack_frac", "seed", "value",
                    "dataset", "extra"])
        for r in rows:
            w.writerow(r + [""] * (7 - len(r)))
    print(f"\nSaved: results/study_*.png, {out}")


if __name__ == "__main__":
    main()

