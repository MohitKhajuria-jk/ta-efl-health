"""Build docs/RESULTS.md from benchmark runs made on THIS machine.

    python scripts\\build_results.py                   # newest runs found in storage\\benchmarks
    python scripts\\build_results.py --main storage\\benchmarks\\benchmark-A.json ^
                                     --absolute storage\\benchmarks\\benchmark-B.json

* ``--main``: a full benchmark (all strategies) with the default *relative* trust signal,
  produced by ``python scripts\\run_benchmarks.py``.
* ``--absolute`` (optional): a trust-only benchmark with the original absolute signal,
  produced by ``python scripts\\run_benchmarks.py --strategies trust --trust-mode absolute``.

Every number comes from the benchmark JSON files and the experiment records they point
to in storage\\experiments; the findings section is computed, not hand-written.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.datasets.registry import DatasetRegistry  # noqa: E402

ATTACKS = ["none", "sign_flip", "scaling", "gaussian_noise", "label_flip"]
ATTACK_NAMES = {"none": "No attack", "sign_flip": "Sign flipping", "scaling": "Update scaling",
                "gaussian_noise": "Gaussian noise", "label_flip": "Label flipping"}
STRATEGIES = ["trust", "fedavg", "median"]
STRATEGY_NAMES = {"trust": "Trust-aware", "fedavg": "FedAvg", "median": "Median"}


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["meta"].setdefault("trust_performance_mode", "absolute")  # files made before the option
    data["path"] = path
    return data


def auto_pick(bench_dir: Path) -> tuple[dict | None, dict | None]:
    main = absolute = None
    for p in sorted(bench_dir.glob("benchmark-*.json"), reverse=True):
        try:
            d = load(p)
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        strategies = {r["strategy"] for r in d["runs"]}
        mode = d["meta"]["trust_performance_mode"]
        if main is None and mode == "relative" and {"fedavg", "median"} <= strategies:
            main = d
        if absolute is None and mode == "absolute" and "trust" in strategies:
            absolute = d
    return main, absolute


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def fmt(m, sd=None, digits=3):
    if m is None:
        return "—"
    return f"{m:.{digits}f}" + ("" if sd is None else f" ± {sd:.{digits}f}")


def exclusion_counts(runs: list[dict], experiments_dir: Path) -> dict:
    """Zero-weight decisions per (dataset, attack) for honest clients and the attacker."""
    out: dict = defaultdict(lambda: [0, 0, 0, 0])
    missing = 0
    for r in runs:
        if r["strategy"] != "trust":
            continue
        p = experiments_dir / f"{r['experiment_id']}.json"
        if not p.is_file():
            missing += 1
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        attackers = {f"client_{m}" for m in rec["config"]["malicious_clients"]}
        c = out[(r["dataset"], r["attack"])]
        for rnd in rec["rounds"]:
            for cl in rnd["clients"]:
                if cl["cid"] in attackers:
                    c[2] += int(cl["excluded"])
                    c[3] += 1
                else:
                    c[0] += int(cl["excluded"])
                    c[1] += 1
    out["_missing"] = missing  # type: ignore[index]
    return out


def dataset_facts(runs: list[dict], experiments_dir: Path) -> dict:
    facts = {}
    for r in runs:
        if r["dataset"] in facts:
            continue
        p = experiments_dir / f"{r['experiment_id']}.json"
        if p.is_file():
            rec = json.loads(p.read_text(encoding="utf-8"))
            facts[r["dataset"]] = {"rows": rec["dataset"]["rows"],
                                   "positives": rec["dataset"]["positives"],
                                   "sha": rec["dataset"].get("file_sha256"),
                                   "flower": rec["framework"]["flower"],
                                   "test_n": r["test_n"], "prev": r["test_prevalence"]}
    return facts


def pct(x: int, n: int) -> str:
    return "—" if n == 0 else f"{x}/{n} ({100 * x / n:.1f}%)"


def build(main: dict, absolute: dict | None, storage: Path) -> str:
    exp_dir = storage / "experiments"
    runs = main["runs"]
    m = main["meta"]
    datasets = [k for k in DatasetRegistry.keys() if any(r["dataset"] == k for r in runs)]
    name = {k: DatasetRegistry.get(k).name for k in datasets}
    groups: dict = defaultdict(list)
    for r in runs:
        groups[(r["dataset"], r["attack"], r["strategy"])].append(r)

    def g(ds, att, st, key):
        return mean_sd([r[key] for r in groups.get((ds, att, st), [])])

    facts = dataset_facts(runs, exp_dir)
    L = ["# TA-EFL Health – experiment results (pipeline v2)", ""]
    L.append(f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/build_results.py` from "
             f"`{main['path'].name}`" + (f" and `{absolute['path'].name}`" if absolute else "")
             + f" on {platform.system()} {platform.release()}, Python {platform.python_version()}"
             + (f", Flower {next(iter(facts.values()))['flower']}" if facts else "") + ".")
    L.append("Pipeline v2: preprocessing fitted from client training rows only. Results from the "
             "earlier pipeline are superseded and not used.")
    L.append("")
    L.append(f"**Setup.** {m['clients']} simulated hospitals ({m['partition']}), {m['rounds']} "
             f"rounds, seeds {m['seeds']}, configured attacker Client {m['attacker']} (1 of "
             f"{m['clients']}); settings from config/settings.json. Mean ± SD over seeds. "
             "**Test** = held-out test split, never used for training, preprocessing, trust or "
             "thresholds. In-process Flower simulation, not a deployed network.")
    L.append("")
    L.append("**Datasets.**")
    L.append("")
    L.append("| Dataset | Records | Positive | Test n | Test prevalence | File SHA-256 |")
    L.append("|---|---|---|---|---|---|")
    for ds in datasets:
        f = facts.get(ds, {})
        sha = (f.get("sha") or "bundled")[:16] + ("…" if f.get("sha") else "")
        L.append(f"| {name[ds]} | {f.get('rows', '—')} | {f.get('positives', '—')} | "
                 f"{f.get('test_n', '—')} | {fmt(f.get('prev'))} | {sha} |")
    L.append("")

    # ---------------------------------------------------------------- section 1
    L.append("## 1. Strategy comparison under attack (test split)")
    L.append("")
    L.append("| Dataset | Attack | Strategy | Test ROC-AUC | Test F1 | Test recall | Test ECE | "
             "Attacker final trust | Attacker last weight |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        for att in ATTACKS:
            for st in STRATEGIES:
                if (ds, att, st) not in groups:
                    continue
                L.append(f"| {name[ds]} | {ATTACK_NAMES[att]} | {STRATEGY_NAMES[st]} | "
                         f"{fmt(*g(ds, att, st, 'test_roc_auc'))} | {fmt(*g(ds, att, st, 'test_f1'))} | "
                         f"{fmt(*g(ds, att, st, 'test_recall'))} | {fmt(*g(ds, att, st, 'test_ece'))} | "
                         f"{fmt(*g(ds, att, st, 'attacker_final_trust'))} | "
                         f"{fmt(*g(ds, att, st, 'attacker_last_weight'))} |")
    L.append("")
    L.append("Trust for FedAvg and Median is monitored only (not used for weighting); Median has "
             "no per-client weights (—).")
    L.append("")

    # ---------------------------------------------------------------- section 2
    rel = exclusion_counts(runs, exp_dir)
    ab = exclusion_counts(absolute["runs"], exp_dir) if absolute else None
    L.append("## 2. Trust performance signal: zero-weight decisions")
    L.append("")
    L.append(f"Client-rounds given zero weight by the trust-aware strategy ({m['rounds']} rounds × "
             f"{m['clients']} clients × {len(m['seeds'])} seeds per row).")
    L.append("")
    if ab:
        L.append("| Dataset | Attack | Honest zeroed (absolute) | Honest zeroed (relative) | "
                 "Attacker zeroed (absolute) | Attacker zeroed (relative) |")
        L.append("|---|---|---|---|---|---|")
    else:
        L.append("| Dataset | Attack | Honest zeroed (relative) | Attacker zeroed (relative) |")
        L.append("|---|---|---|---|")
    for ds in datasets:
        for att in ATTACKS:
            r = rel.get((ds, att))
            if r is None:
                continue
            if ab:
                a = ab.get((ds, att), [0, 0, 0, 0])
                L.append(f"| {name[ds]} | {ATTACK_NAMES[att]} | {pct(a[0], a[1])} | {pct(r[0], r[1])} | "
                         f"{pct(a[2], a[3])} | {pct(r[2], r[3])} |")
            else:
                L.append(f"| {name[ds]} | {ATTACK_NAMES[att]} | {pct(r[0], r[1])} | {pct(r[2], r[3])} |")
    missing = rel.get("_missing", 0) + (ab.get("_missing", 0) if ab else 0)
    if missing:
        L.append("")
        L.append(f"Note: {missing} experiment record(s) were missing, so their rounds are not counted.")
    L.append("")

    # ---------------------------------------------------------------- section 3 (computed)
    L.append("## 3. Findings (computed from the tables above)")
    L.append("")
    for att in ("sign_flip", "gaussian_noise"):
        better, worse, med_better = [], [], []
        for ds in datasets:
            t, _ = g(ds, att, "trust", "test_roc_auc")
            f, _ = g(ds, att, "fedavg", "test_roc_auc")
            md, _ = g(ds, att, "median", "test_roc_auc")
            if t is None or f is None:
                continue
            (better if t > f else worse).append(f"{name[ds]} {t:.3f} vs {f:.3f}")
            if md is not None and md >= t:
                med_better.append(name[ds])
        L.append(f"* **{ATTACK_NAMES[att]}:** trust-aware test ROC-AUC was higher than FedAvg on "
                 f"{len(better)} of {len(better) + len(worse)} datasets"
                 + (f" ({'; '.join(better)})" if better else "")
                 + (f"; lower on {'; '.join(worse)}" if worse else "") + ". "
                 + (f"Median was equal or better than trust-aware on {', '.join(med_better)}."
                    if med_better else "Trust-aware was better than median on every dataset."))
    sc = [rel.get((ds, "scaling"), [0, 0, 0, 0]) for ds in datasets]
    sc_z, sc_n = sum(c[2] for c in sc), sum(c[3] for c in sc)
    L.append(f"* **Update scaling:** the attacker was given zero weight in {pct(sc_z, sc_n)} of its "
             "rounds; its update points in the honest direction, so this policy does not detect it.")
    eff = []
    for ds in datasets:
        n, _ = g(ds, "none", "fedavg", "test_roc_auc")
        lf, _ = g(ds, "label_flip", "fedavg", "test_roc_auc")
        if n is not None and lf is not None:
            eff.append(n - lf)
    if eff:
        L.append(f"* **Label flipping** by one client reduced FedAvg test ROC-AUC by "
                 f"{min(eff):.3f}–{max(eff):.3f} across datasets.")
    hz = sum(rel[(ds, att)][0] for ds in datasets for att in ATTACKS if (ds, att) in rel)
    hn = sum(rel[(ds, att)][1] for ds in datasets for att in ATTACKS if (ds, att) in rel)
    line = f"* **Honest hospitals** were wrongly given zero weight in {pct(hz, hn)} of client-rounds with the relative signal"
    if ab:
        ahz = sum(ab[(ds, att)][0] for ds in datasets for att in ATTACKS if (ds, att) in ab)
        ahn = sum(ab[(ds, att)][1] for ds in datasets for att in ATTACKS if (ds, att) in ab)
        line += f", versus {pct(ahz, ahn)} with the original absolute signal"
    L.append(line + ".")
    eces = [r["test_ece"] for r in runs if r.get("test_ece") is not None]
    if eces:
        L.append(f"* **Calibration:** expected calibration error ranged from {min(eces):.3f} to "
                 f"{max(eces):.3f} across runs, so outputs are model scores, not calibrated risks.")
    low_prev = [name[ds] for ds in datasets if (facts.get(ds, {}).get("prev") or 1) < 0.2]
    if low_prev:
        L.append(f"* **Imbalanced datasets** ({', '.join(low_prev)}): F1/recall at threshold 0.5 are "
                 "low by construction; report ROC-AUC and consider class weighting 'balanced'.")
    L.append("")
    L.append("## 4. Scope")
    L.append("")
    L.append("Simulated hospitals in one program, one configured attacker per run. Low trust is a "
             "policy output, not proof of malicious behaviour, and says nothing about real-world "
             "attack detection. Research prototype; not a medical device.")
    return "\n".join(L) + "\n"


def main() -> int:
    from app.configuration.settings import SettingsManager

    s = SettingsManager().load()
    p = argparse.ArgumentParser()
    p.add_argument("--main", type=Path)
    p.add_argument("--absolute", type=Path)
    p.add_argument("--storage", type=Path, default=s.storage_path)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "RESULTS.md")
    a = p.parse_args()
    auto_main, auto_abs = auto_pick(a.storage / "benchmarks")
    main_d = load(a.main) if a.main else auto_main
    abs_d = load(a.absolute) if a.absolute else auto_abs
    if main_d is None:
        print("No full benchmark found. Run: python scripts\\run_benchmarks.py")
        return 1
    text = build(main_d, abs_d, a.storage)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text, encoding="utf-8")
    print(f"Wrote {a.out}\n  main:     {main_d['path'].name}\n  absolute: "
          f"{abs_d['path'].name if abs_d else '(none – section 2 shows relative only)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
