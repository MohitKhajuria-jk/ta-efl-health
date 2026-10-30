"""Controlled robustness benchmark: strategies x attacks x seeds on genuine datasets.

    python scripts\\run_benchmarks.py                       # every available dataset
    python scripts\\run_benchmarks.py --datasets heart --seeds 1 2 3 --rounds 30

Every run is a full Flower experiment saved under storage\\experiments. A summary is
written to storage\\benchmarks\\<timestamp>.{json,csv,md}. Test metrics come from the
held-out test split; the validation AUC column is reported separately and labelled.
Configured attackers are recorded separately from the trust policy's output.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.configuration.settings import SettingsManager  # noqa: E402
from experiments.store import atomic_write_text  # noqa: E402
from federated.server.runner import run_experiment  # noqa: E402
from ml.datasets.registry import DatasetRegistry  # noqa: E402

STRATEGIES = ["trust", "fedavg", "median"]
ATTACKS = ["none", "sign_flip", "scaling", "gaussian_noise", "label_flip"]


def _mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main(argv=None) -> int:
    s = SettingsManager().load()
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--strategies", nargs="*", default=STRATEGIES)
    p.add_argument("--attacks", nargs="*", default=ATTACKS)
    p.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    p.add_argument("--rounds", type=int, default=s.num_rounds)
    p.add_argument("--clients", type=int, default=s.num_clients)
    p.add_argument("--attacker", type=int, default=None, help="client number (default: last)")
    p.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    p.add_argument("--storage", type=Path, default=s.storage_path)
    p.add_argument("--trust-mode", choices=["relative", "absolute"], default=s.trust_performance_mode)
    a = p.parse_args(argv)
    reg = DatasetRegistry(s.data_path)
    datasets = a.datasets or [k for k in reg.keys() if reg.availability(k).available]
    attacker = a.attacker or a.clients
    runs = []
    total = len(datasets) * len(a.strategies) * len(a.attacks) * len(a.seeds)
    i = 0
    for ds in datasets:
        if not reg.availability(ds).available:
            print(f"skip {ds}: {reg.availability(ds).message}")
            continue
        for strat in a.strategies:
            for att in a.attacks:
                for seed in a.seeds:
                    i += 1
                    cfg = s.experiment_config(
                        dataset_key=ds, strategy=strat, seed=seed, num_rounds=a.rounds,
                        num_clients=a.clients, partition=a.partition,
                        malicious_clients=[] if att == "none" else [attacker],
                        attack_type=att, centralized_baseline=(strat == "fedavg" and att == "none"),
                        trust_performance_mode=a.trust_mode,
                        label="benchmark")
                    rec = run_experiment(cfg, s.data_path, a.storage)
                    tm, vm = rec["test_metrics"], rec["final_validation_metrics"]
                    last = rec["rounds"][-1]["clients"]
                    att_id = f"client_{attacker}"
                    att_rec = next((c for c in last if c["cid"] == att_id), None)
                    honest = [c["trust"] for c in last if c["cid"] != att_id]
                    row = {
                        "dataset": ds, "strategy": strat, "attack": att, "seed": seed,
                        "experiment_id": rec["id"],
                        "test_n": tm["n"], "test_prevalence": tm["prevalence"],
                        "test_roc_auc": tm["roc_auc"], "test_accuracy": tm["accuracy"],
                        "test_precision": tm["precision"], "test_recall": tm["recall"],
                        "test_f1": tm["f1"], "test_brier": tm["brier"], "test_ece": tm["ece"],
                        "validation_roc_auc": vm["roc_auc"],
                        "attacker_final_trust": None if att == "none" or att_rec is None else att_rec["trust"],
                        "attacker_last_weight": None if att == "none" or att_rec is None else att_rec["weight"],
                        "honest_min_trust": min(honest) if honest else None,
                    }
                    runs.append(row)
                    print(f"[{i}/{total}] {ds:13s} {strat:6s} {att:15s} seed={seed} "
                          f"test AUC={tm['roc_auc']:.3f}")
    if not runs:
        print("No runs executed.")
        return 1
    groups: dict[tuple, list] = {}
    for r in runs:
        groups.setdefault((r["dataset"], r["strategy"], r["attack"]), []).append(r)
    summary = []
    for (ds, strat, att), rows in groups.items():
        entry = {"dataset": ds, "strategy": strat, "attack": att, "n_seeds": len(rows)}
        for k in ("test_roc_auc", "test_accuracy", "test_recall", "test_f1", "test_ece",
                  "validation_roc_auc", "attacker_final_trust", "attacker_last_weight",
                  "honest_min_trust"):
            entry[k + "_mean"], entry[k + "_sd"] = _mean_sd([r[k] for r in rows])
        summary.append(entry)
    out = a.storage / "benchmarks"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {"created": stamp, "trust_performance_mode": a.trust_mode, "rounds": a.rounds, "clients": a.clients, "attacker": attacker,
            "partition": a.partition, "seeds": a.seeds,
            "note": "Test metrics: held-out test split. validation_roc_auc: server validation "
                    "split (not test). Attacker = configured; trust/weights are policy outputs."}
    atomic_write_text(out / f"benchmark-{stamp}.json",
                      json.dumps({"meta": meta, "runs": runs, "summary": summary}, indent=1))
    with open(out / f"benchmark-{stamp}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)

    def f(m, sd):
        return "—" if m is None else f"{m:.3f} ± {sd:.3f}"
    lines = [f"# TA-EFL Health benchmark {stamp}", "",
             f"{a.clients} clients ({a.partition}), {a.rounds} rounds, seeds {a.seeds}, trust "
             f"performance signal: {a.trust_mode}, configured "
             f"attacker: Client {attacker}. Mean ± SD over seeds. Test = held-out test split.", "",
             "| Dataset | Strategy | Attack | Test ROC-AUC | Test F1 | Test ECE | Attacker trust | "
             "Attacker weight | Min honest trust |", "|---|---|---|---|---|---|---|---|---|"]
    for e in summary:
        lines.append(f"| {e['dataset']} | {e['strategy']} | {e['attack']} | "
                     f"{f(e['test_roc_auc_mean'], e['test_roc_auc_sd'])} | "
                     f"{f(e['test_f1_mean'], e['test_f1_sd'])} | {f(e['test_ece_mean'], e['test_ece_sd'])} | "
                     f"{f(e['attacker_final_trust_mean'], e['attacker_final_trust_sd'])} | "
                     f"{f(e['attacker_last_weight_mean'], e['attacker_last_weight_sd'])} | "
                     f"{f(e['honest_min_trust_mean'], e['honest_min_trust_sd'])} |")
    atomic_write_text(out / f"benchmark-{stamp}.md", "\n".join(lines) + "\n")
    print(f"\nSummary written to {out / f'benchmark-{stamp}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
