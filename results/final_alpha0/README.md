# FINAL configuration — trust weighting α = 0.0

`TrustEngine(alpha=0.0)` — explanation-consistency only — selected by the E6
sweep. E1, E3 and E5 rerun with identical datasets, seeds (42/43/44),
attacks, κ values, partitions and evaluation; α is the only change.

* `study_e1_e3_e5_rows.csv` — per-seed results
* `study_e1_robustness.png`, `study_e3_kappa.png`, `study_e5_datasets.png`

Comparison points:
* `results/baseline_alpha0.5/` — original α=0.5 results (produced on the
  user's machine, scikit-learn present)
* `results/env_alpha0.5/`      — α=0.5 rerun in the same environment as this
  run, for a like-for-like comparison
