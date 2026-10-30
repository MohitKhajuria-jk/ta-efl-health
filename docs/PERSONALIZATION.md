# Hospital-specific personalisation (local probability calibration)

TA-EFL Health adds an optional **hybrid global–local personalisation** step: the existing
trust-aware global model stays frozen, and every simulated hospital fits a two-parameter
logistic calibrator on its own local calibration rows.

This is **local probability calibration**. It is *not* local fine-tuning, it does not train
a model per hospital, and it does not learn hospital-specific features.

## Architecture

```text
existing Flower run (unchanged) ──► global model artifact (storage/models, read only)
                                            │  z_global = w·x + c   (log-odds)
held-out test split of that run ──► simulated hospitals (Dirichlet label skew or IID)
                                            │  each hospital: calibration rows | test rows
ml/personalization/calibration.py ──► (a_h, b_h) per hospital, fitted on its calibration rows
ml/personalization/evaluator.py   ──► global vs personalised metrics on identical test rows
ml/personalization/storage.py     ──► storage/personalization/<run>/{record.json, calibrators/*.json}
federated/personalized_runner.py  ──► one run = one global model × one hospital partition
scripts/run_personalization.py    ──► G1–G4 study over saved benchmark models + report
api/personalization.py            ──► read-only results + personalised prediction endpoint
web/src/pages/personalization.tsx ──► study reports, history, per-hospital results
web/src/pages/prediction.tsx      ──► Global / Personalised selector and hospital selector
```

No existing aggregation, training, preprocessing, dataset-loading or SHAP code was changed.
Historical experiment records and model artifacts are only read.

## Mathematical formulation

For hospital *h* and global log-odds *z*:

    P_personalized(y = 1 | x) = sigmoid(a_h · z + b_h)

(a_h, b_h) minimise, over that hospital's calibration rows only,

    J(a, b) = Σ_i [ log(1 + e^{s_i}) − y_i s_i ] + (λ/2)·((a − 1)² + b²),   s_i = a·z_i + b

* λ = 1.0 shrinks the calibrator towards the identity (a = 1, b = 0, i.e. the global model).
  It was fixed before any experiment and never tuned. Because it penalises the summed
  log-loss, its influence shrinks as local data grows.
* J is strictly convex; Newton's method with backtracking line search converges to the unique
  minimum and stays finite even on perfectly separable data. The fit is deterministic.
* Stable computation: `logaddexp` for the log-likelihood, a branch-wise sigmoid, and clipped logit.
* **a must be > 0.** A fit with a ≤ 0 would reverse the global model's risk ranking and is
  rejected with an explicit error.
* Consequence of a > 0: the mapping is strictly increasing, so **ROC-AUC within one hospital is
  unchanged by construction**. Probabilities, Brier score, log loss, ECE and threshold-based
  metrics (F1, precision, recall at the fixed 0.5 threshold) can change. Pooled AUC across
  hospitals can also change because each hospital is shifted differently.

## Minimum local sample requirements

A calibrator is fitted only if the hospital's calibration subset has at least **20 rows** and at
least **5 rows of each class**. Otherwise the hospital is recorded as
`insufficient_calibration_data` (or `calibration_rejected` for a non-positive slope). It keeps the
global model, is shown as *not calibrated* in the interface, and is excluded from personalised
comparisons. There is no silent fallback that is reported as personalisation.

## Experimental protocol

* **Global models (frozen):** the no-attack runs of `storage/benchmarks/benchmark-20261014-052600`
  (the benchmark behind `docs/RESULTS.md`): 5 datasets × {FedAvg, FedMedian, trust-aware} × seeds 1, 2, 3.
  5 IID clients, 30 rounds. Nothing is retrained.
* **Split verification:** for each model, the original splits are regenerated from the saved seed and
  fractions. The run proceeds only if the dataset SHA-256, split sizes and the saved test ROC-AUC
  are reproduced exactly.
* **Hospital data:** the model's held-out **test split**, never used for federated training,
  preprocessing, trust scoring or threshold selection. It is partitioned into 3 hospitals with the
  existing `partition_indices` function:
  * primary: Dirichlet label skew, α = 0.5 (the project's existing default), seed = model seed;
  * sensitivity: α = 1.0, and IID.
* Each hospital is split 50/50 (stratified, seeded) into **local calibration** and **local test**
  rows. No row appears in two subsets or two hospitals; this is checked on every run and in tests.
* The server-held validation split is not used. There is no per-hospital or central tuning.
* Partitions depend only on dataset, seed and labels, so G1–G4 for one dataset and seed use
  identical hospital rows (verified through a partition hash).
* **Groups:** G1 FedAvg global, G2 FedMedian global, G3 trust-aware global, G4 = G3 + hospital
  calibration. G3 vs G4 is the primary comparison. Calibrated FedAvg and FedMedian are reported
  as supplementary rows.
* **Matched evaluation:** for each dataset and seed all groups are scored on the same rows, namely
  the local test rows of hospitals that received a G4 calibrator. Reports show mean ± SD over
  seeds, plus every hospital individually.

## Evaluation metrics

ROC-AUC, F1, precision, recall, accuracy, Brier score, log loss and expected calibration error
(10 equal-width bins). All use the same `classification_metrics` function as every historical
experiment, at the model's fixed 0.5 threshold. A single-class test partition reports ROC-AUC as
undefined (`None`) with a note, never a number.

## Results (from the saved reports in `storage/personalization/reports/`)

Primary protocol (Dirichlet α = 0.5, 3 hospitals, seeds 1–3). Only **21 of 45** hospitals had
enough local calibration data: breast cancer 3/9, diabetes 4/9, Framingham 5/9, heart 5/9,
stroke 4/9.

| Dataset | G3 Brier | G4 Brier | G3 ECE | G4 ECE | G3 F1 | G4 F1 | Brier improved (hospitals) |
|---|---|---|---|---|---|---|---|
| Heart | 0.106 ± 0.008 | 0.090 ± 0.008 | 0.147 ± 0.027 | 0.096 ± 0.046 | 0.890 | 0.885 | 5/5 |
| Diabetes | 0.190 ± 0.050 | 0.180 ± 0.045 | 0.192 ± 0.070 | 0.172 ± 0.078 | 0.634 | 0.640 | 3/4 |
| Breast cancer | 0.031 ± 0.016 | 0.022 ± 0.018 | 0.112 ± 0.014 | 0.070 ± 0.039 | 0.986 | 0.970 | 3/3 |
| Framingham | 0.209 ± 0.083 | 0.140 ± 0.075 | 0.164 ± 0.106 | 0.076 ± 0.047 | 0.127 | 0.692 | 5/5 |
| Stroke | 0.174 ± 0.186 | 0.097 ± 0.059 | 0.156 ± 0.225 | 0.083 ± 0.112 | 0.000 | 0.287 | 3/4 |

Within-hospital ROC-AUC changed by exactly 0.000 in every calibrated hospital.

Sensitivity:

* **α = 1.0 (moderate skew, 26/45 hospitals calibrated):** smaller gains. Brier improved in 20 of
  26 hospitals and ECE in 18 of 26. Diabetes ECE was slightly worse on average
  (0.152 → 0.167).
* **IID (no prevalence shift, 36/45 calibrated; breast cancer 0/9 because each hospital has only
  19 calibration rows):** changes were small and mixed. ECE improved in 28 of 36 hospitals, but
  Brier improved in only 18 of 36. Heart pooled AUC fell from 0.921 to 0.913, and heart and
  diabetes F1 fell slightly.

**Interpretation.** Local calibration mainly corrects differences in prevalence between a
hospital and the training population. It helps most when those differences are large, and it
adds little, or slightly hurts, when hospitals resemble the training data. It never changes a
hospital's risk ranking. Sample sizes are small (3–9 calibrated hospitals per dataset and
setting), so no statistical significance is claimed.

## SHAP and calibration

SHAP values (unchanged implementation) explain the **global model's log-odds**. The calibration
is a separate monotone adjustment applied afterwards. The personalised prediction view shows the
global SHAP explanation, the global probability, the calibrated probability, and the hospital's
adjustment ((a − 1)·z + b in log-odds), and states that SHAP does not explain the calibration step.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts/run_personalization.py                        # primary (α = 0.5)
.\.venv\Scripts\python.exe scripts/run_personalization.py --alpha 1.0            # sensitivity
.\.venv\Scripts\python.exe scripts/run_personalization.py --partition iid        # control
.\.venv\Scripts\python.exe scripts/run_personalization.py --report-from <run ids>  # rebuild a report
```

Every invocation creates new run IDs and a new report; earlier runs are never overwritten.
Rerunning the primary protocol reproduced the saved report exactly (apart from its timestamp).

## Assumptions and limitations

* Hospitals are **simulated** partitions of public datasets: label skew only (no covariate or
  measurement shift), 3 hospitals, one partition per seed.
* The global model was trained on IID clients. The "hospitals" are deployment sites that differ
  from training, not the same clients that trained the model.
* Local test sets are small (often 14–60 rows), so per-hospital metrics are noisy; many hospitals
  fall below the minimum sample requirement.
* λ, the minimum sample requirements and the 50/50 split were fixed in advance, not tuned. Other
  choices would give different coverage.
* Calibrated probabilities are still research estimates, not clinically validated risks.
* No privacy mechanism is added. A real deployment would need each hospital to hold its own
  labelled calibration data.
* Reproducibility: deterministic on a given platform. Across platforms, floating-point results
  may differ by about 1e-15, as with the historical experiments.
