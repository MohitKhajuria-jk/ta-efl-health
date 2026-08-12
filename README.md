# TA-EFL Health

**Trust-Aware Explainable Federated Learning for Personalized Healthcare Risk Prediction** — PyQt6 desktop application + [Flower](https://flower.ai)-based research simulation framework, now running over **five clinical datasets**.

## Quick start

```bash
pip install -r requirements.txt
python main.py                              # launch the desktop app
python run_experiment.py                    # headless experiment (plots -> results/)
python run_experiment.py --dataset all      # every dataset in turn
python run_study.py                         # full paper study (E1-E5)
python tests/test_smoke.py                  # smoke tests
```

## Datasets

Five binary-outcome clinical datasets, selectable in the GUI (Federated Training screen) and via `--dataset` on every script:

| key | dataset | task | features | positive class |
|---|---|---|---|---|
| `heart` | UCI/Kaggle Heart Disease (Cleveland) | coronary artery disease | 13 | disease present |
| `diabetes` | Pima Indians Diabetes | 5-year diabetes onset | 8 | onset |
| `breast_cancer` | Breast Cancer Wisconsin (Diagnostic) | malignant vs benign | 10 (mean nucleus features) | malignant |
| `framingham` | Framingham Heart Study | 10-year CHD risk | 14 | CHD within 10 years |
| `stroke` | Healthcare Stroke Prediction | stroke event (~10% prevalence) | 10 | stroke |

Each dataset resolves its source in this order:

1. a **real CSV** in `data/` (`heart.csv`, `diabetes.csv`, `framingham.csv`, `healthcare-dataset-stroke-data.csv`, `wdbc.csv`, …) — column names and categorical encodings from the usual Kaggle releases are adapted automatically;
2. a **built-in real copy** where one exists (Breast Cancer Wisconsin ships with scikit-learn);
3. otherwise a **calibrated synthetic cohort** with realistic marginals, correlations and prevalence, so the app always runs offline.

Adding a sixth dataset means writing one module under `core/datasets/` with a `DatasetSpec` (features, ranges, CSV adapter, synthetic generator) and listing it in `core/datasets/registry.py`. Everything else — GUI form, SHAP labels, partitioning, experiments — picks it up automatically.

## Federated learning with Flower

The federated engine is written against the Flower APIs:

```
core/fl/client_app.py   HospitalClient(NumPyClient) + ClientApp   (one SuperNode per hospital)
core/fl/server_app.py   ServerApp / ServerAppComponents
core/fl/strategy.py     TrustWeightedFedAvg | PlainFedAvg | CoordinateMedian  (flwr Strategy)
core/fl/data.py         deterministic dataset -> Dirichlet partitions (rebuildable per node)
core/fl/runner.py       FederatedRunner: one object tying strategy + clients together
core/fl/sim.py          in-process driver that steps the same Strategy one round at a time
core/fl/compat.py       flwr type imports, with equivalents when flwr is absent
```

Two execution backends, selectable in the GUI ("Execution") or with `--backend`:

* **`local`** (default) — the in-process driver runs the identical `configure_fit → fit → aggregate_fit → evaluate` loop synchronously, one round per call. This is what gives the GUI live progress and a working Stop button, and it keeps the study scripts deterministic.
* **`flower`** — Flower's own simulation runtime (`flwr.simulation.run_simulation`) with real SuperNodes/SuperLink and Ray. Requires `pip install "flwr[simulation]"`; falls back to `local` with a warning if it is unavailable.

The app is also a valid Flower app: `pyproject.toml` registers `core.fl.server_app:app` and `core.fl.client_app:app`, so

```bash
flwr run . --run-config "dataset='framingham' strategy='trust' num-server-rounds=30"
```

works from the project root.

## What it does

1. Patient data is split across N simulated hospitals with a **non-IID Dirichlet partition** (lower α = more heterogeneous populations). Partitions are a pure function of `(dataset, n_clients, α, seed)`, so every SuperNode rebuilds its own shard — no patient data is ever shipped between processes.
2. Each round, every hospital trains a local logistic-regression risk model starting from the global model and submits its weights. Optional **attacks**: label flipping, Gaussian noise, scaled model poisoning.
3. The server computes a **trust score** per hospital from:
   - *Update similarity*: cosine similarity of the client's update to the cohort's coordinate-wise median update, with a magnitude penalty;
   - *Explanation consistency*: cosine similarity of the client's **extrapolated attribution signature** (signed SHAP direction of θ_g + κ(θ_k − θ_g) on a shared public reference set) to the cohort median signature — computed server-side, so it cannot be spoofed;
   - *History*: exponential moving average `T_k(t) = β·T_k(t−1) + (1−β)·[α·s_upd + (1−α)·s_exp]`.
4. Aggregation is **trust-weighted FedAvg**: weights ∝ n_k · T_k^γ, with clients below τ excluded. Baselines: plain FedAvg and coordinate-wise median.
5. Every prediction ships with an **exact SHAP explanation** (closed form for the linear model, log-odds space).

## Results (2 of 6 hospitals malicious, 30 rounds, 3 seeds)

| dataset | TA-EFL AUC | FedAvg | Coord. median | trust separation |
|---|---|---|---|---|
| Heart Disease | **0.784** | 0.745 | 0.783 | +0.24 |
| Pima Diabetes | **0.887** | 0.839 | 0.883 | +0.27 |
| Breast Cancer | 0.919 | 0.893 | **0.922** | +0.27 |
| Framingham CHD | **0.824** | 0.754 | 0.801 | +0.21 |
| Stroke | **0.918** | 0.838 | 0.901 | −0.04 |

Trust weighting beats FedAvg on every dataset. On the heavily imbalanced stroke data the honest/attacker trust *separation* narrows to slightly negative — one honest hospital with a very skewed shard scores below the noise attacker — which is an honest limitation worth reporting: the explanation signal is weakest where honest clients genuinely disagree. See `results/study_e5_datasets.png`.

## Project layout

```
core/datasets/      dataset registry (heart, diabetes, breast_cancer, framingham, stroke)
core/dataset.py     backwards-compatible facade over the registry
core/partition.py   Dirichlet non-IID split
core/model.py       NumPy logistic regression + exact linear SHAP
core/metrics.py     metrics (scikit-learn, with NumPy fallbacks)
core/trust.py       trust engine
core/fl/            Flower client app, server app, strategies, runner, driver
core/session.py     wiring: dataset -> clients -> federated run
core/server.py      compatibility alias for core.fl.runner
gui/app.py          PyQt6 application (all screens)
run_experiment.py   strategy comparison, per dataset
run_study.py        paper study E1-E5
tests/test_smoke.py smoke tests
pyproject.toml      Flower app definition (`flwr run .`)
```

## App screens

Login → Dashboard (global metrics, trust bars) → Clients table → Federated Training (dataset, hospitals, rounds, non-IID α, aggregation strategy, attack scenario, execution backend; live progress) → Trust Management (per-round trust components + history plot) → Model Evaluation (confusion matrix, ROC) → Explainability (per-patient SHAP) → Predictions (dataset-specific patient form, risk + explanation) → Reports (CSV/PNG export).

## Notes & disclaimer

Research prototype for educational purposes. Not a medical device; predictions must not be used for real clinical decisions.
