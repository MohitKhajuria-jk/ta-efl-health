# TA-EFL Health

**Trust-Aware Explainable Federated Learning for Personalized Healthcare Risk Prediction** —
a PyQt6 desktop research application: an in-process Flower federated-learning simulation of
hospitals, trust-aware aggregation compared with FedAvg and coordinate-wise median, validated
predictions and SHAP explanations.

> Research prototype. Not a medical device. Outputs are model classifications with
> model-estimated probabilities, not clinically calibrated risks or diagnoses.

---

## 1. Requirements

* Windows 11, **64-bit Python 3.12** (3.13 also works; SHAP 0.53 needs ≥ 3.12).
* No GPU, Ray or network service needed. Internet only for installing packages and
  downloading datasets.

## 2. Install, check data, run (PowerShell)

```powershell
cd C:\path\to\ta_efl_health
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts\fetch_datasets.py      # gets/validates datasets (see section 4)
python scripts\check_datasets.py      # validation report only

python main.py                        # desktop GUI (create a local account on first launch)
```

### Short experiment from the command line

```powershell
python run_experiment.py --dataset breast_cancer --rounds 10 --strategy trust
python run_experiment.py --dataset heart --rounds 30 --malicious 5 --attack sign_flip
python run_experiment.py --list
```

### Regenerate all results (strategies × attacks × 3 seeds, every available dataset)

```powershell
python scripts\run_benchmarks.py                          # ~2 min per dataset
python scripts\run_benchmarks.py --datasets heart --seeds 1 2 3
```

Summaries are written to `storage\benchmarks\benchmark-<time>.{md,csv,json}`.

### Tests

```powershell
python -m pytest tests\test_gui.py -vv
python -m pytest -q
```

Tests that need a genuine dataset file are skipped with a reason when the file is absent.

### View and export results

GUI → **Results & Analysis**, **Trust Analysis**, **Reports** (PDF / HTML / JSON).

```
storage\experiments\<id>.json      experiment record (config, per-round trust/weights, metrics)
storage\models\<id>\               model artifact (model + fitted preprocessing + schema + threshold)
storage\benchmarks\, storage\sweeps\, storage\reports\, storage\logs\
config\settings.json               persisted settings
```

---

## 3. What changed in v2 (summary — see `docs/CHANGELOG.md`)

1. **GUI crash fixed:** `TypeError: native Qt signal is not callable`. Root cause: the background
   workers declared a signal named `event`, shadowing `QObject.event()`, which Qt calls to deliver
   every event (e.g. the thread-change event from `moveToThread`). Signals are now `progress`,
   `succeeded`, `failed`, `done`; regression tests scan all GUI classes for shadowed Qt names.
2. **Preprocessing leakage fixed:** v1 fitted imputation/scaling on the validation split. v2 fits
   them only from the clients' training rows — each simulated hospital sends aggregate statistics
   through Flower, the server combines them exactly. Tests verify equality with scikit-learn fitted
   on the pooled training rows and that the trained model is **bit-for-bit unchanged** when
   validation/test rows are corrupted. Experiments and models from v1 are marked *superseded*:
   they are hidden from metrics, refused by reports, and their models are not loaded.
3. **Trust signal improved:** a benchmark showed the absolute performance signal wrongly gave
   honest hospitals zero weight on harder datasets. The default is now relative to the round's
   median client AUC (see `docs/RESULTS.md`, section 2).
4. Diabetes dataset added (checksum-verified), dataset panel in the GUI, calibration metrics (Brier,
   ECE, reliability plot), deterministic aggregation order, malformed-update attack, client-failure
   handling tests, benchmark script, reproducibility/limitations in reports, UI wording polish.

---

## 4. Datasets

| Dataset | File in `data\` | Status in this package | How to obtain |
|---|---|---|---|
| Breast Cancer Wisconsin (Diagnostic), 569 rows | — | **Bundled** in scikit-learn | nothing to do |
| Pima Indians Diabetes, 768 rows | `diabetes.csv` | **Included** (made by `fetch_datasets.py` from the original UCI distribution, SHA-256 verified) | re-create: `python scripts\fetch_datasets.py --only diabetes` |
| Heart Failure Prediction, 918 rows | `heart.csv` | not included (Kaggle terms) | Kaggle: fedesoriano/heart-failure-prediction |
| Framingham CHD, 4,240 rows | `framingham.csv` | not included (licence unclear) | Kaggle: aasheesh200/framingham-heart-study-dataset |
| Stroke Prediction, 5,110 rows | `healthcare-dataset-stroke-data.csv` | not included (source "confidential", no redistribution) | Kaggle: fedesoriano/stroke-prediction-dataset |

**Easiest way:** download the dataset from its Kaggle page (button *Download*, sign-in needed),
then in the app open **Settings → Dataset availability → Import…** (or **Federated Training →
Import downloaded file…**) and pick the downloaded `.zip` or `.csv`. The file is validated against
the schema before it is copied into `data\` under the correct name.

`scripts\fetch_datasets.py` uses the Kaggle CLI if installed (`pip install kaggle` + API token in
`%USERPROFILE%\.kaggle\kaggle.json`); otherwise it prints the manual steps: sign in to Kaggle,
download the dataset, unzip, copy the CSV with the exact name above into `data\`. Every file is
validated against its schema **and** its documented row/positive counts (heart 918/508, diabetes
768/268, Framingham 4,240/644, stroke 5,110/249). A missing or invalid file is reported, and
training on it is blocked; another dataset is never substituted.

Dataset rules implemented and tested: binary target mapping (breast cancer recoded so 1 =
malignant), documented zero-coded missing values (heart `RestingBP`/`Cholesterol`; diabetes
`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` — but not `Pregnancies`), stroke
`N/A` BMI, `Unknown` smoking status kept as a category, `id` dropped, unknown categories rejected,
duplicates and class imbalance reported, stratified seeded splits.

## 5. Architecture

```
main.py, run_experiment.py           GUI and command-line entry points
app/ui/                              PyQt6 pages, widgets, theme, workers (QThread / QThreadPool)
app/services/                        context, auth, charts, reports, reference values, jobs
app/configuration/settings.py        validated settings
federated/clients/                   HospitalClient (Flower NumPyClient), attack simulations
federated/server/                    runner (Flower Server.fit), ClientProxy transport
federated/strategies/                trust policy, TA-EFL / FedAvg / FedMedian strategies
ml/                                  datasets, preprocessing (v2), model, prediction, SHAP, metrics
experiments/                         experiment store, sweeps
scripts/                             fetch_datasets, check_datasets, run_benchmarks
docs/                                RESULTS.md, CHANGELOG.md
tests/                               pytest suite
```

**Federated simulation (in-process).** Flower's `Server`, `SimpleClientManager` and strategies
coordinate one `NumPyClient` per simulated hospital in a single program. Messages pass through
Flower's `ClientProxy` interface and are encoded/decoded with Flower's protobuf format; no
sockets are opened. This is not a deployed multi-machine network.

* **Stays on each client:** its raw rows and labels; its local train/evaluation split.
* **Client → server:** aggregate preprocessing statistics of its training rows (counts, means,
  sums of squared deviations, category counts), model parameters `[coef, intercept]`, number of
  training examples, self-reported loss, local evaluation loss/accuracy.
* **Server → client:** global parameters, training configuration, fitted preprocessing parameters.
* **Server-held:** validation split (trust signal, per-round monitoring, SHAP background) and the
  test split (final evaluation only).
* No differential privacy, secure aggregation or encryption.

**Trust-aware aggregation (TA-EFL)**

```
P_k  = clip((AUC_k - 0.5) / (median_j AUC_j - 0.5), 0, 1)   relative (default); absolute: /0.5
S_k  = max(0, cos(w_k - w_g, median_j(w_j - w_g)))
r_k  = λ·P_k + (1-λ)·S_k                                     λ = 0.6
T_k(t) = α·T_k(t-1) + (1-α)·r_k(t),  T_k(0) = 0.5            α = 0.7
ω_k  = T_k·n_k / Σ_j T_j·n_j  for valid updates with T_k ≥ τ (τ = 0.3), else 0
```

AUCs are measured by the server on the validation split from each client's submitted model.
Malformed updates (wrong shapes, NaN/Inf) are rejected by all three strategies; if no update
qualifies, the previous global model is kept. Failed clients keep their trust and get no weight.
Client results are aggregated in client-ID order, so runs are bit-for-bit reproducible.

**Model.** L2-regularised logistic regression trained by genuine mini-batch SGD (scikit-learn
estimator API). A scikit-learn `LogisticRegression` on the pooled training rows is reported only
as a centralised reference.

**Explanations.** `shap.LinearExplainer` (independent masker, server-validation background) in
log-odds space; additivity is checked; one-hot columns are summed per original feature.

## 6. Results

See **`docs/RESULTS.md`** (generated by `run_benchmarks.py`, pipeline v2): strategy comparison on
breast cancer, diabetes and Framingham under five attack settings, and the absolute-vs-relative
trust analysis. Heart and stroke results must be generated on your machine.

## 7. Verification status

| Item | Status |
|---|---|
| Backend, leakage, datasets, federated, trust, attacks, SHAP, reports, settings | `pytest` in sandbox: see `docs/CHANGELOG.md` for the exact count |
| Genuine-data end-to-end (experiment → artifact → prediction → SHAP) | run for breast cancer, diabetes and Framingham (verification copy) |
| GUI logic, all pages | executed headless with a stand-in Qt layer that emulates Qt event dispatch |
| GUI on real PyQt6 (`tests\test_gui.py`, `tests\test_qt_naming.py` live part) | **must be run on Windows** – PyQt6 could not be installed in the build sandbox |

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `No matching distribution for shap==0.53.0` | Use 64-bit Python 3.12 or 3.13 |
| Qt platform plugin error | `pip install --force-reinstall PyQt6 PyQt6-Qt6`; don't mix conda and pip Qt |
| Dataset *missing* / *invalid* | Use the exact file name from section 4 in `data\`, or set Settings → Data Directory |
| Experiments shown as *superseded* | Produced by pipeline v1; rerun them (they are kept on disk, not deleted) |
| Old model "trained with an earlier preprocessing pipeline" | Retrain on the Federated Training page |
| Training diverged | Lower the learning rate (advanced settings) |
| Forgot password | Login → *Forgot Password?* resets local accounts (experiments kept) |

## 9. Limitations

* In-process simulation, not a deployed network; no formal privacy guarantee.
* One linear model family; no non-linear federated models.
* Trust policy assumes an honest majority; the scaling attack is not detected; low trust is a
  policy output, not proof of maliciousness.
* Fixed decision threshold (0.5); probabilities are not calibrated (see ECE in results).
* Mean (not median) imputation, because medians cannot be combined exactly across hospitals.
