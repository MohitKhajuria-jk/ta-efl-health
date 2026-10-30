# Changelog

## v2.3 — GUI visibility fixes

* **Login button invisible:** the login pane used an unscoped stylesheet (`background: white`),
  which in Qt cascades to every child and overrode the blue button style, leaving white text on a
  white button. Panels are now styled by object-name selectors (`#LoginPane`, `#ResultBox`,
  `#TipBox`, `#ReportPaper`).
* **Unreadable text in Windows dark mode:** Qt 6 adopts the dark system palette; unpainted surfaces
  (dropdown lists, dialogs, tab pages, message boxes) turned dark under dark text. `theme.apply_theme`
  now forces Fusion, a complete light palette and the light colour scheme (Qt ≥ 6.8), plus explicit
  rules for popups, menus, dialogs and group boxes.
* Page subtitles wrap instead of clipping; experiment pickers no longer squeeze the header.
* `tests/test_gui_visual.py`: pixel-contrast tests (login button colour and text, every page title,
  sidebar items, tables, tabs, dialogs) run under a simulated dark system palette.

## v2.2

* `tests/test_gui.py`: the GUI fixture now uses an isolated, empty data folder. Previously
  `test_training_page_shows_dataset_status_and_blocks_missing` read the developer's real `data\`
  folder and failed once the stroke file had been imported (the app correctly reported it as
  valid). App code unchanged.

## v2.1

* Fixed `tests/test_gui.py::test_reference_mode_is_labelled`: the test still expected "—" after the
  dashboard was changed to show "Not run" for metrics of experiments that have not been run.
* New in-app dataset import (`ml/datasets/importer.py`, `app/ui/dataset_actions.py`): pick the
  downloaded Kaggle `.zip` or `.csv`; it is validated against the schema before being copied into
  `data\` under the expected name. Buttons on the Training dataset panel and in Settings.
* Dashboard dataset table no longer truncates names; tooltips explain missing files.

## v2 — repair, leakage fix, datasets, robustness study

### Critical bug 1 — `TypeError: native Qt signal is not callable` (GUI tests)

* **Root cause (confirmed):** `ExperimentWorker` and the thread-pool signal holder in
  `app/ui/workers.py` declared `event = pyqtSignal(object)`. `QObject.event(QEvent)` is the
  virtual method Qt calls to deliver every event to an object (for example the
  `ThreadChange` event sent by `moveToThread`, deferred deletes, queued calls). The signal
  replaced that method, so Qt's event loop tried to call the signal object. `ExperimentWorker`
  also stored its thread in `self.thread`, shadowing `QObject.thread()`.
* **Fix:** signals renamed to `progress`, `succeeded`, `failed`, `done` (always emitted last);
  thread stored as `_qthread`; pages connect `done` and call `wait()` before releasing the
  worker; no widget is touched from the worker thread.
* **Regression tests:** `tests/test_qt_naming.py` (static scan of every GUI class for signals or
  attributes that shadow Qt members — fails on the v1 code; plus a live PyQt6 introspection
  test), `tests/test_gui.py` (error, success, cancellation, thread-pool jobs, dataset panel,
  missing-model state, training → prediction → SHAP workflow).

### Critical bug 2 — preprocessing fitted on the validation split

* **Root cause:** `federated/server/runner.py` called `fit_preprocessor(spec, splits.X_val)`.
* **Fix (pipeline v2):** `ml/preprocessing/pipeline.py` now provides `SchemaPreprocessor`, fitted
  only from training rows. In federated runs each client returns aggregate statistics of its
  local training rows through Flower `get_properties`; the server combines them exactly
  (parallel mean/variance) and sends the fitted parameters to clients in the fit/evaluate config.
  Validation and test rows are only transformed. Imputation changed from median to mean
  because medians cannot be combined exactly across hospitals.
* **Other leakage checks:** threshold fixed before training; trust and per-round monitoring use
  the validation split only; test split used only for final metrics; no model selection on test;
  the centralised reference fits its own preprocessing on pooled training rows.
* **Tests:** `tests/test_leakage.py` — equality with scikit-learn fitted on training rows,
  federated statistics == pooled fit, validation/test perturbation does not change the fitted
  transform, and the trained model is bit-for-bit identical when test rows (FedAvg/Median: also
  validation rows) are corrupted.
* **Stale results:** records and artifacts now carry `pipeline_version = 2`. v1 experiments are
  shown as *superseded*, excluded from dashboards/pickers, refused by reports; v1 models are not
  loaded. Nothing is deleted.

### Datasets

* `data/diabetes.csv` added (original UCI Pima distribution, pinned commit, SHA-256 verified,
  documented header added) via `scripts/fetch_datasets.py`.
* Documented row/positive counts and licence notes per dataset; duplicate and class-imbalance
  warnings; minimum record check; file SHA-256 recorded in experiments.
* GUI dataset panel (training page): description, target mapping, usable records, class
  distribution, missing values, validation status, licence; training blocked for missing or
  invalid files.
* `tests/test_all_datasets.py`: schema-fixture tests for all CSV datasets (sentinel zeros, stroke
  `N/A`/`Other`/`id`, unknown categories, missing columns, bad targets, reproducible splits) and
  genuine-data tests that run when the real file is present.

### Federated learning, trust and evaluation

* Client results sorted by client ID before aggregation → bit-for-bit reproducible runs.
* New `invalid_update` (NaN) attack; tests for malformed updates (all strategies), crashing
  clients, and no-valid-update rounds.
* Trust performance signal: new default *relative* mode (AUC relative to the round's median
  client); *absolute* mode kept. Justified by the benchmark in `docs/RESULTS.md`.
* Metrics: prevalence, negatives, Brier score, expected calibration error, reliability bins.
* `scripts/run_benchmarks.py`: strategies × attacks × seeds on every available dataset.

### GUI and reports

* Dashboard: latest-experiment summary (strategy, status, participation, test metrics with n and
  prevalence, trust summary); "Not run" vs "N/A" distinguished; superseded runs labelled.
* Results: Test / Validation / centralised columns, calibration plot, readable labels.
* Trust, Clients, Settings, Training: readable client/attack names, performance-signal option,
  wording that separates configured attackers from trust-policy output.
* Reports: reproducibility section, validation-vs-test distinction, aggregation weights,
  calibration chart, limitations; superseded experiments refused.
* About: research objective, architecture, simulation and non-clinical scope.

### Files changed

| File | Why |
|---|---|
| `app/ui/workers.py` | Qt signal-shadowing fix, worker lifecycle (`done`, `wait`) |
| `app/ui/pages/training.py`, `results.py` | new worker signals; dataset panel; labels; calibration |
| `app/ui/pages/dashboard.py`, `clients.py`, `trust.py`, `settings.py`, `about.py`, `reports.py`, `prediction.py` | current-pipeline data only, wording, options |
| `app/ui/labels.py` (new) | human-readable names for stored identifiers |
| `ml/preprocessing/pipeline.py` | pipeline v2 (training-only, federated statistics) |
| `federated/clients/hospital_client.py` | statistics via Flower, preprocessing from config |
| `federated/server/runner.py` | federated preprocessing, provenance, reproducibility fields |
| `federated/strategies/strategies.py`, `trust.py` | deterministic order, relative performance signal |
| `federated/clients/attacks.py` | malformed-update attack |
| `ml/models/artifact.py` | pipeline-version check, corrupted-artifact handling |
| `ml/datasets/*` | documented counts, licences, duplicates, imbalance, summary service |
| `ml/evaluation/metrics.py` | prevalence, Brier, ECE, reliability bins |
| `ml/explainability/shap_service.py` | reference values from the v2 preprocessor |
| `experiments/store.py`, `sweeps.py` | superseded-record handling, readable sweep names |
| `app/services/reports.py`, `charts.py` | report sections, calibration chart |
| `app/configuration/settings.py` | trust performance mode setting |
| `scripts/fetch_datasets.py`, `run_benchmarks.py` (new) | dataset acquisition/verification, benchmarks |
| `tests/*` | new leakage, dataset, Qt-naming, robustness, report tests; training-only fitting |
| `README.md`, `data/README.md`, `docs/RESULTS.md` | documentation and regenerated results |

### Verification in the build sandbox

`python -m pytest -q`: **104 passed, 8 skipped, 0 failed** (Linux, Python 3.13; Flower 1.26.1,
SHAP 0.53.0 and pytest run from source because PyPI was unreachable). Skipped: `tests/test_gui.py`
and the live part of `tests/test_qt_naming.py` (PyQt6 not installable there) and six genuine-data
tests for heart, Framingham and stroke (files not present). The Framingham genuine-data tests
also passed (2/2) against a verification copy that was not bundled.
