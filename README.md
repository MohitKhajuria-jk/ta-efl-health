# TA-EFL Health

**Trust-Aware Explainable Federated Learning for Personalized Healthcare Risk Prediction**

A React research workspace backed by the original Python engine. Train a genuine Flower simulation, inspect client trust, predict with saved models, compute real SHAP explanations, and explore preserved experiment history.

This is a single-process research simulation. Model probabilities are not clinically calibrated risks or diagnoses. It does not deploy hospital servers or provide differential privacy or secure aggregation.

## Start the application

Requirements: Windows, Python 3.12 (64-bit), Node.js 22.12+ or a compatible newer LTS. The implementation was verified with Python 3.12.10 and Node 24.18.0.

From this project folder, in PowerShell:

```powershell
# If needed: py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
cd web
npm.cmd ci
npm.cmd run build
cd ..
.\.venv\Scripts\python.exe main.py
```

Open **http://127.0.0.1:8780**. FastAPI serves both the production frontend and API. Sign in with the existing local account. Registration appears only when the configured account store is empty. No default password or production demo account is introduced. The dedicated port avoids an older backend on port 8000 that does not implement this frontend's authentication routes.

For reproducible Python installation, `requirements-lock.txt` records the exact tested environment. `requirements-web.txt` describes supported direct dependencies; `web/package-lock.json` pins the JavaScript dependency tree.

If you forget a local account password, run the recovery tool in an interactive PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe scripts/reset_password.py YOUR_USERNAME
```

Enter the new password twice; input stays hidden. The tool backs up the account store and updates only that account. Restart the Python server to sign out previous sessions, then sign in again. Research data and other accounts are retained. The tool honors `TA_EFL_SETTINGS` or an explicit `--settings` file and is not exposed over HTTP.

If port 8780 is occupied, select an unused port:

```powershell
$env:TA_EFL_PORT='8783'
.\.venv\Scripts\python.exe main.py
```

## Frontend development

Start the Python backend as above. In another PowerShell terminal:

```powershell
cd web
npm.cmd run dev
```

Vite runs at **http://127.0.0.1:5173** and proxies `/api` to port 8780. Both processes must be running. For an alternate backend or frontend port, match the proxy target to the selected Python port:

```powershell
$env:TA_EFL_API_TARGET='http://127.0.0.1:8783'
npm.cmd run dev -- --port 5177
```

Vite is a development server. For the production application, use `npm run build` and the Python server. A standalone Vite preview has no API proxy; it is not the full application.

## Workflows

- **Overview:** real dataset validation, preserved experiment count, latest saved metrics, validation history, and current job status.
- **Federated learning:** choose dataset, strategy, client count, rounds, local training, seed, attacks and advanced trust parameters. Start one real simulation at a time. Cancellation takes effect at safe round boundaries.
- **Hospital network:** inspect coordinator/client activity from engine callbacks; no simulated progress timers or remote-hospital claims.
- **Trust monitoring:** saved per-client trust, validation AUC, performance/similarity signals, sample counts, validity and aggregation weights. Median correctly has no per-client weights.
- **Risk prediction:** choose a saved current-pipeline model, enter schema-valid research features, compute a real prediction and SHAP contributions toward the positive class. Patient inputs and prediction outputs are not persisted.
- **Research analytics:** held-out ROC curves, precision, recall, F1, confusion matrices, calibration metrics and compatible-run comparison.
- **Dataset library:** source, licensing notes, verified counts, missing values, quality notes and feature schema. Existing local dataset import tools remain available.
- **Experiment history:** search, dataset/strategy/attack filters, sort, pagination, record details and PDF/HTML/JSON report downloads.
- **Settings:** Light, Dark and System appearance options; validated defaults for new web experiments, saved separately from original desktop settings; environment and research-scope information.

Comparisons require the same dataset version, seed, test-split protocol, pipeline and decision threshold. Different seeds are not silently pooled. Historical metrics are labelled as saved results; no independent replication or confidence interval is invented.

Use the sun/moon button on the login page, landing page or workspace header to switch between light and dark themes. **Settings → Appearance** also offers System, which follows the device preference. Appearance is saved in this browser, applied before the page paints, and synchronized between tabs. It stays separate from research configuration and account data.

## Hospital-specific calibration (optional personalisation)

An optional extension keeps the trust-aware global model **frozen** and fits, for each simulated hospital, a two-parameter local calibrator `P = sigmoid(a·z + b)` on global-model log-odds `z`, using only that hospital's own calibration rows. This is local probability calibration, not local fine-tuning or personalised feature learning; SHAP still explains the global model only. Hospitals are non-IID partitions of each saved model's held-out test split, each divided into disjoint calibration and test rows. Hospitals with fewer than 20 calibration rows, or fewer than 5 of either class, are reported as not calibrated.

In the saved study (Dirichlet α = 0.5, seeds 1–3), calibration lowered Brier score and ECE on all five datasets, but only 21 of 45 hospitals had enough local data. Within-hospital ROC-AUC is unchanged by construction. With IID hospitals the gains were small and mixed. Full method, results and limitations: [docs/PERSONALIZATION.md](docs/PERSONALIZATION.md).

```powershell
.\.venv\Scripts\python.exe scripts/run_personalization.py
```

Results are stored separately in `storage/personalization/`. In the web app use **Hospital calibration** for study results and run history, and **Risk prediction → Prediction mode → Personalised** to compare global and hospital-calibrated probabilities.

## Dataset inventory

All five datasets passed the existing loader and schema checks in this workspace:

| Dataset | Source | Rows | Positive rows | Features |
|---|---|---:|---:|---:|
| Heart Failure Prediction | Local `data/heart.csv` | 918 | 508 | 11 |
| Pima Indians Diabetes | Local `data/diabetes.csv` | 768 | 268 | 8 |
| Wisconsin Diagnostic Breast Cancer | Bundled in scikit-learn | 569 | 212 | 30 |
| Framingham CHD | Local `data/framingham.csv` | 4,240 | 644 | 15 |
| Stroke Prediction | Local `data/healthcare-dataset-stroke-data.csv` | 5,110 | 249 | 10 |

Presence in this local folder does not grant redistribution rights. Dataset-specific provenance and licence notes remain in the registry and the web library. This project contains no BRFSS target; none is represented as a validated future-heart-attack prediction.

```powershell
.\.venv\Scripts\python.exe scripts/check_datasets.py
```

## Architecture and preservation

```text
web/src/                  React pages, typed client, accessible components, charts
web/public/assets/        Local SVG artwork, genuine product-preview capture
api/                      FastAPI contracts, sessions, research adapters, durable jobs
federated/                Original Flower server/clients/strategies, event callbacks
ml/                       Original datasets, preprocessing, models, prediction, SHAP
experiments/              Original experiment store and sweep tools
app/services/             Reused authentication and report implementation
storage/experiments/      Existing and new research records
storage/models/           Existing joblib models and metadata
storage/web/              Web defaults, durable jobs, temporary exports
main.py                   Primary web launcher
desktop.py                Preserved optional desktop launcher
```

The original 576 experiment records and 576 model artifacts were retained. A SHA-256 manifest and pre-web ZIP backup were saved outside the project before editing, and Git baseline commit `c750d6b` preserves the original source. The ZIP excludes regenerable virtual-environment/cache folders but includes source, datasets, configuration, accounts and research artifacts; keep it private. `scripts/verify_preservation.py <manifest.json>` checks original data/storage/config files without rewriting them. Browser verification uses disposable copies under ignored `test-results/`, never the original account store.

Preprocessing remains fitted only on client training statistics. Validation data informs trust and SHAP background; test data is used for final evaluation. The original algorithms were retained. The runner changes provide observational events, an explicit cancellation exception and a corrected module comment. `python desktop.py` retains the original PyQt6 interface if needed; [desktop documentation](docs/DESKTOP.md) is kept for reference.

## Configuration and security

The server reads the existing `config/settings.json`. `TA_EFL_SETTINGS` can select another local configuration file. Storage and dataset locations are configured locally, not accepted as API paths. Web defaults are stored separately at `storage/web/defaults.json`.

Environment variables are documented in `.env.example`; set them in the process environment (the app does not automatically load dotenv files). Use one Uvicorn worker. A storage lock prevents multiple web API instances using the same storage. Avoid simultaneous desktop/CLI training because those original launchers do not acquire the web lock.

Authentication reuses PBKDF2 local accounts. Opaque HttpOnly/SameSite cookies, CSRF verification, origin/Host restrictions, bounded requests, rate limits and safe artifact paths protect HTTP operations. Sessions expire after eight hours and reset on server restart. Logs contain structured request metadata, not submitted patient profiles or passwords. Accounts share the same local research workspace; enterprise roles are not implemented.

For remote hosting, use an authenticated HTTPS reverse proxy, exact `TA_EFL_ALLOWED_HOSTS`, `TA_EFL_SECURE_COOKIES=1` and an appropriately controlled Python host. No public deployment was created. Sites/Cloudflare Worker hosting cannot run this local Python research engine.

## Verification

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:FLWR_TELEMETRY_ENABLED='0'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
cd web
npm.cmd run build
npx.cmd playwright install chromium
npm.cmd run test:e2e
npm.cmd audit
```

The browser suite starts a temporary production API on port 8781 with copied research artifacts and a test-only account. It shuts down that test server afterward. It needs the production frontend build, an installed Chromium, and access to create subprocesses. Set `QA_BROWSER` to an existing Chromium executable when appropriate. Test evidence and screenshots are in ignored `web/test-results/`; durable browser-test storage is under ignored root `test-results/`.

Exact outcomes, limitations and preservation results are recorded in [VERIFICATION.md](docs/VERIFICATION.md).

## Design and documentation

- [Architecture, lifecycle and security](docs/WEB_ARCHITECTURE.md)
- [API routes and contracts](docs/API.md); authenticated live OpenAPI at `/api/schema`
- [Design resources and honest asset provenance](docs/DESIGN_SOURCES.md)
- [Security controls and remaining dependency advisories](docs/SECURITY.md)
- [Historical research results](docs/RESULTS.md)

The visual system uses navy, light analytical surfaces and restrained teal/mint accents, central CSS tokens with Tailwind v4 theme variables, locally bundled Manrope/DM Sans, Radix dialogs, Recharts, and verified Motion Primitives source. Haikei was accessed but export failed; original local SVG fallbacks are documented. Realtime Colors informed the locally defined palette; Godly redirected to Recent for art-direction research. Reduced-motion settings are respected. The landing-page product preview is a labelled screenshot of actual historical data, not live activity.

Known limits: one workstation/one web simulation at a time; trusted local joblib artifacts; no distributed deployment, secure aggregation, differential privacy, clinical calibration, patient history, enterprise identity, or automatic resumption of interrupted training. API restart marks incomplete jobs as interrupted. Full raw diagnostics and dataset imports remain local administrative operations. The preserved Flower version constrains cryptography to a version with published advisories; see [SECURITY.md](docs/SECURITY.md) before considering remote hosting.
