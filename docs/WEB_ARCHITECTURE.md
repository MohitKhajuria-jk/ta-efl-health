# Web architecture

## Ownership and execution

React/TypeScript/Vite owns presentation in `web/`. FastAPI in `api/` owns HTTP authentication, validation, orchestration and safe storage adapters. Existing `ml/`, `federated/`, `experiments/` and report services remain the research implementation. `main.py` is now the web launcher; `desktop.py` preserves the original GUI entry point, and all original GUI code/tests remain available.

The browser uses same-origin `/api` requests. In development Vite proxies those to port 8780, matching the Python launcher's default. This avoids an older incompatible backend on port 8000. In production FastAPI serves the built `web/dist` directory and SPA routes. There are no browser ML models, remote hospital servers, mock API handlers or fabricated progress timers.

FastAPI synchronous endpoints execute in its thread pool. One dedicated `ThreadPoolExecutor` worker runs `run_experiment`; the ASGI loop is not used for training, prediction, SHAP, dataset validation or reports. An OS file lock prevents two web API processes using the same research storage. Run **one Uvicorn worker**. Avoid simultaneous desktop/CLI training while the web worker is running: the older launchers do not acquire the web process lock.

## Scientific boundary

`run_experiment` still partitions data, requests client training statistics via Flower, combines preprocessing statistics, constructs the existing strategies, runs `Server.fit`, evaluates the saved model and persists the artifact. No optimizer, aggregation, trust, partition or preprocessing mathematics were changed. Small observational callbacks were added at client fit/submission, aggregation and server-evaluation boundaries. The misleading validation-preprocessing module comment was corrected.

Prediction loads the existing joblib artifact with its schema, training-fitted preprocessor, threshold and validation-background distribution. The API verifies the requested experiment identity against the artifact. Real `predict` and `explain_prediction` generate the result; explanations target class 1, so positive/negative SHAP values consistently refer to the positive-class score. Patient inputs/results are not stored or logged by the web layer. There is intentionally no patient prediction history.

Comparison requires matching dataset key, dataset fingerprint/row count, pipeline, test/validation fractions, stratification, seed, test evaluation split and decision threshold. Incompatible comparisons return 409. Analytics shows individual saved evaluations; it does not pool incompatible splits or invent repeated-run confidence intervals. Historical results retain their original identifiers, timestamps and origin. New completed web runs add `origin: web` and a web job identifier.

## Durable job lifecycle

`storage/web/jobs/<opaque-id>.json` records configuration, timestamps, stage, completed aggregation rounds, client events, trust records and the resulting experiment identifier. A lock serializes callbacks from Flower client threads. Writes use the existing atomic replace helper. Major lifecycle events and each aggregated round are persisted; in-memory polling also exposes current client activity. The event log retains the latest 250 events; all round aggregation records are retained.

Stages map to real callbacks: initializing → local_training → aggregating → evaluating. Completed/cancelled/failed are terminal job states. Polling is once per second and may miss very brief phases, which remain visible in the event log; no artificial delays are introduced. A cancel request signals the existing stop event and takes effect at a round boundary. A partial engine result remains `stopped`; its web job is `cancelled`. Cancellation before the first completed round creates no model. A request arriving after the final training boundary may finish normally.

On restart, unfinished durable jobs become `interrupted`, with a clear message; they are never silently resumed or relabelled completed. Session cookies cease to work after restart because sessions are held in memory. New jobs are rejected with 409 while another is active.

## Identity and safety

Existing `storage/users.json` PBKDF2 identities are reused without rewriting them. First-account registration is available only when the store is empty; additional account creation requires a valid authenticated session and CSRF token. Password reset is not exposed over HTTP. All local accounts have the same shared-workstation access, matching the original project; there is no claim of enterprise roles or per-patient authorization.

Sessions use random opaque identifiers, an HttpOnly SameSite=Strict cookie, an eight-hour absolute expiry, rotation on login and explicit logout. Mutating authenticated requests need the per-session CSRF header. Cross-origin writes, unexpected Host headers and bodies over 64 KiB are rejected. Login/registration/training/prediction/report rates are bounded. The JSON validation handler removes submitted input values; unexpected errors do not expose tracebacks. No permissive CORS is enabled.

Resource identifiers are restricted, model paths must resolve within the configured model directory, and arbitrary file paths/model uploads are not accepted. Joblib artifacts remain trusted local inputs; metadata checks do not make pickle safe for untrusted files. Report exports use temporary server-chosen names, preserve the existing report implementation and are removed after download. Matplotlib report generation and model prediction are serialized separately.

Production responses set CSP, frame, content-type and referrer headers; API data uses `Cache-Control: no-store`. Fonts and SVG assets are local. For anything beyond loopback, configure HTTPS, `TA_EFL_SECURE_COOKIES=1`, exact allowed hosts and an appropriate authenticated reverse proxy. This work does not assert internet-facing clinical or regulatory readiness.

## Storage and configuration

- Existing `data/`, `storage/experiments/`, `storage/models/`, `storage/benchmarks/`, `storage/users.json`, `config/settings.json`: preserved.
- `storage/web/defaults.json`: independently persisted, validated defaults for new web experiments; does not rewrite desktop or historical configurations.
- `storage/web/jobs/`: restart-aware job status and progress.
- `storage/web/exports/`: temporary report download files.
- `test-results/web-qa-*/`: isolated browser-test copies; actual accounts are not copied.

Dataset management exposes validated summaries and source links, not patient rows or arbitrary paths. Existing local import/fetch tools remain the way to add dataset files. The web uses no database and introduces no destructive migration.

## Frontend structure

`src/api.ts` is the typed fetch/query boundary; React Query manages server state. React Hook Form and Zod validate authentication, training, settings and dynamic prediction schemas. Python repeats all consequential validation. Radix Dialog provides focus trapping, Escape handling and modal semantics. Recharts visualizes stored evaluations and trust values; numeric tables provide the underlying values. Custom SVG renders the interactive coordinator/client network. Network packets animate only from real client callback sequence identifiers while a job is running.

Lazy page modules keep chart-heavy screens separate. Central CSS variables and Tailwind v4 `@theme inline` define colors/fonts. Motion Primitives AnimatedGroup is vendored with its license and used through a reduced-motion-aware wrapper. Route/dialog/navigation states use short reusable transitions; reduced-motion CSS and MotionConfig also disable motion. See DESIGN_SOURCES.md for source access and fallback provenance.
