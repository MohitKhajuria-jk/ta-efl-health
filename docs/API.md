# API contract

All resource endpoints require the local session cookie except health, session status and login/initial registration. JSON mutation requests require `X-CSRF-Token` returned by session/login. The complete generated OpenAPI contract is available to signed-in users at `GET /api/schema`.

| Method | Route | Contract |
|---|---|---|
| GET | `/api/health` | Process health and execution mode; no private data |
| GET | `/api/auth/session` | Authenticated flag, username, registration availability, CSRF token |
| POST | `/api/auth/login` | `{username,password}` → rotated session |
| POST | `/api/auth/register` | First account, or authenticated additional local account |
| POST | `/api/auth/logout` | Invalidate session and clear cookie |
| GET | `/api/dashboard` | Actual experiment/model-reference counts, datasets, latest/current activity |
| GET | `/api/datasets` | Validated metadata, schema, provenance and missing-value counts; no raw rows |
| GET | `/api/experiments` | Search/filter/sort/paginate summaries |
| GET | `/api/experiments/{id}` | Original record, round trust records and evaluation history |
| POST | `/api/experiments/compare` | `{ids:[...]}` → compatible saved test evaluations; 409 if protocols differ |
| GET | `/api/models?dataset=...` | Current-pipeline saved model references |
| POST | `/api/predictions` | `{experiment_id, values, explain:true}` → probability/class/threshold and actual SHAP |
| GET | `/api/federation` | Most recent 50 job summaries; large histories omitted for efficient polling |
| POST | `/api/federation` | Validated TrainingRequest → 202 JobView; 409 if busy |
| GET | `/api/federation/{id}` | Authoritative job state |
| POST | `/api/federation/{id}/cancel` | Cooperative cancellation request; terminal jobs unchanged |
| GET | `/api/reports/{id}?format=pdf` | Download generated PDF, HTML or JSON report |
| GET | `/api/settings` | Validated web defaults and redacted system information |
| PUT | `/api/settings` | `{defaults: TrainingRequest}` → saved web-only defaults |

Experiment list queries: `q`, `dataset`, `strategy`, `attack`, `status`, `sort` (`newest`, `oldest`, `auc`), `page` (1-based), `page_size` (1–100). Response: `{items,total,page,page_size}`. Metrics are copied from actual records, never accepted from the caller. Superseded records remain searchable but are not used for predictions/comparisons.

`TrainingRequest` supports the engine's dataset, strategy, client/round/local-epoch counts, learning rate, batch size, L2, class weighting, partition, Dirichlet alpha, participation, seed, fixed decision threshold, trust parameters, malicious-client numbers (1-based), supported attacks, scale/noise and label. Extra fields are forbidden. The API further bounds finite values and validates against `ExperimentConfig`. Supported strategies/attacks are returned by settings.

`PredictionResponse` includes the associated experiment/dataset, positive-class probability/meaning, predicted label, saved threshold, research caveats and optional actual explanation. An explanation failure does not manufacture SHAP or discard a successful prediction; it returns `explanation_error`. Patient input values are never included in transport error validation details.

`JobView` includes id/status/stage, UTC timestamps, configuration, current round, completed rounds, cancellation flag, experiment id, error, per-client activity, aggregation rounds and a bounded event stream. `running` is a job status; initializing/local_training/aggregating/evaluating are phases. `interrupted` is explicitly distinct from failure/cancellation/completion.

Error codes: 400 invalid resource id, 401 not signed in, 403 origin/CSRF denial, 404 missing resource, 409 conflicting/incompatible state, 413 oversized body, 422 invalid schema/input, 429 rate limit, 503 unavailable account store. No arbitrary file browsing, model uploads, remote training hosts or unprotected diagnostic-log endpoints exist.
