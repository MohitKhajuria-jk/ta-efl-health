# Verification results

Evidence recorded on 30 October 2026. Every number below was produced by running the project's
code or tests; nothing is estimated.

## Test suites

| Suite | Environment | Result |
|---|---|---|
| Python, full suite including PyQt6 GUI tests | Windows, project `.venv` (Python 3.12), `QT_QPA_PLATFORM=offscreen` | **202 passed** |
| Python, without the 3 GUI test files | Linux, Python 3.12.15, exact versions from `requirements-lock.txt` | 183 passed (147 existing + 36 personalisation) |
| Browser (Playwright, Chromium) | Linux, production build served by `scripts/serve_web_qa.py` | **16 passed** (14 existing + 2 personalisation) |
| TypeScript check and production build | `npm run build` | passed |

Browser-test note: historical experiment records store model paths with Windows separators
(`models\...`). On Linux, `api/research.py` cannot resolve them. For the Linux browser run only, a
temporary copy normalised the separator; the shipped `api/research.py` is unchanged. Windows is
unaffected.

## Reproduction of saved research results

* All 75 rows of the strategy table and all 25 rows of the zero-weight table in
  [RESULTS.md](RESULTS.md) were recomputed independently from `storage/experiments`. Every value
  matched.
* All 577 current-pipeline model artifacts were re-scored on regenerated test splits. Each
  reproduced its saved test ROC-AUC exactly (maximum difference 0). Every dataset SHA-256 matched.
* The trust formulas (performance signal, instant score, moving-average trust, aggregation weights,
  exclusion) and the FedAvg weights were recomputed for all 17,310 saved rounds (7,380 trust-aware),
  with 0 mismatches.
* Determinism: benchmark `20261014-050727` and `20261014-052600` contain the same 225 configurations
  and give identical results. A heart run repeated on 14, 15 and 29 October gave identical AUC.
  Twenty fresh reruns on Linux gave identical test ROC-AUC and F1; trust values differed by at most
  about 1e-15.
* One legacy pipeline-v1 record (`heart-trust-20261014-040145-cb11`) remains and is labelled
  non-current by the application.

## Leakage checks

* Preprocessing is fitted only on client training rows; no saved preprocessor used more rows than
  the training split.
* No exact duplicate rows occur between training and test splits for any dataset or seed (1, 2, 3).
* Federated research-code changes since the baseline commit are observational only (event
  callbacks, cancellation, a corrected comment).

## Personalisation

* 135 calibration runs (5 datasets × 3 strategies × 3 seeds × 3 partition settings) completed. Each
  run confirmed that its global model reproduced the saved test ROC-AUC and was byte-identical before
  and after calibration.
* The three reports produced on Linux and later on Windows are identical apart from timestamps and
  line endings.
* Preservation: before the extension, SHA-256 fingerprints were taken of 1,868 data, storage,
  configuration and source files and 40 web source files. Afterwards, 1,865 and 37 were identical.
  The six that differed were the intended edits: `README.md`, `api/main.py`,
  `scripts/serve_web_qa.py`, `web/src/app.tsx`, `web/src/types.ts` and
  `web/src/pages/prediction.tsx`. All 1,746 experiment, model and benchmark files were unchanged.

Method and results: [PERSONALIZATION.md](PERSONALIZATION.md).

## Known limitations

See the README and [RESULTS.md](RESULTS.md): single-process simulation, IID training partitions in
the saved benchmark, one attacker out of five, three seeds and no significance tests. FedMedian
equalled or exceeded the trust-aware strategy under sign-flip and Gaussian-noise attacks.
