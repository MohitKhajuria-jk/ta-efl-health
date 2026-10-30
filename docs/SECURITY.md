# Local workstation security and dependency review

Reviewed on 29 October 2026. This application is a shared local research workstation, with the original Python engine and account store. It has not been certified for clinical use or public multi-tenant hosting.

## HTTP controls

The web layer reuses the existing PBKDF2 password verifier. It adds opaque expiring HttpOnly/SameSite=Strict sessions, session rotation, CSRF checks, same-origin writes, restricted Host headers, request-body limits, rate limits, identifier validation, and resolved-path containment for artifacts. It does not accept model uploads or arbitrary filesystem paths. Reports use generated temporary filenames and are removed after download. Patient values and passwords are excluded from request/error logging; predictions are not persisted.

API tests exercise unauthorized requests, closed registration, session expiry/logout, invalid CSRF/origin, rate limiting, oversized bodies, redacted validation errors, invalid training parameters and artifact-path escape. The browser tests also verify protected navigation after logout. These tests cover the implemented boundaries; they are not a penetration-test certification.

Use one server process on loopback, retain operating-system access controls on the project and backups, and load only trusted local joblib artifacts. The old desktop password-reset workflow remains a local administrative operation and is not exposed over HTTP. All local accounts share access to the same research storage.

## Dependency review

`npm audit --json` returned **0 known vulnerabilities** across the resolved JavaScript dependency tree. Python's `pip check` reported **No broken requirements found**.

The pinned Python environment was checked against the [OSV batch API](https://google.github.io/osv.dev/post-v1-querybatch/). The initial scan flagged certifi 2026.7.22, filelock 3.32.7, Pillow 12.3.0 and cryptography 44.0.3. Official PyPI metadata and the reported advisory fixed versions were inspected. The compatible remediation applied to the web environment is:

| Package | Installed version after remediation | Reason |
|---|---|---|
| certifi | 2026.10.23 | Updated trusted root bundle |
| filelock | 4.2.4 | Updated local filesystem-lock protections |
| Pillow | 13.0.0 | Updated image/report dependency with published parser/encoder fixes |

These lower bounds are included in `requirements-web.txt`; exact versions are captured in `requirements-lock.txt`. The original research algorithms and serialized model artifacts are unchanged. The full Python and browser suites were rerun after these updates; see [verification results](VERIFICATION.md).

### Remaining cryptography advisories

The original `flwr==1.26.1` explicitly requires `cryptography>=44.0.1,<45.0.0`. Version 44.0.3 therefore remains installed. The audit reports these five GHSA entries, with corresponding PYSEC aliases:

| Advisory | Reported issue | Fixed version |
|---|---|---|
| [GHSA-537c-gmf6-5ccf](https://github.com/pyca/cryptography/security/advisories/GHSA-537c-gmf6-5ccf) | Vulnerable bundled OpenSSL | 48.0.1 |
| [GHSA-g6cj-pr64-35w5](https://github.com/pyca/cryptography/security/advisories/GHSA-g6cj-pr64-35w5) | PKCS#7 decryption oracle | 50.0.0 |
| [GHSA-jwv3-5hgf-82ww](https://github.com/pyca/cryptography/security/advisories/GHSA-jwv3-5hgf-82ww) | Certificate path-building resource exhaustion | 49.0.0 |
| [GHSA-m959-cc7f-wv43](https://github.com/pyca/cryptography/security/advisories/GHSA-m959-cc7f-wv43) | DNS name-constraint enforcement | 46.0.6 |
| [GHSA-r6ph-v2qm-q3c2](https://github.com/pyca/cryptography/security/advisories/GHSA-r6ph-v2qm-q3c2) | SECT curve subgroup validation | 46.0.5 |

The implemented loopback API does not expose certificate verification, PKCS#7 decryption or remote Flower clients; local password verification uses Python's `hashlib`, and client communication is in-process. This reduces relevant exposure but does **not** remove the installed dependency findings. Do not treat the Python environment as vulnerability-free.

Resolving this constraint needs a separately validated Flower upgrade and its associated dependency changes. The current Flower release also changes FastAPI, Starlette, Uvicorn and other requirements, so no incompatible cryptography override or broad framework upgrade was silently applied during presentation-layer replacement. Complete that compatibility work before considering network-facing deployment.

## Repeat the checks

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/audit_dependencies.py --output test-results/python-osv-audit.json
cd web
npm.cmd audit
```

The OSV script reads the checked-in exact lockfile and needs network access. Exit 0 means no findings, 1 means findings, and 2 means the audit did not complete. An unavailable advisory service is never reported as a clean scan. Advisory databases change; results reflect the date of the check. OSV records may contain multiple identifiers for the same issue.
