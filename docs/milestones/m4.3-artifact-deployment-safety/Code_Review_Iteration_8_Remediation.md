# Code Review Iteration 8 Remediation — CR-I8-MAJ-01

Fixes the sole finding in `Code_Review_Iteration_8.md`: the Iteration 7
remediation's three-entry `_CERTIFICATE_LABEL_COMPATIBILITY` table was tuned
against one interpreter's pip-vendored `certifi` copy and rejected the
review's own environment's genuine pip-vendored bundle (`certifi==2023.07.22`,
vendored by that environment's `pip==23.3.1`), so
`test_is_verified_ca_bundle_accepts_full_pip_vendored_certifi_bundle` failed
there even though `_label_bound_to_subject`'s byte-exact matching itself was
correct.

## Root cause

`pip`'s vendored `certifi` copy (`pip/_vendor/certifi/cacert.pem`) is source
baked into the installed `pip` package itself. It is a separate, independent
copy from the top-level `certifi` package `requirements.lock` pins, is never
touched by installing or upgrading the top-level package, and instead tracks
whatever `pip` release happens to be on `PATH` — so it legitimately differs
between interpreters. This project has (at least) two real, regularly-used
interpreters on this machine:

- **This project's `venv`** (`venv/bin/python -m pytest`): top-level
  `certifi` pinned by `requirements.lock` (`certifi==2026.7.22` at the time of
  writing); `pip`'s vendored copy tracks whatever `pip` was installed into
  that `venv`.
- **This machine's repository-default Python** (`python3`/`pytest` resolved
  from `PATH` without activating `venv`): the interpreter
  Code_Review_Iteration_8.md's fresh review actually ran under, whose
  `pip==23.3.1` vendors `certifi==2023.07.22`.

The Iteration 7 remediation report incorrectly claimed both the locally
installed and pip-vendored bundles were `certifi==2026.7.22` — that version
number only ever applied to the top-level package; the pip-vendored copy's
actual version was never independently checked. `Code_Review_Iteration_7_
Remediation.md` has been corrected in place to note this.

## Change

### Defined compatibility boundary

The module docstring in `scripts/scan_image_layers.py` now documents the
supported compatibility boundary explicitly as the real bundles exercised
across both interpreters above (see the docstring's three-bundle write-up):
top-level `certifi` per `requirements.lock` in `venv`, `pip`'s vendored copy
in `venv`, and `pip`'s vendored copy under this machine's repository-default
Python — the exact interpreter the Iteration 8 review used.

### Exception table

`_CERTIFICATE_LABEL_COMPATIBILITY` gains five entries, verified by
independently walking the real, unmodified `certifi==2023.07.22` distribution
(downloaded directly from PyPI, not hand-built) with the same
`_CERT_BLOCK_GROUPED_RE` / `_decode_certificate` / `_unicode_escape` logic the
scanner itself uses — reproducing every one of the review's five reported
deviations and confirming no further ones exist in that bundle:

| Certificate SHA-256 | Label |
|---|---|
| `d7a7a0fb...699ef4` | `Comodo AAA Services root` |
| `e75e72ed...026b6c` | `Security Communication Root CA` |
| `cecddc90...86fc1aa2` | `XRamp Global CA Root` |
| `c3846bf2...81fe546ae4` | `Go Daddy Class 2 CA` |
| `1465fa20...fb5fb658` | `Starfield Class 2 CA` |

None of these five certificates (by SHA-256) is present in either bundle the
`venv` interpreter's full-bundle tests exercise — Mozilla has since removed
all five from the curated Mozilla root list — so the two existing full-bundle
tests (installed top-level, `venv` pip-vendored) are unaffected by the
addition; they continue to find only the pre-existing three exceptions
(Entrust.net, certSIGN, OISTE).

The existing full-bundle test
`test_is_verified_ca_bundle_accepts_full_pip_vendored_certifi_bundle` already
exercises whichever `certifi` bundle the interpreter running pytest has
vendored, so no new full-bundle test was needed to cover the
repository-default interpreter's `certifi==2023.07.22` bundle — running the
existing suite under that interpreter (see Verification below) is sufficient
end-to-end coverage.

### Tests added (`tests/unit/test_scan_image_layers.py`)

Rather than embedding all five real certificates' full PEM bytes (a large,
low-signal diff for data the existing full pip-vendored bundle test already
covers end-to-end when run under the repository-default interpreter), the new
tests call `_label_bound_to_subject` directly:

- `test_label_bound_to_subject_accepts_certifi_2023_legacy_exception`
  (parametrized over the five entries): each exact (SHA-256, Label) pair
  binds against a genuine decoded Subject RDN value taken from the real
  certificate (not a hand-built guess) that differs from the Label.
- `test_label_bound_to_subject_rejects_mutated_legacy_label`: a one-character
  mutation of the same real Label, against the same real SHA-256 and Subject
  value, fails — the exception is bound to one exact Label string, not any
  Label for that certificate.
- `test_label_bound_to_subject_rejects_legacy_label_with_wrong_sha256`: the
  same real Label against a wrong SHA-256 fails — the exception is bound to
  one exact certificate, not any certificate carrying this Label text.

No `.strip()`, `.casefold()`, or other loosening was introduced anywhere in
this change; `_label_bound_to_subject`'s ordinary-Label comparison remains
byte-exact, and the three adversarial case/leading/trailing-whitespace tests
from Iteration 7 are unchanged and still pass.

## Verification

- `venv/bin/python -m pytest -q tests/unit/test_scan_image_layers.py
  tests/unit/test_rag_engine_singleton.py` — **112 passed** (venv interpreter;
  top-level `certifi==2026.7.22` per `requirements.lock`, pip-vendored
  `certifi==2025.10.5`).
- `python3 -m pytest -q tests/unit/test_scan_image_layers.py
  tests/unit/test_rag_engine_singleton.py` (repository-default interpreter,
  `venv` not activated) — **112 passed**, including
  `test_is_verified_ca_bundle_accepts_full_pip_vendored_certifi_bundle`
  against the genuine `certifi==2023.07.22` bundle the Iteration 8 review
  used — the finding no longer reproduces.
- `git diff --check` — clean.
- No edits to Requirement/Plan/Traceability/Design/workflow/baseline-checker
  files; no commit/push/merge; no Native Linux/Ollama/DDGS/live/protected/
  self-hosted/image gate invoked.

## Note on venv drift

Before this remediation, `venv`'s top-level `certifi` was installed at
`2025.10.5`, not the `2026.7.22` `requirements.lock` pins — the two had
drifted out of sync. `certifi==2026.7.22` was installed via
`pip install --no-deps certifi==2026.7.22` to bring `venv` back in line with
the lock file before verification; this is the only environment change made.
