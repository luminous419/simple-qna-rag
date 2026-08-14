# Code Review Iteration 7 Remediation — CR-I7-MAJ-01

Fixes the sole finding in `Code_Review_Iteration_7.md`: ordinary Label
binding in `scripts/scan_image_layers.py`'s `_label_bound_to_subject()` used
`label_value.strip().casefold()` against case-folded Subject RDN candidates,
so a case-only or leading/trailing-whitespace mutation of a genuine Label
still bound to the certificate — the seventh field was the only one of the
seven that wasn't exact.

## Change

- `_label_bound_to_subject()` now compares the raw Label value to the exact
  (`_unicode_escape`-rendered, unmodified) Subject RDN values — no
  `.strip()`, no `.casefold()`.
- `_CERTIFICATE_LABEL_COMPATIBILITY` gains two entries, alongside the
  unchanged legacy Entrust.net exception, each an exact
  `(certificate SHA-256, exact label string)` pair rather than a
  case/whitespace equivalence class:
  - `657cfe2f...eda3305` → `"certSIGN Root CA G2"` (Label differs from
    Subject CN `certSIGN ROOT CA G2` only by case)
  - `9ae36232...089a651e6` → `" OISTE Server Root RSA G1"` (Label carries a
    genuine leading space; Subject CN has none)
  These were reported as the exact two entries that differ from every
  Subject RDN value in the locally installed and pip-vendored bundles once
  matching became byte-exact. **Correction (Code_Review_Iteration_8.md
  CR-I8-MAJ-01):** this was only accurate for the top-level `certifi`
  package, whose version `requirements.lock` pins (`certifi==2026.7.22` at
  the time of writing). The claim that the pip-vendored bundle was also
  `certifi==2026.7.22` was wrong: `pip`'s vendored `certifi` copy
  (`pip/_vendor/certifi/cacert.pem`) is baked into the installed `pip`
  package itself, is never touched by installing or upgrading the
  top-level package, and its actual version was never independently
  checked at the time — it tracks whatever `pip` release happens to be on
  `PATH` and legitimately differs between interpreters (this project's
  `venv` vs. this machine's repository-default Python). See
  Code_Review_Iteration_8_Remediation.md for the documented three-boundary
  compatibility policy and the five additional exception-table entries
  that boundary required.
- Module and function docstrings updated to describe the exact-binding
  policy and the fixed exception table.

## Tests added (`tests/unit/test_scan_image_layers.py`)

- `test_is_verified_ca_bundle_rejects_case_or_whitespace_label_variant`
  (parametrized): the three exact mutations the review demonstrated
  (`"entrust Root Certification Authority"`,
  `" Entrust Root Certification Authority"`,
  `"Entrust Root Certification Authority "`) against the otherwise genuine
  Entrust Root fixture now fail.
- `test_is_verified_ca_bundle_accepts_full_pip_vendored_certifi_bundle`: the
  full real `pip/_vendor/certifi/cacert.pem` bundle still verifies end to
  end under exact-equality binding.
- Pre-existing `test_is_verified_ca_bundle_accepts_full_installed_certifi_bundle`
  (installed top-level certifi) continues to pass unchanged, confirming the
  positive path for both real bundles.

## Verification

- `pytest -q tests/unit/test_scan_image_layers.py tests/unit/test_rag_engine_singleton.py`
  — 97 passed (93 baseline + 4 new).
- `git diff --check` — clean.
- No edits to Requirement/Plan/Traceability/Design/workflow/baseline-checker
  files; no commit/push/merge; no live/protected/self-hosted gate invoked.
