# M4.3 Artifact & Deployment Safety — Code Review Iteration 8

Reviewer: Fresh Codex independent code review worker

Baseline: `84f6b407c9dd6d2de262c4d8f708618d11b37766`

Reviewed state: current uncommitted remediation relative to that baseline

Verdict: **FAIL — 9.5/10** (`CRITICAL 0`, `MAJOR 1`, `MINOR 0`, `TRIVIAL 0`)

## Scope and conclusion

I independently reviewed the current uncommitted changes to
`scripts/scan_image_layers.py`, `tests/unit/test_scan_image_layers.py`, and
`tests/unit/test_rag_engine_singleton.py`, plus Code Review Iterations 6 and 7
and the Iteration 7 remediation report. I did not edit implementation, commit,
push, merge, or invoke any Native Linux, Ollama, DDGS, protected/live,
self-hosted, or image gate.

The generic ordinary-Label comparison now has the required byte-exact
semantics: it compares the unmodified Label against decoded Subject RDN values
without stripping, case-folding, or Unicode normalization. Genuine deviations
can pass only through an exact certificate-SHA-256 key to an exact Label value,
and the three requested case/leading/trailing-whitespace mutations fail.
Certificate metadata binding, fail-closed temporary cleanup, and the singleton
object-identity oracle also remain sound.

CR-I7-MAJ-01 is nevertheless not closed as a deliverable because its required
positive compatibility boundary fails in the current review environment. The
new full pip-vendored bundle test fails against the genuine installed pip
bundle, while the remediation report claims that bundle passes.

## Finding

### CR-I8-MAJ-01 — exact exception table rejects the genuine pip-vendored bundle

**Severity:** MAJOR (required compatibility oracle fails; CR-I7-MAJ-01 remains
open).

`_CERTIFICATE_LABEL_COMPATIBILITY` contains only three certificate/Label
pairs. Running the focused suite against the actual
`pip/_vendor/certifi/cacert.pem` installed in this environment
(`certifi==2023.07.22`) produces one test failure:

```text
test_is_verified_ca_bundle_accepts_full_pip_vendored_certifi_bundle FAILED
96 passed, 1 failed
```

Walking the otherwise grammar-valid bundle shows five genuine entries rejected
at Label binding before iteration stops: Comodo AAA Services root, Security
Communication Root CA, XRamp Global CA Root, Go Daddy Class 2 CA, and Starfield
Class 2 CA. Each curated Label differs from every exact decoded Subject RDN
value, and none of the five certificate SHA-256 values is present in the
compatibility table. For example, the first failure has SHA-256
`d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4`, Label
`Comodo AAA Services root`, and Subject values including `Comodo CA Limited`
and `AAA Certificate Services`.

The exact matching rule itself is appropriately non-extensible: ordinary
case/whitespace mutations fail and exceptions require both exact digest and
exact Label. The defect is incomplete bundle-version coverage. The remediation
report's statement that the locally installed pip-vendored bundle is
`certifi==2026.7.22` and passes is not true of the review environment, whose pip
vendor reports `2023.07.22`.

**Required fix:** define and document the supported pip-vendored certifi
compatibility boundary, then include every genuine deviation in that boundary
as an exact certificate-SHA-256 to exact-Label pair. Keep the full real
installed and pip-vendored bundle tests, and require both to pass in the locked
project environment; do not restore any case, whitespace, or normalization
equivalence.

## Other reviewed properties

- Ordinary Label candidates are exact `_unicode_escape` renderings of decoded
  Subject RDN values. No `.strip()`, `.casefold()`, or normalization is used.
- The three adversarial Label mutations from Iteration 7 all return `False`.
  The exact SHA-256/Label exception lookup does not admit a one-character or
  case mutation.
- Issuer, Subject, Serial, MD5, SHA1, and SHA256 remain derived from the exact
  following certificate. Grammar-valid token/path/key-value probes fail.
- `_decode_certificate()` transfers the `mkstemp` descriptor safely, closes it
  on transfer failure, and unlinks in `finally`. A cleanup exception propagates
  rather than producing an allowlist success, so cleanup remains fail-closed.
- Singleton tests retain strong references to candidates and use `is`/`is not`,
  eliminating recyclable integer-ID ambiguity while proving the returned
  engine is the successfully initialized candidate.
- Requirement, Plan, Traceability, Design, workflow, and baseline-state code
  have no diff from `84f6b407`. `M4.1_BLOCKED=true`, protected M3 live
  `NOT_RUN`/`SKIPPED`, `operational_status=BLOCKED`, and
  `overall_release_ready=false` remain protected.

## Verification evidence

| Check | Result |
|---|---|
| `pytest -q tests/unit/test_scan_image_layers.py tests/unit/test_rag_engine_singleton.py` | **FAIL:** 96 passed, 1 failed (genuine pip-vendored full bundle) |
| Installed top-level certifi full bundle (`2025.08.03`) | **PASS** |
| Iteration 7 case/leading/trailing-whitespace adversarial Labels | **PASS:** all rejected |
| Grammar-valid DN and Label smuggling regressions | **PASS:** rejected |
| Singleton fresh-object identity tests | **PASS** |
| `git diff --check 84f6b407` before this report | **PASS** |
| Protected/live/self-hosted/image gates | **NOT RUN**, by scope |

## Gate

Severity count is `CRITICAL 0 / MAJOR 1 / MINOR 0 / TRIVIAL 0`; score is
**9.5/10**. The required PASS gate is `CRITICAL=0`, `MAJOR=0`, and score
`>=9.7`, so Code Review Iteration 8 is **FAIL**. No merge is allowed on this
result.
