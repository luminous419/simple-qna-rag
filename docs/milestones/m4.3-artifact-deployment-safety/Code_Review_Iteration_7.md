# M4.3 Artifact & Deployment Safety — Code Review Iteration 7

Reviewer: Fresh Codex independent code review worker

Baseline: `84f6b407c9dd6d2de262c4d8f708618d11b37766`

Reviewed state: current uncommitted M4.3 remediation relative to that baseline

Verdict: **FAIL — 9.5/10** (`CRITICAL 0`, `MAJOR 1`, `MINOR 0`, `TRIVIAL 0`)

## Scope and conclusion

I independently reviewed the current uncommitted changes to
`scripts/scan_image_layers.py`, `tests/unit/test_scan_image_layers.py`, and
`tests/unit/test_rag_engine_singleton.py`, plus Code Review Iteration 6 and the
M4.3 Requirement, Plan, Traceability, and Design boundaries. I did not edit
implementation, commit, push, merge, or invoke any live, protected,
self-hosted, Docker-image, Ollama, or DDGS gate.

CR-I6-MIN-01 is closed: the singleton tests retain the actual failed and
successful objects and compare them with `is`/`is not`, so CPython integer-ID
reuse can no longer invalidate the identity oracle. Six certificate-derived
fields are exact, the sole legacy label exception is bound to one exact SHA-256
and one exact label, and temporary-file cleanup failures cannot produce an
allowlist success. CR-I6-MAJ-01 is nevertheless not fully closed because the
ordinary Label path deliberately normalizes case and surrounding whitespace;
therefore the seventh field is not exact and multiple non-certificate metadata
values are accepted for the same certificate.

## Finding

### CR-I7-MAJ-01 — ordinary Label binding is not exact

**Severity:** MAJOR (security-scanner allowlist correctness; required exact
seven-field certificate binding is incomplete).

`_label_bound_to_subject()` constructs case-folded Subject RDN candidates and
compares them to `label_value.strip().casefold()`. For the genuine Entrust Root
fixture, independently changing only the Label while leaving the exact
certificate and the other six fields untouched produced:

```text
"entrust Root Certification Authority"  -> True
" Entrust Root Certification Authority" -> True
"Entrust Root Certification Authority " -> True
```

The same mutation method returned `False` for Issuer, Subject, Serial, MD5,
SHA1, and SHA256, but returned `True` for a case-only Label mutation. Thus the
certificate does not determine the exact accepted Label bytes. This is a
bounded equivalence class rather than arbitrary free text, but it contradicts
the requested exact binding for all seven fields and permits metadata that is
not the stanza for that certificate.

The hash-bound compatibility branch is appropriately narrow: the exact
Entrust.net legacy label with certificate SHA-256
`6dc47172e01cbcb0bf62580d895fe2b8ac9ad4f873801e0c10b9c837d21eb177`
passes, a one-character label change fails, and a case variant fails. The
finding concerns the generic Subject-RDN branch, not that exception.

**Required fix:** bind ordinary labels through an exact certificate-specific
mapping or another deterministic exact rendering policy. If upstream certifi
contains genuine case/leading-space variants, encode only the necessary
exceptions as exact `(certificate SHA-256, label)` pairs, as already done for
the Entrust.net legacy entry. Add regression tests proving case-only and
leading/trailing-space variants fail for an otherwise genuine stanza.

## Other reviewed properties

- The stanza grammar still requires exactly seven ordered fields. Direct
  one-field mutation probes rejected Issuer, Subject, Serial, MD5, SHA1, and
  SHA256; the Label result above is the sole observed binding defect.
- The legacy exception cannot generalize by label alone: both the exact digest
  lookup and exact compatible-label equality must succeed.
- `_decode_certificate()` owns the `mkstemp` descriptor correctly across
  `fdopen`, closes it on transfer failure, and attempts unlink in `finally`.
  An injected `PermissionError` from `os.unlink` propagated rather than
  returning `True`; cleanup failure is fail-closed.
- The singleton tests now keep strong references to every observed candidate,
  assert failed-versus-retry identity directly, and assert the returned engine
  is the successful initialized object.
- Requirement, Plan, Traceability, Design, workflow, and baseline checker have
  zero diff from `84f6b407`. Consequently `M4.1_BLOCKED=true`, protected M3
  live `NOT_RUN`/`SKIPPED`, `operational_status=BLOCKED`, and
  `overall_release_ready=false` remain untouched.

## Verification evidence

| Check | Result |
|---|---|
| `pytest -q tests/unit/test_scan_image_layers.py tests/unit/test_rag_engine_singleton.py` | **93 passed** |
| Same command plus `tests/integration/test_health_endpoints.py` | collection blocked by host `email-validator` version mismatch; environment evidence only |
| Independent one-field certificate stanza mutations | six exact fields rejected; ordinary Label case mutation accepted |
| Legacy exact label / one-character change / case change | `True` / `False` / `False` |
| Injected temporary-file unlink failure | propagated `PermissionError`; no allowlist success |
| `python scripts/check_markdown_links.py` before this report | **PASS**, 124 files / 566 links / 0 failures |
| `git diff --check 84f6b407` before this report | **PASS** |
| Protected/live/self-hosted gates | **NOT RUN**, by scope |

## Gate

Severity count is `CRITICAL 0 / MAJOR 1 / MINOR 0 / TRIVIAL 0`; score is
**9.5/10**. The required PASS gate is `CRITICAL=0`, `MAJOR=0`, and score
`>=9.7`, so Code Review Iteration 7 is **FAIL**. No merge is allowed on this
result.
