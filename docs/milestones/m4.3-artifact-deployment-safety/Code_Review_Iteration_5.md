# M4.3 Artifact & Deployment Safety — Code Review Iteration 5

Reviewer: Fresh Codex independent code review worker

Reviewed revision: `30419791a4bf984984ee191190e6ee8b2225b3f0`

Exact range: `1e7fbac4ecda5217a2a315cb5e54621708624edb..30419791a4bf984984ee191190e6ee8b2225b3f0`

Verdict: **FAIL — 9.0/10** (`CRITICAL 0`, `MAJOR 2`, `MINOR 0`, `TRIVIAL 0`)

## Scope and conclusion

I fully read `milestone_dev_orchestration_guide.md`,
[Code Review Iteration 3](Code_Review_Iteration_3.md),
[Code Review Iteration 4](Code_Review_Iteration_4.md),
[Hosted CI Remediation Iteration 3](Hosted_CI_Remediation_Iteration_3.md),
[Implementation Report](Implementation_Report.md), and
[Traceability](Traceability.md), then independently inspected the exact
13-file diff and its tests. The FAISS construction change preserves the
trust-before-pickle ordering and now retains the real `Embeddings` object; the
deterministic sentinel and `Document` fixture match production types; the
production image still excludes `tests/`; the positive smoke mounts the seam
read-only while the negative smoke has no seam mount; and the new subprocess
test exercises the real bare-script invocation.

The remediation does not pass the code-quality gate. The purported certifi
“seven-field” grammar actually accepts any number and order of the seven
prefixes with arbitrary values, which recreates arbitrary-text smuggling.
Separately, the failed engine remains cached by the class singleton and its
artifact reason is never cleared; a later ordinary initialization failure can
therefore be misreported as the earlier artifact failure. Both defects are in
the exact changed code and lack regression oracles.

The diff does not change workflow approval or release-state logic.
`M4.1_BLOCKED=true`, protected M3 live `NOT_RUN` (the hosted job remains
skipped), and `overall_release_ready=false` remain mandatory. No Native
Linux/Ollama/DDGS, protected live, self-hosted runner, or environment approval
action was run or altered.

## Findings

### CR-I5-MAJ-01 — certifi comments can smuggle arbitrary text

**Severity:** MAJOR (pre-merge security scanner correctness).

**Evidence:** `scripts/scan_image_layers.py::_CERTIFI_COMMENT_FIELD` and
`_CERT_BLOCK` (lines 135–144) use `[^\r\n]{0,512}` unrestricted
values and `(?:_COMMENT_LINE)*`. The implementation therefore enforces neither
exactly seven fields nor their documented order, and it applies no field-level
grammar. Direct probes against the reviewed revision all returned `True` after
appending the same genuine X.509 certificate:

```text
# Issuer: API_TOKEN=supersecret
# Label: ../../etc/shadow
# Serial: arbitrary free text
# Issuer: x
# Issuer: y
```

`SSLContext.load_verify_locations()` ignores those comment
bytes, so successful X.509 parsing does not repair the grammar gap. The four new
tests at `tests/unit/test_scan_image_layers.py:646–674` cover a real sample,
an unknown prefix, a wholly unknown comment, and a trailing recognized comment,
but none asserts exact field count/order/uniqueness or adversarial values under
a recognized prefix. Thus the test oracles are not exact for the stated
seven-field contract.

**Required fix:** encode one exact ordered seven-line stanza per certificate,
with field-specific bounded syntax (at minimum strict hexadecimal fingerprint
and serial grammars and a justified non-secret-bearing Issuer/Subject/Label
alphabet), or strip and independently validate an exact upstream stanza before
passing only the certificate blocks to OpenSSL. Add negative tests for missing,
duplicate, reordered, and extra fields and for token/path/key-value payloads in
every recognized field; retain a full real certifi bundle positive oracle and
full-input consumption checks.

### CR-I5-MAJ-02 — failed engine state and stale artifact reason survive retry

**Severity:** MAJOR (pre-merge readiness and negative-control integrity).

**Evidence:** `src/simple_qna_rag/rag_engine.py:RAGEngine.__new__` (lines
196–205) stores every constructed engine in `RAGEngine._instance`, while
`get_rag_engine()` (lines 818–833) now avoids assigning only the second,
module-level `_rag_engine` variable on failure. A retry therefore receives the
same failed object, not a fresh candidate. `RAGEngine.initialize()` records an
artifact reason at lines 265–270 but never resets `_artifact_error_reason` at
the start of a later attempt; its ordinary `except Exception` branch at lines
271–275 also leaves the old value intact. Consequently this sequence is
possible: artifact failure → retry → unrelated ordinary failure →
`EngineArtifactError(old_reason)`, altering the required ordinary
`engine_init_failed` result and potentially making an artifact negative control
pass for the wrong attempt.

The new exception plumbing also accepts arbitrary `.reason` text:
`EngineArtifactError.__init__` stores any string (lines 97–107), and
`server.py::_make_lifespan` copies `getattr(exc, "reason", None)` from any
factory exception (lines 358–368). The default production path currently
originates its reason from bounded trust exceptions, so no direct path leak was
reproduced there, but the boundary itself does not enforce that invariant and
an ordinary exception carrying `.reason` is reclassified. No exact-diff test
covers failed-singleton identity, retry, stale reason, reason allowlisting, or
an ordinary exception with a `reason` attribute.

**Required fix:** ensure a failed construction clears both singleton layers (or
remove the class-level singleton), reset `_artifact_error_reason` before each
initialization attempt, and propagate only an explicit allowlist of public
artifact reason codes from a dedicated exception type. In the lifespan, catch
that type specifically rather than accepting `.reason` from arbitrary
exceptions. Add exact tests for fresh identity on retry, artifact→ordinary
failure, artifact→success, arbitrary/path-like reason rejection, and unchanged
ordinary `engine_init_failed` behavior.

## Other reviewed properties

- `_update_merged_state()` applies ordered layers, exact whiteouts and opaque
  directory masking before same-layer writes. Link resolution is bounded to 40
  hops, cycle-checked, rejects root traversal/dangling/non-regular/untrusted
  targets, reads the selected historical member bytes, and preserves additive
  violation history. Existing tests cover the principal Debian two-hop and
  cross-layer positives plus dangling, cycle, whiteout, traversal, duplicate
  history, and untrusted-target negatives. No separate OCI resolver bypass was
  established in this review.
- `_construct_faiss_from_verified_bytes()` still receives only bytes returned
  by `verify_version()`; changing `FAISS(embeddings.embed_query, ...)` to
  `FAISS(embeddings, ...)` does not move pickle deserialization before trust
  verification. The focused suite confirms construction succeeds.
- `DeterministicTestEmbeddings` now inherits the LangChain `Embeddings` ABC and
  remains under `tests/support`; `deploy/Dockerfile` copies `tests/` only into
  the test stage, not production. The positive smoke explicitly mounts that
  directory read-only, while `build_negative_activation_argv()` has no mount or
  `PYTHONPATH` seam.
- The fixture now stores `Document(page_content=...)`, matching the production
  docstore shape, and imports one deterministic identity sentinel rather than
  duplicating it.
- The bare-script tests launch `sys.executable scripts/container_smoke.py` in a
  fresh subprocess and reached the documented Docker failure/skip contract;
  they do not rely on pytest's import state.
- Hosted/local status is evidence, not a substitute for review. The remediation
  report's local linux/amd64 image scan (`forbidden_count: 0`) and smoke
  (`status: PASS`) are plausible and the hosted scan step independently
  succeeded at the exact SHA, but the local report records no retained image
  digest or raw receipt from which to independently replay that particular
  run.

## Verification evidence

| Check | Result |
|---|---|
| `pytest -q tests/unit/test_scan_image_layers.py tests/unit/test_container_smoke_bare_script.py tests/unit/test_index_verification.py tests/integration/test_health_endpoints.py` | **80 passed**, 3 third-party SWIG deprecation warnings |
| Adversarial recognized-prefix certifi probes | **Bypass reproduced**: all four arbitrary-text/duplicate-field inputs returned `True` |
| `python scripts/check_markdown_links.py` | **PASS**, 120 files / 553 links / 0 failures |
| `git diff --check 1e7fbac4..30419791` | **PASS** |
| Hosted run 31615907683 at review time | exact SHA confirmed; OCI scan **SUCCESS**, frontend **SUCCESS**, container smoke/python/deterministic still in progress, protected M3 live **SKIPPED** |

## Gate

Severity count is `CRITICAL 0 / MAJOR 2 / MINOR 0 / TRIVIAL 0`; score is
**9.0/10**. The required gate is `CRITICAL=0`, `MAJOR=0`, and score >=9.7,
so Code Review Iteration 5 is **FAIL**. Iteration 5 is within the guide's
conditional extension bounds (`CRITICAL=0`, score >=9.0, at most two MAJORs,
concrete fixes), but both findings must be remediated and independently
re-reviewed; hosted success cannot override either code-level defect.

## Closure note (remediation applied, same session, PR #18)

Both findings were remediated in
[Code_Review_Iteration_5_Remediation.md](Code_Review_Iteration_5_Remediation.md)
(same branch, new commit, no merge). Summary, pending a fresh independent
Codex review:

- **CR-I5-MAJ-01**: `is_verified_ca_bundle()`'s certifi comment grammar
  was rewritten from a repeatable `(?:_COMMENT_LINE)*` with free-text
  `[^\r\n]{0,512}` values to a single fixed, non-repeating sequence that
  names each of the seven fields literally exactly once, in the exact
  upstream order, each with its own bounded grammar (decimal-only
  Serial; byte-exact colon-hex fingerprints; a real-DN-derived
  Issuer/Subject/Label alphabet). Missing/duplicate/reordered/extra
  fields and this finding's exact adversarial probe now all return
  `False`; a full real installed `certifi` bundle and a real
  pip-vendored entry (both extracted from an actual built
  `linux/amd64` image) still verify `True`.
- **CR-I5-MAJ-02**: `get_rag_engine()` now discards both the class-level
  `RAGEngine._instance` singleton and `_initialized` flag on any failed
  construction (so a retry always builds a genuinely fresh object),
  `initialize()` resets `_artifact_error_reason` at the start of every
  attempt, `EngineArtifactError` rejects any reason outside
  `index_verification.REASONS` at construction, and `server.py`'s
  lifespan classifies that dedicated exception type rather than
  duck-typing `.reason` off `Exception`. New regression tests were first
  confirmed to fail against the pre-fix code (via `git stash`) before
  the fix was restored.

Re-verification: full local pytest suite 1251 passed, 1 skipped
(1220→1251); real `linux/amd64` `production` image scan
`forbidden_count: 0` (one `1` was actually observed and fixed mid-remediation
— see the remediation doc §1.1); the same image's `container_smoke.py`
`status: PASS` with `production_test_seam_sealed: true`. No OCI resolver
behavior, `M4.1_BLOCKED`, protected M3 live, or `overall_release_ready`
logic was touched.
