# M4.3 Hosted Container Remediation Iteration 5 — Fresh Codex Code Review

Reviewer: Codex (independent review)  
Baseline: `c51387a`  
Scope: `scripts/container_smoke.py`, `tests/unit/test_container_smoke_readiness_diagnostics.py`, and `Hosted_Container_Remediation_Iteration_5.md`  
Hosted failure under diagnosis: run `31800982514`

## Verdict

**PASS for one diagnostic commit/push and a definitive hosted rerun.** This is a narrowly bounded observability change, not a demonstrated root-cause remediation and not final M4.3 merge readiness.

- CRITICAL: 0
- MAJOR: 0
- MINOR: 2
- Score: **9.7 / 10.0**
- Diagnostic-push gate (`CRITICAL=0`, `MAJOR=0`, score `>=9.7`): **PASS**
- Final M4.3 merge/readiness gate: **NOT EVALUATED / NOT READY**

The implementation preserves the existing 60-second positive readiness budget, the exact negative seam expectation (`503` plus `artifact_test_embedding_seam_unavailable`), the five-field `compute_all_ok()` decision, and all protected CI/evidence boundaries. It is safe and useful to commit and push solely to obtain hosted evidence, provided the resulting run is interpreted using the branches in this review and is not described as proof that the failure has been fixed.

## Findings

### CR-HCR5-MIN-01 — `max_bytes` is a character bound, not a byte bound

`_capture_container_logs()` decodes subprocess output as text and applies `len(combined)` / `combined[-max_bytes:]`. The stored tail is therefore bounded to 16,000 Python characters, but its UTF-8 representation can exceed 16,000 bytes. The output remains materially bounded (and `docker logs --tail 200` additionally bounds the selected line count), so this is not a push blocker; however, the helper name, docstring, remediation report, and truncation test overstate the exact byte contract. A later cleanup should either truncate encoded bytes safely or rename/document the limit as characters. Also note that `subprocess.run(capture_output=True)` buffers Docker's selected output before the post-capture truncation, so the bound governs emitted evidence rather than peak subprocess capture memory.

### CR-HCR5-MIN-02 — The report overstates what elapsed time can distinguish

For any positive poll that never reaches HTTP 200, `_poll_ready()` continues until its deadline whether it repeatedly receives 503 or receives no response. Consequently `ready_poll_elapsed_seconds` will be near 60 seconds in both failure branches; a “much shorter elapsed time with 503” is not a possible normal result of the current loop. This does not impair the implementation's decisive signal because `ready_last_http_status` and `ready_last_reason` distinguish the branches. The hosted result must be interpreted as specified below, not by elapsed time alone.

## Correctness and safety review

The positive path now preserves exactly the final status and parsed readiness reason already computed by `_poll_ready()`. No-response attempts leave both values `null`, while a parsed non-200 response preserves its status and reason. Elapsed time is measured with `time.monotonic()` around only the existing poll and does not alter its deadline or mask a timeout.

Both log tails are fail-closed with respect to the smoke verdict: capture is best-effort, exceptions return an empty string, and capture failure cannot convert a smoke failure into a pass or crash the result-producing path. Positive logs are captured only when positive readiness fails; negative-control logs are captured only when the exact seal assertion fails. `main()` emits only non-empty captured tails to stderr. Status/reason metadata is always present after the corresponding poll, including successful runs; this is small structured evidence rather than an unconditional log dump.

The emitted log material is acceptably non-secret for this controlled gate. The container receives only the explicit fixture configuration in `build_docker_run_argv()`—no inherited host environment or credential arguments—and the application startup event uses the payload-safe structured logger with an allowlisted `reason` field. The tail is nevertheless raw container stdout/stderr, so this conclusion depends on retaining the current controlled fixture and explicit environment contract; future additions of credentials or arbitrary exception logging to this container invocation require renewed review.

The negative seam remains exact: the production image still has no test-seam mount or `PYTHONPATH`, and sealing still requires both status `503` and reason `artifact_test_embedding_seam_unavailable`. The new fields do not participate in `compute_all_ok()`, so they cannot weaken or mask a failure.

Schema and evidence compatibility is preserved. The payload keeps schema `m43-container-smoke-v1`; `assemble_m4_evidence.py` checks the existing five semantic booleans but does not freeze the inner `container_smoke.json` key set. Producer receipt filenames, hashes, sizes, and the protected baseline/assembly workflow are unchanged. The exact diff confirms no drift in `.github/workflows/ci.yml`, `scan_image_layers.py`, `assemble_m4_evidence.py`, `check_m4_baseline.py`, baseline evidence, lock generation, or dependencies.

## Test quality and bounded verification

The new tests exercise propagation of a concrete 503 reason, failure-only positive and negative log attachment, successful omission, exact negative status/reason, stdout/stderr combination, tail-not-head truncation, and best-effort capture failure. These assertions have direct defect-detection value: restoring the discarded `_poll_ready()` tuple values, making log capture unconditional, taking the head, or allowing capture exceptions to escape would fail focused tests. The remaining omissions—byte-versus-character behavior and direct `main()` stderr assertions—support the two minor findings but do not weaken the diagnostic-push gate.

Commands run in this independent review:

```text
venv/bin/python -m pytest -q \
  tests/unit/test_container_smoke_readiness_diagnostics.py \
  tests/unit/test_container_smoke_contract.py \
  tests/unit/test_container_smoke_bare_script.py
# 25 passed

venv/bin/python -m pytest -q \
  tests/unit/test_assemble_m4_evidence.py \
  tests/unit/test_check_m4_baseline.py
# 11 passed

git diff --check c51387a -- scripts/container_smoke.py
# exit 0

git diff --exit-code c51387a -- \
  .github/workflows/ci.yml scripts/scan_image_layers.py \
  scripts/assemble_m4_evidence.py scripts/check_m4_baseline.py \
  evaluation/baselines requirements.lock scripts/compile_lock.sh
# exit 0
```

No Native Linux, Ollama, DDGS, live gate, self-hosted runner, or environment mutation was performed. No commit, push, PR, or merge was performed.

## Required hosted interpretation branches

After the diagnostic-only push, inspect the `container` job's console and uploaded `container_smoke.json`:

1. **Last status is `503` with a concrete reason.** Treat this as a deterministic readiness rejection, not a timeout-budget diagnosis. Use the exact reason and matching startup log to scope the next narrow remediation (for example an artifact trust, settings, or engine initialization branch). Do not increase the timeout merely because elapsed time is near 60 seconds; that is expected for every non-converging poll.
2. **Last status and reason are both `null` (no HTTP response observed).** The poll never obtained a parseable readiness response. Use the log tail to distinguish startup still running, startup crash/exit, bind failure, or another transport-level problem. Only if the logs show healthy startup completing just after the deadline does evidence support a data-driven timeout adjustment.
3. **Last status is non-null but reason is `null`.** Treat this as an unexpected/malformed/non-JSON response path and investigate the recorded status and logs; do not classify it as either a healthy cold start or a known readiness rejection.
4. **The container job passes.** This proves the instrumented commit passed that hosted run, but it does not by itself establish that Iteration 5 fixed the prior root cause; the change is observational and the earlier failure may be intermittent. Continue the normal M4.3 evidence and review gates before any merge-readiness claim.
5. **The log tail is absent or empty on failure.** Preserve the smoke failure. This means best-effort log capture itself yielded no evidence; status/reason fields remain authoritative if present, and the result must not be upgraded or masked.

The downstream `m4-assemble` result should be read only after the producer jobs settle. A dependency-driven assemble failure is not a new root cause, and no result from this diagnostic rerun changes the existing M4.1/protected-live limitations.
