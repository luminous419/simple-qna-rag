# M4.3 Hosted CI Remediation Iteration 9 — Fresh Codex Code Review

Reviewer: Codex (independent dispatched review)  
Baseline: `c99419f` plus the current uncommitted Iteration 9 working-tree changes  
Scope: `src/simple_qna_rag/index/lifecycle.py`, `tests/unit/test_index_lifecycle.py`, and `Hosted_CI_Remediation_Iteration_9.md` only

## Verdict

**PASS — 9.8 / 10.0**

- CRITICAL: **0**
- MAJOR: **0**
- MINOR: **1** (the previously documented, intentionally deferred CR-HCIR6-MIN-03 only)
- TRIVIAL: **1** (non-blocking wording precision; see below)
- Pre-push code-quality Gate: **PASS**
- Hosted rerun / merge Gate: **NOT RUN; remains coordinator-owned**

The Gate requires CRITICAL 0, MAJOR 0, and score >= 9.7. The implementation closes the observed hosted Linux failure without weakening the immutable version artifacts, widening group/world write access, changing the trust-boundary error surface, or touching protected M4.1/M3 workflow and evidence boundaries.

## Findings

### CRITICAL — none

### MAJOR — none

### MINOR — CR-HCIR6-MIN-03 remains intentionally deferred

The lock test fixture still establishes seeded-copy stability rather than exercising a real resolver's durability and upgrade behavior. This is inherited, explicitly excluded from Iteration 9, and unrelated to the `current` pointer permission repair; it remains the single non-blocking MINOR from the preceding review.

### TRIVIAL — crash-recovery wording overstates production in-place writes

The new source comment, unit-test docstring, and Iteration 9 report say that crash-recovery paths overwrite `current` in place. The production reconciliation implementation reads the already-committed pointer and repairs history/receipt state; the in-place `Path.write_bytes()` is the fault-injection test's method of simulating the post-rename crash state. This does not invalidate `0o644`: `current` is the operator-owned mutable pointer, owner write does not add group/world write permission, `verify_version` already permits owner-write modes, and an owner of the containing writable directory could atomically replace the entry regardless. Future wording should distinguish the simulation seam from reconciliation behavior, but no code change is required for this Gate.

## Correctness and safety review

`activate()` still verifies the candidate and durably writes the `prepared` transition journal before constructing the pointer. `_write_fsync()` creates a new `0600` temporary file with `O_EXCL`, writes it completely, fsyncs it, and closes it. The new `os.chmod(tmp, 0o644)` then runs before `os.replace`, so the public `current` name is never exposed at `0600`; rename preserves the inode's mode, and the existing parent-directory fsync preserves the atomic activation durability sequence. A failure before rename leaves the previous pointer unchanged and is handled by the existing prepared-journal reconciliation contract.

Mode `0644` provides exactly the missing non-owner read bit required by the production container's explicit `--user 10001:10001`, without group/world write access. It does not alter the `versions/` parent (`0755`), immutable version directories (`0555`), or immutable version members (`0444`). It also does not change `expected_owner_uid`, dirfd/`O_NOFOLLOW` validation, member hash/mode verification, `resolve_current()`'s EACCES classification, or `_load_vectorstore()`'s typed trust-error propagation.

The regression test sets umask `0077` and asserts the final exact mode, so removing the chmod yields `0600` and is detected independently of ambient host defaults. The exact `0644` assertion also rejects broader modes such as `0664`/`0666`. Its owner-context `os.access` checks are redundant rather than harmful; the exact mode assertion plus the real cross-UID container smoke is the meaningful evidence for UID 10001.

The existing activation/rollback loop and fault-injection suite cover parseable atomic pointer replacement, prepared and pointer-committed recovery, history/receipt reconciliation, and unchanged-current failure behavior. The focused run passed all 40 lifecycle and fault-injection tests, including the `0444`-sensitive crash-state simulation.

## Independent verification

```text
venv/bin/python -m pytest -q \
  tests/unit/test_index_lifecycle.py \
  tests/integration/test_index_lifecycle_fault_injection.py
40 passed, 3 warnings in 2.96s

venv/bin/python -m pytest -q
1329 passed, 1 skipped, 4 warnings in 174.02s

git diff --check
exit 0
```

The full-suite count independently matches the Iteration 9 report and is exactly one test above Iteration 8's `1328 passed, 1 skipped`, consistent with the single added regression test.

The reported real Docker validation is also sufficient for this bounded pre-push Gate: it built the production target for `linux/amd64` and ran the actual container with UID/GID `10001:10001`, the read-only/drop-capability/no-new-privileges smoke contract, readiness `200` / reason `ok`, positive mock query/static/root checks, and the sealed production-test-seam negative control `503` / `artifact_test_embedding_seam_unavailable`. That execution directly crosses the host-owner/container-non-owner permission boundary that ordinary owner-context unit tests cannot reproduce. Combined with the independently reproduced full suite and focused crash-recovery tests, it is sufficient evidence that the narrow remediation fixes the hosted failure without a deterministic regression; the next hosted run remains required as post-push acceptance evidence, not as a prerequisite for this code-review Gate.

No Native Linux/Ollama/DDGS/live/self-hosted execution was performed or inferred.

## Protected boundaries and release state

The reviewed code diff is confined to lifecycle pointer creation and its unit regression test. It does not modify `.github/workflows/ci.yml`, `scripts/scan_image_layers.py`, `scripts/assemble_m4_evidence.py`, `scripts/check_m4_baseline.py`, `evaluation/baselines/m3_initial.*`, `requirements.lock`, `requirements.txt`, or `deploy/Dockerfile`. Legacy M3 approved hashes/import behavior and protected live triggers, runner labels, and environment approval remain unchanged.

Accordingly, this PASS is only the Iteration 9 pre-push code-quality decision. It does not synthesize or upgrade M4.1 receipts, does not execute the protected M3 gate, and does not change `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN` / workflow `SKIPPED`, or `overall_release_ready=false`.

## Gate decision

**PASS — 9.8 / 10.0; CRITICAL 0, MAJOR 0, MINOR 1, TRIVIAL 1.** Commit/push and the hosted CI rerun may proceed under coordinator ownership. Overall M4 release readiness remains blocked on its separate protected operational evidence.
