# M4.3 Hosted CI Remediation Iteration 8 — Fresh Codex Code Review

Reviewer: Codex (independent dispatched review)  
Baseline: `2418119` plus the current uncommitted Iteration 6–8 working-tree changes  
Primary Gate scope: complete closure of CR-HCIR7-MAJ-01, with focused regression proof for the previously closed current-pointer EACCES path

## Verdict

**PASS — 9.8 / 10.0**

- CRITICAL: **0**
- MAJOR: **0**
- MINOR: **1** (the intentionally deferred CR-HCIR6-MIN-03 only)
- Pre-merge code-quality Gate: **PASS**
- Diagnostic commit/push Gate: **READY for the coordinator's next phase; not performed by this review**
- Post-merge/hosted acceptance Gate: **NOT RUN / remains separate**

The Gate requires CRITICAL 0, MAJOR 0, and score >= 9.7. CR-HCIR7-MAJ-01 is fully closed: startup selection now precedes the budget early return, and over-budget composition removes the maximal byte-level suffix/prefix overlap between the retained latest startup record and the tail. The two new tests reproduce and kill the exact prior mutations. The sole remaining MINOR is the explicitly out-of-scope lock fixture limitation and does not block this bounded Gate.

## Closure review

### CR-HCIR7-MAJ-01 — PASS / closed

`_capture_container_logs()` splits the combined Docker output into newline-preserving records, identifies startup records by position, retains only the last position, and removes every earlier startup record before testing the UTF-8 byte budget. Therefore a repeated startup sequence that is comfortably below `max_bytes` no longer escapes through the early return; an independent focused test returned exactly the latest `attempt=2` record and no `attempt=1` record.

For an over-budget log, the function reserves the latest startup record first and fills the shared remainder with the newest bytes. If the complete startup bytes already occur in the tail, it returns the tail unchanged. Otherwise it searches from the maximum possible overlap down to one byte for `startup_bytes[-k:] == tail_bytes[:k]` and appends only the non-overlapping tail suffix. This is the required maximal suffix/prefix removal at byte level, including a tail window beginning inside the startup record.

The budget is shared: `startup_bytes + tail_bytes[overlap:]` cannot exceed `len(startup_bytes) + remaining == max_bytes`. A startup record larger than the budget remains head-preserved and bounded. Final decoding is deterministic and performed once with UTF-8 `errors="ignore"`, so a multibyte boundary cannot raise or introduce replacement bytes that grow the result. The no-startup path remains a plain newest-byte tail. Existing multibyte, oversized startup, repeated over-budget startup, complete-overlap, no-startup, and Docker-failure contracts all pass.

### Mutation strength of the two new tests — PASS

`test_capture_container_logs_dedupes_repeated_startup_events_even_under_budget` fixes the exact early-return boundary at 70 encoded bytes with `max_bytes=1000`, asserts one startup marker, asserts latest-event presence, and asserts earlier-event absence. An independent execution of the prior algorithm returned two startup records, so reverting startup selection behind the size check makes the test fail.

`test_capture_container_logs_trims_partial_overlap_when_tail_starts_inside_startup_line` constructs `remaining = len(trail) + 15`, which forces the tail prefix to be a proper suffix of the startup record rather than containing the whole record. Its exact equality assertion requires `startup_line + trail_noise`; the prior complete-containment-only algorithm independently reproduced `startup_line + '_init_failed"}\n' + trail_noise`, so removing the maximal-overlap logic makes the test fail.

### Previously closed current-pointer EACCES path — PASS / no regression

Focused inspection and tests confirm the earlier MAJ-01 closure remains intact. `resolve_current()` translates only direct `current`-open `errno.EACCES` into allowlisted `current_pointer_permission_denied`; `_load_vectorstore()` translates `TrustBoundaryError.reason` into `IndexTrustError`; initialization exposes the typed `EngineArtifactError.reason`; and the existing readiness integration contract produces the bounded `artifact_{reason}` surface. The focused suite exercises both mutation-strength links and the generic readiness disclosure path. ENOENT legacy fallback, ELOOP handling, and unrelated errno behavior were not broadened by Iteration 8.

### CR-HCIR6-MIN-03 — intentionally deferred MINOR

The lock test fixture still proves seeded-copy stability rather than exercising a real resolver's durability and upgrade behavior. Per the assignment, this remains one intentional MINOR only; it is not promoted, remediated, or used to expand Iteration 8 scope.

## Verification evidence

```text
focused container diagnostics + EACCES propagation/readiness tests
58 passed, 3 warnings in 8.90s

independent prior-algorithm mutation probes
under-budget repeated startup: 2 records (expected old-code failure)
partial overlap: duplicated `_init_failed"}\n` fragment (expected old-code failure)

venv/bin/python -m pytest -q
1328 passed, 1 skipped, 4 warnings in 166.38s

python scripts/generate_field_spec.py --check
exit 0

python scripts/logging_callsite_audit.py --check
exit 0

python scripts/check_markdown_links.py
143 files, 597 links, 0 failures

docker build --platform linux/amd64 --target production ...
PASS, image sha256:4141c2758256026d43c1b02da3443e7534cc8d9d43c912e0a74c8146956276c9

venv/bin/python scripts/container_smoke.py --image simple-qna-rag:iter8-review
status PASS; all six semantic booleans true; readiness reason ok;
negative control 503 / artifact_test_embedding_seam_unavailable
```

The ordinary real-container smoke was run against the current worktree using the requested linux/amd64 production target. No Native Linux, Ollama, DDGS, live, self-hosted, protected-environment approval, or protected M3/M4.1 live gate was executed.

## Protected surfaces and release state

`git diff --exit-code` confirms no change to `.github/workflows/ci.yml`, `scripts/scan_image_layers.py`, `scripts/assemble_m4_evidence.py`, `scripts/check_m4_baseline.py`, `evaluation/baselines/m3_initial.*`, `requirements.lock`, or `requirements.txt`. The protected state remains fail-closed: `M4.1_BLOCKED=true`, `operational_status=BLOCKED`, protected M3 live `NOT_RUN` with workflow `SKIPPED`, and `overall_release_ready=false`. This pre-merge PASS does not claim post-merge hosted or overall release readiness.

## Gate decision

The pre-merge code-quality Gate is **PASS** with CRITICAL 0, MAJOR 0, MINOR 1, and 9.8/10.0. CR-HCIR7-MAJ-01 requires no further remediation. Commit, push, hosted CI, and operational acceptance remain coordinator-owned next-phase actions and were not performed by this review.
