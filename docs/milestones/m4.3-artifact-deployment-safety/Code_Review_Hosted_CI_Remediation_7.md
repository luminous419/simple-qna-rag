# M4.3 Hosted CI Remediation Iteration 7 — Fresh Codex Code Review

Reviewer: Codex (independent dispatched review)  
Baseline: `2418119` plus the uncommitted Iteration 6/7 working-tree changes  
Primary Gate scope: CR-HCIR6-MAJ-01 and CR-HCIR6-MAJ-02

## Verdict

**FAIL — 9.3 / 10.0**

- CRITICAL: **0**
- MAJOR: **1**
- MINOR: **1** (the intentionally deferred CR-HCIR6-MIN-03 only)
- Pre-merge code-quality Gate: **FAIL**
- Diagnostic commit/push Gate: **FAIL**
- Post-merge/hosted acceptance Gate: **NOT READY / NOT RUN**

CR-HCIR6-MAJ-01 is closed: the exact direct `os.open("current", ..., dir_fd=root.fd)` EACCES route becomes the allowlisted `current_pointer_permission_denied`, and the reason now survives `TrustBoundaryError` -> `IndexTrustError` -> `EngineArtifactError` -> the existing readiness `artifact_{reason}` surface. CR-HCIR6-MAJ-02 is only partially closed. The implementation enforces the strict UTF-8 byte ceiling and selects the latest startup line on the over-budget path, but it does not satisfy “latest startup at most once” for an under-budget log and deduplicates only a *complete* overlap, not a tail beginning inside the retained startup record.

The Gate requires CRITICAL 0, MAJOR 0, and score >= 9.7. One bounded MAJOR therefore keeps this review at FAIL even though all deterministic tests and the ordinary real-container smoke pass.

## Closure review

### CR-HCIR6-MAJ-01 — PASS / closed

`resolve_current()` now translates only `errno.EACCES` at its direct current-pointer open to `TrustBoundaryError("current_pointer_permission_denied")`; ENOENT legacy fallback, ELOOP handling, and unrelated errno behavior remain unchanged. The literal is in `REASONS`. `_load_vectorstore()` catches a `TrustBoundaryError` from `resolve_current()` before the verified-FAISS call and maps it through the same `IndexTrustError` channel already used for artifact verification failures.

The two new tests are mutation-strength for the two code changes: removing the EACCES branch exposes a raw `PermissionError`, and removing the new `_load_vectorstore()` catch collapses the reason to generic initialization failure. The existing integration test independently proves that an allowlisted `EngineArtifactError.reason` becomes HTTP 503 `artifact_{reason}`. Direct inspection of the server/readiness path confirms there is no lossy translation between those surfaces.

### CR-HCIR7-MAJ-01 (CR-HCIR6-MAJ-02 incomplete) — startup selection and overlap dedupe are not exhaustive

`_capture_container_logs()` returns `combined` immediately whenever its UTF-8 encoding fits within `max_bytes`. Consequently, two or more startup records below the budget are all returned rather than retaining at most the latest one. An independent probe returned two startup events (`70` encoded bytes) with `max_bytes=1000`.

On the over-budget path, `if startup and startup in tail` removes duplication only when the tail contains the complete startup string. If the byte tail starts *inside* that startup record, the function prepends the complete startup and then appends the overlapping suffix. An independent probe produced:

```text
{"event": "startup", "reason": "engine_init_failed"}
_init_failed"}
TAIL
```

The output remains within the byte ceiling, but duplicated overlapping bytes consume the shared diagnostic budget and violate the explicit overlap-deduplication contract. The added repeated-startup test forces the combined log over budget, while the overlap test deliberately gives the tail enough room to contain the entire startup line; neither test detects these two boundary mutations.

**Bounded remediation:** perform startup selection before the size early return whenever startup records exist, retain only the latest record, and compose the latest record with a byte tail after removing any maximal suffix/prefix overlap (including partial overlap). Add one under-budget repeated-startup test and one tail-starts-inside-startup test that asserts exact non-duplication as well as `len(result.encode("utf-8")) <= max_bytes`. Preserve the current deterministic UTF-8 decode policy, shared ceiling, latest-event preference, and plain no-startup tail behavior.

### CR-HCIR6-MIN-03 — intentionally deferred MINOR

The lock fixture still proves seeded-copy stability rather than resolver durability and the true upgrade path. Per the review assignment this is intentionally excluded, remains MINOR, and is neither promoted nor used to expand the present remediation scope.

## Verification evidence

```text
focused MAJ-01/MAJ-02 plus readiness tests
56 passed, 3 warnings in 10.06s

venv/bin/python -m pytest -q
1326 passed, 1 skipped, 4 warnings in 166.43s

python scripts/generate_field_spec.py --check
exit 0

python scripts/logging_callsite_audit.py --check
exit 0

python scripts/check_markdown_links.py
141 files, 597 links, 0 failures

docker build --platform linux/amd64 --target production ...
PASS, image sha256:4141c2758256026d43c1b02da3443e7534cc8d9d43c912e0a74c8146956276c9

venv/bin/python scripts/container_smoke.py --image simple-qna-rag:iter7-review
status PASS; all six semantic booleans true; readiness reason ok;
negative control 503 / artifact_test_embedding_seam_unavailable
```

The generated/markdown checks passed, and the full suite count exactly reproduces the Iteration 7 report. The real ordinary container build and smoke were rerun against the current worktree. They validate the normal container path but do not exercise the two `_capture_container_logs()` boundary counterexamples above.

Protected surfaces have no diff: `.github/workflows/ci.yml`, `scripts/scan_image_layers.py`, `scripts/assemble_m4_evidence.py`, `scripts/check_m4_baseline.py`, `evaluation/baselines/m3_initial.*`, `requirements.lock`, and `requirements.txt`. `M4.1_BLOCKED=true`, `operational_status=BLOCKED`, protected M3 live `NOT_RUN`/workflow `SKIPPED`, and `overall_release_ready=false` remain intact. No Native Linux/Ollama/DDGS/live/self-hosted/environment-approval surface was executed or changed.

## Gate classification and next step

This is a **pre-merge FAIL**. Close only CR-HCIR7-MAJ-01 with the bounded startup-selection/partial-overlap correction and two mutation-strength tests, then rerun deterministic checks and obtain another fresh review. Do not commit, push, merge, or run protected operational gates from this result.
