# M4.3 Artifact & Deployment Safety — Code Review Iteration 10

Reviewer: Fresh Codex independent final pre-merge code and acceptance reviewer

Baseline: `84f6b407c9dd6d2de262c4d8f708618d11b37766`

Reviewed state: entire current working-tree diff relative to that baseline, plus
the committed M4.3 branch history for the `consumer_fenced` watchdog readiness
fix

Verdict: **PASS — 9.8/10** (`CRITICAL 0`, `MAJOR 0`, `MINOR 1`, `TRIVIAL 0`)

## Scope and conclusion

I independently reviewed the milestone development orchestration guide and the
M4.3 Requirement, Plan, Traceability, Design, Implementation, Acceptance, hosted
remediation, design-review, code-review, remediation, and stop-report artifacts.
I reviewed every current tracked delta relative to `84f6b407`: the canonical
`requirements.lock` regeneration, certificate metadata binding in
`scripts/scan_image_layers.py`, scanner tests, singleton identity tests, and the
three updated integration/acceptance documents. I also inspected the committed
branch-history change to `scripts/orchestration_watchdog.py` and its tests,
including terminal-scoped argv, bounded `consumer_fenced` classification,
exact-once journal behavior, and nonzero termination.

The implementation is ready for the pre-merge code-quality gate. The scanner's
ordinary Label comparison is byte-exact, and its compatibility path is an exact
SHA-256-to-exact-Label lookup. An independent walk of the three documented real
bundles found that their deviation union is exactly the eight table entries,
with no missing or surplus compatibility pair. Issuer, Subject, Serial, MD5,
SHA1, SHA256, and Label remain bound to the immediately following certificate;
focused smuggling, wrong-digest, mutated-label, and full-bundle tests pass.

The regenerated lock contains the same 103 package stanzas as the baseline.
Its complete semantic version delta is limited to `pypdf` 6.15.0→6.16.0,
`uvicorn` 0.52.1→0.52.2, and `xxhash` 3.8.1→4.0.0; all other lock changes are
the generated header path or hash-set changes associated with current upstream
artifacts. This is consistent with a canonical fresh resolution rather than an
unexplained platform dependency change. The documented canonical execution used
`python:3.11-slim --platform linux/amd64` with `uv==0.8.15`, ran compilation
twice, passed `compile_lock.sh --verify`, and completed a clean hash-verified
install plus `pip check`. The hosted workflow uses Python 3.11, installs the
same pinned uv version, uses the same extra index and install ordering, and runs
the same verifier, so `compile_lock.sh` is hosted-compatible. I did not rerun the
Linux container or any image operation because this review explicitly forbids
Native Linux/image/live execution; this conclusion is based on code inspection,
the exact lock delta, and the recorded bounded acceptance evidence.

## Finding

### MINOR — CR-I10-MIN-01: acceptance evidence misstates the lock package count

`Acceptance_Report.md` §2.9 and `Implementation_Report.md` §15.2 say the lock
changed from 102 to 103 packages. Direct counting of canonical package stanzas
(`^[A-Za-z0-9_.-]+==`) gives **103 in baseline `84f6b407` and 103 in the current
lock**, and the package-name lists are identical. The reports correctly identify
the three version changes, the 222/195 line delta, reproducibility result, and
clean-install result; only the package-count explanation is false. This does not
invalidate the lock or acceptance outcome, but should be corrected from
`102→103` to `103→103 (package set unchanged)` before merge so the evidence is
internally exact.

## Protected boundaries

- `M4.1_BLOCKED=true` and `operational_status=BLOCKED` remain mandatory.
- Protected M3 live remains `NOT_RUN`/workflow `SKIPPED`.
- `overall_release_ready=false` remains mandatory and is not promoted by the
  deterministic M4.3 PASS.
- Hosted receipts remain `NOT_RUN` until a committed SHA is exercised.
- No Native Linux, Ollama, DDGS, protected live, M4.1 live, self-hosted, Docker
  build, image scan, or container smoke execution was performed in this review.

## Verification evidence

| Check | Result |
|---|---|
| `venv/bin/python -m pytest -q tests/unit/test_scan_image_layers.py tests/unit/test_rag_engine_singleton.py tests/unit/test_orchestration_watchdog.py` | **PASS: 128 passed** |
| Independent venv top-level, venv pip-vendored, and repository-default pip-vendored certifi walk | **PASS:** deviation union exactly equals all 8 compatibility pairs |
| Baseline/current lock package-name comparison | **PASS:** identical 103-package set; exactly 3 version bumps |
| `venv/bin/python -m pytest -q tests/unit/test_dependency_lock.py tests/unit/test_assemble_m4_evidence.py tests/unit/test_check_m4_baseline.py tests/unit/test_ci_workflow_contract.py` | **PASS: 20 passed** |
| `venv/bin/python scripts/check_markdown_links.py` | **PASS:** 129 files, 590 links, 0 failures |
| `git diff --check 84f6b407` before this report | **PASS** |
| Native Linux/image/Ollama/DDGS/live/self-hosted execution | **NOT RUN**, by scope |

## Gate

Severity count is `CRITICAL 0 / MAJOR 0 / MINOR 1 / TRIVIAL 0`; score is
**9.8/10**. The required gate is `CRITICAL=0`, `MAJOR=0`, and score `>=9.7`, so
Code Review Iteration 10 is **PASS**. This is the M4.3 pre-merge code-quality
verdict only; it does not alter the protected operational release boundaries.
