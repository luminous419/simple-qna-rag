# M4.2 Safe Serving Boundary — Final Verification Report

Date: 2026-08-11 (Asia/Seoul)  
Base revision: `0c84795729860848d6290080c9092ea9077fcf52` (`master`, equal to `origin/master`)  
Verification scope: complete uncommitted deterministic M4.2 delta after Code Review Iteration 5 (`PASS — 9.8/10`)

## Decision

**PASS — release-ready for the approved deterministic M4.2 scope.**

The complete locked Python suite, Node suite, deterministic repeat-10 acceptance, retained genuine
mismatch negative control, generated-document audits, link check, compilation check, reproducible
dependency lock verification, secret/artifact review, and whitespace check passed. No production
code or tests were changed by this final gate. Acceptance output and the fresh virtual environment
were kept under `/tmp/m42-final-verify.Uo4r3t`; no acceptance receipt, log, cache, environment,
runtime state, credential, or other generated execution artifact was added to the intended commit.

This decision is deliberately limited to deterministic local M4.2. The opt-in live 12-case
Ollama/network gate, M3 live/14-gate regression, and M4.1 operational acceptance were not run and
are not inferred from this PASS. `M4.1_BLOCKED=true` remains an independent M4 release blocker.

## Exact verification receipts

The fresh locked environment was created with `python3 -m venv
/tmp/m42-final-verify.Uo4r3t/venv`, then installed only from the checked-in lock plus the local
editable package (`python -m pip install -r requirements.lock`; `python -m pip install --no-deps
-e .`).

| Command | Result |
|---|---|
| `/tmp/m42-final-verify.Uo4r3t/venv/bin/python -m pip check` | PASS: `No broken requirements found.` |
| `/tmp/m42-final-verify.Uo4r3t/venv/bin/python -m pytest -q` | PASS: 1040 passed, 1 skipped, 2 warnings in 148.19s. The warnings are the existing Starlette `httpx` deprecation and pytest class-scoped instance-fixture deprecation. |
| `/tmp/m42-final-verify.Uo4r3t/venv/bin/python scripts/run_m42_acceptance.py --profile deterministic --repeat 10 --seed 4202 --output /tmp/m42-final-verify.Uo4r3t/m42-acceptance.json` | PASS, exit 0: 10 profiles, 11 collected nodes, 100 profile results, 110 node receipts, 100 profile conservation rows, 10 aggregate conservation rows, `M4.1_BLOCKED=true`. |
| `/tmp/m42-final-verify.Uo4r3t/venv/bin/python scripts/run_m42_acceptance.py --profile deterministic --repeat 10 --seed 4202 --inject-conservation-mismatch --output /tmp/m42-final-verify.Uo4r3t/m42-negative.json` | Expected negative PASS, exit 1: top-level `status=FAIL`, 100/110/100/10 complete evidence retained, negative-control `status=FAIL`, genuine rejected node receipt retained, diagnostic `conservation_mismatch`, `M4.1_BLOCKED=true`. |
| `npm test` | PASS: 1 test file, 9 tests. |
| `/tmp/m42-final-verify.Uo4r3t/venv/bin/python scripts/generate_field_spec.py --check` | PASS: no generated field-spec drift. |
| `/tmp/m42-final-verify.Uo4r3t/venv/bin/python scripts/logging_callsite_audit.py --check` | PASS: no generated logging-callsite drift. |
| `/tmp/m42-final-verify.Uo4r3t/venv/bin/python scripts/check_markdown_links.py` | PASS after this report: 98 files (80 tracked + 18 untracked), 440 links, 0 failures. |
| `/tmp/m42-final-verify.Uo4r3t/venv/bin/python -m compileall -q src scripts tests` | PASS, exit 0. |
| `bash scripts/compile_lock.sh --verify` | PASS: two independent 102-package resolutions (2.82s and 1.77s), reproducible, no committed drift. |
| `git diff --check` | PASS after this report, exit 0. |

## Scope and hygiene review

`git status --branch --porcelain=v2`, `git diff --stat`, `git diff --name-status`, `git diff
--numstat`, `git diff --cached --stat`, and the complete untracked-file inventory were inspected.
The branch is not ahead of or behind `origin/master`; there are no staged implementation changes.
The delta is consistent with the Requirement, Plan, recovered Design, implementation iterations,
and five code-review iterations. Report conclusions agree: Iteration 7 implements the approved
contract, Code Review Iteration 5 closes all CRITICAL/MAJOR/MINOR findings at 9.8/10, and this gate
independently reproduces the required results.

A repository scan for private-key markers and common AWS, OpenAI, and GitHub token forms found no
secret. Its sole token-like match is the intentionally fake `ghp_AdversarialTokenLike1234567890abcd`
fixture in `tests/integration/test_check_config_cli.py`. Untracked/modified commit candidates contain
no `.env`, log, PID, SQLite, acceptance-output, cache, virtual-environment, or runtime-orchestration
file. Existing ignored/local `.claude`, `runtime`, `evaluation/reports`, and environment material is
outside `git status` and outside the intended commit. The 102-package lock is generated only through
`scripts/compile_lock.sh` and is reproducible.

## Exact intended commit scope

The intended commit is exactly this deterministic M4.2 set (including this final report), and no
other path:

- `docs/generated/settings_field_spec.md`, `requirements.lock`, and
  `scripts/run_m42_acceptance.py`.
- `src/simple_qna_rag/agent.py`, `src/simple_qna_rag/rag_engine.py`,
  `src/simple_qna_rag/settings.py`, `src/simple_qna_rag/web_search.py`.
- `src/simple_qna_rag/observability/{deadline.py,health.py,metrics.py,request_context.py,terminal_ledger.py}`.
- `src/simple_qna_rag/web/{body_limit.py,concurrency.py,errors.py,scheduling.py,server.py}`.
- `tests/conftest.py`, `tests/evaluation/test_m4_safe_serving_load.py`.
- `tests/integration/{test_health_endpoints.py,test_metrics_live_traffic.py,test_output_surface_capture.py,test_request_logging_matrix.py,test_web_bootstrap_matrix.py,test_web_concurrency.py,test_web_disconnect.py,test_web_input_boundary.py}`.
- `tests/unit/{test_dependency_lock.py,test_dependency_snapshot.py,test_m42_acceptance_runner.py,test_observability_metrics.py,test_query_executor.py,test_readiness_saturation.py,test_settings.py,test_settings_inventory.py,test_shutdown_drain.py}`.
- `docs/milestones/m4.2-safe-serving-boundary/{Requirement.md,Plan.md,Design.md,Design_Review_Iteration_1.md,Design_Review_Iteration_2.md,Design_Review_Iteration_3.md,Design_Review_Iteration_4.md,Design_Recovery_Review_Iteration_1.md,Design_Recovery_Review_Iteration_2.md,Design_Recovery_Review_Iteration_3.md,Design_Recovery_Review_Iteration_4.md,Design_Recovery_Validation.md,Implementation_Report.md,Code_Review_Iteration_1.md,Code_Review_Iteration_2.md,Code_Review_Iteration_3.md,Code_Review_Iteration_4.md,Code_Review_Iteration_5.md,Final_Verification_Report.md}`.

Recommended branch: `feat/m4.2-safe-serving-boundary`  
Recommended commit title: `feat: enforce the M4.2 safe serving boundary`  
Recommended PR title: `M4.2: add deterministic safe serving boundary`

## Explicit exclusions

No commit, push, pull request, merge, email, release, live Ollama/network execution, M3 live/14-gate
execution, or M4.1 operational execution was performed. The deterministic M4.2 commit may be
published for review, but an overall M4 release must continue to show `M4.1_BLOCKED` until that
separate gate is resolved or explicitly risk-accepted by the appropriate owner.
