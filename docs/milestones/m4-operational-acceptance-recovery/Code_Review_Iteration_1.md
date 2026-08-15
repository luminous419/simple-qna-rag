# M4 Operational Acceptance Recovery — Code Review Iteration 1

Reviewer: Fresh Codex independent code reviewer  
Date: 2026-08-15  
Diff reviewed: entire uncommitted worktree against `origin/master`

## Gate decision

**FAIL — 9.4 / 10.0**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 1 |
| MINOR | 0 |
| TRIVIAL | 0 |

PASS requires CRITICAL 0, MAJOR 0, and score at least 9.7. The implementation
and focused deterministic tests are strong, but the current public project
status contradicts itself and therefore does not satisfy the documentation and
support-boundary acceptance surface yet.

## Findings

### CR-I1-MAJ-01 — Current project status is followed by a stale terminal-stop state

Exact references:

- `docs/Problem.md:138-152` says implementation is complete pre-merge and the
  Recovery Cycle design Gate passed, but `docs/Problem.md:154-164` immediately
  says the design Gate is stopped, implementation was never started, and a new
  approval is required.
- `docs/Roadmap.md:42` and `docs/Roadmap.md:193-195` label the current position
  and M4 status as “design Gate stopped,” while `docs/Roadmap.md:262-279` says
  the same milestone passed design review and is implemented pre-merge.

Impact: these are active project-status documents, not the explicitly bannered
historical M4.1 runbook. A reader cannot determine whether the implementation
is authorized, complete, or prohibited. This conflicts with Requirement
M4-OAR-REQ-005's truthful public/support boundary and with the milestone
documents' own current `IMPLEMENTATION COMPLETE (PRE-MERGE)` state.

Required fix: retain the Iteration-4 stop as clearly labeled historical
chronology, but move it before the resumption/current update or add an explicit
superseded/resumed heading. Update the Roadmap diagram and M4 status heading to
the current `implementation complete (pre-merge), post-merge validation
pending` state. Keep the historical 8.9 stop facts, but do not present them as
the current terminal outcome.

## Independent implementation assessment

- Schema dispatch is strict: v2 is default; v1 requires
  `--allow-legacy-v1`; mixed/unknown shapes fail exact-key/schema checks; v1
  stays frozen at live `NOT_RUN`, M4.1 `BLOCKED`, blocked true, and overall
  false.
- `NOT_ADOPTED` cannot become PASS: both compatibility gates are exact-pinned,
  `operational_status` is exact-pinned, and adversarial enum/value cases pass.
- Hosted readiness is recomputed solely from the four producer variants.
  Native, full-production, and overall readiness remain false, with overall
  checked as the full-production alias.
- Identity and provenance checks bind optional expected run/SHA fields and
  recompute the two payload-hash aliases from the already validated producer
  map. The post-merge runbook correctly makes all identity flags mandatory.
- The workflow has the exact six-job shape, four deterministic dependencies,
  no ordinary dependency on the live stub, and a false-default explicit
  dispatch marker whose only execution is a pinned hosted informational step.
  Ordinary push/PR therefore has no live, environment, Ollama, or self-hosted
  scheduling surface.
- The allowed-delta audit covers the pinned assembler, decorator spans, raw
  BOM/shebang/cookie comparison, encoding conflicts, and the comment-first
  second-line PEP 263 cookie case that remained after design review.
- `run_m43_acceptance.py` is correctly repinned to
  `test_v1_legacy_strict_schema_and_algebra_matrix`, preserving the v1 M4.3
  acceptance node rather than silently substituting a v2 claim.
- The deployment verification procedure is identity-bound and the historical
  M4.1 procedure has a clear non-executable banner. README and recovery/deploy
  runbooks consistently describe native/Ollama as unsupported/best-effort.

## Verification performed

- `pytest -q tests/unit/test_assemble_m4_evidence.py tests/unit/test_check_m4_baseline.py tests/unit/test_ci_workflow_contract.py tests/unit/test_doc_audit_no_active_native_runner_procedure.py`:
  **216 passed**.
- `python scripts/check_markdown_links.py`: **157 files, 682 links, 0 failures**.
- `git diff --check`: **passed**.
- Entire changed-file list and diff against `origin/master`: inspected; no
  product/runtime code, receipt, live execution, or historical evidence payload
  was changed.

Full local `pytest -q` did not collect: eight modules fail while importing the
ambient FastAPI/Pydantic stack because this shared environment lacks
`email-validator>=2`. This is an environment gap, not evidence of a changed-test
failure; the focused changed suite imports and passes, and the hosted
`python-tests` job performs a clean hash-locked install, `pip check`, web import,
and full pytest. Nevertheless, the pre-merge Gate is not fully evidenced until
that hosted job passes on the reviewed commit.

The production Docker build was not run locally. This Apple Silicon/arm64
Docker environment is known to reject the Linux-x86_64-pinned lock hashes; the
workflow's `container` job runs on GitHub-hosted Ubuntu x86_64 and is the
required architecture for the locked build, layer scan, smoke, and producer
receipt. The arm64 gap is therefore not a product finding, but hosted CI
`container` success remains mandatory and cannot be inferred from static tests.

No live/native Linux/Ollama/self-hosted/protected-environment command was run.

## Required next step

Fix CR-I1-MAJ-01, rerun the deterministic changed tests, markdown links, and
`git diff --check`, then obtain clean hosted `python-tests` and `container`
results. A fresh review can PASS only after the documentation contradiction is
removed and hosted gaps are closed by CI evidence.
