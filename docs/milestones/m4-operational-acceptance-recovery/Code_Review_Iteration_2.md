# M4 Operational Acceptance Recovery — Code Review Iteration 2

Reviewer: Fresh Codex independent code reviewer  
Date: 2026-08-15  
Diff reviewed: entire uncommitted worktree against `origin/master`

## Gate decision

**PASS — 9.8 / 10.0**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 0 |
| MINOR | 0 |
| TRIVIAL | 0 |

PASS requires CRITICAL 0, MAJOR 0, and score at least 9.7. The reviewed
implementation, workflow contract, schema migration, tests, and active project
status satisfy that Gate.

## Iteration-1 finding closure

### CR-I1-MAJ-01 — CLOSED

`docs/Problem.md` now orders the Iteration-4 8.9 stop before the later
resumption and labels it explicitly as historical and superseded. Its current
status says Recovery Cycle 1 passed at 9.8, implementation is complete
pre-merge, and commit/push/PR/merge plus post-merge hosted `python-tests` and
`container` validation remain pending.

`docs/Roadmap.md` now presents the current position and M4 heading as
hosted/OCI policy-change implementation complete pre-merge with post-merge
hosted CI validation pending. It retains the Iteration-4 8.9 stop only as
chronology followed by the Recovery Cycle 1 approval and resumption.

Both documents preserve the exact support boundary: native Linux/Ollama is
`NOT_ADOPTED`, not PASS or WAIVED, and
`native_linux_release_ready`, `full_production_release_ready`, and the legacy
`overall_release_ready` alias remain false. No stale current terminal-stop
claim remains.

## Independent regression assessment

- The v2 assembler emits exact schema/policy constants, derives hosted
  readiness only from all four deterministic producers, and fixes
  native/full/overall readiness false.
- The checker defaults to strict v2, admits v1 only through
  `--allow-legacy-v1`, preserves v1's frozen blocked semantics, rejects mixed
  or unknown schemas, independently recomputes producer algebra and provenance
  aliases, and rejects every tested `NOT_ADOPTED` or false-readiness
  substitution.
- Post-merge verification is fail-closed through the five mandatory identity
  bindings when `--require-identity-binding` is used. The normative deployment
  procedure supplies that complete command.
- The workflow has exactly the four hosted producers, their hosted assembler,
  and the harmless opt-in informational stub. Ordinary push/PR paths neither
  depend on nor schedule native, Ollama, protected-environment, or self-hosted
  execution.
- The M4.3 strict-schema node remains pinned to the legacy-v1 compatibility
  test, preserving the historical acceptance contract while v2 receives its
  own adversarial coverage.
- The allowed-delta audit and regression matrix cover statement/decorator
  spans, raw BOM/shebang/encoding preambles, including the comment-first
  second-line PEP 263 cookie case.
- Public and operational documentation consistently limits certification to
  hosted Python/frontend plus OCI and makes the former M4.1 native runbook
  explicitly superseded and non-executable.

## Verification performed

- `pytest -q tests/unit/test_assemble_m4_evidence.py tests/unit/test_check_m4_baseline.py tests/unit/test_ci_workflow_contract.py tests/unit/test_doc_audit_no_active_native_runner_procedure.py`:
  **216 passed** (one non-failing ambient torchvision image-extension warning).
- `python scripts/check_markdown_links.py`:
  **158 files, 684 links, 0 failures**.
- `git diff --check`: **passed**.
- Complete tracked diff and all untracked M4-OAR documents/tests were
  inspected against `origin/master`; no runtime product code, historical
  receipt, or live evidence payload was changed.

Hosted clean-environment `python-tests` and Linux-x86_64 `container` evidence
cannot exist for this uncommitted pre-merge worktree. They are explicit
post-commit/push/merge release-acceptance evidence to be obtained for the exact
merged SHA, not a defect in the reviewed code. This PASS does not claim that
release acceptance has already completed and does not substitute local results
for those hosted jobs.

No live, native Linux, Ollama, self-hosted runner, or protected-environment
execution was performed.

## Remaining acceptance work

Commit, push, review, and merge the approved diff; then require successful
hosted `python-tests` and `container` jobs and run the identity-bound v2
baseline verification from `docs/operations/deployment_runbook.md` §6.1
against the exact merged-SHA workflow artifact. Native/full/overall readiness
must remain false throughout.
