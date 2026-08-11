# M4.2 Safe Serving Boundary — Code Review Iteration 5

Date: 2026-08-11 (Asia/Seoul)  
Base revision: `0c84795`  
Review scope: complete current uncommitted M4.2 delta, approved Requirement/Plan/Design and recovery
artifacts, Code Review Iterations 1–4, Implementation Report Iteration 7, production serving and
observability modules, tests, deterministic runner, and dependency lock.

## Gate

**PASS — 9.8/10**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 0 |
| MINOR | 0 |

The conditional-extension threshold is met: score is at least 9.7, no CRITICAL or MAJOR finding
remains, and no release-relevant minor finding was identified. M42-CR4-MAJ-001 is closed. The
receipt boundary now rejects every previously accepted contradiction while retaining the complete
request, RAG, executor, stale, identity, epoch, and snapshot evidence needed to diagnose rejection.

## M42-CR4-MAJ-001 adversarial closure

| Required reproduction | Iteration 5 result | Independent evidence |
|---|---|---|
| Request/RAG `success=1` versus executor `internal=1` | **REJECTED** | Exact probe raised `request_executor_terminal_mismatch`; the checked-in test asserts the same diagnostic ([runner test](../../../tests/unit/test_m42_acceptance_runner.py#L101)). |
| Boolean row `ledger_epoch` | **REJECTED** | Exact probe raised `malformed_runtime_receipt`; row identity requires `type(...) is int` ([parser](../../../scripts/run_m42_acceptance.py#L105)). |
| Boolean `snapshot_epoch` | **REJECTED** | Exact probe raised `snapshot_identity_mismatch`; snapshot epoch also requires exact `int` ([parser](../../../scripts/run_m42_acceptance.py#L146)). |
| Negative `stopped_with_running` | **REJECTED** | Exact probe raised `malformed_snapshot_types` ([parser](../../../scripts/run_m42_acceptance.py#L131)). |
| Negative `stopped_with_orphaned` | **REJECTED** | Exact probe raised `malformed_snapshot_types` ([parser](../../../scripts/run_m42_acceptance.py#L131)). |
| Overlapping zero-work second executor | **RETAINED AND REJECTED** | The ledger increments fixed-cardinality `executor_identity_conflicts` before returning false ([ledger](../../../src/simple_qna_rag/observability/terminal_ledger.py#L88)); the parser requires exact zero, and the end-to-end unit probe rejects the serialized conflict ([runner test](../../../tests/unit/test_m42_acceptance_runner.py#L133)). |

The source algebra is no longer the permissive residual from Iteration 4. It preserves all three
maps, requires request and RAG equality, consumes application terminals positionally from executor
terminals, explicitly models disconnect-after-success, retains the exact per-reason remainder as
direct control tickets, and reconciles executor totals with admission and terminal counter deltas
([source algebra](../../../scripts/run_m42_acceptance.py#L152)). Independent mutation of application
disconnect versus executor internal was also rejected as `request_executor_terminal_mismatch`.

## Nearby source-algebra and lifecycle review

No adjacent fail-open substitution or dropped source map was found.

- Disconnect-after-success is the sole non-identical application/executor relation: executor
  cancellation is consumed first and remaining application disconnects must consume executor
  success. Executor cancellation must still equal the immutable cancellation-counter delta.
- Direct control tickets remain separate executor remainders and are included in the whole executor
  count. Admission `not_ready`/`overloaded`, initial and promotion `SubmitFailed`, shutdown rejection,
  queue timeout, execution timeout, caller cancellation, success, and internal completion each have
  a production terminal transition and counter equation.
- Promotion failure still consumes one or two failed FIFO heads without retaining tickets or
  capacity and continues to a startable head. Executor cancellation preserves the running resource
  until the pool future completes; shutdown drains and mandatory non-waiting shutdown remain ordered.
- Request/RAG terminal ownership remains explicit on all response paths; unproven downstream
  no-response remains internal rather than being relabeled as disconnect.

The focused matrix exercised these branches, including 200 actual-ASGI disconnect races, promotion
submit failures, direct executor profiles, lifecycle observer failures, cleanup task-creation
fallback, repeated cancellation, STOPPED publication, exact-owner release, and immediate reacquire.

## Prior closure and approved simplifications

| Prior finding or decision | Iteration 5 status |
|---|---|
| M42-CR1-MAJ-001 / M42-CR2-MAJ-001 / M42-CR3-MAJ-001 / M42-CR4-MAJ-001 | **CLOSED** — immutable producer capabilities, reset/stale isolation, complete source maps, strict typed parsing, exact source algebra, retained genuine negative receipt, and executor-conflict evidence are green. |
| M42-CR1-MAJ-002 | **CLOSED** — early encoding/length rejection remains pre-read; overflow remains bounded at `limit+1`. |
| M42-CR1-MAJ-003 | **CLOSED** — FIFO promotion-submit failure retains progress, exact delivery, and conservation. |
| M42-CR1-MAJ-004 / M42-CR2-MAJ-002 | **CLOSED** — primary identity, mandatory cleanup tail, repeated-cancellation drain, STOPPED/release, and reacquire remain green. |
| M42-CR1-MIN-001 | **CLOSED** — executable conservation replaces the former tautology. |
| Approved pure-ASGI terminal-owner simplification | **INTACT** — proven frame-zero disconnect records the terminal without sending a synthetic 499 response; unproven no-response fails internal. |
| Approved single-active-lifespan simplification | **INTACT** — overlapping second lifespan fails before global/app mutation; immutable same-object sequential reacquire and canonical teardown remain enforced. |

## Requirement and acceptance traceability

| Contract | Iteration 5 result |
|---|---|
| REQ-001 settings | PASS — eight-field validation, generated inventory, identity policy, and full suite are green. |
| REQ-002 admission/FIFO | PASS — admission, direct-ticket, FIFO cancellation, timeout, and promotion-error paths are conserved. |
| REQ-003 timeout/cancellation | PASS — queued/running disconnect races and executor timeout/cancellation paths are deterministic. |
| REQ-004 orphan/conservation | PASS — exact cross-source algebra, typed epochs/counters, stale isolation, identity conflict evidence, and genuine retained negative control are fail-closed. |
| REQ-005 drain/shutdown | PASS — mandatory shutdown and canonical STOPPED/release tail survive observer, creation, and repeated-cancellation failures. |
| REQ-006 readiness/metrics | PASS — saturation/readiness and bounded exact metrics remain green. |
| REQ-007 input boundary | PASS — early rejection and bounded overflow remain green. |
| REQ-008 upstream deadlines | PASS, deterministic local scope. |
| REQ-009 compatibility/security | PASS — full suite and audits found no disclosure or cardinality regression. |
| Deterministic 10-profile acceptance | PASS — 100 profile results, 110 node receipts, 100 profile conservation rows, and 10 aggregate rows. |
| Genuine retained negative control | PASS — expected exit 1, rejected genuine receipt retained, diagnostic `conservation_mismatch`. |

## Independent receipts

| Command or probe | Exact result |
|---|---|
| Focused parser/executor/shutdown/disconnect/input suite | PASS: 57 passed in 90.53s. |
| `venv/bin/python -m pytest -q` | PASS: 1040 passed, 1 skipped, 1 pre-existing warning in 106.92s. |
| Deterministic runner, repeat 10, seed 4202 | PASS: 100 profile results, 110 node receipts, 100 profile conservation rows, 10 aggregate rows; `M4.1_BLOCKED=true`. |
| Injected genuine mismatch control | Expected exit 1; status FAIL, negative-control status FAIL, diagnostic `conservation_mismatch`, rejected receipt retained. |
| Exact CR4 mutation script | PASS: all seven probes rejected, including both nullable counters, conflict evidence, and disconnect-versus-internal. |
| `npm test` | PASS: 1 file, 9 tests. |
| Generated field-spec and logging-callsite audits | PASS. |
| Markdown link checker before this artifact | PASS: 96 files, 432 links, 0 failures. |
| `venv/bin/python -m compileall -q src scripts tests` | PASS. |
| `bash scripts/compile_lock.sh --verify` | PASS: two independent 102-package resolutions, reproducible with no drift. |
| `git diff --check` before this artifact | PASS. |

Acceptance artifacts were directed to `/tmp`; no production, test, or existing report file was
edited by this review. No live Ollama/network, M3 live/14-gate, M4.1 operational, commit, push, PR,
merge, email, or release action was performed. `M4.1_BLOCKED` remains a separate M4 release blocker.

## Decision

**Gate PASS.** The one permitted conditional Code Review Iteration 5 closes the recurring
receipt-validation root rather than reproducing it. M4.2 satisfies its deterministic local safe
serving boundary and may proceed to the coordinator's next approved milestone action; separate live,
M3, and M4.1 gates remain outside this decision.
