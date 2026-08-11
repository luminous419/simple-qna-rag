# M4.2 Safe Serving Boundary — Code Review Iteration 4

Date: 2026-08-11 (Asia/Seoul)  
Base revision: `0c84795`  
Review scope: complete current uncommitted delta, approved Requirement/Plan/Design and recovery
reviews, Code Review Iterations 1–3, Implementation Report Iteration 6, milestone orchestration guide,
production serving and observability modules, tests, deterministic runner, and dependency lock.

## Gate

**FAIL — 9.2/10**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 1 |
| MINOR | 0 |

The required threshold is at least 9.7/10 with CRITICAL=0 and MAJOR=0. Immutable producer
capabilities, reset isolation, separate source maps, finite JSON parsing, genuine-row retention, and
all earlier serving/lifecycle corrections are present and independently green. The gate remains
closed because the supposedly exact typed receipt boundary still accepts contradictory production
terminal reasons, boolean epoch fields, negative nullable counters, and an unretained overlapping
executor identity. A nominal 10-repeat PASS document therefore cannot be release evidence.

## Prior-finding closure matrix

| Prior finding | Iteration 4 status | Evidence |
|---|---|---|
| M42-CR1-MAJ-001 / M42-CR2-MAJ-001 / M42-CR3-MAJ-001 | **OPEN, substantially narrowed; see M42-CR4-MAJ-001** | Production-owned immutable epoch/token bindings, fixed request/RAG/executor/stale maps, stale callback isolation, finite-number rejection, and genuine-row negative control are implemented. Exact cross-source/type/identity rejection remains fail-open. |
| M42-CR1-MAJ-002 | **CLOSED** | Focused actual-ASGI input tests pass; early encoding/length rejection remains pre-read and exactly observable, and chunk overflow remains bounded at `limit+1`. |
| M42-CR1-MAJ-003 | **CLOSED** | Focused executor tests pass the one-/two-head promotion-submit failure matrix with FIFO progress, exact error delivery, and no retained capacity/tickets. |
| M42-CR1-MAJ-004 / M42-CR2-MAJ-002 | **CLOSED** | Actual lifespan tests preserve body-primary identity, propagate task-create failure after the mandatory tail, drain repeated cancellation, publish STOPPED, release the exact owner, and immediately reacquire. |
| M42-CR1-MIN-001 | **CLOSED** | The executor scenario/conservation matrix remains executable and green; the tautology is absent. |

## Finding

### M42-CR4-MAJ-001 — Receipt validation is still not exact across source, type, and executor identity

The parser keeps all three terminal maps, but it selects request/RAG terminals whenever they are
nonzero and validates the executor only with a permissive aggregate equation
([`run_m42_acceptance.py:149`](../../../scripts/run_m42_acceptance.py)). In particular,
`executor.success <= completed` and the residual `internal` bucket at lines 176–179 allow a genuine
request/RAG `success=1` to coexist with executor `internal=1`. An independent mutation of the valid
unit receipt from executor `success=1` to `internal=1` was **accepted** with `request_count=1`,
`accepted=1`, `completed=1`, and `unknown=0`. This is the same contradictory production-source class
that Iteration 3 required the recovery to reject; the checked-in mismatch test only substitutes
`queue_timeout=1`, which happens to violate a narrower counter equality
([`test_m42_acceptance_runner.py:88`](../../../tests/unit/test_m42_acceptance_runner.py)).

The exact typed boundary is also incomplete. Row `ledger_epoch` and `snapshot_epoch` are compared by
value but never required to have exact integer type
([`run_m42_acceptance.py:104`](../../../scripts/run_m42_acceptance.py),
[`run_m42_acceptance.py:143`](../../../scripts/run_m42_acceptance.py)); Python therefore accepts
`True == 1`. Independent mutations setting either field to `true` both parsed successfully.
`stopped_with_running` and `stopped_with_orphaned` require integer-or-null but omit the nonnegative
check applied to the other snapshot counters at lines 132–139; `after.stopped_with_running=-1` also
parsed successfully.

Finally, overlapping executor observation returns `False` but leaves no mismatch bit/count in the
snapshot or serialized artifact
([`terminal_ledger.py:86`](../../../src/simple_qna_rag/observability/terminal_ledger.py)). The caller
ignores that return value, and the checked-in test proves only that the call returned false while the
receipt still contains the first executor identity
([`test_m42_acceptance_runner.py:107`](../../../tests/unit/test_m42_acceptance_runner.py)). Thus an
overlap with no additional terminal delta is indistinguishable from a valid single executor and can
be accepted instead of producing the required retained mismatch evidence.

Exact fix: define node-aware source algebra that proves every application request/RAG reason against
its executor terminal while separately accounting for direct control tickets; do not collapse either
stream into a residual `internal` allowance. Require `type(row.ledger_epoch) is int` and
`type(row.snapshot_epoch) is int`, require every integer-or-null counter to be nonnegative, and retain
a fixed-cardinality executor-identity/epoch conflict counter that the parser requires to be zero.
Add the exact four mutations above and an overlap-with-zero-work-on-the-second-executor receipt test;
retain the rejected genuine receipt and diagnostic for the cross-source mismatch.

## Requirement and acceptance traceability

| Contract | Iteration 4 result | Independent evidence |
|---|---|---|
| REQ-001 settings | PASS | Eight-field validation/inventory and the complete suite pass. |
| REQ-002 admission/FIFO | PASS | Focused executor and promotion-failure coverage pass. |
| REQ-003 timeout/cancellation | PASS, deterministic | Actual-ASGI queued/running disconnect races and executor timeout/cancellation tests pass. |
| REQ-004 orphan/conservation | **OPEN** | M42-CR4-MAJ-001 permits contradictory production sources and malformed identity/counter values. |
| REQ-005 drain/shutdown | PASS | Actual lifespan/fresh-process task-create propagation and mandatory-tail behavior remain closed. |
| REQ-006 readiness/metrics | PASS | Saturation/readiness and bounded exact request metrics remain green. |
| REQ-007 input boundary | PASS | Early rejection and bounded overflow focused tests pass. |
| REQ-008 upstream deadlines | PASS, deterministic | Deadline propagation is covered locally; live upstream behavior remains separate. |
| REQ-009 compatibility/security | PASS, deterministic | Full suite is green and no new disclosure/cardinality regression was found. |
| Deterministic 10-profile acceptance | **Not accepted as a gate** | Runner produced nominal PASS with 100 profile results, 110 node rows, 100 profile conservation rows, and 10 aggregates, but M42-CR4-MAJ-001 proves its parser fail-open. |
| Injected genuine mismatch control | PASS for its narrow claim | It corrupts a harvested row, exits 1, retains the rejected receipt, and reports `conservation_mismatch`; it does not exercise the accepted contradictions above. |

## Independent receipts

| Command or probe | Exact result |
|---|---|
| Focused runner/executor/disconnect/input/shutdown suite | PASS: 50 passed in 90.14s. |
| `venv/bin/python -m pytest -q` | PASS: 1033 passed, 1 skipped, 1 pre-existing warning in 115.72s. |
| Deterministic runner, repeat 10, seed 4202 | Nominal PASS: 100 profile results, 110 runtime node receipts, 100 profile conservation rows, 10 aggregate rows; `M4.1_BLOCKED=true`. |
| Request/RAG success versus executor internal probe | **Accepted incorrectly:** returned `request_count=1`, `accepted=1`, `completed=1`, `unknown=0`. |
| Row ledger epoch `true` probe | **Accepted incorrectly:** boolean compared equal to integer epoch 1. |
| Snapshot epoch `true` probe | **Accepted incorrectly:** boolean compared equal to integer epoch 1. |
| Negative nullable stopped counter probe | **Accepted incorrectly:** `after.stopped_with_running=-1`. |
| `npm test` | PASS: 1 file, 9 tests. |
| generated field-spec and logging-callsite checks | PASS. |
| `bash scripts/compile_lock.sh --verify` | PASS: two independent 102-package resolutions, reproducible with no drift. |
| `git diff --check` | PASS after this artifact. |
| Markdown link checker | PASS after this artifact. |

No live Ollama/network, M3 live/14-gate, M4.1 operational, commit, push, PR, merge, or email action was
performed. `M4.1_BLOCKED` remains a separate M4 release blocker and is not changed by this local
review.

## Iteration-limit decision

This is the fourth base code-review iteration. The milestone guide permits a conditional Iteration 5
because CRITICAL=0, score is at least 9.0, only one MAJOR remains, the result materially improves on
Iteration 3's 8.8, and the remaining parser/ledger changes are concrete and bounded. The conditional
extension stop conditions do not yet require stopping: no user decision is needed and the repair cost
is proportionate. Because the same receipt-validation root class has now survived Iterations 3 and 4,
Iteration 5 must close it completely; another recurrence during the conditional extension triggers
the guide's early-stop rule rather than Iteration 6.

## Decision

**Gate FAIL.** Do not publish or release M4.2. Perform one bounded conditional recovery for
M42-CR4-MAJ-001, then run independent Code Review Iteration 5 against the adversarial mutations above;
PASS still requires score at least 9.7, CRITICAL=0, MAJOR=0, and minimized MINOR findings.
