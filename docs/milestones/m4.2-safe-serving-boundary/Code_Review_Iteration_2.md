# M4.2 Safe Serving Boundary — Code Review Iteration 2

Date: 2026-08-11 (Asia/Seoul)  
Base revision: `0c84795`  
Review scope: complete current working-tree delta, approved Requirement/Plan/Design/recovery PASS,
Implementation Report Iteration 4, production modules, tests, deterministic runner, and dependency
lock.

## Gate

**FAIL — 8.6/10**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 2 |
| MINOR | 0 |

The required threshold is at least 9.7/10 with no CRITICAL or MAJOR findings. The full Python and
Node suites, lock verification, and the nominal deterministic runner are green. The gate remains
closed because catalog receipts still accept test-authored terminal dictionaries as observations,
and the lifecycle dispatcher discards a required task-creation cleanup error.

## Iteration 1 closure matrix

| Prior finding | Iteration 2 status | Evidence |
|---|---|---|
| M42-CR1-MAJ-001 | **OPEN, narrowed** | Per-node files and real `ExecutorSnapshot` deltas replaced exit-code synthesis, but request terminals remain arbitrary arguments to `M42Receipt.record`; see M42-CR2-MAJ-001. |
| M42-CR1-MAJ-002 | **CLOSED** | Request context is outermost; actual-ASGI gzip, malformed/negative/oversized length probes recorded one start/end/counter/duration and one bounded input rejection with zero receive calls. Chunk overflow stopped at `limit+1`. |
| M42-CR1-MAJ-003 | **CLOSED** | One- and two-head promotion-submit failures continued to the next FIFO ticket, returned `SubmitFailed` once per failed head, and ended at queued/running/orphaned/tickets `(0,0,0,0)`. |
| M42-CR1-MAJ-004 | **OPEN, narrowed** | Mandatory tail, repeated cancellation, and reacquire work in direct dispatcher tests; task-creation error identity/secondary propagation remains incorrect. See M42-CR2-MAJ-002. |
| M42-CR1-MIN-001 | **CLOSED** | The `or True` tautology is gone and the scenario matrix exercises completed, queued/running cancellation, both timeouts, drain, admission rejection, and submit failure. |

## Findings

### M42-CR2-MAJ-001 — Runtime terminal receipts are still assertions supplied by each test

[`tests/conftest.py:18`](../../../tests/conftest.py) accepts an arbitrary
`request_terminals` dictionary from the test and serializes it unchanged at lines 19-23. Every
catalog node calls this API with hand-entered counts, for example
[`tests/unit/test_query_executor.py:45`](../../../tests/unit/test_query_executor.py) and
[`tests/integration/test_web_input_boundary.py:107`](../../../tests/integration/test_web_input_boundary.py).
Those counts are not harvested from request-context terminals, response/disconnect frames, or a
production-owned terminal ledger. An adversarial receipt with no application execution, fabricated
`success=1`, and fabricated matching snapshots was accepted by both `harvest_node_receipt()` and
`validate_conservation()` with `unknown=0`.

The negative control also does not corrupt a genuine node row. At
[`scripts/run_m42_acceptance.py:198`](../../../scripts/run_m42_acceptance.py), node values are summed
into a profile dictionary; lines 199-201 then decrement that derived aggregate and set `unknown=1`.
The genuine row preserved in `node_results[].runtime_receipt` is never corrupted. In addition,
malformed snapshot keys/types can raise uncaught `KeyError`/`TypeError` because lines 100-117 assume
the schema and the runner catches only `ValueError` at lines 183-187, so malformed input can exit
without the required machine-readable FAIL artifact.

Exact fix: make the receipt fixture harvest a production-owned terminal ledger and immutable
before/after executor snapshots; do not accept terminal totals from the test body. Validate the
complete typed snapshot schema, invariants, deltas, node identity, exactly-one row, nonzero work, and
conservation under one normalized fail-closed exception boundary. For the negative control, copy
one successfully harvested node row, corrupt one observed field in that row, pass it through the
same parser/validator, and retain the rejected corrupted row plus diagnostic in the FAIL artifact.
Add isolated tests for missing, malformed JSON, missing/type-invalid fields, duplicate, zero-work,
terminal/snapshot mismatch, and genuine-row corruption.

### M42-CR2-MAJ-002 — Successful inline fallback suppresses task-creation failure

The approved Design §4.3.2 requires task-creation failure to be recorded as the first cleanup
secondary before inline fallback. In
[`src/simple_qna_rag/web/server.py:391`](../../../src/simple_qna_rag/web/server.py), the dispatcher
captures `creation_error`, but lines 395-396 return only the inline teardown's error list when that
fallback succeeds. The creation error is included only when inline teardown itself raises at line
398. The checked-in test at
[`tests/unit/test_shutdown_drain.py:122`](../../../tests/unit/test_shutdown_drain.py) incorrectly
codifies this loss by asserting `errors == []` after injected `asyncio.create_task()` failure.

An independent probe injected `RuntimeError("create-task-primary")`: begin, wait, mandatory
nonwaiting shutdown, snapshot, `STOPPED`, release, and immediate reacquire all succeeded, but the
returned receipt was exactly `errors=[]; cancel=None`. Thus the canonical tail is now safe, but
primary/ordered-secondary identity and propagation are not. The tests still exercise helpers rather
than the actual lifespan dispatcher, leaving constructor/body primary precedence and publication
counts unproven on the production entry point.

Exact fix: return `[creation_error, *inline_errors]` after every inline fallback, then let lifespan
preserve an existing primary or propagate the single/ordered cleanup error according to Design.
Replace the contrary assertion and add actual `app.router.lifespan_context()`/fresh-process tests for
task creation failure with and without a body primary, repeated cancellation identity, settings
identity mismatch zero-delta, invalid-loader single publication, mandatory tail ordering, and
immediate reacquire.

## Requirement and acceptance status

| Contract | Status | Review evidence |
|---|---|---|
| REQ-001 settings | PASS | Eight fields, fail-closed validation, generated inventory, and full settings suite pass. |
| REQ-002 admission/FIFO | PASS | Promotion-failure progress and cleanup reproduced; bounded/FIFO tests pass. |
| REQ-003 timeout/cancellation | PASS, deterministic | Actual-ASGI queued/running race nodes and executor scenario tests pass; live behavior remains separate. |
| REQ-004 orphan/conservation | **OPEN** | Executor algebra improved, but M42-CR2-MAJ-001 prevents acceptance of request-terminal conservation. |
| REQ-005 drain/shutdown | **OPEN** | Mandatory tail works, but M42-CR2-MAJ-002 violates the approved error identity/secondary contract. |
| REQ-006 readiness/metrics | PASS | Saturation/readiness and exact early-terminal metrics tests pass with bounded labels. |
| REQ-007 input boundary | PASS | Pre-read encoding/length rejection and bounded chunk overflow are directly observed. |
| REQ-008 upstream deadlines | PASS, deterministic | Context deadline and remaining-budget paths are covered; live upstream behavior was not run. |
| REQ-009 compatibility/security | PASS, deterministic | Full legacy suite passes; safe error/body and fixed-cardinality metrics remain intact. |
| Deterministic 10-profile acceptance | **Not accepted** | Nominal `100/110/100/10` runner document is green, but terminal provenance is test-authored. |
| Injected negative control | **Not accepted** | It corrupts a derived profile aggregate rather than a genuine harvested node row. |

## Independent receipts

| Command or probe | Exact result |
|---|---|
| Focused executor/lifecycle/input/disconnect/runner suite | PASS: 32 passed in 6.28s. |
| `venv/bin/python -m pytest -q` | PASS: 1015 passed, 1 skipped, 1 warning in 28.30s. |
| `npm test` | PASS: 1 file, 9 tests. |
| nominal deterministic runner, repeat 10, seed 4202 | Process PASS; artifact contained status PASS, 100 profile results, 110 node results, 100 profile rows, and 10 aggregate rows. Temporary artifact removed after inspection. |
| checked-in mismatch-control test | PASS as an expected negative: runner returned 1 and produced status FAIL with first aggregate `unknown=1`. |
| fabricated-runtime-row adversarial probe | **Accepted incorrectly:** `request_count=1`, `accepted_lhs=accepted_rhs=1`, `submit_attempt_lhs=submit_attempt_rhs=1`, `unknown=0`. |
| cleanup task-create failure probe | Tail completed and reacquire returned `released`, but dispatcher returned `errors=[]`, losing `RuntimeError("create-task-primary")`. |
| `bash scripts/compile_lock.sh --verify` | PASS: two independent 102-package resolutions, no drift. |
| generated field spec and logging callsite checks | PASS. |
| Markdown link checker | PASS after this artifact: see final validation below. |
| `git diff --check` | PASS after this artifact: see final validation below. |

## Security, concurrency, and residual/live separation

No new data disclosure, unbounded label, slot leak, FIFO stall, timer/ticket retention, or unread-body
regression was found. Early rejections remain pre-read and exactly observable; promotion work is
performed under one state lock with delivery/callback registration outside it; cancellation and
orphan resource ownership remain bounded. Upstream connect/remaining-budget plumbing and lock
reproducibility are deterministic-local PASS.

The two MAJOR findings are deterministic code/proof residuals and must be closed in a fresh review.
The opt-in live 12-case Ollama/network run, M3 live/14-gate receipt, and M4.1 operational acceptance
were not run. `M4.1_BLOCKED` remains a separate M4 release blocker and is not changed by local M4.2
results.

## Decision

**Gate FAIL.** Do not publish or release M4.2. Return the receipt provenance and lifecycle
task-creation error policy to implementation recovery, then run a fresh independent code review.
