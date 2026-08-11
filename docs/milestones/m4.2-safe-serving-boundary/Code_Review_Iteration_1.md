# M4.2 Safe Serving Boundary — Code Review Iteration 1

Date: 2026-08-11 (Asia/Seoul)  
Base revision: `0c84795`  
Review scope: complete working-tree delta from the base, all M4.2 production modules, tests,
acceptance runner, dependency lock, and relevant unchanged request/config/network seams.

## Gate

**FAIL — 6.8/10**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 4 |
| MINOR | 1 |

The required PASS threshold is at least 9.7/10 with CRITICAL=0 and MAJOR=0. The ordinary test
suite is green, but the deterministic acceptance receipt is manufactured rather than observed,
two request terminals escape conservation/observability, an injected promotion failure strands
FIFO work, and lifecycle cleanup does not implement the approved mandatory-tail failure boundaries.

## Findings

### M42-CR1-MAJ-001 — Acceptance conservation is synthesized from pytest exit codes

`scripts/run_m42_acceptance.py:73-85` converts the number of selected pytest nodes into an invented
all-success transaction: every passing node becomes one request, one accepted submit, and one
completed ticket. `scripts/run_m42_acceptance.py:142-178` then validates and publishes those invented
values. It never harvests request terminals, executor snapshots, race counters, or conservation
receipts from the application under test. A test can exercise queue timeout, cancellation, overload,
and drain while the runner reports all of them as `success/completed`. The aggregate rows are created
from another synthetic tuple at lines 165-173. This directly contradicts the approved Requirement
§5 fail-closed receipt contract and makes the Implementation Report's “100 nonzero per-profile
conservation rows” claim non-evidentiary.

Exact fix: define a machine-readable receipt fixture/plugin owned by each catalog node (or profile),
emit actual request terminal counts plus the before/after `ExecutorSnapshot` deltas, and have the
runner parse and validate those values. Reject absent, duplicate, malformed, zero-work, or internally
inconsistent node receipts. Aggregate only observed rows. The negative control must corrupt an
otherwise genuine harvested field and prove validation catches it; it must not construct a separate
hard-coded FAIL document.

### M42-CR1-MAJ-002 — Pre-body encoding/length rejections bypass request and input observability

`src/simple_qna_rag/web/server.py:521-523` installs `BodyLimitMiddleware` outermost. Its early return
paths at `src/simple_qna_rag/web/body_limit.py:44-56` send 400/413 before
`RequestContextMiddleware` or `RagServingMiddleware` runs. Those paths therefore produce neither the
single request start/end/duration/counter nor `rag_input_rejected_total`. An independent actual-app
probe using `Content-Encoding: gzip` returned status 400 with `request_metrics=0` and
`input_rejected_metrics=0`. This breaks request-terminal equality, observability exactly-once, and
REQ-007 rejection accounting at a security-sensitive boundary.

Exact fix: keep the body limiter outside the body consumer but put the request-context owner outside
the limiter, and give the limiter a bounded observer hook for its own early terminal reason. Verify
actual-app identity/non-identity encoding, invalid/negative/oversized Content-Length, chunk overflow,
and disconnect cases each produce exactly one request terminal, request start/end/duration/counter,
and appropriate bounded input-rejection label without reading forbidden bodies.

### M42-CR1-MAJ-003 — Promotion submit failure strands the remaining FIFO and leaks finalized tickets

When `_pool.submit()` fails during promotion, `src/simple_qna_rag/web/concurrency.py:202-213` rejects
only the popped head, leaves that finalized ticket in `_tickets`, and returns `None` without trying the
next queued ticket. No later resource completion is guaranteed to retry promotion. An independent
injected failure after a 1-running/2-queued state produced
`queued=1, running=0, terminal_rejected=1, second_done=True, third_done=False, tickets=2`. Thus an
accepted caller can hang until its unrelated queue timer, FIFO capacity is left idle, and internal
ticket storage is not conserved.

Exact fix: make promotion a loop that removes/finalizes every failed head exactly once, removes each
terminal ticket from `_tickets`, delivers each `SubmitFailed`, and continues until one ticket starts
or the queue is empty. Register the successful future outside the lock. Add adversarial tests for one
and repeated promotion failures, cancellation/timeout adjacent to failure, no idle slot with queued
work, exact FIFO, terminal/resource conservation, and zero retained tickets/timers.

### M42-CR1-MAJ-004 — Lifecycle canonical tail is bypassable at task creation and repeated cancellation

The approved design requires every lifecycle-owner attempt to reach mandatory nonblocking shutdown,
atomic STOPPED publication, and exact-owner release even when teardown argument evaluation, task
creation, shielding, or cancellation fails. In `src/simple_qna_rag/web/server.py:363-378`, however,
`asyncio.create_task(...)` is outside a protective cleanup boundary; a creation failure skips
shutdown, STOPPED, and release. After the first shield cancellation is caught, the second shield await
at line 374 is unprotected, so a second cancellation also escapes before the tail. The current tests
only call `_teardown_lifecycle_owner` directly (`tests/unit/test_shutdown_drain.py:89-113`) and do not
exercise the actual lifespan dispatcher, task-creation/evaluation failures, primary exception
identity, repeated cancellation, identity mismatch zero-delta, invalid-loader publication count, or
concurrent immediate reacquire. Consequently the Implementation Report's claimed complete lifecycle
matrix is not present in the reviewed tree.

Exact fix: implement one non-skippable attempt-class cleanup dispatcher with an inline fallback when
task construction/creation fails and a cancellation-draining loop that awaits the teardown task to
completion before re-propagating the original cancellation identity. Keep mandatory shutdown and the
nonthrowing `STOPPED -> release_exact_owner` tail in a `finally` that cannot be bypassed by observers.
Add actual lifespan tests for every matrix entry named in Design Recovery Review Iteration 4,
including double cancellation and fresh-subprocess mutation spies for mismatch and invalid loader.

### M42-CR1-MIN-001 — A claimed conservation test is an unconditional tautology

`tests/unit/test_query_executor.py:106-108` ends its assertion with `or True`; it cannot fail and
executes no terminal origin. This obscures the missing executor algebra proof and helped the synthetic
runner appear complete.

Exact fix: replace it with scenario-driven assertions covering completed, queue timeout, execution
timeout, queued/running cancellation, drain rejection, admission rejection, and submit failure. At
every linearization point assert both approved equations from actual snapshot deltas and assert no
unknown state or retained finalized ticket.

## Requirement and acceptance closure

| Contract | Review result | Evidence / blocker |
|---|---|---|
| REQ-001 settings | Provisionally closed | Eight fields and validation are present; full suite passes inventory/facade tests. |
| REQ-002 admission/FIFO | **Open** | M42-CR1-MAJ-003; promotion failure violates progress and conservation. |
| REQ-003 timeouts/cancellation | Provisionally closed | Actual-app four-order race tests execute, but their acceptance conservation is not harvested. |
| REQ-004 orphan/conservation | **Open** | M42-CR1-MAJ-001/003 and tautological test. |
| REQ-005 drain/shutdown | **Open** | M42-CR1-MAJ-004; mandatory tail is bypassable. |
| REQ-006 readiness/metrics | **Open** | Early body terminals escape bounded metrics and request accounting. |
| REQ-007 input boundary | **Open** | Byte limiting is bounded, but early encoding/length terminals bypass observers. |
| REQ-008 upstream deadlines | Provisionally closed | Request-owned Ollama clients use remaining budgets; DDGS uses remaining timeout. Live behavior remains unrun. |
| REQ-009 compatibility/security | Partially closed | Full legacy suite is green; early security-boundary observability and lifecycle proof remain open. |
| Deterministic 10-profile acceptance | **Not accepted** | Tests run, but runner receipts are synthetic rather than production observations. |
| Injected negative control | **Not accepted** | It writes a predeclared FAIL artifact instead of corrupting a genuine receipt. |

## Independent command receipts

| Command / probe | Result |
|---|---|
| `venv/bin/python -m pytest -q` | PASS: 1005 passed, 1 skipped, 1 warning, 23.53s. |
| Promotion-failure adversarial probe (`/tmp/m42_adversarial.py`) | Reproduced stranded FIFO and retained tickets: `queued=1`, `running=0`, `third_done=False`, `tickets=2`. |
| Actual-app non-identity encoding probe (`/tmp/m42_early_reject.py`) | Reproduced status 400 with zero request metrics and zero input-rejection metrics. |
| `venv/bin/python scripts/check_markdown_links.py` (pre-artifact) | PASS: 92 files, 415 links, 0 failures. |
| `git diff --check` (pre-artifact) | PASS. |

The clean locked-environment and deterministic-runner receipts in Implementation Report were
reviewed as claims but were not treated as independent proof. The ordinary full suite was rerun;
lock compilation was inspected through its tests and diff, but a second clean environment was not
created because the four deterministic correctness blockers already fail the code-review gate.

## Residual and live separation

Deterministic residuals are the four MAJOR findings and one MINOR finding above. They must be fixed
and independently re-reviewed before M4.2 can pass. The opt-in live 12-case Ollama/network run, M3
live/14-gate receipt, and M4.1 operational acceptance were not run in this review. They remain
separate external/live work; `M4.1_BLOCKED` remains an M4 release blocker and is not superseded by
the green local suite.

## Decision

**Gate FAIL.** Do not proceed to release/Git publication. Return to implementation recovery, then
run a fresh code-review iteration against real application-derived receipts.
