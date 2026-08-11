# M4.2 Safe Serving Boundary — Implementation Report (Iteration 7)

Date: 2026-08-11 (Asia/Seoul)

## Outcome

Implementation Iteration 7 **PASSES the approved deterministic proof contract** and closes the single
bounded conditional-recovery finding M42-CR4-MAJ-001 while preserving every earlier closure. Every
request, RAG, and executor producer now captures an immutable ledger epoch/node token before work;
late publications are retained only in fixed-cardinality stale maps and cannot enter the active node.
Receipts serialize all three source maps without selection, and before/after snapshots carry one
immutable executor identity, the matching ledger epoch, and a retained executor identity/epoch
conflict count that must be zero. The strict JSON-standard typed parser
rejects non-finite numbers, bool-as-int, schema drift, invalid lifecycle values, negative counters,
identity/epoch disagreement, source reason/count disagreement, and snapshot algebra disagreement.

The checked-in
actual application proof executes 100 queued and 100 running races, evenly covering
disconnect-first, result-first, tie-disconnect-inserted-first, and tie-result-inserted-first. Every
iteration proves the per-order winner, zero disconnect frames or one complete result response,
single receive ownership, exactly one request start/end/duration/counter and executor terminal,
single ticket outcome, zero pending race children, and final queued/running/orphaned `(0,0,0)`.

The teardown proof now covers executor absent/present; cleanup task-creation inline fallback;
repeated shield-cancellation draining; begin, applicable wait, mandatory shutdown,
snapshot observer single/combined failures; grace expiry; cancellation identity; exact ordered
secondary aggregation; canonical `STOPPED -> exact-owner release`; and immediate reacquire. Bounded
failing log and metric spies prove observer failures cannot change response, executor accounting, or
the teardown tail. The deterministic runner invokes each catalog node separately and harvests its
required `m42-node-runtime-receipt-v1` file from a production-owned fixed-cardinality terminal ledger.
Request context, RagServing, and QueryExecutor transitions author observations; the fixture can only
reset/read the ledger and serialize immutable snapshots bound to the exact pytest node. The complete
typed parser rejects absent, duplicate, malformed, missing/type-invalid, zero-work, identity,
invariant, and conservation failures under one machine-readable FAIL boundary. The negative control
deep-copies one successfully harvested genuine row, corrupts that row, runs the identical
parser/validator, and retains the rejected receipt and diagnostic.

Lifecycle cleanup now always returns `[creation_error, *inline_errors]` after task-construction
fallback. The actual `app.router.lifespan_context()` path propagates a lone creation error, preserves
an existing body primary by identity while publishing ordered bounded cleanup-secondary diagnostics,
completes mandatory shutdown/STOPPED/exact-owner release, and permits immediate reacquisition.

`M4.1_BLOCKED` remains separate and unchanged. Opt-in live Ollama/network, M3 live/14-gate, and
M4.1 operational acceptance were not run.

## Iteration 7 conditional recovery changes

- Replaced the permissive executor residual with node-aware positional source algebra. Every
  application/RAG reason must consume its matching executor reason; response-side disconnect is the
  sole explicit relation that may consume executor cancellation or executor success when completion
  won before the disconnect was observed. The exact per-reason executor remainder is retained as the
  direct control-ticket stream, and the whole executor map is still reconciled against immutable
  admission and terminal counter deltas. Application success can no longer coexist with executor
  internal.
- Required exact `int` (not `bool`) for row `ledger_epoch` and `snapshot_epoch`. Required every
  integer-or-null stopped snapshot counter to be nonnegative when present.
- Added fixed-cardinality `executor_identity_conflicts` state to the production ledger snapshot,
  serialized it in every node receipt, and required exact integer zero in the parser. A zero-work
  observation by a second executor is therefore retained and rejected instead of being silently
  ignored.
- Added exact adversarial tests for request/RAG success versus executor internal, boolean row and
  snapshot epochs, negative nullable `stopped_with_running` and `stopped_with_orphaned`, nonzero
  executor identity conflict evidence, and zero-work second-executor overlap. The genuine harvested
  receipt negative control remains retained with the exact rejection diagnostic.

## Iteration 6 final base code-review closure changes

- Replaced resettable unscoped recorder calls with immutable `LedgerProducer` capabilities containing
  epoch, opaque node token, and node identity; reset increments the epoch and stale callbacks publish
  only into separate bounded stale maps.
- Bound every `QueryExecutor` to a generated immutable executor identity and captured producer;
  snapshot observation rejects overlapping identities and mismatched epochs.
- Serialized separate fixed-cardinality request, RAG, executor, and stale maps, plus outer/row epoch,
  node token, executor identity, and snapshot epoch. No stream is selected, substituted, or dropped.
- Added source-owned executor reason/counter algebra and exact request/RAG mapping. Direct profile
  control tickets are validated against executor state-machine deltas without being misattributed to
  the smaller ASGI request stream.
- Made receipt parsing exact-key, exact-type, finite-number, enum/lifecycle, nonnegative, invariant,
  identity, and epoch strict; JSON `NaN` and infinities fail at load time under normalized FAIL.
- Added adversarial reset/stale, overlapping producer identity, source reason/count, snapshot identity,
  bool-as-int, nonfinite, and retained genuine mismatch tests and diagnostics.

## Iteration 5 code-review closure changes

- Added `observability/terminal_ledger.py`, a thread-safe bounded production instrumentation seam with
  immutable snapshots and fixed terminal vocabulary; no test API accepts terminal dictionaries.
- Instrumented request-context completion, explicit RagServing terminal decisions, every executor
  admission/terminal transition, and executor snapshot capture.
- Normalized the complete receipt schema and all parser/validator exceptions to fail-closed
  diagnostics with exact outer/row node identity and exactly-one-row enforcement.
- Reworked the negative control to corrupt and retain a deep-copied genuine node receipt.
- Fixed cleanup task-creation error ordering and added actual lifespan tests for no-primary and
  body-primary propagation, canonical tail completion, secondary ordering, and reacquisition.

## Iteration 4 code-review closure changes

- Reordered the ASGI stack to request context -> body limiter -> body consumer. Encoding and
  Content-Length rejection stays pre-read while recording one bounded rejection reason and exactly
  one request start/end/counter/duration.
- Promotion loops through failed FIFO heads, cancels timers, removes/finalizes each ticket, delivers
  each `SubmitFailed` once outside the lock, and continues until a future starts or the queue empties.
- Added a non-skippable lifecycle cleanup dispatcher with task-creation inline fallback and repeated
  cancellation draining while retaining one cleanup owner and the original primary identity.
- Replaced exit-code conservation synthesis and the predeclared negative artifact with per-node
  runtime receipt harvesting and corruption of an observed field.
- Replaced the unconditional conservation tautology and added adversarial production-path tests.

## Prior implementation changes

- Added `web/scheduling.py` with production and manual absolute-deadline schedulers; drain waiters
  use one lock/sequence/CAS winner and cancel their deadline handle on every exit.
- Added the actual-ASGI `RagServingMiddleware`: it consumes the complete limited body before a
  single disconnect observer takes receive ownership, races that observer against the ticket
  result, reaps both children, cancels the exact ticket once, and sends no disconnect response.
- `BodyLimitMiddleware` selects `app.state.settings.MAX_REQUEST_BODY_BYTES` after startup, rejects
  non-identity content encoding before receive, caps downstream consumption at `limit + 1`, stops
  further consumption after overflow, and records wire/consumed/receive observations separately.
- Lifespan loading keeps the candidate local until process-identity verification. Identity mismatch
  is release-only; invalid Settings is a single fail-soft publication; lifecycle owners perform
  drain/wait/mandatory shutdown and finish with nonthrowing `STOPPED -> release_exact_owner`.
- Replaced the profile paths with the approved immutable ordered inventory, added ordered
  collect-only validation, 10 profile and per-node repetitions, a separate injected mismatch
  negative control, and distinct live/M3/M4.1 status fields.
- Recompiled `requirements.lock` only through `bash scripts/compile_lock.sh`; the authoritative
  `requirements.txt` now resolves reproducibly to 102 packages. Updated the two package-count
  assertions and verified a newly created locked environment independently of the contaminated
  project venv.
- Added deterministic claim control points that only activate when an acceptance controller is
  explicitly installed on app state; normal production scheduling remains unchanged.
- Expanded actual-app race proof to 25 repetitions per ordering in each queued/running node and
  added bounded log/metric failure spies.
- Expanded lifecycle proof into the complete executor/error/cancellation/observer table with
  mandatory shutdown, exact tail, and immediate reacquire assertions.
- Replaced placeholder conservation rows with 100 nonzero per-profile rows and 10 nonzero aggregate
  rows; the acceptance validator rejects incomplete, zero, unknown, or nonzero-exit receipts.

## Exact verification receipts

| Command | Result |
|---|---|
| focused Iteration 7 parser/ledger/executor proof | PASS: 33 tests; acceptance parser adversarial proof separately PASS: 25 tests |
| `venv/bin/python -m pytest -q` | PASS: 1040 passed, 1 skipped, 1 pre-existing warning in 113.52s |
| clean locked venv `pip check`; full suite | PASS: no broken requirements; 1040 passed, 1 skipped, 2 pre-existing warnings in 145.21s |
| deterministic runner: repeat 10, seed 4202 | PASS: 100 profile results, 110 node receipts, 100 profile conservation rows, 10 aggregate rows |
| injected genuine mismatch control | expected exit 1; 100/110/100/10 complete evidence retained; rejected genuine receipt diagnostic `conservation_mismatch` |
| exact Iteration 7 adversarial controls | PASS: request success/executor internal, boolean row/snapshot epoch, both negative nullable stopped counters, conflict count, and zero-work second-executor overlap all rejected |
| Node, generated-field, logging, links, compile, lock, diff | PASS: 9 Node tests; field/logging audits; 96 files/432 links; compileall; two reproducible 102-package locks; clean diff |
| focused Iteration 6 ledger/parser/source proof | PASS: 30 focused tests initially; 13 acceptance-runner adversarial tests after final negative-control adjustment |
| `venv/bin/python -m pytest -q` | PASS: 1033 passed, 1 skipped; the first verification run exposed only the subsequently corrected negative-control corruption field |
| clean locked venv full suite | PASS: `pip check` clean; 1033 passed, 1 skipped, 2 pre-existing warnings in 151.65s |
| deterministic runner: repeat 10, seed 4202 | PASS: 10 profiles, 100 profile results, 110 identity/epoch-bound genuine node receipts, 100 nonzero profile rows, 10 aggregate rows |
| injected genuine mismatch control | expected exit 1; genuine row retained; queued-state corruption rejected as `conservation_mismatch` |
| ledger/parser negative controls | PASS: old publication after reset isolated in stale map; overlapping executor identity rejected; source reason mismatch, snapshot epoch mismatch, NaN, positive/negative infinity, bool-as-int, and malformed structures rejected |
| focused Iteration 5 proof (lifecycle/parser plus fresh-process lifespan) | PASS: 22 selected tests, including the fresh-process primary/secondary probe |
| `venv/bin/python -m pytest -q` | PASS: 1023 passed, 1 skipped, 1 pre-existing pytest warning, 114.76s; the subsequently added isolated fresh-process test passed separately |
| clean venv `python -m pip check` | PASS: `No broken requirements found.` |
| clean locked venv `pip install -r requirements.lock`; `pip install --no-deps -e .`; `pip check`; `python -m pytest -q` | PASS: no broken requirements; 1015 passed, 1 skipped, 2 warnings, 27.47s |
| `npm test` | PASS: 1 file, 9 tests |
| deterministic runner: repeat 10, seed 4202 | PASS: 10 profiles, 100 profile results, 110 genuine node runtime receipts, 100 nonzero profile conservation rows, 10 observed aggregate rows |
| injected conservation mismatch control | expected exit 1; deep-copied genuine event-loop row retained after `completed_total` corruption; identical parser rejected it with `conservation_mismatch` |
| `venv/bin/python scripts/generate_field_spec.py --check` | PASS |
| `venv/bin/python scripts/logging_callsite_audit.py --check` | PASS |
| `venv/bin/python scripts/check_markdown_links.py` | PASS: 94 files, 421 links, 0 failures |
| `venv/bin/python -m compileall -q src scripts tests` | PASS |
| `bash scripts/compile_lock.sh --verify` | PASS: two 102-package resolutions identical; committed drift 0 |
| `git diff --check` | PASS |
| project `venv/bin/python -m pip check` | FAIL only from previously installed out-of-lock `langgraph-prebuilt`/`langchain-classic`; clean locked environment proves repository correctness |
| live 12 / M3 live / M4.1 operational | NOT RUN: no explicit safe opt-in configuration; `M4.1_BLOCKED` retained |

## Remaining work

No deterministic M4.2 proof residual remains. Live 12-case Ollama/network, M3 live/14-gate, and
M4.1 operational acceptance remain deliberately unexecuted because this iteration had no explicit
safe opt-in configuration; `M4.1_BLOCKED` remains separate and unchanged.

No commit, push, pull request, email, live network test, M3 run, or M4.1 run was performed.
