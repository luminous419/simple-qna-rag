# M4.2 Safe Serving Boundary — Code Review Iteration 3

Date: 2026-08-11 (Asia/Seoul)  
Base revision: `0c84795`  
Review scope: complete current working-tree delta, approved Requirement/Plan/Design/recovery PASS,
Implementation Report final state, both prior code reviews, production modules, tests, deterministic
runner, and dependency lock.

## Gate

**FAIL — 8.8/10**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 1 |
| MINOR | 0 |

The required threshold is at least 9.7/10 with CRITICAL=0 and MAJOR=0. Lifecycle task-create
failure is now correctly propagated after the mandatory tail, all earlier serving-path defects remain
closed, and the full suite is green. The gate remains closed because the terminal receipt discards
one production provenance stream, permits cross-reset callback contamination, and accepts malformed
non-finite snapshots; consequently the nominal deterministic acceptance result is not trustworthy.

## Prior-finding closure matrix

| Prior finding | Iteration 3 status | Evidence |
|---|---|---|
| M42-CR1-MAJ-001 / M42-CR2-MAJ-001 | **OPEN, narrowed; see M42-CR3-MAJ-001** | Tests can no longer pass terminal dictionaries, and the negative control retains a corrupted genuine row, but request/executor provenance is collapsed and reset has no generation isolation. |
| M42-CR1-MAJ-002 | **CLOSED** | Focused actual-ASGI encoding/length/overflow tests pass; pre-read rejections retain exactly-one request and bounded input-rejection observations. |
| M42-CR1-MAJ-003 | **CLOSED** | The one-/two-head injected promotion-failure matrix passes with continued FIFO progress, no idle queued capacity, and zero retained tickets. |
| M42-CR1-MAJ-004 / M42-CR2-MAJ-002 | **CLOSED** | Actual lifespan and fresh-process probes cover task-create failure with/without body primary, ordered cleanup-secondary identity, mandatory shutdown/STOPPED/release, double cancellation, and immediate reacquire. |
| M42-CR1-MIN-001 | **CLOSED** | The tautology remains removed and the executor origin/conservation scenario matrix is executable and green. |

## Finding

### M42-CR3-MAJ-001 — Terminal provenance is lossy and not reset-isolated

The production ledger separately records request and executor terminals
([`terminal_ledger.py:16`](../../../src/simple_qna_rag/observability/terminal_ledger.py)), but the
receipt fixture selects request totals whenever any request count is nonzero and otherwise substitutes
executor totals ([`tests/conftest.py:20`](../../../tests/conftest.py)). It serializes the selected map
under `request_terminals` and drops the other production stream. The parser therefore cannot prove
that request/RAG/executor terminal observations agree exactly once. An independent production-ledger
probe recorded `request.success=1` and `executor.internal=1`; after the fixture's selection behavior,
the row parsed successfully with `unknown=0` and conservation equalities satisfied.

Reset isolation is also absent. `reset()` replaces shared dictionaries, while recorders carry no
node/generation token ([`terminal_ledger.py:32`](../../../src/simple_qna_rag/observability/terminal_ledger.py),
[`terminal_ledger.py:40`](../../../src/simple_qna_rag/observability/terminal_ledger.py)). An independently
started executor was reset from node `old` to node `new` before its blocked pool future completed;
releasing it caused that old callback to publish `executor.success=1` into `new`. The lock prevents a
data race but not cross-node attribution, so concurrent or delayed callbacks can author the wrong
terminal row.

The claimed strict typed parser is additionally incomplete: `capacity_edge_at` accepts any `int` or
`float` without a finiteness check ([`run_m42_acceptance.py:107`](../../../scripts/run_m42_acceptance.py)).
A receipt containing `NaN` parsed successfully. Python's JSON reader/writer accepts this non-standard
value by default, so malformed input does not always produce the required machine-readable FAIL.
The checked-in malformed matrix does not cover this case or cross-source mismatch.

Exact fix: bind every producer to an immutable ledger epoch/node token captured before work starts;
reject or retain separately all late/stale publications. Serialize both fixed-cardinality request and
executor maps (and an explicit RAG source if it is distinct), validate source-specific terminal
mappings and exactly-once cross-source/snapshot algebra rather than choosing one source, and bind the
before/after snapshots to the same executor identity and epoch. Reject non-finite numbers and invalid
lifecycle/enumerated values under the same normalized parser boundary. Add adversarial tests for
old-callback-after-reset, overlapping producer epochs, request/executor reason and count mismatch,
snapshot identity mismatch, `NaN`/infinity, and proof that a genuine mismatch row is retained in the
machine-readable FAIL artifact.

## Requirement and acceptance traceability

| Contract | Status | Iteration 3 evidence |
|---|---|---|
| REQ-001 settings | PASS | Eight-field validation/inventory and the complete suite are green. |
| REQ-002 admission/FIFO | PASS | Promotion-submit failure and bounded/FIFO matrices pass. |
| REQ-003 timeout/cancellation | PASS, deterministic | Queued/running disconnect races and executor timeout/cancellation scenarios pass. |
| REQ-004 orphan/conservation | **OPEN** | M42-CR3-MAJ-001 prevents end-to-end terminal provenance and node isolation. |
| REQ-005 drain/shutdown | PASS | M42-CR2-MAJ-002 is closed on the actual lifespan path and fresh process. |
| REQ-006 readiness/metrics | PASS | Saturation/readiness and bounded exact request metrics remain green. |
| REQ-007 input boundary | PASS | Encoding/length rejection remains pre-read; chunk overflow stops at `limit+1`. |
| REQ-008 upstream deadlines | PASS, deterministic | Deadline propagation paths remain covered; live upstream behavior is separate. |
| REQ-009 compatibility/security | PASS, deterministic | Full legacy suite is green and no new disclosure/cardinality defect was found. |
| Deterministic 10-profile acceptance | **Not accepted** | Rows are production-authored but lossy and cross-reset contamination is possible. |
| Injected negative control | PASS for its narrow claim | It deep-copies, corrupts, rejects, and retains a genuinely harvested row with `conservation_mismatch`; it does not cover the reopened provenance defects. |

## Independent receipts

| Command or probe | Exact result |
|---|---|
| Focused lifecycle/parser/executor/input/disconnect/readiness/load suite | PASS: 45 passed in 96.21s. |
| `venv/bin/python -m pytest -q` | PASS: 1024 passed, 1 skipped, 1 pre-existing warning in 114.64s. |
| Production-ledger cross-source mismatch probe | **Accepted incorrectly:** ledger held `request.success=1` and `executor.internal=1`; serialized selected request row parsed with `unknown=0`. |
| Production-ledger reset-isolation probe | **Contaminated:** an `old` executor callback completed after reset and produced `new.executor.success=1`. |
| Typed-parser non-finite probe | **Accepted incorrectly:** `after.capacity_edge_at=NaN` returned `request_count=1`. |
| Actual lifespan task-create tests | PASS inside the focused suite: no-body primary propagation, body-primary identity, ordered cleanup secondary, STOPPED, exact release, and fresh-process repetition. |
| Input rejection and promotion-failure closure tests | PASS inside the focused suite, including four pre-body cases, bounded overflow, and one/two failed FIFO promotions. |
| `npm test` | PASS: 1 file, 9 tests. |
| `bash scripts/compile_lock.sh --verify` | PASS: two independent 102-package resolutions, reproducible with no drift. |
| `venv/bin/python scripts/check_markdown_links.py` | PASS after this artifact: 95 files, 426 links, 0 failures. |
| `git diff --check` | PASS after this artifact. |

## Security, concurrency, and live separation

No new serving-path data disclosure, unbounded metric label, FIFO stall, slot/ticket/timer leak, body
over-read, or lifecycle-owner leak was found. The remaining MAJOR is proof infrastructure with direct
concurrency consequences: a late production callback can be attributed to a different acceptance
node, and contradictory production terminal sources can be silently hidden. It is deterministic and
must be fixed before the receipt can gate release.

The opt-in live 12-case Ollama/network run, M3 live/14-gate receipt, and M4.1 operational acceptance
were not run. `M4.1_BLOCKED` remains a separate M4 release blocker and is not altered by the local
M4.2 result.

## Decision

**Gate FAIL.** M42-CR2-MAJ-002 and all serving-path Iteration 1 defects are closed, but
M42-CR2-MAJ-001 remains open as M42-CR3-MAJ-001. Do not publish or release until both provenance
streams are retained and validated, ledger epochs isolate late/concurrent producers, strict parsing
rejects non-finite/malformed snapshots, and a fresh independent review reaches at least 9.7/10.
