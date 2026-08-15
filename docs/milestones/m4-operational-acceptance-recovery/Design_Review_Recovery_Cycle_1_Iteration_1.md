# M4 Operational Acceptance Recovery — Design Review Recovery Cycle 1, Iteration 1

Reviewer: Fresh Codex independent design recovery reviewer  
Date: 2026-08-15  
Reviewed artifact: [Design.md](Design.md)  
Normative inputs: [Requirement.md](Requirement.md), [Plan.md](Plan.md),
[Traceability.md](Traceability.md), [Stop_Report.md](Stop_Report.md),
[Design Review Iteration 1](Design_Review_Iteration_1.md),
[Design Review Iteration 2](Design_Review_Iteration_2.md),
[Design Review Iteration 3](Design_Review_Iteration_3.md),
[Design Review Iteration 4](Design_Review_Iteration_4.md), the milestone
orchestration guide, and the pinned base/current assembler and tests.

## 1. Gate decision

**FAIL — 9.3 / 10.0**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 1 |
| MINOR | 0 |
| TRIVIAL | 0 |

PASS requires `CRITICAL=0`, `MAJOR=0`, and score `>=9.7`. The revision closes
DR-I4-MAJ-01 and DR-I4-MAJ-02 for the enumerated mutant matrix and closes
DR-I4-MIN-01, but the authoritative whole-file oracle still admits a
runtime-changing source delta outside the source span returned for a decorated
top-level definition.

No native Linux, Ollama, protected live, environment-approval, self-hosted, or
historical-evidence execution or mutation was performed.

## 2. Iteration-4 closure verification

| Finding | Status | Mechanical evidence |
|---|---|---|
| DR-I4-MAJ-01 | **CLOSED for the required binding/statement matrix; broader source-span gap reopened below** | Reconstructing `audit_exact_allowed_delta` against `adda1759754b56b514b3ab6252c2dc1032e03d28` rejected added imports, executable expressions, classes, sync/async functions, assignments, annotated assignments, augmented assignments, loops, `with` aliases, exception aliases, named expressions, and duplicate assignment/function bindings. Because the comparison consumes `ast.parse(source).body` without dispatching on node kind, the former incomplete binding-category enumeration is gone. |
| DR-I4-MAJ-02 | **CLOSED for the exact planned sequence and listed negative matrix; broader source-span gap reopened below** | The pinned base has 43 top-level statements. Independently extracting the §3.2 replacement, deriving the §3.3 `main()` replacement from the pinned base by changing only its final return expression, and inserting the five §3.1 constants after the unique anchor produced exactly 48 expected statements; the reconstructed planned v2 source also produced 48 and compared equal. Added imports/executable statements/unrelated names, `assemble` slice changes, non-exit `main` changes, arbitrary or omitted `_build_baseline`, omitted `main` change, wrong/partial constant insertion, deletion, duplicates, and syntax errors fail. Sequence comparison also rejects any actual removal, addition, or reordering of represented statement slices. |
| DR-I4-MIN-01 | **CLOSED** | The impossible whole-workflow `grep -c "self-hosted" == 0` is explicitly withdrawn. The normative checks are now parsed exact equality for `runs-on: ubuntu-latest`, exact job/step/script shape, and the refined executable-surface scan with the canonical scalar actually removed before scanning. |

The old pinned slices are mechanically obtained from
`git show adda1759754b56b514b3ab6252c2dc1032e03d28:scripts/assemble_m4_evidence.py`
and AST selection. The new `_build_baseline` slice is the independent normative
§3.2 code block, the new `main` slice is the pinned old slice with its one
specified return-expression substitution, and the constant slices are the
literal §3.1 pins. This permits a realizable positive fixture without reading
or copying the current implementation as the expected oracle; the separate
actual-file positive test then checks the implementation against those pins.

All prior state-algebra, identity/alias, legacy-v1, workflow exact-shape,
raw-scalar, runbook-banner, rollback, and historical-evidence closures remain
internally consistent in this review.

## 3. Finding

### DR-RC1-I1-MAJ-01 — Decorators are outside the compared top-level statement slice

**Severity:** MAJOR  
**Gate:** Pre-merge evidence-preservation audit / exact allowed source delta  
**References:** Design.md §3.1a `_top_level_statement_slices`, §7.1, §12;
DR-I4-MAJ-01/DR-I4-MAJ-02.

For a decorated `FunctionDef` or `AsyncFunctionDef`, Python records decorators
in `node.decorator_list`, but the function node's `lineno` and source segment
begin at the `def`/`async def` token. Therefore
`ast.get_source_segment(source, node)` omits every preceding decorator line.
The whole-file oracle compares only those returned strings and never separately
compares `decorator_list` source spans.

Mechanical reproduction on the pinned assembler added this line immediately
before the otherwise unchanged top-level `assemble` definition:

```python
@(lambda f: (lambda *a, **k: {}))
```

The mutant parses, changes runtime behavior, and leaves the complete ordered
list returned by `_top_level_statement_slices` equal to the unmodified list.
Applied to the reconstructed positive v2 source, `audit_exact_allowed_delta`
therefore returns `[]`. A simpler `@staticmethod` addition reproduces the same
source-list equality. This violates the stated rule that only the five constant
insertions, exact `_build_baseline` replacement, and exact `main` replacement
may pass, and permits an `assemble` behavior change that DR-I4-MAJ-02 explicitly
requires the oracle to reject.

**Required fix:** make each represented top-level slice start at the earliest
decorator for decorated class/function/async-function statements (using AST
line/column coordinates rather than `ast.get_source_segment` on the definition
node alone), or compare a complete token/source partition that includes
decorator spans. Add negative tests for adding, removing, modifying, and
reordering decorators on an otherwise unchanged protected top-level definition,
including `assemble`; retain the reconstructed positive fixture and all current
mutants.

## 4. Additional verification

- The canonical workflow stub and parsed checks remain compatible: the safe
  explanatory `self-hosted` text is not treated as an executable runner label.
- Broad self-hosted grep appears only in historical/problem explanation, not as
  an acceptance command.
- `python scripts/check_markdown_links.py` passed: 155 files, 647 links, 0
  failures.
- `git diff --check` passed after this report was added.

## 5. Next gate

Repair the decorator-span fail-open and mechanically repeat the actual and
synthetic positive cases, the complete existing negative matrix, and decorator
addition/removal/modification/reordering mutants. PASS remains unavailable
until no CRITICAL or MAJOR finding remains and the score is at least 9.7.
