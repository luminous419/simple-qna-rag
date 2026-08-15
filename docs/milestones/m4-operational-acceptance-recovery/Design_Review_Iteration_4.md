# M4 Operational Acceptance Recovery — Design Review Iteration 4

Reviewer: Fresh Codex independent design reviewer  
Date: 2026-08-15  
Reviewed artifact: [Design.md](Design.md)  
Normative inputs: [Requirement.md](Requirement.md), [Plan.md](Plan.md),
[Traceability.md](Traceability.md), [Stop_Report.md](Stop_Report.md),
[Design Review Iteration 1](Design_Review_Iteration_1.md),
[Design Review Iteration 2](Design_Review_Iteration_2.md),
[Design Review Iteration 3](Design_Review_Iteration_3.md), the milestone
orchestration guide, and the current assembler, checker, workflow, tests, and
runbooks.

## 1. Gate decision

**FAIL — 8.9 / 10.0**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 2 |
| MINOR | 1 |
| TRIVIAL | 0 |

PASS requires `CRITICAL=0`, `MAJOR=0`, and score `>=9.7`. The revision exactly
closes DR-I3-MAJ-01, DR-I3-MAJ-02 for the specified assignment/function
mutants, and DR-I3-MIN-01. The authoritative source-preservation mechanism is
still fail-open, however, for other Python top-level rebinding forms and for
arbitrary edits outside its 26-symbol allowlist.

No native Linux, Ollama, protected live, environment-approval, self-hosted, or
historical-evidence execution or mutation was performed.

## 2. Iteration-3 closure verification

| Finding | Status | Mechanical evidence |
|---|---|---|
| DR-I3-MAJ-01 | **CLOSED** | `PROTECTED_SYMBOLS` contains exactly 26 names: 15 simple top-level assignments followed by 11 top-level functions in pinned base `adda1759754b56b514b3ab6252c2dc1032e03d28`. `_in_span_mutation` inserts one space after the assignment `=` or function-signature `(`, inside the reparsed statement span. Inspection of all 26 pinned slices shows those anchors are present and precede the protected value/body; each mutant remains parsable, its selected `ast.get_source_segment` differs from the base slice, and the specified 26-way test requires exactly `protected_symbol_source_changed:<symbol>`. |
| DR-I3-MAJ-02 | **CLOSED for the required assignment/function matrix; broader gap reopened below** | `_module_source_segments` collects every matching top-level `Assign`/`AnnAssign` and `FunctionDef`. For each of the 15 assignments and 11 functions, removal yields zero current slices and the distinct `protected_symbol_removed:<symbol>` violation; appending a second matching assignment or function yields two slices and the distinct `protected_symbol_duplicate_binding:<symbol>` violation. Base-side zero/multiple bindings also have separate `missing_in_base`/`duplicate_binding_in_base` reasons. The positive test targets the actual implementation-phase v2 file and requires the intended new constants, `_build_baseline` replacement, and `main()` exit change to leave all 26 protected slices unchanged. |
| DR-I3-MIN-01 | **CLOSED** | `_m3_gate_denylist_scan_text` locates the raw `run: |` header, consumes its indented scalar lines, dedents those exact raw lines, compares them to the canonical parsed pin, and removes the original indented body only on exact equality. The positive oracle requires every pinned script line to be absent from the returned scan text. The canonical YAML fixture remains parseable and exact-shaped; each of the 15 forbidden surfaces is represented by a parsable job-level field mutation or an executable line added inside the `run` scalar, so a changed scalar is not removed and the matching denylist reason is exercised. |

All previous Iteration-1 closures and DR-I2-MAJ-01/DR-I2-MIN-01 remain sound:
legacy v1 is frozen blocked; v2 identity, producer algebra, and aliases are
recomputed fail-closed; the stub has an exact hosted/no-checkout shape; ordinary
jobs do not depend on the live gate; the historical runbook is banner-qualified;
and rollback never restores an ordinary-triggered self-hosted path.

## 3. Findings

### DR-I4-MAJ-01 — “All top-level bindings” ignores valid Python rebinding forms

**Severity:** MAJOR  
**Gate:** Pre-merge evidence-preservation audit / fail-closed source integrity  
**References:** Design.md §3.1a `_module_source_segments`, DR-I3-MAJ-02.

The prose makes the audit authoritative and says it collects all top-level
bindings, but the implementation recognizes only `FunctionDef`, `Assign`, and
`AnnAssign`. Python permits later top-level rebinding through forms such as
`from attacker import REQUIRED_PRODUCERS`, `class _evaluate_producer: ...`, a
`for REQUIRED_PRODUCERS in ...` target, or a `with ... as _settings_hash`
target. Each can replace a protected runtime value after the original protected
statement while `_module_source_segments` still returns exactly the one original
slice, causing `audit_protected_symbols` to return `[]`. `AsyncFunctionDef` is
also omitted.

This reopens the security property behind DR-I3-MAJ-02 even though its named
assignment/function duplicate tests now pass.

**Required fix:** either reject every additional module-scope binding of a
protected name using a complete scope-binding analysis (including imports,
classes, async functions, loop/comprehension targets where applicable, `with`
aliases, exception aliases, and named expressions), or fail closed on any
unapproved top-level statement/delta. Add one exact-violation mutant for every
supported binding category and retain the existing 15 assignment plus 11
function duplicate/removal matrices.

### DR-I4-MAJ-02 — The positive audit does not enforce the claimed exact allowed change set

**Severity:** MAJOR  
**Gate:** Pre-merge evidence-preservation audit / plan-scope enforcement  
**References:** Plan.md §2–§3; Design.md §0.1, §3.1a, §7.1, §12.

The requested positive boundary is “new v2 constants, `_build_baseline`, and the
`main()` exit-line change.” The audit only compares the 26 named protected
statements. It does not compare or constrain imports, module-level executable
statements, `assemble`, `main` apart from the intended exit line, existing
unprotected constants/functions, or newly added names. Consequently the actual
planned v2 file can pass, but so can that same file plus an arbitrary rewrite of
`assemble`, extra module-scope execution, or unrelated changes inside `main`.
The positive test proves inclusion of the desired patch; it does not prove
exclusion of undesired patches. This contradicts the design's scoped-change
claim and leaves a second route around the protected producer verification.

**Required fix:** define and mechanically compare the complete base-to-v2 AST
delta. Permit only explicitly pinned new constant statements, the exact planned
`_build_baseline` replacement, and the exact `main()` exit-expression change;
require every other statement/source slice to remain identical. Add negative
mutants for an import, a new executable statement, `assemble`, a non-exit line
inside `main`, and a new unrelated function, plus the positive actual planned
v2 file.

### DR-I4-MIN-01 — The stated whole-workflow `self-hosted` grep cannot pass the canonical stub

**Severity:** MINOR  
**Gate:** Design consistency / implementation checklist accuracy  
**References:** Design.md §5.3 item 2 and canonical stub.

The design says `grep -c "self-hosted" .github/workflows/ci.yml` must be zero,
but the canonical job's pinned informational echo contains “no self-hosted
runner,” and its explanatory comment also contains the token. The structural
contract and refined executable-surface regex correctly distinguish this safe
negation, so this does not reopen the workflow MAJOR; the broad grep is simply
an impossible and misleading audit command.

**Required fix:** remove the broad grep claim or replace it with the same
parsed `runs-on`/exact-key and executable-surface checks used by §7.3.

## 4. Score rationale and next gate

The state algebra, migration, workflow steady state, rollback, runbook boundary,
26 in-span mutants, assignment/function removal and duplicate matrices, and raw
YAML scalar handling are detailed and implementable. The two remaining MAJOR
issues affect the mechanism designated as the authoritative proof that existing
producer verification is unchanged; implementation must not begin while that
mechanism permits runtime-changing source edits to pass.

Next review should mechanically demonstrate a complete top-level binding model
and an exact whole-file allowed-delta oracle, then repeat all prior positive and
negative closures. PASS remains unavailable until both MAJOR findings are closed
and the score reaches at least 9.7.

## 5. Validation

- Markdown-link validation and `git diff --check` were run after writing this
  review.
- No product/design/policy/workflow/test file was edited; only this review
  artifact was created.
- No live, native, self-hosted, Ollama, approval, or acceptance workload was
  executed.
