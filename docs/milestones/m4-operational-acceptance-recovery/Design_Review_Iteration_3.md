# M4 Operational Acceptance Recovery — Design Review Iteration 3

Reviewer: Fresh Codex independent design reviewer  
Date: 2026-08-15  
Reviewed artifact: [Design.md](Design.md)  
Normative inputs: [Requirement.md](Requirement.md), [Plan.md](Plan.md),
[Traceability.md](Traceability.md), [Stop_Report.md](Stop_Report.md),
[Design Review Iteration 1](Design_Review_Iteration_1.md),
[Design Review Iteration 2](Design_Review_Iteration_2.md), the milestone
orchestration guide, operational runbooks, and the current assembler, checker,
workflow, and tests.

## 1. Gate decision

**FAIL — 9.1 / 10.0**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 2 |
| MINOR | 1 |
| TRIVIAL | 0 |

PASS requires `CRITICAL=0`, `MAJOR=0`, and score `>=9.7`. DR-I2-MAJ-01 and
DR-I2-MIN-01 are mechanically closed, but DR-I2-MAJ-02 remains open: its
specified all-symbol mutation test detects none of its 26 generated mutants.
The same audit also fails open on duplicate top-level rebinding of any protected
name.

## 2. Iteration-2 closure verification

| Finding | Status | Mechanical evidence |
|---|---|---|
| DR-I2-MAJ-01 | **CLOSED, with DR-I3-MIN-01** | A canonical §5.3 YAML stub parses to the exact pinned job/script shape. Parsable mutations adding live/Ollama text, a secret interpolation, checkout/`uses`, an environment key, a self-hosted runner, or another executable step necessarily change an exact-pinned field/key set and fail the structural suite. The refined regexes no longer reject the harmless explanatory negations. |
| DR-I2-MAJ-02 | **OPEN — DR-I3-MAJ-01 and DR-I3-MAJ-02** | Running the specified `base_source.replace(segment, segment + " ", 1)` construction against base `adda1759754b56b514b3ab6252c2dc1032e03d28` produced 26/26 mutants for which `ast.get_source_segment` remained identical, so the promised negative test cannot pass. A second mechanical mutant that appends a new `REQUIRED_PRODUCERS = ...` binding also returns an empty audit because the helper selects only the first binding. |
| DR-I2-MIN-01 | **CLOSED** | `_leading_blockquote` consumes every consecutive initial `>` line, including the twelfth line containing `deployment_runbook.md`; it stops at the first non-blockquote line and therefore binds both marker and normative link to the initial banner rather than to an arbitrary later occurrence. |

All Iteration-1 MAJOR closures remain sound: v1 is fixed to historical blocked
semantics; v2 identity binding and alias recomputation are fail-closed; the
workflow steady state is exact-shaped and ordinary jobs have no live dependency;
the historical runbook is banner-qualified and the hosted/OCI runbook is
normative; and rollback never restores the ordinary-triggered self-hosted job.
The former DR-I1-MIN-01 remains reopened through the protected-symbol defects
below.

## 3. Findings

### DR-I3-MAJ-01 — The required 26-way protected-symbol mutation test detects zero mutations

**Gate:** Pre-merge evidence-preservation audit  
**References:** Design.md §3.1a, §7.1, §12; DR-I2-MAJ-02.

For every protected symbol, the design says to take the exact AST source
segment and generate a mutant with
`base_source.replace(segment, segment + " ", 1)`. The appended space begins
after the node's recorded `end_col_offset`; reparsing does not extend the node,
so `ast.get_source_segment(mutant_source, node)` returns the original segment.
Mechanical execution against the pinned base reported
`WHITESPACE_MUTANT_MISSED` for all 26 symbols. Therefore
`audit_protected_symbols(base, mutant)` returns no
`protected_symbol_source_changed:<symbol>` violation, directly contradicting
the required test and the claimed proof that every protected mutation fails.

**Required fix:** Generate a syntax-preserving mutation *inside* each AST node's
source span, not after it. Use a symbol-specific token mutation or insert a
comment/whitespace at a position that remains within the statement node, and
first assert that the mutated source segment differs from the base segment.
Parametrize all 26 names and require the exact violation for each. Retain a
positive test against the actual planned v2 file proving `_build_baseline`, new
constants, and the `main()` exit-line change are permitted.

### DR-I3-MAJ-02 — The named-symbol audit accepts a later rebinding that replaces protected runtime behavior

**Gate:** Pre-merge evidence-preservation audit / fail-closed source integrity  
**References:** Design.md §3.1a `_module_source_segment` and
`audit_protected_symbols`.

`_module_source_segment` returns on the first matching top-level definition or
assignment. It never establishes that the name has exactly one binding. Appending
`REQUIRED_PRODUCERS = ("attacker-job",)` to the pinned base leaves the first
source slice byte-identical, so the audit reports no violation, while Python's
last assignment becomes the runtime value. The same bypass applies to protected
functions by appending a second `def` with the same name. This is a substantive
fail-open gap in the mechanism designated as the authoritative preservation
decision.

**Required fix:** Collect all top-level bindings for every protected name in
both sources; require exactly one binding in the base and exactly one in the
current source, then compare those unique source slices. Reject duplicate
assignments/definitions with a distinct violation. Add assignment and function
duplicate-rebinding mutants, plus removal and in-span mutation cases.

### DR-I3-MIN-01 — The raw YAML helper does not actually remove the exact-pinned script source

**Gate:** Pre-merge workflow defense-in-depth  
**References:** Design.md §7.3 `_m3_gate_denylist_scan_text`.

The helper calls `block.replace(M3_GATE_PINNED_RUN_SCRIPT, "", 1)`, but the raw
YAML block contains indentation before each block-scalar script line whereas
the parsed pinned script does not. Mechanical reproduction showed that the YAML
parses to the exact pinned string while `replace` leaves the raw block unchanged.
The canonical stub still passes because the refined patterns do not match its
negated prose, and the exact-shape checks reject executable mutants, so this is
not a reopened MAJOR; however, the documented source-scan exclusion and its
claimed mutant rationale are false.

**Required fix:** Locate the `run: |` scalar in the raw job block and remove its
indented scalar lines, or derive source spans with a YAML parser that preserves
marks. Add an assertion that the canonical pinned scalar is actually removed
and test parsable executable-field mutants rather than injecting dangerous text
only into comments.

## 4. Other verification

- V2 readiness and support-policy algebra remain fail-closed; native, full, and
  overall readiness cannot become true through candidate aggregates.
- No reviewed rollback combination reintroduces live/self-hosted execution.
- Current repository implementation remains the pre-design v1/live-workflow
  state, consistently labeled implementation-pending by the policy documents;
  no live, Ollama, environment, checkout, or self-hosted execution was performed.
- `python scripts/check_markdown_links.py` passed: 152 files, 624 links, 0
  failures.
- `git diff --check` passed.

## 5. Required next iteration

Repair both protected-symbol audit defects and the raw-scalar exclusion, then
rerun the canonical workflow positive case, parsable live/self-hosted/
environment/checkout/secret mutants, the actual v2 positive source audit, all
26 in-span protected-symbol mutants, and duplicate-rebinding mutants. A fresh
review may issue PASS only with no CRITICAL or MAJOR findings and a score of at
least 9.7.
