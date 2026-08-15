# M4 Operational Acceptance Recovery — Design Review Recovery Cycle 1, Iteration 2

Reviewer: Fresh Codex independent design recovery reviewer  
Date: 2026-08-15  
Reviewed artifact: [Design.md](Design.md)  
Normative inputs: [Requirement.md](Requirement.md), [Plan.md](Plan.md),
[Traceability.md](Traceability.md), [Stop_Report.md](Stop_Report.md),
[Design Review Iteration 1](Design_Review_Iteration_1.md),
[Design Review Iteration 2](Design_Review_Iteration_2.md),
[Design Review Iteration 3](Design_Review_Iteration_3.md),
[Design Review Iteration 4](Design_Review_Iteration_4.md), and
[Recovery Cycle 1 Iteration 1](Design_Review_Recovery_Cycle_1_Iteration_1.md).

## 1. Gate decision

**FAIL — 9.4 / 10.0**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 1 |
| MINOR | 0 |
| TRIVIAL | 0 |

PASS requires `CRITICAL=0`, `MAJOR=0`, and score `>=9.7`. The revision closes
DR-RC1-I1-MAJ-01: decorated definitions are now sliced from the earliest actual
`@` through the definition node's end. The Gate still fails because the claimed
complete-span whole-file oracle ignores the executable file preamble; a changed
encoding declaration or shebang leaves every compared statement slice unchanged.

No native Linux, Ollama, protected live, environment-approval, self-hosted, or
historical-evidence execution or mutation was performed.

## 2. Mechanical reconstruction and closure verification

I independently read the pinned assembler with:

```text
git show adda1759754b56b514b3ab6252c2dc1032e03d28:scripts/assemble_m4_evidence.py
```

I reconstructed `_statement_source_slice`, `_top_level_statement_slices`, and
`audit_exact_allowed_delta` from the revised §3.1a, selected the old
`_build_baseline` and `main` slices from the pinned AST, extracted the normative
new `_build_baseline` from §3.2, derived the new `main` by the single specified
return-expression substitution, and inserted the five literal constants after
the unique pinned anchor. I did not derive expected output from a planned/current
implementation file.

| Verification | Result |
|---|---|
| Pinned base statement count | **43** |
| Independently constructed expected v2 count | **48** |
| Independently constructed planned v2 count | **48** |
| Expected sequence versus planned v2 sequence | **equal / positive fixture passes** |
| Decorated `FunctionDef` slice | Starts at the earliest actual `@`; includes all decorators, intervening comments/blank lines, the definition, and its body through `end_lineno`/`end_col_offset` |
| Decorated `ClassDef` and `AsyncFunctionDef` slices | Same complete-span behavior |
| Spaced/multiline decorator forms | `@ ( lambda f: f )`, multiline call decorators with inline comments, and multiple decorators separated by comments/blank lines all start at `@` and are fully represented |
| Decorator addition/removal/modification/reordering | All change the ordered slice sequence and fail, including `assemble`, other pinned-base functions, and synthetic class/async definitions |
| Comments/blank lines between statements or decorators | Semantically inert and do not independently create AST statements; expected positive cases remain accepted |

The prior negative matrix was also repeated. Added imports, import rebinding,
executable expressions, unrelated functions, class/sync/async shadows,
assignment/annotated-assignment/augmented-assignment rebinding, loop targets,
`with` and exception aliases, named expressions, duplicate assignments/functions,
in-slice whitespace changes, `assemble` changes, non-exit `main` changes, omitted
`main` or `_build_baseline` replacements, arbitrary `_build_baseline` rewrites,
wrong/partial constant insertion, statement deletion/reordering, and syntax
errors all fail. An executable statement appended with a semicolon creates a
second AST node and fails; an otherwise inert trailing semicolon and type comment
remain outside the semantic comparison.

This confirms DR-RC1-I1-MAJ-01 is **CLOSED**. It also preserves the earlier
DR-I1 through DR-I4 state-algebra, identity/alias, legacy-v1, workflow-shape,
documentation-banner, rollback, and historical-evidence closures. The positive
oracle is non-circular: its expected sequence comes from the pinned revision and
normative design literals, while the actual-file test remains a separate consumer.

## 3. Finding

### DR-RC1-I2-MAJ-01 — Executable file preamble is outside the “whole-file” oracle

**Severity:** MAJOR  
**Gate:** Pre-merge evidence-preservation audit / exact allowed source delta  
**References:** Design.md §3.1a, §7.1, §12; DR-I4-MAJ-02.

`_top_level_statement_slices` begins with the module docstring AST node and never
represents source text before that node. Consequently, both of these mutations
leave the complete ordered slice list equal to the pinned base list:

```python
#!/usr/bin/env -S python3 -O
# coding: latin-1
```

The shebang is an execution boundary, not an ordinary inert comment: this file
is executable, and direct invocation delegates interpreter selection/options to
that line. The encoding declaration is a decoding boundary, not an ordinary
inert comment. The pinned file contains UTF-8 non-ASCII text in its module
docstring. Inserting `# coding: latin-1` on line 2 and decoding the same UTF-8
bytes according to Python's detected `iso-8859-1` encoding changes the module
docstring (for example, the em dash becomes mojibake), and therefore changes the
`argparse.ArgumentParser(description=__doc__)` CLI output. Nevertheless,
`ast.parse(current_source)` receives an already decoded string, the cookie is
only a comment at that stage, and the statement slices compare equal.

This is a concrete fail-open against the design's stronger claim that only the
five pinned constant insertions and exact `_build_baseline`/`main` replacements
may pass. It is distinct from DR-RC1-I1-MAJ-01: decorator spans are now complete,
but the source partition is still not complete at the file-loading boundary.

**Required fix:** pin and compare the source prefix before the first represented
top-level statement, at minimum the exact shebang and absence/exact value of an
encoding declaration, or replace the statement-only construction with a
token/byte-aware complete-file partition that explicitly classifies the few
permitted inert gaps. Add negative tests for shebang modification/removal and
encoding-cookie insertion/modification, including a non-ASCII semantic
reproduction; retain positive comment/blank-line tests for genuinely inert gaps,
the 43-to-48 fixture, every prior negative category, and all decorator mutants.

## 4. Additional verification

- `python scripts/check_markdown_links.py` passed: 156 files, 659 links, 0 failures.
- `git diff --check` passed with this report present.
- No live/self-hosted workflow, product code, test, policy, historical artifact,
  or implementation file was edited.

## 5. Next gate

Close DR-RC1-I2-MAJ-01 by covering the executable/decoding preamble, then repeat
the independent positive fixture, complete prior negative matrix, decorator
matrix, and lexical-boundary mutants. PASS remains unavailable until no CRITICAL
or MAJOR finding remains and the score is at least 9.7.
