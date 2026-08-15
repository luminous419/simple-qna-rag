# M4 Operational Acceptance Recovery — Design Review Recovery Cycle 1, Iteration 3

Reviewer: Fresh Codex independent design recovery reviewer

Date: 2026-08-15

Reviewed artifact: [Design.md](Design.md)

Normative inputs: [Requirement.md](Requirement.md), [Plan.md](Plan.md),
[Traceability.md](Traceability.md), [Stop_Report.md](Stop_Report.md), all prior
design reviews through [Recovery Cycle 1 Iteration 2](Design_Review_Recovery_Cycle_1_Iteration_2.md),
the milestone orchestration guide, and pinned base revision
`adda1759754b56b514b3ab6252c2dc1032e03d28`.

## 1. Gate decision

**PASS — 9.8 / 10.0**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 0 |
| MINOR | 1 |
| TRIVIAL | 0 |

PASS requires `CRITICAL=0`, `MAJOR=0`, and score `>=9.7`. The revision closes
DR-RC1-I2-MAJ-01 for the pinned production boundary: the exact base shebang and
absence of a PEP 263 cookie are derived from raw bytes, preamble comparison is
byte-exact, and decoding is controlled by `tokenize.detect_encoding`. The one
remaining finding is a bounded specification/utility-generality defect for a
synthetic no-shebang source layout; it cannot occur when auditing the pinned
assembler because that base has the exact required shebang.

No native Linux, Ollama, protected live, environment-approval, self-hosted,
historical-evidence, or acceptance workload was executed or mutated.

## 2. Independent reconstruction and mechanical evidence

I read the pinned assembler as raw stdout bytes from:

```text
git show adda1759754b56b514b3ab6252c2dc1032e03d28:scripts/assemble_m4_evidence.py
```

I independently reconstructed the exact §3.1a statement slicer (including the
earliest-decorator `@` span), the 43-to-48 expected-sequence construction, and
the §3.1b `_source_preamble`/`audit_exact_allowed_delta_bytes` boundary. Expected
source was derived from the pinned base and normative literals, not from a
current implementation file.

| Verification | Result |
|---|---|
| Raw pinned first line | exactly `b"#!/usr/bin/env python3\n"` |
| Pinned encoding cookie | absent; line 2 starts the module docstring |
| `_source_preamble(pinned_bytes)` | exactly `b"#!/usr/bin/env python3\n"` |
| Pinned/expected/planned statement counts | **43 / 48 / 48** |
| Expected versus planned statement sequence | equal; positive fixture passes |
| Modified `#!/usr/bin/env -S python3 -O` | rejected before statement comparison |
| Removed shebang / inserted shebang against a no-shebang base | rejected by preamble inequality |
| Cookie insertion after the pinned shebang | rejected byte-exactly |
| Cookie modification/removal on a shebang-plus-cookie synthetic pair | rejected byte-exactly |
| UTF-8 bytes plus inserted `# coding: latin-1` | detected as `iso-8859-1`; the em dash becomes mojibake, while the byte-aware oracle rejects the cookie delta |
| UTF-8 BOM insertion | rejected by preamble inequality |
| BOM plus non-UTF-8 cookie | `tokenize.detect_encoding` raises `SyntaxError`; mapped fail-closed to `current_source_encoding_conflict` |
| Identical BOM in synthetic base/current | accepted through `utf-8-sig` decoding |
| Leading non-cookie comment/blank lines after the pinned shebang | accepted as the explicitly declared inert gap, without a byte-identity claim |

The prior source matrix also remains covered. Decorator addition, removal,
modification, and reordering change the complete definition slice for
`FunctionDef`, `ClassDef`, and `AsyncFunctionDef`; the representative `assemble`
decorator mutant fails. Added imports, executable statements, unrelated
functions, assignment/function rebinding, class/async shadows, loop/`with`/
exception aliases, named expressions, in-slice whitespace, `assemble` changes,
non-exit `main` changes, missing/arbitrary pinned replacements, partial or
misplaced constants, statement deletion/reordering, syntax errors, and a
semicolon-added second AST node all fail the exact ordered comparison. The two
positive fixtures, all 43 pinned statements, five inserted statements, and the
previous decorator and negative categories are unaffected because §3.1b only
controls raw-byte preamble comparison and decoding before delegating to the
unchanged §3.1a oracle.

The partition claim is now honest: the oracle proves byte identity only for
BOM/shebang/cookie and literal decoded identity for represented top-level
statement spans. It explicitly permits comments and blank lines in gaps outside
those spans and does not claim arbitrary whole-file byte identity.

## 3. DR-RC1-I2-MAJ-01 closure

**CLOSED.** The production audit entry point now consumes raw bytes. It first
pins the base fact itself, compares the current BOM/shebang/cookie preamble
byte-for-byte, fails closed on encoding conflicts, and only then decodes each
source using the encoding returned by CPython's own
`tokenize.detect_encoding`. Thus every requested pinned-boundary mutant changes
the preamble or fails encoding detection before the AST comparison can accept
it. The non-ASCII reproduction demonstrates that this is a semantic decoding
boundary rather than a cosmetic comment check.

All earlier closures remain sound: the complete ordered top-level default-deny
comparison still covers arbitrary statement kinds and rebinding paths;
decorator spans begin at the earliest actual `@`; the workflow stub, raw YAML
scalar exclusion, frozen v1 behavior, v2 state algebra and aliases, runbook
banner, rollback, and historical-evidence boundaries are unchanged. The design
continues to preserve `NOT_ADOPTED` and no-live/no-Ollama/no-self-hosted scope.

## 4. Finding

### DR-RC1-I3-MIN-01 — `_source_preamble` under-models a valid second-line cookie when line 1 is an ordinary comment

**Severity:** MINOR

**Gate:** Helper contract precision / synthetic fixture completeness

**References:** Design.md §3.1b `_source_preamble`, §7.1.

PEP 263 and `tokenize.detect_encoding` permit an encoding declaration on line 2
when line 1 is blank or comment-only, not only when line 1 is a shebang. The
helper sets `cookie_index = 1` only when `has_shebang`; otherwise it inspects
only `consumed[0]`. Mechanical examples beginning with `# ordinary comment`
and then `# coding: latin-1` or `# coding: cp1252` are detected respectively as
`iso-8859-1` and `cp1252` by `tokenize.detect_encoding`, while
`_source_preamble` returns `b""` for both. Modification or removal of that
second-line cookie can consequently be invisible to this helper for a synthetic
no-shebang base/current pair.

This does not reopen DR-RC1-I2-MAJ-01: the authoritative base is separately
pinned to a first-line shebang, so its only valid cookie position is line 2 and
the implemented branch captures it. Removing or moving that shebang already
causes preamble inequality against the pin. The defect is nevertheless a
contradiction of the helper's general PEP 263 wording and leaves its synthetic
no-shebang behavior incomplete.

**Recommended fix:** classify a cookie using the same two consumed lines and
line-position rule as `tokenize.detect_encoding`, independently of whether the
first comment is a shebang. Add a synthetic no-shebang, comment-first,
second-line-cookie modification/removal matrix. This is non-blocking for the
pinned assembler audit.

## 5. Validation and score rationale

The score reflects complete closure of the only prior MAJOR, preservation of
all earlier substantive boundaries, and one bounded MINOR in an unreachable
production-base layout. Implementation may proceed under the documented
9.7 Gate; the recommended helper correction should be included to make the
general contract match PEP 263 exactly.

- Markdown-link validation passed.
- `git diff --check` passed.
- The final diff contains only this review artifact; no Design, policy, code,
  test, workflow, runbook, or historical-evidence file was edited.
