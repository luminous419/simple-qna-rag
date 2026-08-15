# M4 Operational Acceptance Recovery — Design Review Iteration 2

Reviewer: Fresh Codex independent design reviewer  
Date: 2026-08-15  
Reviewed artifact: [Design.md](Design.md)  
Normative inputs: [Requirement.md](Requirement.md), [Plan.md](Plan.md),
[Traceability.md](Traceability.md), [Stop_Report.md](Stop_Report.md),
[Design Review Iteration 1](Design_Review_Iteration_1.md), current repository
implementation/tests/workflow, and preserved M4.3 evidence.

## 1. Gate decision

**FAIL — 8.8 / 10.0**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 2 |
| MINOR | 1 |
| TRIVIAL | 0 |

PASS requires `CRITICAL=0`, `MAJOR=0`, and score `>=9.7`. The revision closes
all five Iteration-1 MAJOR findings in its intended steady-state contracts, but
two of the proposed acceptance mechanisms cannot pass the design's own required
implementation, and one runbook-banner test does not match the specified banner.

## 2. Iteration-1 closure verification

| Finding | Status | Evidence |
|---|---|---|
| DR-I1-MAJ-01 | **CLOSED** | Design.md §4.4 unconditionally fixes legacy v1 to live `NOT_RUN`, M4.1 `BLOCKED`, `operational_status=BLOCKED`, `M4.1_BLOCKED=true`, and `overall_release_ready=false` before producer algebra, independent of `--expect-operational-blocked`. §7.2 adds mutants using `--allow-legacy-v1` alone. |
| DR-I1-MAJ-02 | **CLOSED** | Design.md §4.3 exact-checks `workflow_run`, validates scalar identity, compares five optional expected identity values, and recomputes both top-level aliases from validated producer payload hashes. §4.6/§8.3 require all five expected values for the normative post-merge hosted-ready command through `--require-identity-binding`; §4.7 accurately limits the checker claim to artifact self-consistency plus operator-supplied identity rather than payload-byte revalidation. |
| DR-I1-MAJ-03 | **REOPENED by DR-I2-MAJ-01** | §7.3 now specifies exact job/step key sets, one exact script, no `uses`/`env`/`with`, an exact workflow job set, and all-job `needs` checks. However, its source denylist rejects the required stub itself. |
| DR-I1-MAJ-04 | **CLOSED, with DR-I2-MIN-01** | §8.3 makes the hosted/OCI deployment runbook normative and identity-bound; §8.4 preserves the M4.1 body while adding a prominent superseded/non-executable banner; §7.4 limits exceptions to the banner-bearing historical path. The concept is sound, though one proposed banner-position assertion is off by the specified line count. |
| DR-I1-MAJ-05 | **CLOSED** | Design.md §11 separates schema/checker rollback from workflow rollback and prohibits every rollback combination from restoring the ordinary-triggered self-hosted path. The matrix retains either deletion or the hosted no-op stub and reapplies the workflow safety suite. |
| DR-I1-MIN-01 | **REOPENED by DR-I2-MAJ-02** | The revision replaces the unsupported byte-equality claim with a base SHA, symbol inventory, and diff audit, but the stated protected interval includes an explicitly mutable function and therefore rejects the required patch. |

## 3. New findings

### DR-I2-MAJ-01 — The raw workflow denylist rejects the exact required no-op stub

**Gate:** Pre-merge workflow contract / ordinary-run closure  
**References:** Requirement.md M4-OAR-REQ-004.2 and REQ-006; Design.md §5.3,
§7.3; current `.github/workflows/ci.yml`:316-365.

Design §7.3 requires `test_m3_live_regression_gate_source_text_has_no_forbidden_substrings`
to slice the raw `m3-live-regression-gate` block and reject case-insensitive
`ollama`, `self-hosted`, and `environment:`. The exact allowlisted block in §5.3
necessarily contains two forbidden tokens: its second echo says "no secrets,
no environment approval, and no self-hosted runner." (The surrounding normative
comment also uses those terms, although the stated slice begins at the job key.)
Consequently the
required implementation cannot satisfy both the exact-script equality test and
the raw-source denylist. This makes the pre-merge gate deterministically red and
also undermines the claimed rollback validation, which mandates rerunning this
same suite.

**Required fix:** Define a single canonical safe stub and make structural
exact-shape/equality the authoritative executable-surface check. If a raw scan
is retained, scan only executable values that are not already exact-pinned, or
use precise dangerous forms such as `${{ secrets.`, `environment:` as a YAML
key, runner labels, endpoint/model tokens, network commands, checkout/fetch/
clone commands, and repository-script execution. Do not forbid explanatory
negations or policy-document names. Add a positive test that the literal §5.3
stub passes the complete contract suite, plus one mutant per forbidden
executable surface.

### DR-I2-MAJ-02 — The protected-symbol audit interval overlaps the required `_build_baseline` replacement

**Gate:** Pre-merge evidence-preservation audit  
**References:** Plan.md §3.1; Design.md §3.1a, §3.2, §12; current
`scripts/assemble_m4_evidence.py`:35-327.

Design §3.1a declares base lines 35-327 protected and says any diff hunk whose
old-side interval overlaps that range must fail. In the reviewed base,
`_build_baseline` is lines 276-312, inside that protected interval, yet the same
section explicitly permits replacing `_build_baseline`, and §3.2 requires that
replacement to emit v2. The prose also describes lines 35-327 as ending
"before `_build_baseline`," although `_dependency_snapshot_sha256` and
`_settings_hash` at lines 315-327 are after it. A normal replacement hunk around
276-312 must therefore be rejected by the mandated audit. Moreover, `grep '^@@'`
only prints hunk headers; it does not implement the promised interval-overlap
decision, making the checklist dependent on manual, error-prone interpretation.

**Required fix:** Protect named symbols, not one contiguous range crossing an
allowed edit. Use an automated AST/source-slice comparison against
`adda1759754b56b514b3ab6252c2dc1032e03d28` for each listed constant/function,
excluding `_build_baseline`, or define accurate disjoint immutable ranges and a
script that parses hunks and fails mechanically. Add a test proving an
`_build_baseline`-only v2 change passes while a one-line mutation in every
protected region fails.

### DR-I2-MIN-01 — The specified banner link may fall outside the asserted first ten lines

**Gate:** Documentation audit reliability  
**References:** Design.md §7.4, §8.4.

The exact banner in §8.4 spans more than ten Markdown source lines, with the
`deployment_runbook.md` link at its end, while
`test_ci_acceptance_runbook_has_superseded_banner_near_top` inspects only
`text.splitlines()[:10]` and requires that filename in the slice. Implementing
the banner verbatim can therefore fail the prescribed test despite placing it
at the very top.

**Required fix:** Either put the normative-runbook link within the first ten
physical lines, or parse the complete initial blockquote/front-matter banner and
assert both marker and link within that block rather than using a magic line
count.

## 4. No additional blocking finding

- V2 readiness remains fail-closed: deterministic status is derived from the
  four producer variants, hosted readiness follows only that status, excluded
  gates are literal `NOT_ADOPTED`, and native/full/overall cannot become true.
- Alias recomputation handles both `OK` and non-`OK` producer variants (`hash`
  versus `None`) after exact producer-shape validation.
- The normative post-merge procedure binds SHA, run ID, attempt, workflow path,
  and event and does not overstate payload-byte verification.
- The rollback matrix never restores the pending ordinary-push live path.
- Current product code exposes no `release_ready` field under `src/` or `web/`;
  the proposed scope remains documentation/schema/workflow/tests only.

## 5. Required next iteration

Revise Design.md to make the workflow contract internally satisfiable, replace
the overlapping/manual protected-range audit with an executable symbol-level or
disjoint-range audit, and align the banner test with the exact banner. No live,
self-hosted, environment-approval, Ollama, product-code, or historical-evidence
change is needed.
