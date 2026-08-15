# M4 Operational Acceptance Recovery stop and policy-decision report

Date: **2026-08-15**  
Former recovery outcome: **STOPPED — dependency unavailable**  
Current decision: **APPROVED SCOPE CHANGE — native Linux/Ollama NOT_ADOPTED**  
Terminal outcome (superseded — see §6): ~~STOPPED — design Gate exhausted at Iteration 4~~  
Current outcome: **RESUMED — design Gate PASS at Recovery Cycle 1, Iteration 3 (9.8/10.0); implementation complete (pre-merge)**

## 1. What stopped

The original plan sought a hardened native Linux x86_64 host, repository-scoped
`[self-hosted, ollama-m3]` runner, canonical read-only data, protected approval,
and an Ollama-backed 14-gate receipt. At the stop point, GitHub environment
`m3-live-regression` and required reviewer `luminous419` existed, but repository
self-hosted runner count was zero and no authorized host owner could provide or
administer the machine. Run `31825950604` at merge SHA
`adda1759754b56b514b3ab6252c2dc1032e03d28` therefore retained a pending
`m3-live-regression-gate` while its ordinary hosted jobs passed.

No environment approval, runner registration, Ollama/live execution, canonical
data access, or live receipt occurred. Pending was never PASS. M4.3 deterministic
PASS and earlier receipts remain valid and unchanged.

## 2. Approved resolution

The user permanently declined the unavailable native/Ollama validation scope
and selected the verified hosted/OCI product boundary. The former prerequisite
is not deferred, silently skipped, or waived: it is typed `NOT_ADOPTED`.
Accordingly, the implementation may clear `M4.1_BLOCKED` only as “no remaining
adopted-scope work,” while `m41_operational` and `m3_live_regression` remain
`NOT_ADOPTED`, never `PASS`.

Hosted readiness is a new narrow claim derived exclusively from deterministic
Python, frontend, OCI container, M4.3, assembler, and independent checker
evidence. Native Linux readiness is false. Full-production readiness and the
legacy `overall_release_ready` alias are false. Documentation and release notes
must name the narrow hosted/OCI claim.

## 3. Workflow stop correction

The protected job must no longer be selected by ordinary `master` pushes, where
absent self-hosted capacity can leave the run permanently pending. The policy
implementation will remove the executable job or constrain a harmless future
reactivation marker to explicit false-default `workflow_dispatch`; it may not
checkout code, request approval/secrets, or target self-hosted labels. Ordinary
push and PR checks must terminate using hosted jobs alone.

This is not authorization to run the old pending job. Canceling or preserving
that historical GitHub run is an administrative choice outside this repository
documentation task and does not change its evidence meaning.

## 4. What remains

The default four design-review iterations are exhausted. Iteration 4 scored
**8.9 / 10.0** with **CRITICAL 0, MAJOR 2, MINOR 1**, below the 9.7 PASS Gate.
It is not eligible for the conditional Iteration 5–6 extension because the
score is below 9.0 and regressed from Iteration 3's 9.1. The Run therefore
stops at the design Gate; this is not an implementation or acceptance PASS.

The exact remaining design findings are:

- **DR-I4-MAJ-01:** complete module-scope/top-level binding analysis for every
  protected name, including all valid Python rebinding forms, with exact
  violation mutants for each supported category.
- **DR-I4-MAJ-02:** an exact whole-file allowed-delta oracle that permits only
  the pinned new v2 constants, `_build_baseline` replacement, and `main()`
  exit-expression change while rejecting every other source change.
- **DR-I4-MIN-01:** remove or replace the impossible whole-workflow broad
  `self-hosted` grep with the parsed `runs-on`, exact-shape, and executable-
  surface checks already specified by the design.

The approved specification still requires a normal implementation and review:
v2 baseline schema, strict migration behavior, independent readiness algebra,
workflow contract change, adversarial tests, support/runbook wording, hosted
pre-merge gates, and exact-merge post-merge artifact validation. Until those
land, current artifacts retain v1 blocked semantics and no new readiness claim
is made.

Native/Ollama reactivation is possible only through a future explicit policy
reversal and a new reviewed milestone with owned infrastructure and security
evidence. Historical tooling and receipts may be retained for audit, but cannot
participate in current release algebra.

No implementation, product code, workflow, test, Git, or release work was
performed in this terminal-stop update. No live, native, self-hosted, Ollama,
protected-environment, or other acceptance workload was executed. The Run may
resume only after the user explicitly reapproves either a new design-iteration
cycle or a revised scope; an automatic Iteration 5 is prohibited.

## 5. Prohibited conclusions

- Do not call run `31825950604` or its pending job PASS.
- Do not map `NOT_ADOPTED` to `PASS`, `SKIPPED`, or `WAIVED`.
- Do not set `native_linux_release_ready`, `full_production_release_ready`, or
  `overall_release_ready` true.
- Do not infer hosted readiness from policy constants or an assembler claim;
  require all four deterministic producers and independent checking.
- Do not rewrite or delete M3, M4.1, M4.2, or M4.3 historical receipts.
- Do not perform live/self-hosted execution as acceptance for this change.

## 6. Resumption record (2026-08-15) — design Gate later PASSED

§1-§5 above are preserved unchanged as the historical record of the
Iteration-4 design-review stop; they are not rewritten. This section
records what happened after that stop, which the header fields above now
point to.

The user approved a new design-iteration cycle (the "라우팅 단순화 재설계
사이클" mechanism was not used; this is an ordinary continuation of the
same design document under `milestone_dev_orchestration_guide.md` §Gate
rules). Design.md iteration 5 closed DR-I4-MAJ-01/DR-I4-MAJ-02/DR-I4-MIN-01
by replacing the enumerated "protected symbol" allowlist with the
whole-file default-deny `audit_exact_allowed_delta` oracle (§3.1a). A
subsequent Recovery Cycle (separate from, and following, the numbered
Design_Review_Iteration_1–4 series) ran three further iterations:

- **Recovery Cycle 1, Iteration 1** found DR-RC1-I1-MAJ-01 (the oracle's
  slice generator omitted decorator lines) and closed it via
  `_statement_source_slice`'s decorator-span extension.
- **Recovery Cycle 1, Iteration 2** found DR-RC1-I2-MAJ-01 (the oracle's
  `base_source`/`current_source` strings were already-decoded `str`, so a
  shebang/encoding-cookie change never reached the comparison) and closed
  it via the §3.1b raw-bytes preamble boundary
  (`_source_preamble`/`audit_exact_allowed_delta_bytes`).
- **Recovery Cycle 1, Iteration 3** scored **PASS — 9.8/10.0** (CRITICAL 0,
  MAJOR 0, MINOR 1 — [review](Design_Review_Recovery_Cycle_1_Iteration_3.md)),
  clearing the 9.7 Gate. The one remaining MINOR, DR-RC1-I3-MIN-01
  (`_source_preamble` under-modeled a valid comment-first second-line PEP
  263 cookie), was explicitly non-blocking for implementation and was
  closed during implementation (Design.md §19).

Implementation then proceeded exactly as this report's §2-§3 require:
baseline schema v2 with `NOT_ADOPTED` never mapped to `PASS`,
`hosted_release_ready` derived only from same-run deterministic producer
evidence, `native_linux_release_ready`/`full_production_release_ready`/
`overall_release_ready` fixed `false`, `m3-live-regression-gate`
constrained to an explicit false-default `workflow_dispatch` opt-in with
no checkout/secrets/environment/self-hosted label, and no live/native/
Ollama/self-hosted command executed at any point. See
[Traceability.md](Traceability.md) for the requirement-to-implementation
matrix and current pre-merge state.
