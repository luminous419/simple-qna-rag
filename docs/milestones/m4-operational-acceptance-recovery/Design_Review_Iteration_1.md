# M4 Operational Acceptance Recovery — Design Review Iteration 1

Reviewer: Fresh Codex independent design reviewer  
Date: 2026-08-15  
Reviewed artifact: [Design.md](Design.md)  
Normative inputs: [Requirement.md](Requirement.md), [Plan.md](Plan.md),
[Traceability.md](Traceability.md), [Stop_Report.md](Stop_Report.md), repository
implementation/tests/workflow, and preserved M4.3 evidence.

## 1. Gate decision

**FAIL — 7.8 / 10.0**

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 5 |
| MINOR | 1 |
| TRIVIAL | 0 |

The core v2 state separation is sound: `NOT_ADOPTED` is not equated with
`PASS`; the two compatibility gates are fixed to `NOT_ADOPTED`; hosted
readiness is derived only from the four deterministic producer statuses; and
native/full/overall readiness is fixed false. The Gate nevertheless fails
because the proposed checker does not authenticate the baseline's run/SHA and
payload-derived provenance, legacy v1 is not restricted to the required frozen
blocked state, workflow tests permit hidden live execution, the support/runbook
plan leaves an active provisioning checklist in place, and rollback explicitly
restores the permanently pending self-hosted workflow.

PASS requires `CRITICAL=0`, `MAJOR=0`, and score `>=9.7`; this iteration does
not qualify.

## 2. Findings

### DR-I1-MAJ-01 — Legacy v1 compatibility accepts states other than the required frozen blocked semantics

**Gate:** Pre-merge Code Quality / schema compatibility  
**References:** Requirement.md:80-87; Design.md:509-573, 575-594, 805-823;
current `scripts/check_m4_baseline.py`:73-164.

Requirement M4-OAR-REQ-003.2 does not merely ask to retain the old general
algebra. It defines the accepted legacy state as live `NOT_RUN`, M4.1
`BLOCKED`, `M4.1_BLOCKED=true`, and `overall_release_ready=false`. The proposed
`_check_v1_legacy` preserves the generic old checker and only enforces those
fixed values when `expect_operational_blocked` is optionally supplied. Thus
`--allow-legacy-v1` alone can accept, for example, both operational gates
`PASS`, `operational_status=PASS`, and `overall_release_ready=true`. It also
never relates `M4.1_BLOCKED` to the gates unless the optional expectation flag
is set. That contradicts the named compatibility contract and makes the legacy
mode broader than the historical artifact meaning recorded in
Traceability.md:19-25 and the M4.3 evidence.

**Exact fix:** Make `_check_v1_legacy` unconditionally require
`gates.m3_live_regression == "NOT_RUN"`, `gates.m41_operational == "BLOCKED"`,
`operational_status == "BLOCKED"`, `M4.1_BLOCKED is True`, and
`overall_release_ready is False`, in addition to the existing four-producer
recalculation. Keep `--expect-operational-blocked` only as a compatibility CLI
assertion/alias if needed; it must not be what activates safe semantics. Add
mutants for live `PASS`, live `SKIPPED`, M4.1 `PASS`, `M4.1_BLOCKED=false`, and
overall `true`, all using only `--allow-legacy-v1`.

### DR-I1-MAJ-02 — The “independent checker” does not bind the artifact to the expected merge SHA/run or recheck provenance aliases

**Gate:** Pre-merge Code Quality / post-merge checker executability  
**References:** Requirement.md:57-73, 77-87, 126-132, 136-140;
Plan.md:35-38, 47-63, 95-114; Design.md:60-65, 362-418, 421-507,
987-1008, 1087-1102; current `scripts/assemble_m4_evidence.py`:97-127,
167-275; current `scripts/check_m4_baseline.py`:73-164.

The assembler correctly binds producer receipts to expected SHA, run ID,
attempt, workflow path, and event before hashing payload bytes. The proposed
checker, however, validates only producer status variants and hash *shape*. It
does not accept expected identity arguments; validate `git_sha` or
`workflow_run` types/values; bind the downloaded candidate to the merge SHA and
run selected by the operator; or recompute the top-level
`image_digest`/`m43_deterministic_receipt_sha256` aliases from the corresponding
producer payload hashes. It likewise only checks that receipt/payload hashes
look like 64 hex characters. A baseline can therefore be copied from another
run, have its run metadata replaced arbitrarily, or have top-level provenance
aliases changed while still passing `--expect-hosted-release-ready`. The
runbook's instruction to download from an “exact merge-SHA workflow run” is a
human convention, not a fail-closed CLI contract.

**Exact fix:** Add checker CLI arguments for the expected SHA, run ID, run
attempt, workflow path, and event (or a single reviewed identity receipt that
contains them), require them for the hosted-ready post-merge assertion, and
strictly validate `workflow_run`'s exact key set and scalar types. Recompute
`image_digest` from `producers.container.payload_hashes["container_smoke.json"]`
and `m43_deterministic_receipt_sha256` from
`producers["m43-deterministic"].payload_hashes["m43.json"]`, including the
non-OK/null cases. Update the deployment command and tests with cross-SHA,
cross-run, attempt, workflow, event, malformed metadata, and top-level alias
tampering mutants. If payload-byte verification is intentionally assembler-only,
state that trust boundary precisely rather than calling the baseline-only
checker independent provenance verification.

### DR-I1-MAJ-03 — Workflow contract tests allow hidden secrets, checkout alternatives, and live commands

**Gate:** Pre-merge Code Quality / workflow safety  
**References:** Requirement.md:93-109, 126-132; Plan.md:39-43, 47-57;
Design.md:716-759, 902-920.

The proposed YAML job itself is harmless, but its mutation tests are too weak
for the normative “no live execution” boundary. They forbid only an
`actions/checkout` `uses:` prefix, a top-level `environment`, and self-hosted
runner labels. A second `run:` step could invoke `curl`, Ollama, a repository
script from a preinstalled path, or `${{ secrets.* }}`; another `uses:` action
could fetch/execute code; job/step `env` could reference secrets; and all five
listed tests would still pass. The test named “has no checkout step” also does
not exclude non-`actions/checkout` repository acquisition.

**Exact fix:** Make the stub an exact-shape contract: exact job key set, exactly
one step, no `uses`, no job/step `env`, no `with`, no permissions elevation,
and an exact allowlisted informational script (or delete the job entirely, as
the Plan prefers). Add source-level assertions that the stub contains no
`${{ secrets`, `RUN_LIVE_LLM_TESTS`, Ollama endpoint/model tokens, repository
script invocation, network command, checkout/fetch/clone, self-hosted label, or
environment. Include mutants that add each forbidden surface. Also assert that
ordinary triggers and every ordinary job dependency closure contain only the
five adopted hosted jobs, rather than testing only `m4-assemble.needs`.

### DR-I1-MAJ-04 — The support plan preserves an active native-runner provisioning runbook contrary to REQ-005

**Gate:** Pre-merge Code Quality / support boundary  
**References:** Requirement.md:111-124; Plan.md:23-28;
Design.md:91-95, 921-1047; current
`docs/milestones/m4.1-configuration-observability/CI_Acceptance_Runbook.md`:132-138,
164-195, 323-402, 560-561.

Design §8.4 declares the M4.1 CI Acceptance Runbook edit-forbidden because it is
historical. The file is nevertheless named and written as an actionable
runbook and contains current commands/checklists to provision
`self-hosted,ollama-m3`, configure the protected environment, trigger the live
job, and require its receipt. Requirement M4-OAR-REQ-005.3 specifically says
runbooks MUST remove native runner provisioning and live approval from the
release checklist. Historical truth can be preserved without leaving an
unqualified active procedure that directly conflicts with the new supported
release path.

**Exact fix:** Preserve the historical body but add a prominent superseded,
non-executable historical-record banner linking to the new hosted/OCI runbook,
or move/copy it into an explicitly archival location while removing it from
current release navigation. The new deployment/release checklist must be the
only normative current procedure and must contain the exact v2 identity-bound
checker invocation from MAJ-02. Expand the documentation audit beyond three
positive-state strings to find actionable runner provisioning, environment
approval, live-job, and unlabeled Ollama instructions; classify historical
matches through an explicit allowlist/banner test.

### DR-I1-MAJ-05 — Rollback restores the exact permanently pending live workflow that this milestone removes

**Gate:** Recovery / ordinary-run termination  
**References:** Requirement.md:93-109; Plan.md:119-126;
Stop_Report.md:37-48; Design.md:1104-1119.

Design §11 step 2 says rollback restores the pre-design
`m3-live-regression-gate`, including the `push:master` path and absent
self-hosted capacity. Calling that restoration “not enabling” does not change
the effect: every ordinary master push again schedules the unavailable job and
the run can remain pending indefinitely. This contradicts the Plan's rollback
rule to retain v1 blocked semantics without enabling live as a workaround and
violates the ordinary-run termination requirement during the precise failure
window when rollback is needed.

**Exact fix:** Define rollback components independently. Schema/checker rollback
may restore the v1 writer/checker and blocked artifact semantics, but workflow
rollback must retain deletion or the harmless hosted no-op stub; it must never
restore an ordinary-triggered self-hosted job. Provide a tested rollback matrix
for (a) schema-only failure, (b) workflow-only failure, and (c) both, with the
invariant that ordinary push/PR always terminates and no live path becomes
executable.

### DR-I1-MIN-01 — The assembler “byte-for-byte unchanged” claim and test do not establish byte-level preservation

**Gate:** Pre-merge Code Quality / evidence regression  
**References:** Design.md:28-35, 83-90, 208-212, 825-853, 1130-1145;
Plan.md:31-34.

The design repeatedly says four assembler functions will be “byte-for-byte” or
“one character” unchanged, but the proposed test
`test_assemble_v2_producers_and_m43_receipt_sha_shape_unchanged` checks output
shape/value only, and the checklist's `_evaluate_producer 이하` diff boundary is
ambiguous because the protected functions span multiple regions. A refactor
could materially alter receipt verification while retaining the tested happy
path.

**Exact fix:** Replace the rhetorical byte-level promise with an auditable
symbol list and a scoped diff command against the reviewed base revision, or
accept semantic preservation and add the complete existing adversarial matrix
to the required gate. Record the base SHA and exact protected ranges/symbols;
do not claim byte equality from output-shape tests.

## 3. Areas reviewed with no blocking finding

- **Truthful `NOT_ADOPTED`:** Design.md:54-59 and 441-448 correctly prevent
  either excluded gate from being represented as `PASS`, `WAIVED`, `SKIPPED`,
  or another enum value.
- **V2 readiness algebra:** Design.md:165-188 and 461-506 correctly recompute
  deterministic status from the four producer statuses, derive hosted
  readiness from that result, and force native/full false. The overall alias
  cannot validate true because full readiness is independently required to be
  the literal `False`.
- **Ordinary workflow intent:** The proposed job condition in Design.md:730-741
  removes push/PR selection and uses hosted infrastructure with no checkout or
  environment. The design defect is test strength and rollback, not the shown
  steady-state YAML.
- **Historical M4.3 meaning:** The design does not rewrite the preserved M4.3
  deterministic PASS or claim that the historical protected job ran. The
  reviewed M4.3 Acceptance/Implementation/Traceability evidence consistently
  records deterministic PASS while the former live/M4.1 state remained
  `NOT_RUN`/`BLOCKED` and overall readiness false.
- **Product/UI scope:** No runtime UI change is proposed, and the current
  repository has no `release_ready` exposure under `src/` or `web/`.

## 4. Required next iteration

Revise Design.md to close all five MAJOR findings, update the exact test/mutant
tables and CLI/runbook contracts, and correct rollback. A second fresh review
should then verify the revised design against the unchanged Requirement/Plan
and current repository state. No implementation, live execution, self-hosted
runner action, environment approval, or historical evidence rewrite is needed
to close this design-review Gate.
