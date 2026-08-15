# M4 Operational Acceptance Recovery policy-change requirements

Status: **APPROVED POLICY CHANGE / IMPLEMENTATION COMPLETE (PRE-MERGE)**  
Design Gate: **PASS — Recovery Cycle 1, Iteration 3, 9.8/10.0** ([review](Design_Review_Recovery_Cycle_1_Iteration_3.md), DR-RC1-I3-MIN-01 closed in implementation)  
Decision date: **2026-08-15**  
Historical baseline: merge SHA `adda1759754b56b514b3ab6252c2dc1032e03d28`, PR #18 run `31825950604`  
Preserved evidence: **M4.3 deterministic PASS**

## 1. Decision and historical record

The product will ship only the verified ordinary hosted Python/frontend and OCI
container scope. Native Linux x86_64, self-hosted GitHub Actions, local Ollama
`gpt-oss:20b`, and the protected live M3 regression are permanently outside the
adopted release scope. Their typed state is `NOT_ADOPTED`, never `PASS`.

This supersedes the earlier recovery proposal to acquire a host and execute the
live job. That proposal remains visible in Git history and in
[Stop_Report.md](Stop_Report.md); it was stopped because repository runner count
was zero and no authorized native host owner existed. Run `31825950604` had
ordinary hosted jobs PASS and `m3-live-regression-gate` pending. The environment
and reviewer configuration, the pending job, and the old M4.1 exception are
historical receipts only: none proves live execution.

No native/live/self-hosted execution is authorized by this specification. This
task changes documentation only; implementation requires a separately reviewed
change.

## 2. Normative typed state model

### M4-OAR-REQ-001 — Closed enums and compatibility

1. Baseline schema `m4-baseline-v2`, version `2.0.0`, MUST use:
   - deterministic gates: `PASS | FAIL`;
   - adopted-scope status: `PASS | FAIL`;
   - non-adopted capability state: `NOT_ADOPTED`;
   - booleans for readiness fields.
2. `NOT_ADOPTED` means deliberately excluded from the supported product and
   release claim. It is not success, failure, skipped execution, pending work,
   or a temporary waiver. `WAIVED` MUST NOT be emitted because it could imply an
   unmet adopted requirement was excused.
3. Required top-level v2 fields are the v1 fields plus
   `support_policy`, `hosted_release_ready`, `native_linux_release_ready`, and
   `full_production_release_ready`. The legacy `overall_release_ready` field is
   retained as a compatibility alias for full-production readiness and MUST
   equal `full_production_release_ready`.
4. `support_policy` has exactly:
   `schema="m4-support-policy-v1"`, `adopted_scope="HOSTED_OCI"`,
   `native_linux_ollama="NOT_ADOPTED"`, and
   `decision_date="2026-08-15"`.
5. The gate keys `m3_live_regression` and `m41_operational` remain present for
   compatibility but both MUST equal `NOT_ADOPTED`. `operational_status` MUST
   equal `NOT_ADOPTED`; `M4.1_BLOCKED` MUST be `false` because there is no
   unfinished adopted-scope requirement. This does not convert M4.1 history to
   PASS.

### M4-OAR-REQ-002 — Truthful readiness algebra

The checker MUST independently recompute, and the assembler MUST emit:

```text
deterministic_status = PASS iff all four deterministic gates are PASS, else FAIL
hosted_release_ready = (deterministic_status == PASS)
native_linux_release_ready = false
full_production_release_ready = false
overall_release_ready = full_production_release_ready  # compatibility alias
```

`hosted_release_ready` may become true only from the four same-run deterministic
producers: `python-tests`, `frontend-tests`, `container`, and
`m43-deterministic`. Policy constants, absent jobs, historical live receipts,
manual input, environment configuration, or self-reported aggregate fields
MUST NOT make it true. `native_linux_release_ready`,
`full_production_release_ready`, and `overall_release_ready` MUST never be true
under the `HOSTED_OCI` policy.

### M4-OAR-REQ-003 — Baseline migration and backward compatibility

1. The assembler writes v2 only. It preserves existing producer receipt schema
   `m43-producer-receipt-v1`, payload validation, provenance binding, and M4.3
   receipt hashes unchanged.
2. The checker accepts v2 by default. A v1 baseline is accepted only with a
   named compatibility mode (`--allow-legacy-v1`) and its original fail-closed
   algebra: live `NOT_RUN`, M4.1 `BLOCKED`, `M4.1_BLOCKED=true`, and
   `overall_release_ready=false`. Compatibility mode MUST NOT migrate, rewrite,
   or call v1 hosted-ready.
3. Unknown schema/version, mixed v1/v2 fields, extra keys, `WAIVED`, live
   `PASS`, policy/status disagreement, or any true native/full/overall flag
   fails closed. Historical artifacts remain immutable and interpretable.
4. Replace `--expect-operational-blocked` for v2 with
   `--expect-hosted-release-ready` and `--expect-hosted-not-ready`. The legacy
   option remains valid only with `--allow-legacy-v1`; incompatible combinations
   are usage errors.

### M4-OAR-REQ-004 — Workflow contract

1. Ordinary `push` and `pull_request` runs MUST schedule only hosted jobs and
   MUST terminate without a permanently queued/pending self-hosted job.
2. `m3-live-regression-gate` MUST be removed from ordinary dependency paths.
   The preferred implementation deletes the executable live job. If a future
   reactivation stub is retained, it MUST be `workflow_dispatch`-only with an
   explicit opt-in input whose default is false, and the current policy MUST
   make the job resolve immediately on hosted infrastructure without checkout,
   secrets, environment approval, or self-hosted labels. It MUST NOT execute
   live code under this policy.
3. Baseline assembly depends only on the four deterministic producers and uses
   the hosted-ready checker. Branch protection SHOULD require only deterministic
   hosted checks documented in the plan.
4. Future reactivation requires a new policy decision, requirements/design
   review, threat model, owned native runner, and separate implementation. It
   MUST NOT be enabled by toggling an undocumented secret or repository variable.

### M4-OAR-REQ-005 — User, documentation, and support boundary

1. UI and public docs MUST describe supported deployment as hosted Python
   service/frontend plus OCI container behavior verified by deterministic
   tests. They MUST NOT advertise native Linux/Ollama production readiness.
2. Health/readiness UI MUST report runtime health only; it MUST NOT expose
   `hosted_release_ready` or infer certification from a reachable Ollama server.
3. Runbooks MUST remove native runner provisioning and live approval from the
   release checklist, document artifact download plus v2 checker commands, and
   state that native Ollama use is unsupported/best-effort with no release SLA.
4. Support triage may reproduce hosted/OCI issues. Native Linux/Ollama-specific
   incidents are outside the supported matrix unless also reproducible in the
   adopted scope. Existing generic local-development Ollama instructions may
   remain only when clearly labeled development-only.

### M4-OAR-REQ-006 — Security, evidence, and history

No workflow or test for this change may contact Ollama, register/use a
self-hosted runner, request environment approval, access canonical live data,
or manufacture a live receipt. Preserve M3 baselines, run `31825950604`, M4.1
exception/stop reports, M4.2 receipts, and M4.3 PASS documents without rewriting
their historical conclusions.

## 3. Acceptance criteria

The policy implementation is accepted only when all hosted pre-merge commands
pass, adversarial state-algebra and workflow-contract tests pass, ordinary push
runs reach a terminal conclusion with no self-hosted dependency, the merged-SHA
post-merge baseline validates as v2 with `hosted_release_ready=true`, and the
three broader readiness fields remain false. This milestone closes the former
operational blocker by scope removal, not by operational PASS.
