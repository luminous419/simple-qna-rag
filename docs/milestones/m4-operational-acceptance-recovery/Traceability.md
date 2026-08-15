# M4 Operational Acceptance Recovery policy traceability

Status: **POLICY APPROVED / IMPLEMENTATION COMPLETE (PRE-MERGE)**  
Claim boundary: **hosted/OCI may become ready; native/full/overall remain false**
Design Gate: **PASS — Recovery Cycle 1, Iteration 3, 9.8/10.0** ([review](Design_Review_Recovery_Cycle_1_Iteration_3.md));
DR-RC1-I3-MIN-01 closed by implementation ([Design.md §19](Design.md)).

## 1. Requirement-to-change matrix

| Requirement | Implementation | Verification | Current state |
|---|---|---|---|
| M4-OAR-REQ-001 typed policy | `assemble_m4_evidence.py` v2 constants/`assemble`; `check_m4_baseline.py` schema dispatch | Exact-key/enums/policy tests; reject `WAIVED`, live `PASS`, and mixed schemas | Implemented; `tests/unit/test_assemble_m4_evidence.py`/`test_check_m4_baseline.py` pass locally |
| M4-OAR-REQ-002 readiness algebra | Assembler output and independent checker recomputation | Four-producer PASS/each failure variant; true native/full/overall adversarial cases | Implemented; M4.3 PASS preserved, hosted readiness now evidence-derived and independently rechecked |
| M4-OAR-REQ-003 migration | v2 writer; explicit `--allow-legacy-v1`; hosted expectation flags | v1 default rejection, explicit legacy acceptance, immutable historical fixture | Implemented; frozen-blocked v1 legacy path enforced unconditionally under `--allow-legacy-v1` |
| M4-OAR-REQ-004 workflow | `.github/workflows/ci.yml`; `test_ci_workflow_contract.py` | Push/PR terminal without self-hosted/environment; deterministic needs only; optional dispatch stub false by default | Implemented; `m3-live-regression-gate` is workflow_dispatch-opt-in-only, `ubuntu-latest`, no checkout/secrets/environment |
| M4-OAR-REQ-005 support boundary | Roadmap, Problem, release/deployment/user docs | Markdown links; terminology search; no UI certification claim | Implemented; README/deployment_runbook/recovery_runbook updated, CI_Acceptance_Runbook.md superseded banner added, doc-audit test passes |
| M4-OAR-REQ-006 evidence/history | No live commands; historical docs/artifacts untouched | Diff audit; M4.3 tests/hashes; no receipt fabrication | Preserved; no live/native/self-hosted command executed during implementation |

Pending: Codex code review, `git commit`/push/PR/merge, and the post-merge
exact-SHA baseline verification in §6.1 of
[deployment_runbook.md](../../operations/deployment_runbook.md#61-hostedoci-baseline-verification-pre-deployment)
(none of these can occur before this dispatch's `commit`/`push` boundary,
per this worker's scope).

## 2. State transition

```text
HISTORICAL v1 (immutable artifact semantics)
  deterministic_status=PASS
  m3_live_regression=NOT_RUN
  m41_operational=BLOCKED
  M4.1_BLOCKED=true
  overall_release_ready=false

POLICY-CHANGE v2 before deterministic evidence is complete
  native_linux_ollama=NOT_ADOPTED
  m3_live_regression=NOT_ADOPTED
  m41_operational=NOT_ADOPTED
  hosted_release_ready=false
  native_linux_release_ready=false
  full_production_release_ready=false
  overall_release_ready=false

VERIFIED hosted/OCI v2
  four deterministic producers=PASS
  deterministic_status=PASS
  hosted_release_ready=true
  native_linux_release_ready=false
  full_production_release_ready=false
  overall_release_ready=false
```

The transition is a schema/policy migration, not a mutation of v1 and not a
live acceptance event. M4.3 PASS carries forward only through its existing
typed receipt and hashes.

## 3. Historical receipts retained

| Receipt | Truth preserved |
|---|---|
| M4.1 exception and stop report | Operational acceptance was not completed; later milestones proceeded under a bounded exception. |
| Run `31825950604` at `adda175...e03d28` | Ordinary hosted jobs passed; protected live job was pending; no native/Ollama PASS. |
| M4.2 final verification | Deterministic safe-serving evidence passed; live/Ollama was not run. |
| [M4.3 Acceptance Report](../m4.3-artifact-deployment-safety/Acceptance_Report.md) | Deterministic profile and artifact/deployment safety PASS remain valid. |
| Former recovery stop report | No runner/host authority existed; no approval, registration, or live execution occurred. |

## 4. Acceptance evidence map

| Gate | Evidence | Accepted conclusion |
|---|---|---|
| Pre-merge Python | full `pytest -q`, dataset validation | hosted deterministic input only |
| Pre-merge frontend | `npm ci`, `npm test`, vendor sync/diff | hosted deterministic input only |
| Pre-merge OCI | image build plus container smoke/layer evidence | OCI deterministic input only |
| Pre-merge M4.3 | deterministic profile repeat 10 seed 4303 plus negative control | M4.3 PASS preserved |
| Workflow contract | static YAML tests and terminal ordinary run | no pending self-hosted dependency |
| Post-merge baseline | fresh artifact plus strict v2 checker | hosted/OCI ready only |

No native Linux, self-hosted runner, protected environment, Ollama endpoint, or
live corpus is an acceptance input.

## 5. Documentation validation

```bash
python scripts/check_markdown_links.py
git diff --check
rg -n "overall_release_ready=true|native_linux_release_ready=true|m3_live_regression=PASS" \
  docs/milestones/m4-operational-acceptance-recovery docs/Roadmap.md docs/Problem.md
```

Any positive match that presents those strings as an accepted current state is
a documentation failure; adversarial examples clearly labeled as rejected are
allowed.
