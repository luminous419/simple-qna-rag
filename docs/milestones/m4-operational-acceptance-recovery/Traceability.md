# M4 Operational Acceptance Recovery policy traceability

Status: **RELEASED — hosted/OCI baseline verified on exact merge SHA**
Claim boundary: **hosted/OCI is ready; native/full/overall remain false**
Design Gate: **PASS — Recovery Cycle 1, Iteration 3, 9.8/10.0** ([review](Design_Review_Recovery_Cycle_1_Iteration_3.md));
DR-RC1-I3-MIN-01 closed by implementation ([Design.md §19](Design.md)).

## 0. Release evidence (exact identity binding)

| Item | Value |
|---|---|
| PR | [#19](https://github.com/luminous419/simple-qna-rag/pull/19) `agent/m4-operational-acceptance-recovery` → `master` |
| Implementation commit | `f6ff86cd920f97e732973a6141c0d17cd16c3a1c` |
| Hosted-CI fix commit | `24dfc8b49ddcebbd766202be3e1934e524c56e18` — `fetch-depth: 0` added to the `python-tests` job's checkout after the first PR-head hosted run (`31889309407`) failed `python-tests`: `audit_exact_allowed_delta`'s `git show <base_revision>:...` calls require the pinned base commit `adda1759754b56b514b3ab6252c2dc1032e03d28` to exist locally, which `actions/checkout@v4`'s default shallow clone (depth 1) does not provide. No test/product logic changed. |
| PR-head hosted CI (post-fix) | run [`31889748729`](https://github.com/luminous419/simple-qna-rag/actions/runs/31889748729), commit `24dfc8b49ddcebbd766202be3e1934e524c56e18` — `python-tests`/`frontend-tests`/`container`/`m43-deterministic`/`m4-assemble` all `success`; `m3-live-regression-gate` `skipped` (never scheduled on `pull_request`) |
| Merge SHA | `8e203abe5ed6e17e6e8b6e292975121749374a52` (merge commit, `gh pr merge --merge`) |
| Post-merge hosted CI (exact merge SHA) | run [`31890598812`](https://github.com/luminous419/simple-qna-rag/actions/runs/31890598812), event `push`, `head_sha=8e203abe5ed6e17e6e8b6e292975121749374a52`, `run_attempt=1` — `python-tests`/`frontend-tests`/`container`/`m43-deterministic`/`m4-assemble` all `success`; `m3-live-regression-gate` `skipped` (never scheduled on ordinary `push`) |
| Downloaded artifact | `m4-baseline` from run `31890598812`, `git_sha=8e203abe5ed6e17e6e8b6e292975121749374a52` in the artifact body matches the run's own `head_sha` |
| Checker verdict | `python scripts/check_m4_baseline.py --candidate m4-baseline.json --expect-hosted-release-ready --require-identity-binding --expect-sha 8e203abe5ed6e17e6e8b6e292975121749374a52 --expect-run-id 31890598812 --expect-run-attempt 1 --expect-workflow-path .github/workflows/ci.yml --expect-event push` → `{"ok": true, "issues": []}`, exit 0. Adversarial sanity check (wrong `--expect-sha`) → `{"ok": false, "issues": ["identity_sha_mismatch:..."]}`, exit 1, confirming fail-closed behavior at the exact artifact used for this release. |
| Artifact literals | `deterministic_status=PASS`, `operational_status=NOT_ADOPTED`, `gates.m3_live_regression=NOT_ADOPTED`, `gates.m41_operational=NOT_ADOPTED`, `M4.1_BLOCKED=false`, `hosted_release_ready=true`, `native_linux_release_ready=false`, `full_production_release_ready=false`, `overall_release_ready=false` — exact match to Plan.md §6's required literal block |
| Branch protection | `gh api repos/luminous419/simple-qna-rag/branches/master/protection` → `404 Branch not protected` (informational; Requirement M4-OAR-REQ-004.3 is phrased "SHOULD", not "MUST") |
| Release claim | **hosted/OCI release ready.** Not "production ready." Not "overall release ready." `native_linux_release_ready`, `full_production_release_ready`, and `overall_release_ready` are `false`; native Linux/Ollama remains `NOT_ADOPTED`. |

No live, native Linux, Ollama, self-hosted runner, or protected-environment execution occurred at any point in this release.

## 1. Requirement-to-change matrix

| Requirement | Implementation | Verification | Current state |
|---|---|---|---|
| M4-OAR-REQ-001 typed policy | `assemble_m4_evidence.py` v2 constants/`assemble`; `check_m4_baseline.py` schema dispatch | Exact-key/enums/policy tests; reject `WAIVED`, live `PASS`, and mixed schemas | Implemented; `tests/unit/test_assemble_m4_evidence.py`/`test_check_m4_baseline.py` pass locally |
| M4-OAR-REQ-002 readiness algebra | Assembler output and independent checker recomputation | Four-producer PASS/each failure variant; true native/full/overall adversarial cases | Implemented; M4.3 PASS preserved, hosted readiness now evidence-derived and independently rechecked |
| M4-OAR-REQ-003 migration | v2 writer; explicit `--allow-legacy-v1`; hosted expectation flags | v1 default rejection, explicit legacy acceptance, immutable historical fixture | Implemented; frozen-blocked v1 legacy path enforced unconditionally under `--allow-legacy-v1` |
| M4-OAR-REQ-004 workflow | `.github/workflows/ci.yml`; `test_ci_workflow_contract.py` | Push/PR terminal without self-hosted/environment; deterministic needs only; optional dispatch stub false by default | Implemented; `m3-live-regression-gate` is workflow_dispatch-opt-in-only, `ubuntu-latest`, no checkout/secrets/environment |
| M4-OAR-REQ-005 support boundary | Roadmap, Problem, release/deployment/user docs | Markdown links; terminology search; no UI certification claim | Implemented; README/deployment_runbook/recovery_runbook updated, CI_Acceptance_Runbook.md superseded banner added, doc-audit test passes |
| M4-OAR-REQ-006 evidence/history | No live commands; historical docs/artifacts untouched | Diff audit; M4.3 tests/hashes; no receipt fabrication | Preserved; no live/native/self-hosted command executed during implementation |

Complete: Codex code review (PASS, [Code_Review_Iteration_2.md](Code_Review_Iteration_2.md)),
pre-merge acceptance (PASS, [Acceptance_Report.md](Acceptance_Report.md)),
`git commit`/push/PR #19/merge, and the post-merge exact-SHA baseline
verification in §6.1 of
[deployment_runbook.md](../../operations/deployment_runbook.md#61-hostedoci-baseline-verification-pre-deployment)
— see §0 above for the exact PR/commit/merge-SHA/run-ID evidence.

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
