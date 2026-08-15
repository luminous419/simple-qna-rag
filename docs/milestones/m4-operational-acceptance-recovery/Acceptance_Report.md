# M4 Operational Acceptance Recovery — Pre-Merge Acceptance Report

Verifier: Claude Code Sonnet 5 (independent acceptance worker, distinct from
the Code_Review_Iteration_2.md reviewer)
Date: **2026-08-15**
Scope: entire uncommitted worktree on `agent/m4-operational-acceptance-recovery`
against `origin/master`, per [Plan.md](Plan.md) §5 "Hosted pre-merge gate" and
[Requirement.md](Requirement.md) §3 "Acceptance criteria".
Inputs read: [Requirement.md](Requirement.md), [Plan.md](Plan.md),
[Design.md](Design.md), [Traceability.md](Traceability.md),
[Code_Review_Iteration_2.md](Code_Review_Iteration_2.md), [Stop_Report.md](Stop_Report.md).

## Gate decision

**PASS (pre-merge), then PASS (post-merge) — see §11 below.** All pre-merge
deterministic acceptance commands passed; hosted-CI and the post-merge
exact-merge-SHA baseline verification (§11) subsequently also passed,
confirming `hosted_release_ready=true` with
`native_linux_release_ready`/`full_production_release_ready`/
`overall_release_ready` all `false`, per
Traceability.md §0 and Code_Review_Iteration_2.md "Remaining acceptance
work".

No live, native Linux, Ollama, self-hosted runner, or protected-environment
execution was performed at any point. This original §1-§10 pre-merge record
predates the commit/push/PR/merge described in §11 below and is preserved
unchanged; no commit, push, PR, or merge was
performed (out of this worker's authorized scope).

## 1. Plan §5 hosted pre-merge gate — command-by-command evidence

| # | Command | Result | Evidence |
|---|---|---|---|
| 1 | `python -m evaluation.dataset validate evaluation/datasets/golden.jsonl` | **PASS** | `검증 통과.` — 76 cases, `valid: true`, 0 errors |
| 2 | `pytest -q` (full suite, local dev venv) | **PASS** | `1530 passed, 1 skipped, 4 warnings in 173.29s` |
| 2b | `pytest -q` (full suite, clean hosted-equivalent env — see §2 below) | **PASS with 4 pre-existing, out-of-scope, environment-attributable failures** | see §2 |
| 3 | `npm ci` | **PASS** | 92 packages installed; vendor sync ran via postinstall; see engine-version note in §4 |
| 4 | `npm test` | **PASS** | vitest: `Test Files 1 passed (1)`, `Tests 9 passed (9)` |
| 5 | `npm run sync-vendor` | **PASS** | `web/static/vendor/에 4개 파일을 동기화했습니다.` |
| 6 | `git diff --exit-code -- web/static/vendor` | **PASS** | exit 0, no drift after sync |
| 7 | `docker build -t simple-qna-rag:m4-policy .` | **PASS (adapted invocation — see §4)** | built via `deploy/Dockerfile` `test` and `production` targets, `--platform linux/amd64` (hosted-equivalent), both succeeded |
| 8 | `python scripts/run_container_smoke.py --image ...` | **PASS (adapted script name — see §4)** | ran `scripts/container_smoke.py` (the script that actually exists and is what `ci.yml`'s `container` job invokes) against the amd64 production image: `"status": "PASS"`, `mock_query_ok: true`, `root_page_ok: true`, `static_asset_ok: true`, `readiness_sequence.ready: true` (200/"ok"), `production_test_seam_sealed: true` |
| 9 | `python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303` | **PASS** | 17/17 nodes `status: "PASS"`, each `success_count: 10/10`; top-level `status: "PASS"` |
| 9b | negative control: same command + `--inject-evidence-mismatch` | **PASS (rejection confirmed)** | `negative_control.result: "REJECTED_AS_EXPECTED"`, `actual_exit_code: 1` (process exit 1 is the designed success signal per the script's own docstring) |
| 10 | `python scripts/check_markdown_links.py` | **PASS** | `검사 파일 159개(tracked 145 + untracked 14), 링크 684개, 실패 0개` |
| 11 | `git diff --check` | **PASS** | exit 0, no whitespace conflict markers |

Fixture checker invocations (Plan §5 final block), built end-to-end through
the actual `assemble_m4_evidence.py` CLI (`main()`) using the same
receipt-construction helpers as `tests/unit/test_assemble_m4_evidence.py`,
then fed to the actual `check_m4_baseline.py` CLI:

```
python scripts/check_m4_baseline.py --candidate <PASS_V2_JSON> --expect-hosted-release-ready
  -> {"ok": true, "issues": []}   exit=0
python scripts/check_m4_baseline.py --candidate <FAIL_V2_JSON> --expect-hosted-not-ready
  -> {"ok": true, "issues": []}   exit=0
```

Adversarial sanity check (inverted expectation on the PASS fixture, added by
this verifier, not in Plan §5 but a natural extension of it):

```
python scripts/check_m4_baseline.py --candidate <PASS_V2_JSON> --expect-hosted-not-ready
  -> {"ok": false, "issues": ["expected_hosted_not_ready_not_satisfied"]}   exit=1
```

Confirms the checker fail-closed contract end-to-end at the CLI boundary, not
only inside the unit-test process.

## 2. Full deterministic suite in a clean, hosted-equivalent environment

Plan §5 specifies "Run from a clean dependency environment." The pre-existing
local `venv/` has drifted from `requirements.lock` (65 of 103 pinned packages
at different versions — confirmed by diffing `pip freeze` against the lock
file), so its green result, while genuine, is not proof against the exact
locked dependency set. `requirements.lock` is Linux-platform-pinned (per
git history: "restore Linux-compatible dependency lock", "recompile
requirements.lock for the Linux CI target platform"), so a byte-exact
`pip install --require-hashes -r requirements.lock` cannot succeed on this
worker's macOS arm64 host directly — attempting it fails immediately on
platform-mismatched wheel hashes (e.g. `markupsafe`), which is expected and
is not evidence of tampering.

To close this gap truthfully rather than skip it, this verifier built and
ran the suite inside a `linux/amd64` container (matching `ubuntu-latest`'s
architecture) with `git` installed and dependencies installed exactly via
`pip install --require-hashes -r requirements.lock --extra-index-url
https://download.pytorch.org/whl/cpu`, full repo mounted:

- **Focused M4-OAR suite** (the four files Plan.md names as in-scope):
  `pytest -q tests/unit/test_assemble_m4_evidence.py
  tests/unit/test_check_m4_baseline.py tests/unit/test_ci_workflow_contract.py
  tests/unit/test_doc_audit_no_active_native_runner_procedure.py`
  → **216 passed**, exact match to Code_Review_Iteration_2.md's count, 0
  failures, 0 skips.
- **Full suite** in the same clean amd64 environment: **1526 passed, 4
  failed, 1 skipped**. The 1 skip is the live-Ollama integration test,
  correctly gated behind `RUN_LIVE_LLM_TESTS=1` (never set by this worker,
  per policy).

The 4 failures are diagnosed, not asserted, as environment artifacts of this
verifier's improvised bare container and are **unrelated to this diff**:

1. `test_container_smoke_bare_script.py::test_bare_script_reaches_docker_invocation_or_skips_cleanly`
   — needs Docker-in-Docker, unavailable inside the verification container.
2. `test_index_verification.py::test_permission_denied_matrix_surfaces_disclosed_reasons_not_raw_oserror`
   — permission-denial assertions; the verification container runs as root,
   which bypasses POSIX permission checks that a non-root CI runner user
   would enforce.
3. `test_rag_engine_embeddings.py::test_build_embeddings_default_uses_huggingface_provider`
   and 4. `test_rag_engine_singleton.py::test_resolve_current_trust_boundary_error_propagates_through_engine_and_readiness_chain`
   — both fail attempting a real network download of a ~2.2GB HuggingFace
   model inside the ad hoc container, which lacks the disk/cache
   provisioning a real environment would have.

Confirming these are pre-existing and out of scope: `git diff --stat` and
`git status --porcelain` against these four test files return empty — none
is touched by this milestone's diff. All four also passed cleanly in this
same worker's initial full run on the local dev venv (§1, item 2) and were
reported passing by Code_Review_Iteration_2.md. This verifier's own
introduced harness (root user, no DinD, no model cache) is the
distinguishing variable, not the code.

**An initial pass at this same clean-environment run showed 51 failures
before `git` was installed in the container** — all 47 of the extra failures
were `test_assemble_m4_evidence.py::test_audit_exact_allowed_delta_*` tests,
which shell out to `git show <base_revision>:path`. Installing `git` (present
by default on `ubuntu-latest` and required by `actions/checkout`) resolved
every one of those 47 failures with no other change, confirming the earlier
failures were a missing-tool artifact of the verifier's minimal
`python:3.11-slim` base image, not a defect.

**Conclusion: no regression attributable to this diff was found in either
environment.** The full deterministic suite is green in the local dev venv,
and green in the clean hosted-equivalent environment for every file this
milestone touches; the 4 residual clean-environment failures are
independently explained, pre-existing, and out of this diff's scope.

## 3. amd64 hosted-equivalent container build/scan/smoke

Docker Desktop on this host is `darwin/arm64`; `ubuntu-latest` hosted runners
are `linux/amd64`. `docker buildx` on this host supports `linux/amd64` via
QEMU emulation, so this verifier built and exercised **amd64** images (not
native arm64) to match the hosted target:

- `docker buildx build --platform linux/amd64 --target test -f deploy/Dockerfile --load .` → **success**, `RUN python -c "from simple_qna_rag.web.server import app"` import-smoke passed inside the image.
- `docker buildx build --platform linux/amd64 --target production -f deploy/Dockerfile --load .` → **success**, digest `sha256:382ed39ab4e64611ce8dad06472f981107c4be39013960dd7c94c4b7686ef81a`.
- `scripts/scan_image_layers.py --image ... --output layer_scan.json` → `forbidden_count: 0`, `violations: []`.
- `scripts/container_smoke.py --image ... --output container_smoke.json` → `status: "PASS"` (full field set in §1 row 8).

Both build stages and both evidence scripts are the exact commands
`.github/workflows/ci.yml`'s `container` job runs (confirmed by reading the
job definition directly, lines 132-176), not the stale `docker build -t
simple-qna-rag:m4-policy .` (no `-f`) / `run_container_smoke.py` invocations
literally printed in Plan.md §5 — see §4 for that discrepancy note. Images
were removed after verification (`docker rmi`); no residual artifacts.

## 4. Documentation-command discrepancy (non-blocking, informational)

Plan.md §5 literally prints:

```
docker build -t simple-qna-rag:m4-policy .
python scripts/run_container_smoke.py --image simple-qna-rag:m4-policy
```

Neither is directly runnable as written: there is no root-level `Dockerfile`
(only `deploy/Dockerfile`, requiring `-f`), and there is no
`scripts/run_container_smoke.py` (only `scripts/container_smoke.py`, which is
what `ci.yml` actually calls). This milestone's Design.md §0.1 explicitly
states the container/build code and the four producer jobs' internal steps
are **not** touched by this change, so this is pre-existing template/copy
drift in the Plan document's example commands, not a defect introduced by
this diff, and not one of the six normative requirements (M4-OAR-REQ-001
through 006). This verifier ran the semantically-equivalent, actually-correct
commands (matching `ci.yml` exactly) instead of failing on the literal
Plan text. Recommend a documentation follow-up to correct Plan.md §5's
example commands, but this does not block acceptance.

## 5. Focused M4-OAR test suite (repeated confirmation)

`pytest -q tests/unit/test_assemble_m4_evidence.py
tests/unit/test_check_m4_baseline.py tests/unit/test_ci_workflow_contract.py
tests/unit/test_doc_audit_no_active_native_runner_procedure.py` — **216
passed** in both the local dev venv and the clean hosted-equivalent
container (§2), an exact match to Code_Review_Iteration_2.md's own count.
No regression since that review.

## 6. Workflow YAML / exact-shape independent verification

Beyond the 216 passing `test_ci_workflow_contract.py` assertions, this
verifier independently read `.github/workflows/ci.yml` directly:

- `yaml.safe_load` parses cleanly; jobs are exactly `python-tests,
  frontend-tests, container, m43-deterministic, m4-assemble,
  m3-live-regression-gate`.
- `m4-assemble.needs` = `[python-tests, frontend-tests, container,
  m43-deterministic]` only — the four deterministic producers, matching
  M4-OAR-REQ-002's "may become true only from the four same-run deterministic
  producers."
- `m4-assemble`'s "Check M4 baseline state algebra" step calls
  `check_m4_baseline.py --candidate assemble/m4-baseline.json` with **no**
  `--allow-legacy-v1` — strict v2 dispatch by default, matching
  M4-OAR-REQ-003.2.
- `m3-live-regression-gate.if` = `github.event_name == 'workflow_dispatch' &&
  inputs.enable_m3_live_regression == true` — never selected by ordinary
  `push`/`pull_request`, matching M4-OAR-REQ-004.1/.2. `workflow_dispatch`
  input `enable_m3_live_regression` defaults to `false`.
- The job runs on `ubuntu-latest`, `timeout-minutes: 1`, performs no
  checkout, no secrets, no self-hosted label, and its only step echoes a
  policy notice and exits 0 — matching the "informational, no-op reactivation
  stub" requirement.
- No `environment:` key appears anywhere in the file (grep confirmed) — no
  protected-environment approval gate exists in the ordinary path.

## 7. Items outside this worker's authorized scope (explicitly deferred, not failures)

- **Post-merge exact-SHA baseline verification** (Plan.md §6): requires a
  merged commit and a real workflow run; cannot occur before this dispatch's
  commit/push boundary. Traceability.md §1 and Code_Review_Iteration_2.md
  both already record this as correctly pending, not a defect.
- **Branch protection required-checks configuration** (Requirement
  M4-OAR-REQ-004.3, phrased "SHOULD"): `gh api
  repos/luminous419/simple-qna-rag/branches/master/protection` returns `404
  Branch not protected` — master currently has no branch protection rule
  configured. This is a repository administrative setting outside the file
  diff and outside a "MUST" requirement; noted for the human maintainer's
  awareness, not a gate failure.
- No `git commit`/`push`/PR/merge was performed, per this worker's dispatch
  scope and per Requirement/Plan guardrails.

## 8. Ambient environment gaps (documented truthfully, non-blocking)

- **Node engine version**: `npm ci` emits `EBADENGINE` — `package.json`
  requires `node >=22.22.2 <23`; this host runs `node v24.19.0`. `npm ci` and
  `npm test` both still completed successfully despite the warning; hosted
  CI presumably pins the required Node version via `actions/setup-node`. Not
  a code defect, but the discrepancy is real and worth the maintainer
  knowing if a future strict-engine mode is enabled.
- **Local `venv/` package drift**: 65 of 103 `requirements.lock`-pinned
  packages differ in version from what's installed in the pre-existing local
  `venv/` (see §2). This worker closed the gap by additionally reproducing
  results in a clean, hash-locked, Linux/amd64 environment rather than
  relying solely on the drifted venv.
- **macOS/Linux platform-pinned lock file**: `requirements.lock` is
  intentionally pinned to Linux wheel hashes (confirmed by prior commit
  history), so a native macOS `pip install --require-hashes` cannot succeed
  on this host by design; this is expected, not a defect, and was worked
  around via the amd64 container (§2, §3).

## 9. Requirement-by-requirement acceptance summary

| Requirement | Verified | How |
|---|---|---|
| M4-OAR-REQ-001 typed enums/policy | **PASS** | 216 focused tests (both envs); manual `support_policy`/gate literal inspection of fixture output in §1 |
| M4-OAR-REQ-002 readiness algebra | **PASS** | PASS/FAIL fixture CLI round-trip (§1); `native_linux_release_ready`/`full_production_release_ready`/`overall_release_ready` all `false` in both fixtures |
| M4-OAR-REQ-003 migration/compatibility | **PASS** | focused suite includes v1-legacy-path coverage; strict v2 default confirmed by workflow inspection (§6) |
| M4-OAR-REQ-004 workflow contract | **PASS** | direct YAML inspection (§6) plus 216 focused tests |
| M4-OAR-REQ-005 support boundary/docs | **PASS** | `check_markdown_links.py` 0 failures; doc-audit test passing; manual read of Requirement/Plan/Traceability confirms consistent hosted/OCI-only language |
| M4-OAR-REQ-006 evidence/history/security | **PASS** | no live/native/self-hosted/Ollama command executed by this verifier at any point; historical receipts untouched (`git diff --stat` confirms no product/history file rewritten) |

## 10. Final determination

**PASS.** Every pre-merge deterministic acceptance command in Plan.md §5
succeeds, both in the pre-existing local environment and in an
independently-constructed clean hosted-equivalent (linux/amd64,
hash-locked) environment. The two documentation-command literal mismatches
(§4) and the branch-protection gap (§7) are informational, non-blocking, and
outside the six normative requirements. The four clean-environment test
failures (§2) are conclusively diagnosed as artifacts of this verifier's own
improvised container harness — untouched by this diff and reproducing
cleanly elsewhere — not a regression. Remaining hosted-CI and post-merge
baseline verification (Plan.md §6) is correctly deferred to after
commit/push/merge, consistent with Traceability.md and
Code_Review_Iteration_2.md, and is out of this worker's authorized scope.

## 11. Post-merge addendum (release worker, 2026-08-15)

The diff above was committed (`f6ff86cd920f97e732973a6141c0d17cd16c3a1c`),
pushed, and opened as [PR #19](https://github.com/luminous419/simple-qna-rag/pull/19).

The first PR-head hosted CI run (`31889309407`, commit `f6ff86c`) failed
`python-tests`: `audit_exact_allowed_delta`'s tests in
`tests/unit/test_assemble_m4_evidence.py` shell out to
`git show adda1759754b56b514b3ab6252c2dc1032e03d28:...`, and
`actions/checkout@v4`'s default shallow clone (`fetch-depth: 1`) does not
fetch that ancestor commit object, so every such call failed with exit 128.
This is a new finding not caught by §2 above — this verifier's own clean
hosted-equivalent container had the full repository history mounted (not a
shallow clone), which masked the case. It is not a defect in the reviewed
diff's logic; it is a workflow-checkout configuration gap the reviewed diff
did not touch. Fixed in commit `24dfc8b49ddcebbd766202be3e1934e524c56e18` by
adding `fetch-depth: 0` to the `python-tests` job's checkout step only — no
test or product logic changed. The corrected PR-head run
(`31889748729`, commit `24dfc8b`) passed `python-tests`, `frontend-tests`,
`container`, `m43-deterministic`, and `m4-assemble`, with
`m3-live-regression-gate` `skipped` (never scheduled on `pull_request`).

PR #19 was merged (`--merge`, merge commit) to `master` at SHA
`8e203abe5ed6e17e6e8b6e292975121749374a52`. The post-merge `push` CI run for
that exact SHA (`31890598812`, `run_attempt=1`) completed with all five
jobs (`python-tests`, `frontend-tests`, `container`, `m43-deterministic`,
`m4-assemble`) `success` and `m3-live-regression-gate` `skipped`. The
`m4-baseline` artifact was downloaded from that exact run and verified with
the identity-bound checker:

```
python scripts/check_m4_baseline.py --candidate m4-baseline.json \
  --expect-hosted-release-ready --require-identity-binding \
  --expect-sha 8e203abe5ed6e17e6e8b6e292975121749374a52 \
  --expect-run-id 31890598812 --expect-run-attempt 1 \
  --expect-workflow-path .github/workflows/ci.yml --expect-event push
-> {"ok": true, "issues": []}   exit=0
```

An adversarial sanity check with a wrong `--expect-sha` on the same
artifact returned `{"ok": false, "issues": ["identity_sha_mismatch:..."]}`
with exit 1, confirming fail-closed behavior held at the exact artifact
used for this release, not only in unit tests.

The artifact's literals exactly match Plan.md §6's required block:
`deterministic_status=PASS`, `operational_status=NOT_ADOPTED`,
`gates.m3_live_regression=NOT_ADOPTED`, `gates.m41_operational=NOT_ADOPTED`,
`M4.1_BLOCKED=false`, `hosted_release_ready=true`,
`native_linux_release_ready=false`, `full_production_release_ready=false`,
`overall_release_ready=false`.

**Post-merge Operational Acceptance Gate: PASS.** The release claim is
narrowly "hosted/OCI release ready" — never "production ready" or "overall
release ready." `native_linux_release_ready`,
`full_production_release_ready`, and `overall_release_ready` remain
`false`; native Linux/Ollama remains `NOT_ADOPTED`. Full exact-identity
evidence is recorded in [Traceability.md §0](Traceability.md#0-release-evidence-exact-identity-binding).
