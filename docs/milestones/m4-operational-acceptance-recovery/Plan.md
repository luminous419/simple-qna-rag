# M4 Operational Acceptance Recovery policy-change plan

Status: **APPROVED SPECIFICATION / IMPLEMENTATION COMPLETE (PRE-MERGE)**  
Requirements: [Requirement.md](Requirement.md)

## 1. Guardrails

Implementation is repository-only and uses hosted CI. Do not provision or
inspect a native host, register a runner, approve an environment, dispatch the
old live job, contact Ollama, run `RUN_LIVE_LLM_TESTS=1`, or create synthetic
live evidence. Preserve historical receipts and M4.3 PASS.

## 2. Exact implementation scope

| File | Symbols/sections | Required change |
|---|---|---|
| `scripts/assemble_m4_evidence.py` | constants; `assemble`; `main` | Add v2/support-policy constants and readiness fields; keep four producers and their verification unchanged; emit `NOT_ADOPTED` compatibility gates and hosted-only algebra. |
| `scripts/check_m4_baseline.py` | schema constants; `check`; `main` | Strictly dispatch v2 versus explicit legacy v1; independently recompute policy and all readiness fields; add hosted expectation CLI flags. |
| `tests/unit/test_assemble_m4_evidence.py` | baseline/schema/algebra cases | Assert exact v2 key set and policy, deterministic true/false cases, and immutable M4.3 evidence behavior. |
| `tests/unit/test_check_m4_baseline.py` | candidate fixture and adversarial matrix | Cover every invalid typed state, alias, policy, schema, and readiness combination plus explicit v1 compatibility. |
| `.github/workflows/ci.yml` | triggers; `assemble-m4`; `m3-live-regression-gate` | Ensure ordinary runs never wait on self-hosted capacity; assemble from deterministic needs only; use hosted-ready checker; remove executable live path or retain only a non-executing explicit opt-in reactivation stub. |
| `tests/unit/test_ci_workflow_contract.py` | workflow static contract | Replace assertions that preserve the live runner with assertions for terminal ordinary runs, no self-hosted/approval path, deterministic dependencies, and explicit future-reactivation contract. |
| `docs/Roadmap.md`, `docs/Problem.md` | M4/P1 status | Record hosted/OCI scope approval and distinguish hosted readiness from full production readiness. |
| Release/deployment/user docs found by `rg` | support matrix and runbook sections | Label native Ollama development-only/unsupported and document v2 artifact verification. No runtime UI change unless current UI makes a release-readiness claim. |

`scripts/ci_acceptance_contract.py`, `scripts/preflight_ollama.py`, and their
tests remain as historical tooling unless dead-code removal is separately
approved. They MUST NOT appear in adopted release gates.

## 3. Implementation sequence

1. Add v2 assembler constants and exact output shape. Keep
   `REQUIRED_PRODUCERS`, receipt/payload identity checks, M4.3 node set, hashes,
   and same-run arguments unchanged.
2. Refactor the checker into strict schema-specific paths. V2 recomputes four
   producer gates, deterministic status, `hosted_release_ready`, fixed false
   broader flags, and the compatibility alias without trusting candidate
   aggregates. V1 is rejected unless explicitly allowed.
3. Change workflow dependencies and checker invocation. Prefer removing the
   live job entirely. If retaining reactivation documentation in YAML, use a
   false-default `workflow_dispatch` input and a hosted, no-checkout, no-secret,
   no-environment informational job; never reference `[self-hosted, ollama-m3]`
   in an executable job.
4. Update static and adversarial tests, then support docs. Do not alter product
   behavior or claim native capability.

## 4. Adversarial acceptance matrix

Tests MUST reject: `PASS`, `BLOCKED`, `NOT_RUN`, `SKIPPED`, `UNKNOWN`, or
`WAIVED` substituted for either required `NOT_ADOPTED`; wrong support-policy
schema/date/scope; missing/extra/mixed-version keys; strings used as booleans;
true native/full/overall readiness; alias disagreement; hosted true with any
deterministic producer missing, failed, skipped, duplicated, malformed,
path-traversing, cross-run/SHA/attempt/workflow/event, payload-tampered, or M4.3
negative control not rejected; self-reported deterministic PASS over failed
producer evidence; implicit v1 acceptance; and an ordinary workflow trigger
that can select an environment or self-hosted runner.

Positive tests MUST prove: exact four-producer same-run evidence yields
deterministic PASS and hosted ready; any deterministic failure yields hosted
not-ready; v1 is readable only under the explicit compatibility flag and keeps
its original blocked/false meaning; historical M4.3 receipt hashes are carried
unchanged.

## 5. Hosted pre-merge gate

Run from a clean dependency environment; none is a live test:

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
pytest -q
npm ci
npm test
npm run sync-vendor
git diff --exit-code -- web/static/vendor
docker build --target production -f deploy/Dockerfile -t simple-qna-rag:m4-policy .
python scripts/container_smoke.py --image simple-qna-rag:m4-policy
python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303
python scripts/check_markdown_links.py
git diff --check
```

Also assemble fixture evidence for one all-PASS and one failed deterministic
case, then run respectively:

```bash
python scripts/check_m4_baseline.py --candidate <PASS_V2_JSON> --expect-hosted-release-ready
python scripts/check_m4_baseline.py --candidate <FAIL_V2_JSON> --expect-hosted-not-ready
```

Gate: all commands pass, review has no unresolved CRITICAL/MAJOR finding, and
the diff contains no live execution, product code, fabricated receipt, or
historical artifact rewrite.

## 6. Post-merge gate

On the exact merge SHA, require terminal success for `python-tests`,
`frontend-tests`, `container`, `m43-deterministic`, and `assemble-m4`. Confirm no
job is queued for `self-hosted`, no `m3-live-regression` environment approval is
requested, and the workflow run reaches a terminal conclusion. Download
`m4-baseline` into a fresh directory and run the hosted-ready checker. Inspect
the artifact for:

```text
deterministic_status=PASS
operational_status=NOT_ADOPTED
m3_live_regression=NOT_ADOPTED
m41_operational=NOT_ADOPTED
M4.1_BLOCKED=false
hosted_release_ready=true
native_linux_release_ready=false
full_production_release_ready=false
overall_release_ready=false
```

Publish only the claim “hosted/OCI release ready.” Never shorten it to
“production ready” or “overall release ready.”

## 7. Rollback and future reactivation

If migration or workflow changes fail, revert the policy implementation while
retaining the v1 blocked baseline and historical receipts; do not enable the
live job as a workaround. Reactivation is a new milestone requiring explicit
scope adoption, owned infrastructure, security/design review, new evidence
schema, and approval. `NOT_ADOPTED` cannot be changed to `PASS` by configuration
alone.
