# M3 Code Review — Iteration 1

- Review date: 2026-08-07
- Scope: all uncommitted changes since `HEAD`, M3 normative documents, Phase 4 and `m3-final` live artifacts
- Score: **8.4 / 10**
- Gate: **STOP / REJECT**
- Findings: **CRITICAL 0, MAJOR 3, MINOR 1, TRIVIAL 0**
- Required gate: score >= 9.7, CRITICAL 0, MAJOR 0

## Executive assessment

The static implementation is broad and well tested: the full Python suite passes (640 passed, 1 skipped), the web suite passes (9 passed), the golden dataset validates, Markdown link validation reports zero failures, `git diff --check` is clean, and the immutable M2 dataset/baseline files are unchanged from `HEAD`. The standalone final Retrieval, Routing, and Answer reports also show strong raw metrics.

The M3 checkpoint is nevertheless not promotable. The authoritative `m3-final` aggregate explicitly has `gate_evaluation.overall_pass=false`, its routing child report is not schema-compatible with the M3 contract, the Phase 4 production default was adopted from an agent-scored worksheet rather than the required human review, and the checked-in traceability documents still describe the live phases as pending. These are acceptance-evidence and checkpoint-safety defects, not cosmetic report issues.

## CRITICAL

None.

## MAJOR

### M1 — The authoritative `m3-final` checkpoint fails its own promotion gate

Evidence:

- `evaluation/reports/m3/m3-final/baseline_20260807T132559219593Z.json` records `gate_evaluation.overall_pass=false`; both `retrieval_latency_mean_ms` and `retrieval_latency_p95_ms` have `metric=null` and `pass=null`.
- The same aggregate points at `evaluation/reports/m3/m3-final/retrieval/retrieval_20260807T053804883072Z.json`, whose `warmup.performed=true`, `mmr_instrumentation.fallback_case_count=0`, and latency values are 2191.36 ms mean / 2368.56 ms p95. The child report therefore contains qualifying measurements that the aggregate did not carry into its gate result.
- `evaluation/compare.py:106-122` only blocks these values when warm-up is false or MMR fallback occurred and otherwise accepts either `mean_ms`/`p95_ms` or `mean`/`p95`.
- `evaluation/baseline.py:548-555` defines the final gate evaluation as the result produced from the in-memory child payloads. A saved aggregate that disagrees with its referenced child artifact is not a safe or reproducible checkpoint.

Impact: Requirement §4.1 and M3-REQ-010 require the final comparison/approval record to pass all gates. A separate standalone report cannot override an authoritative aggregate that says the gate is unresolved and failed.

Required fix: rerun `m3-final` from one clean, consistent process after fixing or diagnosing the payload mismatch; verify all 14 gate items resolve to `pass=true`, `overall_pass=true`, and the referenced child report timestamps/fingerprints are the artifacts actually used by the aggregate. Add an integration test that serializes/reloads the exact child payloads referenced by the final baseline and asserts the same gate result.

### M2 — Baseline-generated Routing reports violate the M3 1.1 schema and omit required evaluator metadata

Evidence:

- `evaluation/reports/m3/m3-final/routing/routing_20260807T102111841184Z.json` has `schema_version="1.0.0"`, `routing_policy=null`, and `router_prompt_sha256=null`.
- `evaluation/routing.py:630-686` shows the canonical Routing CLI computes `router_prompt_sha256`, `routing_policy`, explicitly assigns `SCHEMA_VERSION`, and declares `SCHEMA_VERSION="1.1.0"` at `evaluation/routing.py:42`.
- The alternate orchestration path in `evaluation/baseline.py:381-404` builds and writes a routing payload directly but never supplies `router_prompt_sha256`/`routing_policy` and never assigns the Routing schema version.
- Design §3.2-3.3 requires evaluator report schema 1.1.0 and the routing policy/prompt fields; M3-REQ-001 requires model/routing settings and evaluator version in each candidate run.

Impact: the most important production-like runner produces a materially weaker and mislabeled artifact than the standalone evaluator. Reviewers cannot verify which routing policy or prompt yielded the reported 76/76 result, and schema-aware consumers can treat a new-format payload as legacy 1.0.

Required fix: extract one public payload/report builder in `evaluation.routing` and use it from both CLI and baseline, or invoke the canonical evaluator path. Add a baseline integration test asserting schema 1.1.0, non-null prompt hash in live mode, complete routing policy metadata, and parity with standalone output.

### M3 — Phase 4 adoption does not satisfy the required human blind-review gate

Evidence:

- `evaluation/reports/m3/m3-p4-intent-ab/ADR.md:37-44` correctly states that a human must score the blind worksheet.
- The later decision at `ADR.md:64-75` says an agent performed proxy blind scoring and immediately changes the production default to `ANSWER_TEMPLATE_MODE="default"`.
- The same document remains internally contradictory: `ADR.md:80-98` says the decision is unconfirmed until actual human scoring and lists human review as a resume condition.
- `src/simple_qna_rag/config.py:269-270` has already adopted `"default"` as the production default.
- Requirement §4.2 and M3-REQ-007 require paired blind review and a preserved approval decision; `docs/milestones/m3-retrieval-domain-quality/Traceability.md:16,62-64` still identifies live/human Phase 4 review as pending.

Impact: the classifier/template production behavior was changed without satisfying the milestone's specified reviewer gate. The worksheet may be useful preliminary evidence, but it cannot be represented as the final human approval record.

Required fix: obtain and preserve an actual human blind review (including reviewer identity/role or an explicit approval record), aggregate it, resolve the ADR's contradictory sections, and only then retain the adopted default. Until that happens, restore the pre-adoption default or explicitly mark the candidate as non-promoted.

## MINOR

### m1 — Checked-in milestone status and metrics are stale relative to the live artifacts

Evidence:

- `docs/milestones/m3-retrieval-domain-quality/Traceability.md:3,16,19,45-46,60-65` says Phase 4 and the `m3-final` run are still pending, although both report directories now contain live outputs.
- `Traceability.md:35-44` records earlier Retrieval/Routing numbers (including routing values `[76,75,76]`) while the current `m3-final` reports show Retrieval 2191.36/2368.56 ms and Routing `[76,76,76]`.
- `Traceability.md:54` says 639 tests passed; the current run is 640 passed, 1 skipped.

Impact: reviewers following the checked-in traceability document cannot determine the current milestone state or distinguish preliminary phase evidence from final evidence.

Required fix: update Traceability and the Phase 4 ADR after the blockers above are resolved. Clearly label historical numbers, link the exact final artifacts, and state the actual gate outcome.

## Positive observations

- M2 checkpoint safety is preserved: `evaluation/datasets/golden.jsonl` and `evaluation/baselines/m2_initial.{json,md}` have no diff from `HEAD`; the golden dataset SHA-256 remains `61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a`.
- The standalone final Retrieval report satisfies the quality and latency floors, uses stored vectors, records one query embedding per successful case, zero candidate embedding calls, 2100 lookup hits, and zero fallback cases.
- The standalone final Answer report is official evaluator v2, records the reviewed-variants SHA and rules fingerprint, has no v2 regressions, achieves 100% abstention accuracy, 100% source any-hit, and 95.45% mean source recall.
- Production fallback and compatibility seams have focused tests, including trace-disabled MMR fallback, routing policy behavior, vector mapping validation, evaluator fail-closed behavior, and RAG response seam tests.
- Detailed live reports remain under the ignored `evaluation/reports/` tree, reducing accidental publication of prompts/answers and local execution metadata.

## Verification performed

| Check | Result |
|---|---|
| `pytest -q` | 640 passed, 1 skipped, 1 environment warning |
| `npm test` | 9 passed |
| `python -m evaluation.dataset validate evaluation/datasets/golden.jsonl` | passed, 76 valid cases |
| `python scripts/check_markdown_links.py` | 41 files, 100 links, 0 failures |
| `git diff --check HEAD` | passed |
| M2 golden/baseline diff against `HEAD` | unchanged |

## Gate recommendation

**STOP / REJECT for promotion.** The implementation may proceed to a corrective review iteration, but it must not be committed as an approved M3 baseline or merged as milestone-complete until M1-M3 are closed, the final aggregate reports `overall_pass=true`, and an independent re-review assigns at least 9.7/10 with CRITICAL 0 and MAJOR 0.
