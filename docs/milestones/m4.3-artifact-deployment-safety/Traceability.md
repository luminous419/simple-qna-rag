# M4.3 Artifact & Deployment Safety 요구사항 추적표

상태: **구현 완료 — 로컬 결정론적 검증 PASS, hosted CI/merge 증거 NOT_RUN(미커밋)**  
기준 revision: `648e3ab` (작업 트리 변경 미커밋 — no commit/push/PR)  
근거: [Requirement.md](Requirement.md), [Plan.md](Plan.md), [Design.md](Design.md),
[Implementation_Report.md](Implementation_Report.md),
[Code_Review_Iteration_1.md](Code_Review_Iteration_1.md),
[Code_Review_Iteration_1_Remediation.md](Code_Review_Iteration_1_Remediation.md),
[M4 복구 결정](../m4-production-readiness/Recovery_Decision.md)

Code Review Iteration 1(판정 FAIL 7.8/10)이 지적한 `CR-I1-MAJ-01/02/03`,
`CR-I1-MIN-01` 4개 finding은 같은 세션의 후속 remediation에서 전부
수정·재검증했다 — 상세는
[Code_Review_Iteration_1_Remediation.md](Code_Review_Iteration_1_Remediation.md)
참조. 이 remediation은 M4.3-REQ-001(§3 trust boundary)/M4.3-REQ-003(crash
recovery)/M4.3-REQ-005(container smoke)의 기존 PASS 판정이 근거했던 실제
코드 결함을 닫은 것이므로, 아래 표의 해당 행은 remediation 이후 재검증된
상태를 반영한다.

상태 enum은 `PLANNED`, `IN_PROGRESS`, `PASS`, `FAIL`, `BLOCKED`, `NOT_RUN`이다.
`NOT_RUN`/`BLOCKED`를 PASS로 계산하지 않는다. 아래 "PASS"는 이 세션이 로컬에서
직접 실행해 확인한 deterministic 증거만을 근거로 한다 — hosted GitHub Actions
receipt는 이 작업 트리가 아직 commit/push되지 않았으므로 존재하지 않으며, 그
Gate는 `NOT_RUN`으로 유지한다(Implementation_Report.md §검증 결과 참조).

## 1. Requirement → Phase → 구현 → 테스트 → 증거

| ID | Phase | 구현/문서 | positive/negative 검증 | 증거 | 상태 |
|---|---:|---|---|---|---|
| M4.3-REQ-001 canonical index/provenance | 1~2 | `index/manifest.py`, `index/verification.py` | canonical 100회 round-trip PASS; schema/non-finite/hash/symlink/owner/mode mismatch 전수 거부, `load_local` 0회(재오픈 없음) 실측 확인; remediation 이후: member/manifest/current 읽기가 `expected_size+1`로 bounded, manifest/current exact canonical-byte 비교 추가 | `tests/unit/test_index_manifest.py`(19), `tests/unit/test_index_verification.py`(9→19, CR-I1-MAJ-03/MIN-01 remediation) — 로컬 PASS | **PASS(로컬)** |
| M4.3-REQ-002 legacy/staging | 1~3 | `index/lifecycle.py`, lifecycle CLI | 실제 tracked M3 baseline hash pair로 import 성공(실측), 임의 hash 거부, 원본 bytes 불변 확인 | `tests/unit/test_index_lifecycle.py`, `test_pinned_baseline_provenance.py` — 로컬 PASS; CLI e2e 수동 재현 성공 | **PASS(로컬)** |
| M4.3-REQ-003 activation/rollback/retention | 1~3 | atomic pointer/lock/retention/transition journal | activate/rollback 100회 반복 partial pointer 0; crash-recovery reconcile 검증; lock symlink/contention 거부; cleanup dry-run/apply 보호 확인; remediation 이후: `.transition` journal에 exact-schema/enum/32-hex op_id/pointer-regex/timestamp를 강제하는 strict parser 추가, malformed journal은 어떤 mutation도 없이 fail-closed 거부 | `tests/unit/test_index_lifecycle.py`, `tests/integration/test_index_lifecycle_fault_injection.py`(4→24, CR-I1-MAJ-02 remediation — 18-way malformed-journal parametrize + 원본 finding 재현 + traversal-lookalike 경계) — 로컬 PASS | **PASS(로컬)** |
| M4.3-REQ-004 CLI/evidence | 4,7 | `cli/index_lifecycle.py`, canonical JSON receipt | exit 0/1/2/3 matrix, domain 오류 receipt 변환 확인 | `tests/unit/test_index_lifecycle_cli.py` — 로컬 PASS; 수동 e2e(build 대체 import-legacy→activate→list→rollback→cleanup) 성공 | **PASS(로컬)** |
| M4.3-REQ-005 OCI image | 1,5 | `deploy/Dockerfile`, `.dockerignore`, `scan_image_layers.py`, `container_smoke.py`, test seam | 스캐너 positive/negative/traversal/whiteout fixture 전수 PASS; argv 계약 PASS; `docker build --target test`(amd64 emulated)가 hash-verified pip install까지 성공(호스트 Docker Desktop 디스크 소진으로 최종 레이어 미완주); remediation 이후: `check_static_asset()`가 실제 `GET /static/app.js` 200+content-type+non-empty body를 검증하고 `static_asset_ok`가 `all_ok`에 포함됨(이전에는 상수 `False`로 이 필드가 hosted gate를 필연적으로 실패시켰음); **Hosted CI Remediation Iteration 1**: hosted run 31593816593이 base 이미지의 정상 OS/certifi CA 신뢰 저장소 파일 153개를 `.pem` 패턴이 전부 credential로 오탐(`forbidden_count=153`)해 `container` job이 실패했다 — path+content 이중 조건의 fail-closed 허용목록을 추가해 신뢰 저장소 경로의 순수 `CERTIFICATE` PEM만 예외 처리하고, 개인키/CSR 혼입·경로 불일치·파싱 실패는 여전히 credential로 남도록 했다(상세는 [Hosted_CI_Remediation_Iteration_1.md](Hosted_CI_Remediation_Iteration_1.md)) | `tests/unit/test_scan_image_layers.py`(6→20, Hosted CI Remediation Iteration 1 — 시스템 CA/certifi 허용 4 + 악성 PEM/경로 불일치 거부 5 + `is_verified_ca_bundle` 단위 3 + deletion-history leakage 회귀 1 + 기존 pfx/env 회귀 1), `test_container_smoke_contract.py`(4→15, CR-I1-MAJ-01 remediation — stubbed-HTTP 6 + `compute_all_ok` 배선 3 + `main()` negative control 2) — 로컬 PASS; 실제 이미지 build/scan/smoke는 호스트 Docker Desktop 디스크 소진(VM 가상 디스크 100%, `docker builder prune`/`docker image prune` 후에도 537MB만 회수)으로 이 세션에서도 미완주 | **IN_PROGRESS** — 로직/계약 PASS, 실제 이미지 Gate는 hosted CI(x86_64, 디스크 여유) 필요 |
| M4.3-REQ-006 runbook | 6 | `docs/operations/*.md`, `deploy_drill.py` | mock deploy/rollback 3회, manifest 손상/disk-full/lock contention/settings mismatch 4종 fault 모두 `current` 불변 확인 | `tests/unit/test_deploy_drill.py`, 수동 `deploy_drill.py --repeat 3` 실행 — 로컬 PASS | **PASS(로컬)** |
| M4.3-REQ-007 single workflow | 7 | `.github/workflows/ci.yml`, `write_ci_producer_receipt.py`/`assemble_m4_evidence.py` | 정적 workflow 계약 테스트 PASS; 로컬 4-producer 시뮬레이션 assemble PASS(container payload는 로컬 시뮬레이션); negative matrix(10건 이상) PASS; protected block 텍스트 무변경 확인 | `tests/unit/test_ci_workflow_contract.py`, `test_assemble_m4_evidence.py` — 로컬 PASS; hosted 실행은 미커밋으로 `NOT_RUN` | **IN_PROGRESS** — 로직 PASS, hosted receipt는 commit/push 이후에만 발생 |
| M4.3-REQ-008 M4 baseline | 8 | `check_m4_baseline.py` | exact-schema/producer→gate algebra 24개 이상 파라미터화 케이스 PASS; 실측 baseline candidate(로컬 4-producer 시뮬레이션)에 대해 `--expect-operational-blocked` PASS | `tests/unit/test_check_m4_baseline.py` — 로컬 PASS | **PASS(로컬 로직)** — 실제 hosted candidate는 REQ-007과 동일하게 `NOT_RUN` |
| M4.3-REQ-009 compatibility/readiness fix | 0,8 | `scripts/orchestration_watchdog.py`(`_classify_runner_error`, `run_loop` 분기) | 기존 8개 테스트 무변경 유지 + 신규 8개 exact-argv/terminal-scope/consumer_fenced/dry-run 테스트 전부 PASS(16/16) | `tests/unit/test_orchestration_watchdog.py` — 로컬 PASS | **PASS(로컬)** |

## 2. 비기능 추적

| ID | 검증 | 상태 |
|---|---|---|
| M4.3-NFR-001 재현성 | canonical manifest round-trip 100회 동일 확인(로컬); clean hosted rebuild는 미실행 | IN_PROGRESS |
| M4.3-NFR-002 신뢰성 | crash/disk-full/contention/receipt-failure fault matrix에서 active 불변 — 로컬 실측 확인 | **PASS(로컬)** |
| M4.3-NFR-003 보안 | contained-open/pickle prevalidation 로컬 실측 확인; OCI layer scan 로직 PASS(실제 이미지 미완주); remediation 이후: pre-verification 읽기가 선언 크기+1로 bounded(무제한 메모리 적재 결함 제거), crash-recovery journal이 strict-schema fail-closed(corrupted journal이 PASS로 승격되던 결함 제거) | IN_PROGRESS |
| M4.3-NFR-004 호환성 | M3 baseline bytes 무변경 확인(`git diff --exit-code`); 전체 로컬 pytest suite 1173 PASS(1 skip)(1132 → 1173, Code Review Iteration 1 remediation이 추가한 41개 신규 테스트 반영) | **PASS(로컬)** |
| M4.3-NFR-005 검증성 | repeat-10 acceptance 실행 완료(PASS), genuine evidence mutation과 same-parser rejection 실측 확인 | **PASS(로컬)** |
| M4.3-NFR-006 복구성 | mock deploy/rollback 3회, exact identity 복귀 로컬 실측 확인 | **PASS(로컬)** |

## 3. Gate 추적

| Gate | 위치/시점 | 필수 증거 | 현재 상태 |
|---|---|---|---|
| Requirement/Plan 품질 | pre-merge | 링크, diff, 독립 review 9.7+ | 기존 승인 유지(변경 없음) |
| Design 품질 | pre-merge | executable prototypes, 독립 review, CRITICAL/MAJOR 0 | 기존 Iteration 6 PASS 9.7/10 유지(변경 없음) |
| Python hosted | PR + master push | locked install, `pip check`, pytest, dataset/generated/link checks | 로컬 동등 명령 전부 PASS; hosted receipt는 미커밋으로 `NOT_RUN` |
| Frontend hosted | PR + master push | `npm ci/test/sync-vendor`, vendor diff 0 | 로컬 PASS; hosted receipt `NOT_RUN` |
| Container hosted | PR + master push | build, security/mock smoke, layer scan | Dockerfile hash-verified install까지 로컬 확인(호스트 디스크 소진으로 최종 미완주); `container_smoke.py`의 `static_asset_ok`가 이제 실제 HTTP 검증 결과(CR-I1-MAJ-01 remediation, stubbed 단위 테스트로 로직 PASS 확인) — 실제 이미지 대상 실행은 여전히 hosted CI 필요; hosted receipt `NOT_RUN` |
| M4.3 deterministic | local/hosted | index fault/receipt/rollback repeat, negative control | **로컬 PASS**(repeat=10, seed=4303, positive+negative 모두 실행; Code Review Iteration 1 remediation 이후 재실행, 17개 node 전부 10/10 유지 확인) |
| Same-workflow assemble | PR + master push | all hosted `needs=success`, same run/attempt/SHA, exact-one artifact | 정적 계약 PASS; 로컬 4-producer 시뮬레이션 assemble PASS; hosted 실행 `NOT_RUN`(미커밋) |
| M4 baseline candidate | post-merge hosted | deterministic PASS, operational BLOCKED 분리 | checker 로직 PASS; 로컬 시뮬레이션 candidate에서 `overall_release_ready=false`/`M4.1_BLOCKED=true` 확인; 실제 hosted candidate `NOT_RUN` |
| M4.1 operational | protected post-merge | live 14-gate + 정상 receipt | BLOCKED |
| Protected M3 live | self-hosted/environment 승인 | 실제 protected job receipt | NOT_RUN |
| 전체 M4 release | post-merge operational | 위 모든 필수 operational evidence 또는 별도 release-risk 승인 | BLOCKED |

## 4. 선행 승인/예외 보존

| 항목 | 근거 | M4.3 사용 방식 | 불변 상태 |
|---|---|---|---|
| M3 baseline | `evaluation/baselines/m3_initial.{json,md}` | legacy approved hash와 회귀 reference | byte 변경 금지 |
| M4.1 pre-merge 구현 | M4.1 문서/merge `fd14eec` | settings/logging/metrics/health interface 입력 | 운영 PASS로 확대 금지 |
| M4.1 operational exception | [예외 결정](../m4.1-configuration-observability/Operational_Acceptance_Exception.md) | `M4.1_BLOCKED=true` | BLOCKED |
| M4.2 deterministic acceptance | [최종 검증](../m4.2-safe-serving-boundary/Final_Verification_Report.md), merge `648e3ab` | regression input | live 결과로 확대 금지 |
| protected live workflow | `.github/workflows/ci.yml::m3-live-regression-gate` | immutable contract snapshot | trigger/runner/environment 변경 금지 |

## 5. 현재 코드 gap과 planned closure

| 현재 경계 (`648e3ab`) | Gap | closure |
|---|---|---|
| `cli/index_documents.py::create_vectorstore`가 active output을 삭제 후 직접 저장 | 실패 시 정상 index 손실 | staging + immutable publish + atomic pointer |
| `rag_engine.py::_load_vectorstore`가 path 존재만 확인 후 dangerous pickle load | provenance/trust 검증 없음 | manifest/contained-open/hash/settings 검증 후 load |
| root Dockerfile/`.dockerignore`/runbook 없음 | 재현 배포·최소 권한 증거 없음 | Phase 5~6 |
| workflow hosted Python/frontend + protected live만 존재 | container/M4.3/assemble 없음 | 같은 workflow hosted DAG 추가, protected block 보존 |
| M4 baseline 없음 | deterministic/operational 상태 결합 위험 | typed 분리 baseline + fail-closed checker |
| `scripts/orchestration_watchdog.py` 미커밋 coordinator readiness fix | base `e57fe1c` 이후 terminal-bound query이며 기존 test가 exact argv를 assert하지 않음 | fix 보존; exact argv/read-only peek/failure tests와 bound-Run dry-run 후 review·commit scope 포함 |

## 6. Acceptance 명령 추적

현재 문서 Gate:

```bash
python scripts/check_markdown_links.py
git diff --check
```

구현 후 필수 deterministic command family:

```bash
bash scripts/compile_lock.sh --verify
python -m pip check
python -m pytest -q
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
npm ci && npm test && npm run sync-vendor
git diff --exit-code -- web/static/vendor/
python scripts/generate_field_spec.py --check
python scripts/logging_callsite_audit.py --check
python scripts/check_markdown_links.py
python -m compileall -q src scripts tests evaluation
python scripts/run_m42_acceptance.py --profile deterministic --repeat 10 --seed 4202 --output <tmp>/m42.json
python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 --output <tmp>/m43.json
python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 --inject-evidence-mismatch --output <tmp>/m43-negative.json
docker build --target test -f deploy/Dockerfile .
docker build --target production -f deploy/Dockerfile -t simple-qna-rag:m43-candidate .
python scripts/scan_image_layers.py --image simple-qna-rag:m43-candidate
python scripts/assemble_m4_evidence.py --fresh-dir <tmp>/m4-assemble --expected-sha "$(git rev-parse HEAD)"
python scripts/check_m4_baseline.py --candidate <tmp>/m4-assemble/m4-baseline.json --expect-operational-blocked
git diff --check
```

계획된 command가 아직 존재하지 않는 현재 단계에서는 실행하지 않고 `PLANNED`로 둔다.
negative-control command는 exit 1이어야 PASS다. Native Linux/Ollama/DDGS/M3 live 명령은
이 목록에 의도적으로 없으며, 그 부재는 전체 M4 release blocker를 닫지 않는다.

## 7. 판정 불변식

```text
M4.3_DETERMINISTIC_PASS
  = all_required_hosted_and_local_deterministic_gates_are_PASS

M4_OVERALL_RELEASE_READY
  = M4.3_DETERMINISTIC_PASS
    AND M4.1_OPERATIONAL_PASS
    AND PROTECTED_M3_LIVE_PASS
    AND exact_post_merge_release_identity_is_verified
```

현재 `M4.1_OPERATIONAL_PASS=false`, `PROTECTED_M3_LIVE_PASS=false(NOT_RUN)`이므로
`M4_OVERALL_RELEASE_READY=false`다. 어떤 assembler, baseline writer 또는 문서도 이 값을
누락·default·합성으로 true로 만들 수 없다.
