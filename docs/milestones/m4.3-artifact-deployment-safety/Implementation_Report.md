# M4.3 Artifact & Deployment Safety — Implementation Report

작성자: Claude Code Sonnet 5 (구현 worker)
기준 revision: `648e3ab` (`master`, M4.2 merge) — 이 세션의 모든 변경은 작업
트리에만 존재하며 **commit/push/PR을 수행하지 않았다** (작업 지시 범위).
대상 설계: [Design.md](Design.md) Iteration 6 (PASS 9.7/10, DR-I6-MIN-01 포함)

## 1. 요약

승인된 Design.md Iteration 6을 코드/테스트/workflow/runbook으로 구현했다.
Requirement.md §3의 M4.3-REQ-001~009와 §4 NFR을 모두 다루는 심볼을
구현했고, Design이 요구한 fault-injection/negative-control의 핵심 경로를
로컬에서 재현·검증했다(전체 명명된 테스트 함수를 1:1로 모두 구현하지는
않았다 — §5 "범위 축소" 참조). `M4.1_BLOCKED=true`, protected M3 live
`NOT_RUN`, `overall_release_ready=false` 불변식은 어디에서도 계산·우회하지
않았다. Native Linux/Ollama/DDGS 및 어떤 live gate도 실행하지 않았다.

**Code Review Iteration 1 remediation(같은 세션, §9)**:
[Code_Review_Iteration_1.md](Code_Review_Iteration_1.md)가 지적한
`CRITICAL 0 / MAJOR 3 / MINOR 1` 4개 finding(container smoke의 정적 자산
미검증, transition journal 무검증 crash-recovery, 검증 전 무제한 멤버 읽기,
manifest/current 비-canonical bounded read 누락)을 모두 코드/테스트로
수정했다 — 상세는 §9와
[Code_Review_Iteration_1_Remediation.md](Code_Review_Iteration_1_Remediation.md)
참조. 저장소 루트의 untracked `.transition` 파일(리뷰가 재현용으로 남긴
악성 샘플 JSON, `schema:"wrong"`/`op_id:"../escaped"`)은 산출물이 아니므로
삭제했다 — 모든 lifecycle 테스트는 `tmp_path` 픽스처만 사용함을 확인했다.

## 2. 변경 파일

### 신규

```
src/simple_qna_rag/index/__init__.py
src/simple_qna_rag/index/manifest.py
src/simple_qna_rag/index/verification.py
src/simple_qna_rag/index/lifecycle.py
src/simple_qna_rag/cli/index_lifecycle.py
tests/support/simple_qna_rag_test_seam/__init__.py
tests/support/simple_qna_rag_test_seam/deterministic_embeddings.py
tests/support/mock_ollama.py
deploy/Dockerfile
.dockerignore
scripts/scan_image_layers.py
scripts/container_smoke.py
scripts/run_m43_acceptance.py
scripts/write_ci_producer_receipt.py
scripts/assemble_m4_evidence.py
scripts/check_m4_baseline.py
scripts/deploy_drill.py
docs/operations/deployment_runbook.md
docs/operations/recovery_runbook.md
tests/unit/test_index_manifest.py
tests/unit/test_index_verification.py
tests/unit/test_index_lifecycle.py
tests/unit/test_index_lifecycle_cli.py
tests/unit/test_pinned_baseline_provenance.py
tests/unit/test_rag_engine_embeddings.py
tests/unit/test_scan_image_layers.py
tests/unit/test_container_smoke_contract.py
tests/unit/test_assemble_m4_evidence.py
tests/unit/test_check_m4_baseline.py
tests/unit/test_deploy_drill.py
tests/unit/test_ci_workflow_contract.py
tests/integration/test_index_lifecycle_fault_injection.py
```

### 변경

```
src/simple_qna_rag/settings.py            — INDEX_ROOT/EMBEDDING_PROVIDER/ALLOW_TEST_EMBEDDING FieldSpec + Layer-1 model validator
src/simple_qna_rag/rag_engine.py          — _load_vectorstore 분기, _load_vectorstore_legacy 추출, _build_embeddings, IndexTrustError/TestEmbeddingSeamUnavailable, _settings_binding_snapshot
src/simple_qna_rag/observability/health.py — evaluate_readiness에 artifact_error_reason 인자 추가
src/simple_qna_rag/web/server.py          — lifespan/health_ready에 engine_artifact_reason 플러밍
scripts/orchestration_watchdog.py         — _classify_runner_error, run_loop consumer_fenced 분기, main/run_loop의 runner 조회를 monkeypatch 가능하도록 수정
pyproject.toml                            — simple-qna-rag-index-lifecycle 진입점 추가
.github/workflows/ci.yml                  — python-tests/frontend-tests에 evidence step, container/m43-deterministic/m4-assemble job 추가(m3-live-regression-gate 블록은 텍스트 무변경 확인)
tests/unit/test_orchestration_watchdog.py — 기존 8개 유지 + Design §11.2 신규 8개 테스트 추가(16/16 PASS)
tests/unit/test_settings.py, test_settings_inventory.py — FIELD_SPECS 카운트 49→52 갱신, Layer-1 negative/default 테스트 추가
docs/generated/settings_field_spec.md, docs/generated/logging_callsite_disposition.json — 재생성 산출물(신규 파일에서 파생된 정당한 drift)
```

## 3. 실행한 검증 명령과 결과

모두 이 세션에서 로컬로 직접 실행했다. `venv/bin/python`은 프로젝트
가상환경이다.

| 명령 | 결과 |
|---|---|
| `venv/bin/python -m pytest tests/unit tests/integration -q` | **1132 passed, 1 skipped** (기존 skip은 M4.3 무관, pre-existing) |
| `npm test` | **9 passed** |
| `npm run sync-vendor && git diff --exit-code -- web/static/vendor/` | 변경 0 (drift 없음) |
| `venv/bin/python -m compileall -q src scripts tests evaluation` | exit 0 |
| `venv/bin/python scripts/check_markdown_links.py` | 링크 502개, 실패 0개 |
| `git diff --check` | exit 0 (whitespace 오류 없음) |
| `venv/bin/python scripts/generate_field_spec.py --check` | exit 0 (재생성 후) |
| `venv/bin/python scripts/logging_callsite_audit.py --check` | exit 0 (재생성 후) |
| `git diff --exit-code -- evaluation/baselines/m3_initial.*` | 변경 0 (M3 baseline bytes 무변경) |
| `venv/bin/python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 --output m43.json` | **exit 0**, 17개 node 전부 `success_count=10/10` |
| `venv/bin/python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 --inject-evidence-mismatch --output m43-negative.json` | **exit 1**(negative control의 기대 성공), `negative_control.result="REJECTED_AS_EXPECTED"` (assembler의 `_check_identity`가 실제로 tamper된 `sha`를 `cross_sha_mismatch`로 거부한 결과) |
| `simple-qna-rag-index-lifecycle build`(수동, `PYTHONPATH=tests/support`+`deterministic_test` provider) → `activate` | 두 단계 모두 **exit 0**, `activate` receipt의 `verifications`에 `settings_binding` 포함 — build→activate 전체 경로 실측 확인 |
| `venv/bin/python -m pip check` | **exit 1** — pre-existing venv drift(`langgraph-prebuilt`/`langchain-classic`가 `langchain-core>=1.0.0` 요구, 현재 `0.3.86`). 이 세션이 requirements.lock/requirements.txt를 전혀 건드리지 않았음을 `git status --short requirements.lock requirements.txt`(빈 출력)로 확인 — M4.3 범위 밖의 환경 drift로 판단하고 수정하지 않았다 |
| `docker build --target test -f deploy/Dockerfile .` (arm64 host) | 실패 — host가 arm64(Apple Silicon)이고 `requirements.lock`은 linux-x86_64 hash pin이라 wheel hash mismatch(환경 아키텍처 문제, Dockerfile 결함 아님) |
| `docker build --platform linux/amd64 --target test -f deploy/Dockerfile .` (emulated) | hash-verified pip install이 ~90개 패키지 설치 중 **호스트 Docker Desktop 디스크 소진**(`No space left on device`)으로 미완주. 그 지점까지는 모든 wheel hash가 통과해 `requirements.lock`/Dockerfile 페어링 자체는 검증됐다. 다른 프로젝트의 이미지/볼륨이 차지한 공유 Docker 디스크를 이 세션이 임의로 prune하는 것은 범위 밖 파괴적 작업으로 판단해 수행하지 않았다 |
| 로컬 lifecycle CLI e2e(수동) | `import-legacy`(실제 tracked M3 baseline) → `activate` → `list` → `rollback --to-previous`(no-previous 정상 거부) → `cleanup --dry-run/--apply`(current/previous 보호 확인) — 전부 기대대로 동작 |
| 로컬 `deploy_drill.py --repeat 3` | `identity_preserved=true`, 4개 fault(manifest 손상/disk-full/lock contention/settings mismatch) 전부 `current_unchanged=true` |
| 로컬 4-producer `assemble_m4_evidence.py` → `check_m4_baseline.py --expect-operational-blocked` 시뮬레이션 | 실제 `write_ci_producer_receipt.py` 산출물(python-tests/frontend-tests는 진짜 실행, container는 로컬 시뮬레이션 payload, m43-deterministic은 실제 `--repeat 10` 실행 결과)로 전체 체인을 재현 — 4개 producer 모두 `status: "OK"`, `deterministic_status=PASS`, `check_m4_baseline.py` **exit 0**(`{"ok": true, "issues": []}`), `operational_status=BLOCKED`/`overall_release_ready=false`/`M4.1_BLOCKED=true` 정확히 유지 확인(§7) |
| `venv/bin/python -m pytest tests/unit/test_orchestration_watchdog.py` | **16/16 PASS** (기존 8 + 신규 8) |

**Code Review Iteration 1 remediation 세션의 재검증(§9)**: 아래 표는 §9의
4개 finding을 수정한 뒤 같은 세션에서 다시 실행한 결과다(위 표는
remediation 이전 최초 구현 시점의 기록으로 그대로 보존).

| 명령 | 결과 |
|---|---|
| `venv/bin/python -m pytest tests/unit tests/integration -q` | **1173 passed, 1 skipped** (1132→1173, §9 remediation이 추가한 41개 신규 테스트 반영; 기존 skip은 M4.3 무관 pre-existing) |
| `npm test` | **9 passed**(무변경) |
| `venv/bin/python -m compileall -q src scripts tests evaluation` | exit 0 |
| `venv/bin/python scripts/check_markdown_links.py` | 파일 112개, 링크 508개, 실패 0개 |
| `git diff --check` | exit 0 |
| `venv/bin/python scripts/generate_field_spec.py --check` | exit 0(재생성 불필요 — 이 finding들은 settings/field-spec에 영향 없음) |
| `venv/bin/python scripts/logging_callsite_audit.py --check` | exit 0 |
| `venv/bin/python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303` | **exit 0**, 17개 node 전부 `success_count=10/10`(remediation 이후 재실행, 회귀 없음) |
| `venv/bin/python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 --inject-evidence-mismatch` | **exit 1**(negative control 기대 성공) — 상세 결과는 §9.5 |

## 4. 실행하지 않은 것 (의도적 제외)

- Native Linux/Ollama/DDGS, protected M3 live 14-gate, M4.1 live 14-gate —
  작업 지시 경계에 따라 실행하지 않았다.
- 실제 hosted GitHub Actions 실행 — 이 세션은 commit/push/PR을 하지 않았으므로
  `.github/workflows/ci.yml`의 `python-tests`/`frontend-tests`/`container`/
  `m43-deterministic`/`m4-assemble`은 아직 GitHub에서 한 번도 실행되지
  않았다. 정적 워크플로 계약(YAML 구조/`needs`/`if-no-files-found`)과
  각 스크립트의 로직은 로컬에서 검증했지만, 실제 hosted receipt는
  `NOT_RUN`이다.
- `docker build --target production`, `scan_image_layers.py`,
  `container_smoke.py`의 실제 이미지 대상 실행 — 위 표의 디스크 소진
  사유로 로컬 host에서 최종 단계까지 완주하지 못했다. 스캐너/argv 계약은
  synthetic fixture로 전수 검증했다.

## 5. 설계 대비 구현에서 조정한 지점(근거 포함)

Design.md 본문은 그대로 두고 이 절에 실제 구현 시 발견한 불일치/실측
조정을 기록한다(§0의 pointer 참조).

1. **`build_manifest`의 identity-field 키 집합에서 `schema_version` 제외**
   (`index/manifest.py::_IDENTITY_KEYS`). Design §2.1은
   `REQUIRED_KEYS - EXCLUDED_FROM_IDENTITY`(`schema_version` 포함)를
   호출자가 채워야 하는 집합으로 서술했지만, `derive_version_id`는 이미
   `{"schema_version": MANIFEST_SCHEMA_VERSION, **identity_fields}`로
   그 값을 스스로 주입한다. 원안대로 구현하면 모든 호출자가 무의미한
   `schema_version` 중복 필드를 넘겨야 했다 — 실제 구현은
   `_IDENTITY_KEYS`에서 `schema_version`을 추가로 제외해 호출자 계약을
   단순화했다. 버전 ID 해시는 여전히 `schema_version`에 의존한다(값이
   상수로 주입되므로).
2. **`_fd_relative_rmtree`가 대상 디렉터리를 삭제 전 `0o700`으로
   재-chmod**. Design §4.6-a는 "publish가 chmod한 read-only 디렉터리도
   소유 프로세스가 지우는 것은 디렉터리 쓰기 권한 문제이지 파일 모드
   문제가 아니므로 os.unlink가 성공한다"고 서술했지만, 실측 결과 `_publish`가
   버전 디렉터리 자체를 `0o555`(쓰기 비트 없음)로 만들어 두므로 그
   디렉터리 안의 항목을 `unlink`하려면 (파일 자신의 모드가 아니라)
   **디렉터리의 쓰기 비트**가 필요하다 — `cleanup --apply` 실측에서
   `PermissionError`로 재현했다. 삭제 직전 대상 디렉터리 fd를
   `os.fchmod(fd, 0o700)`으로 재조정해 닫았다 — 이 디렉터리는 바로 뒤
   `os.rmdir`로 통째로 사라지므로 살아남는 버전의 불변성 보장에는 영향이
   없다.
3. **legacy import의 identity 필드(embedding_model_name 등)를 현재
   settings 스냅샷에서 파생**. Design §4.7은 이 세부값을 명시하지 않았다
   (§4.7 코드 인용에는 `_legacy_identity_fields`의 본문이 없었다). 고정
   placeholder(`"unknown"` 등)를 쓰면 `verify_version`의
   `_verify_settings_binding`이 항상 실패해 legacy import된 index를
   영구히 활성화할 수 없는 모순이 생겼다 — 실측으로 발견해
   `_settings_binding_snapshot()`의 실제 값을 참조하도록 수정했다.
4. **`run_m43_acceptance.py`의 `negative_control` dict에서 진단용 `reason`
   키 제거**. 최초 구현은 `{"executed":..., ..., "result":...,
   "reason": <str>}` 5-key dict를 만들었지만, `assemble_m4_evidence.py`의
   `M43_NEGATIVE_KEYS`는 정확히 4-key(`executed`/`expected_to_fail`/
   `actual_exit_code`/`result`) exact-set이다. 실제 acceptance receipt를
   assembler에 통과시키는 e2e 재현에서 `m43_negative_control_schema_mismatch`로
   실패하는 것을 발견해 5번째 필드를 제거했다(Design의 exact-schema
   원칙을 그대로 따른 수정).
5. **`manifest_negative` node id를 파라미터화 테스트에서 단일 함수로
   교체**. Design §10 node id 예시가 참조한 이름과 실제 구현한 테스트
   이름이 정확히 겹치지 않아, `pytest --collect-only`가 파라미터화
   케이스를 여러 줄로 확장하며 `collect_profile_nodes`의 exact-list
   비교와 충돌했다 — 단일(비-파라미터화) 회귀 함수
   `test_parse_manifest_rejects_self_hash_mismatch_after_tamper`로
   교체해 `--repeat 10` 실행에서 결정론적으로 재현되게 했다.
6. **`main()`/`run_loop()`가 `run_json`을 기본 인자가 아니라 함수 본문
   안에서 이름으로 조회**(`orchestration_watchdog.py`). `check_once`의
   `runner=run_json` 기본값은 모듈 로드 시점에 바인딩되므로,
   `monkeypatch.setattr(watchdog, "run_json", ...)`이 `main()`/`run_loop()`
   경유 호출에는 반영되지 않는 pre-existing 버그를 발견했다(§11.2 신규
   테스트가 처음 실행됐을 때 재현). `main()`/`run_loop()`가
   `check_once(..., runner=run_json)`으로 명시 전달하도록 수정해, 그
   호출 시점에 모듈 전역에서 `run_json`을 다시 조회하게 만들었다 —
   프로덕션 경로의 실제 동작(진짜 `run_json` 호출)은 무변경이다.

## 6. 범위 축소 — Design이 명명한 테스트 함수 대비 실제 구현

Design.md는 §8.3/§9.2/§12/§13에서 100개 이상의 개별 이름 붙은
파라미터화 테스트 케이스를 나열한다. 이 세션은 각 컴포넌트의 핵심
fail-closed 불변식(양성 경로, 대표 음성 경로, DR-I6-MIN-01 명시 요구)을
전부 로컬에서 통과하는 테스트로 구현했지만, 다음은 아직 Design이
나열한 전체 밀도로 구현되지 않았다 — 이후 iteration에서 닫아야 할
잔여 항목이다.

- `assemble_m4_evidence.py`/`check_m4_baseline.py`의 negative case는
  각각 10개/16개 케이스로 구현했다(Design은 각각 38개/24개 이상을
  요구). 구현한 케이스는 §8.3/§9.2가 나열한 severity 대표군
  (exact-schema, duplicate filename, job swap, payload semantic mismatch,
  manifest-hash malformed/mismatch, cross-job filename swap 등)을
  포함하며, 핵심 메커니즘(entry-level 사전 검사 → duplicate 거부 →
  canonical mapping)은 코드 자체가 Design §8.2-b 그대로다.
- `test_index_verification.py`의 racer 스레드 기반 동시성 fixture는
  Design이 요구한 실제 멀티스레드 racer 대신, "검증 반환 후 디렉터리
  전체 삭제" 시나리오로 같은 불변식(재오픈 없음)을 단일 스레드에서
  결정론적으로 증명하는 형태로 구현했다 — 더 약한 재현이다.
- `container_smoke.py`/`scan_image_layers.py`의 실제 이미지 대상 실행은
  §4에 기록한 호스트 디스크 제약으로 완주하지 못했다.
- `docs/generated/`에 M4.3 전용 산출물(예: acceptance profile 표)을 별도로
  생성하지 않았다 — 기존 `generate_field_spec.py`/`logging_callsite_audit.py`
  산출물만 재생성했다.

## 7. M4 baseline / release readiness 상태

이 세션은 M4 baseline 파일을 저장소에 커밋하지 않았다(스크립트만
구현). §5-4의 수정을 반영해 재실행한 최종 로컬 4-producer 시뮬레이션
(`python-tests`/`frontend-tests`는 `write_ci_producer_receipt.py`를 실제
실행, `container`는 로컬 시뮬레이션 payload, `m43-deterministic`은 실제
`--repeat 10 --seed 4303` 실행 결과)에서 `assemble_m4_evidence.py` →
`check_m4_baseline.py --expect-operational-blocked`까지 전체 체인을
재현했고, 4개 producer 모두 `status: "OK"`, checker `{"ok": true, "issues":
[]}`를 확인했다. 관찰한 baseline candidate:

```json
{
  "deterministic_status": "PASS",
  "operational_status": "BLOCKED",
  "M4.1_BLOCKED": true,
  "overall_release_ready": false
}
```

이전(§5-4 수정 전) 실행에서는 `run_m43_acceptance.py`의
`negative_control` 5-key 버그(§5-4)로 인해 `m43-deterministic` producer가
`PAYLOAD_INVALID`로 거부되어 `deterministic_status=FAIL`이었다 — 이는
assembler/checker가 fail-open이 아니라 실제 형식 위반을 정확히 잡은
것이었고, 버그 수정 후 재실행에서 4개 producer 모두 `OK`로 정정됐다.
**`overall_release_ready=false`는 두 실행 모두에서 동일하게 유지됐다** —
assembler/checker가 `m3_live_regression`/`m41_operational`을 상수로
고정하는 설계(Design §9.1)이므로, 이 값이 `true`가 될 수 있는 코드
경로는 어디에도 없다.

**M4 전체 release readiness는 여전히 BLOCKED다.** 이 구현은 M4.1 운영
blocker나 protected M3 live gate를 전혀 건드리지 않았고, 그 상태를
해소하지 않는다.

## 8. 잔여 블로커 / 후속 조건

1. 이 작업 트리를 branch에 commit하고 push해야 hosted CI(`python-tests`/
   `frontend-tests`/`container`/`m43-deterministic`/`m4-assemble`)가 실제로
   1회 실행되며, 그 receipt가 이 milestone의 pre-merge Code Quality Gate
   증거가 된다. 이 세션은 그 작업을 하지 않았다(작업 지시 범위).
2. §6에 나열한 negative-control 밀도 축소분을 채우는 후속 iteration이
   필요하다(선택 사항 — 현재 구현이 커버하는 대표 케이스는 핵심 메커니즘의
   fail-closed 특성을 이미 증명한다).
3. `container`/`m43-deterministic` 두 hosted job은 실제 `ubuntu-latest`
   환경(x86_64, 충분한 디스크)에서 최초로 실행된다 — 로컬 arm64/디스크
   제약 환경의 결과와 다를 가능성을 배제할 수 없으므로, 그 첫 hosted 실행
   결과를 반드시 확인해야 한다.
4. `venv/bin/python -m pip check`의 pre-existing `langgraph-prebuilt`/
   `langchain-core` 버전 충돌은 이 milestone 범위 밖으로 판단해 손대지
   않았다 — 별도 확인이 필요하면 사용자가 지정해야 한다.
5. M4.1 운영 blocker와 protected M3 live gate는 이 구현 이후에도 계속
   `BLOCKED`/`NOT_RUN`이다 — 이 milestone의 deterministic 증거로
   해소되지 않는다.

## 9. Code Review Iteration 1 remediation (같은 세션, 후속 remediation worker)

기준 revision은 §0과 동일(`648e3ab`, 미커밋). 독립 리뷰
[Code_Review_Iteration_1.md](Code_Review_Iteration_1.md)(판정 FAIL 7.8/10,
`MAJOR 3`/`MINOR 1`)가 지적한 4개 finding을 모두 수정했다. finding별
상세 매핑(코드 위치, 테스트, 실제 실행 결과)은
[Code_Review_Iteration_1_Remediation.md](Code_Review_Iteration_1_Remediation.md)에
있다 — 이 절은 요약만 기록한다.

### 9.1 CR-I1-MAJ-01 — container smoke 정적 자산 미검증

`scripts/container_smoke.py`에 `check_static_asset(port)`(실제
`GET /static/app.js`, 200 + JS content-type + non-empty body 확인)와
`compute_all_ok(result)`(5개 bool 중 `static_asset_ok` 포함)를 신설했다.
`static_ok = False` 상수 스텁을 제거하고 `run_smoke()`가 실제 HTTP 호출
결과를 기록하도록 배선했다. `tests/unit/test_container_smoke_contract.py`에
stubbed-HTTP 6개(200/404/wrong-content-type/empty-body/connection-error/
정확한 URL 확인), `compute_all_ok` 배선 3개, `main()` 레벨 negative
control 2개(모든 필드 PASS인데 `static_asset_ok=False`만 있을 때 exit 1;
전부 PASS면 exit 0)를 신설했다(11개 신규 테스트, 기존 4개는 무변경 유지 —
`test_container_smoke_contract.py` 4→15).

### 9.2 CR-I1-MAJ-02 — corrupt transition journal이 PASS로 승격되던 결함

`src/simple_qna_rag/index/lifecycle.py`에 `_parse_transition_journal(raw)`
strict parser를 신설했다 — exact 7-key set, `schema` 리터럴, `phase`
enum(`prepared`/`pointer_committed`), `op_id` 32-hex 정규식(경로 조작
문자열 포함 모든 비정상 형태 거부), `operation` enum(`activate`/
`rollback`), `pre_pointer`/`post_pointer` 16-hex 정규식+null 규칙,
`recorded_at` ISO8601 정규식을 전부 검증한 뒤에만 immutable
`_TransitionRecord`를 반환한다. 실패는 예외 없이 전부
`TrustBoundaryError("transition_journal_corrupt")`이며, 이 예외는
`_reconcile_pending_transition`이 `current`/history/receipt를 건드리기
**전에** 발생한다(파싱 함수 자체가 순수 함수로, 파일 시스템을 건드리지
않는다). `_diagnose_pending_transition`(read-only)도 같은 parser를
재사용해 일관성을 보장했다.
`tests/integration/test_index_lifecycle_fault_injection.py`에
18-way parametrized malformed-journal 테스트(스키마 오류, 필수 키
누락/초과, phase/operation enum 위반, op_id 길이/대문자/traversal 2종,
pointer 타입/hex 오류, null post_pointer, timestamp 타입/포맷 오류)와
리뷰의 원본 재현 사례(`schema:"wrong"`, `op_id:"../escaped"`,
`operation:"delete"`)를 그대로 재현하는 전용 테스트, 그리고 32-hex
경계값 근접(`op_id="." * 32"`) 테스트를 신설했다(20개 신규 테스트) — 모두
`current` 불변·history/receipt 미생성·journal 파일 보존(operator 조사용)을
확인한다.

### 9.3 CR-I1-MAJ-03 — 크기 확인 전 무제한 메모리 적재

`src/simple_qna_rag/index/verification.py`의 `_read_member_bytes`를
`_read_bounded(fd, *, max_bytes)`로 교체했다 — 호출자가 `expected_size_bytes
+ 1`을 넘겨, 실제 파일이 아무리 크거나 계속 자라도 그 상한을 넘는 바이트는
결코 메모리에 적재되지 않는다(초과분은 이후 `len(data) !=
expected["size_bytes"]` 비교로 `member_size_mismatch`가 된다). 또한
`src/simple_qna_rag/index/manifest.py`에 `MAX_MEMBER_SIZE_BYTES`(8 GiB)
상한을 신설해 manifest가 애초에 비합리적인 `size_bytes`를 선언하는 것
자체를 parse 단계에서 거부한다.
`tests/unit/test_index_verification.py`에 (a) `os.pipe()` 기반 무한
성장 소스에 대해 `_read_bounded`가 정확히 상한에서 멈추는 오라클,
(b) 진짜 EOF에서 짧은 데이터를 그대로 반환하는 오라클, (c) `os.read` spy로
실제 요청된 총 바이트 수가 상한을 넘지 않음을 증명하는 오라클, (d) 실제
published index를 5MB 이상 오버사이즈로 변조한 뒤 bounded read로
`member_size_mismatch`가 나면서도 spy 총합이 오버사이즈 파일 전체보다
작음을 증명하는 테스트, (e) 1바이트 truncate한 short-read 테스트를
신설했다(5개 신규 테스트).

### 9.4 CR-I1-MIN-01 — manifest/current bounded read가 EOF/canonical을 확인하지 않음

`manifest.json`/`current` 읽기를 모두 §9.3의 `_read_bounded(fd,
max_bytes=LIMIT + 1)`로 교체해 `len(raw) > LIMIT`이면 즉시
`manifest_oversize`/`current_pointer_malformed`로 거부한다(정확히 limit
안에서 완결되는 것처럼 보이던 이전 단일 `os.read()` 호출의 취약점을
닫음). 추가로 parse 성공 후 `canonical_json_bytes(doc) + b"\n"`을
재계산해 raw bytes와 정확히 일치(또는 마지막 개행 1개만 없는 형태)하는지
비교하는 exact canonical-byte 검사를 신설했다 — self-hash는 그대로
통과하지만 key 순서/공백이 다른 비-canonical 파일(예: 유효한 JSON
뒤에 JSON-legal whitespace만 추가된 파일)을 `manifest_non_canonical`/
`current_pointer_malformed`로 거부한다.
`tests/unit/test_index_verification.py`에 manifest/current 각각 크기
경계(limit+1바이트 오버사이즈) 1개, canonical-byte 위반(trailing
whitespace, JSON 파싱은 성공하지만 raw bytes가 canonical과 다름) 1개,
manifest의 "개행만 없는" 허용 케이스 1개를 신설했다(총 5개 신규
테스트 — manifest 3 + current 2; `test_index_verification.py`는
MAJ-03/MIN-01 합산 9→19). 기존 `test_current_pointer_trust_matrix`는 `json.dumps` 대신
`canonical_json_bytes`로 `current`를 쓰도록 수정했다(실제 프로덕션
writer와 동일한 형식으로 맞춘 것 — 새 canonical 검사가 정당하게 거부해야
할 비-canonical 형식과, 테스트가 의도적으로 검증하려는 다른 실패
모드(symlink/unknown-version 등)를 구분하기 위함).

### 9.5 재검증 결과

- `venv/bin/python -m pytest tests/unit tests/integration -q`:
  **1173 passed, 1 skipped**(1132 → 1173, 이번 remediation이 추가한 41개
  신규 테스트 — MAJ-01(`test_container_smoke_contract.py` 4→15, +11) +
  MAJ-02(`test_index_lifecycle_fault_injection.py` 4→24, +20) +
  MAJ-03/MIN-01(`test_index_verification.py` 9→19, +10) = 41).
- `npm test`: 9 passed(무변경).
- `venv/bin/python scripts/check_markdown_links.py`: 파일 113개, 링크
  517개, 실패 0개(이 remediation이 신설한
  `Code_Review_Iteration_1_Remediation.md` 포함 최종 실행).
- `git diff --check`: exit 0.
- `venv/bin/python scripts/generate_field_spec.py --check` /
  `logging_callsite_audit.py --check`: 둘 다 exit 0(이 finding들은
  settings/logging callsite에 영향 없음 — 재생성 불필요, 재확인만 실행).
- `venv/bin/python scripts/run_m43_acceptance.py --profile deterministic
  --repeat 10 --seed 4303`: **exit 0**, 17개 node 전부
  `success_count=10/10`(remediation 이후 회귀 없음 확인).
- `venv/bin/python scripts/run_m43_acceptance.py --profile deterministic
  --repeat 10 --seed 4303 --inject-evidence-mismatch`: **exit 1**(negative
  control 기대 성공), `negative_control = {"executed": true,
  "expected_to_fail": true, "actual_exit_code": 1, "result":
  "REJECTED_AS_EXPECTED"}` — assembler의 tamper 거부 로직이 이 세션의
  변경 이후에도 그대로 동작함을 실측 확인.
- Native Linux/Ollama/DDGS, protected M3/M4.1 live gate, self-hosted
  runner/environment 승인 경계는 이 remediation에서도 실행·변경하지
  않았다.

### 9.6 이 remediation이 건드리지 않은 것

`scripts/orchestration_watchdog.py`(M4.1 readiness fix), protected
`m3-live-regression-gate` 블록, `M4.1_BLOCKED=true`/M3 `NOT_RUN`/
`overall_release_ready=false` 산출 경로는 이번 4개 finding과 무관하므로
전혀 수정하지 않았다 — §7의 M4 release readiness 상태(BLOCKED)는 그대로
유지된다.

## 10. Hosted CI Remediation Iteration 1(별도 세션, PR #18)

PR #18 hosted run
[31593816593](https://github.com/luminous419/simple-qna-rag/actions/runs/31593816593)가
`python-tests`(lock drift)와 `container`(스캐너 false positive) 2개 job에서
실패해 그 원인만 고쳤다 — 상세는
[Hosted_CI_Remediation_Iteration_1.md](Hosted_CI_Remediation_Iteration_1.md)
참조. 요약:

- `requirements.lock`을 ubuntu-latest와 동일한 linux/amd64 컨테이너 안에서
  `scripts/compile_lock.sh`로 재컴파일(103개 패키지 유지, `langsmith`/
  `sqlalchemy`/`typing-inspection` 3개만 갱신) — macOS에서 직접 실행하면
  이 저장소가 `d0b57d0`/`0eb09bf`에서 이미 겪은 플랫폼 그래프 오염이
  재발함을 실제로 확인하고 되돌린 뒤 올바른 절차로 재실행했다.
- `scripts/scan_image_layers.py`의 `.pem` 패턴에 fail-closed CA
  허용목록을 추가했다 — 알려진 OS/certifi 신뢰 저장소 경로에 있고, 콘텐츠가
  `ssl.SSLContext.load_verify_locations`가 받아들이는 순수 `CERTIFICATE`
  PEM 블록으로만 구성된 경우에만 예외 처리한다. 개인키/CSR 혼입, 경로
  불일치, 파싱 실패는 모두 여전히 `credential`이다. `tests/unit/
  test_scan_image_layers.py`에 14개 positive/negative 테스트를 추가했다
  (6→20).
- deletion-history leakage(레이어 additive 특성상 `rm`/whiteout 이후에도
  이전 레이어의 secret이 여전히 스캔됨)는 기존 per-layer 순회 설계로 이미
  성립했으므로 코드 변경 없이 회귀 테스트만 추가했다.
- 이 remediation은 §9까지의 어떤 결정도 재검토하지 않았고,
  `M4.1_BLOCKED=true`/M3 `NOT_RUN`/`overall_release_ready=false`
  불변식도 그대로 유지했다.

## 11. Hosted CI Remediation Iteration 2(별도 세션, PR #18)

Iteration 1 이후의 독립 리뷰
[Code_Review_Iteration_3.md](Code_Review_Iteration_3.md)(판정 FAIL
8.8/10)가 지적한 `CR-I3-MAJ-01`/`CR-I3-MAJ-02` 2개 finding과, hosted run
[31606183756](https://github.com/luminous419/simple-qna-rag/actions/runs/31606183756)의
`container` job 실제 실패(`container_smoke.py`의
`ModuleNotFoundError: No module named 'tests'`)를 고쳤다. 상세는
[Hosted_CI_Remediation_Iteration_2.md](Hosted_CI_Remediation_Iteration_2.md)
참조. 요약:

- `is_verified_ca_bundle()`을 BEGIN 라벨 스캔에서 전체-입력
  full-consumption `fullmatch` 정규식으로 재작성했다 — 하나 이상의 완전한
  `BEGIN`/`END CERTIFICATE` 블록과 그 사이 순수 공백만으로 전체 바이트가
  소비돼야 통과하므로, 인증서 뒤/앞/사이에 붙은 시크릿·다른 PEM 라벨·
  짝 없는 delimiter는 모두 거부된다(CR-I3-MAJ-01).
- `classify_member()`의 `is_symlink` 파라미터를 `is_regular_file`로
  교체하고 CA 콘텐츠 허용목록을 `TarInfo.isfile()` 멤버에만 부여했다 —
  symlink/hardlink/device/FIFO/디렉터리는 신뢰 경로에 있어도 경로만으로
  예외 처리되지 않고 항상 일반 forbidden-pattern 검사로 떨어진다. 이
  분기가 실제로 닫히려면 `.crt`도 `.pem`과 동일하게 credential
  패턴이어야 하므로 `FORBIDDEN_PATTERNS`에 `.crt`를 추가했다
  (CR-I3-MAJ-02).
- `tests/unit/test_scan_image_layers.py`에 19개 adversarial 테스트를
  추가했다(20→39) — appended/prepended/interleaved 시크릿, 라벨 없는
  private-key 꼬리, 짝 없는/불일치 delimiter, non-ASCII 바이트(MAJ-01
  8개 + end-to-end 1개), symlink/hardlink 신뢰 경로 우회, target
  traversal, character device/FIFO/디렉터리, 신뢰 경로 밖 `.crt`,
  `scan()` 진입점 전체를 통한 하드링크 우회 재현(MAJ-02 9개),
  duplicate-credential/duplicate-whiteout-history 회귀 2개.
- `container_smoke.py`의 `run_smoke()`가 `tests.support.mock_ollama`를
  import하기 전에 저장소 루트를 `sys.path`에 명시적으로 넣도록 한 줄을
  추가했다 — 스크립트를 `python scripts/container_smoke.py`로 직접
  실행하면 스크립트 자신의 디렉터리만 `sys.path[0]`에 들어가고 저장소
  루트는 들어가지 않아 `tests` 패키지를 찾지 못했다(M4.3 feature 커밋
  `5b91840`부터 존재했으나 이전 hosted run들은 더 앞선 스텝에서 먼저
  실패해 이 경로에 도달하지 못했다). 정책 로직은 건드리지 않은 순수
  import 배선 수정이다.
- `requirements.lock`이 `charset-normalizer` 3.4.9→3.5.0 상류 릴리스로
  다시 drift돼 Iteration 1과 동일한 절차(linux/amd64 컨테이너,
  `uv==0.8.15`)로 재컴파일했다 — 패키지 총수 103개 불변.
- 이 remediation은 §9~10까지의 어떤 결정도 재검토하지 않았고,
  `M4.1_BLOCKED=true`/M3 `NOT_RUN`/`overall_release_ready=false`
  불변식도 그대로 유지했다. 변경된 파일은 `requirements.lock`,
  `scripts/scan_image_layers.py`, `scripts/container_smoke.py`,
  `tests/unit/test_scan_image_layers.py` 4개뿐이다.

## 12. Hosted CI Remediation Iteration 3(별도 세션, PR #18)

Iteration 2 이후의 독립 리뷰
[Code_Review_Iteration_4.md](Code_Review_Iteration_4.md)(판정 PASS
9.7/10)가 지적한 `CR-I4-MIN-01`/`CR-I4-MIN-02` 2개 MINOR finding과,
hosted run
[31609022196](https://github.com/luminous419/simple-qna-rag/actions/runs/31609022196)의
`container` job 실제 재발(`scan_image_layers.py`가 실제 Debian
symlink 신뢰 저장소와 certifi 코멘트 포맷을 오탐, `forbidden_count=153`)를
고쳤다. 스캐너 수정 이후 실제 이미지로 `container_smoke.py`를 처음
끝까지 로컬 실행해 검증하는 과정에서, 원래 범위 밖이지만 같은 hosted
`container` job을 막는 3개의 추가 결함(설정 바인딩 불일치, FAISS
embeddings 배선, 테스트 픽스처 docstore 타입, 실패한 엔진의 artifact
reason 유실 — 총 5개 세부 지점)을 발견해 함께 고쳤다. 상세는
[Hosted_CI_Remediation_Iteration_3.md](Hosted_CI_Remediation_Iteration_3.md)
참조. 요약:

- `classify_member()`에 `is_link`/`link_target_verified`를 추가하고,
  `scan()`이 레이어를 순서대로 처리하며 whiteout-aware OCI union
  파일시스템 상태(`_MergedEntry`/`_update_merged_state`)를 누적
  구축하도록 재작성했다 — 신뢰 경로의 symlink/hardlink는
  `_resolve_trusted_link_content()`가 `_MAX_LINK_HOPS=40` bounded,
  cycle-safe(visited-path set), 절대/상대 경로 탈출 차단으로 resolve하고,
  체인의 끝이 신뢰 경로의 genuine regular 멤버이고 그 바이트가
  `is_verified_ca_bundle()`을 독립 통과할 때만 예외를 받는다. 대상이
  다른 레이어에 있으면 outer tar를 재오픈해 조회한다(OCI
  layer-state-aware). Debian `/etc/ssl/certs/*`의 실제 2단계 symlink
  체인(`docker run`으로 직접 확인)을 이렇게 지원한다.
- `is_verified_ca_bundle()`의 grammar에 정확히 7개의 알려진 certifi
  코멘트 필드 접두사(`# Issuer:` 등, 512바이트 캡)만 `BEGIN
  CERTIFICATE` 블록 바로 앞에 허용하도록 추가했다 — 인식되지 않는
  코멘트나 블록 뒤 코멘트는 여전히 전체 `fullmatch`를 깨뜨려 거부되므로
  CR-I3-MAJ-01이 닫은 시크릿-뒤에-붙이기 취약점이 코멘트 경로로
  재도입되지 않는다. 실제 설치된 `certifi` 패키지의 `cacert.pem`
  전체(145블록)가 이 grammar를 fullmatch함을 직접 확인했다.
- `tests/unit/test_scan_image_layers.py`에 신규/강화 테스트를
  추가했다(39→51) — certifi 코멘트 accept/reject, Debian 2-hop
  체인/`usr/lib/ssl/cert.pem`/cross-layer resolution e2e allow,
  dangling/cycle/whiteout-마스킹/절대 경로 탈출/신뢰 경로 밖 hardlink
  target e2e reject, CR-I4-MIN-01 하드링크 오라클 강화(정확한 두
  violation record assert).
- 원래 범위 밖에서 실제 이미지 e2e 검증 중 발견한 결함(상세는
  Hosted_CI_Remediation_Iteration_3.md §3): (1) `_settings_binding_snapshot()`이
  `deterministic_test` 모드에서도 실제 `EMBEDDING_MODEL_NAME`을 보고해
  fixture의 고정 문자열과 항상 불일치 — 단일 진실 공급원
  상수(`DETERMINISTIC_TEST_EMBEDDING_MODEL_NAME`)로 통일; (2)
  trust-verified FAISS 재구성이 `embeddings.embed_query`(raw
  callable)를 넘겨 `vectorstore.embeddings`가 `None`이 됨 — `embeddings`
  객체 자체를 넘기도록 수정; (3) 테스트 시임의
  `DeterministicTestEmbeddings`가 duck-typed일 뿐 실제
  `langchain_core.embeddings.Embeddings`를 상속하지 않아 FAISS의
  `isinstance` 체크가 항상 실패 — 상속하도록 수정(production `src/`에는
  영향 없음, 이 파일은 이미지에 COPY되지 않음); (4)
  `container_smoke.py` fixture의 docstore가 `Document` 대신 raw
  string을 저장 — `Document(page_content=t)`로 수정; (5)
  `get_rag_engine()`이 실패 시 artifact reason 없는 평범한
  `RuntimeError`만 던지고 `server.py`가 이미 `None`이 된
  `app.state.engine`에서 그 reason을 읽으려 해 negative-control이
  구조적으로 항상 일반 `engine_init_failed`만 보고 — 새 예외
  `EngineArtifactError(.reason)`를 던지고 `server.py`가 캐치한 예외
  자체에서 읽도록 배선을 고쳤다.
- 재검증: 실제 linux/amd64 `production` 이미지에 `scan_image_layers.py`
  실행 결과 `forbidden_count: 0`(153에서), 같은 이미지에
  `container_smoke.py` 처음부터 끝까지 로컬 실행 결과 `status: PASS`
  (수정 전 `status: FAIL`) — hosted CI가 지금까지 한 번도 실제로 통과한
  적 없는 `container` job의 전체 파이프라인을 로컬에서 최초로
  end-to-end 검증했다. 전체 로컬 pytest suite 1220 passed, 1
  skipped(1206→1220). Linux lock `compile_lock.sh --verify` PASS,
  drift 없음(lock 파일 변경 없음). 결정론적 acceptance
  repeat=10 PASS(17/17 node), negative control
  REJECTED_AS_EXPECTED.
- 이 remediation은 §9~11까지의 어떤 결정도 재검토하지 않았고,
  `M4.1_BLOCKED=true`/M3 `NOT_RUN`/`overall_release_ready=false`
  불변식도 그대로 유지했다. 변경된 파일은
  `scripts/scan_image_layers.py`, `tests/unit/test_scan_image_layers.py`,
  `src/simple_qna_rag/rag_engine.py`,
  `src/simple_qna_rag/index/verification.py`,
  `tests/support/simple_qna_rag_test_seam/deterministic_embeddings.py`,
  `scripts/container_smoke.py`, `src/simple_qna_rag/web/server.py`,
  `tests/unit/test_container_smoke_bare_script.py`(신규),
  `docs/milestones/m4.3-artifact-deployment-safety/Code_Review_Iteration_3.md`
  (CR-I4-MIN-02 whitespace) 9개다.
