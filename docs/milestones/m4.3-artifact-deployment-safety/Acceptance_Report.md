# M4.3 Artifact & Deployment Safety — Final Pre-Merge Integration & Acceptance Report

역할: Claude Code Sonnet 5 integration/acceptance worker
기준 revision (working tree base, HEAD, uncommitted): `648e3abcca7c321e7f4dd13a7fbed1a4f1886c3e`
선행 조건: 독립 Code Review Iteration 2 — **PASS 9.8/10**
(`CRITICAL 0 / MAJOR 0 / MINOR 0 / TRIVIAL 0`), 근거
[Code_Review_Iteration_2.md](Code_Review_Iteration_2.md)
실행 시각: 2026-08-12 (KST 오후, UTC 07:xx)
No commit / push / PR / merge performed by this session.

## 0. 근거 문서

`milestone_dev_orchestration_guide.md` 전체와 이 디렉터리의
[Requirement.md](Requirement.md), [Plan.md](Plan.md), [Design.md](Design.md),
[Design_Review_Iteration_1~6](Design_Review_Iteration_6.md),
[Traceability.md](Traceability.md), [Implementation_Report.md](Implementation_Report.md),
[Code_Review_Iteration_1.md](Code_Review_Iteration_1.md),
[Code_Review_Iteration_1_Remediation.md](Code_Review_Iteration_1_Remediation.md),
[Code_Review_Iteration_2.md](Code_Review_Iteration_2.md)를 읽고, 현재 working
tree(코드/스크립트/워크플로/테스트)를 대조 확인한 뒤 아래 절차를 수행했다.

## 1. 목적과 범위

이 세션은 **pre-merge Code Quality Gate 이후, merge 이전의 최종 통합/인수
단계**다. 코드 변경은 수행하지 않았고(발견된 결함 없음), 이미 리뷰를 통과한
구현을 독립적으로 재현·검증했다. 가이드 §11(Pre-merge/Post-merge Gate 분리)에
따라 hosted CI/protected environment/self-hosted runner 증거는 이 세션의
책임 범위 밖이며 `NOT_RUN`으로 유지한다.

## 2. 실행 명령과 결과 (전체 명시적 재현)

### 2.1 Python 컴파일/정적/생성/링크/diff 계약

| # | 명령 | 결과 |
|---|---|---|
| 1 | `venv/bin/python -m compileall -q src scripts tests evaluation` | exit 0 |
| 2 | `venv/bin/python -m pip check` | **exit 1** — 아래 §3.1 참조(사전 존재 환경 조건, M4.3 무관) |
| 3 | `bash scripts/compile_lock.sh --verify` | exit 0, `Resolved 102 packages in 3.29s`, `requirements.lock` git diff 없음(재현 가능, drift 없음) |
| 4 | `venv/bin/python scripts/generate_field_spec.py --check` | exit 0 |
| 5 | `venv/bin/python scripts/logging_callsite_audit.py --check` | exit 0 |
| 6 | `venv/bin/python scripts/check_markdown_links.py` | exit 0 — 파일 114개(tracked 98 + untracked 16), 링크 519개, 실패 0개 |
| 7 | `git diff --check` | exit 0 |
| 8 | `venv/bin/python -m evaluation.dataset validate evaluation/datasets/golden.jsonl` | exit 0 — `"valid": true`, `"errors": []`, total 76 |

### 2.2 전체 Python unit + integration suite

```
venv/bin/python -m pytest tests/unit tests/integration -q
```

결과: **1173 passed, 1 skipped, 4 warnings in 127.11s** — Code Review
Iteration 2 근거 수치와 정확히 일치. skip 1건은 기존에 문서화된 M4.3 무관
pre-existing skip이다. 신규 회귀 없음.

### 2.3 Frontend 테스트/vendor drift

| 명령 | 결과 |
|---|---|
| `npm ci` | exit 0 — 92 packages, postinstall `sync-vendor.js`가 vendor 4개 파일 동기화 |
| `npm test` | exit 0 — **9 passed** (vitest) |
| `npm run sync-vendor` | exit 0 |
| `git diff --exit-code -- web/static/vendor/` | exit 0 — drift 없음 |

### 2.4 Dockerfile/layer/container workflow 정적 계약 (targeted)

```
venv/bin/python -m pytest tests/unit/test_ci_workflow_contract.py \
  tests/unit/test_scan_image_layers.py tests/unit/test_container_smoke_contract.py -q
```

결과: **26 passed**(§2.2 전체 suite에도 포함된 동일 테스트의 focused 재확인).
`deploy/Dockerfile`을 직접 재검토해 numeric non-root user(`10001:10001`),
test-seam(`tests/support`)의 production stage 물리적 미포함, `--require-hashes`
locked install, minimal `COPY` surface를 재확인했다.

### 2.5 Protected M3 live block 보존 검증 (byte 단위)

`.github/workflows/ci.yml`의 `m3-live-regression-gate:` 잡 블록을 `master`와
현재 working tree에서 각각 추출해 SHA-256으로 비교했다.

```
git show master:.github/workflows/ci.yml | awk '/^  m3-live-regression-gate:/,0' > master.txt
awk '/^  m3-live-regression-gate:/,0' .github/workflows/ci.yml > current.txt
diff master.txt current.txt   # 출력 없음
sha256sum master.txt current.txt
# 6fbb2a13432c7b216e7d871f156d52bd5faa554a46fdcacb99ee2f2723b314cb  master.txt
# 6fbb2a13432c7b216e7d871f156d52bd5faa554a46fdcacb99ee2f2723b314cb  current.txt
```

49줄 블록이 **byte 단위로 완전히 동일**(SHA-256 일치). trigger, `[self-hosted,
ollama-m3]` runner labels, `environment: m3-live-regression` 승인 요구가
전혀 변경되지 않았음을 확인했다. `git diff master -- .github/workflows/ci.yml`
전체를 별도로 검토해 신규 `+` 라인은 `python-tests`/`frontend-tests`의 evidence
step과 신규 `container`/`m43-deterministic`/`m4-assemble` job에만 있고, 기존
`python-tests`/`frontend-tests`/`m3-live-regression-gate` 블록 본문에는 삭제(`-`)
라인이 전혀 없음을 확인했다.

### 2.6 M4.3 deterministic acceptance — positive

```
venv/bin/python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 \
  --output <tmp>/m43-positive.json
```

결과: **exit 0**, top-level `"status": "PASS"`, 17개 node 전부
`success_count: 10/10`, `negative_control.executed: false`(positive 실행에서는
negative control 미실행이 정상).

Node 목록(17개, 전부 10/10 PASS): `activation_rollback`,
`assemble_payload_verification`, `baseline_strict_schema`,
`container_static_and_connectivity`, `crash_recovery_journal`,
`embedding_provider_seam_guard`, `layer_scanner`, `legacy_baseline_pin`,
`legacy_import`, `lock_contention`, `lock_untrusted_symlink`,
`manifest_canonical`, `manifest_negative`, `retention`, `staging_fault`,
`verification_reopen_race`, `verification_trust`.

### 2.7 M4.3 deterministic acceptance — negative control (expected failure)

```
venv/bin/python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 \
  --inject-evidence-mismatch --output <tmp>/m43-negative.json
```

결과: **exit 1**(negative control의 기대되는 성공) — 프로세스 exit code가
정확히 1, top-level `"status": "REJECTED_AS_EXPECTED"`,
`negative_control = {"executed": true, "expected_to_fail": true,
"actual_exit_code": 1, "result": "REJECTED_AS_EXPECTED"}`. tampered `sha`
필드를 `assemble_m4_evidence.py::_check_identity`의 동일 parser가 거부함을
실측 확인했다.

### 2.8 Deploy drill

```
venv/bin/python scripts/deploy_drill.py --root <tmp>/deploy_drill_root --repeat 3 \
  --output <tmp>/deploy_drill.json
```

결과: exit 0. `identity_preserved: true`. `start_identity_current`와
`final_identity_current`가 동일(`version_id: 22df8aba840fcec6`). `repeat=3`의
activate/rollback 5-step 시퀀스 전부 `outcome: PASS`, `ready: true`. 4종
fault injection(`manifest_corruption`, `disk_full_build`, `lock_contention`,
`readiness_settings_mismatch`) 전부 `current_unchanged: true`(마지막
fault는 정상적으로 `outcome: FAIL`/`error_code: settings_mismatch`를
반환하는 것이 설계상 기대 동작이며, 이것이 `current` 불변을 깨지 않았음을
같은 필드로 확인).

### 2.9 Docker build/scan/smoke — 시도 및 정확한 실패 증거

호스트 환경: macOS, Docker Desktop 29.6.2, Docker VM `aarch64`
(`linux/amd64` requirements.lock hash pin과 불일치하므로 native build는
아키텍처상 불가 — 기존 문서화된 제약과 동일).

**시도 1 — emulated amd64 test-stage build:**

```
docker build --platform linux/amd64 --target test -f deploy/Dockerfile .
```

- 빌드가 `requirements.lock`의 **104개 wheel을 hash-verified 상태로 전부
  다운로드**하고 `Installing collected packages: ...`(90개 패키지 나열)
  단계까지 진행했다.
- 63.82초 지점에서 `Installing collected packages`가 시작된 뒤 78.88초
  지점에서 **`ERROR: Could not install packages due to an OSError: [Errno
  28] No space left on device`**로 실패(exit 1).
- 이 실패는 [Implementation_Report.md](Implementation_Report.md)에 기록된
  이전 세션의 정확히 동일한 현상(Docker Desktop VM 디스크 소진, `requirements.lock`/
  Dockerfile 페어링 자체는 hash 검증까지 통과)을 **재현**한다.
- 빌드 도중 호스트 파일시스템 여유 공간을 10초 간격으로 감시했다
  (`df -k /`): 시작 시 14GiB, 종료 시 14GiB — **호스트 디스크는 소진되지
  않았다.** 소진된 것은 Docker Desktop VM 자체의 가상 디스크
  (`docker system df`: Images 24.57GB/Volumes 31.34GB/Build Cache
  18.95GB, 총 사용량이 이미 크고 reclaimable이 45GB+ 존재)다.
- **다른 프로젝트가 점유한 공유 Docker 이미지/볼륨을 이 세션이 임의로
  prune하는 것은 범위 밖 파괴적 작업으로 판단해 수행하지 않았다** —
  Implementation_Report.md §5의 동일 결정을 재확인·계승.
- production-stage build는 동일한 `builder` 단계(`requirements.lock` pip
  install)를 공유하므로 별도로 시도하지 않았다 — 같은 지점에서 동일하게
  실패할 것이 확실하다.
- 결론: **실제 이미지 build/scan/smoke는 이 로컬 환경에서 완주 불가**.
  hosted CI(ubuntu-latest, x86_64, 충분한 디스크)로 이연한다.

### 2.10 4-producer receipt / assembler / check_m4_baseline 시뮬레이션

`GITHUB_SHA=648e3abcca7c321e7f4dd13a7fbed1a4f1886c3e`,
`GITHUB_RUN_ID=local-sim-m43-acceptance`, `GITHUB_RUN_ATTEMPT=1`,
`GITHUB_EVENT_NAME=workflow_dispatch`로 고정해 4개 producer 전부를
`write_ci_producer_receipt.py`(실제 hosted step과 동일 스크립트)로 생성했다.

| Producer | Payload 출처 |
|---|---|
| `python-tests` | 실제 — §2.2의 실제 pytest 실행 후 receipt 작성(payload 없음, hosted step과 동일 계약) |
| `frontend-tests` | 실제 — §2.3의 실제 `npm test` 실행 후 receipt 작성(payload 없음) |
| `container` | **SIMULATED** — §2.9에서 실제 이미지 build가 실패했으므로, `scan_image_layers.py`/`container_smoke.py`의 실제 출력 schema(`m43-layer-scan-v1`/`m43-container-smoke-v1`)를 그대로 따르는 payload를 수기로 작성하고 `_note` 필드에 시뮬레이션임을 명시. `forbidden_count: 0`, `_ALL_OK_KEYS`(`host_gateway_reachable`/`mock_query_ok`/`root_page_ok`/`static_asset_ok`/`production_test_seam_sealed`) 전부 `true`로 설정 — assembler가 실제로 검증하는 필드만 채웠으며 실제 이미지의 증거라고 주장하지 않는다. 이 관행은 Implementation_Report.md §7이 이미 사용한 것과 동일한 방법이다. |
| `m43-deterministic` | 실제 — §2.6/§2.7에서 방금 실행한 진짜 `m43.json`/`m43-negative.json`을 payload로 사용 |

```
venv/bin/python scripts/assemble_m4_evidence.py --fresh-dir <tmp>/m4-assemble \
  --expected-sha 648e3abcca7c321e7f4dd13a7fbed1a4f1886c3e \
  --expected-run-id local-sim-m43-acceptance --expected-run-attempt 1 \
  --expected-workflow-path .github/workflows/ci.yml --expected-event workflow_dispatch \
  --needs-result python-tests=success --needs-result frontend-tests=success \
  --needs-result container=success --needs-result m43-deterministic=success \
  --evidence python-tests=... --evidence frontend-tests=... \
  --evidence container=... --evidence m43-deterministic=... \
  --output <tmp>/m4-assemble/m4-baseline.json
```

결과(exit 0), baseline candidate:

```json
{
  "deterministic_status": "PASS",
  "operational_status": "BLOCKED",
  "M4.1_BLOCKED": true,
  "overall_release_ready": false,
  "gates": {
    "python_tests": "PASS", "frontend_tests": "PASS",
    "container": "PASS", "m43_deterministic": "PASS",
    "m41_operational": "BLOCKED", "m3_live_regression": "NOT_RUN"
  },
  "producers": {
    "python-tests": {"status": "OK"}, "frontend-tests": {"status": "OK"},
    "container": {"status": "OK"}, "m43-deterministic": {"status": "OK"}
  }
}
```

```
venv/bin/python scripts/check_m4_baseline.py --candidate <tmp>/m4-assemble/m4-baseline.json \
  --expect-operational-blocked
```

결과: `{"ok": true, "issues": []}`, **exit 0**.

**확인된 판정 불변식(요구된 4개 값 전부 정확히 재현):**

- `M4.1_BLOCKED = true`
- `m3_live_regression 게이트 = NOT_RUN`(protected M3 live는 이 세션은 물론
  hosted candidate에서도 아직 실행되지 않음)
- `operational_status = BLOCKED`
- `overall_release_ready = false`

## 3. 발견 사항과 예외 처리

### 3.1 `pip check` 실패 — M4.3 무관 사전 존재 환경 조건 (예외로 처리)

```
langgraph-prebuilt 1.0.2 has requirement langchain-core>=1.0.0, but you have langchain-core 0.3.86.
langchain-classic 1.0.0 has requirement langchain-core<2.0.0,>=1.0.0, but you have langchain-core 0.3.86.
langchain-classic 1.0.0 has requirement langchain-text-splitters<2.0.0,>=1.0.0, but you have langchain-text-splitters 0.3.11.
```

`git stash`로 이 세션의 모든 working tree 변경을 제거하고 기준 revision
`648e3ab`(M4.3 변경 없는 상태)에서 동일 명령을 재실행해 **동일한 3건의
경고가 그대로 재현됨**을 확인했다(`git stash pop`으로 복원). 즉 이 조건은
venv에 이미 설치된 패키지 조합의 사전 존재 상태이며 M4.3 코드/의존성
변경으로 유발되지 않았다. `requirements.lock`은 `compile_lock.sh --verify`로
재현 가능함이 확인됐고 이 pip check 경고는 lock 자체의 결함이 아니라 로컬
venv에 설치된 실제 패키지 버전 간의 사전 존재 mismatch다. 가이드
"Gate가 가져야 할 진행 여부 결정의 지침" §2-(1) "환경 상의 제약으로 성공할
수 없는 케이스는 예외로 간주하고 넘어간다"에 따라 **예외로 처리**하며 M4.3
pre-merge Gate 판정에 영향을 주지 않는다. 코드 수정을 하지 않았다(범위 밖).

### 3.2 실제 OCI 이미지 build/scan/smoke — 호스트 디스크 제약 (hosted CI로 이연)

§2.9 참조. Dockerfile 자체의 계약(hash-verified install, numeric non-root
user, minimal COPY surface)은 정적으로 재확인했고 unit 테스트(`test_scan_image_layers.py`,
`test_container_smoke_contract.py`)로 스캐너/스모크 로직은 전수 검증됐다.
실제 이미지 대상 build/scan/smoke만 로컬 환경 제약으로 미완주하며, 이는
코드 결함이 아니라 이 세션이 반복 확인한 동일 인프라 제약이다.

### 3.3 신규 코드 결함

**발견 없음.** 이 세션은 코드를 수정하지 않았다. 모든 pytest/스크립트
실행이 Code Review Iteration 2가 근거로 삼은 수치와 정확히 일치했으며 신규
회귀가 전혀 없었다.

## 4. 의도적으로 실행하지 않은 것 (지시된 경계)

- Native Linux 실행, Ollama, DDGS 실제 네트워크/모델 호출
- protected M3 live 14-gate(self-hosted, `ollama-m3` label, `m3-live-regression`
  environment 승인)
- M4.1 live 14-gate(운영 승인 대상)
- 실제 hosted GitHub Actions 실행(이 세션은 commit/push/PR을 하지 않음)
- self-hosted runner/environment 승인 설정 변경(전혀 건드리지 않음)

## 5. Requirement Traceability 재확인

[Traceability.md](Traceability.md)의 각 행을 이 세션이 재실행한 명령/증거와
대조했다. **모든 claim이 검증된 상태로 확인되어 문서 수정이 필요한
불일치를 발견하지 못했다** — 따라서 Traceability.md/Implementation_Report.md는
이 세션에서 변경하지 않는다(가이드 지시: "Update Traceability/Implementation_Report
only for verified truth" — 검증 결과가 기존 문서와 완전히 일치하므로 갱신할
"새로운 진실"이 없다).

| Requirement | 이 세션 재확인 근거 |
|---|---|
| M4.3-REQ-001~004(canonical index/lifecycle/CLI) | §2.2 전체 suite 1173 passed 포함 |
| M4.3-REQ-005(OCI image) | §2.4 정적 계약 26 passed, §2.9 실제 build 재시도(동일 실패 재현), Dockerfile 재검토 |
| M4.3-REQ-006(runbook) | §2.8 deploy drill 실측 재실행 |
| M4.3-REQ-007(single workflow) | §2.4, §2.5(byte 단위 protected block 검증), §2.10(4-producer 시뮬레이션) |
| M4.3-REQ-008(M4 baseline) | §2.10 checker `{"ok": true, "issues": []}` |
| M4.3-REQ-009(watchdog) | §2.2에 포함(orchestration_watchdog 16/16) |
| M4.3-NFR-001~006 | §2.2, §2.6, §2.8 |

## 6. Pre-merge Gate 판정

**Pre-merge Code Quality Gate: PASS 유지(9.8/10, 변경 없음)** — Code Review
Iteration 2의 판정을 이 세션이 독립적으로 재현·재확인했다. 신규 CRITICAL/MAJOR/MINOR
없음. §3.1/§3.2의 두 항목은 모두 기존에 문서화된 환경 제약의 재확인이며
새로운 코드 품질 finding이 아니다.

**Post-merge Operational Acceptance Gate: 전제 조건 대로 미충족**
(가이드 §11에 따라 pre-merge 판정의 선행조건이 아님):

```
M4_OVERALL_RELEASE_READY
  = M4.3_DETERMINISTIC_PASS(true, 로컬)
    AND M4.1_OPERATIONAL_PASS(false — BLOCKED)
    AND PROTECTED_M3_LIVE_PASS(false — NOT_RUN)
    AND exact_post_merge_release_identity_is_verified(NOT_RUN — 미커밋)
  = false
```

`M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`, `overall_release_ready=false`
— 이 세션의 §2.10 실측이 이 판정 불변식을 정확히 재확인했다. 어떤 스크립트도
이 값을 합성·default로 `true`로 만들 수 없음을 코드 검토로 재확인했다.

## 7. Release-worker 인계 체크리스트

머지를 진행하는 release worker는 다음을 순서대로 수행해야 한다.

1. **커밋 범위 확인**: 이 working tree의 modified/untracked 파일 전체가
   M4.3 범위와 일치하는지 최종 `git status`/`git diff --stat`로 확인.
   `runtime/`, `.env`, 개인 스크래치 파일이 섞여 있지 않은지 확인.
2. **commit → push → PR 생성**. PR 본문에 이 Acceptance_Report.md와
   Code_Review_Iteration_2.md를 링크.
3. **hosted CI 관찰**: PR push 후 `python-tests`, `frontend-tests`, `container`,
   `m43-deterministic`, `m4-assemble` 5개 job이 모두 실행되는지 확인.
   `container`/`m4-assemble` job은 이 세션이 로컬에서 완주하지 못한
   **첫 실제 hosted 실행**이므로 특히 주의 깊게 관찰할 것 — 실제
   x86_64/충분한 디스크 환경에서 §2.9가 예측한 대로 성공하는지가 이
   머지의 가장 중요한 미검증 지점이다.
4. **`m3-live-regression-gate` 무변경 재확인**: hosted 실행 전에 PR diff에서
   해당 job 블록에 어떤 변경도 없는지 GitHub UI로 재확인(§2.5의 로컬
   byte-비교를 hosted 관점에서 재확인).
5. **`m4-baseline` artifact 다운로드 후 검사**: hosted 실행이 끝나면
   업로드된 `m4-baseline.json`을 받아 `M4.1_BLOCKED=true`,
   `operational_status=BLOCKED`, `overall_release_ready=false`,
   `gates.m3_live_regression=NOT_RUN`을 재확인. 이 중 하나라도
   다르면 merge를 중단하고 즉시 조사.
6. **container job 실패 시**: §2.9의 로컬 재현과 같은
   `No space left on device`가 hosted에서도 발생한다면 이는 (a) 로컬
   emulation 특유의 문제였거나 (b) Dockerfile/lock 자체의 결함일 수 있다.
   hosted runner는 보통 넉넉한 디스크를 가지므로, hosted에서도 실패하면
   pre-merge PASS 판정과 무관하게 **fresh code review 대상**으로 반드시
   재분류해야 한다(이 세션은 이 가능성을 배제하지 못했다 — 로컬 hash-verified
   install까지만 확인했다).
7. **merge 후**: M4.1 operational exception과 protected M3 live 승인은
   이 머지로 해소되지 않는다. M4 전체 release는 계속 BLOCKED다. 별도
   운영 승인 절차(M4.1 live 14-gate, M3 live 14-gate)가 완료되기 전까지
   `overall_release_ready`를 true로 표시하는 어떤 문서/스크립트 변경도
   생성하지 말 것.

## 8. Hosted CI 기대치

- `python-tests`/`frontend-tests`: 이 세션이 이미 동일 명령을 로컬에서
  실행해 확인했으므로 hosted에서도 동일하게 PASS해야 한다. 차이가 나면
  환경 차이(Python/Node 버전, lock drift)를 먼저 의심할 것.
- `container`: **로컬에서 완주하지 못한 유일한 job.** §2.9의 hash-verified
  install까지는 아키텍처/디스크와 무관하게 결정론적이므로 hosted
  ubuntu-latest(native x86_64, 통상 충분한 디스크)에서는 `pip install`
  단계를 통과할 것으로 예상하지만, 이 세션은 그 이후 단계(`test`/`production`
  stage build, `scan_image_layers.py`, `container_smoke.py`)를 실제로
  한 번도 실행하지 못했다 — hosted가 이 로직의 **최초 실제 실행**이다.
- `m43-deterministic`: 로컬에서 실제로 정확히 같은 명령으로 실행해
  positive/negative 모두 확인했으므로 hosted에서도 동일 결과가 예상된다.
- `m4-assemble`: 로컬 4-producer 시뮬레이션(§2.10)이 assembler/checker
  로직 자체를 실제 스크립트로 검증했다. hosted에서 실제 4개 job 산출물을
  다운로드해 조립하는 것은 `needs`/artifact 배선의 최초 실제 실행이며,
  `container` producer가 처음으로 진짜(시뮬레이션이 아닌) payload를 갖게
  된다.
- `m3-live-regression-gate`: 이 워크플로 실행 대상이 아니다(별도 트리거,
  self-hosted, environment 승인). 이 PR의 hosted run에서 트리거되지 않아야
  정상이다.

## 9. 최종 요약

- **로컬 결정론적 M4.3 사이클: PASS**(Python/Frontend 전체 suite, 정적/생성/링크
  계약, positive/negative acceptance repeat=10 seed=4303, deploy drill,
  4-producer 시뮬레이션 전부 재현 확인).
- **신규 코드 결함: 0건.** 코드 변경 없음.
- **실제 이미지 build/scan/smoke: 로컬 환경 제약으로 미완주**(정확한 증거
  기록, hosted CI로 이연).
- **`M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`,
  `overall_release_ready=false`를 이 세션이 독립적으로 재확인했다.**
- Traceability.md/Implementation_Report.md는 기존 claim이 모두 검증되어
  갱신 없음.
- Pre-merge Code Quality Gate: **PASS 유지(9.8/10)**. Post-merge
  Operational Acceptance Gate: 여전히 미충족(설계상 정상, 별도 운영 승인
  절차 필요).
