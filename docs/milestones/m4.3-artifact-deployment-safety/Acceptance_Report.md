# M4.3 Artifact & Deployment Safety — Final Pre-Merge Integration & Acceptance Report

역할: Claude Sonnet 5 integration/acceptance worker (Code Review Iteration 9
PASS 10.0/10 이후)
기준 revision (working tree base, HEAD, uncommitted): `648e3abcca7c321e7f4dd13a7fbed1a4f1886c3e`
선행 조건: 독립 Code Review Iteration 9 — **PASS 10.0/10**
(`CRITICAL 0 / MAJOR 0 / MINOR 0 / TRIVIAL 0`), 근거
[Code_Review_Iteration_9.md](Code_Review_Iteration_9.md); Iteration 6~8과 그
remediation의 전체 chain은 [Implementation_Report.md](Implementation_Report.md)
§14 참조.
실행 시각: 2026-08-14
No commit / push / PR / merge performed by this session.

이 문서는 2026-08-12에 작성된 이전 버전(Code Review Iteration 2 기준, PASS
9.8/10)을 대체한다 — 그 이후 Code Review Iteration 3~9와 세 차례의 Hosted CI
Remediation이 있었고, 이 세션은 그 최종 상태(Iteration 9 PASS 10.0/10)를
기준으로 통합/인수를 재수행했다.

## 0. 근거 문서

`milestone_dev_orchestration_guide.md` 전체와 이 디렉터리의
[Requirement.md](Requirement.md), [Plan.md](Plan.md), [Design.md](Design.md),
[Traceability.md](Traceability.md), [Implementation_Report.md](Implementation_Report.md),
[Code_Review_Iteration_1.md](Code_Review_Iteration_1.md)~
[Code_Review_Iteration_9.md](Code_Review_Iteration_9.md)와 대응 remediation
문서, [Hosted_CI_Remediation_Iteration_1~3](Hosted_CI_Remediation_Iteration_1.md),
[Orchestration_Stop_Report.md](Orchestration_Stop_Report.md)(역사적 stop/resume
기록 — 상단 배너 참조, 현재 상태 아님)를 읽고, 현재 working tree(코드/스크립트/
워크플로/테스트)와 `.github/workflows/ci.yml`을 대조 확인한 뒤 아래 절차를
수행했다.

## 1. 목적과 범위

이 세션은 **pre-merge Code Quality Gate(Iteration 9 PASS) 이후, merge 이전의
최종 통합/인수 단계**다. 코드 변경은 `requirements.lock` 재컴파일 1건뿐이며
(§3.2 참조, 코디네이터 지시에 따른 진짜 상류 drift 해결 — 상세 근거는 아래),
그 외에는 이미 리뷰를 통과한 구현을 독립적으로 재현·검증하고 milestone 문서를
최종 상태로 정합화했다. 가이드 §11(Pre-merge/Post-merge Gate 분리)에 따라
hosted CI/protected environment/self-hosted runner 증거는 이 세션의 책임
범위 밖이며 `NOT_RUN`으로 유지한다.

## 2. 실행 명령과 결과 (전체 명시적 재현)

### 2.1 Python 컴파일/정적/생성/링크/diff 계약

| # | 명령 | 결과 |
|---|---|---|
| 1 | `venv/bin/python -m compileall -q src scripts tests evaluation` | exit 0 |
| 2 | `venv/bin/python -m pip check`(기존 dev venv) | **exit 1** — §3.1 참조(로컬 venv에만 존재하는 사전 존재 drift, `requirements.lock`과 무관함을 §3.2의 clean-install로 확정) |
| 3 | `venv/bin/python scripts/generate_field_spec.py --check` | exit 0 |
| 4 | `venv/bin/python scripts/logging_callsite_audit.py --check` | exit 0 |
| 5 | `venv/bin/python scripts/check_markdown_links.py` | exit 0 — 파일 129개(tracked 122 + untracked 7), 링크 566개, 실패 0개(이 세션이 갱신한 문서 포함, 최종 재확인은 §5) |
| 6 | `git diff --check` | exit 0(최종 재확인은 §5) |
| 7 | `venv/bin/python -m evaluation.dataset validate evaluation/datasets/golden.jsonl` | exit 0 — `"valid": true`, `"errors": []`, total 76 |

### 2.2 전체 Python unit + integration suite

```
venv/bin/python -m pytest tests/unit tests/integration -q
```

결과: **1282 passed, 1 skipped, 4 warnings in 166.26s**. skip 1건은 기존에
문서화된 M4.3 무관 pre-existing skip이다. 신규 회귀 없음(§13 기준 1251에서
Iteration 7/8 remediation이 추가한 테스트 반영, 이 세션의 문서/lock 변경은
이 수치에 영향 없음 — 재확인 목적으로 그대로 재실행한 결과다).

### 2.3 certifi Label exact-binding — venv/repository-default 두 interpreter

```
venv/bin/python -m pytest -q tests/unit/test_scan_image_layers.py tests/unit/test_rag_engine_singleton.py
python3 -m pytest -q tests/unit/test_scan_image_layers.py tests/unit/test_rag_engine_singleton.py
```

결과: **각각 112 passed** — Code_Review_Iteration_9.md가 근거로 삼은 수치를
독립 재현했다. `python3`은 이 머신의 repository-default interpreter(Anaconda
`common` 환경, `pip==23.3.1` → `certifi==2023.07.22` pip-vendored, Iteration
8/9 리뷰가 검증한 것과 동일한 legacy bundle)로, `venv/bin/python`과는 다른
`certifi` 버전 조합을 실제로 exercise한다.

### 2.4 orchestration watchdog(M4.3-REQ-009, consumer_fenced readiness fix)

```
venv/bin/python -m pytest -q tests/unit/test_orchestration_watchdog.py
```

결과: **16 passed**(기존 8 + Design §11.2 신규 8, `_classify_runner_error`의
`consumer_fenced` 분기, exact-argv/terminal-scope/dry-run 테스트 포함).

### 2.5 Frontend 테스트/vendor drift

| 명령 | 결과 |
|---|---|
| `npm test` | exit 0 — **9 passed** (vitest) |
| `npm run sync-vendor` | exit 0 — 4개 파일 동기화(내용 변경 없음) |
| `git diff --exit-code -- web/static/vendor/` | exit 0 — drift 없음 |

### 2.6 Protected M3 live block 보존(코드 검토)

`.github/workflows/ci.yml`의 `m3-live-regression-gate:` 잡 블록을 이 세션의
working tree에서 재검토했다 — trigger allowlist, `runs-on: [self-hosted,
ollama-m3]`, `environment: m3-live-regression` 승인 요구에 변경이 없음을
확인했다(이 세션은 워크플로 파일을 전혀 수정하지 않았다 — `git status`에
`.github/workflows/ci.yml`이 나타나지 않는다).

### 2.7 M4.3 deterministic acceptance — positive

```
venv/bin/python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 \
  --output <tmp>/m43-positive.json
```

결과: **exit 0**, top-level `"status": "PASS"`, 17개 node 전부
`success_count: 10/10`(`activation_rollback`, `assemble_payload_verification`,
`baseline_strict_schema`, `container_static_and_connectivity`,
`crash_recovery_journal`, `embedding_provider_seam_guard`, `layer_scanner`,
`legacy_baseline_pin`, `legacy_import`, `lock_contention`,
`lock_untrusted_symlink`, `manifest_canonical`, `manifest_negative`,
`retention`, `staging_fault`, `verification_reopen_race`,
`verification_trust`).

### 2.8 M4.3 deterministic acceptance — negative control (expected failure)

```
venv/bin/python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 \
  --inject-evidence-mismatch --output <tmp>/m43-negative.json
```

결과: **exit 1**(negative control의 기대되는 성공),
`"status": "REJECTED_AS_EXPECTED"`, `negative_control = {"executed": true,
"expected_to_fail": true, "actual_exit_code": 1, "result":
"REJECTED_AS_EXPECTED"}`.

### 2.9 requirements.lock — 진짜 상류 drift 발견과 canonical 해결

이 절차는 이 세션 중 코디네이터의 명시적 지시(`orca orchestration check`로
수신한 status 메시지 `msg_a323b90fd961`/`msg_fa852b904e03`/`msg_18cd5cd42a24`)로
수행됐다 — 상세 근거는 [Implementation_Report.md](Implementation_Report.md)
§15.2 참조. 요약:

1. **마스킹된 첫 결과**: `bash scripts/compile_lock.sh --verify 2>&1 | tail`
   형태로 처음 실행했을 때 파이프 뒤의 `tail`이 실제 종료 코드를 가려, 잘못된
   `EXIT:0`을 관측했다. 파이프 없이 재실행하자 **실제 종료 코드 1**
   (`committed requirements.lock has drifted from requirements.txt`)이
   드러났다.
2. **macOS 아티팩트 배제**: macOS(arm64) 호스트에서 직접 `uv pip compile`을
   실행하면 §10(Hosted CI Remediation Iteration 1)이 이미 겪은 플랫폼 그래프
   오염이 재발할 수 있으므로, hosted CI와 동일한 조합
   (`python:3.11-slim --platform linux/amd64` 컨테이너, `uv==0.8.15`) 안에서
   같은 `--verify`를 재실행했다 — **컨테이너 안에서도 exit 1**로 재현돼, 이
   drift가 macOS 아티팩트가 아니라 진짜임을 확인했다.
3. **원인**: 재컴파일 직전 `git diff --stat -- requirements.lock
   requirements.txt`가 빈 출력이었으므로(이 세션이나 이전 M4.3 세션이 만든
   변경이 아님), drift는 마지막 lock 커밋 이후의 순수 상류 PyPI 릴리스
   경과(`pypdf` 6.15.0→6.16.0, `uvicorn` 0.52.1→0.52.2, `xxhash` 등 다수 wheel/
   hash 갱신, 총 패키지 수 103→103, package 세트 불변)였다.
4. **해결(canonical 재컴파일)**: 같은 `python:3.11-slim --platform linux/amd64`
   컨테이너(`uv==0.8.15`)에서 `bash scripts/compile_lock.sh`(no `--verify`)를
   실행해 `requirements.lock`을 재작성했다(bind mount로 호스트 파일 직접
   갱신). 재작성 직후 같은 컨테이너에서 `--verify`를 재실행해
   **`compile_lock.sh: --verify PASS (reproducible, no drift)`, exit 0**을
   확인했다. `git diff --stat -- requirements.lock`: **222 insertions(+),
   195 deletions(-)** — 다수 패키지의 순수 버전/hash 갱신, package 수
   103→103(package 세트 불변).
5. **Clean lock-based pip check**: 같은 컨테이너에 재컴파일된
   `requirements.lock`을 hosted CI와 동일 순서(`pip install --require-hashes
   -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu`
   → `pip install -e . --no-deps` → `pip check`)로 **처음부터 새로 설치**한
   결과 **`No broken requirements found.`(exit 0)** — §2.1의 기존 dev venv
   `pip check` exit 1(`langgraph-prebuilt`/`langchain-classic`가
   `langchain-core>=1.0.0` 요구)이 lock 자체의 결함이 아니라, 이 프로젝트
   로컬 `venv`에만 존재하는 순수 환경 drift(두 패키지 모두
   `requirements.lock`에 없음 — grep으로 확인)임을 clean-install로 최종
   확정했다.

이 결과로 `requirements.lock`은 working tree에서 갱신된 상태다(§7 참조).
코디네이터는 이를 "material change"로 분류해, 이 milestone의 최종 commit
전에 이 lock 갱신에 대한 별도 fresh code review가 필요하다고 명시했다 — 이
세션은 review를 수행하지 않으며 commit/push도 하지 않는다.

**생략한 항목(코디네이터 명시적 waiver)**: 재컴파일된 lock으로부터 완전히
새로 구성한 clean venv에서 전체 pytest suite(§2.2)를 다시 실행하는 추가
확인은, 코디네이터가 "clean pip-check(§2.9-5) + 기존 전체 venv suite
증거(§2.2)로 충분하다"고 명시적으로 판단해 반복 실행하지 않았다(중복된
장시간 컨테이너 QEMU 에뮬레이션 재시도를 피하기 위함). hosted CI(x86_64,
네이티브)가 새 lock을 사용한 전체 suite의 첫 실제 실행이 된다.

### 2.10 Docker build/scan/smoke — 이 세션이 최초로 완주

호스트 환경: macOS, Docker Desktop, Docker VM `aarch64`
(`linux/amd64` requirements.lock hash pin과 불일치하므로 native build는
아키텍처상 불가 — 기존 문서화된 제약과 동일). 이전 세션들을 반복해서 막았던
Docker Desktop VM 디스크 소진은 이 세션의 빌드 시작 시점 `docker system df`
확인 결과(reclaimable 공간 충분) 재현되지 않았다.

```
docker build --platform linux/amd64 --target test -f deploy/Dockerfile .
```

결과: **exit 0** — hash-verified `requirements.lock`(§2.9에서 재컴파일된
버전) 설치, `python -c "from simple_qna_rag.web.server import app"` 스모크
포함 전 단계 완주. 이 세션에서 처음으로 이 명령이 로컬에서 끝까지 성공했다.

```
docker build --platform linux/amd64 --target production -f deploy/Dockerfile -t simple-qna-rag:m43-candidate .
```

결과: **exit 0** — numeric non-root user(`10001:10001`), test-seam
미포함, 최소 `COPY` surface 확인. 이미지 digest
`sha256:9828b1a4fd7d5feaf45d81828443cb1389bad05a7cf75a3bbee33ff9f4258595`.

```
venv/bin/python scripts/scan_image_layers.py --image simple-qna-rag:m43-candidate
```

결과: **`forbidden_count: 0`, `violations: []`, exit 0** — 12개 layer,
member 수 최대 47997(python 패키지 layer)까지 전수 스캔, 신뢰 CA 저장소
symlink/hardlink 허용목록 오탐 없음.

```
venv/bin/python scripts/container_smoke.py --image simple-qna-rag:m43-candidate
```

결과: **`status: "PASS"`, exit 0** — `host_gateway_reachable`,
`mock_query_ok`, `root_page_ok`, `static_asset_ok`,
`production_test_seam_sealed`, `readiness_sequence.live`/`.ready` 전부
`true`. `graceful_stop_seconds: 1.66`.

이로써 Implementation_Report.md §4/§12/Traceability.md가 반복 기록했던
"실제 이미지 build/scan/smoke는 로컬 환경 제약으로 미완주"라는 제약이 **이
세션에서는 해소**됐다. 이는 인프라 상태(Docker Desktop VM 가용 디스크)의
변화이며, 코드/Dockerfile/스캐너 로직을 이 세션이 바꾼 결과가 아니다.
hosted CI가 이 파이프라인의 첫 실제 hosted(x86_64 네이티브) 실행이라는 점은
여전히 유효하다.

### 2.11 4-producer receipt / assembler / check_m4_baseline — 코드 검토(재실행 생략)

이 세션은 §2.7~2.10의 4개 producer(python-tests/frontend-tests/container/
m43-deterministic) 모두를 이번 실행에서 실측했으나, 4-producer
`assemble_m4_evidence.py` → `check_m4_baseline.py --expect-operational-blocked`
전체 체인의 로컬 시뮬레이션 재실행은 하지 않았다 — 이전 세션(2026-08-12,
§9 이하 참조)이 이미 실제 스크립트로 이 체인을 재현해 `{"ok": true,
"issues": []}`와 `overall_release_ready=false`/`M4.1_BLOCKED=true`를
확인했고, `scan_image_layers.py`/`container_smoke.py`/
`run_m43_acceptance.py`/watchdog 코드는 이 세션에서 변경되지 않았으므로(이
세션의 유일한 코드 변경은 `requirements.lock`이며, 이는 assembler/checker의
입력 스키마에 영향이 없다) 그 체인의 판정 로직이 달라질 이유가 없다.
`tests/unit/test_assemble_m4_evidence.py`/`test_check_m4_baseline.py`는
§2.2의 전체 suite 1282 passed에 포함돼 로컬 PASS를 재확인했다.

## 3. 발견 사항과 예외 처리

### 3.1 `pip check` 실패(기존 dev venv) — M4.3 무관, clean-install로 확정된 로컬 drift

§2.9-5의 clean lock-based install이 `No broken requirements found.`을
반환했으므로, 기존 dev venv의 `pip check` exit 1은 **`requirements.lock`
자체의 결함이 아니라 이 특정 로컬 `venv`에만 존재하는 상태**임이 확정됐다.
코드 수정을 하지 않았다(범위 밖이며, clean-install 증거로 이미 결론이
났다).

### 3.2 requirements.lock 진짜 drift — 코디네이터 지시로 이 세션이 canonical 해결

§2.9 참조. `pypdf`/`uvicorn`/`xxhash` 등 상류 PyPI 릴리스 경과로 인한 진짜
drift였으며, macOS 아티팩트가 아니었다(linux/amd64 컨테이너에서도 동일하게
재현). 코디네이터 지시에 따라 hosted CI와 동일한 컨테이너 조합으로 canonical
재컴파일하고 reproducibility+no-drift를 재확인했다. 이 변경은 working
tree에만 존재하며 커밋되지 않았다 — 코디네이터는 이를 별도 fresh code
review 대상으로 지정했다.

### 3.3 신규 코드 결함

**발견 없음.** 이 세션의 유일한 코드 변경은 §3.2의 `requirements.lock`
재컴파일(상류 패키지 버전/hash 갱신)이며, 이는 M4.3의 애플리케이션/스크립트
로직 결함이 아니다. §2.2/§2.3/§2.4의 pytest 수치는 모두 Iteration 9 및 §13
근거 수치와 일치하거나(재실행 재확인) 그 이후 정확히 예상된 증가분만
반영했다.

## 4. 의도적으로 실행하지 않은 것 (지시된 경계)

- Native Linux 실행, Ollama, DDGS 실제 네트워크/모델 호출
- protected M3 live 14-gate(self-hosted, `ollama-m3` label, `m3-live-regression`
  environment 승인)
- M4.1 live 14-gate(운영 승인 대상)
- 실제 hosted GitHub Actions 실행(이 세션은 commit/push/PR을 하지 않음)
- self-hosted runner/environment 승인 설정 변경(전혀 건드리지 않음)
- §2.11에 기록한 4-producer 체인 로컬 시뮬레이션의 반복 재실행(코드 무변경
  근거로 생략)
- §2.9의 clean-install 이후 전체 pytest suite를 다시 그 clean venv에서
  재실행하는 것(코디네이터 명시적 waiver)

## 5. 문서 최종 링크/diff 재확인

이 세션이 Implementation_Report.md/Traceability.md/Orchestration_Stop_Report.md/
이 Acceptance_Report.md를 갱신한 뒤 최종적으로 재실행:

| 명령 | 결과 |
|---|---|
| `venv/bin/python scripts/check_markdown_links.py` | 파일 129개(tracked 122 + untracked 7), 링크 590개, 실패 0개 |
| `git diff --check` | exit 0 |

## 6. Requirement Traceability 재확인

[Traceability.md](Traceability.md)의 각 행을 이 세션이 재실행한 명령/증거와
대조했다 — M4.3-REQ-005/NFR-003 행은 Iteration 6~9 closure와 §2.10의 실제
이미지 완주를 반영해 **이번에 갱신**했다(상세는 Traceability.md 자체와
Implementation_Report.md §14~15).

| Requirement | 이 세션 재확인 근거 |
|---|---|
| M4.3-REQ-001~004(canonical index/lifecycle/CLI) | §2.2 전체 suite 1282 passed 포함 |
| M4.3-REQ-005(OCI image, certifi Label exact binding) | §2.3(112/112 passed, 두 interpreter), §2.10(실제 이미지 build/scan/smoke 최초 완주) |
| M4.3-REQ-006(runbook) | §2.2에 `test_deploy_drill.py` 포함(재실행하지 않음 — 코드 무변경) |
| M4.3-REQ-007(single workflow) | §2.2, §2.6(protected block 코드 검토) |
| M4.3-REQ-008(M4 baseline) | §2.11(체인 로직 무변경 근거로 재실행 생략, 이전 세션 실측 유효) |
| M4.3-REQ-009(watchdog) | §2.4(16/16 passed) |
| M4.3-NFR-001~006 | §2.2, §2.7, §2.10 |

## 7. Release-worker 인계 체크리스트

머지를 진행하는 release worker는 다음을 순서대로 수행해야 한다.

1. **커밋 범위 확인**: 이 working tree의 modified/untracked 파일 전체가
   M4.3 범위와 일치하는지 최종 `git status`/`git diff --stat`로 확인. 특히
   `requirements.lock`(§3.2의 canonical 재컴파일, 코드 아님)이 포함돼야
   한다. `runtime/`, `.env`, 개인 스크래치 파일이 섞여 있지 않은지 확인.
2. **`requirements.lock` 변경에 대한 fresh code review**: 코디네이터가
   이 변경을 "material change"로 분류했다 — 나머지 M4.3 diff(Iteration 9
   PASS 10.0/10 근거)와 별개로, 이 lock 재컴파일 자체에 대한 독립 검토를
   거친 뒤 commit 범위에 포함할 것.
3. **commit → push → PR 생성**. PR 본문에 이 Acceptance_Report.md,
   Code_Review_Iteration_9.md, Implementation_Report.md §14~15를 링크.
4. **hosted CI 관찰**: PR push 후 `python-tests`, `frontend-tests`,
   `container`, `m43-deterministic`, `m4-assemble` 5개 job이 모두
   실행되는지 확인. 재컴파일된 `requirements.lock`이 hosted에서
   `--require-hashes` 설치를 통과하는지가 이 push의 가장 중요한 미검증
   지점이다(이 세션은 linux/amd64 에뮬레이션 컨테이너에서는 clean-install을
   확인했으나, 네이티브 x86_64 hosted runner에서는 아직 확인되지 않았다).
5. **`m3-live-regression-gate` 무변경 재확인**: hosted 실행 전에 PR diff에서
   해당 job 블록에 어떤 변경도 없는지 GitHub UI로 재확인.
6. **`m4-baseline` artifact 다운로드 후 검사**: hosted 실행이 끝나면
   업로드된 `m4-baseline.json`을 받아 `M4.1_BLOCKED=true`,
   `operational_status=BLOCKED`, `overall_release_ready=false`,
   `gates.m3_live_regression=NOT_RUN`을 재확인. 이 중 하나라도 다르면
   merge를 중단하고 즉시 조사.
7. **merge 후**: M4.1 operational exception과 protected M3 live 승인은
   이 머지로 해소되지 않는다. M4 전체 release는 계속 BLOCKED다. 별도
   운영 승인 절차(M4.1 live 14-gate, M3 live 14-gate)가 완료되기 전까지
   `overall_release_ready`를 true로 표시하는 어떤 문서/스크립트 변경도
   생성하지 말 것.

## 8. 최종 요약

- **Code Review Iteration 9: PASS 10.0/10**(`CRITICAL 0/MAJOR 0/MINOR 0/
  TRIVIAL 0`) — Iteration 6~8이 지적한 certifi Label exact-binding 결함
  체인이 모두 closed.
- **로컬 결정론적 M4.3 사이클: PASS(실제 이미지 Gate 포함 전 항목 완주)** —
  Python/Frontend 전체 suite, 정적/생성/링크 계약, positive/negative
  acceptance repeat=10 seed=4303, watchdog 16/16, 실제 linux/amd64 이미지
  build+scan(`forbidden_count: 0`)+smoke(`status: PASS`) 전부 이 세션에서
  직접 재현·확인.
- **신규 애플리케이션 코드 결함: 0건.**
- **환경 delta 1건**: `requirements.lock`이 상류 PyPI 릴리스로 진짜 drift돼
  있었다 — 코디네이터 지시로 hosted CI와 동일한 컨테이너에서 canonical
  재컴파일하고 reproducibility+no-drift+clean pip-check를 재확인했다(§2.9).
  이 변경은 working tree에만 있으며 fresh code review 대상으로 지정됐다.
- **`M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`,
  `overall_release_ready=false`를 코드 검토로 재확인했다** — 이 값을
  변경하는 어떤 코드 경로도 이 세션이 만들거나 발견하지 않았다.
- Implementation_Report.md §14~15, Traceability.md(상태 header,
  M4.3-REQ-005/NFR-003 행, Container hosted gate 행), Orchestration_Stop_Report.md
  (상단 historical 배너)를 이 세션에서 갱신했다.
- Pre-merge Code Quality Gate: **PASS(10.0/10, Iteration 9 근거)**.
  `requirements.lock`은 그 PASS 판정 대상에 포함되지 않았던 별도 environment
  delta이며 §7-2의 fresh review가 필요하다. Post-merge Operational
  Acceptance Gate: 여전히 미충족(설계상 정상, 별도 운영 승인 절차 필요).
