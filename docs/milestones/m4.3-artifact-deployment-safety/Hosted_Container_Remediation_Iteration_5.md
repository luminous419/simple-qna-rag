# M4.3 Artifact & Deployment Safety — Hosted Container Remediation Iteration 5

작성자: Claude Sonnet 5 (hosted-container remediation worker)
대상: PR #18, hosted run
[31800982514](https://github.com/luminous419/simple-qna-rag/actions/runs/31800982514)
(판정 FAILURE — `container` job FAILURE, `m4-assemble` FAILURE(하류 종속
실패); `python-tests`/`frontend-tests`/`m43-deterministic` SUCCESS,
`m3-live-regression-gate` SKIPPED)
기준 revision: `c51387a`(`agent/m4-3-artifact-deployment-safety`, PR #18
HEAD) — 이 문서가 기술하는 변경은 같은 브랜치에 새 커밋으로 추가될 예정이며
PR을 새로 열지 않는다. **merge는 수행하지 않는다** — fresh Codex 리뷰가
필요하다.

## 0. 범위와 결론 요약

`c51387a`(Hosted_CI_Remediation_Iteration_4.md)는 `_poll_ready`의 포지티브
readiness 폴링 예산을 10초→60초로 늘렸다. 이 hosted run(31800982514)은 그
수정이 **hosted 실패를 고치지 못했음**을 증명한다 — 60초 예산 아래에서도
정확히 같은 실패 지문(`readiness_sequence.live=true`, `ready=false`,
`mock_query_ok=false`, 다른 4개 필드는 true)이 재발했다.

Task 지시대로 **예산을 다시 늘리지 않았다.** 대신 코드를 읽고 재현을
시도한 결과, `container_smoke.py`가 근본 원인을 판정할 수 있는 정보를
스스로 관측한 직후 버려왔다는 실제 결함을 발견했다:

1. `run_smoke()`의 포지티브 readiness 폴링 호출부가
   `ready_ok, _, _ = _poll_ready(host_port, expect_status=200,
   max_seconds=60)`로 `_poll_ready`가 이미 계산한 마지막 HTTP status와
   `/health/ready`의 JSON body가 담고 있던 정확한 거부 사유(reason —
   `observability/health.py`의 `evaluate_readiness()`가 만드는
   `artifact_*`/`settings_invalid`/`engine_init_failed`/... 중 하나)를
   즉시 버렸다.
2. negative-control 봉인 확인도 동일하게 `ok, status, reason =
   _poll_ready(...)`로 계산한 뒤 `sealed = ok`만 쓰고 `status`/`reason`을
   버렸다.
3. 컨테이너 자신의 stdout/stderr(구조화 JSON 로그, `startup` 이벤트가
   정확히 이 `reason`을 다시 담고 있다)는 애초에 한 번도 캡처되지
   않았다 — 성공이든 실패든 스크립트는 최종 판정 JSON 한 덩어리만 찍는다.

즉 hosted 로그(§2.1 인용)가 보여주는 `ready=false`라는 결과는 다음 두
가설 중 어느 쪽인지 이 스크립트 자체가 절대 구분할 수 없는 상태였다:

- **가설 A(순수 타이밍)**: 컨테이너가 정상적으로 기동 중이지만 60초
  예산을 아주 약간 넘겨서 폴링 루프가 먼저 포기했다.
- **가설 B(결정론적 거부)**: 앱은 몇 초 안에 기동을 마쳤지만
  `/health/ready`가 60초 내내 반복적으로 같은 구체적 사유로 503을 반환하고
  있었다(느림이 아니라 실제 거부).

§2에서 로컬 재현으로 가설 A에 대한 반증에 가까운 증거를 얻었지만
100% 반증은 아니다(§2.3). Task의 명시적 지시("do not merely increase
timeout without evidence")에 따라, 이 remediation은 **가설을 확정하는
증거를 만드는 최소 fail-closed 수정**을 전달한다: `_poll_ready`가 이미
계산해 버려지던 마지막 status/reason을 결과 JSON에 보존하고, readiness가
기대 status에 도달하지 못했을 때만(성공 경로는 건드리지 않음) 컨테이너
자신의 로그 tail을 캡처해 결과 JSON과 hosted job의 콘솔(stderr) 양쪽에
남긴다. **다음 hosted 재실행이 이 새 필드들을 채우면, 가설 A/B 중 어느
것이 사실인지 그 결과 JSON 하나만으로 확정된다** — 재추측이나 재시행 없이.

이 remediation이 건드리지 않은 것: `M4.1_BLOCKED=true`, protected M3 live
`NOT_RUN`, `overall_release_ready=false` 산출 경로, `m3-live-regression-gate`
블록, Native Linux/Ollama/DDGS, self-hosted runner/environment 승인 경계,
`_poll_ready`의 `max_seconds` 값(60초 그대로), readiness 판정 로직
(`evaluate_readiness()`), negative control의 판정 임계값(여전히
`expect_status=503`/`expect_reason="artifact_test_embedding_seam_unavailable"`),
`compute_all_ok`/`_ALL_OK_KEYS`(5개 boolean 그대로), `scan_image_layers.py`,
`assemble_m4_evidence.py`, `check_m4_baseline.py`, `.github/workflows/ci.yml`,
`requirements.lock`/`compile_lock.sh` 중 어느 것도 수정하지 않았다.

## 1. hosted 실패 재확인

`gh api repos/luminous419/simple-qna-rag/actions/jobs/94768636443/logs`
(run 31800982514, `container` job, "Container security/mock smoke" step,
12:39:51.82 시작 ~ 12:41:05.59 최종 JSON 출력, 약 74초):

```json
{
  "embedding_provider": "deterministic_test",
  "graceful_stop_seconds": 1.14,
  "host_gateway_reachable": true,
  "mock_query_ok": false,
  "production_test_seam_sealed": true,
  "readiness_sequence": {"live": true, "ready": false},
  "root_page_ok": true,
  "schema": "m43-container-smoke-v1",
  "static_asset_ok": true,
  "status": "FAIL"
}
```

이는 Hosted_CI_Remediation_Iteration_4.md §2.1이 진단한 것과 **글자
그대로 동일한 지문**이다 — 다른 점은 이번엔 이미 60초 예산 아래서
발생했다는 것뿐이다. `b072fc9`→`c51387a` 사이에 서빙 경로
(`rag_engine.py`/`server.py`/`observability/health.py`/
`index/verification.py`)를 건드린 커밋은 없다(§4에서 diff로 재확인) —
이번 실패도 코드 회귀가 아니다.

## 2. 로컬 재현 시도와 관측된 메커니즘

### 2.1 lifespan이 `/health/live`와 `/health/ready`를 함께 봉쇄한다

`web/server.py`의 `_make_lifespan()`을 읽으면, `app.state.engine =
engine_factory(candidate)`(엔진 초기화 — 설정 바인딩, artifact 해시
검증, FAISS/embeddings canary, `RAGEngine._initialize_llm()`의
`llm.invoke("test")`까지 포함) 호출이 `yield` **이전**, 즉 ASGI lifespan
`startup` 단계 안에서 동기적으로 실행된다. uvicorn은 이 `startup`이
끝나기 전까지 어떤 HTTP 요청도 처리하지 않는다 — health route가 가장
먼저 등록되더라도 예외는 아니다. 즉 `/health/live`와 `/health/ready`는
"느리게 응답"하는 게 아니라 **엔진 초기화가 끝나기 전까지 둘 다 완전히
응답 불가능**하고, 초기화가 끝나는 순간 둘 다 동시에 응답 가능해진다.

이 메커니즘이 정확히 §0의 가설 A/B가 모두 hosted 로그와 들어맞는
이유다: 가설 A(초기화가 60초를 살짝 넘김)면 폴링 루프가 60초 내내
connection-refused만 보다가 포기하고, 그 직후 별도의 단발성 `/health/live`
체크가 마침 그 직후에 막 열린 포트를 잡아 `true`를 반환한다. 가설
B(초기화는 몇 초 안에 끝나지만 `/health/ready`가 반복적으로 같은 사유로
503을 반환)면 앱은 초반부터 살아있고 `/health/live`는 항상 true,
`/health/ready`는 60초 내내 503(다른 status, "실패"가 아니라 "거부")을
반환한다 — 이 경우도 최종 JSON은 동일하게 `live=true, ready=false`로
찍힌다. **기존 스크립트는 이 둘을 구분할 정보를 절대 만들지 않았다.**

### 2.2 로컬 재현(linux/amd64, `c51387a` 이미지, 1 vCPU 스로틀)

```
docker build --platform linux/amd64 --target production \
  -f deploy/Dockerfile -t simple-qna-rag:repro-c51387a .   # 성공
```

`build_docker_run_argv()`가 만드는 것과 동일한 보안 플래그
(`--read-only`, `--tmpfs /tmp:...noexec,nosuid,size=64m`,
`--cap-drop ALL`, `--security-opt no-new-privileges`, `--user
10001:10001`, `--add-host host.docker.internal:host-gateway`)에
`--cpus 1.0`을 추가해 hosted의 공유 vCPU를 흉내 낸 컨테이너를 macOS
Apple Silicon 위 Docker Desktop의 `linux/amd64` 에뮬레이션(Rosetta) 아래서
직접 기동하고, `/health/live`·`/health/ready`를 0.5초 간격으로 폴링하며
`docker logs`를 함께 캡처했다:

```
[5.71s] live=200
[5.71s] ready=200
```

컨테이너 자신의 구조화 로그(`docker logs`):

```json
{"event": "startup", "level": "info", "reason": "ok", ...,
 "timestamp": "2026-08-14T13:03:11.127332+00:00", ...}
```

즉 `live`와 `ready`가 정확히 같은 순간(5.71초)에 함께 true로
전환됐다 — §2.1이 예측한 그대로다. 이는 hosted 프로덕션 이미지·동일한
보안 플래그·동일한 코드에서, 이미 에뮬레이션 오버헤드가 있는 1-vCPU
스로틀 조건에서도 초기화가 60초 예산의 1/10 수준에서 정상 완료됨을
보여준다. `venv/bin/python scripts/container_smoke.py --image
simple-qna-rag:repro-c51387a`(실제 스크립트, 보안 플래그·negative
control 전부 포함) 실행도 `status: PASS`, `readiness_sequence.ready=true`,
새로 추가된 `ready_poll_elapsed_seconds: 6.04`로 재확인했다(§3).

### 2.3 이 재현이 증명하는 것과 증명하지 못하는 것

이 로컬 결과는 가설 A(순수 타이밍 부족)를 약화시킨다 — 에뮬레이션
오버헤드가 있는 1-vCPU 조건에서도 10배 이상의 여유가 있다면, hosted의
실제 네이티브 x86_64 2-vCPU 러너가 60초를 넘기려면 극단적인 노이즈
이웃/스케줄링 변동이 필요하다. 하지만 macOS Docker Desktop의
`linux/amd64` 에뮬레이션 컨테이너는 실제 GitHub Actions `ubuntu-latest`
호스트와 커널·바인드 마운트 권한 처리·네트워크 스택이 다르다 — 예를
들어 `-v index_root:/app/runtime/index:ro` 바인드 마운트가 실제 Linux
호스트에서 UID 10001 기준 권한을 어떻게 평가하는지는 macOS의 파일
공유 계층과 동일하다는 보장이 없다. 즉 이 재현은 가설 A를 **약화**시킬
뿐 가설 B를 **확증**하지는 못한다 — 어느 쪽이 맞는지는 hosted
환경에서만 직접 관측 가능하다. 이것이 정확히 §3의 수정이 존재하는
이유다: 다음 hosted 실패에서 그 관측을 직접 얻는다.

## 3. 수정 (`scripts/container_smoke.py`)

타임아웃 값(`max_seconds=60`)은 전혀 바꾸지 않았다. 대신 세 가지를
추가했다:

1. **포지티브 readiness 폴링의 마지막 status/reason 보존.**
   `ready_ok, _, _ = _poll_ready(...)`를 `ready_ok, ready_last_status,
   ready_last_reason = _poll_ready(...)`로 바꾸고, 폴링에 실제로 소요된
   시간(`ready_poll_elapsed_seconds`, `time.monotonic()` 기준)과 함께
   `result["readiness_sequence"]`에 추가했다:
   ```python
   result["readiness_sequence"] = {
       "live": live_ok,
       "ready": ready_ok,
       "ready_last_http_status": ready_last_status,
       "ready_last_reason": ready_last_reason,
       "ready_poll_elapsed_seconds": ready_poll_elapsed_seconds,
   }
   ```
   `ready_poll_elapsed_seconds`가 60.0에 근접하면 가설 A(정말 예산을
   다 썼다), 60초보다 훨씬 짧으면서 `ready_last_http_status=503`이면
   가설 B(빠르게 반복 거부)로 다음 hosted 실행에서 즉시 판정된다.
   `ready_last_reason`이 확정되면 `observability/health.py`의
   `evaluate_readiness()` 분기 중 정확히 어느 것이 원인인지도 바로
   드러난다(예: `settings_invalid`라면 컨테이너 환경변수 문제,
   `artifact_*`라면 index/embedding 신뢰 경계 문제, `engine_init_failed`면
   `llm.invoke("test")`를 포함한 다른 예외).

2. **readiness 실패 시에만 컨테이너 로그 캡처.** 새 함수
   `_capture_container_logs(container_id, max_bytes=16000)`가
   `docker logs --tail 200 <container_id>`의 stdout+stderr를 합쳐 마지막
   `max_bytes`바이트만(앞이 아니라 **뒤** — 가장 최근 이벤트인 `startup`
   로그와 최종 거부 응답이 여기 있다) 잘라 반환한다. `docker` 서브프로세스
   자체가 실패해도 예외를 삼키고 빈 문자열을 반환한다(캡처 실패가 판정
   결과 생성 자체를 깨서는 안 된다). `ready_ok`가 False일 때만
   `result["container_log_tail"]`에 채운다 — 통과하는 모든 정상 실행의
   증거 아티팩트를 불필요한 로그 덤프로 오염시키지 않는다.
   negative-control 봉인도 대칭적으로 처리했다: `ok, status, reason =
   _poll_ready(...)`가 버리던 `status`/`reason`을
   `production_test_seam_seal_last_http_status`/
   `_last_reason`으로 보존하고, 봉인 실패 시에만
   `result["negative_control_log_tail"]`을 채운다.

3. **hosted job 콘솔에 즉시 노출.** `main()`이 `container_log_tail`/
   `negative_control_log_tail`이 존재하면 stderr로 바로 출력한다 — 업로드된
   `container_smoke.json` 아티팩트를 별도로 내려받지 않아도 hosted job의
   step 로그 화면에서 곧바로 원인 진단이 보인다.

이 세 가지 모두 **판정 로직을 하나도 바꾸지 않는다**: `compute_all_ok()`가
읽는 `_ALL_OK_KEYS` 5개 boolean, `expect_status`/`expect_reason` 임계값,
`max_seconds=60` 전부 그대로다. `assemble_m4_evidence.py`의
`REQUIRED_PAYLOADS["container"]["container_smoke.json"]` 검증은 5개
boolean 필드가 정확한 값인지만 `payload_doc.get(f)`로 확인하고
`container_smoke.json`의 전체 키 집합을 고정(frozen)하지 않으므로(코드
확인, `RECEIPT_TOP_KEYS`와 달리 이 페이로드에는 별도의 exact-key-set
가드가 없다), 새로 추가된 키들은 M4 evidence 조립·baseline 판정 경로에
영향을 주지 않는다.

## 4. 재검증 결과

- `git diff --stat`(이 remediation의 전체 변경): `scripts/container_smoke.py`
  (수정, 순수 추가 — 기존 판정 로직 라인은 옮기지 않음)와
  `tests/unit/test_container_smoke_readiness_diagnostics.py`(신규, 8개
  테스트) 두 파일뿐.
- `bash -n`은 Python 스크립트에 적용되지 않으므로 대신
  `python -c "import ast; ast.parse(...)"`로 구문 확인: **PASS**.
- `docker build --platform linux/amd64 --target production -f
  deploy/Dockerfile -t simple-qna-rag:repro-c51387a .`(`c51387a` 기준):
  **성공**.
- `venv/bin/python scripts/container_smoke.py --image
  simple-qna-rag:repro-c51387a --output .../smoke_result.json`(실제
  docker, 보안 플래그·negative control 전부 포함, macOS
  `linux/amd64` 에뮬레이션): **`status: PASS`**, 6개 필드 전부 true, 새
  필드 확인:
  ```json
  "readiness_sequence": {
    "live": true, "ready": true,
    "ready_last_http_status": 200, "ready_last_reason": "ok",
    "ready_poll_elapsed_seconds": 6.04
  },
  "production_test_seam_seal_last_http_status": 503,
  "production_test_seam_seal_last_reason": "artifact_test_embedding_seam_unavailable"
  ```
  성공 경로에서 `container_log_tail`/`negative_control_log_tail`이
  결과에 **없음**을 확인 — §3의 "실패 시에만 캡처" 계약이 정상 경로에서
  실제로 지켜진다.
- `venv/bin/python -m pytest -q
  tests/unit/test_container_smoke_readiness_diagnostics.py`(신규,
  docker 불필요 — `_run`/`_poll_ready`/`_build_fixture_index`를
  monkeypatch로 대체하고 나머지는 unbound loopback 포트에 대한 실제
  fail-fast 예외 경로를 그대로 통과시킨다): **8 passed**. 커버리지:
  포지티브 폴링이 200에 도달하지 못했을 때 `ready_last_http_status`/
  `ready_last_reason`/`ready_poll_elapsed_seconds`가 정확히 전파되는지,
  `container_log_tail`이 실패 시에만/성공 시엔 부재하는지, negative
  control의 status/reason이 대칭적으로 전파되는지, 봉인 실패/유지 각각에서
  `negative_control_log_tail`의 존재/부재, `_capture_container_logs`가
  stdout+stderr를 합치는지·앞이 아니라 뒤를 자르는지·`docker` 실패 시
  예외 없이 빈 문자열을 반환하는지.
- `venv/bin/python -m pytest -q tests/unit/test_container_smoke_contract.py
  tests/unit/test_container_smoke_bare_script.py`(기존 회귀 오라클): **17
  passed** — 회귀 없음.
- `venv/bin/python -m pytest -q`(전체 unit+integration, macOS 로컬):
  **1306 passed, 1 skipped**(Hosted_CI_Remediation_Iteration_4.md 기준
  1298 passed + 이번 신규 8건 = 1306, 회귀 없음).
- `python scripts/generate_field_spec.py --check`,
  `python scripts/logging_callsite_audit.py --check`: 둘 다 **exit 0**.
- `python scripts/check_markdown_links.py`: 검사 파일 135개(이 문서
  포함), 링크 595개, 실패 0개.
- `git diff --exit-code -- .github/workflows/ci.yml
  scripts/scan_image_layers.py scripts/assemble_m4_evidence.py
  scripts/check_m4_baseline.py evaluation/baselines/m3_initial.*`
  (protected 경계): **exit 0**.
- `requirements.lock`/`scripts/compile_lock.sh`: 변경 없음(이번
  remediation은 §1.의 lock/헤더 계약을 전혀 건드리지 않았다).
- Native Linux/Ollama/DDGS, protected M3/M4.1 live gate, self-hosted
  runner/environment 승인 경계는 이 remediation에서도 실행·변경하지
  않았다. `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`,
  `overall_release_ready=false` 산출 경로는 변경된 파일에 포함되지
  않는다(변경 파일은 `scripts/container_smoke.py`와 신규 테스트 파일
  1개, 그리고 이 문서뿐).

## 5. 이 remediation이 건드리지 않은 것

`_poll_ready`의 `max_seconds` 값(60초 그대로, 다시 늘리지 않았다),
`evaluate_readiness()` 판정 로직, negative control의 판정 임계값
(`expect_status=503`/`expect_reason="artifact_test_embedding_seam_unavailable"`),
`compute_all_ok`/`_ALL_OK_KEYS`, `M4.1_BLOCKED=true`, protected M3 live
`NOT_RUN`, `overall_release_ready=false` 산출 경로, `m3-live-regression-gate`
블록, `scan_image_layers.py`/certifi Label exact-binding 로직, index
lifecycle/`assemble_m4_evidence.py`/`check_m4_baseline.py`,
`requirements.lock`/`compile_lock.sh`, Native Linux/Ollama/DDGS,
self-hosted runner/environment 승인 경계, 위 결함과 무관한 어떤 파일도
수정하지 않았다.

## 6. 남은 hosted 검증 필요

이 커밋은 기존 PR #18에 push될 예정이다. **merge하지 않는다** — fresh
Codex 리뷰가 필요하다. §2.3에서 명시했듯 로컬 재현은 가설 A(순수 타이밍
부족)를 약화시키는 정황 증거이지 hosted 환경에서의 확정 증거가 아니다
— macOS Docker Desktop의 `linux/amd64` 에뮬레이션은 실제 GitHub
Actions `ubuntu-latest` 러너의 커널/바인드 마운트 권한 처리/네트워크
스택과 다르다. **이 push 이후 hosted `container` job이 다시 실패하면,
이번에 추가된 `readiness_sequence.ready_last_http_status`/
`ready_last_reason`/`ready_poll_elapsed_seconds`와 (실패 시에만 채워지는)
`container_log_tail`을 업로드된 `container_smoke.json` 아티팩트 또는
job 콘솔(stderr)에서 직접 읽으면 가설 A/B 중 어느 쪽이 실제 원인인지
그 자리에서 확정된다** — 그 결과에 따라 다음 iteration의 수정 범위가
결정된다(가설 B로 확정되면 원인이 되는 정확한 `reason`을 대상으로 한
좁은 수정, 가설 A로 확정되면 `ready_poll_elapsed_seconds`가 60초에 얼마나
근접했는지를 근거로 한 데이터 기반 예산 조정 — 이번처럼 근거 없이
숫자만 올리는 것이 아니라). `python-tests`/`container`/`m4-assemble`(하류
종속) 세 job 모두 이 push 이후 재확인이 필요하다.
