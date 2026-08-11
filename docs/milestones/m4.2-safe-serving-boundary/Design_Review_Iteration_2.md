# M4.2 Safe Serving Boundary — Design Review Iteration 2

검토 대상: [Requirement.md](Requirement.md), [Plan.md](Plan.md), [Design.md](Design.md), [Design Review Iteration 1](Design_Review_Iteration_1.md), [개발 orchestration guide](../../../milestone_dev_orchestration_guide.md), 기준 revision `0c84795`의 repository code/tests/config.

## Executive Summary

**FAIL — 7.4/10.0.** Iteration 2는 caller outcome/resource completion 분리, loop-close resource 회계, submit rollback, capacity-edge debounce, per-call Ollama client, metric 보존식의 핵심 방향을 고쳐 M42-DR1-001~006의 직접 결함은 대부분 닫았다. 그러나 `create_app()` 선로딩이 기존 fail-soft startup/readiness 계약을 깨고, ASGI disconnect를 handler cancellation로 간주해 running caller cancellation이 실제로 연결되지 않으며, compressed payload와 deterministic drain/acceptance profile에도 실행 가능한 폐쇄 계약이 없다. 따라서 Phase 2 구현으로 진행하면 안 된다.

| Severity | 수 |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 5 |
| MINOR | 0 |

가이드 Gate는 score >= 9.7/10, CRITICAL=0, MAJOR=0, MINOR 최소화를 모두 요구한다. 현재는 MAJOR 5건과 requirement-to-design 누락이 남아 **Gate FAIL**이다.

## Prior-finding closure matrix

| Finding | Iteration 2 판정 | 검증 근거 |
|---|---|---|
| M42-DR1-001 | CLOSED | `Design.md:70-91`, `201-278`: `caller_finalized`와 `resource_finalized`를 분리하고 ABANDONED에서도 future 완료 전까지 running slot을 유지한다. 양 race order에서 resource completion만 running/orphan을 감소시킨다. |
| M42-DR1-002 | CLOSED | `Design.md:293-332`: asyncio-free resource completion을 notification과 분리하고 loop-close × abandon 4조합을 고정했다. |
| M42-DR1-003 | CLOSED | `Design.md:123-180`: `pool.submit()` Future 반환을 commit point로 두고 direct/promotion submit 실패 rollback 및 negative symbol을 정의했다. |
| M42-DR1-004 | CLOSED | `Design.md:96-112`, `729-756`: 실제 capacity edge의 version/timestamp를 executor lock에서 기록해 probe 사이 clear/re-full을 보존한다. |
| M42-DR1-005 | CLOSED | `Design.md:610-646`: 설치 API 제약에 맞춰 각 router/answer 호출마다 remaining budget의 비공유 `ollama.Client`를 만들고 transport를 닫는다. 고정 singleton timeout 문제는 제거됐다. |
| M42-DR1-006 | CLOSED | `Design.md:91-94`, `772-817`: accepted와 terminal outcomes를 분리한 보존식, rejection taxonomy, scrape 직전 atomic snapshot gauge sync, metrics side-effect 격리를 정의했다. |
| M42-DR1-007 | OPEN (MAJOR) | `Design.md:819-853`에 11행 표는 생겼으나 일부 profile이 제시 fixture로 실행 불가능하거나 Requirement의 측정 계약과 모순된다. M42-DR2-005 참조. |
| M42-DR1-008 | REGRESSED (MAJOR) | `Design.md:520-545`의 단일 로드는 이중 loader를 없앴지만 app factory 자체를 실패시키는 방식으로 M4.1 fail-soft import/startup readiness 및 REQ-005 STARTING 실패 상태를 파괴한다. M42-DR2-001 참조. |

## New and remaining findings

### M42-DR2-001 — MAJOR — 단일 settings load 수정이 startup failure/readiness 호환성을 파괴한다

- 위치: `Design.md:520-547`; 현재 `src/simple_qna_rag/web/server.py:1-10`, `63-98`, `166-183`; REQ-005.1, REQ-009.2.
- 증거: 현재 서버는 module-level `app = create_app()`가 Settings 오류와 무관하게 import되고, lifespan이 `SettingsError`를 잡아 `settings_invalid` readiness 503으로 표현한다. 반면 새 설계는 `create_app()` 시작 전에 `settings_loader()`를 호출하고 오류를 전파한다. 그러면 module import/app construction이 실패하여 lifecycle이 `STARTING`에 머물 수도, `/health/live`와 `/health/ready`가 응답할 수도 없다. `start_server()`의 CLI exit 2와 ASGI module import의 fail-soft health surface도 구분되지 않는다.
- 영향: M42-DR1-008을 다른 요구사항 위반으로 치환했으며 기존 M4.1 health/bootstrap 호환과 lifecycle state machine을 깨뜨린다.
- exact required fix: app factory가 bootstrap/health routes를 항상 만들 수 있는 계약을 유지하면서 validated Settings를 **한 번만 시도/저장**하는 owner를 설계한다. 예를 들어 lifespan의 단일 load 결과를 app state에 저장하고 body limiter가 startup 전에는 별도의 불변 hard ceiling만 적용한 뒤 loaded setting과 일치하는 fail-closed 정책을 사용하도록 명시하라. module import, CLI override, invalid settings 각각의 expected exit/readiness를 별도 상태표와 `test_module_app_import_survives_invalid_settings`, `test_single_loader_result_shared_after_startup`에 고정하라.

### M42-DR2-002 — MAJOR — HTTP disconnect가 `CancelledError`를 만든다는 가정으로 caller cancellation이 배선되지 않는다

- 위치: `Design.md:485-499`; 현재 `src/simple_qna_rag/web/server.py:138-155`; REQ-003.3/003.4, caller-cancellation profile.
- 증거: 설계의 유일한 연결은 `await ticket.result()`의 `except asyncio.CancelledError`다. ASGI의 연결 종료는 `http.disconnect` receive message이며 일반적인 FastAPI/Starlette handler task가 자동 취소된다는 계약이 아니다. handler는 body를 이미 읽은 뒤 executor Future만 await하므로 disconnect를 더 이상 receive하지 않는다. 따라서 실제 client disconnect에서 queued ticket이 제거되지 않고 running ticket도 ABANDONED/orphan으로 표시되지 않을 수 있다. 제시된 executor 직접 cancel race 테스트는 HTTP 배선 결함을 검출하지 못한다.
- exact required fix: ASGI disconnect를 명시적으로 관찰하는 request-scoped task/receive-owner 또는 검증된 server cancellation hook을 symbol/signature와 cleanup ownership까지 설계하라. result completion과 disconnect wait를 race시키되 body receive의 단일 소비자 규칙을 지키고, loser task 취소/회수 및 response-send 금지를 정의하라. 실제 ASGI `http.disconnect`를 주입하는 queued/running 각 100회 integration race를 11-profile runner에 연결하라.

### M42-DR2-003 — MAJOR — body limiter가 “decompressed body” 한도를 보장하지 않는다

- 위치: `Design.md:549-586`; REQ-007.1.
- 증거: `limited_receive()`는 ASGI `http.request.body`의 길이만 합산한다. 설계에는 `Content-Encoding` 처리, 압축 해제 owner, 압축 형식 거부 정책이 없다. 서버가 compressed bytes를 그대로 전달하면 작은 gzip payload가 한도를 통과한 뒤 애플리케이션/상위 계층에서 큰 JSON으로 확장될 수 있고, 서버가 미리 decompress한다면 그 사실을 보장하는 uvicorn/ASGI 계약 증거가 없다. 현재 payload profile도 chunking/Content-Length만 다루고 compressed expansion negative control이 없다.
- exact required fix: `/rag`에서 non-identity `Content-Encoding`을 admission 전에 고정 `invalid_request` 또는 413으로 거부하거나, bounded streaming decompression을 명시적으로 소유해 decompressed 누적치를 `limit+1`에서 중단하라. 선택한 정책을 gzip bomb/false Content-Length/no-length fixture와 receive-byte/decompressed-byte assertions에 추가하라.

### M42-DR2-004 — MAJOR — `wait_drained()`가 fake clock과 wall-clock polling을 혼합해 deterministic/bounded shutdown을 증명하지 못한다

- 위치: `Design.md:347-378`, `Design.md:838`; Requirement §5의 “모든 concurrency 테스트는 Event/barrier와 fake monotonic clock, wall-clock sleep 기반 순서 판정 금지”.
- 증거: `wait_drained()`의 deadline은 injected `_clock`이지만 대기는 `asyncio.sleep(0.05)` wall time이다. frozen FakeClock에서는 grace가 영원히 만료되지 않고, test가 clock을 외부에서 전진시키면 polling wake 순서와 최대 50ms 지연에 의존한다. `SHUTDOWN_GRACE_SECONDS=0` 외에는 bounded-return proof가 scheduler/poll cadence에 섞이며 running completion도 명시적인 wake primitive가 없다.
- exact required fix: running==0 resource transition이 알리는 loop-safe drain event/condition과 injectable deadline scheduler를 설계해 polling sleep을 제거하라. grace expiry와 resource completion의 양 race order, zero grace, loop-close 이후 callback을 fake clock advance + Event/barrier만으로 구동하고 `shutdown(wait=False, cancel_futures=True)` 호출 및 residual snapshot을 정확히 한 번 assert하라.

### M42-DR2-005 — MAJOR — 11개 정량 profile 표가 생겼지만 1:1 executable traceability는 아직 닫히지 않았다

- 위치: `Design.md:819-853`; Requirement §5/§6; M42-DR1-007.
- 증거 1: queue-timeout 행은 running 1 + queued A/B에서 A timer만 발화시키면서 B가 즉시 promoted되고 `submit_delta==1`이라고 요구한다. running slot release event가 fixture에 없어 그 상태에서는 B promotion이 불가능하다.
- 증거 2: normal mock load는 “200ms virtual work ×40”과 `expected=8.000s`, `makespan<=9.600s`를 쓰지만 virtual work/clock을 ThreadPoolExecutor 실행 및 wall-time acceptance와 연결하는 scheduler symbol이 없다. clock만 200ms씩 전진하면 실제 wall-time 목표를 검증하지 않고, 실제 200ms 대기는 금지된 sleep 기반 순서/시간 판정이 된다.
- 증거 3: caller-cancellation 행은 executor guard counters만 측정하고 M42-DR2-002의 실제 ASGI disconnect를 실행하지 않는다. opt-in live 행도 “exactly 12 IDs”의 manifest 경로/schema와 M3 14-gate invocation/exit conservation이 고정되지 않았다.
- exact required fix: 각 모순을 제거한 fixture event sequence를 표에 적고 실제 runner가 호출할 importable symbol/signature를 고정하라. queue-timeout은 A 만료 후 running release를 명시해 그 resource completion에서 B 단일 승격을 검증하거나 PASS assertion을 slot 상태에 맞게 수정하라. mock load는 wall time을 재는 결정론적 Event-driven 200ms service simulator(순서 판정 sleep 없음) 또는 명시적 virtual scheduler와 그에 맞는 virtual-time 기준으로 계약을 일치시켜라. cancellation은 ASGI disconnect integration을, live는 12-case manifest와 M3 runner command/artifact schema를 명시하라. repeat-10 runner가 앞 10개 profile 전부의 pytest node ID를 실제 수집하고 conservation 실패 시 nonzero exit하는 negative control도 추가하라.

## Requirement traceability verdict

| REQ | 판정 | 근거 |
|---|---|---|
| 001 | 부분 충족 | 8개 필드/범위/cross-field는 구체적이나 single-load lifecycle 해법이 REQ-005/009와 충돌한다. |
| 002 | 충족 | bounded admission, FIFO, commit/rollback, resource slot ownership이 폐쇄됐다. |
| 003 | 실패 | executor 내부 race는 개선됐으나 실제 HTTP disconnect 전달 경로가 없다. |
| 004 | 충족 | orphan 불변식, 안전 오류 body, counter 분류가 구체적이다. |
| 005 | 실패 | invalid startup state/readiness와 deterministic drain wait가 폐쇄되지 않았다. |
| 006 | 충족 | edge debounce, bounded labels, scrape-time gauges와 cardinality 상한이 정의됐다. |
| 007 | 실패 | compressed/decompressed payload 경계가 없다. |
| 008 | 충족(설계 수준) | Ollama remaining-budget clients와 DDGS/orphan stall 경계가 설치 API에 맞게 정의됐다. 구현 시 pinned private close seam 검증은 필수다. |
| 009 | 실패 | M4.1 fail-soft app import/readiness 호환을 깨뜨린다. |
| 정량 Gate | 실패 | 11행은 존재하지만 queue-timeout, cancellation, mock load, live separation이 아직 1:1 실행 가능하지 않다. |

`M4.1_BLOCKED`는 계속 독립 위험으로 유지됐고 M4.3/M5 범위 침범은 발견하지 않았다.

## Verification evidence

- Requirement, Plan, Design, Iteration 1 review, orchestration guide의 Gate/iteration 규칙을 전부 대조했다.
- 현재 `web/server.py`의 fail-soft module app/lifespan, health routes, request handler와 metrics registry wiring을 설계의 변경안과 비교했다.
- 현재 `agent.py`, `rag_engine.py`, `web_search.py`, settings/config facade와 관련 unit/integration test inventory를 확인해 기존 singleton/network/API 호환 경계를 재검증했다.
- 상태 머신은 queued/running terminal, abandon-first/completion-first, loop alive/closed, submit commit/failure, drain 상태를 순서별로 재연산했다.
- Requirement §5의 11개 profile, deterministic repeat-10, executor/request conservation, opt-in live/M3 분리를 `Design.md` 표와 행별 대조했다.

## Re-review entry conditions

1. settings single-load와 M4.1 fail-soft startup/readiness를 동시에 만족하는 lifecycle을 다시 고정한다.
2. 실제 ASGI disconnect owner와 compressed payload 정책을 integration-test 가능한 symbol로 추가한다.
3. polling 없는 deterministic drain deadline/wake 계약을 정의한다.
4. M42-DR1-007의 네 profile 모순과 live manifest/runner 누락을 닫고 repeat/conservation negative control을 완성한다.
5. 변경된 Design에 markdown link validation과 `git diff --check`를 통과시킨 뒤 fresh Iteration 3 리뷰를 수행한다.
