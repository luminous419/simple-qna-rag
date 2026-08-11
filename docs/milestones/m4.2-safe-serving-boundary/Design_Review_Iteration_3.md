# M4.2 Safe Serving Boundary — Design Review Iteration 3

검토 대상: [Requirement.md](Requirement.md), [Plan.md](Plan.md), [Design.md](Design.md),
[Design Review Iteration 1](Design_Review_Iteration_1.md),
[Design Review Iteration 2](Design_Review_Iteration_2.md),
[개발 orchestration guide](../../../milestone_dev_orchestration_guide.md), 기준 revision
`0c84795`의 repository code/tests/config.

## Executive Summary

**FAIL — 7.8/10.0.** Iteration 3 설계는 Iteration 2의 다섯 문제에 각각 대응했고,
compressed encoding의 fail-closed 거부, 실제 `http.disconnect` 관찰 방향, event 기반 drain,
11행 acceptance 표까지 설계 의도는 크게 개선됐다. 그러나 현재 settings cache seam을 고려하면
lifespan의 loader 뒤 engine import가 두 번째 `Settings` 검증을 일으키며, `/rag` 전용 ASGI
adapter는 inventory 외에 실행 가능한 send/receive ownership 계약이 없고, body limiter는 큰
ASGI chunk를 `limit+1`보다 많이 소비한다. drain tie ownership과 exact pytest node mapping도
서로 모순되어, 요청된 fail-soft 단일 attempt·실제 disconnect race·payload byte boundary·
polling-free deterministic drain·11-profile 1:1 traceability를 아직 증명하지 못한다.

| Severity | 수 |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 5 |
| MINOR | 0 |

가이드 Gate인 score >= 9.7/10, CRITICAL=0, MAJOR=0, MINOR 최소화를 만족하지 못하므로
**Gate FAIL**이며 Phase 2 구현으로 진행하면 안 된다.

## Prior-finding closure matrix

| Finding | Iteration 3 판정 | 검증 근거 |
|---|---|---|
| M42-DR1-001 | CLOSED | `Design.md:69-94`, `230-281`: caller/resource guard를 분리하고 ABANDONED slot을 future 완료까지 유지한다. |
| M42-DR1-002 | CLOSED | `Design.md:296-335`: asyncio-free resource completion과 loop notification을 분리한다. |
| M42-DR1-003 | CLOSED | `Design.md:123-180`: `pool.submit()` Future 반환 commit과 direct/promotion rollback을 고정한다. |
| M42-DR1-004 | CLOSED | `Design.md:96-112`, `799-818`: capacity edge timestamp/version이 probe 사이 전이를 보존한다. |
| M42-DR1-005 | CLOSED | `Design.md:682-716`: 실제 Ollama 0.6.0 API에 맞춘 per-call remaining-budget client와 close seam을 정의한다. |
| M42-DR1-006 | CLOSED | `Design.md:887-891`: accepted/terminal 보존식, scrape-time gauge sync, metric side-effect 격리를 정의한다. |
| M42-DR1-007 | OPEN (MAJOR) | 표와 runner API는 추가됐으나 exact node ID가 본문과 불일치하고 drain tie가 executable하지 않다. M42-DR3-004/005 참조. |
| M42-DR1-008 | OPEN (MAJOR) | fail-soft app construction은 복구됐지만 process cache를 채우지 않아 engine import에서 실제 settings validation이 재시도된다. M42-DR3-001 참조. |
| M42-DR2-001 | OPEN (MAJOR) | startup 상태표의 “정확히 1회/동일 identity” 주장이 현재 `get_settings()` cache seam과 연결되지 않는다. M42-DR3-001 참조. |
| M42-DR2-002 | OPEN (MAJOR) | observer/race helper는 생겼지만 `RagASGIRoute`의 실제 handler 호출 및 conditional send ownership이 정의되지 않았다. M42-DR3-002 참조. |
| M42-DR2-003 | PARTIAL (MAJOR) | non-identity encoding은 receive 전 400으로 닫혔으나 identity chunk가 한 번에 limit를 크게 넘으면 `limit+1` 수신 상한을 깨뜨린다. M42-DR3-003 참조. |
| M42-DR2-004 | PARTIAL (MAJOR) | polling은 제거됐지만 simultaneous completion/deadline winner와 stale re-arm API가 폐쇄되지 않았다. M42-DR3-004 참조. |
| M42-DR2-005 | OPEN (MAJOR) | 11행 표는 구체화됐지만 disconnect exact symbol 불일치와 drain profile 모순 때문에 collect-and-run 1:1 계약이 아니다. M42-DR3-005 참조. |

## Findings

### M42-DR3-001 — MAJOR — fail-soft lifespan은 `Settings` 단일 validation attempt를 실제 코드 seam에서 보장하지 않는다

- 위치: `Design.md:566-614`; 현재 `src/simple_qna_rag/web/server.py:57-67`,
  `src/simple_qna_rag/settings.py:674-695`, `src/simple_qna_rag/config.py:88`;
  M4.2-REQ-001.2, REQ-005.1, REQ-009.2.
- exact evidence: 설계의 lifespan은 `settings_loader()` 결과를 `app.state.settings`에만 저장하고
  곧 `engine_factory(settings)`를 호출한다. 현재 default engine factory는 그 시점에
  `rag_engine.py`를 import하며, transitive `config.py`는 module import에서 `get_settings()`를
  호출한다. `Settings.from_sources()`는 `_settings_cache`를 채우지 않고, cache를 채우는 공개
  seam은 `set_settings_for_process(settings)`뿐이다. 따라서 기본 loader가
  `Settings.from_sources()`인 설계대로라면 lifespan 1회 + config import 1회, 총 두 번
  validation할 수 있고 두 번째 env 관찰이 실패하거나 다른 identity를 만들 수 있다.
- 재현 상태: loader spy가 validated `s1` 반환 → app state는 `s1` → engine import → cache가
  `None`이므로 `get_settings()`가 env를 다시 읽어 `s2` 생성/실패 → 설계 상태표의 “총 validation
  1회”와 “limiter/executor/engine 동일 Settings identity”가 동시에 깨진다.
- exact required fix: lifespan의 successful load commit에서 기존
  `set_settings_for_process(s)`를 호출하는지, 또는 config/rag-engine import가 process cache를
  전혀 통과하지 않도록 하는지 하나를 명시하고 rollback ownership까지 고정하라. module import
  0회, valid/invalid lifespan 1회, CLI preflight closure 1회 각각에 대해
  `Settings.from_sources`와 `get_settings` 양쪽 spy count 및 engine이 본 object identity를
  assert하는 executable test를 추가하라.

### M42-DR3-002 — MAJOR — 실제 ASGI disconnect의 receive/send owner가 `RagASGIRoute` 수준에서 미정이다

- 위치: `Design.md:14`, `Design.md:33`, `Design.md:470-545`; 현재
  `src/simple_qna_rag/web/server.py:132-155`; M4.2-REQ-003.3/003.4.
- exact evidence: 설계는 `rag_query()`가 `request.json()`으로 body를 읽고
  `race_result_or_disconnect(ticket, receive=request.receive, ...)`를 호출한다고 쓰는 동시에,
  disconnect winner에서 `_NO_RESPONSE`를 반환한다고 쓴다. FastAPI handler의 반환값은
  Starlette routing layer가 Response로 직렬화하고 send하므로 handler sentinel만으로 send 0을
  보장할 수 없다. 이를 해결한다는 `RagASGIRoute`는 symbol inventory에만 있고 constructor,
  `get_route_handler()`/ASGI call signature, handler result 전달, sentinel intercept, outer
  cancellation/finally ownership이 전혀 정의되지 않았다.
- 재현 상태: 실제 `http.request(more_body=False)` 뒤 `http.disconnect` → observer가 이김 →
  handler가 sentinel 반환 → 기본 FastAPI route는 response validation 또는 response start를
  수행한다. 반대로 custom route가 raw receive를 먼저 소유하면 `Request` body parser와 동시
  receive 경쟁을 피하는 인계 규칙이 없다. helper 단위 pseudocode만으로는
  `concurrent receive<=1`, disconnect winner send=0을 증명하지 못한다.
- exact required fix: `/rag`에 실제 설치되는 route class/ASGI wrapper의 executable signature와
  호출 순서를 제시하라. body owner 종료 → disconnect owner 시작의 handoff, result/disconnect
  tie linearization, sentinel intercept 전 response 생성 금지, 양 loser cancel+await, outer
  cancellation cleanup을 한 control flow에 고정하고 실제 app을 호출하는 ASGI trace에서 queued와
  running 각각 100회 양 order/tie, response frames 0/1, receive max 1, pending task 0을 assert하라.

### M42-DR3-003 — MAJOR — identity payload limiter가 `MAX_REQUEST_BODY_BYTES+1` 수신 상한을 보장하지 않는다

- 위치: `Design.md:616-656`; M4.2-REQ-007.1 및 payload 정량 Gate.
- exact evidence: `limited_receive()`는 upstream message 전체를 먼저 `await receive()`한 뒤
  `len(message["body"])`를 더한다. ASGI는 한 `http.request` chunk 크기를
  `remaining_limit+1`로 제한하지 않는다. 예를 들어 limit=16,384이고 첫 chunk가 1 MiB이면
  middleware는 1 MiB를 이미 수신한 뒤 413을 내므로 설계가 명시한 `received <= limit+1`과
  요구사항의 “limit+1에서 더 읽지 않음”을 깨뜨린다.
- compressed boundary 판정: non-identity `Content-Encoding`을 body receive/decompression 전에
  400으로 거부하는 정책 자체는 gzip bomb/false-length/no-length 우회를 fail-closed로 닫는다.
  남은 결함은 identity representation에서도 large-chunk over-read가 가능한 점이다.
- exact required fix: ASGI server가 전달한 단일 message 크기와 application-level consumed byte를
  구분해 요구사항을 구현 가능한 계약으로 정리하라. strict `limit+1` 수신 상한이 필수라면
  그보다 큰 ASGI message를 애플리케이션이 이미 수신한다는 경계 때문에 server/proxy receive
  cap 또는 그에 상응하는 owner가 필요하다. 최소한 one-chunk `limit+N`, multi-chunk boundary,
  false/no Content-Length를 포함해 wire-received/application-consumed 값을 별도 기록하고 assertion과
  error contract를 일치시켜야 한다.

### M42-DR3-004 — MAJOR — polling-free drain의 simultaneous winner와 re-arm 계약이 결정론적이지 않다

- 위치: `Design.md:350-396`; M4.2-REQ-005.3/005.4 및 drain 정량 Gate.
- exact evidence 1: `wait_drained()`은 `asyncio.wait(FIRST_COMPLETED)` 뒤 `if waiter in done`을
  먼저 검사한다. 같은 loop turn에 resource-zero와 deadline future가 모두 done이면 실제
  scheduler sequence와 무관하게 resource가 승자가 된다. 그런데 설계는 completion-first/
  deadline-first/tie를 sequence로 고정해 “먼저 linearize된 쪽만 결과를 소유”한다고 주장한다.
- exact evidence 2: stale event branch는 `deadline_handle.cancel()`을 먼저 호출한 뒤
  `deadline_handle.remaining()`으로 재귀 timeout을 계산하지만 `DeadlineScheduler`가 반환하는
  `Cancellable` 공개 계약에는 `remaining()`이 없다. cancel 후 remaining 의미도 정의되지 않아
  해당 branch를 구현/테스트할 수 없다.
- exact required fix: resource completion과 deadline callback이 하나의 lock/monotonic sequence
  아래 단일 winner field를 CAS하도록 하거나, done set 양쪽을 명시적 linearization token으로
  판정하라. scheduler handle protocol에 필요한 deadline/remaining API와 cancel 이후 의미를
  선언하거나 absolute deadline을 waiter가 직접 보존하라. completion-first, deadline-first,
  exact tie, stale event, timeout=0 각각에서 result, loser reap, shutdown call 1회, residual을
  `ManualDeadlineScheduler`만으로 검증하라.

### M42-DR3-005 — MAJOR — 11-profile 표와 runner의 exact node collection이 자체 불일치한다

- 위치: `Design.md:524-545`, `Design.md:893-951`; M4.2-REQ-009.3, Requirement §5/§6.
- exact evidence: §4.1은 실제 ASGI race tests를
  `test_asgi_disconnect_queued_100_races`와
  `test_asgi_disconnect_running_100_races` 두 node로 고정하지만 §10의 유일한
  `PROFILE_NODE_IDS["caller_cancellation"]` 행은 존재가 보장되지 않은 단일 node
  `test_asgi_disconnect_queued_and_running_100_races`를 지정한다. runner는 exact node를
  `pytest --collect-only`로 검증하고 누락/추가/중복이면 exit 1이라고 하므로 두 계약은 동시에
  구현될 수 없다. drain profile 또한 M42-DR3-004의 tie winner를 report한다고 하지만 실행
  알고리즘은 그 winner를 보존하지 않는다.
- 이미 닫힌 부분: queue-timeout은 running R release 뒤 B 승격으로 수정됐고, normal mock load는
  event-driven virtual 8.000s로 일치하며, live manifest 12건과 별도 M3 command/artifact,
  deterministic repeat=10, conservation negative exit는 명시됐다.
- exact required fix: 각 profile의 `PROFILE_NODE_IDS` tuple과 본문 test catalog를 byte-for-byte
  동일하게 만들고, caller cancellation을 두 nodes로 둘지 한 parameterized node로 둘지 하나만
  선택하라. `test_profile_node_inventory_exact`가 10 deterministic profiles의 ordered mapping,
  실제 collection, 중복 0을 검증하게 하고, repeat 10 receipt에는 매 profile node와 conservation
  결과가 정확히 10개씩 존재함을 assert하라. live 12/M3 artifact/M4.1_BLOCKED는 deterministic
  aggregate와 별도 top-level status로 유지하라.

## Requirement traceability verdict

| REQ | 설계 연결 | 판정 |
|---|---|---|
| 001 | §7, §4.3 | 실패 — field/validator는 구체적이나 default engine 경로에서 settings validation이 재시도될 수 있다(M42-DR3-001). |
| 002 | §2.1~2.8 | 충족 — bounded atomic admission, explicit FIFO, submit commit/rollback, future-owned slot 회계가 구체적이다. |
| 003 | §2.5~2.7, §4.1 | 실패 — executor race는 닫혔으나 실제 ASGI route의 disconnect/send owner가 미정이다(M42-DR3-002). |
| 004 | §2.2~2.10, §3, §9 | 충족 — orphan 불변식, terminal/resource 분리, fixed safe error, bounded counters가 정의됐다. |
| 005 | §2.9, §4.3 | 실패 — lifecycle 방향은 맞지만 drain tie/re-arm이 deterministic/executable하지 않다(M42-DR3-004). |
| 006 | §8~9 | 충족 — readiness precedence, edge debounce, bounded series와 scrape-time gauges가 폐쇄됐다. |
| 007 | §4.2/4.4 | 실패 — compressed encoding 거부는 닫혔지만 large identity chunk가 limit+1 수신 상한을 깨뜨린다(M42-DR3-003). |
| 008 | §5~6 | 충족(설계 수준) — context deadline, per-call Ollama client, DDGS remaining timeout과 stall fake가 설치 API에 맞다. |
| 009 | §4/8/9/10 | 실패 — M4.1 fail-soft surface는 보존하지만 단일 settings identity와 executable acceptance mapping이 미완료다. |
| 정량 Gate | §10 | 실패 — 11행은 모두 있으나 disconnect node mapping과 drain winner 때문에 repeat-10 runner가 1:1 실행 계약이 아니다. |

## Validation evidence

- Requirement §4/§5/§6의 기능 요구사항 9개, 고정 profile 11개, repeat-10, terminal
  conservation, live 12/M3 14-gate/M4.1 separation을 Design의 symbol·fixture·assert·report field와
  행별 대조했다.
- 현재 `web/server.py`, `settings.py`, `config.py`, `rag_engine.py`, `agent.py`,
  `web_search.py`, health/metrics 모듈과 관련 unit/integration/CLI tests를 확인해 design-only
  symbol과 실제 cache/import/ASGI/CLI seam을 구분했다.
- timeout/cancel/completion, loop alive/closed, drain completion/deadline/tie, large ASGI chunk,
  settings loader/cache/engine import를 순서별로 재연산했다.
- non-identity compressed payload는 receive 0의 fail-closed 정책으로 검토했으며 gzip 지원으로
  오인하지 않았다. live M3 결과와 `M4.1_BLOCKED`도 deterministic M4.2 PASS로 합성되지 않았음을
  확인했다.
- 문서 작성 후 `python scripts/check_markdown_links.py`와 `git diff --check`를 실행한다.

## Re-review entry conditions

1. lifespan successful load를 process cache/engine import와 연결해 `Settings` validation 1회와
   동일 object identity를 executable test로 증명한다.
2. `RagASGIRoute`의 실제 request-handler-send control flow와 queued/running `http.disconnect`
   100회 race를 고정한다.
3. large single identity chunk를 포함해 payload `limit+1` 계약을 구현 가능한 계층과 측정값으로
   다시 정의한다.
4. drain completion/deadline/tie를 단일 linearization token으로 결정하고 undefined
   `remaining()` seam을 제거한다.
5. caller-cancellation exact node mapping을 하나로 통일해 10 deterministic profiles × repeat 10,
   conservation negative control, 별도 live 12/M3/M4.1 receipt를 collect-and-run으로 증명한다.

