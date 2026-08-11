# M4.2 Safe Serving Boundary — Design Recovery Review Iteration 1

검토 대상: [Requirement.md](Requirement.md), [Plan.md](Plan.md), [Design.md](Design.md),
[Design Recovery Validation](Design_Recovery_Validation.md),
[Design Review Iteration 1](Design_Review_Iteration_1.md),
[Design Review Iteration 2](Design_Review_Iteration_2.md),
[Design Review Iteration 3](Design_Review_Iteration_3.md),
[Design Review Iteration 4](Design_Review_Iteration_4.md),
[개발 orchestration guide](../../../milestone_dev_orchestration_guide.md), 현재 repository
request-context/settings/server code 및 설치 Starlette `0.50.0` source.

사용자 승인 복구 범위를 판정 기준으로 삼았다. client disconnect는 wire response가 없는 정상
terminal이며 내부 `client_disconnected`/499-equivalent로만 분류한다. request-context만 pure
ASGI로 교체하고, process마다 active lifespan 하나만 지원하며 concurrent second는 모든 global
mutation 전에 거부해야 한다.

## Executive summary와 Gate

**FAIL — 9.2/10.0.** Pure-ASGI request-context 전환은 M42-DR4-001의
`BaseHTTPMiddleware` no-response 실패를 올바른 계층에서 제거하며, 499를 실제 response로 보내지
않는 계약과 actual-app exactly-once acceptance도 구체적이다. Single-active guard도 concurrent
second를 mutation 전에 거부하는 단순화 자체는 타당하다. 그러나 순차 lifespan이 다른
`Settings`를 commit할 때 이미 materialize된 `config.py` facade/engine import identity가 갱신되지
않고, shutdown cancellation 또는 drain 오류 때 guard는 release되지만 executor shutdown이
건너뛰어질 수 있으므로 구현 진입 조건을 아직 만족하지 않는다.

| Severity | 수 |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 2 |
| MINOR | 0 |

Gate 기준 `score >= 9.7`, CRITICAL 0, MAJOR 0, MINOR 최소화 중 score와 MAJOR 조건이
실패하므로 **Gate FAIL**이다. Phase 2 product 구현으로 진행하면 안 된다.

## Closure matrix

| Finding | recovery 판정 | 근거 |
|---|---|---|
| M42-DR1-001 | CLOSED | §2.2/§2.6이 caller terminal과 future-owned resource finalize를 분리해 abandoned running slot의 조기 반환을 막는다. |
| M42-DR1-002 | CLOSED | §2.7이 asyncio-free resource completion과 best-effort loop notification을 분리한다. |
| M42-DR1-003 | CLOSED | §2.4가 `pool.submit()` Future 반환을 commit point로 두고 direct/promotion 실패를 rollback한다. |
| M42-DR1-004 | CLOSED | §2.3/§8.2가 saturation edge timestamp/version을 executor lock 아래 보존한다. |
| M42-DR1-005 | CLOSED | §6.2가 호출별 remaining-budget client와 transport close seam을 고정한다. |
| M42-DR1-006 | CLOSED | §2.3/§9가 conservation, scrape-time snapshot과 bounded labels를 고정한다. |
| M42-DR1-007 | CLOSED | §10의 Requirement §5 11행, deterministic 10-profile inventory, repeat/conservation 및 별도 live/M3/M4.1 상태가 일치한다. |
| M42-DR1-008 | OPEN (MAJOR) | concurrent overlap은 제거됐지만 순차 lifespan의 second Settings identity가 materialized facade까지 전파되지 않는다. M42-RR1-001. |
| M42-DR2-001 | OPEN (MAJOR) | 최초 owner의 single validation/identity는 닫혔지만 지원되는 release→reacquire에서 다른 identity가 process 전체에 일치하지 않는다. M42-RR1-001. |
| M42-DR2-002 | CLOSED | §4.1/§4.1.1이 실제 stack에서 request-context만 pure ASGI로 교체하고 frame-0 disconnect를 정상 terminal로 소유한다. |
| M42-DR2-003 | CLOSED | §4.4가 wire-delivered와 application-consumed byte를 정직하게 분리하고 encoded input을 receive 전에 거부한다. |
| M42-DR2-004 | CLOSED | §2.9가 resource/deadline 단일 CAS winner와 waiter-owned absolute deadline을 사용한다. |
| M42-DR2-005 | CLOSED | §10 mapping/catalog/repeat/live 분리가 executable symbol 수준으로 일치한다. |
| M42-DR3-001 | OPEN (MAJOR) | overlapping rollback 문제는 범위 제거됐지만 sequential replacement의 stale facade라는 같은 Settings identity 계열 문제가 남는다. M42-RR1-001. |
| M42-DR3-002 | CLOSED | pure-ASGI outer terminal owner와 actual-app queued/running race acceptance가 middleware observability까지 포함한다. |
| M42-DR3-003 | CLOSED | oversized single ASGI message를 예방한다고 주장하지 않고 consumed prefix만 `limit+1`로 제한한다. |
| M42-DR3-004 | CLOSED | drain deadline/resource winner 자체의 tie·stale·zero 계약은 단일 lock/sequence/CAS로 정해졌다. |
| M42-DR3-005 | CLOSED | caller cancellation 두 node와 ordered profile inventory가 byte-for-byte 일치한다. |
| M42-DR4-001 | CLOSED (design) | request-context 한 곳의 pure-ASGI rewrite가 downstream frame 0을 예외/허위 500으로 바꾸지 않는다. actual project acceptance는 구현 후 필요하다. |
| M42-DR4-002 | OPEN (MAJOR) | concurrent reject-before-mutation은 닫혔지만 owner teardown의 executor cleanup과 supported sequential owner identity가 완전하지 않다. M42-RR1-001/002. |

## Findings와 exact fixes

### M42-RR1-001 — MAJOR — 순차 lifespan 재획득이 process-wide Settings identity를 보존하지 않는다

- 위치: `Design.md` §4.3.1~4.3.2/§10, 현재
  `settings.py::{get_settings,set_settings_for_process}` 및 `config.py` module facade;
  M4.2-REQ-001.3, REQ-005.1, REQ-009.1/009.2.
- exact evidence: 복구 설계는 owner shutdown 뒤 cache를 그대로 두고, 다음 단독 lifespan이
  guard를 획득해 새 validated `sB`로 cache를 교체한다고 명시한다. 그러나 현재 `config.py`는 첫
  import에서 `_settings = get_settings()`를 실행하고 41개 facade global을 즉시 materialize한다.
  첫 lifespan A의 lazy engine import 뒤에는 `_settings is sA`와 facade 값이 고정된다. A 종료 후
  B가 `set_settings_for_process(sB)`를 호출해도 이미 import된 `config.py._settings`, facade globals,
  그리고 이를 import한 module state는 `sA`에 남는다. 따라서 cache/app/engine이 모두 `is sB`라는
  §4.3 identity 주장은 지원되는 순차 reacquire에서 성립하지 않는다.
- regression sequence: A acquire → cache `sA` → lazy engine/config import → A shutdown/release →
  B acquire → cache `sB` → engine factory가 이미 materialize된 facade 사용. active lifespan은 한
  개뿐이지만 B의 settings/cache/engine network configuration이 서로 달라질 수 있다. 이는
  overlapping lifespan 문제가 아니므로 single-active guard만으로 제거되지 않는다.
- exact required fix: process 설정 identity를 최초 successful commit 뒤 immutable로 고정하고
  후속 lifespan은 동일 object만 허용하도록 계약을 좁히거나, 모든 legacy facade consumer가
  per-owner immutable `Settings`를 명시적으로 받도록 설계하라. 단순 module reload나 global
  facade 재대입은 concurrent worker/import 안전성이 없으므로 허용하지 않는다. executable matrix는
  A release 뒤 same-identity reacquire와 different-identity attempt 각각의 loader count, mutation
  point, cache/config/engine/executor `is`, rejection type/message, guard release/reacquire를 검증해야
  한다. 선택한 정책은 CLI preflight와 module ASGI 양쪽에 동일해야 한다.

### M42-RR1-002 — MAJOR — teardown 예외/cancellation이 executor shutdown을 건너뛴 뒤 guard를 공개한다

- 위치: `Design.md` §2.9, §4.3.1 pseudocode와 shutdown-path catalog;
  M4.2-REQ-005.1~005.4 및 drain 정량 Gate.
- exact evidence: pseudocode의 cleanup은 한 순차 `try` 안에서
  `begin_drain(); await wait_drained(); shutdown()`을 호출하고 바깥 `finally`에서 lease를
  release한다. `begin_drain()` 또는 `wait_drained()`가 예외를 내거나 task가 await 중 cancel되면
  `shutdown()`은 실행되지 않지만 lease는 즉시 release된다. 새 lifespan은 old executor의 running
  threads/resources가 정리되지 않은 상태에서 acquire할 수 있다. 본문의 “drain/shutdown 예외와
  yield 이후 cancellation을 포함한 모든 경로에서 inner cleanup을 시도”한다는 주장과 코드가
  모순된다.
- exact required fix: teardown을 exception-safe state machine으로 고정하라. drain initiation,
  bounded wait, `shutdown(wait=False, cancel_futures=True)`를 독립 guard로 exactly once 수행하고,
  cancellation은 cleanup 동안 지연/차폐한 뒤 원래 cancellation을 재전파해야 한다. 여러 cleanup
  오류가 있으면 primary/secondary 보존 규칙을 정하고 lease는 mandatory shutdown attempt와 app
  STOPPED state publication 이후에만 release하라. `begin_drain` error, wait error, grace expiry,
  shutdown error, cancellation at each await, normal zero/running residual 각각에서 begin/wait/shutdown/
  release count, ordering, final lifecycle, residual, immediate reacquire 가능 시점을 executable fake로
  검증하라.

## Middleware와 request observability 판정

Pure-ASGI replacement의 계층 선택은 타당하다. `BaseHTTPMiddleware.call_next()`가 downstream
첫 response frame을 요구하는 실제 Starlette 0.50.0 제약을 제거하면서 route의 send ownership은
바꾸지 않는다. `client_disconnected`는 bounded internal outcome이고 status-equivalent 499는 log와
기존 `4xx` metric clamp에만 쓰며 wire frame은 0이라는 구분도 정확하다.

다만 pure middleware prototype의 `starts == 0` 판정은 최소 characterization일 뿐 일반적인
“client disconnect 검출” 증거는 아니다. 임의의 downstream no-response 정상 반환도 같은 값으로
분류한다. 설계의 실제 app에서는 `RagASGIRoute`만 정상 frame-0 terminal을 만들도록 제한되어 있으므로
구현 acceptance는 route winner/observer trace와 middleware terminal을 같은 요청에서 결합해
`disconnect observed == true`, start/end/duration/counter 각 1, request ID set/reset 각 1, frames 0,
exception/pending 0을 검증해야 한다. downstream programming-error no-send negative fixture도
`client_disconnected`로 오분류되지 않아야 한다.

## Requirements와 acceptance traceability

| 대상 | 판정 | 근거/진입 조건 |
|---|---|---|
| REQ-001 | FAIL | 8개 field와 validation 규칙은 구체적이나 sequential owner의 Settings facade identity가 미정이다(M42-RR1-001). |
| REQ-002 | PASS (design) | atomic bounded admission, explicit FIFO, commit/rollback과 resource ownership이 닫혔다. |
| REQ-003 | PASS (design) | queue/execution budget 분리, actual-ASGI disconnect, exactly-once race owner가 구체적이다. |
| REQ-004 | PASS (design) | orphan invariant, conservation, safe fixed error contract가 닫혔다. |
| REQ-005 | FAIL | guard release는 있으나 teardown error/cancel에서 mandatory executor shutdown과 STOPPED-before-release가 보장되지 않는다(M42-RR1-002). |
| REQ-006 | PASS (design) | readiness precedence/edge debounce/bounded metrics가 executable fake와 연결된다. |
| REQ-007 | PASS (design) | identity prefix consumption과 non-identity early reject가 구현 가능한 수치로 정직하게 표현됐다. |
| REQ-008 | PASS (design) | context-local deadline과 per-call remaining-budget adapters가 설치 API seam에 연결된다. |
| REQ-009 | FAIL | request observability는 보존 설계가 있으나 stale sequential facade가 CLI/server compatibility를 깨뜨릴 수 있다. |
| deterministic 10 profiles | BLOCKED | ordered mapping과 10-repeat/conservation 계약은 완전하지만 lifecycle/identity negative matrix 수정이 선행돼야 한다. |
| Requirement §5 11행 | COMPLETE CATALOG, NOT RUN | 10 deterministic profiles와 별도 opt-in live 1행이 정확히 11행이며 project implementation test 결과로 주장되지 않았다. |
| live 12/M3/M4.1 | SEPARATE | live 12, M3 14-gate artifact, `M4.1_BLOCKED`가 deterministic PASS와 합성되지 않는다. |

11-profile catalog의 profile/node mapping, caller-cancellation 두 nodes, profile/node별 repeat 10,
conservation 10, 별도 negative control 1은 서로 일치한다. event-loop watchdog은 hang 방지용이고
PASS latency 순서 판정에 쓰지 않으며, normal mock load는 virtual 8.000s를 측정해 Requirement의
wall-time 문구를 결정론적으로 구체화한다. 이 catalog는 구현 예정 계약이지 현재 존재하는 project
tests의 실행 receipt가 아니다.

## Evidence quality

`Design_Recovery_Validation.md`의 command와 기록 결과는 기술적으로 재현 가능한 bounded
characterization이다. 설치 Starlette source와 동일하게 기존 pass-through
`BaseHTTPMiddleware`의 frame-0 `RuntimeError("No response returned.")`를 재현하고, 최소 pure-ASGI
wrapper가 no-response를 허용하며, toy guard가 single owner/release/reacquire를 수행함을 보인다.
각 coroutine을 `asyncio.wait_for(..., 2.0)`으로 제한한 것도 hang 방지에 적절하다.

증거의 한계도 문서가 명시해 과장하지 않았다. pure wrapper는 실제 logging/request-ID/metrics
구현이 아니고, guard prototype의 `mutations` dict는 factories/cache와 연결되지 않은 상수이므로
actual reject-before-mutation이나 failure cleanup을 증명하지 않는다. 결과가 inline prototype이며
product code/test PASS receipt가 아니라고 §1/§4/§5에서 반복해 구분했고, 현재 project 구현
테스트가 존재하거나 통과했다고 주장하지 않았다. 따라서 characterization 자체는 **SOUND BUT
LIMITED**이며 두 MAJOR의 closure evidence로는 부족하다.

## Implementation entry conditions

1. M42-RR1-001의 sequential same/different Settings 정책을 하나로 고정하고 cache/config/engine/
   executor identity matrix와 CLI preflight를 설계에 반영한다.
2. M42-RR1-002의 cancellation-safe teardown state machine, mandatory shutdown attempt,
   STOPPED-before-lease-release 순서와 error aggregation을 pseudocode와 deterministic tests로 고정한다.
3. actual `create_app()`에서 disconnect-first/result-first/tie/outer cancel 및 downstream erroneous
   no-send를 실행해 request start/end/duration/counter, request ID reset, frame/send/receive, loser/
   pending/finalize가 각각 exactly once임을 검증한다.
4. concurrent second가 loader/cache/engine/executor/app-state mutation 전에 실패하고 startup 각
   failure 및 teardown 각 error/cancellation 뒤 guard가 정확히 한 번 release되는 project tests를
   acceptance catalog에 넣는다.
5. 수정 설계의 CRITICAL/MAJOR/MINOR가 0/0/최소이고 재리뷰 score가 9.7 이상이어야 Phase 2로
   진입한다. M4.1은 계속 별도 `M4.1_BLOCKED`다.

사용자 결정이 필요한 새 제품 범위는 없다. 위 두 finding은 승인된 simplified scope 안에서
identity와 teardown ownership을 완결하는 구현 전 설계 수정이다.
