# M4.2 Safe Serving Boundary — Design Recovery Review Iteration 2

검토 대상: [Requirement.md](Requirement.md), [Plan.md](Plan.md), [Design.md](Design.md),
[Design Recovery Validation](Design_Recovery_Validation.md),
[Design Recovery Review Iteration 1](Design_Recovery_Review_Iteration_1.md),
[Design Review Iteration 1](Design_Review_Iteration_1.md),
[Design Review Iteration 2](Design_Review_Iteration_2.md),
[Design Review Iteration 3](Design_Review_Iteration_3.md),
[Design Review Iteration 4](Design_Review_Iteration_4.md),
[개발 orchestration guide](../../../milestone_dev_orchestration_guide.md), 현재 repository의
request-context/settings/config/server/CLI code와 관련 test/runtime seam.

사용자가 승인한 simplified scope를 authoritative contract로 적용했다. request-context는 pure
ASGI이며 증명된 disconnect만 내부 `client_disconnected`/499-equivalent로 기록하고 wire response는
0이다. process당 active lifespan은 하나이고, 최초 successful `Settings` identity는 process 동안
immutable하며 순차 same-object reacquire만 허용하고 different object는 mutation 전에 거부한다.
teardown은 모든 경로에서 mandatory shutdown을 시도하고 STOPPED를 publish한 뒤 guard를 마지막에
release하며, 그 후 원래 cancellation/primary error를 전파해야 한다.

## Executive summary와 Gate

**FAIL — 9.4/10.0.** Recovery Iteration 2는 M42-RR1-001/002의 목표 정책, proven-disconnect
분류, CLI/module 공통 commit primitive, subprocess identity isolation, cleanup error 우선순위와
11-profile inventory를 설계 수준에서 대부분 구체화했다. 그러나 제시된 lifecycle 실행 순서에는
different identity를 검증하기 전에 `app.state.settings_load_attempted`를 쓰는 mutation이 있고,
loader가 `SettingsError`를 내면 초기화되지 않은 `s`에서 grace를 읽다가 mandatory teardown 자체를
시작하지 못하는 경로가 있다. 따라서 승인된 핵심 불변식 두 개가 pseudocode로 실행되지 않으며
Phase 2 구현 진입은 아직 허용할 수 없다.

| Severity | 수 |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 2 |
| MINOR | 0 |

Gate 기준 `score >= 9.7`, CRITICAL 0, MAJOR 0, MINOR 최소화 중 score와 MAJOR 조건이
실패하므로 **Gate FAIL**이다.

## Full closure matrix

| Finding | Iteration 2 판정 | exact evidence |
|---|---|---|
| M42-DR1-001 | CLOSED | caller terminal과 future-owned resource finalize가 분리되고 ABANDONED future 완료 전 slot을 유지한다. |
| M42-DR1-002 | CLOSED | resource completion은 asyncio-free이며 loop notification은 별도 best-effort다. |
| M42-DR1-003 | CLOSED | `pool.submit()` Future 반환이 commit point이고 direct/promotion 실패 rollback이 고정됐다. |
| M42-DR1-004 | CLOSED | saturation edge의 timestamp/version을 executor lock 아래 기록한다. |
| M42-DR1-005 | CLOSED | router/answer 호출별 remaining-budget client와 transport close를 사용한다. |
| M42-DR1-006 | CLOSED | accepted/terminal conservation, scrape-time gauges와 bounded metric labels가 일치한다. |
| M42-DR1-007 | CLOSED | Requirement §5의 11행, deterministic 10-profile mapping, repeat/conservation 및 별도 live/M3/M4.1 상태가 일치한다. |
| M42-DR1-008 | OPEN (MAJOR) | immutable identity 방향은 맞지만 different-object reject 전에 app-state mutation이 발생한다. M42-RR2-001. |
| M42-DR2-001 | OPEN (MAJOR) | fail-soft single-load 방향은 유지되나 invalid loader 경로가 teardown 진입 전에 unbound `s`를 읽는다. M42-RR2-002. |
| M42-DR2-002 | CLOSED (design) | pure-ASGI request-context와 route marker/observed disconnect가 actual stack의 frame-0 terminal을 소유한다. |
| M42-DR2-003 | CLOSED | non-identity encoding은 receive 전에 거부하고 identity input은 consumed prefix와 wire bytes를 분리한다. |
| M42-DR2-004 | CLOSED | drain resource/deadline은 single CAS winner와 absolute deadline을 사용한다. |
| M42-DR2-005 | CLOSED | catalog의 exact node IDs, runner collection, repeat receipts와 live separation이 일치한다. |
| M42-DR3-001 | OPEN (MAJOR) | stale replacement은 제거됐지만 reject-before-mutation의 실제 ordering이 아직 어긋난다. M42-RR2-001. |
| M42-DR3-002 | CLOSED (design) | actual-app queued/running disconnect races와 outer middleware observability가 연결됐다. |
| M42-DR3-003 | CLOSED | oversized single ASGI message의 wire 전달을 되돌린다고 주장하지 않고 application consumption만 제한한다. |
| M42-DR3-004 | CLOSED | completion/deadline/tie/stale/zero가 동일 lock/sequence/CAS 계약을 사용한다. |
| M42-DR3-005 | CLOSED | caller-cancellation 두 nodes와 ordered inventory가 byte-for-byte 일치한다. |
| M42-DR4-001 | CLOSED (design) | pure-ASGI middleware는 proven disconnect의 send 0을 예외나 허위 500으로 바꾸지 않는다. product acceptance는 미실행이다. |
| M42-DR4-002 | OPEN (MAJOR) | single-active와 immutable commit 정책은 선택됐지만 startup failure teardown과 reject ordering이 불완전하다. M42-RR2-001/002. |
| M42-RR1-001 | OPEN (MAJOR) | same-object reacquire/different-object reject 정책은 맞으나 different reject 이전 mutation 0이라는 own contract를 pseudocode가 위반한다. M42-RR2-001. |
| M42-RR1-002 | OPEN (MAJOR) | cleanup state machine의 정상 진입 후 순서는 맞으나 invalid loader가 cleanup task 생성 전에 실패할 수 있다. M42-RR2-002. |

## Findings와 exact fixes

### M42-RR2-001 — MAJOR — different Settings identity가 mutation 전에 거부되지 않는다

- 위치: `Design.md` §4.3.1, §4.3.4 pseudocode와 supported-path table;
  M4.2-REQ-001.3, REQ-009.1/009.2, approved immutable identity contract.
- exact evidence: 설계는 different object를 app/cache/config/engine/executor mutation 전에 고정
  `RuntimeError("process_settings_identity_mismatch")`로 거부하고 해당 case의 app-state mutation도
  0이라고 명시한다. 그러나 pseudocode는 loader 호출 및 `commit_process_settings_once(s)`보다 먼저
  `app.state.settings_load_attempted = True`를 실행한다. 첫 commit `sA`가 있는 process에서 순차
  owner가 `sB`를 반환하면 app state가 이미 변경된 뒤 identity mismatch가 발생한다. 같은 app
  object를 재사용하면 이전 successful state와 새 실패 시도의 observability도 섞일 수 있다.
- CLI/module consistency: CLI preflight와 module lifespan이 공통 commit/verify primitive를 쓰는
  방향은 타당하지만, module 경로만 commit 전 app mutation을 갖기 때문에 두 entry point가 동일한
  reject-before-mutation 계약을 수행하지 않는다. fresh subprocess를 쓰겠다는 test 문구도 현재
  실행 순서의 결함을 고치지 않는다.
- exact required fix: loader result는 local 변수로만 보존하고 `commit_process_settings_once(candidate)`가
  first/same을 반환한 뒤에만 app state, cache-dependent facade, engine, executor를 변경하라. invalid
  loader의 fail-soft 진단 publication은 identity mismatch와 분리된 explicit failure transaction으로
  정의하라. first commit, same-object sequential reacquire, equal-value different object, concurrent
  second, CLI preflight 각각에서 mutation trace를 commit CAS 전/후로 기록하고 different case는
  `app/cache/config/engine/executor` delta 0을 fresh subprocess에서 검증해야 한다.

### M42-RR2-002 — MAJOR — initial loader failure가 mandatory teardown 전에 unbound settings를 읽는다

- 위치: `Design.md` §4.3.2~4.3.4의 `_make_lifespan` pseudocode와 startup/cleanup matrix;
  M4.2-REQ-005.1~005.4, approved teardown contract.
- exact evidence: pseudocode는 `s`를 `settings_loader()` 성공 시에만 대입하지만 최외곽
  `finally`에서 항상 `_teardown(executor, app, lease, s.SHUTDOWN_GRACE_SECONDS)`를 평가한다.
  `settings_loader()`가 `SettingsError`를 내면 fail-soft catch 뒤 `s`는 존재하지 않는다. cleanup
  task를 만들기 위한 argument evaluation이 `UnboundLocalError`로 실패하므로 `begin_drain`, mandatory
  `shutdown`, STOPPED publication, guard release와 primary/secondary policy가 모두 실행되지 않는다.
  process guard가 영구 점유될 수 있고 다음 valid owner도 `lifespan_already_active`로 거부된다.
- lifecycle coverage: executor construction 전에는 begin/wait/shutdown count 0이라는 catalog 자체는
  합리적이지만 STOPPED/release는 여전히 1이어야 한다. 현재 pseudocode는 settings failure,
  identity reject, startup cancellation이 `s` 대입 전 발생하는 모든 경로에 안전한 grace source를
  제공하지 않는다. prototype은 항상 만들어진 fake executor를 직접 teardown하므로 이 진입 실패를
  실행하지 않았다.
- exact required fix: `candidate=None`, `executor=None`, bootstrap-safe bounded grace 또는
  `grace=0.0`을 lease 획득 직후 local로 초기화하고, validated same/first Settings가 확정된 뒤에만
  configured grace로 교체하라. teardown task 생성 자체도 primary-preserving outer guard에 넣고,
  executor가 없으면 begin/wait/shutdown은 명시적으로 skipped하되 STOPPED publish와 release는 반드시
  수행하라. loader error, identity mismatch, cancellation before/during loader, engine error, executor
  constructor error, cancellation at every cleanup await/shield boundary, begin/wait/shutdown single+combined
  errors를 검증하고 trace의 마지막 두 durable actions가 항상 STOPPED→release인지 확인해야 한다.

## Request terminal과 erroneous no-response 판정

Pure-ASGI 계층과 분류 표는 승인 범위에 맞다. route가 설정한
`scope["state"]["rag_terminal"] == "client_disconnected"` 또는 middleware가 직접 관측한
`http.disconnect`만 frame-0 499-equivalent를 허용한다. 증거 없이 downstream이 frame을 보내지 않고
정상 반환하면 `RuntimeError("downstream_no_response")`, internal/500-equivalent, wire frame 0이며
`client_disconnected`나 4xx metric으로 분류하지 않는다. downstream exception과 outer cancellation도
새 보상 response를 만들지 않는다.

이 계약은 M42-DR4-001을 설계 수준에서 닫지만 실제 project acceptance는 남아 있다. actual
`create_app()` 전체 stack에서 disconnect-first/result-first/tie/outer-cancel 및 erroneous no-response를
실행해 request ID set/reset, start/end/duration/counter, terminal marker, frames, receive ownership,
loser reap과 pending task가 각각 exact인지 확인해야 한다.

## Requirements와 acceptance traceability

| 대상 | 판정 | 근거/구현 진입 조건 |
|---|---|---|
| REQ-001 | FAIL | 8개 field/validator와 immutable identity 정책은 구체적이나 different object가 app mutation 뒤 거부된다(M42-RR2-001). |
| REQ-002 | PASS (design) | bounded atomic admission, explicit FIFO, submit commit/rollback과 future-owned slot이 닫혔다. |
| REQ-003 | PASS (design) | queue/execution budget과 actual-ASGI proven disconnect owner가 구체적이다. |
| REQ-004 | PASS (design) | orphan invariant, terminal conservation과 fixed safe error body가 닫혔다. |
| REQ-005 | FAIL | loader failure가 mandatory cleanup/STOPPED/release 전에 실패할 수 있다(M42-RR2-002). |
| REQ-006 | PASS (design) | readiness precedence, edge debounce와 bounded series가 executable fixture에 연결됐다. |
| REQ-007 | PASS (design) | non-identity early reject와 honest wire/application byte accounting이 구현 가능하다. |
| REQ-008 | PASS (design) | context-local remaining deadline과 per-call clients가 current runtime seam에 맞다. |
| REQ-009 | FAIL | CLI/module 공통 primitive 방향은 맞으나 module reject ordering과 failed-start lifecycle이 호환 계약을 완결하지 못한다. |
| deterministic 10 profiles | COMPLETE CATALOG, NOT RUN | ordered mapping, node collection, repeat 10/conservation 계약은 일치하나 project symbols/results는 아직 구현되지 않았다. |
| Requirement §5 11행 | COMPLETE CATALOG, NOT RUN | deterministic 10행과 별도 opt-in live 1행으로 정확히 11행이다. |
| live 12/M3/M4.1 | SEPARATE | live 12, M3 14-gate artifact와 `M4.1_BLOCKED`는 deterministic aggregate에 합성되지 않는다. |

`PROFILE_NODE_IDS`는 `event_loop`, `bounded_admission`, `fifo_cancel`, `queue_timeout`,
`execution_timeout`, `caller_cancellation`, `drain`, `saturation_readiness`, `payload`,
`normal_mock_load` 순서의 deterministic 10 profiles다. caller cancellation만 queued/running 두 exact
nodes를 가지며 catalog와 literal tuple이 일치한다. profile/node별 10 receipts, conservation 10,
aggregate 밖 negative control 1, 별도 opt-in live 12/M3/M4.1 status 계약도 유지됐다.

## Evidence quality

`Design_Recovery_Validation.md`의 prototype evidence는 **SOUND BUT LIMITED**다. 설치
Starlette `0.50.0`의 `BaseHTTPMiddleware` no-response 실패와 최소 pure-ASGI frame-0 정상 반환을
재현했고, toy guard/identity/teardown/classifier의 고정 입력에서는 command exit 0과 명시된 trace를
얻었다. 2초/1초 bounded wrapper도 characterization hang 방지로 적절하다.

그러나 prototype은 product modules를 import하지 않고 actual `create_app()`, config facade, CLI,
engine/executor factories 또는 request observability를 실행하지 않는다. identity prototype은 commit
전에 app-state mutation을 하지 않는 별도 toy 순서를 사용하고, teardown prototype은 loader 실패나
unbound grace evaluation을 거치지 않고 이미 구성된 fake executor에 직접 진입한다. `all_errors`도
begin failure 뒤 wait를 skip하므로 begin+wait+shutdown 세 오류를 동시에 집계하는 증거가 아니며,
actual shield cancellation 재전파와 original exception identity는 의도적으로 미검증이다. 따라서
prototype 결과는 설계 아이디어의 bounded characterization이지 M42-RR1-001/002 closure나 project
acceptance PASS receipt가 아니다.

## Implementation entry conditions

1. M42-RR2-001을 수정해 candidate validation/identity verify 전 global/app mutation을 0으로 만들고,
   CLI와 module ASGI가 동일 commit/verify transaction과 exact error contract를 사용하게 한다.
2. M42-RR2-002를 수정해 settings가 없는 모든 startup failure/cancellation에서도 teardown을 생성하고
   STOPPED publication 뒤 guard를 마지막에 release하게 한다.
3. cleanup error aggregation은 original primary/cancellation identity 우선, 없으면 단일 원본 error,
   복수면 ordered `ExceptionGroup`으로 고정하고 secondary bounded log receipt를 project fake로 검증한다.
4. identity matrix는 first/same/different 각 독립 fresh subprocess로 실행해 process-immutable commit이
   다른 test case에 오염되지 않게 하고 loader/cache/config/engine/executor/app-state identity/count를
   기록한다.
5. actual app에서 proven disconnect의 internal 499-equivalent/wire 0과 erroneous downstream
   no-response의 internal 500-equivalent/wire 0을 분리해 exactly-once observability와 pending 0을 증명한다.
6. 구현 전 fresh review가 CRITICAL 0, MAJOR 0, MINOR 최소, score 9.7 이상을 달성해야 한다.

사용자 결정이 필요한 새 범위는 없다. 두 finding은 승인된 simplified scope의 ordering과 failure-path
완결성 문제이며, `M4.1_BLOCKED`는 계속 별도 release blocker다.
