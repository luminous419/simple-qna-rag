# M4.2 Safe Serving Boundary — Design Recovery Review Iteration 4

검토 대상: [Requirement.md](Requirement.md), [Plan.md](Plan.md), [Design.md](Design.md),
[Design Recovery Validation](Design_Recovery_Validation.md),
[Design Recovery Review Iteration 1](Design_Recovery_Review_Iteration_1.md),
[Design Recovery Review Iteration 2](Design_Recovery_Review_Iteration_2.md),
[Design Recovery Review Iteration 3](Design_Recovery_Review_Iteration_3.md),
[Design Review Iteration 1](Design_Review_Iteration_1.md),
[Design Review Iteration 2](Design_Review_Iteration_2.md),
[Design Review Iteration 3](Design_Review_Iteration_3.md),
[Design Review Iteration 4](Design_Review_Iteration_4.md),
[개발 orchestration guide](../../../milestone_dev_orchestration_guide.md), 현재 repository의
settings/config/server/CLI/request-context/health/metrics runtime seam.

이번 검토는 Recovery Iteration 3의 M42-RR3-001/002 exact fix와 모든 이전 finding을 독립적으로
재평가했다. 구현 또는 product acceptance를 실행했다고 간주하지 않고, normative 설계와 bounded
prototype evidence를 분리해 판정했다.

## Executive summary와 Gate

**PASS — 9.8/10.0.** Iteration 4는 startup attempt를 identity mismatch, invalid loader,
started/partially-started lifecycle owner로 분리해 M42-RR3-001/002를 설계 범위에서 폐쇄했다.
Identity mismatch는 exact-owner lease release 외 app/health/log/metric/cache/config/engine/executor와
STOPPED delta가 0이다. Invalid loader는 atomic `settings_invalid` transaction 하나만 publish하고
generic STOPPED publication 없이 exact-owner release한다. Lifecycle-owning teardown은 모든 fallible
observer/snapshot/error aggregation을 끝낸 뒤 non-throwing atomic STOPPED publication과 즉시 이어지는
atomic guard release를 마지막 두 durable external action으로 고정한다. Release 뒤 diagnostic은
best-effort/non-durable이며 reacquire를 막지 않는다.

| Severity | 수 |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 0 |
| MINOR | 0 |

Gate 기준 `score >= 9.7`, CRITICAL 0, MAJOR 0, MINOR 최소화를 모두 만족한다. 따라서 **Design Gate
PASS**이며 아래 implementation entry conditions를 지키는 조건으로 Phase 2 구현 진입을 승인한다.
M4.1 Operational Acceptance의 `M4.1_BLOCKED`는 별도 release blocker로 유지된다.

## Recovery Iteration 3 finding 재평가

| Finding | Iteration 4 판정 | exact evidence |
|---|---|---|
| M42-RR3-001 | **CLOSED (design)** | `Design.md` §4.3.1의 attempt-class table과 §4.3.4 algorithm은 identity verify를 모든 publication/factory보다 앞에 두고 mismatch를 release-only로 끝낸다. Invalid loader만 atomic `settings_invalid` 1회를 소유하며 generic stopped observer는 0이다. |
| M42-RR3-002 | **CLOSED (design)** | §4.3.2는 lifecycle owner에만 canonical tail을 적용한다. begin/applicable wait/mandatory shutdown 및 모든 fallible observers/snapshot/aggregation이 먼저 끝나고 `_publish_stopped_atomic` 직후 `release_exact_owner`가 실행된다. 이후 처리는 non-durable best-effort다. |

### M42-RR3-001 closure 검증

Normative order는 `acquire -> local candidate/executor/grace/trace -> loader -> commit_or_verify`다.
Different identity는 `commit_or_verify`에서 고정 오류로 끝나며 app lifecycle owner로 전환되지 않는다.
따라서 full attempt 동안 허용된 외부 동작은 exact lease owner release 하나뿐이다. `app.__dict__`,
health/readiness state, service log/metric sinks, process settings cache, config facade, engine/executor
factory와 STOPPED publication 모두 delta 0이라는 계약이 prose, pseudocode, startup matrix와 acceptance
spy에 일치한다. Loader 호출과 lease-local trace는 이 외부 delta taxonomy에 포함되지 않으며 loader
result는 local candidate이므로 reject 전에 global identity를 바꾸지 않는다.

Invalid loader는 identity mismatch와 다른 fail-soft bootstrap owner다. Settings validation failure는
commit/cache/factory를 호출하지 않고 atomic `settings_invalid` health transaction을 정확히 한 번
publish한다. Cleanup dispatcher는 이 class에 generic STOPPED observer를 적용하지 않고 exact-owner
release만 실행하므로 diagnostic overwrite/add가 없다. Invalid lifespan이 health surface를 제공하는
동안 추가 startup diagnostic owner도 없다.

### M42-RR3-002 closure 검증

Lifecycle ownership이 확정된 path만 `_teardown` state machine을 탄다. Executor가 있으면 begin을
최대 1회, begin 성공 시 bounded wait를 최대 1회, 선행 error/cancellation과 무관하게 non-waiting
shutdown을 정확히 1회 시도한다. Executor가 없으면 세 동작은 모두 0이지만 lifecycle publication이
이미 시작됐다면 observer/snapshot/aggregation 뒤 같은 canonical tail을 실행한다. Teardown argument
평가, coroutine/task creation, shield cancellation과 inline fallback도 primary-preserving boundary에
포함돼 mandatory tail을 건너뛰지 않는다.

모든 fallible observer, residual snapshot과 ordered cleanup aggregation은 STOPPED 전에 완료된다.
`_publish_stopped_atomic`은 non-throwing atomic snapshot publication이고, 바로 다음
`release_exact_owner`는 lock 아래 exact token을 확인해 owner를 `None`으로 durable clear하는
non-throwing primitive다. 두 동작 사이에는 observer/log/metric/aggregation이 없으며 release 뒤의
diagnostic-code 처리는 실패를 삼키는 best-effort/non-durable 경로라 새 owner의 acquire와 durable
state를 지연하거나 되돌리지 않는다. Primary exception/cancellation은 cleanup 뒤 원 identity로
전파되고, primary가 없을 때만 단일 cleanup error 또는 ordered `ExceptionGroup`이 전파된다.

## 전체 이전 finding closure matrix

| Finding | Iteration 4 최종 판정 | 근거 |
|---|---|---|
| M42-DR1-001 | CLOSED (design) | caller outcome과 future-owned resource completion guard가 분리되고 ABANDONED slot은 future 완료까지 유지된다. |
| M42-DR1-002 | CLOSED (design) | asyncio-free resource completion이 먼저 끝나고 loop notification은 별도 best-effort다. |
| M42-DR1-003 | CLOSED (design) | Future 반환이 submit commit point이며 direct/promotion failure rollback과 terminal accounting이 고정됐다. |
| M42-DR1-004 | CLOSED (design) | capacity edge timestamp/version이 executor lock 아래 probe 사이 clear/re-full을 보존한다. |
| M42-DR1-005 | CLOSED (design) | router/answer마다 remaining-budget Ollama client를 만들고 owned transport를 닫는다. |
| M42-DR1-006 | CLOSED (design) | accepted/terminal 및 submit-attempt 보존식, scrape snapshot gauge, sink 격리가 일치한다. |
| M42-DR1-007 | CLOSED (catalog only) | Requirement §5의 11행, ordered deterministic 10-profile mapping, repeat/conservation과 별도 live status가 일치한다. |
| M42-DR1-008 | CLOSED (design) | guard sole owner가 single load/commit을 수행하고 immutable process identity와 attempt-class cleanup을 사용한다. |
| M42-DR2-001 | CLOSED (design) | import는 loader 0, owner lifespan만 loader 1회이며 invalid fail-soft diagnostic과 CLI exit 2가 분리된다. |
| M42-DR2-002 | CLOSED (design) | actual `http.disconnect` observer, route marker, conditional send와 pure-ASGI outer owner가 연결됐다. |
| M42-DR2-003 | CLOSED (design) | non-identity encoding early reject와 identity input의 wire/application-consumed accounting이 정직하다. |
| M42-DR2-004 | CLOSED (design) | drain resource/deadline은 single lock/sequence/CAS와 waiter-owned absolute deadline을 쓴다. |
| M42-DR2-005 | CLOSED (catalog only) | exact node collection, repeat receipts, negative/live/M3/M4.1 분리가 일치한다. |
| M42-DR3-001 | CLOSED (design) | loader result는 local candidate이고 first/same commit 뒤에만 cache-dependent construction을 수행하며 replacement/rollback은 없다. |
| M42-DR3-002 | CLOSED (design) | actual-app queued/running disconnect races가 route와 outer request observability까지 포함한다. |
| M42-DR3-003 | CLOSED (design) | already-delivered wire bytes와 downstream-consumed `limit+1` prefix를 분리해 가능한 보장만 주장한다. |
| M42-DR3-004 | CLOSED (design) | completion/deadline/tie 양 order/stale/zero가 같은 CAS winner와 deterministic scheduler를 쓴다. |
| M42-DR3-005 | CLOSED (catalog only) | caller cancellation 두 exact nodes와 ordered literal inventory가 byte-for-byte 일치한다. |
| M42-DR4-001 | CLOSED (design) | pure-ASGI request-context는 proven disconnect의 frame 0 정상 terminal을 허용하고 erroneous no-response를 internal error로 분리한다. |
| M42-DR4-002 | CLOSED (design) | process당 active lifespan 하나, immutable committed identity와 exact-owner release/reacquire로 stale rollback을 제거했다. |
| M42-RR1-001 | CLOSED (design) | first commit과 same-object reacquire만 허용하고 different object는 모든 external mutation 전에 거부한다. |
| M42-RR1-002 | CLOSED (design) | primary/cancellation 보존, mandatory shutdown attempt와 final STOPPED→release ordering이 닫혔다. |
| M42-RR2-001 | CLOSED (design) | candidate-only load와 verify-before-publication ordering에 attempt-class release policy가 결합됐다. |
| M42-RR2-002 | CLOSED (design) | acquire 직후 initialized locals와 class-specific cleanup이 loader/constructor/cancellation 진입 실패를 안전하게 소유한다. |
| M42-RR3-001 | CLOSED (design) | mismatch full external delta 0/release 1, invalid loader diagnostic 1/generic overwrite 0/release 1이 normative다. |
| M42-RR3-002 | CLOSED (design) | lifecycle owner의 모든 fallible work가 final non-throwing atomic STOPPED→release 앞에서 끝난다. |

## Request terminal, process identity와 acceptance catalog

Request terminal 분리는 sound하다. Route marker 또는 wrapped receive가 실제 관측한
`http.disconnect`만 proven disconnect다. 이때 outcome은 internal `client_disconnected`,
status-equivalent 499, wire response frame 0이며 HTTP 499를 전송하지 않는다. 증거 없이 downstream이
frame 0으로 정상 반환하면 `RuntimeError("downstream_no_response")`, internal/500-equivalent,
wire frame 0이다. Outer cancellation은 observed disconnect가 있을 때만 499-equivalent이며 그렇지
않으면 cancellation으로 유지된다. Result-first는 정상 response stream 하나, disconnect-first는 send
0, tie는 callback insertion sequence의 single CAS winner이고 loser task는 항상 회수된다.

Process settings identity는 최초 successful `Settings` object reference로 immutable하다. First commit,
same-object sequential reacquire, different-object deterministic reject만 지원한다. Equal-value object도
다른 identity면 거부하며 reset/rebind/reload/rollback/generation API는 production scope에 없다. Concurrent
second lifespan은 loader와 모든 mutation 전에 `lifespan_already_active`로 거부된다. Module ASGI와 CLI
preflight는 같은 commit/verify primitive를 쓰되 invalid CLI exit 2와 fail-soft ASGI health surface는
서로 다른 owner다.

Requirement §5 catalog는 정확히 11 profiles다. Ordered deterministic 10은 `event_loop`,
`bounded_admission`, `fifo_cancel`, `queue_timeout`, `execution_timeout`, `caller_cancellation`, `drain`,
`saturation_readiness`, `payload`, `normal_mock_load`이고 caller cancellation만 queued/running 두 exact
pytest nodes를 가진다. 각 profile/node별 repeat 10과 conservation 10, inventory 밖 negative control 1,
별도 opt-in live 12/M3 14-gate/M4.1 status가 서로 합성되지 않는다. 이는 설계 catalog이며 아직
project runner receipt가 아니다.

## Requirements와 implementation entry conditions

| 대상 | 최종 판정 | 구현 진입 조건 |
|---|---|---|
| REQ-001 | PASS (design) | 49-field inventory, validator와 immutable identity를 actual settings/config tests로 구현한다. |
| REQ-002 | PASS (design) | bounded FIFO executor, commit/rollback과 두 finalize guard를 구현한다. |
| REQ-003 | PASS (design) | queue/execution budgets와 actual-ASGI disconnect race owner를 구현한다. |
| REQ-004 | PASS (design) | orphan/resource accounting, conservation과 fixed safe errors를 구현한다. |
| REQ-005 | PASS (design) | mandatory shutdown matrix와 attempt-class-specific final publication/release를 구현한다. |
| REQ-006 | PASS (design) | readiness precedence/edge debounce와 bounded metrics spies를 구현한다. |
| REQ-007 | PASS (design) | encoding early reject 및 honest wire/consumed byte boundary를 구현한다. |
| REQ-008 | PASS (design) | context-local deadline과 per-call owned network adapters를 구현한다. |
| REQ-009 | PASS (design) | CLI/module compatibility, proven disconnect observability와 M4.1 preservation을 구현한다. |
| deterministic 10 | COMPLETE CATALOG, NOT RUN | exact ordered node inventory, repeat-10 runner와 conservation receipts가 필요하다. |
| Requirement §5 11행 | COMPLETE CATALOG, NOT RUN | deterministic 10과 opt-in live 1을 별도 status로 실행해야 한다. |

구현 진입 조건은 다음과 같다.

1. Attempt-class dispatcher를 단일 normative owner로 구현해 identity mismatch에서 exact-owner release
   외 real app `__dict__`, health/readiness, service log/metric sinks, cache/config facade,
   engine/executor factory와 STOPPED delta가 모두 0임을 fresh subprocess spy로 검증한다.
2. Invalid loader의 atomic `settings_invalid` transaction 1, generic stopped/overwrite/add 0,
   commit/factory 0, exact-owner release 1을 actual module ASGI path에서 검증한다.
3. Lifecycle owner의 executor-none/present, begin/wait/shutdown/observer/snapshot 단일·복합 error,
   grace expiry, teardown task creation/evaluation failure와 모든 shield cancellation boundary에서
   final durable tail이 정확히 `STOPPED→release`이고 즉시 reacquire 가능함을 검증한다.
4. Release primitive는 exact token clear를 non-throwing atomic operation으로 만들고, post-release
   diagnostic sink failure/재진입/concurrent reacquire가 durable cleanup이나 새 owner를 바꾸지 않음을
   검증한다. Primary exception/cancellation identity와 ordered secondary/`ExceptionGroup`도 보존한다.
5. Current `BaseHTTPMiddleware`와 synchronous `/rag` seam을 설계한 pure-ASGI route/context stack으로
   교체하고, actual app에서 proven disconnect 499-equivalent/wire 0과 erroneous no-response
   500-equivalent/wire 0, request ID/log/metric exactly once와 pending task 0을 검증한다.
6. Exact 11-profile catalog와 deterministic repeat-10/conservation, 별도 negative control,
   opt-in live 12/M3 artifact/M4.1 blocker 분리를 구현한다. Prototype stdout를 product PASS receipt로
   승격하지 않는다.
7. 구현 뒤 독립 code review, clean full suite, regression/acceptance와 release 절차를 Plan 순서대로
   수행한다. `M4.1_BLOCKED`는 해소 또는 별도 risk 승인 전 M4 release-ready 판정을 막는다.

## Prototype evidence와 현재 runtime seam

[Design Recovery Validation §11](Design_Recovery_Validation.md)의 spies는 normative behavior를 정확히
characterize한다. Mismatch의 일곱 external snapshot delta 0/STOPPED 0/release 1, invalid loader의
transaction 1/generic stopped 0/release 1, started owner의 `observer,snapshot,aggregate,STOPPED,release`
tail과 post-release diagnostic failure 뒤 reacquire를 bounded 실행했다. Earlier prototype의 모든-case
STOPPED trace는 RR2 characterization이며 Iteration 4 attempt-class refinement가 supersede한다.

다만 이 evidence는 **PROTOTYPE-ONLY**다. Toy `World`/`Guard`는 product modules를 import하지 않고
real app state, Prometheus/logging sinks, settings cache/config facade, factory, actual coroutine shield와
concurrent reacquire를 실행하지 않는다. 현재 code는 아직 41-field settings, eager config facade,
`BaseHTTPMiddleware`, synchronous `/rag`, teardown 없는 M4.1 lifespan을 사용한다. 따라서 이번 PASS는
구현 가능한 설계의 Gate PASS이지 M4.2 product implementation/test/acceptance PASS가 아니다.

## Iteration 5와 반복-root 중단 규칙

Gate를 통과했으므로 Iteration 5 extension은 **불필요하며 열지 않는다**. M42-RR3-001/002는 이전
root의 단순 재진술이 아니라 attempt-class ownership과 canonical tail의 normative 수정으로 이번
회차에 모두 폐쇄됐다. 따라서 guide의 조건부 연장 중단 조건인 “동일 근본 문제 2회 연속 재발”도
현재 적용되지 않는다. 구현 또는 code review에서 같은 zero-delta/publication-owner나
STOPPED/release ordering root가 다시 나타나면 새 design iteration으로 무한 연장하지 말고 guide의
repeated-root stop rule에 따라 중단·재개 조건을 기록해야 한다.

사용자 결정이 필요한 새 제품 범위는 없다.
