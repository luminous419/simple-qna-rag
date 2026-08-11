# M4.2 Safe Serving Boundary — Design Recovery Review Iteration 3

검토 대상: [Requirement.md](Requirement.md), [Plan.md](Plan.md), [Design.md](Design.md),
[Design Recovery Validation](Design_Recovery_Validation.md),
[Design Recovery Review Iteration 1](Design_Recovery_Review_Iteration_1.md),
[Design Recovery Review Iteration 2](Design_Recovery_Review_Iteration_2.md),
[Design Review Iteration 1](Design_Review_Iteration_1.md),
[Design Review Iteration 2](Design_Review_Iteration_2.md),
[Design Review Iteration 3](Design_Review_Iteration_3.md),
[Design Review Iteration 4](Design_Review_Iteration_4.md),
[개발 orchestration guide](../../../milestone_dev_orchestration_guide.md), 현재 repository의
settings/config/server/CLI/request-context code와 runtime seam.

사용자가 승인한 simplified scope와 Recovery Review Iteration 2의 exact fixes를 authoritative
contract로 적용했다. 특히 different identity는 guard/local trace/cleanup을 제외하고
app/cache/config/engine/executor delta가 정확히 0이어야 하고, invalid loader만 authoritative
fail-soft health diagnostic을 publish할 수 있다. executor가 없으면 begin/wait/shutdown을 모두
건너뛰되 `STOPPED`와 guard release가 순서대로 마지막 두 durable action이어야 한다.

## Executive summary와 Gate

**FAIL — 9.3/10.0.** Iteration 3은 loader 결과를 local candidate에 둔 뒤 identity commit/verify를
app/cache-dependent construction보다 앞세웠고, `candidate=None`, `executor=None`, `grace=0.0`을
초기화해 loader/constructor/cancellation 경로가 teardown에 진입하도록 고쳤다. 그러나 normative
`_teardown`은 different-identity reject 뒤에도 `app.state.lifecycle = "STOPPED"`와 stopped observer를
실행해 claimed app delta 0을 위반하고, executor-none을 포함한 모든 경로에서 `STOPPED`와 release
사이에 fallible observer publication을 넣어 두 동작이 마지막 두 durable action이라는 계약도
위반한다. Iteration 3 prototype은 app mutation counter에서 lifecycle/observer를 제외하고 teardown을
`STOPPED, release` 두 동작으로 축약했으므로 두 모순을 검출하지 못한다.

| Severity | 수 |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 2 |
| MINOR | 0 |

Gate 기준 `score >= 9.7`, CRITICAL 0, MAJOR 0, MINOR 최소화 중 score와 MAJOR 조건이 실패한다.
따라서 **Gate FAIL**이며 Phase 2 구현 진입을 승인하지 않는다.

## Recovery Iteration 2 findings 재평가

| Finding | Iteration 3 판정 | exact evidence |
|---|---|---|
| M42-RR2-001 | **OPEN (MAJOR)** | candidate-only load와 commit-before-construction은 수정됐지만 identity mismatch의 mandatory teardown이 app lifecycle/observer를 변경한다. 문서가 주장한 app delta 0과 normative pseudocode가 충돌한다. M42-RR3-001. |
| M42-RR2-002 | **OPEN (MAJOR)** | initialized locals와 executor-none skip은 수정됐고 executor-present shutdown attempt도 명시됐다. 그러나 stopped observer가 `STOPPED`와 release 사이에 있어 마지막 두 durable action 계약과 failure/cancellation teardown의 canonical tail을 깨뜨린다. M42-RR3-002. |

## Findings와 exact fixes

### M42-RR3-001 — MAJOR — different identity reject 뒤 teardown이 app delta 0을 깨뜨린다

- 위치: `Design.md` §4.3.1, §4.3.2, §4.3.4와 §10.6; M42-RR2-001;
  M4.2-REQ-001.3, REQ-009.1/009.2.
- exact evidence: lifecycle pseudocode는 loader result를 local `candidate`에 두고
  `commit_process_settings_once(candidate)`가 different identity를 거부할 때까지 app/cache/config/
  engine/executor를 touch하지 않는다. 이 부분은 올바르다. 그러나 같은 scope의 unconditional
  `finally`가 `_teardown(executor=None, app, lease, grace=0.0)`을 호출하고, normative `_teardown`은
  `app.state.lifecycle = "STOPPED"`를 쓴 뒤 `_publish_stopped_observers(app, ...)`도 호출한다. 따라서
  reject 전 delta가 0이어도 전체 rejected attempt의 app delta는 0이 아니다. §4.3.1/§4.3.4/§10.6의
  “app delta 0”, “health transaction도 없음”과 직접 모순된다.
- invalid-loader authority: invalid loader도 `settings_invalid` transaction 뒤 같은 unconditional
  STOPPED/observer publication을 거친다. 이 publication이 lifecycle/readiness/metric observer를
  변경할 수 있으므로 invalid loader의 authoritative fail-soft diagnostic 하나만 publish한다는
  계약이 normative algorithm에서 보장되지 않는다. identity mismatch와 invalid loader의 teardown
  tail을 동일하게 둔 채 mutation taxonomy만 prose로 구별할 수 없다.
- prototype gap: Recovery Validation §10의 `counts['app']`은 explicit
  `settings_invalid_transaction`/`app_config_grace`만 세고, toy `teardown()`의 `STOPPED`는 app에 쓰지
  않는다. 그래서 reported different-identity counts 0은 normative app-state delta의 receipt가 아니다.
- current seam: 현재 `src/simple_qna_rag/web/server.py::_make_lifespan`은 settings/engine health state를
  app state로 표현하며, `observability.health.evaluate_readiness`가 그 state를 읽는다. 즉 lifecycle 및
  stopped observer write를 “관측 불가능한 local cleanup”으로 간주할 근거가 없다.
- exact required fix: supported attempt 종류별 publication owner를 명시하라. different identity는
  app/health/metric observer를 받지 않는 cleanup-only tail로 guard를 release하고, invalid loader는
  정확히 하나의 atomic `settings_invalid` transaction을 보존한 뒤 generic STOPPED observer가 이를
  덮거나 추가 publication하지 않게 해야 한다. 또는 app delta 계약을 lifecycle mutation을 허용하도록
  변경하지 말고, cleanup state를 lease-local/process-guard-local로 분리하라. fresh subprocess spy는
  app `__dict__`, health snapshot, metric/log sink, cache/config facade, engine/executor factory를 attempt
  전후 비교해 different identity 전 항목 delta 0, invalid loader는 정확히 한 diagnostic delta만
  assert해야 한다.

### M42-RR3-002 — MAJOR — `STOPPED`와 release가 teardown의 마지막 두 durable action이 아니다

- 위치: `Design.md` §4.3.2 pseudocode와 §4.3.4 startup matrix; M42-RR2-002;
  M4.2-REQ-005.1~005.4.
- exact evidence: normative `_teardown` 순서는 `app.state.lifecycle="STOPPED"` →
  `_publish_stopped_observers(...)` → `lease.release()`다. observer는 fallible이고 오류를
  `stopped_observer_failed` secondary로 기록하므로 명백한 실행 action이며 publication이라는 이름과
  contract상 durable side effect도 가질 수 있다. 따라서 `STOPPED` 다음 즉시 guard release가 오고
  둘이 마지막 두 durable action이라는 요구를 만족하지 않는다.
- failure/cancellation effect: observer가 실패해도 release를 시도하는 방향은 좋지만, STOPPED와
  release 사이 primary/secondary가 추가되고 observer가 부분 publication한 뒤 실패할 수 있다.
  executor-none 경로도 begin/wait/shutdown 0 다음 동일 gap을 거친다. 그러므로 loader 전/중 cancel,
  loader failure, identity mismatch, constructor failure의 canonical tail이 문서와 catalog가 요구한
  exact `STOPPED -> release`가 아니다.
- prototype gap: Recovery Validation §10 toy teardown은 observer를 아예 모델링하지 않고 trace에
  `STOPPED`, `release`만 append한다. §6 prototype도 동일하다. 따라서 prototype exit 0은 normative
  ordering receipt가 아니며 teardown task creation/evaluation, shield 재취소, ordered aggregation도
  문서 스스로 project acceptance로 이월했다.
- preserved strengths: guard acquire 성공 직후 candidate/executor/grace가 fallible loader보다 먼저
  초기화된다. executor `None`이면 begin/wait/shutdown 0이고, executor가 존재하면 앞선 begin/wait
  오류나 cancellation과 무관하게 shutdown을 정확히 한 번 시도하도록 단계가 분리돼 있다. primary
  exception/cancellation identity 우선, secondary 발생 순서, primary가 없을 때 단일 원본 또는 ordered
  `ExceptionGroup` 정책도 설계 문구는 결정적이다. 결함은 그 정책 뒤 canonical durable tail의 순서다.
- exact required fix: 모든 fallible observer/snapshot/log/error aggregation work를 STOPPED 전에 끝내고,
  최종 non-throwing atomic STOPPED publication과 exact-owner guard release를 teardown의 마지막 두 durable
  actions로 만들라. release failure를 지원할 필요가 있다면 release primitive 자체가 owner token을
  원자적으로 해제한 뒤 diagnostic을 반환하도록 정의해 “attempt”와 “durable release”를 혼동하지
  않게 하라. executor-none/present, task-create failure inline fallback, cancellation at every shield
  boundary, begin/wait/shutdown/observer 단일·복합 오류 모두 exact tail과 release 후 reacquire를 project
  fake trace에서 검증해야 한다.

## 전체 이전 finding closure matrix

| Finding | Iteration 3 판정 | 근거 |
|---|---|---|
| M42-DR1-001 | CLOSED (design) | caller outcome/resource completion guard가 분리되고 abandoned resource 완료 전 slot을 유지한다. |
| M42-DR1-002 | CLOSED (design) | resource completion은 asyncio-free이고 loop notification은 별도 best-effort다. |
| M42-DR1-003 | CLOSED (design) | Future-return commit point와 direct/promotion submit rollback이 고정됐다. |
| M42-DR1-004 | CLOSED (design) | capacity edge timestamp/version이 probe 사이 clear/re-full을 보존한다. |
| M42-DR1-005 | CLOSED (design) | 호출별 remaining-budget Ollama client와 close seam이 고정됐다. |
| M42-DR1-006 | CLOSED (design) | accepted/terminal conservation, scrape snapshot gauge, sink 격리가 일치한다. |
| M42-DR1-007 | CLOSED (catalog only) | exact 11행과 runner/report mapping이 일치하며 아직 실행 receipt는 아니다. |
| M42-DR1-008 | OPEN via M42-RR3-001 | single owner/load는 해결됐지만 rejected attempt의 app mutation 0이 아직 normative teardown과 충돌한다. |
| M42-DR2-001 | OPEN via M42-RR3-001/002 | fail-soft import/load owner는 구체적이나 invalid diagnostic의 단일 authority와 teardown tail이 불완전하다. |
| M42-DR2-002 | CLOSED (design) | actual `http.disconnect` observation과 route marker를 shared arbiter에 연결한다. |
| M42-DR2-003 | CLOSED (design) | non-identity encoding early reject와 identity prefix accounting이 정직하다. |
| M42-DR2-004 | CLOSED (design) | polling 없는 single CAS drain winner와 absolute deadline scheduler가 구체적이다. |
| M42-DR2-005 | CLOSED (catalog only) | exact node collection/repeat/conservation/live separation이 일치한다. |
| M42-DR3-001 | OPEN via M42-RR3-001 | stale restoration은 제거됐으나 different identity full-attempt app delta 0은 아직 거짓이다. |
| M42-DR3-002 | CLOSED (design) | actual-app queued/running disconnect race와 outer observability 계약이 연결된다. |
| M42-DR3-003 | CLOSED (design) | wire-delivered/application-consumed byte accounting을 분리한다. |
| M42-DR3-004 | CLOSED (design) | completion/deadline/tie/stale/zero가 같은 lock/sequence/CAS를 쓴다. |
| M42-DR3-005 | CLOSED (catalog only) | literal profile/node inventory와 catalog가 byte-for-byte 일치한다. |
| M42-DR4-001 | CLOSED (design) | pure-ASGI wrapper는 proven disconnect와 erroneous no-response를 구별한다. |
| M42-DR4-002 | OPEN via M42-RR3-001/002 | single-active guard는 타당하지만 reject/full failure teardown의 mutation/tail 계약이 남았다. |
| M42-RR1-001 | OPEN via M42-RR3-001 | immutable first/same/different 정책은 맞지만 different attempt 전체 delta 0이 아니다. |
| M42-RR1-002 | OPEN via M42-RR3-002 | mandatory shutdown/primary policy는 맞지만 STOPPED→release canonical tail이 아니다. |
| M42-RR2-001 | OPEN (M42-RR3-001) | candidate commit ordering 수정만으로 teardown app mutation까지 제거되지 않았다. |
| M42-RR2-002 | OPEN (M42-RR3-002) | initialized locals 수정만으로 마지막 durable action 순서까지 닫히지 않았다. |

## Request terminal과 erroneous no-response 판정

이 계약은 설계 수준에서 sound하다. `scope["state"]["rag_terminal"] ==
"client_disconnected"` 또는 wrapped receive가 실제 관측한 `http.disconnect`만 proven disconnect다.
그 경우 내부 499-equivalent/`client_disconnected`, wire response frame 0이며 HTTP 499를 보내지 않는다.
증거 없이 downstream이 frame 0으로 정상 반환하면 고정
`RuntimeError("downstream_no_response")`, internal/500-equivalent, wire frame 0이다. outer cancellation은
관측된 disconnect가 있을 때만 disconnect로 분류하고 아니면 `cancelled`로 남긴다. result-first,
disconnect-first, tie, loser reap, receive 단일 owner와 request ID/log/metric exactly-once가 actual-app
project tests로 이월된 것도 명확하다. prototype의 옛 `frames==0` 추론은 closure evidence가 아니라고
Recovery Validation이 바로잡았다.

## Requirements와 acceptance traceability

| 대상 | 판정 | 근거/구현 진입 조건 |
|---|---|---|
| REQ-001 | FAIL | field/validator와 immutable identity는 구체적이나 different rejected attempt의 app delta 0이 아니다(M42-RR3-001). |
| REQ-002 | PASS (design) | bounded admission, FIFO, submit commit/rollback, future-owned slot이 닫혔다. |
| REQ-003 | PASS (design) | queue/execution budget과 actual-ASGI proven disconnect owner가 구체적이다. |
| REQ-004 | PASS (design) | orphan invariant, terminal conservation, bounded fixed errors가 닫혔다. |
| REQ-005 | FAIL | mandatory shutdown 시도는 있으나 STOPPED와 release가 마지막 두 durable action이 아니다(M42-RR3-002). |
| REQ-006 | PASS (design) | readiness precedence, edge debounce, bounded labels가 executable fixtures에 연결됐다. |
| REQ-007 | PASS (design) | encoding reject와 honest wire/application byte accounting이 구현 가능하다. |
| REQ-008 | PASS (design) | context-local deadline과 current runtime seam의 per-call adapters가 구체적이다. |
| REQ-009 | FAIL | module/CLI 공통 commit 방향은 맞지만 rejected/invalid startup publication contract가 모순된다. |
| deterministic 10 profiles | COMPLETE CATALOG, NOT RUN | ordered mapping과 project symbols는 exact하나 구현/실행 receipt가 없다. |
| Requirement §5 11행 | COMPLETE CATALOG, NOT RUN | deterministic 10행과 별도 opt-in live 1행으로 정확히 11행이다. |
| live 12/M3/M4.1 | SEPARATE | live 12, M3 14-gate artifact, `M4.1_BLOCKED`는 deterministic aggregate에 합성되지 않는다. |

11-profile catalog는 정확하다. deterministic 순서는 `event_loop`, `bounded_admission`,
`fifo_cancel`, `queue_timeout`, `execution_timeout`, `caller_cancellation`, `drain`,
`saturation_readiness`, `payload`, `normal_mock_load`이며 caller cancellation만 queued/running 두 exact
nodes를 가진다. profile/node별 repeat 10, conservation receipt 10, aggregate 밖 negative control 1,
별도 opt-in live 12의 계약도 서로 모순되지 않는다. 이는 prototype/설계 catalog이지 current product
implementation PASS receipt가 아니다.

## Prototype evidence 품질

Recovery Validation의 bounded commands는 **SOUND BUT LIMITED**다. 설치 Starlette 0.50.0의
`BaseHTTPMiddleware` erroneous no-response 실패를 재현하고 pure-ASGI 최소 wrapper, toy guard,
immutable identity, teardown error matrix, proven-disconnect classifier와 Iteration 3 startup ordering을
bounded 실행했다. 문서도 product code/test/config를 import하지 않은 inline prototype이라고 반복해
정직하게 한계를 밝혔다.

그러나 prototype은 구현 receipt가 아니다. 특히 Iteration 3 toy는 lifecycle/observer app mutation을
세지 않고, stopped observer를 모델링하지 않으며, executor-present mandatory shutdown은 별도 toy에서만
확인한다. teardown task creation/evaluation failure, shield cancellation 재전파, original exception
identity, ordered primary/secondary/`ExceptionGroup`, actual config facade/CLI/app/request stack도 실행하지
않았다. 따라서 prototype은 candidate/identity/local initialization 방향의 characterization일 뿐
M42-RR3-001/002를 폐쇄할 수 없다.

## Implementation entry conditions

1. M42-RR3-001을 수정해 different identity full attempt의 app/cache/config/engine/executor와 health/
   metric/log observer delta를 정확히 0으로 만들고, invalid loader만 하나의 atomic authoritative
   `settings_invalid` diagnostic을 publish하게 한다.
2. M42-RR3-002를 수정해 모든 fallible observer/aggregation을 canonical tail 앞에 두고, non-throwing
   STOPPED publication 뒤 exact-owner guard release를 마지막 두 durable actions로 고정한다.
3. executor-none은 begin/wait/shutdown 0, executor-present는 mandatory shutdown attempt 정확히 1이며,
   loader 전/중 cancel, loader/identity/engine/executor failure, task-create fallback과 모든 shield
   cancellation/error 조합이 STOPPED→release로 끝나는 project fake matrix를 추가한다.
4. primary exception/cancellation identity 우선, ordered bounded secondary log, primary가 없을 때 단일
   원본 또는 ordered `ExceptionGroup`을 actual coroutine/task boundaries에서 검증한다.
5. actual app에서 proven disconnect의 internal 499-equivalent/wire 0과 erroneous no-response의
   internal 500-equivalent/wire 0, request observability exactly once와 pending task 0을 검증한다.
6. exact 11-profile catalog를 구현하고 deterministic repeat-10/conservation, 별도 negative control,
   opt-in live 12/M3/M4.1 분리 receipt를 생성한다. prototype 결과를 product PASS로 승격하지 않는다.
7. 수정 설계의 fresh review가 CRITICAL 0, MAJOR 0, MINOR 최소, score 9.7 이상이어야 Phase 2 구현으로
   진입한다. `M4.1_BLOCKED`는 계속 독립 release blocker다.

사용자 결정이 필요한 새 제품 범위는 없다. 두 finding은 이미 승인된 zero-delta와 teardown ordering
계약을 normative pseudocode와 prototype/acceptance spies에 동일하게 반영하는 설계 완결성 문제다.
