# M4.2 Safe Serving Boundary — Design Review Iteration 4

검토 대상: [Requirement.md](Requirement.md), [Plan.md](Plan.md), [Design.md](Design.md),
[Design Review Iteration 1](Design_Review_Iteration_1.md),
[Design Review Iteration 2](Design_Review_Iteration_2.md),
[Design Review Iteration 3](Design_Review_Iteration_3.md),
[개발 orchestration guide](../../../milestone_dev_orchestration_guide.md), 기준 revision
`0c84795`의 repository code/tests/config 및 현재 설치된 Starlette `0.50.0` 동작.

## Executive Summary

**FAIL — 9.1/10.0.** Iteration 4는 wire-delivered/application-consumed payload를 구현
가능한 계층별 계약으로 분리했고, drain winner를 단일 lock/sequence/CAS로 결정하며, ordered
10-profile node inventory와 repeat/conservation/live 분리를 byte-for-byte 일치시켰다. 그러나
실제 설치 app에는 `RequestContextMiddleware(BaseHTTPMiddleware)`가 `/rag` route 바깥에 있어
disconnect winner의 의도적 send 0을 `RuntimeError("No response returned.")`로 바꾸며, process
Settings cache의 previous-value CAS rollback은 겹치는 lifespan 종료 순서에서 이미 중단된 app의
identity를 되살릴 수 있다. 따라서 실제 ASGI disconnect와 process cache ownership은 아직
exactly-once/identity-safe하지 않다.

| Severity | 수 |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 2 |
| MINOR | 0 |

Gate 조건 score >=9.7/10, CRITICAL=0, MAJOR=0, MINOR 최소화 중 score와 MAJOR 조건을
충족하지 못하므로 **Gate FAIL**이다. Phase 2 구현으로 진행하면 안 된다.

## Full closure matrix

| Finding | Iteration 4 판정 | exact evidence |
|---|---|---|
| M42-DR1-001 | CLOSED | `Design.md` §2.2/§2.6은 caller/resource guard를 분리하고 ABANDONED future 완료 전까지 running slot을 유지한다. |
| M42-DR1-002 | CLOSED | §2.7은 resource completion을 asyncio-free로 먼저 끝내고 loop notification을 별도 best-effort 단계로 둔다. |
| M42-DR1-003 | CLOSED | §2.4는 `pool.submit()` Future 반환을 commit point로 두고 direct/promotion failure rollback을 고정한다. |
| M42-DR1-004 | CLOSED | §2.3/§8.2는 capacity edge timestamp/version을 executor lock에서 기록해 probe 사이 clear/re-full을 보존한다. |
| M42-DR1-005 | CLOSED | §6.2는 router/answer 호출마다 remaining-budget `ollama.Client`를 새로 소유하고 transport를 닫는다. |
| M42-DR1-006 | CLOSED | §2.3/§9는 accepted/terminal 보존식, scrape-time gauge snapshot, metric side-effect 격리를 정의한다. |
| M42-DR1-007 | CLOSED | §10의 11 requirement rows, deterministic 10-profile inventory, repeat receipts, conservation, 별도 live/M3/M4.1 status가 일치한다. |
| M42-DR1-008 | OPEN (MAJOR) | 단일 load/fail-soft ceiling은 닫혔지만 cache rollback ownership이 overlapping lifespan에서 stale identity를 복원한다. M42-DR4-002. |
| M42-DR2-001 | OPEN (MAJOR) | import/invalid/CLI 단일 validation은 닫혔지만 process identity lifecycle이 완전히 닫히지 않았다. M42-DR4-002. |
| M42-DR2-002 | OPEN (MAJOR) | route 내부 owner는 구체화됐지만 실제 outer `BaseHTTPMiddleware`가 send-0 completion을 허용하지 않는다. M42-DR4-001. |
| M42-DR2-003 | CLOSED | §4.4는 non-identity encoding을 receive 0으로 거부하고 identity prefix만 `limit+1`까지 소비한다. |
| M42-DR2-004 | CLOSED | §2.9는 polling을 제거하고 injected deadline scheduler와 단일 CAS waiter를 사용한다. |
| M42-DR2-005 | CLOSED | §10은 queue release, virtual load, ASGI cancellation, live manifest/M3 artifact를 실행 symbol에 고정한다. |
| M42-DR3-001 | OPEN (MAJOR) | load→cache→engine 동일 identity는 해결됐으나 previous-cache 복원의 lifetime/lease가 없다. M42-DR4-002. |
| M42-DR3-002 | OPEN (MAJOR) | `RagASGIRoute.handle`의 local control flow는 닫혔지만 installed middleware까지 포함한 send ownership은 닫히지 않았다. M42-DR4-001. |
| M42-DR3-003 | CLOSED | §4.4와 payload catalog는 wire-delivered와 application-consumed를 분리하고 가능한 assertion만 한다. |
| M42-DR3-004 | CLOSED | §2.9는 resource/deadline 양쪽이 동일 sequence/CAS winner를 claim하고 absolute deadline을 waiter가 보존한다. |
| M42-DR3-005 | CLOSED | literal mapping과 catalog의 두 cancellation nodes, 각 node/profile 10 receipts, 별도 negative/live receipt가 일치한다. |

## Findings

### M42-DR4-001 — MAJOR — 실제 설치 middleware가 disconnect winner의 send 0 계약을 거부한다

- 위치: `Design.md` §0/§4.1/§4.4; 현재
  `src/simple_qna_rag/observability/request_context.py::RequestContextMiddleware`,
  `src/simple_qna_rag/web/server.py::create_app`; M4.2-REQ-003.3/003.4, REQ-009.2.
- exact evidence: 설계의 실제 stack은 outer `BodyLimitMiddleware` →
  `RequestContextMiddleware` → `RagASGIRoute`다. 현재 request-context middleware는 설치된
  Starlette `0.50.0`의 `BaseHTTPMiddleware` subclass다. 그 `call_next()`는 downstream이
  반환하기 전에 첫 `http.response.start`를 memory stream에서 읽으며, downstream route가
  아무 frame도 보내지 않고 반환하면 `anyio.EndOfStream`을 받아 정확히
  `RuntimeError("No response returned.")`를 발생시킨다. 따라서 §4.1 step 5의
  `RagASGIRoute` send 0 정상 반환은 실제 `create_app()` 전체에서는 정상 terminal이 아니고,
  existing middleware의 request-end도 기본 status 500으로 분류된다.
- 재현 sequence: body 완료 → ticket admission → `http.disconnect`가 arbiter 승자 → route가
  executor cancel 및 child reap 후 frame 0으로 반환 → `BaseHTTPMiddleware.call_next()`의
  receive stream EOF → RuntimeError. 이 경로는 §4.1 integration assertion의 “actual app,
  frames `[]`, pending 0”을 예외 없는 성공으로 만족할 수 없으며 M4.1 logging/metric schema에도
  허위 500을 남긴다.
- exact required fix: request-context 경계를 pure ASGI middleware로 바꾸어 downstream send 0을
  명시적 disconnect terminal로 받아들이고 request-end/outcome을 bounded enum으로 기록하거나,
  route보다 바깥의 단일 pure-ASGI owner가 disconnect/no-send를 소유하도록 stack을 다시 고정하라.
  실제 `create_app()` 전체 middleware stack을 호출해 disconnect-first/result-first/tie에서
  exception 0, disconnect frames 0, result response 1, receive max 1, request log/metric 정확히 1,
  pending 0을 검사해야 한다. Starlette `BaseHTTPMiddleware`를 그대로 둔 채 route-unit trace만
  통과시키는 것은 closure evidence가 아니다.

### M42-DR4-002 — MAJOR — previous-value CAS rollback은 overlapping lifespan에서 stale Settings identity를 부활시킨다

- 위치: `Design.md` §1/§4.3; 현재 `src/simple_qna_rag/settings.py` cache seam;
  M4.2-REQ-001.2/001.3, REQ-005.1, REQ-009.2.
- exact evidence: 설계는 successful lifespan마다 `previous_cache`를 snapshot하고 process cache를
  `s`로 덮은 뒤 종료 때 identity CAS `s -> previous_cache`를 수행한다. 단일 owner와 중간
  competing writer에는 안전하지만 ownership generation/lease가 없어서 겹치는 app lifetimes의
  비-LIFO 종료를 처리하지 못한다.
- 재현 sequence: cache=`x` → app A가 `sA` commit(previous=`x`) → app B가 `sB`
  commit(previous=`sA`) → A가 먼저 종료해 CAS(`sA`,`x`) 실패, A는 중단됨 → B가 종료해
  CAS(`sB`,`sA`) 성공. 최종 process cache는 더 이상 살아 있는 owner가 없는 stale `sA`다.
  이후 `config.get_settings()`는 재검증 없이 중단된 app A identity를 반환한다. 설계가 주장하는
  “같은 process에서 app factory를 여러 번 쓰는 경우의 복원”과 cache/limiter/executor/engine
  identity 보존을 위반한다.
- exact required fix: cache write를 opaque generation/token이 있는 lease로 만들고 release가 현재
  head뿐 아니라 이미 종료된 predecessor를 건너뛰어 유효 predecessor를 복원하도록 하거나,
  lifespan이 process-global cache를 임시 override/복원하지 않는 구조로 바꿔라. `get/set/peek/CAS`
  모든 접근은 같은 lock/token protocol을 사용해야 한다. A/B의 LIFO와 non-LIFO 종료, engine
  failure, concurrent writer, `previous is s`, CLI preflight를 포함해 final cache identity,
  `Settings.from_sources` count, engine `is s`, stale identity 0을 executable matrix로 고정하라.

## Requirement traceability verdict

| REQ | 판정 | 근거 |
|---|---|---|
| 001 | 실패 | 8개 field/validator와 single validation은 구체적이나 process cache rollback identity가 overlapping lifespan에서 깨진다(M42-DR4-002). |
| 002 | 충족 | bounded atomic admission, explicit FIFO, submit commit/rollback, resource slot ownership이 폐쇄됐다. |
| 003 | 실패 | executor/route 내부 race는 구체적이나 installed middleware가 disconnect send-0 terminal을 예외로 바꾼다(M42-DR4-001). |
| 004 | 충족 | orphan invariant, terminal conservation, fixed safe error body가 닫혔다. |
| 005 | 실패 | drain CAS 자체는 충족하지만 process lifecycle shutdown이 stale Settings cache identity를 남길 수 있다(M42-DR4-002). |
| 006 | 충족 | edge debounce, bounded labels/series, scrape-time gauges가 구체적이다. |
| 007 | 충족 | non-identity fail-closed와 identity wire/application accounting이 구현 가능한 수준으로 일치한다. |
| 008 | 충족(설계 수준) | remaining-budget per-call clients, context deadline, DDGS stall/orphan 경계가 설치 API에 맞다. |
| 009 | 실패 | actual disconnect가 M4.1 request middleware에서 RuntimeError/500이 되고 cache facade identity도 stale해질 수 있다. |
| 정량 Gate | 실패 | ordered 10-profile/repeat/live inventory는 닫혔지만 caller-cancellation nodes가 실제 installed app에서 요구 assertion을 달성할 수 없다. |

`M4.1_BLOCKED`는 deterministic M4.2, live 12, M3 artifact와 분리된 독립 상태로 유지됐다.
M4.3/M5 범위 침범이나 새 사용자 제품 결정은 발견하지 않았다.

## Acceptance inventory verification

`PROFILE_NODE_IDS`의 deterministic inventory는 정확히 10개 profile이며 insertion order는
`event_loop`, `bounded_admission`, `fifo_cancel`, `queue_timeout`, `execution_timeout`,
`caller_cancellation`, `drain`, `saturation_readiness`, `payload`, `normal_mock_load`다. literal
tuple과 §10 catalog node string은 byte-for-byte 동일하고 caller cancellation만 queued/running
두 node를 갖는다. repeat receipt는 profile 및 각 node별 정확히 10개, conservation receipt
정확히 10개이고 negative control은 aggregate 밖 1개다. opt-in live 12-case manifest,
M3 14-gate artifact, `M4.1_BLOCKED`는 deterministic aggregate와 서로 합성되지 않는다.

Payload Gate의 achievable 계약도 일치한다. ASGI application이 이미 전달된 single
`limit+N` message의 wire 크기를 되돌릴 수 없으므로 `wire_delivered_bytes=limit+N`을 별도로
기록하고 downstream/application consumption만 `limit+1`로 slice하며 이후 receive를 0회로
막는다. 이는 strict wire cap을 허위 주장하지 않으면서 413/admission 0을 검증한다.

Drain Gate는 resource-zero와 deadline callback이 동일 executor lock, monotonic sequence,
winner CAS를 사용하고 waiter가 absolute deadline을 소유한다. completion-first,
deadline-first, tie 양 insertion order, stale transition, running 상태의 zero-grace에서 첫 CAS만
결과를 소유하며 shutdown guard가 pool shutdown을 정확히 한 번 호출한다. Requirement의
`running==0` 즉시 drained 계약과 충돌하지 않도록 zero-grace fixture는 running residual을
명시해야 한다.

## Iteration 5 extension eligibility

| Guide condition | 판정 | 근거 |
|---|---|---|
| CRITICAL=0 | 충족 | 0건 |
| score>=9.0 | 충족 | 9.1 |
| MAJOR<=2 | 충족 | 2건 |
| 이전 iteration 대비 실질 개선 | 충족 | 7.8→9.1, MAJOR 5→2; payload/drain/inventory 세 묶음 폐쇄 |
| 남은 문제가 구체적·해결 가능 | 충족 | pure-ASGI outer ownership과 generation-aware cache lease로 범위가 한정됨 |
| 동일 근본 문제 2회 연속 재발 없음 | **불충족** | ASGI end-to-end ownership은 M42-DR2-002→M42-DR3-002→M42-DR4-001, cache identity ownership은 M42-DR3-001→M42-DR4-002로 연속 재발 |

정량 연장 조건만 보면 Iteration 5 후보지만, guide의 조기 중단 규칙인 동일 근본 문제의
2회 연속 재발에 해당하므로 **Iteration 5 연장은 부적격**으로 판정한다. 재개하려면 단순 문구
보강이 아니라 installed pure-ASGI middleware stack의 executable trace와 generation-aware cache
lease/rollback state machine을 먼저 재설계하고, coordinator가 guide상 별도 재개를 승인해야 한다.

## Validation evidence

- 현재 `RequestContextMiddleware`와 설치 Starlette `0.50.0`
  `BaseHTTPMiddleware.__call__/call_next`를 대조해 no-response EOF의 exact RuntimeError 경로를
  확인했다.
- settings cache의 load/commit/engine import/engine failure/shutdown을 단일 owner, competing
  writer, overlapping A/B non-LIFO 종료 순서로 재연산했다.
- Requirement 9개와 §5의 11 rows, deterministic 10-profile mapping, repeat/node/conservation,
  live/M3/M4.1 분리를 Design symbol/assert/report field에 대조했다.
- 문서 작성 후 markdown link validation과 `git diff --check`를 실행한다.
