# M4.2 Safe Serving Boundary 요구사항

상위 결정: [M4 복구 결정](../m4-production-readiness/Recovery_Decision.md)

선행 예외: [M4.1 운영 acceptance 예외](../m4.1-configuration-observability/Operational_Acceptance_Exception.md)

기준 revision: `0c84795` (`master`, 2026-08-10)

## 1. 목적과 선행 위험

M4.2는 현재 FastAPI `/rag` handler가 event loop에서 동기
`simple_qna_rag.agent.route_query()` 전체를 직접 실행하는 경계를 안전한 단일 프로세스
serving 경계로 바꾼다. 수락, 대기, 실행, 포기, drain을 유한 상태와 유한 자원으로
표현하고, 입력과 외부 네트워크 호출에도 한도를 적용한다.

M4.1 post-merge Operational Acceptance는 **미완료**다. live 14-gate aggregate와
정상 `m4-regression-report` receipt가 없으며 이는 M4 release blocker다. 사용자의 예외는
M4.2 착수만 허용했을 뿐 M4.1 PASS나 품질 보존을 뜻하지 않는다. M4.2의 clean 검증과
post-merge acceptance는 이 부채를 숨기거나 대체할 수 없고, 최종 traceability에는
`M4.1_BLOCKED` 선행 위험으로 남겨야 한다.

## 2. 범위

포함 범위는 blocking `route_query` offload, bounded admission/FIFO queue,
queue/execution timeout, caller cancellation, 실행 thread orphan accounting, graceful
drain/shutdown, saturation readiness reason, payload/input/network boundary, 결정론적
concurrency/load acceptance다.

다음은 제외한다.

- M4.3: index build/activate/rollback lifecycle, index format/provenance 변경, container와
  deployment assembly
- M5: 외부·분산 queue, 다중 프로세스 조정, autoscaling/Kubernetes
- 인증·사용자별 quota, RAG/routing 알고리즘과 모델 품질 tuning
- Python thread의 강제 종료 또는 process hard-timeout 보장

## 3. 현재 코드 기준

- `web/server.py::rag_query()`는 `async def` 안에서 동기
  `agent.route_query(question, metrics_registry=...)`를 직접 호출한다.
- lifespan은 settings/engine 초기화만 하고 shutdown/drain 상태가 없다.
- `settings.py::FIELD_SPECS`는 정확히 41개이며 concurrency, queue, request timeout,
  body/question 한도, shutdown grace 설정이 없다.
- `observability.health.evaluate_readiness()`는 세 오류만 입력받고,
  `metrics.py`에는 queue/running/orphan 계열이 없다.
- `RAGEngine`, agent router와 config facade는 process singleton/global 상태를 사용한다.
  concurrency=2를 안전하다고 가정할 근거가 없으므로 기본 실행 동시성은 1이다.

## 4. 기능 요구사항

### M4.2-REQ-001 — fail-closed serving settings

1. `FIELD_SPECS` 단일 원본에 아래 설정을 추가하고 `Settings.from_sources()`와
   `--check-config` 계약을 보존한다.

   | 설정 | 기본값 | 검증 |
   |---|---:|---|
   | `QUERY_CONCURRENCY_LIMIT` | 1 | 정수, `1..2` |
   | `QUERY_QUEUE_LIMIT` | 4 | 정수, `0..64` |
   | `QUERY_QUEUE_TIMEOUT_SECONDS` | 5.0 | 유한 실수, `0 < x <= 30` |
   | `QUERY_EXECUTION_TIMEOUT_SECONDS` | 90.0 | 유한 실수, `1 <= x <= 600` |
   | `SHUTDOWN_GRACE_SECONDS` | 30.0 | 유한 실수, `0 <= x <= 120` |
   | `MAX_REQUEST_BODY_BYTES` | 16,384 | 정수, `256..1,048,576` |
   | `MAX_QUESTION_CHARS` | 4,000 | 정수, `1..32,000` |
   | `UPSTREAM_CONNECT_TIMEOUT_SECONDS` | 5.0 | 유한 실수, `0 < x <= 30` |

2. `QUERY_QUEUE_TIMEOUT_SECONDS < QUERY_EXECUTION_TIMEOUT_SECONDS`가 아니면 전체 설정
   로드를 exit 2로 거부한다. 알 수 없는 env key, NaN/무한대, 범위 밖 값은 기본값으로
   대체하지 않는다.
3. 새 설정은 secret-safe field inventory와 config facade 호환성 테스트에 포함한다.

### M4.2-REQ-002 — bounded admission과 FIFO 상태 머신

1. 새 `web/concurrency.py::QueryExecutor`가 유일한 admission 소유자다. 외부 공개
   동작은 `submit(callable)`, `begin_drain()`, `wait_drained()`, `shutdown()`과 read-only
   snapshot으로 한정한다.
2. ticket 상태는 `QUEUED -> RUNNING -> DONE` 또는 `QUEUED -> REJECTED|TIMED_OUT|CANCELLED`,
   `RUNNING -> DONE|ABANDONED`만 허용한다. 각 ticket은 정확히 한 terminal response와
   정확히 한 resource finalize를 갖는다.
3. running은 `QUERY_CONCURRENCY_LIMIT`, queued는 `QUERY_QUEUE_LIMIT`를 절대 넘지 않는다.
   두 한도의 확인과 ticket 삽입은 event-loop 위 단일 critical section에서 원자적이어야
   한다. queue가 가득 차거나 draining이면 thread/future를 만들지 않고 거부한다.
4. queue는 명시적 ticket sequence의 FIFO다. 선두 취소/timeout은 제거되고 다음 살아
   있는 ticket 하나만 승격된다. private semaphore wake 순서에 의존하지 않는다.
5. `ThreadPoolExecutor(max_workers=QUERY_CONCURRENCY_LIMIT)`에서 **전체**
   `agent.route_query()`를 offload한다. endpoint, readiness와 metrics handler는 executor
   작업 완료를 기다리는 동안 event loop를 점유하지 않는다.

### M4.2-REQ-003 — 분리된 timeout과 caller cancellation

1. queue timeout은 admission 시 monotonic absolute deadline으로 시작하고 실행 시작 시
   종료한다. 만료된 queued ticket은 `queue_timeout`이다.
2. execution timeout은 worker future가 실제 제출된 시점부터 별도 monotonic deadline을
   사용한다. 만료된 running ticket은 caller 관점 `execution_timeout`, 내부 상태
   `ABANDONED`다. queue 대기 시간이 execution budget을 소비하지 않는다.
3. HTTP caller cancellation은 queued면 ticket을 제거하고 capacity를 즉시 회수한다.
   running이면 응답을 쓰지 않고 `ABANDONED`로 표시하되 running slot은 underlying future
   완료 전까지 회수하지 않는다.
4. timeout/cancel과 future completion race는 단일 finalize 경로로 합류한다. slot 음수,
   이중 응답, 이중 metric, 다음 ticket 이중 승격은 허용하지 않는다.

### M4.2-REQ-004 — orphan 회계와 overload 오류 계약

1. orphan은 caller가 execution timeout/cancel로 떠났지만 worker future가 아직 끝나지
   않은 `ABANDONED` ticket이다. orphan은 running에도 포함되며 완료 callback에서만 둘 다
   감소한다. queued timeout/cancel은 orphan이 아니다.
2. snapshot 불변식은 `0 <= orphaned <= running <= concurrency_limit`와
   `0 <= queued <= queue_limit`이다. accepted/rejected/queue-timeout/execution-timeout/
   cancelled/completed counter로 모든 ticket을 분류한다.
3. `/rag` 오류 body는 항상
   `{"success":false,"answer":"<고정 안전 문구>","sources":[],"search_type":"unknown","error":{"code":"<enum>","retryable":<bool>}}`
   형태이며 질문, upstream body, 예외 문자열, 절대 경로를 포함하지 않는다.
4. 상태 계약은 다음과 같다.

   | code | HTTP | retryable | 조건 |
   |---|---:|---|---|
   | `invalid_request` | 400 | false | JSON/schema/question 검증 실패 |
   | `payload_too_large` | 413 | false | body byte 한도 초과 |
   | `not_ready` | 503 | true | startup 실패 또는 draining |
   | `overloaded` | 503 | true | queue 가득 참 |
   | `queue_timeout` | 503 | true | queue deadline 만료 |
   | `execution_timeout` | 504 | true | 실행 deadline 만료 |
   | `internal` | 500 | false | 그 밖의 안전하게 정규화된 실패 |

### M4.2-REQ-005 — graceful drain과 shutdown

1. lifecycle은 `STARTING -> READY -> DRAINING -> STOPPED`다. startup 실패는
   `STARTING`에 머물며 readiness 503이다. shutdown 시작은 `begin_drain()`을 정확히
   한 번 호출한다.
2. DRAINING 진입 뒤 신규 admission은 `not_ready`로 즉시 거부한다. 아직 실행되지 않은
   queued ticket도 `not_ready`로 깨우고 queue를 0으로 만든다.
3. 이미 running인 future는 grace 동안 완료할 수 있다. `running==0`이면 즉시 drained다.
   grace 만료 뒤 `shutdown(wait=False, cancel_futures=True)`를 호출하고 STOPPED로 전이한다.
4. Python은 이미 실행 중인 thread를 중단할 수 없으므로 grace는 process 종료 시간의
   hard bound가 아니다. 남은 orphan/running 수를 고정 필드로 기록하고 readiness는 계속
   503이어야 한다. 이를 성공적 drain으로 보고하지 않는다.

### M4.2-REQ-006 — readiness와 bounded metrics

1. 기존 readiness 우선순위 뒤에 lifecycle/saturation을 명시적으로 추가한다. startup
   오류가 없을 때 `DRAINING|STOPPED -> draining`, `orphaned == concurrency_limit ->
   orphan_workers`, `running == concurrency_limit && queued == queue_limit ->
   queue_saturated`, 그 밖은 `ok`다.
2. `queue_saturated`는 순간적인 한 번의 full 관찰로 readiness를 흔들지 않도록 연속
   1.0초 full일 때 503, full 해제 후 연속 1.0초 뒤 200으로 전이한다. fake monotonic
   clock으로 검증한다. admission 자체의 overload 거부는 이 debounce와 무관하게 즉시다.
3. `rag_readiness` allowlist에 `draining`, `orphan_workers`, `queue_saturated`를 추가하고
   기존 reason을 보존한다. queue depth, running, orphan gauge와 결과 enum counter를
   무라벨 또는 고정 enum label로 추가한다. 질문/request-id/thread-id를 label로 쓰지 않는다.
4. 1,000개 고유 질문 후 time series 수는 fresh registry의 이론 상한과 같고 질문 수에
   따라 증가하지 않아야 한다.

### M4.2-REQ-007 — payload와 입력 경계

1. ASGI receive 단계에서 decompressed HTTP body 누적 byte를 세어
   `MAX_REQUEST_BODY_BYTES+1`에서 더 읽지 않고 413으로 종료한다. `Content-Length`가
   없거나 거짓이어도 같은 한도가 적용되며, 선언값이 한도를 넘으면 body를 읽기 전에
   거부한다.
2. media type은 `application/json`만 허용한다. JSON object의 유일한 key는 `question`이고
   값은 string이어야 한다. trim 후 빈 문자열, NUL, Unicode control category `Cc`/`Cs`,
   `MAX_QUESTION_CHARS` 초과는 400이다. 정상 질문 원문은 응답 answer 생성 외 로그/metric/
   오류에 복제하지 않는다.
3. validation 실패는 admission 전에 끝나며 queue/running counter를 변경하지 않는다.

### M4.2-REQ-008 — 외부 네트워크 경계

1. Ollama router/answer client와 DDGS 생성에 유한 timeout을 명시한다. Ollama connect는
   `UPSTREAM_CONNECT_TIMEOUT_SECONDS`, 전체 호출 budget은 남은 execution deadline을
   넘지 않는다. 새 자동 retry는 0회다.
2. deadline은 context-local 값으로 `rag_query -> QueryExecutor worker -> route_query`에
   전달한다. 전역 singleton client를 요청마다 mutate하지 않는다. 현재 LangChain client가
   per-call deadline을 보장하지 못하면 concurrency=1을 유지하고, 설계 리뷰 Gate에서
   검증 가능한 adapter 또는 process 경계를 선택하기 전 concurrency=2를 허용하지 않는다.
3. DDGS/library 내부 thread가 timeout을 무시할 가능성을 결정론적 stall fake로 검사한다.
   호출자가 deadline 안에 반환하지 못하는 구현은 수락하지 않으며 실제 thread 강제 종료는
   요구하지 않는다(REQ-003/004 orphan 계약 적용).

### M4.2-REQ-009 — 호환성과 품질 보호

1. CLI의 동기 `route_query()` API와 정상 `/rag` `QueryResponse` 성공 body를 보존한다.
2. `/health/live`, deprecated `/health`, 기존 logging schema와 M4.1 metric family를
   깨지 않는다. `/health/ready`에는 새 reason만 additive하게 추가한다.
3. 기존 Python/Node 전체 suite와 M3 회귀 wrapper를 통과해야 한다. 단, M4.1 live
   acceptance 미완료를 PASS로 재분류하거나 그 receipt를 합성해서는 안 된다.

## 5. 정량 수용 기준

모든 concurrency 테스트는 `threading.Event`/barrier와 fake monotonic clock을 사용하며
wall-clock sleep 기반 순서 판정은 금지한다.

| Gate | 고정 profile | PASS 조건 |
|---|---|---|
| event loop | 2초 barrier fake 1건 실행 중 live 20회 | `/health/live` 20/20, p95 <=100ms, max <=250ms |
| bounded admission | concurrency=1, queue=2, 동시 5건 | max running=1, max queued=2, overloaded=2, capacity 초과 0 |
| FIFO/cancel | running 1 + queued A/B, A 취소 | 실행 순서 running,B; queue=0; orphan=0; finalize 각 1회 |
| queue timeout | fake clock, queued 2건 중 선두 만료 | 선두만 503 `queue_timeout`, 다음 ticket 승격, worker 제출 1회 |
| execution timeout | running stall 1건 | 504 <= configured deadline+100ms, orphan=running=1 until release, release 뒤 둘 다 0 |
| caller cancellation | queued와 running 각각 100회 race 반복 | 이중 응답/finalize 0, 음수 counter 0, 종료 뒤 queue/running/orphan 0 |
| drain | running stall + queued 2 | queued 즉시 `not_ready`, 신규 즉시 거부, release 시 grace 내 STOPPED; 별도 grace 만료 case는 bounded return |
| saturation readiness | fake clock | 1.0초 전 ok, 1.0초 후 `queue_saturated`; 해제 1.0초 후 ok |
| payload | chunked/no-length/거짓 length/oversize | 초과 모두 413, executor submit 0, 수신 byte <= limit+1 |
| 정상 mock load | 200ms fake 40건, concurrency=1, queue=64 | success 40, max running=1, 유실 0, wall time <= single-thread 예상치 +20% |
| opt-in live | 고정 12 case, concurrency=1, 동일 host/warm profile | accepted 12/12, 5xx/timeout 0, M3 14-gate 결과 별도 기록; M4.1 blocker는 독립 유지 |

전체 deterministic suite는 같은 seed/profile로 10회 반복해 flake 0이어야 한다. 각 run의
총 요청 수는 terminal 결과 enum 합계와 정확히 같아야 하고 미분류 상태는 실패다.

## 6. 완료와 traceability

각 요구사항은 상세 설계 section, 구현 symbol, positive/negative/race test, clean 실행
receipt에 연결돼야 한다. 설계 리뷰와 독립 code review의 MAJOR/MINOR가 모두 폐쇄되고,
clean 환경 검증 및 GitHub PR/merge 후 acceptance가 끝나야 M4.2를 완료로 판정한다.
M4.1 운영 acceptance는 별도 행에서 계속 `BLOCKED`; 해소 또는 별도 M4 release-risk
승인 전에는 M4 전체를 release-ready로 판정할 수 없다.
