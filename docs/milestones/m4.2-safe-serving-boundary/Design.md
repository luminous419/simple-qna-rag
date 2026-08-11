# M4.2 Safe Serving Boundary 상세 설계

요구사항: [Requirement.md](Requirement.md) · 계획: [Plan.md](Plan.md)
선행 예외: [M4.1 Operational Acceptance Exception](../m4.1-configuration-observability/Operational_Acceptance_Exception.md) — **BLOCKED, 이 설계로 해소되지 않음**

기준 revision: `0c84795` (`master`). 표기 규칙: **[CURRENT]** = 이미 존재하는 symbol(수정 여부는 개별 명시), **[NEW]** = 이 설계가 도입하는 symbol. 범위는 Requirement §2를 그대로 따른다 — index lifecycle(M4.3), 외부/분산 queue·다중 프로세스·autoscaling(M5), 인증/quota/모델 품질, 강제 thread 종료는 이 설계에 없다. 기본 `QUERY_CONCURRENCY_LIMIT=1`을 설계 전제로 삼으며, concurrency=2는 §11에서 명시적으로 **미승인**으로 남긴다.

복구 결정(2026-08-10): [Design Recovery Validation](Design_Recovery_Validation.md)의
설치 환경 characterization을 근거로 (1) client disconnect를 응답 frame이 없는 정상 terminal
`client_disconnected`(내부 status-equivalent `499`)로 분류하고, (2) process마다 active app
lifespan을 정확히 하나만 허용한다. 이 결정은 M42-DR4-001/002를 복잡한 route-side
workaround나 overlapping-lifespan cache generation/lease 없이 닫는다. `499`는 관측용 내부
등가값일 뿐 연결이 끊어진 뒤 HTTP response로 전송하지 않는다.

## 0. 아키텍처 개요

```
HTTP caller
  -> BodyLimitMiddleware [NEW]           (ASGI 최외곽, byte 누적 한도)
  -> RequestContextMiddleware [CURRENT, REWRITE: pure ASGI]
                                           (request_id, terminal log/metric exactly once)
  -> RagASGIRoute [NEW]                  (body 단일 receive -> result/disconnect race -> conditional send)
  -> rag_query() [CURRENT, REWRITE]      (JSON/schema 검증 -> QueryExecutor.submit)
       -> QueryExecutor [NEW]            (admission, FIFO, timeout, cancel, orphan)
            -> ThreadPoolExecutor(max_workers=concurrency_limit)
                 -> _run_ticket()        (Deadline contextvar bind -> callable_())
                      -> agent.route_query() [CURRENT]
                           -> web_search.search_web() [CURRENT, MODIFIED]  (DDGS)
                           -> rag_engine.RAGEngine.query() [CURRENT, MODIFIED]  (Ollama)
```

세 개의 독립된 시간 축이 있다: (1) queue deadline — admission부터 실행 시작까지, (2) execution deadline — 실행 시작부터 완료/포기까지, (3) HTTP caller의 생존(연결) — 위 둘과 별도로 언제든 끊길 수 있다. 이 설계의 핵심은 세 축이 어떤 순서로 끝나도 정확히 한 번의 finalize만 일어나게 만드는 것이다(§2.6).

## 1. Symbol Inventory

| 모듈 | Symbol | 상태 |
|---|---|---|
| `web/concurrency.py` | `TicketState`, `_Ticket`, `TicketHandle`, `QueryExecutor`, `ExecutorSnapshot`, `AdmissionRejected`, `QueueTimeoutError`, `ExecutionTimeoutError` | NEW |
| `web/errors.py` | `ApiError`, `ERRORS`, `error_response()` | NEW |
| `web/body_limit.py` | `BodyLimitMiddleware`, `_content_encoding`, `BOOTSTRAP_BODY_CEILING` | NEW |
| `web/disconnect.py` | `DisconnectObserver`, `ResultOrDisconnect`, `RaceArbiter`, `race_result_or_disconnect`, `serve_rag_scope`, `RagASGIRoute` | NEW |
| `web/scheduling.py` | `DeadlineScheduler`, `AsyncioDeadlineScheduler`, `ManualDeadlineScheduler` | NEW |
| `observability/deadline.py` | `Deadline`, `current_deadline()`, `bind_deadline()` | NEW |
| `web/server.py` | `rag_query` | CURRENT, REWRITE |
| `web/server.py` | `_SingleActiveLifespanGuard`, `_LifespanLease`, `_make_lifespan`, `create_app`, `app`, `start_server` | NEW/CURRENT, MODIFIED (process당 active lifespan 정확히 1개, fail-soft single-load, drain wiring, CLI override) |
| `settings.py` | `FIELD_SPECS`(41→49), `MODEL_VALIDATORS`(+1), `_make_range_validator` | CURRENT, MODIFIED; overlapping-lifespan generation/lease/cache rollback seam은 도입하지 않음 |
| `observability/request_context.py` | `RequestContextMiddleware` | CURRENT, REWRITE (`BaseHTTPMiddleware` 제거, pure ASGI terminal owner) |
| `observability/health.py` | `evaluate_readiness` | CURRENT, MODIFIED (signature 확장) |
| `observability/health.py` | `SaturationDebounce` | NEW |
| `observability/metrics.py` | `build_metrics_registry`, `READINESS_REASONS` | CURRENT, MODIFIED (additive) |
| `observability/metrics.py` | `record_ticket_outcome`, `record_input_rejected`, `sync_executor_gauges`, `clamp_ticket_outcome` | NEW |
| `rag_engine.py` | `RAGEngine._initialize_llm`, `RAGEngine.query`(generate 단계) | CURRENT, MODIFIED |
| `agent.py` | `_get_router_llm` | CURRENT, MODIFIED |
| `web_search.py` | `search_web` | CURRENT, MODIFIED |

## 2. `web/concurrency.py` — QueryExecutor

### 2.1 상태 머신

```
TicketState = Enum("TicketState", ["QUEUED", "RUNNING", "DONE",
                                    "REJECTED", "TIMED_OUT", "CANCELLED", "ABANDONED"])
```

허용 전이(Requirement §4 REQ-002.2와 동일): `QUEUED -> RUNNING -> DONE`, `QUEUED -> {REJECTED, TIMED_OUT, CANCELLED}`, `RUNNING -> {DONE, ABANDONED}`. `TIMED_OUT`은 queue deadline 만료 전용, execution timeout은 `ABANDONED`(caller 관점 `execution_timeout`)다. `REJECTED`는 admission 거부(`overloaded`)와 draining 중 queue 축출(`not_ready`), queued promotion submit 실패를 포괄하되, 원인은 `_Ticket.reject_reason: Literal["overloaded","not_ready","submit_failed"]`로만 구분한다 — 상태 개수를 늘리지 않는다.

`RUNNING -> ABANDONED`는 **caller 쪽 종료일 뿐 resource 쪽 종료가 아니다** — `ticket.state == ABANDONED`인 동안에도 그 ticket은 여전히 `running`에 포함된 slot을 점유하며, `pool_future`는 계속 실행 중이다. §2.2/§2.6이 이 두 종료를 별개의 guard로 명시적으로 분리한다(M42-DR1-001).

### 2.2 `_Ticket` — caller-outcome guard와 resource-completion guard의 분리

```python
@dataclass
class _Ticket:
    ticket_id: int
    state: TicketState
    queue_deadline: float | None          # monotonic, QUEUED에서만 유효
    execution_deadline: float | None       # monotonic, RUNNING 진입 시 설정
    loop_future: asyncio.Future            # rag_query가 await하는 대상
    queue_timer: asyncio.TimerHandle | None = None
    exec_timer: asyncio.TimerHandle | None = None
    pool_future: "concurrent.futures.Future | None" = None
    reject_reason: str | None = None
    caller_finalized: bool = False         # exactly-once: caller에게 전달될 terminal outcome
    resource_finalized: bool = False       # exactly-once: running/orphaned slot 반환 + 승격
```

**두 guard는 서로 다른 것을 소유한다.** `caller_finalized`는 "이 ticket의 terminal outcome이 정확히 한 번 정해지고 caller가 정확히 한 번 그 결과를 받는다(혹은 받을 필요가 없어진다)"를 보장한다 — `TIMED_OUT`/`CANCELLED`(QUEUED 기원)/`REJECTED`/`ABANDONED`/`DONE` 다섯 상태 전부가 이 guard 하나로 exactly-once를 얻는다. `resource_finalized`는 오직 **RUNNING까지 진행했던 ticket**에만 존재하며 "`running`/`orphaned` 카운터가 정확히 한 번 감소하고 다음 승격이 정확히 한 번 일어난다"만 보장한다 — QUEUED에서 끝난 ticket은 애초에 `running`을 점유한 적이 없으므로 `resource_finalized`가 의미를 갖지 않는다(생성되지 않은 pool future에 대한 resource-finalize는 없다).

`ABANDONED`는 `caller_finalized=True`가 되는 시점에 `resource_finalized`가 아직 `False`일 수 있는 **유일한 상태**다 — timeout/cancel이 caller 쪽을 먼저 닫아도 pool future가 실제로 끝날 때까지 `running`/slot은 유지된다(REQ-003.3). `DONE`은 정상적으로는 `resource_finalized`가 먼저(또는 동시에) `True`가 되면서 `caller_finalized`도 그 자리에서 함께 `True`가 되는 상태다(§2.6 Order B). 두 guard 모두 `_lock` 보유 중에만 읽고 쓴다.

`loop_future`/`*_timer`는 이벤트 루프 스레드만 만들고 직접 건드린다(TimerHandle.cancel()은 스레드 안전성이 문서화되지 않았다). worker 스레드는 `_Ticket`을 절대 참조하지 않는다 — `_run_ticket`은 `(callable_, execution_deadline)`만 클로저 인자로 받는 순수 함수라 락이 필요 없다(§2.8).

### 2.3 `QueryExecutor` 생성자와 불변식

```python
class QueryExecutor:
    def __init__(self, *, concurrency_limit: int, queue_limit: int,
                 queue_timeout: float, execution_timeout: float,
                 loop: asyncio.AbstractEventLoop,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = threading.Lock()          # 모든 카운터/deque/tickets/saturation 보호
        self._pool = ThreadPoolExecutor(max_workers=concurrency_limit)
        self._queue: deque[int] = deque()       # FIFO — ticket_id만 보관
        self._tickets: dict[int, _Ticket] = {}
        self._next_id = 0
        self._running = 0
        self._orphaned = 0
        self._lifecycle = "READY"               # STARTING은 서버 lifespan이 담당
        self._counters = {k: 0 for k in
            ("terminal_rejected", "queue_timeout", "execution_timeout", "cancelled", "completed")}
        self._admission_rejected = {k: 0 for k in ("not_ready", "overloaded", "submit_failed")}
        self._accepted_total = 0                # lifecycle counter(§9) — terminal 5종에 포함되지 않음(M42-DR1-006)
        self._capacity_version = 0
        self._capacity_full = False
        self._capacity_edge_at = clock()       # §8.2의 단일 monotonic clock domain
```

불변식(설계 시점 고정, 테스트 §9에서 매 race 후 검증): `0 <= running <= concurrency_limit`, `0 <= queued <= queue_limit`, `0 <= orphaned <= running`, `len(self._queue) == queued`. caller가 이미 끝난 `ABANDONED`를 `running`과 terminal outcome에 이중 산입하지 않도록 accepted 보존식은 `accepted_total == queued + (running - orphaned) + completed + queue_timeout + execution_timeout + cancelled + terminal_rejected`로 고정한다(§9). **ThreadPoolExecutor는 FIFO/backpressure를 갖지 않는다** — `self._pool.submit()`은 ticket이 `RUNNING`으로 승격되는 순간에만, 승격 1회당 정확히 1회 호출한다. Pool 자체의 내부 큐는 절대 대기실로 쓰지 않는다(REQ-002.5 요구를 만족하면서 REQ-002.3/004의 명시적 FIFO 소유권을 지키는 핵심 설계 결정).

```python
def _currently_full(self) -> bool:
    return self._running == self._concurrency_limit and len(self._queue) == self._queue_limit
```

`running`/`queued`를 변경하는 모든 지점은 카운터 변경 직후 `_record_capacity_edge_locked()`를 호출한다. 이 함수는 새 full 값이 이전 값과 다를 때만 `capacity_version += 1`, `capacity_full = new`, `capacity_edge_at = clock()`을 같은 critical section에서 기록한다. readiness는 이 immutable snapshot만 소비하므로 probe가 없던 중간 clear/re-full도 새 version/timestamp로 보존된다(§8.2, M42-DR1-004).

### 2.4 `submit()` — 원자적 admission과 submit 실패 rollback

```python
def submit(self, callable_: Callable[[], Any]) -> TicketHandle:
    committed_future = None
    with self._lock:
        if self._lifecycle != "READY":
            self._admission_rejected["not_ready"] += 1
            raise AdmissionRejected("not_ready")
        if self._running < self._concurrency_limit:
            ticket = self._new_ticket()
            try:
                self._admit_running(ticket, callable_)
            except (RuntimeError, BrokenThreadPool) as exc:  # shutdown 경쟁 또는 broken worker initializer
                self._rollback_uncommitted_running_locked(ticket)
                self._admission_rejected["submit_failed"] += 1
                raise SubmitFailed() from exc
            committed_future = ticket.pool_future
            handle = TicketHandle(ticket.ticket_id, ticket.loop_future)
        elif len(self._queue) < self._queue_limit:
            ticket = self._new_ticket()
            self._admit_queued(ticket)
            self._queue.append(ticket.ticket_id)
            self._pending_callable[ticket.ticket_id] = callable_
            return TicketHandle(ticket.ticket_id, ticket.loop_future)
        else:
            self._admission_rejected["overloaded"] += 1
            raise AdmissionRejected("overloaded")
    # 완료 Future에 등록하면 이 스레드에서 즉시 callback되므로 lock 밖이어야 한다.
    committed_future.add_done_callback(self._on_pool_future_done)
    return handle
```

두 한도 확인과 ticket 생성/삽입이 `self._lock` 한 critical section 안에서 일어난다(REQ-002.3). 거부 시 ticket/thread/future가 전혀 생성되지 않는다. `submit()`은 이벤트 루프 스레드에서만 호출된다(`rag_query` 코루틴 내부) — `loop.call_later`를 잠금 보유 중 호출해도 안전하다(동일 스레드, 재진입 없음).

**commit point는 `pool.submit()`이 Future를 반환한 순간**이다. 호출 시작은 commit이 아니다. `_admit_running`은 submit 전에 timer/callback/pending callable을 만들지 않고, 반환 뒤에만 ticket을 accepted로 publish한다:

```python
def _admit_running(self, ticket: _Ticket, callable_: Callable[[], Any]) -> None:
    self._running += 1
    ticket.state = TicketState.RUNNING
    ticket.execution_deadline = self._clock() + self._execution_timeout
    ticket.pool_future = self._pool.submit(_run_ticket, callable_, ticket.execution_deadline)  # 반환이 commit
    ticket.exec_timer = self._loop.call_later(self._execution_timeout, self._on_execution_timeout, ticket.ticket_id)
    self._accepted_total += 1
    self._record_capacity_edge_locked()

def _rollback_uncommitted_running_locked(self, ticket: _Ticket) -> None:
    assert ticket.pool_future is None
    self._running -= 1
    ticket.execution_deadline = None
    ticket.state = TicketState.REJECTED
    self._pending_callable.pop(ticket.ticket_id, None)
    self._tickets.pop(ticket.ticket_id, None)
    self._record_capacity_edge_locked()
```

`pool.submit()`이 `RuntimeError("cannot schedule new futures after shutdown")` 또는 `BrokenThreadPool`을 던지면 Future가 존재하지 않으므로 rollback 함수가 `running`, ticket, deadline, pending callable을 모두 되돌린다. timer/callback/`accepted_total`은 commit 뒤에만 생성·증가하므로 rollback 대상이 아니다. 반환 직후 future가 이미 완료될 수 있으므로 실제 구현은 submit/commit까지 lock 안에서 하되 `add_done_callback` 등록은 lock 밖에서 한다(완료 Future는 등록 호출 스레드에서 callback을 즉시 실행하므로 일반 `Lock` 재진입 deadlock 방지). callback 등록 자체가 실패하면 Future가 이미 실행 가능한 committed 상태이므로 rollback하지 않고 resource 결과를 동기 수거해 caller에 `internal` terminal을 한 번 전달한다. `SubmitFailed`는 오직 uncommitted 실패이며 `rag_query`가 안전한 500 `internal`을 한 번 반환하고 `rag_admission_rejected_total{reason="submit_failed"}`를 증가시킨다.

`_admit_queued`는 `queued += 1`, `queue_deadline = clock()+queue_timeout`, `queue_timer = loop.call_later(queue_timeout, self._on_queue_timeout, ticket_id)`, `accepted_total += 1`, `_record_capacity_edge_locked()`를 수행한다 — 이 경로는 pool을 건드리지 않으므로 submit 실패 rollback이 필요 없다.

`AdmissionRejected(reason)`/`SubmitFailed()`를 `submit()`이 예외로 던지는 이유: `rag_query`가 `TicketHandle` 유무로 분기하지 않고 단일 `try/except` 체인으로 표현하기 위함이며, 두 경로 모두 REQ-004.2 카운터에 정확히 1회만 반영된다(ticket이 남지 않으므로 별도 finalize 불필요 — exactly-once가 자명하다).

queued 승격도 같은 commit helper를 사용한다. `_promote_if_capacity()`는 lifecycle이 `READY`이고 slot이 있을 때만 head 하나를 dequeue해 낙관적 RUNNING으로 바꾸고 submit한다. submit이 `RuntimeError`/`BrokenThreadPool`이면 running을 롤백하되 이미 accepted된 ticket은 삭제하지 않고 caller terminal `REJECTED(reject_reason="submit_failed")`, outcome `rejected`, HTTP `internal`로 정확히 한 번 끝낸 뒤 다음 head를 **승격하지 않는다**(broken/shutdown pool에 반복 submit하는 fail-open 방지). DRAINING에서는 promotion 0이며 `begin_drain()`의 queued 축출은 `terminal_rejected`(reason=`not_ready`)로 끝난다. `test_submit_shutdown_and_broken_pool_rollback`은 direct admission과 queued promotion 두 기원을 각각 고정한다.

`begin_drain()`(§2.9)이 `self._pool.shutdown()`을 호출하기 전에 `self._lifecycle`을 `DRAINING`으로 바꾸므로, 정상 경로에서는 `submit()`의 첫 분기(`lifecycle != READY`)가 이미 신규 admission을 막는다 — `_admit_running`의 rollback 경로는 오직 **shutdown()과 submit() 사이의 진짜 경쟁**(예: `shutdown()`이 `_lifecycle=STOPPED`로 쓰기 전, `self._pool.shutdown()`이 먼저 호출된 비정상 순서나 `BrokenThreadPool` 같은 pool 자체의 내부 오류) 또는 pool 구현체가 방어적으로 거부하는 경우를 위한 안전망이다. 두 경쟁 모두 negative test로 traceability에 연결한다(§10).

### 2.5 FIFO 취소/timeout — 선두 제거와 단일 승격

```python
def cancel(self, ticket_id: int) -> None:
    with self._lock:
        ticket = self._tickets[ticket_id]
        if ticket.state == TicketState.QUEUED:
            self._finalize_queued_locked(ticket_id, TicketState.CANCELLED)
        elif ticket.state == TicketState.RUNNING:
            self._abandon_running_locked(ticket_id, cause="caller_cancel")
        # DONE/REJECTED/TIMED_OUT/ABANDONED: 이미 caller_finalized — no-op
```

`_finalize_queued_locked`(QUEUED 기원 전용, §2.6)는 `self._queue.remove(ticket_id)`로 **큐에서 즉시 제거**하고 capacity(`queued -= 1`)를 회수한다 — 선두가 아니어도 O(n) 제거지만 `queue_limit<=64`로 상한이 있어 허용 가능한 비용이다. `_on_queue_timeout(ticket_id)`도 동일 경로를 탄다. 두 경우 모두 finalize 말미에 `_promote_if_capacity()`를 호출해 **다음 살아있는 ticket 정확히 하나**를 승격한다 — `self._queue`에는 죽은 ticket이 절대 남지 않으므로(취소/timeout이 즉시 제거) `popleft()`만으로 다음 승격 대상이 곧 "살아있는" ticket임이 보장된다. private semaphore/wake 순서에 의존하지 않는다 — 승격은 오직 `_lock` 보유 중 `_promote_if_capacity()` 호출로만 일어난다.

정확한 구현 signature는 `_promote_if_capacity_locked() -> PromotionCommit | None`이다. lock 안에서는 최대 head 하나를 submit/commit하고 `(pool_future, ticket_id)`만 반환하며, 모든 caller는 lock을 놓은 뒤 `_register_done_callback(commit)`을 호출한다. 이 규칙은 direct submit과 promotion 모두 완료 Future callback의 동기 실행에 의한 lock 재진입을 막는다. 따라서 아래 pseudocode의 `_promote_if_capacity()` 표기는 반환 commit을 바깥 wrapper가 등록한다는 축약이며, race 테스트는 한 resource completion당 commit 0 또는 1과 callback 등록 1을 spy로 assert한다.

RUNNING ticket의 취소는 `_abandon_running_locked`(§2.6)를 타므로 `_queue`/`queued`/`running`을 **전혀** 건드리지 않는다 — slot은 pool future가 실제로 끝날 때(`_complete_resource_locked`)까지 유지된다.

### 2.6 caller-outcome guard와 resource-completion guard — exactly-once의 근원(M42-DR1-001)

QUEUED에서 끝나는 세 상태(`TIMED_OUT`/`CANCELLED`/`REJECTED`)는 pool future가 아예 없으므로 caller-finalize와 resource-finalize가 같은 순간에 함께 일어난다 — 단일 함수로 충분하다:

```python
def _finalize_queued_locked(self, ticket_id: int, terminal: TicketState, *, reject_reason: str | None = None) -> None:
    ticket = self._tickets[ticket_id]
    if ticket.caller_finalized:
        return                                  # 이미 다른 경로가 먼저 도착 — no-op
    ticket.caller_finalized = True
    ticket.resource_finalized = True            # QUEUED 기원은 resource를 가진 적이 없다 — 항상 참
    ticket.state = terminal
    if ticket.queue_timer is not None: ticket.queue_timer.cancel()
    self._queued -= 1
    self._queue.remove(ticket_id)
    counter = {"TIMED_OUT": "queue_timeout", "CANCELLED": "cancelled",
               "REJECTED": "terminal_rejected"}[terminal.name]
    self._counters[counter] += 1
    self._record_capacity_edge_locked()
    self._promote_if_capacity()
    if terminal != TicketState.CANCELLED:            # CANCELLED는 caller 자신이 이미 연결을 끊은 것 — wake 불필요(RUNNING의 caller_cancel과 동일 규칙)
        self._wake_caller_error(ticket, _ERROR_FOR_TERMINAL[terminal])
```

RUNNING에서 시작하는 ticket은 **두 개의 독립된 진입점**을 갖는다 — 이것이 M42-DR1-001이 요구하는 분리다.

**(A) caller-outcome guard** — timeout/cancel이 caller 쪽 응답을 정하지만 slot은 건드리지 않는다:

```python
def _abandon_running_locked(self, ticket_id: int, *, cause: Literal["execution_timeout", "caller_cancel"]) -> None:
    ticket = self._tickets[ticket_id]
    if ticket.caller_finalized:
        return                                  # Order B: resource가 이미 먼저 끝났다 — no-op
    ticket.caller_finalized = True
    ticket.state = TicketState.ABANDONED
    if ticket.exec_timer is not None: ticket.exec_timer.cancel()   # 이미 발화한 경우도 cancel()은 안전(idempotent)
    self._orphaned += 1                          # running은 그대로 유지 — slot 조기 반환 없음
    self._counters["execution_timeout" if cause == "execution_timeout" else "cancelled"] += 1
    if cause == "execution_timeout":
        self._wake_caller_error(ticket, ExecutionTimeoutError())
    # cause == "caller_cancel": caller가 이미 연결을 끊었으므로 wake 시도 자체를 하지 않는다(REQ-003.3)
```

**(B) resource-completion guard** — pool future가 실제로 끝났을 때만, 오직 여기서만 `running`/`orphaned`가 감소하고 승격이 일어난다:

```python
def _complete_resource_locked(self, ticket_id: int, exc: BaseException | None, result: Any) -> None:
    ticket = self._tickets[ticket_id]
    if ticket.resource_finalized:
        return                                  # 중복 done-callback(§2.7) — no-op
    ticket.resource_finalized = True
    was_orphaned = ticket.state == TicketState.ABANDONED
    self._running -= 1
    if was_orphaned:
        self._orphaned -= 1
    needs_wake = False
    if not ticket.caller_finalized:               # Order B: 아무도 abandon하지 않았다 — 정상 완료
        ticket.caller_finalized = True
        ticket.state = TicketState.DONE
        self._counters["completed"] += 1
        needs_wake = True
    self._record_capacity_edge_locked()
    self._promote_if_capacity()                    # running이 실제로 줄어든 이 지점에서만, 정확히 한 번
    self._pending_callable.pop(ticket_id, None)
    self._tickets.pop(ticket_id, None)              # resource completion에서만 RUNNING ticket 제거
    return needs_wake, ticket, exc, result          # 호출자가 §2.7 규칙에 따라 wake를 마셜링
```

**양 순서(order) race 표** — 네 조합 모두 `caller_finalized`/`resource_finalized`가 서로 다른 시점에 `True`가 되지만 각 guard는 정확히 한 번만 효과를 낸다:

| Order | 원인 | 1번째 도착 이벤트 | `caller_finalized` | `running`/`orphaned` 변화 | 2번째 도착 이벤트 | 최종 `running`/`orphaned` 변화 | 승격 시점 |
|---|---|---|---|---|---|---|---|
| A(abandon-first) | execution timeout | `_on_execution_timeout` → `_abandon_running_locked` | True(1st에서) | 변화 없음, `orphaned+=1` | `_on_pool_future_done` → `_complete_resource_locked` | `running-=1, orphaned-=1` | 2번째(resource) |
| A(abandon-first) | caller cancel | `rag_query`의 `except CancelledError` → `cancel()` → `_abandon_running_locked` | True(1st에서) | 변화 없음, `orphaned+=1` | `_on_pool_future_done` → `_complete_resource_locked` | `running-=1, orphaned-=1` | 2번째(resource) |
| B(completion-first) | (실행이 정상적으로 execution timeout보다 먼저 끝남) | `_on_pool_future_done` → `_complete_resource_locked` | True(1st에서, `state=DONE`) | `running-=1`(orphaned 관련 없음, 애초에 orphan 아니었음) | `_on_execution_timeout`(이미 취소됐거나 뒤늦게 발화) → `_abandon_running_locked` | 변화 없음(guard가 즉시 반환) | 1번째(resource) |
| B(completion-first) | (실행이 caller cancel 신호보다 먼저 끝남) | `_on_pool_future_done` → `_complete_resource_locked` | True(1st에서, `state=DONE`) | `running-=1` | `cancel()` → `_abandon_running_locked` | 변화 없음(guard가 즉시 반환) | 1번째(resource) |

불변식 확인: 모든 행에서 `running`은 정확히 한 번(2번째 event가 A일 때, 1번째 event가 B일 때)만 감소하고, `orphaned`는 A order에서만 `+1` 후 `-1`로 정확히 상쇄되며 B order에서는 애초에 건드려지지 않는다 — `orphaned <= running`이 어느 시점에도 깨지지 않는다. **caller가 받는 outcome과 counter increment는 항상 1번째 도착 이벤트가 결정한다** — 2번째 이벤트는 오직 아직 안 끝난 절반(주로 resource)만 마저 처리한다.

```python
def _on_execution_timeout(self, ticket_id: int) -> None:          # loop.call_later 콜백, 루프 스레드
    with self._lock:
        self._abandon_running_locked(ticket_id, cause="execution_timeout")

def _handle_completion_on_loop(self, ticket_id: int, exc: BaseException | None, result: Any) -> None:
    with self._lock:
        outcome = self._complete_resource_locked(ticket_id, exc, result)
    needs_wake, ticket, exc, result = outcome
    if needs_wake:
        self._deliver_caller(ticket, exc, result)                 # §2.7 — 루프 스레드에서만 호출됨이 보장된 지점
```

### 2.7 loop-close 안전성 — resource-only 경로는 asyncio 객체를 절대 만지지 않는다(M42-DR1-002)

`ThreadPoolExecutor`의 `Future.add_done_callback`은 **결과가 확정된 스레드에서** 콜백을 실행한다(이미 완료 상태에서 추가하면 추가한 스레드에서 즉시 실행) — 즉 `_on_pool_future_done`은 worker 스레드 또는 루프 스레드 어느 쪽에서나 불릴 수 있다.

```python
def _on_pool_future_done(self, pool_future: "concurrent.futures.Future") -> None:
    ticket_id, exc, result = self._lookup_by_pool_future(pool_future)
    with self._lock:
        needs_wake, ticket, exc, result = self._complete_resource_locked(ticket_id, exc, result)  # 항상 실행, asyncio-free
    if not needs_wake:
        return                                   # Order A(§2.6) — caller는 이미 abandon 시점에 통지됨, 여기선 아무것도 안 만짐
    try:
        self._loop.call_soon_threadsafe(self._deliver_caller, ticket, exc, result)
    except RuntimeError:
        return                                   # 루프가 이미 close됨 — resource 회계는 위에서 이미 끝났다, wake는 포기
```

**`_complete_resource_locked`는 `running`/`orphaned`/`_counters`/capacity edge/dict 항목만 다루는 순수 Python 함수이며 `asyncio.Future`, `TimerHandle`, `call_soon_threadsafe` 어느 것도 참조하지 않는다** — worker 스레드에서 직접 호출돼도 항상 안전하다. `exec_timer.cancel()`은 Order B에서 `_deliver_caller`(루프 스레드 전용, 아래) 안으로 옮겼다 — Order A에서는 timer가 이미 발화해 스스로를 소모했으므로 cancel 대상이 없고, Order B에서 loop가 이미 close된 경우는 timer가 다시 발화할 loop 자체가 없으므로 cancel을 생략해도 안전하다.

```python
def _deliver_caller(self, ticket: _Ticket, exc: BaseException | None, result: Any) -> None:
    # 루프 스레드에서만 호출됨이 call_soon_threadsafe 성공으로 보장된다.
    if ticket.exec_timer is not None:
        ticket.exec_timer.cancel()
    if not ticket.loop_future.done():
        ticket.loop_future.set_exception(exc) if exc is not None else ticket.loop_future.set_result(result)
```

`_on_execution_timeout`/`_on_queue_timeout`/`cancel()` 세 경로는 애초에 이벤트 루프 스레드에서만 호출되므로(`loop.call_later`로 등록되거나 `rag_query` 코루틴 안에서 직접 호출) `_abandon_running_locked`의 `_wake_caller_error` 호출도 항상 루프 스레드에서 일어난다 — 문제는 오직 worker 스레드가 트리거할 수 있는 `_on_pool_future_done` 콜백 하나뿐이며, 위 구조는 그 콜백의 asyncio 접촉을 `_deliver_caller`(루프 스레드 전용) 하나로 완전히 격리한다.

**loop-close × caller-abandon 4개 조합** — `needs_wake`는 §2.6의 Order(A/B)만으로 결정되고, loop 생존 여부는 `needs_wake=True`일 때만 결과(marshal 성공/실패)에 영향을 준다:

| # | Order(§2.6) | `needs_wake` | loop 상태(자원 완료 시점) | `call_soon_threadsafe` 결과 | asyncio 접촉 여부 | resource counter | caller 통지 |
|---|---|---|---|---|---|---|---|
| 1 | B(completion-first) | True | 살아있음 | 성공 → `_deliver_caller` 실행 | 있음(루프 스레드에서만) | `running-=1`, `resource_finalized=True`, `completed+=1` | 정확히 1회, 정상 결과 |
| 2 | B(completion-first) | True | close됨 | `RuntimeError` 발생 → catch, 즉시 return | resource phase는 없음; notification phase가 loop method만 호출하고 Future/TimerHandle은 안 만짐 | 동일(1과 같음) | 없음 — loop 종료 정책상 응답 전달 불가 |
| 3 | A(abandon-first) | False | 살아있음 | 시도 자체 없음 | 없음 | `running-=1, orphaned-=1`, `resource_finalized=True` | abandon 시점에 이미 통지됨(루프가 그때는 반드시 살아있었다 — abandon은 루프 스레드에서만 발생) |
| 4 | A(abandon-first) | False | close됨 | 시도 자체 없음 | 없음 | 동일(3과 같음) | 동일(3과 같음) |

네 행 모두 먼저 실행하는 `_complete_resource_locked`는 asyncio 객체를 **참조하거나 호출하지 않는 resource-only path**다. 그 함수가 반환한 뒤에만 별도 notification phase가 `needs_wake=True`일 때 loop marshal을 시도한다. 행 2는 loop 객체의 marshal 메서드 호출은 하지만 worker가 `Future`/`TimerHandle`을 조작하지 않으며, 행 3·4는 notification phase 자체가 없다. 유일한 Future/TimerHandle 접촉은 행 1의 성공한 marshal 뒤 루프 스레드가 실행하는 `_deliver_caller`다.

### 2.8 `_run_ticket` — worker 스레드, 락 없음

```python
def _run_ticket(callable_: Callable[[], Any], execution_deadline: float) -> Any:
    token = bind_deadline(Deadline(execution_deadline))   # observability/deadline.py, §5
    try:
        return callable_()
    finally:
        _DEADLINE.reset(token)
```

모듈 최상위 함수(또는 `staticmethod`)로 두어 `self`/`_Ticket`/`_lock`을 절대 참조하지 않는다는 것을 타입으로도 드러낸다. `ThreadPoolExecutor.submit(fn, *args)`는 호출 스레드의 `contextvars.Context`를 복사하지 않는다(그 동작은 `asyncio.to_thread`에만 있다) — 따라서 `_DEADLINE.set()`을 worker 스레드 진입 직후 이 함수 안에서 직접 호출하는 것이 유일하게 올바른 전파 지점이다(§5).

### 2.9 drain/shutdown 시퀀스 — transition notification과 주입 deadline scheduler

`web/scheduling.py`의 공개 계약은 다음과 같다. production scheduler는 `loop.call_later`를, test scheduler는 fake monotonic domain의 명시적 `advance_to()`를 사용하며 어느 쪽도 polling/sleep을 사용하지 않는다.

```python
class DeadlineScheduler(Protocol):
    def schedule_at(self, when: float, callback: Callable[[], None]) -> Cancellable: ...

class AsyncioDeadlineScheduler:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None: ...
    def schedule_at(self, when: float, callback: Callable[[], None]) -> Cancellable: ...

class ManualDeadlineScheduler:
    def __init__(self, clock: FakeClock) -> None: ...
    def schedule_at(self, when: float, callback: Callable[[], None]) -> Cancellable: ...
    def advance_to(self, when: float) -> None: ...  # due callback을 (deadline, sequence) 순으로 동기 실행
```

`QueryExecutor.__init__(..., deadline_scheduler: DeadlineScheduler | None=None)`는 현재 loop에 묶인 `_drained_event = asyncio.Event()`를 만들고 초기 `running==0`이므로 set한다. `0 -> 1` running 전이는 loop thread에서 clear한다. worker callback의 `_complete_resource_locked()`가 `running -> 0`을 만든 경우에는 lock 안에서 immutable `DrainTransition(version)`만 만들고, lock 밖에서 `loop.call_soon_threadsafe(_publish_drained, transition)`를 시도한다. `_publish_drained`는 같은 loop에서 version을 deduplicate하고 event를 set한다. loop가 닫혀 marshal이 실패해도 resource accounting은 이미 끝났고 shutdown은 더 이상 await할 loop가 없으므로 안전하다.

```python
def begin_drain(self) -> None: ...  # READY->DRAINING 한 번, queued 전부 not_ready

async def wait_drained(self, timeout: float) -> bool:
    absolute_deadline = self._clock() + max(0.0, timeout)
    waiter = _DrainWaiter(absolute_deadline=absolute_deadline)
    with self._lock:
        self._install_drain_waiter_locked(waiter)
        if self._running == 0:
            self._claim_drain_waiter_locked(waiter, "resource_zero")
    deadline_handle = self._deadline_scheduler.schedule_at(
        absolute_deadline,
        lambda: self._claim_drain_waiter(waiter, "deadline"),
    )
    try:
        winner = await waiter.future
        return winner == "resource_zero"
    finally:
        deadline_handle.cancel()
        self._remove_drain_waiter(waiter)

def shutdown(self) -> None: ...  # idempotent STOPPED snapshot, pool.shutdown exactly once
```

Lifecycle은 `STARTING -> READY -> DRAINING -> STOPPED`다. `begin_drain()`은 queued를 즉시 `not_ready`로 축출하고 신규 admission을 막는다. `shutdown()`은 lock 아래 `_shutdown_called` guard로 residual `running/orphaned` snapshot과 STOPPED를 한 번만 commit한 뒤 lock 밖에서 `pool.shutdown(wait=False, cancel_futures=True)`를 정확히 한 번 호출한다. grace는 process hard bound가 아니며 residual이 있으면 successful drain으로 기록하지 않는다.

`_DrainWaiter`는 `absolute_deadline`, loop-bound `future`, `winner: Literal["resource_zero","deadline"] | None`, `winner_sequence: int | None`를 가진다. resource completion의 `running -> 0` 경로와 deadline callback은 모두 **같은 `QueryExecutor._lock`, 같은 `_next_linearization_sequence` 증가, 같은 `_claim_drain_waiter_locked()` compare-and-set**을 사용한다. 첫 callback만 `winner`와 sequence를 commit하고 lock 밖에서 future 통지를 예약하며, 뒤 callback은 loser no-op이다. 따라서 `asyncio.wait()`의 unordered done set이나 callback 실행 뒤 membership 우선순위가 결과를 바꾸지 않는다. stale `_drained_event`는 winner 입력이 아니며 제거한다; transition version은 관측/report용으로만 남는다. waiter가 absolute deadline을 직접 보존하므로 `Cancellable.remaining()`은 계약에 없고 cancel 전후 의미도 필요 없다. outer cancellation은 handle을 cancel하고 waiter registry에서 제거한 뒤 이미 예약된 notification task를 cancel+await한다.

`test_wait_drained_all_event_orders`는 `ManualDeadlineScheduler`의 `(deadline, insertion_sequence)` 실행과 명시적 resource completion hook으로 completion-first=`True`, deadline-first=`False`, exact tie 두 insertion order의 해당 첫 CAS winner, stale 이전 transition 뒤 새 waiter의 독립 판정, timeout=0의 deadline winner를 고정한다. 각 case는 `winner_sequence` 단조 증가, loser의 CAS 실패, pending callback/task 0을 assert한다. 각 case 뒤 `shutdown()`을 중복 호출해도 pool shutdown spy는 정확히 1이며, winner 직후의 residual snapshot과 STOPPED snapshot이 일치한다. loop-close 뒤 worker callback도 resource accounting을 완료하되 asyncio 통지는 best-effort이고, 전 profile의 `sleep`/poll count는 0이다.

`web/server.py::_make_lifespan`(§4)이 `yield` 이후 `begin_drain() -> await wait_drained(SHUTDOWN_GRACE_SECONDS) -> shutdown()`을 순서대로 호출한다.

### 2.10 `ExecutorSnapshot`과 `TicketHandle`

```python
@dataclass(frozen=True)
class ExecutorSnapshot:
    lifecycle: str
    running: int
    queued: int
    orphaned: int
    concurrency_limit: int
    queue_limit: int
    accepted_total: int              # lifecycle counter(§9) — 5개 terminal outcome에 포함되지 않음
    admission_rejected_total: int     # ticket 미생성: overloaded/not_ready/submit_failed 합
    queue_timeout_total: int          # terminal outcome 5종 중 1
    execution_timeout_total: int      # terminal outcome 5종 중 1
    cancelled_total: int              # terminal outcome 5종 중 1 (QUEUED-cancel + RUNNING-cancel 합산)
    completed_total: int              # terminal outcome 5종 중 1
    terminal_rejected_total: int      # terminal outcome 5종 중 1 (accepted 뒤 drain/submit failure)
    capacity_full: bool               # §8.2: 마지막 실제 capacity edge의 값
    capacity_edge_at: float           # 동일 executor clock의 edge timestamp
    capacity_version: int             # edge마다 +1; probe 누락 중간 전이 검출
    stopped_with_running: int | None
    stopped_with_orphaned: int | None

def snapshot(self) -> ExecutorSnapshot:
    with self._lock:
        return ExecutorSnapshot(..., capacity_full=self._capacity_full,
            capacity_edge_at=self._capacity_edge_at, capacity_version=self._capacity_version)

@dataclass(frozen=True)
class TicketHandle:
    ticket_id: int
    _loop_future: asyncio.Future
    async def result(self) -> Any:
        return await self._loop_future
```

`snapshot()`은 read-only 공개 API 전부다(REQ-002.1: `submit/begin_drain/wait_drained/shutdown` + snapshot). `evaluate_readiness`(§6)와 `/metrics`(§7)가 이 함수 하나만 호출한다.

## 3. `web/errors.py` — 고정 오류 계약 (REQ-004.3/004.4)

```python
_FIXED_ANSWER = "요청을 처리할 수 없습니다. 잠시 후 다시 시도해주세요."

@dataclass(frozen=True)
class ApiError:
    code: str; http_status: int; retryable: bool

ERRORS: dict[str, ApiError] = {
    "invalid_request":    ApiError("invalid_request", 400, False),
    "payload_too_large":  ApiError("payload_too_large", 413, False),
    "not_ready":          ApiError("not_ready", 503, True),
    "overloaded":         ApiError("overloaded", 503, True),
    "queue_timeout":      ApiError("queue_timeout", 503, True),
    "execution_timeout":  ApiError("execution_timeout", 504, True),
    "internal":           ApiError("internal", 500, False),
}

def error_response(code: str) -> JSONResponse:
    e = ERRORS[code]
    return JSONResponse(status_code=e.http_status, content={
        "success": False, "answer": _FIXED_ANSWER, "sources": [],
        "search_type": "unknown", "error": {"code": e.code, "retryable": e.retryable},
    })
```

질문 원문·upstream body·예외 문자열·절대 경로는 이 함수가 유일한 오류 응답 생성 경로이므로 구조적으로 누출될 수 없다(파라미터가 `code: str` 하나뿐).

## 4. `web/server.py` — admission과 lifecycle 배선

### 4.1 실제 설치되는 `/rag` ASGI 경계와 단일 send/receive owner

`@app.post` handler가 sentinel을 반환하는 구조는 사용하지 않는다. `create_app()`은 health/home route를 등록한 뒤 정확히 한 번 `app.router.routes.append(RagASGIRoute(path="/rag", endpoint=rag_query, methods={"POST"}, response_model=QueryResponse))`를 실행한다. `RagASGIRoute`는 Starlette `APIRoute`의 response 생성 이후를 감싸는 class가 아니라 `/rag` scope 전체의 ASGI callable을 제공하며, sentinel을 response validation/serialization **전에** 가로챈다.

```python
_NO_RESPONSE = object()

class RagASGIRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Awaitable[Response | object]]: ...
    def matches(self, scope: Scope) -> tuple[Match, Scope]: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        await serve_rag_scope(scope, receive, send, endpoint=self.endpoint)

async def rag_query(request: Request, body: object) -> Response | QueryResponse | object: ...
async def serve_rag_scope(scope: Scope, receive: Receive, send: Send, *,
                          endpoint: Callable[..., Awaitable[object]]) -> None: ...
async def race_result_or_disconnect(*, ticket: TicketHandle, receive: Receive,
                                    executor: QueryExecutor,
                                    arbiter: RaceArbiter) -> ResultOrDisconnect: ...
```

`serve_rag_scope`가 유일한 top-level owner이며 순서는 하나뿐이다.

1. `BodyLimitMiddleware`가 넘긴 `receive`를 `receive_body_once()`가 단독 소유해 마지막 `http.request(more_body=False)`까지 모은다. JSON/media/schema/question 검증과 admission은 그 body로 수행한다. 이 함수가 반환하기 전 observer task는 존재하지 않는다.
2. endpoint는 검증 후 ticket을 만들고 response가 아닌 `_PendingRag(ticket)`을 route에 반환한다. route는 이 값을 response validation/serialization에 넘기지 않고 즉시 race로 들어간다. 이 시점에 body owner는 종료됐으며 raw `receive` 소유권이 `DisconnectObserver` 하나로 handoff된다(`concurrent_receive` counter 0→1→0).
3. `RaceArbiter`는 단일 `asyncio.Lock`, monotonic sequence, `winner` CAS를 가진다. result task와 disconnect task가 완료 callback에서 같은 `_claim(kind)`을 호출한다. 첫 claim만 winner이고 exact tie는 callback insertion sequence로 결정된다; unordered `done` set membership으로 우선순위를 재판정하지 않는다.
4. result winner이면 disconnect loser를 cancel하고 **await하여 회수한 뒤** `QueryResponse` 또는 고정 `error_response`를 처음 생성하고, route만 `await response(scope, receive_never_called, send)`를 호출한다. `send`의 유일한 owner는 route이며 start 1/body 1개 이상을 정확히 한 response stream으로 보낸다.
5. disconnect winner이면 executor cancel을 exactly once 수행하고 result loser를 cancel+await한 뒤 `_NO_RESPONSE`를 route 내부에서 intercept하여 Response 생성·validation·`send` 호출 없이 반환한다. queued/running cancel의 resource 의미는 §2와 같다.
6. endpoint/race/route outer cancellation은 `finally`에서 존재하는 두 child 모두 cancel+await하고 ticket이 있으면 executor cancel을 exactly once 한 뒤 `CancelledError`를 재전파한다. response send가 시작된 뒤의 server cancellation은 response task도 cancel+await하며 두 번째 오류 response를 보내지 않는다.

`DisconnectObserver.wait()`는 handoff 뒤 `http.disconnect`만 terminal로 인정하고 추가 `http.request`는 protocol error로 정규화한다. body owner와 observer가 공유하는 executable receive spy는 in-flight 호출을 세어 최대 1을 강제한다. `tests/integration/test_web_disconnect.py::test_asgi_disconnect_queued_100_races`와 `::test_asgi_disconnect_running_100_races`는 각각 **실제 `create_app()`**을 `asgi_exchange`로 호출해 disconnect-first, result-first, exact-tie 두 insertion order를 포함한 100 iteration을 수행한다. 매 iteration에서 winner sequence/kind, disconnect winner frames `[]`, result winner의 정확히 한 `http.response.start`와 terminal `http.response.body(more_body=False)`, receive in-flight max 1, finalize count 1, loser/pending asyncio task 0을 assert하고, running fixture는 release 후 최종 `(queued,running,orphaned)==(0,0,0)`까지 기다린다. 별도 outer-cancel case도 동일 cleanup/frame 불변식을 검사한다.

#### 4.1.1 pure-ASGI `RequestContextMiddleware` terminal 계약

현재 `BaseHTTPMiddleware` 상속은 제거하고 `__call__(scope, receive, send)`만 구현한다.
middleware는 downstream을 직접 await하고 `observed_send`로 첫
`http.response.start.status`만 관측한다. `call_next()`/memory stream/Response 재생성은 없다.
따라서 downstream이 disconnect winner로 frame 0개를 보내고 정상 반환하는 것은 예외가 아닌
지원 terminal이다.

HTTP scope마다 request ID ContextVar set/reset, `request_start`, terminal claim,
`request_end`, `rag_requests_total`, duration observe는 한 outer `try/except/finally` owner가
각각 정확히 한 번 수행한다. `frames==0`만으로 disconnect를 추론하지 않는다. route가
`scope["state"]["rag_terminal"]="client_disconnected"`를 설정한 뒤 정상 반환했거나 middleware가
감싼 `receive`에서 실제 `http.disconnect`를 관측한 경우에만 **proven disconnect terminal**이다.
그 증거 없이 downstream이 정상 반환하고 frame을 하나도 보내지 않으면
`RuntimeError("downstream_no_response")`인 programming error이며 `internal`로 기록하고 재전파한다.
terminal 분류는 다음 닫힌 표만 사용한다.

| downstream 결과 | 실제 response frame | 내부 outcome | status-equivalent | 전송 동작 |
|---|---:|---|---:|---|
| 정상 response | start 1 | `response` | 실제 status | downstream frame만 전달 |
| client disconnect 정상 반환 | 0 | `client_disconnected` | 499 | **아무 frame도 보내지 않음** |
| disconnect 증거 없는 정상 no-response | 0 | `internal` | 500 | `RuntimeError("downstream_no_response")` 재전파; 보상 response 없음 |
| downstream 예외 | 0 또는 이미 시작 | `internal` | 500 또는 이미 시작 status | 예외 재전파; 보상 499/500 response 없음 |
| outer cancellation | 0 또는 이미 시작 | `client_disconnected` if disconnect observed, else `cancelled` | 499-equivalent 또는 500 clamp | cancellation 재전파; 새 response 없음 |

`client_disconnected`는 bounded internal enum이며 HTTP wire status가 아니다. 기존 M4.1 log
schema의 `status_code`에는 숫자 499를 넣고 `error_code="client_disconnected"`를 넣는다.
metrics는 기존 `clamp_status` allowlist에 `4xx`로 귀속하므로 새 unbounded label이 없다.
request ID는 disconnect terminal의 start/end logs에도 같은 값으로 정확히 한 번 나타난다.
`REQUEST_ID.reset(token)`은 관측 기록 뒤 최종 `finally`에서 항상 한 번 실행된다.

actual-app disconnect acceptance는 `request_start==1`, `request_end==1`, duration observation 1,
`rag_requests_total{route="rag",status="4xx"}` delta 1, `error_code=client_disconnected`,
status-equivalent 499, frames `[]`를 추가로 assert한다. result winner는 기존 실제 status의 end/log/
metric delta가 각 1이고, exception/tie/outer-cancel에서도 terminal record 총합이 요청 수와 같다.
이 actual stack 증거가 route-unit trace보다 우선한다.
`test_downstream_no_response_without_disconnect_is_internal`은 disconnect marker/관측 없이 반환하는
fixture가 `client_disconnected`나 4xx로 분류되지 않고 고정 RuntimeError, internal end/counter 각 1,
frames 0을 남기는지 검사한다.

### 4.2 `_validate_question` (REQ-007.2)

```python
def _validate_question(body: Any, max_chars: int) -> str | None:
    if not isinstance(body, dict) or set(body) != {"question"}:
        return None
    q = body["question"]
    if not isinstance(q, str):
        return None
    q = q.strip()
    if not q or "\x00" in q or len(q) > max_chars:
        return None
    if any(unicodedata.category(ch) in ("Cc", "Cs") for ch in q):
        return None
    return q
```

`FastAPI`의 기존 `QueryRequest` pydantic 모델은 유지하되(REQ-009.1 CLI/성공 body 보존), `/rag` 핸들러는 pydantic 자동 바인딩 대신 수동 `request.json()` 경로를 타 400 오류 body를 REQ-004 고정 계약으로 통일한다 — pydantic의 기본 422 응답은 이 계약을 만족하지 않으므로 우회가 필요하다.

### 4.3 `_make_lifespan` — single-active guard와 immutable process Settings identity

#### 4.3.1 active lifespan guard — 모든 전역 mutation보다 먼저

모듈 전역 `_SingleActiveLifespanGuard`는 `threading.Lock`, 현재 owner token 또는 `None`,
단조 증가 token sequence만 가진다. 별도 `_ProcessSettingsCommit`은 같은 종류의 process lock 아래
최초 성공한 `Settings` object reference를 영구 보존한다. production lifetime에는 reset API가 없고
module reload, `config.py` facade rebinding, previous-cache rollback, generation lease도 없다.
테스트가 다른 identity를 필요로 하면 fresh subprocess를 사용한다. `_make_lifespan`의 첫 실행 문장은
`lease = _ACTIVE_LIFESPAN_GUARD.acquire()`다. acquire는 lock 안에서 owner가 `None`일 때만
새 opaque `_LifespanLease`를 반환한다. owner가 이미 있으면 고정
`RuntimeError("lifespan_already_active")`로 startup을 실패시키며, 그 전에
`Settings.from_sources`/`set_settings_for_process`/engine factory/`QueryExecutor` 호출과 app
state의 settings/cache/engine/executor mutation은 모두 0이다. 이 fail-fast 오류는 두 app 중
어느 것이 owner인지와 무관하게 같은 type/message를 갖는다.

owner lifespan은 최외곽 `try/finally` 하나로 lease를 소유한다. acquire 직후에는
`candidate=None`, `executor=None`, `grace=0.0`과 startup/cleanup ordering을 검증하기 위한
**local trace만** 초기화한다. loader가 반환한 값은 local `candidate`에만 넣고
`commit_or_verify(candidate)`를 거친다. process identity가 비어 있으면 cache write와 함께
`candidate`를 최초 commit하고, 이미 같은 object이면 cache/config identity를 바꾸지 않는
idempotent reacquire다. 이미 다른
object면 값의 동등성과 무관하게 고정 `RuntimeError("process_settings_identity_mismatch")`로 거부한다.
이 reject는 app state/cache/config facade/engine/executor mutation 전에 일어나며 local trace와
lease cleanup 외의 관측 가능한 변경이 0이다. 특히 identity mismatch를 health/readiness 오류로
publish하지 않는다. settings validation 실패는 최초 commit을 만들지 않는다.

attempt class와 cleanup owner는 다음처럼 고정한다.

| attempt class | lifecycle publication ownership | 허용된 durable publication | cleanup owner/tail |
|---|---|---|---|
| identity mismatch | 없음. guard acquire 뒤 identity verify에서 끝나 app lifecycle owner가 되지 않는다 | 없음 | lease-local cleanup만 exact-owner release; 외부 observer 전체 delta 0 |
| invalid loader | fail-soft bootstrap owner | atomic `settings_invalid` health transaction 정확히 1개 | generic stopped observer 없이 exact-owner release |
| started/partially-started | app lifecycle owner | 설정/engine 진단과 최종 STOPPED snapshot | fallible observer/snapshot/error aggregation 뒤 canonical `STOPPED -> release` tail |

identity mismatch는 guard acquire 뒤지만 app/health/log/metric/cache/config/engine/executor publication
전에 발생한다. cleanup은 lease-local이며 exact guard owner만 release한다. `app.__dict__`, health/log/metric
sinks, process cache/config와 engine/executor factories를 포함한 모든 external observer의 full-attempt
before/after delta는 0이다. lifecycle owner가 된 적이 없으므로 `STOPPED`도 publish하지 않는다.
invalid loader는 별도 fail-soft bootstrap owner다. REQ-009.2의 atomic `settings_invalid` health
transaction을 정확히 한 번 publish하며 generic stopped observer가 이를 overwrite하거나 두 번째
diagnostic을 추가할 수 없다.

`release_exact_owner(token)`는 lock 안에서 exact token을 확인하고 owner를 `None`으로 durable clear하는
비-throwing atomic primitive다. clear가 완료된 뒤 bounded diagnostic code만 반환한다. 이 반환값의
log/metric 처리는 non-durable best-effort이고 release 이후 reacquire를 막거나 durable cleanup action을
추가할 수 없다. 중복/비-owner는 bounded programming-diagnostic code로만 표현된다.

#### 4.3.2 exception/cancellation-safe teardown state machine (M42-RR1-002)

이 state machine은 lifecycle ownership을 publish한 started/partially-started attempt에만 적용한다.
상태는 `OWNED -> DRAIN_ATTEMPTED -> WAIT_ATTEMPTED|WAIT_SKIPPED -> SHUTDOWN_ATTEMPTED ->
FALLIBLE_OBSERVERS_DONE -> STOPPED_PUBLISHED -> RELEASED`이고 뒤로 가지 않는다. body/yield에서 발생한 첫 exception 또는
`CancelledError`를 `primary`로 먼저 보존한다. cleanup은 별도 task로 만들고 `asyncio.shield`로
완료시킨다. cleanup await 중 들어온 cancellation은 기록만 하고 cleanup task를 cancel하지 않으며
task가 끝날 때까지 다시 shield한다. `begin_drain()`은 executor construction이 commit됐으면 정확히 한 번 독립
`try`에서 호출한다. 성공했을 때만 `wait_drained(timeout=grace)`를 정확히 한 번 bounded await하며,
실패/취소/`False`(grace expiry)는 모두 다음 단계로 진행한다. `shutdown(wait=False,
cancel_futures=True)`은 앞선 모든 오류와 무관하게 정확히 한 번 시도한다. 그 mandatory attempt 뒤
모든 fallible observer, residual snapshot 계산과 cleanup error aggregation을 끝낸다. canonical tail은
non-throwing atomic app lifecycle `STOPPED` publication과 그 직후의 `release_exact_owner(lease)`이며,
이 둘이 final two durable external actions다. 둘 사이와 release 뒤에는 durable/fallible 작업이 없다.
따라서 lifecycle-owning attempt의 새 lifespan은 `STOPPED_PUBLISHED`와 `RELEASED` 둘 다 끝나기 전
acquire할 수 없다. 이 `STOPPED -> release` invariant는 lifecycle ownership을 publish한 attempt에만
적용하며 identity mismatch에는 적용하지 않는다.

teardown coroutine의 argument 평가, coroutine 생성, `create_task`와 shield await 자체도 같은
primary-preserving 경계 안에 둔다. task 생성이 실패하면 그 오류를 첫 cleanup secondary로 기록한
뒤 cancellation-deferred inline fallback으로 같은 `_teardown(executor, app, lease, grace)`를 끝까지
실행한다. task가 만들어졌으면 outer cancellation은 task를 취소하지 않고 완료까지 재-shield한다.
따라서 cleanup infrastructure 오류도 mandatory shutdown/STOPPED/release를 건너뛰거나 이미 보존한
primary를 대체할 수 없다.

오류 정책은 고정한다. body/yield의 primary exception 또는 원래 cancellation이 있으면 그것을
동일 identity로 재전파하고 cleanup 오류는 발생 순서대로 bounded code
`teardown_task_create_failed|begin_drain_failed|wait_drained_failed|shutdown_failed|snapshot_failed|observer_failed`와
exception type만 `cleanup_secondary` log에 남긴다(문자열/경로 없음). primary가 없고 cleanup 오류가
하나면 그 오류를, 둘 이상이면 순서 보존 `ExceptionGroup("lifespan_cleanup_failed", errors)`를
raise한다. cancellation은 언제나 모든 mandatory cleanup 뒤 원래 `CancelledError`가 우선한다.

```python
async def _teardown(executor, app, lease, grace):
    errors = []
    began = False
    if executor is not None:
        try:
            executor.begin_drain(); began = True             # exactly once
        except BaseException as exc: errors.append(("begin_drain_failed", exc))
        if began:
            try: await executor.wait_drained(timeout=grace)  # bounded, exactly once
            except BaseException as exc: errors.append(("wait_drained_failed", exc))
        try: executor.shutdown(wait=False, cancel_futures=True)  # mandatory, exactly once
        except BaseException as exc: errors.append(("shutdown_failed", exc))
    snapshot = _collect_snapshot_and_observers(executor, app, errors)  # all fallible work
    aggregate = _freeze_cleanup_errors(errors)               # non-fallible value
    _publish_stopped_atomic(app, snapshot)                    # penultimate durable action
    release_diagnostic = release_exact_owner(lease)          # final durable action
    _emit_release_diagnostic_best_effort(release_diagnostic) # non-durable; never blocks reacquire
    return aggregate
```

#### 4.3.3 supported startup/reacquire paths

설계가 명시적으로 지원하는 수명 계약은 **process당 active app lifespan 정확히 하나**다.
따라서 previous-cache snapshot/CAS rollback, generation chain, predecessor lease, overlapping A/B
LIFO·non-LIFO 복원은 모두 범위에서 제거한다. 최초 successful commit 뒤 cache는 process 설정
identity로 영구 유지된다. 다음 단독 lifespan은 **정확히 같은 object**만 재사용할 수 있고 다른
identity로 교체할 수 없다.

#### 4.3.4 fail-soft import와 owner lifespan 초기화

`create_app(*, settings_loader=Settings.from_sources)`는 Bootstrap, metrics, health routes와 `BOOTSTRAP_BODY_CEILING=1_048_576`의 middleware만 만들며 settings loader를 호출하지 않는다. module-level `app = create_app()`는 환경이 잘못돼도 import된다. owner lease를 얻은 lifespan만 loader를 **정확히 한 번** 시도한다. loader 성공값은 local `candidate`에만 보존하며 lazy engine import나 app-state write보다 먼저 `commit_process_settings_once(candidate)`로 first-commit/same-identity 검증한다. 따라서 `rag_engine -> config -> get_settings()`는 재검증하지 않고 `candidate`와 같은 object를 본다.

```python
app = create_app()  # module import 경로: loader 호출 0, health route 즉시 구성
```

```python
lease = _ACTIVE_LIFESPAN_GUARD.acquire()
candidate = None
executor = None
grace = 0.0
trace = []                                      # local startup/cleanup trace only
attempt_class = "prepublication"                # release-only until a publication owner is chosen
try:
    try:
        trace.append("loader_start")
        candidate = settings_loader()            # local only; app/global mutation 0
        trace.append("loader_return")
    except SettingsError as exc:
        # REQ-009.2/M4.1이 허용한 invalid-loader fail-soft transaction만 publish한다.
        attempt_class = "invalid_loader"
        _publish_settings_invalid_transaction(app, safe_settings_reason(exc))
        yield                                    # live/ready health surface 유지
    else:
        commit_process_settings_once(candidate)  # first/same verify가 다음 유일한 동작
        trace.append("identity_verified")
        attempt_class = "lifecycle_owner"

        # 이 아래에서만 app state, cache-dependent facade, engine/executor/grace를 touch한다.
        app.state.settings_load_attempted = True
        app.state.settings = candidate
        app.state.settings_error = None
        grace = candidate.SHUTDOWN_GRACE_SECONDS
        try:
            app.state.engine = engine_factory(candidate)
            executor = QueryExecutor.from_settings(candidate)
            app.state.query_executor = executor
        except Exception as exc:
            _publish_engine_invalid_transaction(app, safe_engine_reason(exc))
        yield
finally:
    primary = sys.exception()                    # loader/commit/constructor/body/cancel 보존
    cleanup_errors = await _run_cleanup_for_attempt_class(
        attempt_class=attempt_class, executor=executor, app=app, lease=lease, grace=grace,
    ) # mismatch=release-only; invalid-loader=no generic STOPPED; lifecycle-owner=canonical tail
    _propagate_primary_or_cleanup(primary, cleanup_errors)
```

M4.2는 `settings.py`에 cache lease/generation/previous-value CAS seam을 추가하지 않는다.
`commit_process_settings_once(candidate)`는 guard owner/CLI preflight만 engine import 전에 호출한다.
invalid loader와 동시 두 번째 lifespan은 cache write 0이다. invalid loader의 fail-soft publication은
REQ-009.2가 보존하도록 요구한 기존 `settings_invalid` health transaction 하나뿐이며 commit을
시도하지 않는다. 반대로 identity mismatch는 startup exception으로 전파되고 health transaction도
없어 app/cache/config/engine/executor delta가 모두 0이다. engine failure 뒤에도 stale predecessor를 복원하지
않으며 outer guard release가 다음 owner의 clean reacquire를 보장한다.

settings 실패면 lifecycle은 STARTING, executor/engine은 `None`, readiness는 `settings_invalid`, `/rag`는 `not_ready`이며 live/deprecated health/metrics는 계속 응답한다. engine 실패도 executor를 만들지 않는다.

| 진입 경로 | loader attempt | 결과/exit | health/readiness | app state |
|---|---:|---|---|---|
| `import simple_qna_rag.web.server; server.app` | 0 | import 성공 | lifespan 전 route 존재 | `settings=None`, `attempted=False`, STARTING |
| ASGI lifespan, valid env | isolated lazy-import spy에서 `Settings.from_sources=1`, `config.get_settings=1`, 그 get의 `from_sources` 재호출 0 | startup 계속 | ready 200 | loader/cache/config/limiter/executor/engine 모두 `is s`; shutdown 뒤 guard release, process cache는 `s` 유지 |
| ASGI lifespan, invalid env | `Settings.from_sources=1`, `get_settings=0`, cache write=0 | 승인된 explicit failed-start diagnostic transaction 뒤 startup는 fail-soft 계속 | live 200, ready 503 `settings_invalid`, `/rag` 503 | settings-invalid atomic transaction 정확히 1, generic stopped observer 0, exact-owner release 1 |
| concurrent second lifespan | loader/cache write/engine/executor 모두 0 | startup `RuntimeError("lifespan_already_active")` | 두 번째 app은 미기동; owner health는 영향 없음 | owner state/cache identity 불변, guard owner 불변 |
| startup failure/cancellation 뒤 재시작 | prepublication cancel/identity mismatch는 release-only; invalid loader는 단일 transaction→release; lifecycle owner constructor 오류는 STOPPED→release | 다음 acquire 성공 | invalid loader/constructor만 승인된 기존 fail-soft 진단; cancel은 원 primary 재전파 | executor가 없으면 begin/wait/shutdown 0; attempt-class별 publication 유지, stale owner 0 |
| 정상 shutdown 뒤 same-identity 재시작 | loader 1회가 exact committed `s` 반환 | 다음 acquire 성공 | ready 200 | cache/config/engine/executor/app state 모두 `is s`; cache write 0, engine/executor 각 1 |
| 정상 shutdown 뒤 equal-value different identity | loader 1회 | `RuntimeError("process_settings_identity_mismatch")` | 새 app 미기동 | `app.__dict__`/health/log/metric/cache/config/factories full-attempt delta 모두 0; STOPPED 0, exact-owner release 1 |
| `start_server(settings_override=None)` | CLI preflight `Settings.from_sources=1`; lifespan closure는 같은 `s`; engine import의 `get_settings`는 cache hit | invalid는 uvicorn 전 exit 2 | server 미기동 | 총 validation 1, config/engine `is s` |
| `start_server(settings_override=s)` | 0회 loader, 이미 validated `Settings` identity 사용 | uvicorn 실행 | lifespan이 동일 객체 소비 | explicit CLI/test override 우선 |

`start_server(settings_override: Settings | None = None, *, settings_loader=Settings.from_sources) -> int`만 exit 2를 소유한다. preflight가 성공하면 uvicorn 전에 `commit_process_settings_once(s)`하고 `create_app(settings_loader=lambda: s)`를 넘긴다. 최초 commit이면 cache/config lazy import가 `s`를 보고 lifespan은 same-object verify만 한다. 이미 다른 identity가 commit된 process에서의 CLI preflight는 고정 identity mismatch로 uvicorn/app construction 전에 실패한다. initial validation 실패는 commit/cache/uvicorn/import 0이다. module ASGI lifespan도 같은 `commit_process_settings_once`를 사용하되 CLI exit를 모사하지 않는다.

`tests/integration/test_web_settings_identity.py`의 executable spies는 `Settings.from_sources`,
`get_settings`, `commit_process_settings_once`, engine factory, executor factory와 guard owner를 함께
기록한다. `test_module_app_import_does_not_load_settings_or_engine`은 count 0;
`test_valid_lifespan_commits_before_lazy_engine_import_and_releases_guard`는 from_sources 1,
engine-import cache hit, 모든 identity `is s`, shutdown guard `None`;
`test_concurrent_second_lifespan_rejected_before_global_mutation`은 owner active 중 두 번째의
loader/cache/engine/executor count 0과 고정 오류;
`test_startup_failure_releases_guard_and_reacquires`는 loader 전/중 cancellation, settings loader 오류와
engine/executor 각 constructor failure path;
executor가 construction commit 전에 실패했으면 begin/wait/shutdown count는 모두 0이지만 lifecycle
ownership을 이미 publish했다면 canonical STOPPED/release는 그대로 1이다. `test_same_identity_reacquires_and_different_identity_rejects_before_mutation`은 sequential 두 경로의
loader/cache/config/engine/executor/app identity와 counts 및 trace의
`acquire→locals→loader→commit/verify→publish/factory` 순서를 검사하고 다른 identity case는 fresh
subprocess에서 `app.__dict__`, health/log/metric sinks, process cache/config, engine/executor factories의
before/after snapshot을 비교해 각각 delta 0, STOPPED 0, release 1을 검사한다.
`test_invalid_loader_single_transaction_and_release`는 `settings_invalid` atomic transaction 1,
generic stopped observer 0, diagnostic overwrite/add 0, exact-owner release 1을 검사한다.
`test_shutdown_cleanup_matrix`는 begin/wait/shutdown 오류의 단일/조합,
teardown argument/task 생성 오류, 각 await cancellation, zero/running residual, grace expiry를 검사한다. 모든 행에서 begin<=1,
wait<=1(only if begin success), shutdown=1이며 모든 fallible observer/snapshot/error aggregation이
trace의 마지막 두 durable external action인 exact `STOPPED→release` 전에 있다. LIFO/non-LIFO cache rollback 및 competing predecessor matrix는 지원하지 않는 동시
lifespan을 전제하므로 catalog에서 제거한다. CLI valid/invalid identity tests는 그대로 유지한다.

### 4.4 미들웨어 순서와 body limit (REQ-007.1)

```python
app.add_middleware(RequestContextMiddleware)
app.add_middleware(BodyLimitMiddleware, bootstrap_max_bytes=BOOTSTRAP_BODY_CEILING)
```

`add_middleware`는 나중 추가한 것이 바깥쪽이다. 두 middleware 모두 pure ASGI이며 실제 stack은
`BodyLimitMiddleware -> RequestContextMiddleware -> router`다. request-context가
`BaseHTTPMiddleware`를 다시 상속하거나 route의 frame-0 반환을 Response로 강제 변환하면
M42-DR4-001 회귀로 간주한다. limiter는 `scope["app"].state.settings`가 있으면 검증값, 없으면 bootstrap hard ceiling을 선택한다. settings/engine 미준비 상태의 `/rag`는 body 한도 적용 뒤 `not_ready`이고 admission은 0이다. 초과는 전용 input metric만 남긴다.

```python
class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, bootstrap_max_bytes: int = BOOTSTRAP_BODY_CEILING) -> None: ...

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        max_bytes = _effective_max(scope, self.bootstrap_max_bytes)
        encoding = _content_encoding(scope)
        if encoding not in (None, "", "identity"):
            return await _send_invalid_request(send)  # admission/receive 전 fail-closed
        declared = _content_length(scope)
        if declared is not None and declared > max_bytes:
            return await _send_413(send)          # body를 읽기 전에 거부

        wire_delivered_bytes = 0
        application_consumed_bytes = 0
        stopped = False
        async def limited_receive():
            nonlocal wire_delivered_bytes, application_consumed_bytes, stopped
            if stopped:
                raise RuntimeError("receive after body limit terminal")
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                wire_delivered_bytes += len(chunk)  # upstream가 이미 전달한 message 전체
                remaining_plus_probe = max_bytes - application_consumed_bytes + 1
                prefix = chunk[:remaining_plus_probe]
                application_consumed_bytes += len(prefix)
                message = {**message, "body": prefix}
                if len(chunk) > len(prefix) or application_consumed_bytes > max_bytes:
                    stopped = True
                    raise _PayloadTooLarge(
                        wire_delivered_bytes=wire_delivered_bytes,
                        application_consumed_bytes=application_consumed_bytes,
                    )
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _PayloadTooLarge:
            await _send_413(send)
```

정책은 **모든 non-identity `Content-Encoding`을 400 `invalid_request`로 body receive/admission 전에 거부**하는 것이다. 이 서비스는 압축 해제를 소유하지 않으며 proxy/server의 암묵 decompression에 의존하지 않는다; 중복/쉼표 encoding, `gzip`, `br`, 대소문자 변형은 identity의 단일 token 정규형 외에는 모두 거부한다. 따라서 application-consumed decompressed 크기는 identity prefix 크기와 같다.

여기서 두 byte 수치는 의도적으로 다르다. `wire_delivered_bytes`는 upstream ASGI server가 한 `receive()` 결과로 이미 애플리케이션에 넘긴 message body 전체이고, `application_consumed_bytes`는 middleware가 downstream parser에 허용하거나 overflow 판정에 소비한 prefix다. middleware는 매 call에서 `remaining + 1` prefix만 slice하고 application-consumed 합계가 `max_bytes+1`을 넘지 않게 하며, overflow 뒤에는 upstream `receive()`를 다시 호출하지 않는다. 상태/metric에는 두 값을 별도 필드로 남기되 body content는 남기지 않는다. 413/error body/admission=0 계약은 wire chunk 크기와 무관하게 유지된다.

ASGI application은 `receive()`가 반환하기 전에 server가 만든 message 크기를 통제할 수 없다. 따라서 첫 message가 `limit+N`이면 가능한 보장은 `application_consumed_bytes==limit+1`, `wire_delivered_bytes==limit+N`, receive calls=1이지 wire bytes `<=limit+1`이 아니다. 이 설계는 Uvicorn/proxy request-buffer cap을 새로 지정하지 않으므로 이미 전달된 oversized message를 예방한다고 주장하지 않는다. 향후 strict wire cap이 필요하면 구체적 server/proxy max-request/message 설정과 그 배포 test를 별도 요구사항으로 추가해야 한다.

`test_body_profiles_stop_at_limit_plus_one`은 (a) single chunk `limit+N`, (b) multichunk가 정확히 limit 뒤 다음 chunk에서 overflow, (c) 거짓 작은 `Content-Length`, (d) length 없음, (e) 선언 length `>limit` 조기 거부를 포함한다. (a)는 wire=`limit+N`, consumed=`limit+1`, receive=1; (b)는 wire가 실제 전달 chunk 합계이고 consumed=`limit+1`, overflow 뒤 추가 receive=0; (c)/(d)는 같은 prefix/stop 계약; (e)는 둘 다 0을 assert한다. 모두 413, executor submit 0이다. gzip bomb/false-length/no-length tests는 400, wire/consumed/receive/decompression/submit 모두 0을 assert한다.

## 5. `observability/deadline.py` — context-local deadline (REQ-008.2)

```python
_DEADLINE: ContextVar["Deadline | None"] = ContextVar("_DEADLINE", default=None)

class Deadline:
    def __init__(self, monotonic_deadline: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._deadline, self._clock = monotonic_deadline, clock
    def remaining(self) -> float:
        return max(0.0, self._deadline - self._clock())
    def expired(self) -> bool:
        return self.remaining() <= 0.0

def bind_deadline(deadline: Deadline) -> "contextvars.Token":
    return _DEADLINE.set(deadline)

def current_deadline() -> Deadline | None:
    return _DEADLINE.get()
```

`request_context.py`의 `REQUEST_ID` ContextVar와 동일한 선례를 따라 관측 모듈에 둔다. 전파 경로는 §2.8의 `_run_ticket`이 유일한 진입점이다 — `ThreadPoolExecutor.submit()`은 호출 스레드의 context를 복사하지 않으므로(`asyncio.to_thread`와 달리), worker 함수 본문 맨 앞에서 직접 `bind_deadline()`을 호출하는 것이 유일하게 올바른 지점이다. `route_query -> web_search.search_web / rag_engine.RAGEngine.query`는 모두 같은 worker 스레드 안에서 동기 호출로 이어지므로 이 ContextVar 값을 그대로 본다.

## 6. 외부 네트워크 경계 (REQ-008)

### 6.1 설치된 패키지 API 검증 결과

| 대상 | 실제 API (`venv/lib/python3.11/site-packages/`) | 결론 |
|---|---|---|
| `ollama.Client`(`ollama/_client.py:79-112`) | `BaseClient.__init__`이 `httpx.Client(timeout=..., **kwargs)`를 **생성 시 1회만** 구성. `Client.generate/chat`(`_client.py:192-`)의 시그니처는 `timeout` kwarg를 받지 않고 `self._client.request()`로 그대로 전달하지 않음 | **per-call timeout override 불가.** 생성 시점 고정값만 가능 |
| `langchain_ollama.OllamaLLM`/`ChatOllama`(`llms.py:337`) | `self._client = Client(host=..., **sync_client_kwargs)` — 생성자 1회 | 위와 동일한 제약을 그대로 상속 |
| `ddgs.DDGS`(`ddgs/ddgs.py`) | `DDGS(timeout=X)`는 매 호출부에서 **새 인스턴스**로 재생성 가능(현재 `web_search.py`가 이미 그렇게 함). `text()->_search_sync()`는 내부적으로 `with ThreadPoolExecutor(max_workers=..., thread_name_prefix="DDGS") as executor:`(`ddgs.py:411`)로 백엔드별 요청을 fan-out하고, 별도로 `_get_cache_executor()`(`ddgs.py:87-93`, `max_workers=2`, daemon 아님)가 캐시 쓰기를 위해 process-lifetime 동안 유지됨 | **per-call timeout 가능**(새 인스턴스 생성 방식이므로 mutate 아님). 단 내부 `with ThreadPoolExecutor` 블록은 종료 시 `shutdown(wait=True)`를 암묵 수행 — 우리가 준 `timeout=`이 라이브러리 버그로 무시되면 그 `with` 자체가 우리 execution deadline보다 오래 블록될 수 있음 |

설치 API는 per-`generate/chat` kwarg를 허용하지 않지만 **생성자에는 `timeout`을 전달할 수 있으므로 매 호출 새 client를 만들면 remaining budget을 실제 httpx timeout으로 고정할 수 있다.** 따라서 요구사항 변경은 필요하지 않다. concurrency=1은 singleton thread-safety 근거가 없는 별도 이유로 계속 유지한다.

### 6.2 Ollama 어댑터 — 매 호출 remaining-budget client

새 `observability/deadline.py::ollama_call_client()`는 공유 객체를 받거나 mutate하지 않는 resource factory다.

```python
@contextmanager
def ollama_call_client(*, host: str, connect_timeout: float):
    deadline = current_deadline()
    remaining = deadline.remaining() if deadline is not None else 0.0
    if remaining <= 0:
        raise UpstreamDeadlineExceeded()
    timeout = httpx.Timeout(
        connect=min(connect_timeout, remaining),
        read=remaining, write=remaining, pool=min(connect_timeout, remaining),
    )
    client = ollama.Client(host=host, timeout=timeout)  # ollama 0.6.0 생성자 → httpx.Client(timeout=...)
    try:
        yield client
    finally:
        client._client.close()  # 0.6.0에 public close가 없어 version-pinned adapter 한 곳에서만 transport 소유/폐기
```

router는 `_get_router_llm` singleton을 호출 경로에서 제거하고 `route_query()`의 각 router 호출마다 위 context 안에서 `client.chat(model=..., messages=..., tools=..., stream=False, options=...)`를 한 번 호출한 뒤 응답의 `message.tool_calls`를 기존 tool dispatch 입력으로 변환한다. answer는 `RAGEngine`에 model/base URL/template 데이터만 장수명으로 보존하고 `generate_answer()`의 각 호출마다 `client.generate(model=..., prompt=rendered_prompt, stream=False, options={"temperature": 0.1})`를 한 번 수행한다. 즉 LangChain `ChatOllama`/`OllamaLLM`의 생성 시 고정 client를 요청 경로에서 사용하지 않으며, CLI `route_query()`의 외부 signature/반환 shape는 그대로다.

각 router/answer 호출 직전에 `remaining()`을 새로 읽으므로 router가 budget 대부분을 소비하면 answer client의 read/write timeout은 실제 남은 값으로 축소된다. factory가 생성한 client/transport는 해당 호출의 worker thread만 소유하고 `finally`에서 닫으며 cache/singleton에 넣지 않는다. `_client` 접근은 설치된 `ollama==0.6.0`에 public close가 없는 데 따른 격리된 compatibility seam이며 dependency snapshot과 `test_ollama_adapter_transport_closed`가 field 존재·정확히 1회 close를 고정한다. 향후 public `close()`가 생기면 이 seam만 교체한다.

### 6.3 DDGS 어댑터 — remaining-budget 축소

```python
# web_search.py::search_web — MODIFIED
def search_web(query, max_results=None):
    ...
    deadline = current_deadline()
    remaining = deadline.remaining() if deadline is not None else WEB_SEARCH_TIMEOUT
    if remaining <= 0:
        log_event("web_search", stage="web_search", duration_ms=0.0, error_code="timeout")
        return []
    timeout = min(WEB_SEARCH_TIMEOUT, remaining)
    with DDGS(timeout=timeout) as ddgs:
        results = list(ddgs.text(query=query, region=WEB_SEARCH_REGION,
                                  max_results=max_results, timelimit=None))
    ...
```

`DDGS(timeout=...)`는 매 호출 새 인스턴스이므로 mutate가 아니다. 자동 retry는 기존에도 없고(단일 `ddgs.text()` 호출) 이번에도 추가하지 않는다(REQ-008.1 "새 자동 retry는 0회"). expired-before-call은 `remaining<=0` 분기로 처리하고 빈 결과(기존 실패 시맨틱과 동일한 `[]`)를 반환한다 — `route_query()`의 기존 "web_search 실패 시 document_qa로 재시도" 폴백 경로를 그대로 재사용하므로 이 지점에 새 오류 코드가 필요 없다.

### 6.4 stall fake 테스트 설계 (REQ-008.3)

새 `tests/unit/test_network_deadline.py`는 client constructor fake로 router 전 90초, router가 fake clock 89초를 소비한 뒤 answer 전 1초인 예를 만들고 두 번째 constructor의 `httpx.Timeout.read/write == 1.0`, `connect/pool == min(config, 1.0)`, 두 client가 서로 다른 identity, global singleton write 0, transport close 각 1회를 assert한다. 별도 stall fake는 `threading.Event`를 기다리게 하여 QueryExecutor caller가 `execution_deadline+100ms` 이내 `ExecutionTimeoutError`를 받는지 측정하고, resource는 `event.set()` 뒤에만 `running/orphaned`에서 빠지는지 확인한다.

## 7. `settings.py` — 8개 필드 (REQ-001)

```python
def _make_range_validator(lo, hi, *, lo_inclusive=True, hi_inclusive=True, finite=False):
    def _check(v):
        if finite and not math.isfinite(v):
            raise ValueError("must be finite")
        ok = (v >= lo if lo_inclusive else v > lo) and (v <= hi if hi_inclusive else v < hi)
        if not ok:
            raise ValueError(f"must be within ({lo},{hi})")
        return v
    return _check
```

| # | name | annotation | default | validator | env_alias |
|---|---|---|---:|---|---|
| 42 | `QUERY_CONCURRENCY_LIMIT` | int | 1 | `_make_range_validator(1,2)` | `SIMPLE_QNA_RAG_QUERY_CONCURRENCY_LIMIT` |
| 43 | `QUERY_QUEUE_LIMIT` | int | 4 | `_make_range_validator(0,64)` | `SIMPLE_QNA_RAG_QUERY_QUEUE_LIMIT` |
| 44 | `QUERY_QUEUE_TIMEOUT_SECONDS` | float | 5.0 | `_make_range_validator(0,30,lo_inclusive=False,finite=True)` | `SIMPLE_QNA_RAG_QUERY_QUEUE_TIMEOUT_SECONDS` |
| 45 | `QUERY_EXECUTION_TIMEOUT_SECONDS` | float | 90.0 | `_make_range_validator(1,600,finite=True)` | `SIMPLE_QNA_RAG_QUERY_EXECUTION_TIMEOUT_SECONDS` |
| 46 | `SHUTDOWN_GRACE_SECONDS` | float | 30.0 | `_make_range_validator(0,120,finite=True)` | `SIMPLE_QNA_RAG_SHUTDOWN_GRACE_SECONDS` |
| 47 | `MAX_REQUEST_BODY_BYTES` | int | 16384 | `_make_range_validator(256,1_048_576)` | `SIMPLE_QNA_RAG_MAX_REQUEST_BODY_BYTES` |
| 48 | `MAX_QUESTION_CHARS` | int | 4000 | `_make_range_validator(1,32_000)` | `SIMPLE_QNA_RAG_MAX_QUESTION_CHARS` |
| 49 | `UPSTREAM_CONNECT_TIMEOUT_SECONDS` | float | 5.0 | `_make_range_validator(0,30,lo_inclusive=False,finite=True)` | `SIMPLE_QNA_RAG_UPSTREAM_CONNECT_TIMEOUT_SECONDS` |

`assert len(FIELD_SPECS) == 41` → `== 49`(MODIFIED, 한 줄). float 필드는 모두 `finite=True`다 — `float("nan")`/`float("inf")`는 Python이 파싱 자체는 성공시키므로(예외를 던지지 않음) validator에서 명시적으로 거부해야 REQ-001.2 "NaN/무한대는 기본값으로 대체하지 않는다(=거부)"를 만족한다. 새 `MODEL_VALIDATORS` 원소:

```python
def _check_queue_timeout_lt_execution_timeout(self):
    if self.QUERY_QUEUE_TIMEOUT_SECONDS >= self.QUERY_EXECUTION_TIMEOUT_SECONDS:
        raise ValueError("QUERY_QUEUE_TIMEOUT_SECONDS must be < QUERY_EXECUTION_TIMEOUT_SECONDS")
    return self
```

기본값(5.0 < 90.0)은 이 제약을 통과한다. `unknown key`/`알 수 없는 env`는 `_from_sources()`의 기존 `unknown` 검사가 이미 `ENV_PREFIX`로 시작하는 모든 키를 대상으로 하므로 8개 신규 필드가 자동으로 커버된다 — 이 부분은 **CURRENT, 변경 없음**. `field_disclosure()`는 신규 필드가 모두 `int`/`float`이므로 `_BOUNDED_VALUE_ANNOTATIONS`에 이미 걸려 자동으로 `"value"`(평문 노출 허용) 판정을 받는다 — **CURRENT, 변경 없음**, secret-safe 계약이 신규 필드에도 그대로 적용됨을 코드 검토만으로 증명 가능하다(REQ-001.3).

## 8. `observability/health.py` — lifecycle/saturation readiness (REQ-006)

### 8.1 `evaluate_readiness` 신호 확장

```python
def evaluate_readiness(
    bootstrap_error: str | None, settings_error: str | None, engine_error: str | None,
    *, lifecycle: str | None = None, saturated: bool = False,
    orphaned: int = 0, concurrency_limit: int = 0,
) -> tuple[int, str]:
    if bootstrap_error is not None: return 503, "static_mount_failed"
    if settings_error is not None: return 503, "settings_invalid"
    if engine_error is not None: return 503, "engine_init_failed"
    if lifecycle in ("DRAINING", "STOPPED"): return 503, "draining"
    if concurrency_limit > 0 and orphaned == concurrency_limit: return 503, "orphan_workers"
    if saturated: return 503, "queue_saturated"
    return 200, "ok"
```

기존 세 우선순위는 손대지 않고(REQ-009.2 M4.1 계약 유지) keyword-only 신규 인자로 순수 확장한다 — 위치 인자 호출부(있다면)는 깨지지 않는다. `saturated`는 호출자(§8.2 `SaturationDebounce`)가 미리 계산해 넘긴다 — 이 함수 자체는 여전히 순수 함수, I/O·시계 없음(기존 docstring 불변식 유지).

### 8.2 executor edge snapshot 기반 `SaturationDebounce` (M42-DR1-004)

```python
class SaturationDebounce:
    _HOLD_SECONDS = 1.0
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._active = False
        self._seen_version = -1

    def evaluate(self, *, full: bool, edge_at: float, version: int) -> bool:
        now = self._clock()
        if version != self._seen_version:
            self._seen_version = version
        if now - edge_at >= self._HOLD_SECONDS:
            self._active = full
        return self._active
```

`ExecutorSnapshot`은 동일 lock에서 `capacity_full`, `capacity_edge_at`, `capacity_version`을 함께 복사한다. debounce는 probe 시각이 아니라 마지막 실제 edge 시각으로 연속 1초를 판정한다. 예를 들어 `t=0 full(v1) → t=.4 clear(v2) → t=.6 re-full(v3) → t=1.0 probe`이면 snapshot은 `(full=True, edge_at=.6, version=3)`이므로 아직 0.4초뿐이라 `ok`다; `t=1.6`에야 `queue_saturated`가 된다. 반대 clear 방향도 같은 규칙이다. fake clock negative control은 중간 probe 없이 이 exact sequence와 `v1<v2<v3`를 assert한다. admission 거부는 debounce와 무관하게 현재 카운트로 즉시 판단한다.

### 8.3 `/health/ready` 배선(MODIFIED)

```python
@app.get("/health/ready")
async def health_ready(request: Request):
    state = request.app.state
    snap = state.query_executor.snapshot() if state.query_executor else None
    saturated = state.saturation_debounce.evaluate(
        full=snap.capacity_full, edge_at=snap.capacity_edge_at,
        version=snap.capacity_version,
    ) if snap else False
    status_code, reason = evaluate_readiness(
        state.bootstrap_error, state.settings_error, state.engine_error,
        lifecycle=snap.lifecycle if snap else None,
        saturated=saturated,
        orphaned=snap.orphaned if snap else 0,
        concurrency_limit=snap.concurrency_limit if snap else 0,
    )
    _update_readiness_gauge(state, reason)                     # 아래
    return JSONResponse(status_code=status_code, content={"status": "ok" if status_code==200 else "not_ready", "reason": reason})
```

M4.1 `rag_readiness` gauge는 runtime reason 전이 때 갱신한다. `_update_readiness_gauge`는 snapshot으로 계산한 새 reason의 전체 allowlist 값을 하나의 metrics-side lock 아래 0/1로 교체한다. `/health/ready`는 이 갱신을 best-effort로 시도하지만 실패를 readiness HTTP 판정에 역전파하지 않는다. executor gauges는 `/metrics` scrape 직전 §9의 atomic snapshot으로 동기화하므로 readiness probe 유무와 독립적이다.

## 9. `observability/metrics.py` — 신규 계측 (REQ-006.3/006.4)

```python
READINESS_REASONS = frozenset({
    "ok", "settings_invalid", "engine_init_failed", "static_mount_failed",
    "draining", "orphan_workers", "queue_saturated", "other",
})   # MODIFIED — additive, 기존 5개 값 보존

TICKET_OUTCOMES = frozenset({"completed", "rejected", "queue_timeout", "execution_timeout", "cancelled"})
ADMISSION_REJECT_REASONS = frozenset({"not_ready", "overloaded", "submit_failed"})
INPUT_REJECT_REASONS = frozenset({"payload_too_large", "invalid_request"})

def clamp_ticket_outcome(v: str) -> str: return _clamp_label(v, TICKET_OUTCOMES, default="rejected")
def clamp_input_reject_reason(v: str) -> str: return _clamp_label(v, INPUT_REJECT_REASONS, default="invalid_request")
```

`build_metrics_registry`에 추가:

| Metric | 종류 | 라벨 | 라벨 상한 |
|---|---|---|---|
| `rag_queue_depth` | Gauge | 없음 | 1 series |
| `rag_running` | Gauge | 없음 | 1 series |
| `rag_orphaned_workers` | Gauge | 없음 | 1 series |
| `rag_query_outcomes_total` | Counter | `result`∈TICKET_OUTCOMES | 5 series |
| `rag_admission_rejected_total` | Counter | `reason`∈ADMISSION_REJECT_REASONS | 3 series |
| `rag_input_rejected_total` | Counter | `reason`∈INPUT_REJECT_REASONS | 2 series |

```python
def sync_executor_gauges(registry, snapshot: "ExecutorSnapshot | None") -> None:
    if registry is None or snapshot is None: return
    registry.rag_queue_depth.set(snapshot.queued)
    registry.rag_running.set(snapshot.running)
    registry.rag_orphaned_workers.set(snapshot.orphaned)

def record_ticket_outcome(registry, result: str) -> None:
    if registry is None: return
    registry.rag_query_outcomes_total.labels(result=clamp_ticket_outcome(result)).inc()

def record_input_rejected(registry, reason: str) -> None:
    if registry is None: return
    registry.rag_input_rejected_total.labels(reason=clamp_input_reject_reason(reason)).inc()
```

outcome 분류는 다음처럼 닫힌다. admission 전에 거부된 `not_ready`/`overloaded`/uncommitted `submit_failed`는 `rag_admission_rejected_total`이며 `accepted_total`에 들어가지 않는다. drain 또는 queued promotion submit 실패로 이미 accepted된 QUEUED를 축출하면 `terminal_rejected`이고 public outcome은 `rejected`다. queue/execution timeout, caller cancel, completed는 각각 동명 terminal이다. 내부 보존식은 `accepted = queued + (running-orphaned) + completed + queue_timeout + execution_timeout + cancelled + terminal_rejected`; admission 시도 보존식은 `submit_attempts = accepted + not_ready + overloaded + submit_failed`다. HTTP acceptance runner는 입력 거부까지 포함하는 별도 `request_terminal` enum을 사용하므로 executor 식과 혼합하지 않는다.

executor lock 안에서는 내부 dict/state만 갱신하고 `Metric.inc/set`을 절대 호출하지 않는다. finalize가 만든 immutable `MetricEvent`를 lock 밖 best-effort sink에 전달하며 sink 예외는 잡아 `metrics_side_effect_failures` 진단만 남기고 caller outcome/resource state를 되돌리거나 재실행하지 않는다. 중복 방지는 `caller_finalized`가 event 생성을 소유한다. `/metrics` handler는 exposition을 만들기 **직전** `snap = query_executor.snapshot()` 한 번으로 세 gauge 값을 `registry.metrics_lock` 아래 함께 set한 뒤 render한다; 따라서 직접 scrape도 최신의 동일-version triple을 보고 `/health/ready` 호출에 의존하지 않는다. snapshot 이후 새 전이는 다음 scrape에 반영되는 정상 linearization이다.

**1,000개 고유 질문 cardinality 증명(REQ-006.4)**: 신규 series 상한은 gauges 3 + outcome 5 + admission reason 3 + input reason 2 = 13이며 질문 문자열은 라벨에 없다. 테스트는 fresh registry에서 모든 allowlist label을 materialize한 이론 상한과 1,000 unique 입력 후 실제 series 수가 같음을 assert한다.

## 10. Traceability — Requirement와 정량 Gate의 실행 계약

기능 요구사항은 §2 caller/resource guards(REQ-002~005), §4 fail-soft 단일 settings attempt/disconnect/input(REQ-001/003/007), §6 network client(REQ-008), §8~9 readiness/metrics(REQ-006), 기존 suite와 runner(REQ-009)에 직접 연결한다. 아래는 구현자가 그대로 import할 공개 acceptance API다.

```python
# scripts/run_m42_acceptance.py
PROFILE_NODE_IDS: Mapping[str, tuple[str, ...]]              # 아래 deterministic 10행과 정확히 동일
REQUEST_TERMINALS: tuple[str, ...]
def run_profile(name: str, *, seed: int, repeat_index: int,
                scheduler_factory: Callable[[], ManualDeadlineScheduler]) -> GateReceipt: ...
def collect_profile_nodes(profiles: Sequence[str]) -> tuple[str, ...]: ...
def validate_conservation(receipt: RunReceipt) -> None: ...
def load_live_manifest(path: Path) -> LiveManifest: ...          # schema/id/count fail-closed
def main(argv: Sequence[str] | None = None) -> int: ...

# tests/support/m42_load.py
class EventDrivenService:
    def __init__(self, *, service_seconds: float, scheduler: ManualDeadlineScheduler) -> None: ...
    def __call__(self, request_id: str) -> dict: ...             # started Event 후 release(id)까지 block
    def release(self, request_id: str) -> None: ...              # virtual +.200 후 해당 worker Event set
def drive_serial_load(service: EventDrivenService, request_ids: Sequence[str]) -> LoadTrace: ...

# tests/support/asgi.py
def asgi_exchange(app: ASGIApp, *, request_messages: Sequence[ASGIMessage],
                  disconnect_gate: threading.Barrier | None=None) -> ASGITrace: ...
```

`PROFILE_NODE_IDS`는 일반 mapping이 아니라 아래 insertion order와 byte string을 보존하는 immutable ordered inventory다. tuple 내부와 profile 사이 node 중복은 모두 금지한다.

```python
PROFILE_NODE_IDS = MappingProxyType({
    "event_loop": ("tests/integration/test_web_concurrency.py::test_event_loop_remains_responsive",),
    "bounded_admission": ("tests/unit/test_query_executor.py::test_profile_1_2_5_exact_receipt",),
    "fifo_cancel": ("tests/unit/test_query_executor.py::test_fifo_cancel_head_promotes_once",),
    "queue_timeout": ("tests/unit/test_query_executor.py::test_queue_head_timeout_single_promotion",),
    "execution_timeout": ("tests/unit/test_query_executor.py::test_abandoned_holds_slot_until_future_done",),
    "caller_cancellation": (
        "tests/integration/test_web_disconnect.py::test_asgi_disconnect_queued_100_races",
        "tests/integration/test_web_disconnect.py::test_asgi_disconnect_running_100_races",
    ),
    "drain": ("tests/unit/test_shutdown_drain.py::test_running_queued_and_grace_expiry_profiles",),
    "saturation_readiness": ("tests/unit/test_readiness_saturation.py::test_edge_timestamp_debounce_with_intermediate_clear",),
    "payload": ("tests/integration/test_web_input_boundary.py::test_body_profiles_stop_at_limit_plus_one",),
    "normal_mock_load": ("tests/evaluation/test_m4_safe_serving_load.py::test_single_thread_40",),
})
```

아래 표는 Requirement §5의 **11개 행을 순서와 개수 그대로** test/runner/fixture/formula/assert/report field에 1:1 고정한다. 각 exact node ID는 실제 파일의 module-level test 함수이며 `collect_profile_nodes()`가 `pytest --collect-only -q` subprocess로 존재/중복 없음까지 확인한다.

| Gate | exact pytest symbol | runner/profile | 결정론 fixture/입력 | 측정식 | exact PASS assert | report field |
|---|---|---|---|---|---|---|
| event loop | `tests/integration/test_web_concurrency.py::test_event_loop_remains_responsive` | `event_loop` | `BlockingRoute(started: Event, release: Event)`; watchdog가 2.0s 뒤 release(순서 판정에는 미사용), live 20회 | nearest-rank p95, max | `ok==20`, `p95<=.100`, `max<=.250` | `gates.event_loop.{ok,p95_ms,max_ms}` |
| bounded admission | `tests/unit/test_query_executor.py::test_profile_1_2_5_exact_receipt` | `bounded_admission` | conc=1,q=2, barrier 동시 5 | snapshot maxima+terminal count | `max_running==1`, `max_queued==2`, `overloaded==2`, violations=0 | `gates.bounded_admission.{max_running,max_queued,outcomes,violations}` |
| FIFO/cancel | `tests/unit/test_query_executor.py::test_fifo_cancel_head_promotes_once` | `fifo_cancel` | running gate + queued A/B, A cancel event | worker start trace | `trace==[running,B]`, q/orphan=0, finalize count/ticket=1 | `gates.fifo_cancel.{start_order,finalize_counts}` |
| queue timeout | `tests/unit/test_query_executor.py::test_queue_head_timeout_single_promotion` | `queue_timeout` | running R 시작 Event→A/B queue→clock A deadline→A timer fire→R release Event→R resource complete | R release 전 submit delta=0, 이후 delta=1 | A=`queue_timeout`, B는 **R resource completion에서만** 단일 승격, 총 worker submit R+B=2 | `gates.queue_timeout.{statuses,promoted,worker_submits}` |
| execution timeout | `tests/unit/test_query_executor.py::test_abandoned_holds_slot_until_future_done` | `execution_timeout` | stall Event+FakeClock/scheduler | virtual return-deadline; snapshots before/after release | slip<=.100 virtual, pre `(1,1)`, post `(0,0)` | `gates.execution_timeout.{deadline_slip_ms,pre_release,post_release}` |
| caller cancellation | `tests/integration/test_web_disconnect.py::test_asgi_disconnect_queued_100_races`; `tests/integration/test_web_disconnect.py::test_asgi_disconnect_running_100_races` | `caller_cancellation` | pure-ASGI middleware가 설치된 실제 app/`http.disconnect`; 각 node 100회에 양 order와 두 tie insertion order | ASGI frames, receive in-flight, arbiter sequence, task inventory, request log/metric deltas | 각 node 100, receive max=1, disconnect frames=0/예외=0/`client_disconnected`·499-equivalent, result start=1+terminal body, start/end/duration/request counter 각 1, loser/pending=0, finalize=1, final triple=0 | `gates.caller_cancellation.{nodes,iterations,winners,frames,receive_max,pending,request_events,request_metrics,final}` |
| drain | `tests/unit/test_shutdown_drain.py::test_running_queued_and_grace_expiry_profiles` | `drain` | stall 1+queued 2; `ManualDeadlineScheduler`; completion-first/deadline-first/tie/zero-grace/loop-close; begin/wait/shutdown single+combined errors; cancellation before cleanup, at wait await, while shield-awaiting cleanup, after wait; zero/running residual | transition versions+cleanup trace/count/error receipts+shutdown spy | queued/new=`not_ready`, polling=0, winner deterministic; begin<=1, applicable wait<=1, mandatory shutdown=1 despite errors/cancel, STOPPED=1 then release=1 last; original primary/cancel identity preserved, secondary policy exact; reacquire only after both; residual exact | `gates.drain.{trace,rejected,winner,poll_count,begin_calls,wait_calls,shutdown_calls,stopped_calls,release_calls,primary,secondary,residual,reacquire_at}` |
| saturation readiness | `tests/unit/test_readiness_saturation.py::test_edge_timestamp_debounce_with_intermediate_clear` | `saturation_readiness` | FakeClock `0 full,.4 clear,.6 full,1.0/1.6 probe`, reverse clear | `probe-edge_at` | 1.0 ok, 1.6 saturated, clear 후 1초에 ok; versions strictly increase | `gates.saturation.{edges,versions,probes}` |
| payload | `tests/integration/test_web_input_boundary.py::test_body_profiles_stop_at_limit_plus_one` | `payload` | identity `limit+N` single/multichunk/no-length/false-length + encoded cases | wire-delivered와 application-consumed 별도 | identity 초과=413, consumed<=limit+1, overflow 뒤 receive=0; single wire=`limit+N`; encoded=400 before receive; submit=0 | `gates.payload.{cases,statuses,wire_delivered_bytes,application_consumed_bytes,receive_calls,decompression_calls,submits}` |
| 정상 mock load | `tests/evaluation/test_m4_safe_serving_load.py::test_single_thread_40` | `normal_mock_load` | `EventDrivenService(.200, ManualDeadlineScheduler)`, IDs 00..39; 매 started Event 뒤 scheduler +.200, release Event | virtual makespan only | success=40, max running=1, lost=0, virtual makespan=8.000s(<=9.600s), wall time은 hang watchdog만이며 PASS 수치 아님 | `gates.normal_mock_load.{success,max_running,lost,virtual_makespan_ms,expected_ms}` |
| opt-in live | `tests/live/test_m42_live.py::test_fixed_12_cases` | `live` (`RUN_LIVE_LLM_TESTS=1`) | `evaluation/datasets/m42_safe_serving_live.json`, schema `m42-live-manifest-v1`, unique `case_id` 정확히 12, 각 `{case_id,question,expected_success:true}`; SHA-256, warm host, conc=1 | accepted/5xx/timeout + 별도 M3 command | 12/12, 5xx=timeout=0; `venv/bin/python scripts/run_m4_regression_gate.py --output-dir evaluation/reports/m4_regression` exit 0 및 14개 gate를 별도 artifact로 기록 | `live.{manifest_path,manifest_schema,manifest_sha256,case_results,accepted,five_xx,timeouts,m3_command,m3_exit,m3_14_gate_artifact}` |

`test_profile_node_inventory_exact`는 `tuple(PROFILE_NODE_IDS)`가 위 10 profile 순서와 정확히 같고 각 tuple이 위 catalog cell을 byte-for-byte 재현하며, flatten node가 pytest collect 결과와 ordered equality이고 duplicate count 0임을 검사한다. `venv/bin/python scripts/run_m42_acceptance.py --profile deterministic --repeat 10 --seed 4202 --output evaluation/reports/m42_acceptance.json`는 이 validation 뒤 각 repeat를 새 subprocess/registry로 실행한다. 누락/추가/순서 drift/중복 node, repeat!=10, 어느 subprocess nonzero도 runner exit 1이다. 최종 receipt는 profile별 result가 정확히 10개이고, 각 profile에 속한 **각 node ID별 result도 정확히 10개**이며 `(profile, repeat_index)`와 `(node_id, repeat_index)`가 각각 unique임을 assert한다. conservation validation result도 repeat별 정확히 하나, 총 10개여야 한다. 각 run은 `repeat_index`, revision, settings hash, seed, Python/OS/CPU, exact command, ordered collected node IDs, per-gate fields, artifact SHA-256를 기록하며 flake 0만 PASS다.

terminal conservation은 `tests/unit/test_query_executor.py::test_terminal_conservation_all_origins`와 모든 deterministic run에 공통 적용한다. executor의 두 식과 `request_count == sum(request_terminal[x] for x in REQUEST_TERMINALS)`, `unknown==0`을 매 run 검사한다. `tests/unit/test_m42_acceptance_runner.py::test_negative_conservation_mismatch_exits_nonzero`는 deterministic profile inventory 밖의 **별도 negative control**로 한 completed count를 삭제해 `main()` exit 1, artifact `status=FAIL`, `unknown/mismatch` 진단을 검증한다. positive repeat receipt의 conservation 10개와 negative-control receipt 1개를 합치지 않는다. report에는 `conservation.{request_count,request_terminal_sum,accepted_lhs,accepted_rhs,submit_attempt_lhs,submit_attempt_rhs,unknown}`를 쓴다.

live는 `venv/bin/python scripts/run_m42_acceptance.py --profile live --manifest evaluation/datasets/m42_safe_serving_live.json --output evaluation/reports/m42_live.json`으로만 실행하며 deterministic aggregate 밖의 top-level `live` status로 12 case를 정확히 기록한다. M3 command/14-gate artifact는 별도 top-level `m3_regression` status이고, M4.1은 receipt 유무와 관계없이 별도 top-level `M4.1_BLOCKED`다. 세 status는 deterministic PASS나 서로의 count로 합성되지 않는다.

추가 negative/race symbols는 `test_submit_shutdown_and_broken_pool_rollback`(uncommitted submit 두 예외에서 ticket/timer/pending/running=0, 단일 internal), `test_completion_and_abandon_both_orders`, `test_loop_close_cross_abandon_matrix`(§2.7 네 행), `test_metrics_scrape_without_readiness_probe_is_current`, `test_metric_sink_failure_does_not_change_executor_state`, `test_ollama_remaining_budget_clients_are_distinct_and_closed`,
`test_concurrent_second_lifespan_rejected_before_global_mutation`,
`test_initial_validation_failure_leaves_process_uncommitted_and_releases_guard`,
`test_same_identity_reacquire_preserves_all_identities`,
`test_different_identity_rejected_before_all_mutation`,
`test_loader_failure_publishes_only_settings_invalid_transaction`,
`test_cancel_before_or_during_loader_stops_and_releases_without_executor`,
`test_constructor_failure_uses_initialized_zero_grace_cleanup`,
`test_shutdown_cleanup_error_and_cancellation_matrix`와 §4.3의 module/valid/invalid/CLI identity
matrix다. identity-isolation runner는 first commit/same reacquire/different reject case마다 fresh
subprocess를 쓰고 loader/cache/config/engine/executor/app-state identity와 count를 기록한다.
overlapping-lifespan cache restoration tests는 없다. 모든 ordering은
`Event`/barrier/FakeClock이며 wall-clock sleep을 쓰지 않고, fixture `finally`가 stall event를 release하고 executor를 shutdown해 test worker thread 0을 assert한다.

### 10.1 Design Review Iteration 1 폐쇄 대조

| Finding | 리뷰의 실패 state/line | 폐쇄 설계 |
|---|---|---|
| M42-DR1-001 | 옛 133~161: abandon이 `running`을 조기 감소하고 future callback no-op | §2.2/§2.6 두 guard, abandon-first/completion-first 표; future completion만 slot/orphan 감소·단일 promotion |
| M42-DR1-002 | 옛 163~178: loop-close fallback이 wake와 resource cleanup 혼합 | §2.7 asyncio-free resource 함수와 별도 notification, loop×abandon 4조합 |
| M42-DR1-003 | 옛 97~119: submit 예외 후 running/ticket 누수 | §2.4 Future-return commit point, shutdown/BrokenThreadPool direct·promotion rollback과 terminal internal |
| M42-DR1-004 | 옛 567~594: probe 사이 clear/re-full 유실 | §2.3/§8.2 executor edge timestamp/version snapshot과 `.4 clear/.6 re-full` negative control |
| M42-DR1-005 | 옛 449~474: singleton 고정 90초 timeout | §6.2 매 router/answer 호출 remaining-budget `ollama.Client`, 비공유 transport close, global mutation 0 |
| M42-DR1-006 | 옛 90~669: accepted/terminal 모순과 readiness 의존 stale gauge | §2.3/§9 두 보존식, rejection taxonomy, scrape atomic sync, metrics side-effect 격리 |
| M42-DR1-007 | 옛 657~695: 11 profiles/runner/report 추적 누락 | §10의 11개 1:1 행, repeat-10·flake·conservation·opt-in live 별도 계약 |
| M42-DR1-008 | 옛 384~391: create_app/lifespan 이중 loader와 16KiB fallback | §4.3~4.4 guard sole owner의 단일 attempt/result, successful cache commit-before-engine, immutable maximum bootstrap ceiling, 동일 Settings identity 공유; concurrent second는 mutation 전 reject |

### 10.2 Design Review Iteration 2 폐쇄 대조

| Finding | 폐쇄 evidence |
|---|---|
| M42-DR2-001 | §4.3 state table과 identity spies: module import loader 0, owner lifespan validation 1, concurrent second mutation 0, invalid health 유지, CLI exit 2 분리, cache commit-before-import, failure/shutdown guard release |
| M42-DR2-002 | §4.1/§4.1.1 실제 설치 pure-ASGI stack: body→observer handoff, shared arbiter, pre-Response sentinel, route-owned send, frame-0 disconnect terminal, loser/outer cleanup, 실제 app 각 100회 log/metric 포함 |
| M42-DR2-003 | §4.4 non-identity 400-before-receive와 identity consumed `limit+1`; oversized ASGI wire bytes는 별도 기록하고 server cap 부재를 명시 |
| M42-DR2-004 | §2.9 단일 lock/sequence/CAS winner와 waiter-owned absolute deadline; polling 없는 completion/deadline/tie/stale/zero 및 shutdown once |
| M42-DR2-005 | §10 ordered exact 10-profile mapping, caller 두 node, node/profile별 receipt 10개, 별도 negative/live 12/M3/M4.1 status |

### 10.3 Design Review Iteration 3 폐쇄 대조

| Finding | 상태 | exact closure evidence |
|---|---|---|
| M42-DR3-001 | CLOSED (recovery scope) | §4.3은 global mutation 전 single-active guard를 acquire하고 owner만 loader 결과를 cache commit한 뒤 engine/executor를 만든다. overlapping lifespan 자체를 deterministic reject하므로 stale predecessor 복원 상태가 없고, 모든 startup/shutdown failure에서 release/reacquire를 실행 검증한다. |
| M42-DR3-002 | CLOSED (recovery scope) | §4.1/§4.1.1은 route frame-0 terminal을 허용하는 pure-ASGI request-context까지 실제 stack에 포함한다. queued/running 각 100회 actual-app test가 frames/receive/tasks와 request ID/log/metric exactly-once를 함께 검증한다. |
| M42-DR3-003 | CLOSED | §4.4는 upstream `wire_delivered_bytes`와 sliced `application_consumed_bytes`를 분리한다. remaining+1만 소비하고 overflow 뒤 receive를 중단하며 single `limit+N`, multichunk, false/no length를 검사한다. 구체적 server/proxy cap 없이 이미 전달된 ASGI message를 예방한다고 주장하지 않는다. |
| M42-DR3-004 | CLOSED | §2.9는 resource-zero와 deadline이 같은 lock/monotonic sequence/CAS winner를 쓰고 absolute deadline을 waiter가 보존한다. completion/deadline first, tie 양 insertion order, stale, zero에서 exact winner/loser/pending/residual과 shutdown 정확히 1회를 검증한다. |
| M42-DR3-005 | CLOSED | §10의 literal `PROFILE_NODE_IDS`와 catalog는 byte-for-byte 동일하며 caller cancellation tuple은 queued/running 두 node다. ordered 10-profile/no-duplicate collection, profile/node/conservation별 정확히 10 results, 별도 negative control과 top-level live 12/M3/M4.1 status를 고정한다. |

### 10.4 Design Review Iteration 4 복구 폐쇄 대조

| Finding | 상태 | simplified-scope closure evidence |
|---|---|---|
| M42-DR4-001 | **CLOSED** | §4.1.1의 `RequestContextMiddleware`는 `BaseHTTPMiddleware`를 제거한 pure ASGI wrapper다. installed Starlette 0.50.0 prototype은 기존 no-response가 `RuntimeError("No response returned.")`임을 재현했고 pure wrapper가 frames 0, exception 0, `client_disconnected`/499-equivalent end·metric 각 1을 허용함을 보였다. 구현 acceptance는 actual `create_app()`에서 request ID/start/end/duration/counter 각 1까지 고정한다. 499 response는 보내지 않는다. |
| M42-DR4-002 | **CLOSED** | §4.3은 process당 active lifespan 1개만 지원한다. 두 번째 acquire는 settings/cache/engine/executor mutation 전 고정 오류로 실패하고, sole owner는 startup failure와 모든 shutdown path에서 guard를 release한다. prototype이 first acquire, concurrent reject, failure release, shutdown release, 두 reacquire를 확인했다. previous-cache generation/lease/rollback 복잡성은 설계와 test catalog에서 제거됐다. |

### 10.5 Design Recovery Review Iteration 1 폐쇄 대조

| Finding | 상태 | approved simplified-scope closure evidence |
|---|---|---|
| M42-RR1-001 | **CLOSED** | §4.3은 최초 successful commit의 exact `Settings` object를 production process lifetime 동안 immutable로 고정한다. sequential same identity만 재획득하고 equal-value different identity는 loader 뒤 app/cache/config/engine/executor mutation 전에 고정 오류로 거부한다. module ASGI와 CLI preflight가 같은 commit/verify primitive를 쓰며 reset/reload/rebind/rollback/generation은 없다. fresh-subprocess matrix가 initial validation failure, first commit, same reacquire, different reject의 loader count와 모든 `is`를 고정한다. |
| M42-RR1-002 | **CLOSED** | §4.3.2는 primary/cancellation을 먼저 보존하고 begin, applicable bounded wait, mandatory non-waiting shutdown을 독립 exactly-once guard로 수행한다. STOPPED publish 뒤 guard를 마지막에 release하며 cleanup cancellation은 shield/defer한다. fixed primary/secondary policy와 deterministic error/cancellation/residual matrix가 ordering, counts, propagation, reacquire point를 고정한다. |

### 10.6 Design Recovery Review Iteration 2 폐쇄 대조

| Finding | 상태 | Iteration 3 exact closure evidence |
|---|---|---|
| M42-RR2-001 | **CLOSED** | §4.3.1/§4.3.4는 guard acquire 직후 local `candidate=None`, `executor=None`, `grace=0.0`, trace만 초기화하고 loader 결과를 local candidate에 둔다. `commit_process_settings_once(candidate)`가 first/same을 확인한 뒤에만 app state, cache-dependent facade, engine, executor와 configured grace를 touch한다. different identity는 health transaction 없이 app/cache/config/engine/executor delta 0이며, invalid loader만 REQ-009.2가 허용한 atomic `settings_invalid` failed-start diagnostic을 별도 publish한다. §10 fresh-subprocess spies와 [Recovery Validation §10](Design_Recovery_Validation.md)이 ordering과 delta를 실행 characterize한다. |
| M42-RR2-002 | **CLOSED, RR3 refined** | §4.3.2/§4.3.4의 outer finally는 초기화된 executor/grace와 attempt class로 cleanup에 진입한다. executor `None`이면 begin/wait/shutdown 0이고, lifecycle ownership을 publish한 constructor/cancellation path만 STOPPED→release를 수행한다. identity mismatch는 lease-local release만, invalid loader는 단일 diagnostic 뒤 release만 수행한다. teardown argument/task 생성과 shield evaluation까지 primary-preserving이며 executor가 존재하면 shutdown을 정확히 한 번 시도한다. [Recovery Validation §11](Design_Recovery_Validation.md)이 refined ordering을 bounded 실행한다. |

### 10.7 Design Recovery Review Iteration 3 폐쇄 대조 (final base iteration)

| Finding | 상태 | Iteration 4 exact closure evidence |
|---|---|---|
| M42-RR3-001 | **CLOSED** | §4.3.1은 identity mismatch를 pre-publication lease-local attempt로 분류한다. exact-owner guard release 외 `app.__dict__`, health/log/metric sinks, process cache/config, engine/executor factories와 STOPPED의 full-attempt delta가 모두 0이다. invalid loader는 별도 fail-soft bootstrap owner로 atomic `settings_invalid` transaction 정확히 1개만 소유하고 generic stopped observer는 overwrite/add하지 않는다. [Recovery Validation §11](Design_Recovery_Validation.md)의 bounded spies가 이를 characterize한다. |
| M42-RR3-002 | **CLOSED** | §4.3.2는 canonical `STOPPED→release`를 lifecycle ownership을 publish한 started/partially-started attempt에만 한정한다. 모든 fallible observer/snapshot/error aggregation은 먼저 끝나고 non-throwing atomic STOPPED publication과 exact-owner release가 final two durable external actions다. release primitive는 durable owner clear 뒤 bounded diagnostic만 반환하며 post-release 처리는 non-durable/best-effort라 reacquire를 막지 않는다. [Recovery Validation §11](Design_Recovery_Validation.md)이 ordering과 reacquire를 bounded characterize한다. |

Iteration 4의 나머지 closure 행(M42-DR1-001~007, DR2-003~005, DR3-003~005)은
기존 판정을 유지한다. DR1-008/DR2-001/DR3-001은 M42-DR4-002의 단일-lifespan scope로,
DR2-002/DR3-002는 M42-DR4-001의 pure-ASGI terminal owner로 함께 폐쇄된다. 따라서 복구 후
전체 설계 리뷰 closure matrix는 CRITICAL 0, MAJOR 0, MINOR 0이며 recovery review의
M42-RR1-001/002와 M42-RR2-001/002도 승인된 simplified scope 안에서 명시적으로 CLOSED다. 이는 prototype evidence이지 project product test
PASS 주장이 아니며 구현 후 §10 catalog가 별도로 통과해야 한다.

## 11. 잔여 위험과 미해결 항목

1. Ollama 0.6.0의 public close 부재 때문에 §6.2 adapter 한 곳만 version-pinned `_client.close()` seam을 사용한다. remaining budget 자체는 매 호출 constructor timeout으로 폐쇄됐지만 dependency upgrade 시 이 seam의 contract test가 실패하면 구현 Gate를 닫지 않는다.
2. caller execution timer와 upstream socket timeout이 동시에 임박할 수 있어 scheduler jitter로 worker가 caller보다 늦게 끝나는 짧은 orphan 구간은 남는다. 이는 §2의 정확한 resource accounting 대상이며 slot을 조기 반환하지 않는다.
3. M4.1 Operational Acceptance는 이 설계로 전혀 해소되지 않는다 — Traceability.md(Phase 6 산출물)에 `M4.1_BLOCKED` 행을 독립적으로 유지해야 한다(Requirement §6, Plan §1/§8).
4. M4.3(index lifecycle)/M5(외부 queue·다중 프로세스) 범위 침범 없음 — 이 설계가 다루는 파일은 §1 Symbol Inventory로 닫혀 있고 index/container/배포/외부 queue 관련 symbol을 하나도 도입하지 않는다.
5. 같은 process에서 동시에 둘 이상의 app lifespan을 active로 두는 것은 명시적 비지원이다.
   두 번째는 global mutation 전 deterministic startup failure다. 이는 multi-process worker 수를
   제한하는 계약이 아니라 각 process 내부의 app lifespan ownership만 제한한다.

**사용자 결정이 필요한 열린 항목: 없음.** concurrency=1/queue=4 기본값과 수치 계약은 Plan.md §9가 이미 승인한 안전한 초기값이며, 이 설계는 그 범위 안에서만(하향 조정 방향으로만) 세부를 확정했다.
