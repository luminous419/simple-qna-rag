# M4.2 Safe Serving Boundary — Design Review Iteration 1

검토 대상: [Requirement.md](Requirement.md), [Plan.md](Plan.md), [Design.md](Design.md), [개발 orchestration guide](../../../milestone_dev_orchestration_guide.md), 기준 revision `0c84795`의 현재 코드, 현재 설치 환경의 `ollama 0.6.0`, `langchain-ollama 0.3.3`, `ddgs 9.14.4`, `httpx 0.28.1` API.

## 판정

**FAIL — 5.8/10.0**

| Severity | 수 |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 7 |
| MINOR | 1 |

Gate 조건인 CRITICAL/MAJOR 0, MINOR 최소화(Plan의 더 엄격한 기준은 MINOR 0), score >= 9.7, requirement-to-design 누락 0을 충족하지 못한다. Phase 2 구현으로 진행하면 안 되며, 설계 작성자가 아래 항목을 닫은 새 설계 iteration 뒤 fresh 독립 리뷰가 필요하다.

## 발견사항

### M42-DR1-001 — MAJOR — `ABANDONED`가 caller terminal과 resource terminal을 분리하지 않는다

- 위치: `Design.md:133`, `Design.md:140`, `Design.md:150`, `Design.md:159`, `Design.md:161`
- 재현 상태 전이: `RUNNING(running=1, orphaned=0, finalized=False)` → execution timer가 선점 → `_finalize_locked(..., ABANDONED)`가 `finalized=True`, `running-=1`을 수행 → caller에 504. 이때 underlying pool future는 여전히 stall 중인데 snapshot은 `running=0, orphaned=0`이고 다음 queued ticket을 승격한다. 이후 pool future 완료 callback은 `finalized` guard에서 no-op가 된다.
- 요구사항 영향: M4.2-REQ-002.2/002.3, 003.2~003.4, 004.1/004.2, 005.3/005.4와 execution-timeout/cancellation/drain 정량 Gate를 직접 위반한다. 특히 `orphaned <= running`만 수치상 유지하면서 실제 실행 thread 수가 concurrency limit를 초과할 수 있어 가장 위험한 fail-open이다.
- 수정 조건: caller outcome exactly-once와 resource finalize exactly-once를 별도 ownership/guard로 모델링한다. timeout/cancel은 `RUNNING -> ABANDONED`, caller wake/counter 1회, `orphaned += 1`만 수행하고 `running`/slot은 유지해야 한다. 오직 pool future completion이 `running -= 1`, `orphaned -= 1`, resource-finalized 표시, 단일 승격을 수행해야 하며, 양 순서의 race 표와 executable pseudocode가 필요하다.

### M42-DR1-002 — MAJOR — loop-close fallback도 동일 finalize를 호출해 asyncio wake와 resource 정리를 분리하지 못한다

- 위치: `Design.md:163`, `Design.md:168`, `Design.md:173`, `Design.md:175`, `Design.md:178`
- 재현 상태 전이: shutdown이 STOPPED로 전이하고 loop close → worker 완료 → `call_soon_threadsafe`가 `RuntimeError` → worker thread가 `_finalize_locked` 호출 → 그 함수의 무조건적인 `_wake_caller(ticket)`가 closed-loop 소유 `asyncio.Future`를 worker thread에서 조작한다. 이미 caller-finalized ABANDONED라면 반대로 guard에서 resource 정리가 아예 생략된다.
- 요구사항 영향: M4.2-REQ-003.4, 004.1, 005.3/005.4 및 callback-after-loop-close Gate. “caller wake-up만 skip”이라는 설명과 제시된 pseudocode가 서로 다르다.
- 수정 조건: loop 독립적인 resource-completion 경로를 만들고 asyncio 객체를 절대 만지지 않게 한다. loop가 살아 있을 때만 별도의 caller-completion 함수를 marshal하며, loop-close 전/후와 caller-abandon 전/후의 4개 조합에서 resource counter와 ticket 제거가 정확히 한 번임을 상태표로 증명한다.

### M42-DR1-003 — MAJOR — pool submit 실패가 admission 원자성을 깨뜨린다

- 위치: `Design.md:97`, `Design.md:105`, `Design.md:119`, `Plan.md:107`
- 재현 상태 전이: READY, `running < limit` → `_new_ticket()` → `_admit_running()`이 `running += 1` 및 deadline 설정 → `ThreadPoolExecutor.submit()`이 shutdown 경쟁 또는 `BrokenThreadPool`로 예외 → handle도 caller outcome도 없이 running/ticket capacity가 유실된다. Plan은 “submit 실패” 테스트를 요구하지만 Design에는 rollback/finalize 계약이 없다.
- 요구사항 영향: M4.2-REQ-002.2/002.3, 003.4, 004.2, 005.1~005.4.
- 수정 조건: submit 성공 전후의 commit point를 명시하고 실패 시 ticket/카운터/timer/pending callable을 원자적으로 되돌리며 안전한 terminal error를 1회 반환해야 한다. `shutdown`/`begin_drain`/submit 경쟁을 포함한 negative test를 traceability에 연결한다.

### M42-DR1-004 — MAJOR — readiness debounce는 상태 전이가 아니라 polling 관찰을 debounce한다

- 위치: `Design.md:567`, `Design.md:578`, `Design.md:587`, `Design.md:594`
- 재현 상태 전이: t=0 ready probe에서 full 관찰 → t=0.4 full 해제 → t=0.6 재포화 → 중간 probe 없음 → t=1.0 probe는 `currently_full=True == pending`이므로 최초 t=0부터 연속 full로 오판하고 503을 낸다. 반대 방향도 동일하게 연속 1초 해제를 보장하지 못한다.
- 요구사항 영향: M4.2-REQ-006.2 및 saturation readiness 정량 Gate.
- 수정 조건: admission/resource-finalize가 만드는 실제 full/non-full edge를 단일 clock domain에서 기록하거나, readiness가 연속성을 증명할 수 있는 executor timestamp/version snapshot을 소비하게 한다. 중간에 해제 후 재포화되는 negative control을 fake monotonic clock으로 추가한다.

### M42-DR1-005 — MAJOR — Ollama 고정 timeout은 남은 execution deadline 계약을 만족하지 않는다

- 위치: `Requirement.md:165`, `Design.md:449`, `Design.md:457`, `Design.md:463`, `Design.md:472`, `Design.md:474`, `Design.md:697`
- API 검증: 설치된 `ollama.Client.generate()`/`chat()`은 per-call `timeout` kwarg를 받지 않고, `ollama._client.BaseClient.__init__`은 생성 시 `httpx.Client(timeout=...)`를 고정한다. `OllamaLLM`과 `ChatOllama`의 `sync_client_kwargs`도 생성 시 Client 구성에만 쓰인다. 이 부분의 API 관찰은 맞지만 설계 결론은 요구사항을 충족하지 않는다.
- 재현 상태 전이: execution budget 90초 중 router가 89초 소비 → answer 호출 전 1초 잔여지만 singleton client read timeout은 여전히 90초 → caller는 상위 timer로 1초 후 ABANDONED가 되어도 underlying call은 최대 약 90초 더 실행된다. “전체 호출 budget은 남은 execution deadline을 넘지 않는다”가 깨지고 concurrency=1 slot이 장기간 orphan으로 고정된다.
- 요구사항 영향: M4.2-REQ-008.1/008.2. REQ-008.3의 caller 반환 보장은 underlying upstream deadline 계약을 대체하지 않는다.
- 수정 조건: 매 호출 남은 budget으로 구성한 비공유 client/adapter 또는 검증 가능한 process 경계를 선택하고 실제 signature와 lifecycle을 고정한다. 불가능하다면 Requirement 변경 승인을 받아야 하며, 단순 concurrency=1 유지나 expired-before-call 검사만으로 PASS할 수 없다.

### M42-DR1-006 — MAJOR — metric/outcome exactly-once 회계가 모순되고 live gauge 갱신 경로가 불완전하다

- 위치: `Design.md:90`, `Design.md:121`, `Design.md:153`, `Design.md:622`, `Design.md:640`, `Design.md:655`, `Design.md:669`
- 재현 상태 전이: admitted ticket은 `accepted` 증가 대상이지만 완료 시 `completed`도 증가한다. 그런데 §10은 모든 ticket이 6개 counter 중 “정확히 하나”라고 주장한다. 또한 gauge sync는 제시된 wiring상 `/health/ready`에서만 호출되어 `/metrics`를 직접 scrape하면 최근 admission/completion 뒤 stale 값을 노출한다. Prometheus increment를 lock 해제 직후 한다는 설명도 두 작업 사이 snapshot/exception에 대한 일관성 규칙이 없다.
- 요구사항 영향: M4.2-REQ-004.2, 006.3/006.4와 “총 요청 수 = terminal 결과 enum 합계” 정량 Gate.
- 수정 조건: accepted를 lifecycle counter로 명시해 terminal outcome 합계에서 제외하고, rejected/not_ready/overloaded 및 queued drain의 분류를 포함한 보존식을 정확히 정의한다. `/metrics` scrape 시 atomic snapshot으로 gauge를 동기화하거나 state transition에서 갱신하며, Prometheus 실패가 executor state를 손상시키지 않는 side-effect 경계를 설계한다.

### M42-DR1-007 — MAJOR — 정량 Gate 전부에 대한 실행 가능한 traceability가 없다

- 위치: `Requirement.md:186`, `Requirement.md:205`, `Design.md:657`, `Design.md:683`, `Design.md:695`
- 재현 상태: Requirement의 11개 고정 profile 및 repeat-10/terminal conservation 중 Design의 race catalog는 R1~R7만 제시한다. event-loop 20회 latency 산출 방식, 정상 mock load의 single-thread 예상치 계산, bounded admission exact receipt, opt-in live 12 case와 M3 14-gate 별도 기록, 전체 repeat runner/report symbol이 상세 설계의 symbol/test에 완전히 연결되지 않는다. §10의 일반적인 테스트 문구만으로 clean receipt를 생성할 정확한 runner/API가 결정되지 않았다.
- 요구사항 영향: M4.2-REQ-009.3, Requirement §5/§6 및 requirement-to-design 누락 0 Gate. 누락 수는 최소 1(정량 acceptance 묶음)이다.
- 수정 조건: 각 정량 Gate를 고유 test/runner symbol, fixture clock/event, 입력 profile, 측정식, pass assertion, report field에 1:1로 연결한다. repeat 10, flake 0, terminal conservation과 opt-in/live 분리까지 포함한 표를 완성한다.

### M42-DR1-008 — MINOR — body-limit 설정 획득이 fail-closed 단일 로드 계약과 불일치한다

- 위치: `Requirement.md:51`, `Requirement.md:67`, `Design.md:384`, `Design.md:391`, 현재 `src/simple_qna_rag/web/server.py:168`
- 재현 상태: `create_app()`에서 `settings_loader()`를 한 번 호출해 middleware 값을 얻고 lifespan에서 다시 호출하는 설계는 stateful/custom loader에서 서로 다른 설정을 만들 수 있다. 첫 로드 실패 시 16,384로 대체한다는 문구는 invalid 값을 기본값으로 대체하지 않는다는 fail-closed 표현과도 충돌한다.
- 요구사항 영향: M4.2-REQ-001.2, 007.1, 009.2.
- 수정 조건: validated Settings를 정확히 한 번 로드해 lifespan과 limiter가 같은 immutable 값을 소비하거나, startup 전에도 fail-closed body ceiling을 보장하는 별도 bootstrap 상수와 제품 설정의 관계를 명시한다. invalid 설정을 정상 default load로 표현하지 않는다.

## 요구사항 추적성 요약

| REQ | 설계 연결 | 판정 |
|---|---|---|
| 001 | §7, body middleware 초기화 | 부분 충족 — M42-DR1-008 |
| 002 | §2 | 실패 — M42-DR1-001/003 |
| 003 | §2.4~2.8, §4.1 | 실패 — M42-DR1-001/002 |
| 004 | §2.6/2.10, §3, §9 | 실패 — M42-DR1-001/006 |
| 005 | §2.9, §4.3 | 실패 — M42-DR1-001/002/003 |
| 006 | §8~9 | 실패 — M42-DR1-004/006 |
| 007 | §4.2/4.4 | 부분 충족 — M42-DR1-008 |
| 008 | §5~6 | 실패 — M42-DR1-005 |
| 009 | §4/8/9/10 | 실패 — M42-DR1-007 |
| 정량 Gate | §10~11 | 누락/불완전 — M42-DR1-007 |

`M4.1_BLOCKED`는 Requirement, Plan, Design에 계속 명시되어 있으며 PASS로 재분류되지 않았다. index lifecycle/container/deployment/external queue/multi-process를 도입하지 않아 M4.3/M5 범위 침범도 발견하지 않았다.

## 재리뷰 진입 조건

1. caller terminal finalize와 underlying resource finalize를 별도 상태/guard로 재설계하고 모든 timeout/cancel/completion/loop-close 순서를 표로 고정한다.
2. submit 실패, drain/grace, FIFO 단일 승격, saturation edge debounce에 executable pseudocode와 negative control을 추가한다.
3. Ollama remaining-deadline adapter를 실제 설치 API로 구현 가능하게 결정하거나 Requirement 변경 승인을 받는다.
4. outcome 보존식과 live metrics 갱신 시점을 고정하고 11개 정량 Gate 각각을 test/runner/report symbol에 연결한다.
5. 새 Design iteration에 대해 markdown link 검사와 `git diff --check`를 통과시킨 뒤 fresh 독립 설계 리뷰를 수행한다.
