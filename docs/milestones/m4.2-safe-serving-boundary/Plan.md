# M4.2 Safe Serving Boundary 개발 계획

요구사항: [Requirement.md](Requirement.md)

운영 선행 위험: [M4.1 Operational Acceptance Exception](../m4.1-configuration-observability/Operational_Acceptance_Exception.md)

## 1. 실행 원칙과 순서

작업은 다음 순서를 건너뛰지 않는다.

```text
요구사항 -> 상세 설계 -> 설계 리뷰 Gate -> 구현 phases
-> 독립 code review iterations -> clean 검증
-> GitHub release(PR/merge) -> post-merge acceptance
```

Codex는 요구사항/계획과 독립 리뷰를, Claude Code는 상세 설계와 구현·Git 작업을 맡는
상위 [개발 orchestration guide](../../../milestone_dev_orchestration_guide.md)의 역할을
따른다. 같은 작성자가 자신의 설계 또는 코드를 최종 승인하지 않는다. Gate FAIL이면
발견 ID, 재현, 요구사항 연결을 포함한 개선 iteration 후 같은 Gate를 다시 연다.

M4.1 운영 acceptance는 미완료이며 M4 release blocker다. 이 계획은 사용자 예외에 따라
M4.2를 진행하지만 M4.1 PASS를 선행조건이나 예상 결과로 두지 않는다. 모든 traceability와
최종 보고서에 `M4.1_BLOCKED`를 유지한다.

## 2. Phase 0 — 요구사항 기준선 고정

파일/근거:

- `docs/milestones/m4.2-safe-serving-boundary/Requirement.md`, `Plan.md`
- `web/server.py::rag_query`, `_make_lifespan`
- `agent.py::route_query`, `settings.py::FIELD_SPECS`
- `observability.health.evaluate_readiness`, `metrics.build_metrics_registry`

작업:

1. `0c84795`의 symbol/signature/test inventory를 기록하고 상위 M4 문서의 미구현 가상
   API와 현재 코드를 구분한다.
2. M4.2 요구사항 ID를 새 `Traceability.md` 초안에 옮기고 각 행을 PLANNED로 둔다.
3. M4.3/M5 제외 범위와 M4.1 blocker를 별도 행으로 고정한다.

완료 조건: 모든 요구사항에 수치, terminal state/error, 최소 한 개 negative/race test가
있고 열린 제품 결정이 없다. 변경은 M4.2 문서에 한정한다.

## 3. Phase 1 — 상세 설계와 설계 리뷰 Gate

상세 설계 파일: `docs/milestones/m4.2-safe-serving-boundary/Design.md`.

설계가 고정할 symbol/API:

- `web/concurrency.py::{TicketState,QueryExecutor,ExecutorSnapshot}`와 단일 finalize 알고리즘
- `web/server.py::{_make_lifespan,rag_query}`의 lifecycle/admission/error mapping
- `observability/health.py::evaluate_readiness`의 lifecycle/saturation 입력과 우선순위
- `settings.py::{FIELD_SPECS,MODEL_VALIDATORS}`의 8개 설정과 cross-field 검증
- `observability/metrics.py::build_metrics_registry`의 bounded series 계산
- request body limiter와 deadline context/네트워크 adapter의 정확한 모듈·signature

설계에는 ticket 상태표, cancel/timeout/future-completion race별 ownership, drain sequence,
readiness debounce clock, ASGI chunk 수신, upstream deadline 전파, singleton thread-safety
증거를 포함한다. concurrency=2는 승인된 기본이 아니며 component stress 증거가 있을 때만
별도 변경 대상으로 검토한다.

독립 Codex 설계 리뷰는 다음을 모두 확인한다.

- queue insert/grant/remove와 resource finalize가 exactly-once인가
- 실행 timeout 후 slot을 조기 반환하지 않는가
- loop close 뒤 callback이 안전하고 shutdown이 실행 thread를 기다리지 않는가
- body limit가 Content-Length 없는 chunked 입력에도 적용되는가
- LangChain/Ollama/DDGS timeout이 실제 constructor/per-call API와 맞는가
- metric/readiness label 상한과 기존 M4.1 호환성이 계산됐는가
- 테스트가 sleep 대신 event/barrier/fake clock으로 race를 결정하는가

Gate: MAJOR 0, MINOR 0, requirement-to-design 누락 0일 때만 Phase 2로 진행한다. 리뷰 문서는
`Design_Review_Iteration_N.md`, 개선은 새 iteration으로 보존한다.

## 4. 구현 phases

### Phase 2 — settings, 오류와 input boundary

파일/symbol:

- `src/simple_qna_rag/settings.py::FIELD_SPECS`, `MODEL_VALIDATORS`
- `src/simple_qna_rag/web/server.py::rag_query` 및 body-limit seam
- 필요 시 새 `src/simple_qna_rag/web/errors.py`
- `tests/unit/test_settings*.py`, 새 `tests/integration/test_web_input_boundary.py`
- generated field spec과 관련 inventory 생성기 산출물

테스트: 각 설정 min/max/NaN/무한대/unknown key/cross-field, content-length 조기 거부,
chunked limit+1, JSON media/schema/control char/question length, 고정 오류 body와 payload 누출
negative fixture.

완료 조건: invalid input은 executor submit 0, settings invalid는 exit 2, 기존 CLI/settings
호환 suite PASS, generated check diff 0.

Rollback: 새 setting 소비를 제거하고 기존 41-field generated artifact로 되돌리는 단일
commit이 가능해야 한다. input limiter flag로 우회하는 fail-open rollback은 허용하지 않는다.

### Phase 3 — QueryExecutor, timeout/cancel과 orphan

파일/symbol:

- 새 `src/simple_qna_rag/web/concurrency.py`
- `web/server.py::{_make_lifespan,rag_query}`
- 새 `tests/unit/test_query_executor.py`
- 새 `tests/integration/test_web_concurrency.py`

테스트: admission 1/2/5 profile, FIFO, queue head cancel/timeout, submit 실패, cancel-before-submit,
cancel-versus-completion, timeout-versus-completion, callback-after-loop-close, 100회 반복 race,
event-loop health probe. 모든 stall은 `threading.Event`로 명시적으로 release한다.

완료 조건: Requirement §5 executor gate 전부 PASS, 정확한 terminal enum 합계, 반복 10회
flake 0, suite 종료 후 살아 있는 test worker thread 0.

Rollback: `/rag` offload 변경과 executor 모듈을 함께 revert한다. queue를 unbounded로
바꾸거나 timeout 때 slot을 조기 반환하는 부분 rollback은 금지한다. 긴 작업이 문제면
concurrency=1/queue=0으로 더 보수적으로 설정한다.

### Phase 4 — lifecycle, readiness와 metrics

파일/symbol:

- `web/server.py::_make_lifespan`
- `web/concurrency.py::{begin_drain,wait_drained,shutdown,snapshot}`
- `observability/health.py::evaluate_readiness`
- `observability/metrics.py::{READINESS_REASONS,build_metrics_registry}`
- 새 `tests/integration/test_shutdown_drain.py`,
  `tests/integration/test_readiness_saturation.py`
- 기존 `tests/integration/test_health_endpoints.py`, `test_metrics_live_traffic.py`

테스트: state transition 전수, idle 즉시 drain, running 완료, queued wake/reject, grace 만료,
shutdown idempotency, fake-clock 1초 saturation enter/exit, 1,000 unique question cardinality.

완료 조건: DRAINING 이후 submit 0, queued 0, grace 만료 path가 worker를 기다리지 않고 반환,
orphan/running 불변식 유지, 기존 health 계약과 7개 metric family 회귀 PASS.

Rollback: lifecycle/readiness/metric 확장을 executor와 원자적으로 revert한다. readiness만
`ok`로 강제하는 rollback은 금지한다.

### Phase 5 — upstream deadline/network boundary

파일/symbol:

- `agent.py::{_get_router_llm,route_query}`
- `rag_engine.py::{_initialize_llm,query}`
- `web_search.py::search_web`
- 상세 설계가 정한 deadline context/adapter 모듈
- 새 `tests/unit/test_network_deadline.py`, 기존 agent/web-search tests

테스트: connect timeout 전달, remaining budget 감소, expired-before-call, retry 0, Ollama/DDGS
stall fake, timeout error 정규화, 전역 client mutation 0. live 외부 호출은 opt-in 별도 profile로
두고 unit Gate의 대체물이 되지 않는다.

완료 조건: 모든 외부 호출이 유한 budget을 갖고 호출자 반환은 execution deadline+100ms
이내다. 보장할 수 없는 library 경로는 설계 Gate로 되돌리며 문서만으로 PASS하지 않는다.

Rollback: network adapter만 이전 client construction으로 되돌릴 수 있으나 전체 executor
deadline은 유지한다. 되돌린 경로가 무한 network wait를 재도입하면 release candidate를
폐기한다.

### Phase 6 — 결정론적 load acceptance와 tuning

파일/symbol:

- 새 `evaluation/m4_safe_serving_load.py`와 단위 테스트
- 새 `scripts/run_m42_acceptance.py` 또는 상세 설계가 정한 단일 runner
- `docs/milestones/m4.2-safe-serving-boundary/Traceability.md`

고정 report는 revision, settings hash, seed, Python/OS/CPU, profile, 요청별 terminal enum,
max queue/running/orphan, latency percentile와 artifact SHA-256을 기록한다. mock gate는 §5의
profiles를 10회 실행한다. opt-in live는 concurrency=1 고정 12 case와 M3 gate를 실행하되
M4.1 운영 receipt와 혼동하지 않는다.

완료 조건: Requirement §5 전부 PASS, 미분류 요청 0, report 재실행 시 schema/분류 동일,
concurrency 기본 1 유지. 2로 올리는 변경은 별도 stress evidence와 설계 리뷰가 필요하다.

Rollback: 성능 목표 실패 시 한도는 낮출 수 있다. queue 확대나 external queue/autoscaling은
M4.2 rollback/tuning 수단이 아니다.

## 5. 독립 code review iterations

모든 구현 phase 후 fresh Codex reviewer가 전체 diff와 current code를 읽고 다음 순서로
검토한다.

1. Requirement/Design/Traceability 누락과 범위 침범
2. executor ownership/race/drain correctness
3. fail-closed input/network/error 계약과 payload leakage
4. deterministic tests의 실제 결함 검출력(negative control 포함)
5. 기존 M3/M4.1 API·logging·metrics 회귀

결과는 `Code_Review_Iteration_N.md`에 severity, file:line, 재현, 요구사항 ID로 기록한다.
MAJOR/MINOR가 하나라도 있으면 Claude Code 개선 Task와 새 독립 iteration을 수행한다.
Gate는 MAJOR 0, MINOR 0, 전체 suite PASS, `git diff --check` PASS일 때만 닫는다.

## 6. clean 검증 Gate

추적되지 않은 local artifact와 공유 venv에 의존하지 않는 fresh checkout/venv에서 아래를
실행한다. 정확한 명령은 Design에서 package manager와 runner에 맞춰 고정한다.

```bash
bash scripts/compile_lock.sh --verify
python -m pip check
python -m pytest -q
npm ci
npm test
python scripts/generate_field_spec.py --check
python scripts/logging_callsite_audit.py --check
python scripts/check_markdown_links.py
python scripts/run_m42_acceptance.py --profile deterministic --repeat 10
git diff --check
```

`RUN_LIVE_LLM_TESTS=1` profile은 신뢰할 수 있는 Ollama/vectorstore host에서 별도로 실행한다.
clean Gate는 deterministic 필수 gate와 opt-in live 결과를 구분해 기록하며, live 환경 부재를
PASS로 바꾸지 않는다. 완료 조건은 모든 필수 local gate PASS, artifact hash/command/exit
receipt, M4.1 blocker의 명시적 잔존이다.

## 7. GitHub release와 post-merge acceptance

이 Task에서는 실행하지 않는다. 구현·리뷰·clean Gate 승인 후 Claude Code가 범위를 확인해
단일 feature branch를 commit/push하고 draft PR을 만든다. PR에는 Requirement/Design/
Traceability, review iterations, deterministic report, known M4.1 blocker와 rollback 절차를
연결한다. required CI가 모두 성공하고 독립 reviewer가 merge SHA를 승인한 뒤에만 merge한다.

merge 후에는 `master`의 정확한 SHA로 lock/install, 전체 suite, deterministic 10회 load,
opt-in live 12-case와 M3 gate를 재실행하고 GitHub run/job/artifact provenance를 검증한다.
실패하면 M4.2를 완료로 표시하지 않고 release candidate를 revert하거나 수정 PR을 연다.
M4.2 acceptance가 성공해도 M4.1 Operational Acceptance는 자동으로 해소되지 않으며,
M4 release 전 별도 live 14-gate receipt 또는 사용자의 별도 release-risk 승인이 필요하다.

## 8. 위험과 통제

| 위험 | 통제/중단 기준 |
|---|---|
| singleton model/index thread race | 기본 concurrency=1; 2는 component stress+설계 리뷰 전 금지 |
| timeout/cancel 뒤 실행 thread 잔존 | slot을 future 완료까지 유지, orphan gauge, overload fail-closed |
| queue race로 capacity 누수/초과 | event-loop 단일 owner, explicit FIFO ticket, single finalize, barrier tests |
| shutdown이 process를 붙잡음 | bounded grace 후 pool `wait=False`; 잔존 수 기록, hard bound라고 주장 금지 |
| readiness flap | fake-clock 1초 enter/exit debounce; admission 거부는 즉시 |
| body streaming 우회 | ASGI 누적 byte limit+1, Content-Length 유무 양쪽 negative test |
| library timeout이 실제로 무시됨 | stall fake로 caller deadline 측정; 실패 시 adapter/process 경계 재설계 |
| metric/log payload 또는 cardinality 증가 | allowlist, 1,000 unique input gate, forbidden-token capture |
| M4.1 미완료가 묻힘 | Traceability `M4.1_BLOCKED`, PR/release checklist의 필수 blocker |
| M4.3/M5 scope creep | index/container/deploy/external queue/autoscaling diff가 있으면 Gate FAIL |

## 9. 사용자 결정

현재 착수에 필요한 열린 사용자 결정은 없다. 기본 concurrency=1, queue=4와 수치 계약은
안전한 초기값으로 설계 리뷰와 load Gate에서 **하향 조정만** 가능하다. 범위 확대,
concurrency=2 기본화, 오류 HTTP 계약 변경, M4.1 release-risk 수용은 별도 사용자 결정이
필요하다.
