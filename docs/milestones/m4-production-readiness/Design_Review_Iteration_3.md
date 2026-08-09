# M4 Production Readiness 상세 설계 리뷰 — Iteration 3

검토일: 2026-08-08  
검토 대상: [Requirement](Requirement.md), [Plan](Plan.md), 최신
[Design](Design.md), [Traceability](Traceability.md),
[Iteration 1 리뷰](Design_Review_Iteration_1.md),
[Iteration 2 리뷰](Design_Review_Iteration_2.md), 현재 제품 코드·CI·packaging 구조  
프로세스 기준: [milestone 개발 orchestration guide](../../../milestone_dev_orchestration_guide.md)  
검토 방식: Iteration 2의 M2-01~M2-06 및 m2-01~m2-04를 먼저 실제 코드 seam과
대조하고, race/trust boundary/clean build/fresh evidence 실행을 실패시키는 방향으로
공격적으로 검증

## 1. 결론

**Gate: FAIL**  
**점수: 4.6 / 10.0**  
**발견사항: CRITICAL 0 / MAJOR 7 / MINOR 2 / TRIVIAL 0**

Iteration 3 설계는 이전 리뷰의 용어와 표를 상당히 구체화했지만 구현 Gate를 열 수
있는 상태는 아니다. `QueryExecutor`는 QUEUED ticket이 grant되는 순간 caller가
취소되면 future를 만들기도 전에 slot을 영구 누수한다. `DeadlineBudget`은 caller의
응답 deadline만 제공할 뿐 router/answer/DDGS worker를 overall deadline 안에
회수하지 못하고, 제안한 request-scoped LLM 복사 seam도 현재 객체 형태와 맞지 않는다.
`ObservationSink` 의사코드는 실제 `RetrievalTrace` 계약에 없는 메서드와 필드를 쓴다.

보안·검증 경계도 닫히지 않았다. legacy import는 상위 디렉터리 교체 TOCTOU를 막지
못하며, Dockerfile의 마지막 stage가 `test`라 target 없는 production build가 mock
코드를 포함한다. gate DAG는 전체 pytest 안에서 자기 자신을 다시 실행하는 재귀,
fresh directory와 선행 container artifact의 양립 불가, live/container evidence
수집 경로 부재를 동시에 가진다. 따라서 guide 기준인 9.7 이상, MAJOR 0, MINOR
최소화에 미달한다.

## 2. Iteration 2 발견사항 폐쇄 감사

| 이전 발견 | 판정 | 근거와 잔여 사항 |
|---|---|---|
| M2-01 executor 단일 finalize/race | **미해결** | worker future 완료 이후 release는 개선됐지만, queue grant와 caller cancel 경쟁에서 아직 future/callback 없는 RUNNING ticket이 남는다. 아래 M3-01. |
| M2-02 absolute overall deadline/seam | **미해결** | 같은 budget 객체 전파는 명시했으나 실제 router는 `RunnableBinding`, answer는 engine chain, DDGS는 별도 client다. worker 전체 종료 경계도 없다. 아래 M3-02. |
| M2-03 ObservationSink/trace | **미해결** | event allowlist는 맞췄지만 의사코드가 현재 `RetrievalTrace` API와 호환되지 않고 injected sink 예외 격리도 일관되지 않다. 아래 M3-03. |
| M2-04 legacy expected hash/TOCTOU | **미해결** | 필수 expected hash와 final-component `O_NOFOLLOW`는 추가됐지만 parent component 교체 및 승인 provenance가 남는다. 아래 M3-04. |
| M2-05 clean Docker/package/layer scan | **미해결** | 표준 venv와 layer archive 해제는 개선됐지만 기본 build가 test stage이고 clean packaging/evidence/canary 검증이 성립하지 않는다. 아래 M3-05. |
| M2-06 fresh 14-gate DAG/attestation | **미해결** | `GATE_DAG` 골격은 생겼지만 실제 실행 순서가 재귀·freshness·CI attestation 계약과 모순된다. 아래 M3-06. |
| m2-01 drain/public pool API | **부분 해결** | `begin_drain()`/`shutdown_pool()` public API와 idle event는 명시됐다. 다만 executor slot 누수와 cancellation-safe finalize가 없어 drain 완료성은 여전히 보장되지 않는다(M3-01). |
| m2-02 body limiter request ID/response start | **해결** | 공통 `resolve_request_id()`, response-start 추적, double-start 방지 및 8개 테스트가 설계됐다([Design §6.6a](Design.md#66a-body-크기-제한--raw-asgi-receive-wrapper-m-02-대응)). |
| m2-03 Prometheus 실제 sample 예산 | **부분 해결** | lock 버전·실제 sample 계산·139 상한은 구체화됐으나 created-series 비활성화 구현이 settings 외 직접 환경 접근 0건 gate와 정면 충돌한다. 아래 M3-07. |
| m2-04 typed settings 완전 inventory | **미해결** | 완전 목록을 설계에서 확정하지 않고 Phase 1 탐색으로 미뤘으며 env consumer만 열거해 hard-coded 운영 설정을 누락한다. 아래 M3-07. |

## 3. MAJOR 발견사항

### M3-01 — single-finalizer 이전에 RUNNING이 된 ticket은 cancel에서 영구 누수한다

근거:

- QUEUED waiter가 `await ticket.event.wait()` 중인 정확한 순간
  `_wake_next_locked()`가 ticket을 deque에서 꺼내 `RUNNING`, `_running += 1`로
  전이시킨 뒤 caller cancellation이 전달될 수 있다. `CancelledError` 분기는
  `ticket in self._deque`가 false이므로 아무것도 반환하지 않고 곧바로 raise한다
  ([Design.md:981-997](Design.md)). 이 경로는 `run_in_executor()`에 도달하지 않아
  future, done callback, `_finalize()` 모두 존재하지 않는다.
- `run_in_executor()` 자체 실패 분기의 `await self._finalize(ticket)`도 cancellation에
  shield되지 않았다([Design.md:1007-1015](Design.md)). caller가 이 await를 취소하면
  "동기적으로 release"한다는 주석과 달리 `_finalize()`가 lock을 얻기 전에 중단될 수
  있다. repeated cancellation을 future callback 경로만 시험하는 현재 테스트 표는 이
  직접 finalize 경로를 포함하지 않는다.
- 완료 callback은 `asyncio.Future.add_done_callback()`에 의해 이미 loop thread에서
  실행되는데 다시 `call_soon_threadsafe(_resume)`로 한 turn 뒤로 미룬다
  ([Design.md:1061-1073](Design.md)). callback enqueue 성공 후 `_resume` 실행 전에
  loop가 종료되면 예외도 기록되지 않고 finalize도 실행되지 않는다. 적어도 shutdown
  중 drain 판단은 callback task가 실제 완료될 때까지 보장되지 않는다.
- `begin_drain()`은 `_lock` 보호 계약을 우회하면서 "await가 없으므로 안전"하다고
  설명한다. 단일 loop thread에서는 중간 preemption이 없지만, 이 예외 규칙은
  `_deque/_running/ticket.state`가 모두 같은 lock 아래라는 설계 불변식과 증명 스케치의
  전제를 깨뜨린다([Design.md:1095-1114](Design.md)).

영향: M4-REQ-005.5, M4-REQ-006.2~3과 M4-NFR-002를 위반한다. 정상적인 client
disconnect 한 번으로 concurrency slot을 영구 점유하고 이후 shutdown drain도 grace
만료 전 정상 완료할 수 없다.

수정안:

1. queue wait의 cancel/timeout 처리를 lock 아래 하나의 `cancel_or_observe_grant()`
   전이로 합친다. 이미 grant됐다면 future 제출 전 ticket을 DONE으로 finalize하거나,
   제출을 계속해 callback을 등록한 뒤 ABANDONED로 표시하는 한 정책을 명시한다.
2. slot grant와 future submit/callback registration 사이에 cancellation point가 없다는
   사실에 의존하지 말고, `try/finally` ownership token으로 "future callback 또는
   caller cleanup 중 정확히 하나"가 반드시 책임을 인수하게 한다.
3. callback 없는 `run_in_executor` 실패 finalize는 별도 task 생성 +
   `asyncio.shield()`로 끝까지 실행하고, caller cancellation state와 release 완료를
   barrier 테스트로 분리한다.
4. `queue_grant_then_cancel_before_submit`, `cancel_during_submit_failure_finalize`,
   `callback_queued_then_loop_shutdown` 테스트를 추가한다.

### M3-02 — DeadlineBudget은 router/answer/DDGS의 실제 overall worker deadline이 아니다

근거:

- executor의 `asyncio.timeout_at()`은 caller 응답만 끝낸다. 문서도 thread를 강제
  종료하지 못한다고 인정하며, timeout된 worker는 실제 network client가 반환할 때까지
  slot을 계속 점유한다([Design §6.4](Design.md#64-bounded-concurrency--queryexecutor-webconcurrencypy)).
  따라서 M2-02가 요구한 "absolute deadline 전에 worker를 반환"하는 보장은 각
  upstream client 자체가 제공해야 하는데 설계의 3단계는 이를 제공하지 않는다.
- `httpx.Timeout(connect, read, write, pool)` 각 값은 전체 호출 합산 상한이 아니다.
  `connect+read <= remaining`만 증명해도 pool wait, write, redirect, response parsing이
  순차로 각각 예산을 소비할 수 있다. "스트리밍 경로가 있다면" budget을 검사한다는
  조건문은 LangChain Ollama 내부 response 소비에 주입할 실행 seam이 아니다
  ([Design.md:1441-1452](Design.md)).
- 현재 router cache는 `ChatOllama`가 아니라 `llm.bind_tools(...)`가 반환한
  `RunnableBinding`이다(`agent.py::_get_router_llm`). 설계는 이 객체에
  `OllamaLLM.model_copy(update={"client_kwargs": ...})`를 적용하는 것처럼 서술하며,
  binding의 bound tools를 보존하면서 underlying model을 request-scoped로 교체하는
  API를 정의하지 않는다([Design.md:1454-1471](Design.md)).
- answer path는 `RAGEngine.generate_answer()`가 `self.llm`을 chain에 직접 삽입한다
  (`rag_engine.py::generate_answer`). `RAGEngine.query(question, budget)` 시그니처만
  추가해서는 request-scoped LLM이 chain에 들어가지 않는다. 어느 메서드가 LLM 복사본을
  생성하고 `generate_answer`로 넘기는지 설계가 없다.
- DDGS는 httpx client가 아니며 현재 `DDGS(timeout=WEB_SEARCH_TIMEOUT)` 하나만
  제공한다(`web_search.py::search_web`). `compute_upstream_timeout()` 객체를 DDGS에
  어떻게 적용하고, 여러 backend/redirect/result iteration 전체를 absolute deadline에
  중단시키는지 구체 seam이 없다.
- `loop.time()` 값과 `time.monotonic()` 값이 같다는 설명은 CPython 현재 구현에
  의존한다. 둘 다 monotonic이라는 계약만으로 epoch가 같아지는 것은 아니다. 공식
  profile을 CPython 기본 loop로 제한하려면 startup assertion 또는 하나의 injected
  clock으로 실제 동일성을 검증해야 한다.

영향: M4-REQ-006.3과 M4-REQ-007.4가 충족되지 않는다. caller는 90초에 응답해도
두 worker가 장시간 남아 모든 후속 요청을 overload로 만들 수 있다.

수정안: 현재 lock된 `langchain-ollama`/`ollama`/`ddgs` 버전의 실제 생성자와 transport
API를 작은 executable spike로 먼저 확정한다. router binding은 immutable model config와
tool schema를 분리해 request마다 timeout이 반영된 model을 생성한 후 bind하고, answer는
request-scoped LLM을 `generate_answer(..., llm=...)`에 명시 전달한다. client가 overall
deadline을 지원하지 않으면 watchdog thread가 아니라 deadline-aware transport/response
iteration 또는 종료 가능한 subprocess 격리를 사용해야 한다. connect/read/trickle
테스트는 caller 응답뿐 아니라 worker 종료, `_running==0`, 다음 요청 admit까지 단언한다.

### M3-03 — ObservationSink 의사코드는 실제 RetrievalTrace와 호환되지 않는다

근거:

- 설계의 `_measure_substage()`는 `trace.record(name, duration_ms, outcome)`을 호출하지만
  현재 `RetrievalTrace`에는 `record()`가 없고 `stages: list[RetrievalStageTrace]`만 있다.
  `RetrievalStageTrace`도 `name`, `latency_ms`, `candidate_count` 세 필드뿐이며
  `outcome` 필드는 없다(`rag_engine.py:RetrievalStageTrace/RetrievalTrace`). "평가용
  필드/이름을 변경하지 않는다"는 설명과 의사코드를 동시에 구현할 수 없다
  ([Design.md:497-511](Design.md)).
- 기존 retrieval helper는 결과에 `len(result)`를 적용해 `candidate_count`를 기록하고,
  query embedding은 별도 `candidate_count=0` 규칙을 쓴다. 새 helper는 임의 반환형에
  공통 적용하면서 candidate count를 전혀 계산하지 않아 M3 trace 값 보존 계약을 깬다.
- `ProductObservationSink` 내부는 안전하지만 injected sink 자체가 예외를 던지는 테스트를
  요구한다. retrieval substage만 `safe_observe(sink.retrieval_substage, ...)`로 감싸고,
  서술된 `sink.stage(...)` 및 `sink.fallback(...)` 직접 호출에는 같은 wrapper가 없다
  ([Design.md:513-529](Design.md)). 테스트 3의 "sink 메서드 자체가 예외여도
  route_query/RAGEngine 결과 정상"은 이 호출 계약으로는 통과하지 않는다.
- `stage_started`/`retrieval_substage_started`가 allowlist에 있지만 sink protocol에는
  완료형 메서드만 있고 구현도 completed event만 낸다. Requirement는 완료/오류 연결을
  요구하므로 started가 필수는 아니지만, schema에 존재하는 이벤트가 언제 누가 내는지
  불명확해 정확한 started/completed 쌍을 기대하는 구현으로 갈라질 수 있다.

영향: M4-REQ-003.3/.5, M4-REQ-004.2와 M3 평가 호환성을 동시에 만족시킬 수 없다.
제시된 코드를 그대로 구현하면 첫 retrieval에서 `AttributeError`가 발생한다.

수정안: 기존 `RetrievalStageTrace`를 생성하는 단일 helper를 확장하되
`candidate_count`와 trace schema를 그대로 보존하고, product outcome은 sink에만 투영한다.
모든 외부 sink 호출은 `safe_sink_call()` 한 함수로 감싸 stage/substage/fallback 세 경로를
동일하게 격리한다. started event를 실제로 낼지 삭제할지 확정하고 schema/test를 맞춘다.

### M3-04 — legacy import의 expected hash와 O_NOFOLLOW가 전체 trust boundary를 닫지 못한다

근거:

- `O_NOFOLLOW`는 마지막 path component만 보호한다. 사전 `resolve()`/`realpath()` 검사
  이후 `os.open(path, O_NOFOLLOW)` 전에 writable parent directory가 rename 또는 symlink로
  교체되면 open은 새 parent 아래의 `index.pkl`을 정상적으로 연다
  ([Design.md:1866-1883](Design.md)). 문서가 주장하는 "TOCTOU 창을 없앤다"는 증명은
  parent component에 대해 성립하지 않는다.
- `os.path.realpath()` 결과와 "원본 Path"가 다르면 symlink라고 판정한다는 규칙은
  상대 경로를 절대 realpath와 직접 비교하면 정상 `runtime/vectorstore`도 항상 다르다.
  비교 대상의 절대/정규화 형식이 정의되지 않았다.
- 두 expected hash를 같은 명령의 임의 CLI 문자열로 받는 것만으로는 Requirement의
  "운영자가 소유한 manifest가 승인한 pickle" provenance를 입증하지 않는다. 공격자나
  잘못된 자동화가 파일과 그 파일의 hash를 함께 공급하면 둘은 일치한다. 설계가 제시한
  `--expected-fingerprint-file` 대안도 서명/owner/mode 검증 없이 단순 경로라면 같은
  문제가 남는다.
- 정상 시나리오가 import 단계에서 `FAISS.load_local` 0회를 기대한 뒤 activate에서만
  deserialize한다. 그러나 import와 activate 사이 staging root의 owner/mode 및 파일
  descriptor 기반 재검증 정책이 없어 승인된 복사본을 누가 바꿀 수 있는지 신뢰 경계가
  완결되지 않는다.

영향: 공격자 제어 parent 또는 staging이 가능한 배포에서 승인되지 않은 pickle이
`allow_dangerous_deserialization=True`에 도달할 수 있어 M4-REQ-008.2와 M4-NFR-004를
위반한다.

수정안: operator-owned approved root를 먼저 fd로 열고 각 component를 `openat` 계열
(`dir_fd`, `O_DIRECTORY|O_NOFOLLOW`)로 내려가며, 최종 두 파일도 그 directory fd 기준으로
연다. `fstat()`로 regular file, owner/mode, inode/device를 검증한다. 승인값은 commit된
M3 baseline의 정확한 schema/key 또는 owner/mode가 검증된 승인 manifest 한 종류로
고정하고 arbitrary hash pair와 모호한 대안을 제거한다. staging/index root owner/mode도
activate 전 검증한다.

### M3-05 — target 없는 Docker build가 production이 아니라 test image를 만든다

근거:

- Dockerfile 의사코드는 `runtime` 뒤에 `test-builder`, 마지막으로 `FROM runtime AS
  test`를 선언한다([Design.md:1984-2016](Design.md)). Docker는 `--target`이 없으면
  **마지막 stage**를 결과로 만들므로 `docker build ... -t qna-rag:ci .`와
  `qna-rag:verify`는 설계 주장과 달리 test-only mock code 및
  `simple-qna-rag-web-testonly` ENTRYPOINT를 포함한다. 이는 CI production scan과
  실제 배포 tag 모두를 무효화하는 신뢰 경계 위반이다.
- builder는 `pyproject.toml`의 dynamic dependencies가 읽는 `requirements.txt`를
  복사하지 않고 lock 파일만 다른 basename으로 복사한 뒤 `pip install --no-deps .`를
  실행한다([Design.md:1971-1982](Design.md), 현재 `pyproject.toml`). clean PEP 517
  metadata 생성에서 입력 파일이 없거나 빈 dependency metadata가 만들어질 수 있다.
- YAML에 보인 container job은 layer scan에서 끝나며 `write_evidence`, artifact upload가
  없다([Design.md:2087-2128](Design.md)). §10.1a가 "job 마지막 단계"라고 설명하는
  attestation command 및 `actions/upload-artifact`는 실제 CI 설계 블록과 연결되지 않는다.
- `known_secret_canary.bin`은 scan 명령의 비교 입력일 뿐 Dockerfile 어느 stage에도
  의도적으로 넣었다가 `.dockerignore`가 제외하는 fixture가 아니다. canary가 이미지에
  없는 성공은 scanner가 실제 layer 내용 검사를 수행했다는 positive control이 되지 않는다.
- forbidden prefix는 `runtime/documents/` 등인데 이미지 tar 경로가 `/app/runtime/...`
  형태라면 정규화 후 `app/runtime/...`가 된다. scanner의 leading slash/WORKDIR prefix
  canonicalization 계약이 없어 경로 검사가 실제 유출 위치를 놓칠 수 있다.

영향: M4-REQ-009와 M4-NFR-001/004를 위반한다. production tag가 mock server로
실행되고 production layer scan도 실제로 test image를 검사하게 된다.

수정안: 최종 stage를 명시적 `FROM runtime AS production`으로 다시 두거나 모든
production/compose/CI 명령에 `--target runtime`을 강제하고 이를 테스트한다. packaging
metadata 입력을 모두 복사하거나 dependency 선언을 lock과 분리된 정상 PEP 517 구조로
바꾼다. CI YAML에 `trap` 기반 smoke cleanup, evidence 생성, artifact upload를 실제
단계로 포함한다. scanner unit fixture에는 forbidden path/content가 있는 synthetic layer가
반드시 실패하고 clean layer가 성공하는 positive/negative control을 둔다.

### M3-06 — 14-gate runner는 fresh 실행·CI attestation·자기 테스트를 동시에 수행할 수 없다

근거:

- `static_regression` runner가 전체 `pytest -q`를 실행하고, 그 전체 suite에는
  `test_run_m4_gates_self_test.py`가 포함된다. self-test는 다시 `run_m4_gates`를
  실행하고 그 DAG가 다시 static regression 전체 pytest를 실행하므로 무한 재귀/프로세스
  폭증이 발생한다([Design.md:2263-2287](Design.md), [Design.md:2422-2430](Design.md)).
- runner는 "처음부터 비어 있는 fresh directory"에서 14개를 만든다고 주장하지만
  container runner는 CI에서 미리 다운로드된 evidence의 **존재만** 검사한다
  ([Design.md:2400-2403](Design.md)). directory가 정말 비어 있으면 container는 없고,
  container를 먼저 다운로드하면 더 이상 fresh empty directory가 아니다.
- `--skip-container` mock-only self-test는 container만 제외해 13개라고 쓰지만 DAG의
  `live_smoke`는 실제 live Ollama/model을 요구한다. mock-only CI가 live evidence를
  만드는 별도 runner/profile 계약이 없어 "13개 모두 생성" 테스트를 재현할 수 없다.
- 최종 명령은 runner가 이미 `live_smoke`를 수행한 뒤 다시
  `evaluation.m4_load live`를 evidence wrapper 밖에서 실행한다
  ([Design.md:2437-2450](Design.md)). container artifact download 명령과
  `m4_gate --ci-run-id`도 없어 container attestation 검사는 항상 UNKNOWN이 된다.
- artifact containment helper는 `(root/path).resolve()`한 뒤
  `resolved.is_symlink()`를 검사한다([Design.md:2209-2226](Design.md)). `resolve()`는
  symlink를 이미 따라갔으므로 이 검사는 symlink 자체를 발견하지 못한다. 검사와
  `read_bytes()` 사이 regular-file 교체 TOCTOU도 남는다.
- "다른 run artifact로 교체" self-test는 result artifact 자체에 `run_id` binding이
  정의되지 않았다. evidence의 `run_id`는 그대로 두고 artifact만 교체하면 hash mismatch일
  뿐 stale-evidence 판정이 아니며, hash도 함께 갱신하면 aggregator는 결과 파일이 다른
  run에서 왔음을 알 수 없다.
- container의 `ci_run_id`를 "현재 m4_gate 실행 CI run"과 같게 요구하지만 Phase 7 최종
  판정이 로컬/후속 workflow에서 실행될 수 있는 구조와 양립하지 않는다. artifact를 만든
  source run ID를 명시적으로 신뢰·검증해야지 consumer run ID와 동일할 필요는 없다.

영향: M4-REQ-010.2/.4의 사람 개입 없는 단일 판정은 실행 불가능하다. 정상 구현을
시도해도 recursion, NOT_RUN 또는 attestation UNKNOWN 중 하나로 끝난다.

수정안:

1. static gate의 pytest selection에서 M4 orchestrator self-test를 명시적으로 제외하고,
   self-test는 작은 fake runner registry를 주입해 subprocess 재귀 없이 DAG/evidence
   semantics만 검증한다.
2. local 13-gate 결과 root와 downloaded container attestation inbox를 분리한 뒤,
   aggregator용 immutable candidate root로 원자 assemble한다. assemble은 destination이
   미존재/empty임을 강제하고 source artifact를 copy+rehash한다.
3. live gate와 mock load gate를 명확히 분리하며, 최종 한 command가 live 실행 또는
   검증된 live artifact import 중 하나를 선택하도록 한다. 필수 gate에는 skip을 허용하지
   않는다.
4. CI workflow에 container evidence 생성/upload, source commit/run/attempt/image digest,
   artifact download/assemble, final aggregate 단계를 모두 실제 YAML로 명시한다.
5. artifact는 parent부터 `lstat/openat(O_NOFOLLOW)`하고 열린 fd에서 hash+parse한다.
   canonical result JSON 자체에도 candidate/run/gate를 넣어 evidence와 양방향 일치시킨다.

### M3-07 — typed settings와 Prometheus 환경 설정이 서로의 gate를 실패시킨다

근거:

- §4.3b는 완전 inventory를 상세 설계에서 확정하지 않고 "Phase 1 착수 시 grep"으로
  미룬다고 명시한다([Design.md:341-355](Design.md)). 실제 `config.py`에는 retrieval K,
  chunk size/overlap, hybrid/MMR/reranker 값, web max/timeout/region, routing hint 등
  다수 운영값이 있는데 env helper 소비만 grep하면 hard-coded 값 대부분을 놓친다.
- 예시 dataclass의 `routing_signal_override: str | None`은 현재 bool 설정과 타입부터
  다르고, `use_mmr`, `mmr_fetch_k`, `use_hybrid_search`, `reranker_model`, `chunk_size`,
  `web_search_timeout` 등 실제 consumer가 요구하는 필드가 없다
  ([Design.md:223-264](Design.md), 현재 `config.py`). 구현자가 inventory를 새로
  설계해야 하므로 상세 설계가 실행 가능한 수준으로 닫히지 않았다.
- Requirement는 "모든 운영 설정"을 단일 객체에서 읽도록 요구하지만 정적 테스트는
  환경변수 접근만 금지한다. hard-coded module constant가 Settings 밖에 남아도 테스트가
  통과하므로 요구사항 전체를 증명하지 못한다.
- metrics 모듈 최상단에서
  `os.environ.setdefault("PROMETHEUS_DISABLE_CREATED_SERIES", "True")`를 쓰도록
  요구하면서, 바로 앞 절은 `settings.py` 외 모든 제품 모듈의 `os.environ` 접근 0건을
  AST로 강제한다([Design §4.3b](Design.md#43b-typed-settings-완전-인벤토리와-직접-osenviron-조회-0건-m2-04-대응),
  [Design §5.4](Design.md#54-metric-registry)). 제시된 구현은 settings gate에서
  반드시 실패한다.

영향: M4-REQ-001.1/.3, M4-REQ-004.4 및 dependency/settings/metrics gate를 동시에
통과할 수 없다.

수정안: 상세 설계 안에 실제 현재 consumer 기준 전체 필드·type·default·alias·validation
표를 확정하고, 코드 생성 또는 machine-readable schema로 Settings와 inventory를 같은
원본에서 만든다. created-series 설정은 process 시작 전에 deployment env로 고정하거나
prometheus client의 supported API를 사용하고, settings 밖 env 접근 금지와 충돌하지 않게
한다. tests는 public config facade export와 Settings 필드의 1:1 consumer mapping도
검사해야 한다.

## 4. MINOR 발견사항

### m3-01 — Docker smoke 실패 시 container cleanup이 보장되지 않는다

§9.4 YAML은 `docker exec` 중 하나가 실패하면 `docker stop smoke`에 도달하지 않는다
([Design.md:2108-2116](Design.md)). GitHub hosted runner 종료로 결국 정리되더라도 이후
scan/evidence 단계가 같은 이름의 container 때문에 오염될 수 있다. `trap 'docker rm -f
smoke || true' EXIT` 또는 job-level always cleanup을 사용하고, readiness retry는 고정
`sleep 5` 대신 bounded poll로 구현하라.

### m3-02 — 설계의 소스 line 근거와 섹션 표현이 반복 개정 후 불안정하다

Design은 `rag_engine.py:224-245`, `agent.py:103-120`처럼 현재 파일 line을 근거로
사용하지만 다음 구현에서 즉시 drift한다. Traceability는 절 번호만 있고 핵심 seam의 symbol
단위 연결이 부족하다. 구현 단계부터 `module::symbol`을 주 근거로 쓰고 line은 보조로만
표시해야 review/변경 추적이 안정적이다.

## 5. 긍정적으로 확인한 사항

- executor release 책임을 request return 경로가 아니라 worker future completion에 두려는
  방향, explicit ticket enum, FIFO deque, absolute queue deadline은 올바른 기반이다.
- body limiter는 공통 request-ID helper와 response-start 추적까지 구체화돼 m2-02를
  설계 수준에서 닫았다.
- legacy 자동 서비스 fallback 제거, expected hash 선검사, import/activate 분리는 이전보다
  명확하고 fail-closed 방향과 일치한다.
- Docker layer outer archive를 먼저 해제하고 각 layer를 별도로 검사하는 방향은 삭제된
  layer의 secret까지 찾는 올바른 모델이다.
- evidence에 candidate/run/fingerprint/result hash를 넣고 14개 모두 PASS일 때만
  `overall_pass=true`로 만드는 판정 원칙은 요구사항과 일치한다.

## 6. Iteration 4 필수 수정 체크리스트

- [ ] queue grant 직후 cancel과 callback 없는 submit 실패까지 ownership이 보장되는
      executor 상태 머신 및 세 race barrier 테스트
- [ ] 현재 `RunnableBinding`/`OllamaLLM` chain/DDGS API에 실제 적용되고 worker 종료까지
      검증하는 overall deadline seam
- [ ] 현재 `RetrievalTrace`의 candidate count/schema를 보존하는 ObservationSink 통합과
      모든 sink 호출의 예외 격리
- [ ] parent component 및 승인 provenance/staging mode까지 닫는 legacy import trust boundary
- [ ] 기본 production target, clean PEP 517 metadata, 실제 CI evidence upload와 scanner
      positive/negative control
- [ ] pytest 자기 재귀 없이 fresh local/live/container evidence를 assemble하는 실행 가능한
      14-gate DAG와 source-CI attestation
- [ ] 전체 operational settings inventory 확정 및 Prometheus env 접근 모순 제거

위 항목을 반영한 Iteration 4 독립 리뷰 전에는 구현 단계로 진행할 수 없다. 다음 Gate도
CRITICAL/MAJOR 0, MINOR 최소화, 9.7/10 이상을 동일하게 적용한다.
