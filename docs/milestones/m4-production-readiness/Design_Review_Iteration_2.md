# M4 Production Readiness 상세 설계 리뷰 — Iteration 2

검토일: 2026-08-08  
검토 대상: [Requirement](Requirement.md), [Plan](Plan.md), 최신
[Design](Design.md), [Traceability](Traceability.md),
[Iteration 1 리뷰](Design_Review_Iteration_1.md), 현재 제품 코드·CI·배포 구조  
프로세스 기준: [milestone 개발 orchestration guide](../../../milestone_dev_orchestration_guide.md)  
검토 방식: Iteration 1 발견사항의 실제 폐쇄 여부를 우선 재현하고, 수정 과정에서
생긴 새 모순과 구현 가능성을 전체 요구사항 범위에서 독립적으로 교차 검증

## 1. 결론

**Gate: FAIL**  
**점수: 5.9 / 10.0**  
**발견사항: CRITICAL 0 / MAJOR 6 / MINOR 4 / TRIVIAL 0**

Iteration 1의 version ID 순환, 공통 staging/lock, legacy 자동 서비스 폴백,
settings hash 의미와 잔여 의사코드 문구는 설계 수준에서 개선됐다. 그러나 구현
Gate를 열 수 있는 상태는 아니다. executor는 동기 함수 예외나 정상 완료 직후
cancel에서 slot을 영구 누수하고, upstream timeout은 `overall` deadline을 실제로
강제하지 않는다. 관측 이벤트 allowlist와 제품 retrieval trace도 서로 맞지 않는다.
Dockerfile은 `.dockerignore` 때문에 build 자체가 실패하며 test image mock 모듈도
Python import 경로에 설치되지 않는다. 최종 명령은 14개 evidence 대부분을 만들지
않으므로 aggregator가 구조적으로 `overall_pass=true`를 낼 수 없다. 따라서 guide의
기준(9.7 이상, CRITICAL/MAJOR 0, MINOR 최소)에 미달한다.

## 2. Iteration 1 발견사항 폐쇄 감사

| 이전 발견 | 판정 | 근거와 잔여 사항 |
|---|---|---|
| M-01 executor FIFO/deadline/exactly-once | **미해결** | 명시적 deque와 absolute deadline은 추가됐으나 worker 예외 및 정상 완료 후 cancel에서 release가 없다. 아래 M2-01. |
| M-02 legacy 자동 폴백/pickle | **부분 해결** | 서비스 자동 폴백은 제거됐으나 `import-legacy`가 기대 fingerprint 없이 임의 `--from` pickle을 load-smoke한다. 아래 M2-04. |
| M-03 version ID/staging/lock | **해결** | `version_id` 제외 canonical content payload, 공통 staging, 단일 lifecycle lock, 충돌/멱등 정책이 정의됐다([Design §8.2~8.5](Design.md#82-manifest-schema와-순환-없는-version-id-indexmanifestpyindexmanifest-m-03-대응)). |
| M-04 upstream budget | **미해결** | per-phase `httpx.Timeout`만 만들며 overall deadline이 없고 기존 singleton에 매 호출 주입하는 seam도 없다. 아래 M2-02. |
| M-05 retrieval/fallback observability | **미해결** | metric 표는 확장됐지만 event allowlist·제품 trace 경로가 상충한다. 아래 M2-03. |
| M-06 Docker ignore/명령/mock DI | **미해결** | ignore 위치와 UID 명령은 고쳤으나 build 입력과 test stage 설치 경로가 깨졌고 layer scan이 유효하지 않다. 아래 M2-05. |
| M-07 evidence aggregator | **미해결** | schema는 생겼으나 runner/최종 명령/CI 수집이 연결되지 않았다. 아래 M2-06. |
| m-01 graceful shutdown | **부분 해결** | 순서 표는 생겼으나 idle drain, pool 접근자, 프로세스 종료 한계가 남는다. 아래 m2-01. |
| m-02 raw ASGI body limit | **부분 해결** | raw receive seam은 맞지만 oversize error의 request ID와 response-start 안전성이 없다. 아래 m2-02. |
| m-03 settings hash | **해결** | redacted canonical hash이며 secret 회전 비감지 한계와 stdout/stderr 경계를 명시했다([Design §4.3a](Design.md#43a-설정-hash의-의미와-stdoutstderr-계약-m-03-대응)). |
| t-01 잔여 문구 | **해결** | 해당 문구가 제거됐다. |

## 3. MAJOR 발견사항

### M2-01 — executor 종료는 worker 결과와 독립적으로 exactly-once가 아니다

근거:

- `await asyncio.shield(future)`가 제품 함수 예외를 던지면 `TimeoutError`와
  `CancelledError` 어느 분기에도 들어가지 않고, `else`의
  `_transition_done()`도 실행되지 않는다. `_running`은 영구히 감소하지 않고
  다음 FIFO ticket도 깨우지 않는다
  ([Design.md:770-784](Design.md)). 실제 `route_query()`와 하위 model/web
  호출은 예외를 낼 수 있으므로 정상적인 production failure 경로다.
- future가 정상 완료한 후 task가 `await _transition_done()`의 lock 획득 중
  취소되면 이 취소는 앞선 `try`의 `except CancelledError`가 잡지 않는다
  (`else` suite에서 발생). callback도 정상 실행에는 등록하지 않았으므로 같은
  slot 누수가 생긴다([Design.md:773-824](Design.md)).
- 문서는 `RUNNING -> ORPHANED`를 상태로 사용하지만 enum에는 `ORPHANED`가 없고
  ticket state도 실제로 전이시키지 않는다([Design.md:692-712](Design.md)).
  따라서 증명 스케치가 말하는 상호 배타 상태를 코드로 검사할 수 없다.
- fake clock은 ticket deadline 계산에만 쓰이고 `asyncio.timeout(remaining)`은
  실제 event-loop clock을 사용한다. "real sleep 없는 deadline 제어" 테스트는
  제시된 seam만으로 결정론적으로 구현할 수 없다
  ([Design.md:731-753](Design.md), [Design.md:896-903](Design.md)).

영향: M4-REQ-006.2~3과 M4-NFR-002를 직접 위반한다. 한 번의 예상 가능한 worker
예외만으로 bounded executor가 영구 포화되고, FIFO/slot leak gate가 실패한다.

수정안:

1. future 생성 즉시 모든 future에 완료 callback을 한 번 등록하고, **slot 반환은
   그 callback이 event loop로 전달하는 단일 finalize 함수만** 수행하게 한다.
   request task는 결과/timeout/cancel 응답과 `abandoned` 표지만 담당한다.
2. `QUEUED/RUNNING/ABANDONED/DONE`을 실제 enum으로 만들고 모든 counter/state
   전이를 같은 lock 아래 한 함수에서 수행한다. worker 성공·예외·future cancel
   모두 DONE으로 수렴시킨다.
3. callback 등록 전 `run_in_executor` 자체가 실패하는 경로도 동기적으로 release한다.
4. worker exception, 정상 완료 직후 cancel, finalize lock 대기 중 repeated cancel,
   `run_in_executor` 실패를 barrier 테스트에 추가한다.
5. 시간 테스트는 loop와 같은 clock을 쓰는 `timeout_at(ticket.deadline)` 및 테스트
   event/barrier로 경계를 강제하거나, timeout primitive 자체를 주입한다.

### M2-02 — upstream 설계는 overall deadline을 강제하지 않고 현재 객체 생명주기에도 적용할 수 없다

근거:

- `compute_upstream_timeout()`은 connect/read/write/pool **각 단계의 timeout**을
  반환할 뿐, 호출 전체를 감싸는 overall timeout은 없다. httpx의 네 값은 합산
  wall-clock budget이 아니므로 connect 후 read, redirect/stream 처리 등이 query
  deadline을 넘을 수 있다([Design.md:1041-1053](Design.md)). 이는
  connect/read/**overall**을 요구한 [Requirement.md:180-181](Requirement.md)과
  맞지 않는다.
- `read = max(min_read, remaining - connect)`는 remaining이 min_read보다 작을 때
  남은 예산보다 큰 read timeout을 만들며, `remaining == 0`이면
  `connect=0, read=min_read`가 되어 deadline 이후에도 새 upstream 호출을
  허용한다([Design.md:1041-1053](Design.md)).
- 설계는 `OllamaLLM(... timeout=...)`을 "매 호출마다 주입"한다고 하지만 현재
  답변 LLM은 엔진 초기화 때 singleton으로 만들어지고
  (`rag_engine.py:224-245`), routing `ChatOllama`도 별도 singleton이다
  (`agent.py:103-120`). §6.6b는 router 호출의 timeout 적용 seam과 두 singleton을
  deadline별로 재설정/호출하는 방법을 정의하지 않는다.
- `asyncio` loop time으로 만든 deadline을 동기 코드에서 `time.monotonic()`과
  혼용하는 계약도 구현 독립적으로 보장하지 않았다([Design.md:700-737](Design.md),
  [Design.md:1029-1049](Design.md)).

영향: HTTP 응답은 90초에 timeout될 수 있어도 router/answer worker는 그 이후에도
오래 slot을 점유할 수 있으며 M4-REQ-007.4와 M2-01의 capacity 회복 목표가
성립하지 않는다.

수정안: 하나의 `DeadlineBudget`(동일 monotonic clock)을 request에서 sync call까지
전달하고, 매 upstream 호출 직전 `remaining <= 0`이면 네트워크를 시작하지 않는다.
httpx client 단계 timeout은 모두 `remaining-safety` 이하로 cap하고, 호출 전체도
client가 지원하는 overall 경계 또는 별도 watchdog/process 경계로 감싼다. router와
answer LLM에 request-scoped client/options를 전달할 구체 API를 현재
`ChatOllama`/`OllamaLLM` 버전에서 검증해 설계하고, 불가능하면 singleton 생성
정책을 명시적으로 바꾼다. fake server의 connect/read/stream stall 각각이
absolute deadline 전에 worker를 반환하는 통합 테스트를 추가한다.

### M2-03 — retrieval/fallback 관측 계약이 event allowlist 및 실제 제품 trace 경로와 모순된다

근거:

- `EVENT_NAMES`에는 `retrieval_substage`나 `fallback` event가 없는데 §5.3은
  `retrieval_substage` 로그를 남기고 fallback 단계 완료/오류도 요구한다
  ([Design.md:330-342](Design.md), [Design.md:379-408](Design.md)).
- `stage_started/stage_completed`의 stage는 상위 `STAGE` 네 값만 허용한다.
  따라서 retrieval 6개와 fallback을 같은 request ID로 연결한다는
  M4-REQ-003.3을 제시된 allowlist로 표현할 수 없다.
- 현재 제품 retrieval은 `trace=None`이면 `RetrievalTrace`와 span을 만들지 않는
  zero-cost 계약이다(`rag_engine.py:66-84,468-486`). 설계는 제품 경로에서
  "개별 span이 완료될 때마다" trace 값을 읽겠다고 하지만 제품 query가 trace를
  생성하는지, callback sink를 주입하는지, 평가 trace와 중복 측정을 어떻게
  피하는지 정의하지 않는다([Design.md:397-408](Design.md)).
- `safe_log`/`safe_observe` 적용 범위를 `web/server.py` 호출로만 제한하지만 실제
  stage/fallback hook은 `agent.py`와 `rag_engine.py`에 있다
  ([Design.md:484-489](Design.md)).

영향: 구현자는 allowlist를 깨거나 필수 retrieval/fallback 로그를 누락하는 둘 중
하나를 선택해야 한다. logging gate와 M4-REQ-003/004를 동시에 만족시킬 수 없다.

수정안: event와 field의 단일 schema를 확정해 `stage_*`가 상위/substage/fallback을
모두 bounded 값으로 수용하게 하거나 별도 allowlisted event를 추가한다. 제품용
`ObservationSink`/callback을 retrieval 각 계측 seam에 주입하고 평가용
`RetrievalTrace`는 동일 측정값을 소비하도록 하여 timer 중복을 막는다.
`agent.py`/`rag_engine.py`도 모든 sink 호출을 safe wrapper로 강제하고 성공,
예외, fallback 세 분기를 통합 테스트한다.

### M2-04 — legacy import가 "승인된 pickle만"이라는 trust boundary를 입증하지 못한다

근거:

- 서비스의 legacy 자동 폴백 제거는 명확하다
  ([Design.md:1372-1404](Design.md)). 그러나 CLI는
  `import-legacy --from <path>`로 받은 두 파일을 스스로 hash해 그 값으로 manifest를
  만든 뒤, activate 단계에서 `allow_dangerous_deserialization=True`로 load-smoke한다
  ([Design.md:1246-1267](Design.md), [Design.md:1313-1318](Design.md),
  [Design.md:1406-1420](Design.md)).
- 문장은 "운영자가 승인된 fingerprint를 알고" 검증한다고 주장하지만 CLI 계약에
  `--expected-faiss-sha256`/`--expected-pkl-sha256` 또는 승인 manifest 입력이 없고,
  어느 단계에서도 기대값과 실제값을 비교하지 않는다. 자체 계산 hash는 무결성
  표식일 뿐 승인 증거가 아니다.
- Requirement는 운영자 소유 root의 manifest가 승인한 pickle만 load하고 임의 경로
  입력으로 load하지 않도록 요구한다
  ([Requirement.md:193-204](Requirement.md)). 현재 `--from` 경계는 이 조건을
  기계적으로 강제하지 않는다.

영향: 로컬 경로를 바꿀 수 있는 공격자나 운영 실수로 임의 pickle이 activation
검증 과정에서 코드 실행될 수 있다. M4-NFR-004의 핵심 신뢰 경계가 fail-open이다.

수정안: import 시 두 **기대 SHA-256을 필수 인자** 또는 서명/승인된 M3 fingerprint
파일로 받고, copy 전후 hash가 모두 기대값과 일치할 때만 staging manifest를 만든다.
입력은 승인된 legacy root 아래의 정규화된 두 고정 파일명으로 제한하고 symlink를
거부한다. hash 불일치 시 deserialization 전에 exit 3이어야 한다. 테스트는 임의
경로, symlink escape, TOCTOU 교체, 기대 hash 불일치에서 `FAISS.load_local` 호출이
0회임을 검증한다.

### M2-05 — 제시된 Dockerfile/test image/layer scan은 clean CI에서 실행되지 않는다

근거:

- root `.dockerignore`가 `*.md`를 제외하지만 Dockerfile builder는
  `COPY pyproject.toml README.md LICENSE ./`를 실행한다
  ([Design.md:1453-1461](Design.md), [Design.md:1497-1515](Design.md)).
  root context에서 `README.md`가 제외되므로 clean build는 COPY 단계에서 실패한다.
- package는 `pip --target /opt/venv`로 설치되고 `PYTHONPATH=/opt/venv`인데 test
  stage는 mock 파일을 `/opt/venv/lib/python3.11/site-packages/...`에 복사한다
  ([Design.md:1457-1483](Design.md)). 이 하위 경로는 PYTHONPATH가 아니며
  설치된 `simple_qna_rag` package 경로와도 다르다. 또한 신규
  `simple-qna-rag-web-testonly` console script를 생성하는 단계가 없어 test
  ENTRYPOINT가 존재한다는 보장이 없다.
- §9.0은 test image가 "명시적으로 `--entrypoint`"일 때만 mock을 쓴다고
  선언하지만 Dockerfile의 test stage는 이미 test-only ENTRYPOINT를 기본값으로
  설정하고 CI도 override 없이 실행한다
  ([Design.md:1438-1449](Design.md), [Design.md:1476-1484](Design.md),
  [Design.md:1567-1574](Design.md)).
- `docker save ... | tar -xO > /tmp/img.tar`는 outer OCI/Docker archive의 서로
  다른 파일(JSON, manifest, compressed layer)을 이어 붙인 바이트를 만들 뿐
  유효한 layer tar가 아니다. 이어지는 `tar -tf`는 layer 내용을 신뢰성 있게
  검사하지 못하고 known-secret **내용**도 검색하지 않는다
  ([Design.md:1576-1580](Design.md)). Requirement/Plan은 runtime/report/cache,
  Git metadata와 known-secret fixture의 layer 포함 0건을 요구한다
  ([Plan.md:275-281](Plan.md)).

영향: 필수 container gate가 build 단계에서 실패하거나, mock smoke가 시작되지
않거나, 민감 artifact 포함을 놓치는 거짓 PASS를 만들 수 있다.

수정안: Dockerfile 전용 ignore에서 필요한 `README.md`를 `!README.md`로 다시
포함하거나 package metadata가 README를 요구하지 않는 wheel-build 흐름으로 바꾼다.
builder에서 wheel/venv를 만들고 test 전용 package/entrypoint도 정상 packaging으로
설치한다. test-only 기본 ENTRYPOINT 정책을 한 가지로 통일한다. layer 검사는
`docker image save` outer archive를 풀고 각 layer tar를 순회하여 금지 경로와
known-secret byte를 모두 검사하는 검증 스크립트로 구현하며, 실패 자체도 container
evidence에 기록한다.

### M2-06 — evidence schema는 생겼지만 최종 실행과 aggregator 사이에 실행 가능한 DAG가 없다

근거:

- `run_pytest_gate`는 pytest JUnit만 기록한다고 정의하면서
  `static_regression.json`은 pytest, npm, vendor diff, Markdown link,
  `git diff --check`를 **취합**한다고 주장한다
  ([Design.md:1672-1678](Design.md), [Design.md:1725-1729](Design.md)).
  npm/link/diff의 결과 schema와 취합 runner가 없다.
- 최종 명령은 npm, locked install, config check, index pytest, Docker build를
  evidence wrapper 밖에서 직접 실행한다. `dependency`, `settings`, `logging`,
  `metrics`, `health`, `event_loop`, `index`, `container` evidence를 만드는 명령이
  없거나 불완전하다([Design.md:1744-1769](Design.md)). 이 상태에서
  `m4_gate`는 필연적으로 다수를 `NOT_RUN` 처리한다.
- container evidence는 "CI artifact로 저장 후 Phase 7에서 수집"한다고만 하고
  artifact 이름, download/검증 명령, CI run/commit 결합, host가 다른 fingerprint
  필드 정책이 없다([Design.md:1684-1687](Design.md)).
- `write_evidence()`는 임의 `result_artifact_path`를 저장하며 aggregator는 나중에
  그 경로를 그대로 연다. candidate root 밖 absolute path, `..`, symlink를
  거부하지 않아 evidence가 다른 실행의 artifact를 참조할 수 있다
  ([Design.md:1638-1667](Design.md), [Design.md:1706-1715](Design.md)).
- 정적 gate의 JUnit XML처럼 JSON이 아닌 artifact도 있는데 aggregator는 모든
  성공 evidence의 "실제 결과 JSON"을 읽어 threshold를 적용한다고 선언한다
  ([Design.md:1713-1715](Design.md), [Design.md:1727](Design.md)).

영향: M4-REQ-010.2/4의 사람 판단 없는 필수 gate는 구현 불가능하다. schema가
존재해도 실제 명령 실행을 증명하지 못하며, 최종 `overall_pass`는 항상 false이거나
검증되지 않은 artifact를 신뢰하게 된다.

수정안: `evaluation/run_m4_gates.py` 하나가 고정된 gate DAG 전체를 subprocess로
실행하고, gate별 canonical result JSON과 evidence를 원자 생성하도록 한다. 각
result schema 및 threshold parser를 명시하고 static composite는 모든 하위 command
결과/hash를 하나의 JSON에 기록한다. CI container attestation은 commit SHA/run ID로
다운로드하고 서명된 metadata와 image digest를 검증한다. artifact path는 candidate
root 기준 상대 경로만 허용하고 resolve 후 root containment/symlink 거부를 검사한다.
최종 문서에는 14개 evidence가 fresh empty directory에서 생성된 뒤 aggregator가
PASS하는 단일 명령과, 한 evidence 삭제/변조/교체 시 FAIL하는 self-test를 넣는다.

## 4. MINOR 발견사항

### m2-01 — graceful shutdown 의사코드의 idle·pool 계약이 실행되지 않는다

- lifespan은 `query_executor.pool.shutdown(...)`을 호출하지만 executor가 노출한
  필드는 `_pool`뿐이다([Design.md:520-529](Design.md),
  [Design.md:714-728](Design.md)). 그대로 구현하면 shutdown에서
  `AttributeError`가 난다.
- shutdown 진입 시 `_running == 0`이면 `reject_queued()`도
  `drain_complete.set()`을 호출하지 않아 매번 전체 30초를 기다린다
  ([Design.md:516-540](Design.md), [Design.md:837-849](Design.md)).
- `ThreadPoolExecutor.shutdown(wait=False)`는 실행 중 thread를 종료하지 않으며
  Python interpreter는 종료 전 executor thread를 join한다. grace 이후 프로세스
  종료가 bounded라는 인상을 주지 말고 server/process supervisor 동작과
  non-returning worker 대응을 runbook에 정확히 명시해야 한다.

수정안: `await query_executor.begin_drain()` 하나가 lock 아래 draining 설정,
queued reject, idle event set을 수행하고 `shutdown_pool()` public API를 제공한다.
idle/active/grace-expired 실제 process 종료 테스트와 supervisor hard-stop 조건을
추가한다.

### m2-02 — body limit의 413은 stable error/request-ID와 ASGI response 상태를 보장하지 않는다

Body limiter가 가장 바깥이라 request context가 생성되기 전에 `_send_413`을 보내며
request ID 생성/응답 header 계약이 없다([Design.md:969-1018](Design.md)).
또 하위 app이 response start를 이미 보낸 경우 두 번째 413 start를 보내지 않는다는
것을 "일반적 처리 순서"에만 의존한다. M4-REQ-007.2는 실패 응답에도 request ID를
요구한다. limiter에서 공통 request-ID helper를 호출하고 start 여부를 추적해, 시작
전에는 canonical ErrorResponse 413, 시작 후에는 안전한 disconnect/abort 정책을
적용하라. middleware 순서와 duplicate header 6개 테스트에서 body/header 모두
검증해야 한다.

### m2-03 — metric series 예산 계산이 Prometheus 실제 sample 계약과 정확히 맞지 않는다

Histogram은 boundary 8개일 때 `_bucket`이 `+Inf`까지 9개이고 `_sum`, `_count`,
`_created`가 추가될 수 있다. Counter도 Python client 기본 설정에서 `_created`
sample을 노출할 수 있다. 설계의 "×10 근사"와 counter당 1개 계산
([Design.md:439-456](Design.md))은 실제 `generate_latest()` series 수와
다르다. 테스트에서 실제 collector sample을 세므로 설계 상한 139가 재현되지 않을
가능성이 높다. 사용하는 prometheus-client 버전과 `_created` 설정을 고정하고 실제
sample 명명 규칙으로 worst-case를 다시 계산하라. 필요하면 bucket/metric을 줄여
150 이하 여유를 확보하라.

### m2-04 — typed settings 필드 목록이 단일 설정 계약을 완결하지 않는다

§2.2는 모든 기존 flag를 이관한다고 하지만 실제 `Settings` dataclass에는
`ANSWER_TEMPLATE_MODE`, `ROUTING_SIGNAL_OVERRIDE`, `MMR_VECTOR_SOURCE`, hybrid/MMR/
reranker/chunk 설정 등 현재 `config.py` 소비 필드가 없다
([Design.md:102-108](Design.md), [Design.md:194-246](Design.md)).
Requirement는 template/routing/retrieval flags를 최소 설정으로 요구한다
([Requirement.md:79-93](Requirement.md)). 구현자가 추측하지 않도록
`config.py`의 모든 public setting consumer를 필드·type·default·env alias·검증표로
완전히 열거하고, 제품 모듈의 직접 `os.environ` 조회 0건을 정적 테스트로 닫아라.

## 5. 긍정적으로 확인한 사항

- `content_digest`에서 identity/time/self-reference를 제외하고 완성 manifest의
  외부 hash를 분리해 version ID 순환을 제거했다.
- build/import가 같은 staging 모양과 activation 함수를 사용하고, 모든 lifecycle
  변경을 하나의 non-blocking OS lock 아래 두며 destination 충돌 정책을 정의했다.
- M4 서비스가 `CURRENT`/manifest 실패 시 legacy pickle로 자동 폴백하지 않고
  readiness 503으로 유지되는 방향은 맞다.
- root build context에 맞는 `.dockerignore` 위치와 UID 검사 entrypoint override를
  바로잡았다.
- settings hash가 redacted operational hash이며 secret 회전을 검출하지 않는다는
  한계를 명시했고, t-01 잔여 문구를 제거했다.
- 반복 index activation에서 이전 Prometheus label child를 제거하는 장기
  cardinality 방향과 100회 stress test 제안은 타당하다.

## 6. 다음 Iteration 필수 수정 체크리스트

- [ ] worker 성공·예외·cancel·timeout 모두 단일 callback finalize로 수렴하는
      executor 상태 머신과 결정론적 race 테스트
- [ ] router/answer/DDGS 전부에 실제 적용되는 동일-clock absolute overall deadline
- [ ] retrieval sub-stage/fallback event allowlist와 제품 observation seam 통일
- [ ] expected fingerprint·경로 containment·TOCTOU 방어가 선행되는 legacy import
- [ ] clean build 가능한 Docker context/package/test stage와 실제 layer content scan
- [ ] 14개 gate를 fresh directory에서 모두 생성·수집·검증하는 단일 evidence DAG
- [ ] idle drain/public pool API/body-limit request ID/실제 Prometheus sample 예산 보완
- [ ] typed settings 전체 inventory 및 직접 환경변수 조회 0건 검증

위 항목을 반영한 Iteration 3 독립 리뷰 전에는 구현 단계로 진행할 수 없다.
다음 Gate도 CRITICAL/MAJOR 0, MINOR 최소화, 9.7/10 이상을 동일하게 적용한다.
