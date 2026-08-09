# M4 Production Readiness 상세 설계 리뷰 — Iteration 1

검토일: 2026-08-08  
검토 대상: [Requirement](Requirement.md), [Plan](Plan.md), [Design](Design.md)  
프로세스 기준: [milestone 개발 orchestration guide](../../../milestone_dev_orchestration_guide.md)  
검토 방식: 요구사항·계획·설계와 현재 `src/`, `evaluation/`, CI 및 배포 구조를 독립적으로 교차 검증

## 1. 결론

**Gate: FAIL**  
**점수: 6.8 / 10.0**  
**발견사항: CRITICAL 0 / MAJOR 7 / MINOR 3 / TRIVIAL 1**

범위 분해와 단계 구성, 성공 API 보존, payload 금지 원칙, index staging/pointer
방향은 좋다. 그러나 현재 설계를 그대로 구현하면 timeout/cancel 경쟁에서 executor
slot이 이중 반환되거나 음수가 될 수 있고, manifest 없는 pickle을 계속 자동
로드하며, 컨테이너 build context 차단이 실제로 적용되지 않는다. 또한 최종 gate가
테스트 실행 결과를 기계적으로 소비하는 계약이 없어 `overall_pass`를 신뢰할 수
없다. 따라서 guide의 진행 기준(9.7 이상, CRITICAL/MAJOR 0)에 미달하며 구현 Gate를
열 수 없다.

## 2. MAJOR 발견사항

### M-01 — QueryExecutor가 전체 timeout, FIFO 및 exactly-once slot 반환을 보장하지 못한다

근거:

- [Design §6.4](Design.md#64-bounded-concurrency--queryexecutor-webconcurrencypy)의
  의사코드는 `Condition.wait()`가 끝난 뒤에야 `wait_for(..., timeout=...)`를
  시작하지만, 같은 절은 timeout을 제출 시점부터 대기+실행 합산이라고 선언한다
  (`Design.md:529-545,586-590`).
- timeout 경로는 done callback을 등록한 뒤 `finally`에서도 `future.done()`이면
  `_release()`한다. 완료와 timeout의 경계 경쟁에서 같은 slot을 두 번 반환해
  `_running < 0`이 될 수 있다(`Design.md:546-570`).
- cancel 경로는 `_orphaned`를 증가시키지 않고 `_release_orphan()`은 무조건
  감소시켜 orphan gauge가 음수가 된다(`Design.md:550-570`).
- `asyncio.Condition`은 waiter의 FIFO를 공개 계약으로 보장하지 않는데 설계는
  이를 FIFO 근거로 사용한다(`Design.md:513-517`).

영향: M4-REQ-006.2~3, NFR-002와 bounded-load gate를 직접 위반한다. 부하 중
상한 초과, slot 누수/과다 반환, 뒤 요청의 starvation이 발생할 수 있다.

수정안:

1. 명시적인 FIFO ticket/deque와 `submitted_at`/absolute deadline을 사용한다.
2. queue 진입부터 `asyncio.timeout_at(deadline)`을 적용하고, queue timeout과
   실행 timeout의 상태 전이를 분리한다.
3. 작업마다 lock으로 보호되는 `released: bool` 또는 단일 event-loop owner 상태를
   두어 release를 한 함수에서 정확히 한 번만 수행한다.
4. timeout/cancel 후 실제 worker 종료까지 running을 유지하되 orphan 증감도 같은
   상태 전이 함수에서 처리한다. callback은 생성 시 저장한 event loop의
   `call_soon_threadsafe()`만 사용한다.
5. timeout-completion race, queue 중 cancel, 실행 중 cancel, callback 후 shutdown을
   barrier로 강제하는 결정론적 테스트를 표에 추가한다.

### M-02 — 자동 legacy fallback이 pickle 신뢰 경계와 fail-closed readiness를 위반한다

근거:

- 요구사항은 운영자 소유 index root 아래에서 **manifest가 승인한** `index.pkl`만
  서비스가 로드하도록 한다(`Requirement.md:193-195`).
- 설계는 `runtime/index/`가 없으면 manifest 검증 없이 기존
  `runtime/vectorstore/index.pkl`을 자동 직접 로드한다(`Design.md:850-865`).
- 같은 설계의 readiness는 index manifest 검증 완료를 ready 조건으로 선언한다
  (`Design.md:436-445`). 두 계약을 동시에 만족할 수 없다.

영향: 임의 pickle deserialization 경계를 다시 열고 M4-REQ-005.2,
M4-REQ-008.2/5, NFR-004를 위반한다.

수정안: M4 서비스는 `CURRENT`와 승인 manifest가 없으면 `index_invalid`로
fail-closed해야 한다. M3 복구는 별도 legacy service version rollback으로만
허용하거나, 명시적인 one-shot `import-legacy`가 승인 fingerprint를 검증해
manifest/version을 생성한 뒤에만 M4가 읽도록 한다. 자동 fallback은 제거한다.

### M-03 — version ID, import, lock 계약이 서로 모순되어 원자 lifecycle을 구현할 수 없다

근거:

- `version_id`를 manifest SHA-256 prefix로 정의하면서 manifest 자체에
  `version_id`를 포함한다(`Design.md:736-773`). 어떤 필드를 hash에서 제외하는지
  정의하지 않아 순환 참조가 생긴다.
- `import-legacy`는 새 파일을 바로 `versions/<version_id>`에 복사하지만, 이어지는
  activation 명령은 `--build-id`로 staging 경로를 기대한다
  (`Design.md:782-791,867-875`).
- 요구사항은 동시 **build/activate**를 하나의 OS lock으로 제한하지만
  activate 절차만 step 4에서 lock을 얻고 build의 lock 구간은 설계하지 않았다
  (`Requirement.md:199-201`, `Design.md:809-823`).
- activation은 대상 `versions/<version_id>`가 이미 존재할 때의 충돌/멱등성
  정책을 정의하지 않는다.

영향: 동일 입력의 ID가 재현되지 않거나 import 결과를 활성화할 수 없고, 동시
작업에서 예측 가능한 exit code/불변 디렉터리 계약이 깨질 수 있다.

수정안: `content_digest`는 `version_id`를 제외한 명시적 canonical payload로
계산하고 `version_id=<timestamp>-<content_digest[:8]>`로 확정한 뒤 전체 manifest를
직렬화한다. build/import 모두 staging에 동일한 완성 artifact를 만들고 동일한
`activate(staging_id)` 경로를 사용한다. root lock의 소유 범위, lock 획득 순서,
기존 destination 처리(동일 hash면 idempotent, 다르면 exit 3), cleanup을 상태표로
명시한다.

### M-04 — 90초 query timeout보다 600초인 upstream read timeout이 worker 용량을 장시간 고갈시킨다

근거:

- query 전체 timeout 기본은 90초다(`Requirement.md:161-166`).
- 외부 호출에는 connect/read/**overall** timeout이 모두 있어야 한다
  (`Requirement.md:180-181`).
- 설계는 Ollama read timeout 600초를 그대로 유지하고 overall timeout을 정의하지
  않는다(`Design.md:654-658`). timeout 응답 이후에도 두 worker가 최대 600초
  점유되면 모든 후속 query가 거절된다.

영향: HTTP timeout 응답은 빨라도 내부 서비스는 수 분 동안 복구되지 않으며,
REQ-007.4와 production readiness 목적을 충족하지 못한다.

수정안: upstream connect/read/write/pool/overall 예산을 query absolute deadline에서
파생하고 각 단계의 remaining budget 이하로 설정한다. 최소한 Ollama overall/read
timeout을 query timeout보다 짧게 두고, timeout 이후 orphan saturation 시 readiness
reason 또는 overload 상태가 어떻게 변하는지 정의한다. DDGS에도 retry 수 0과
overall budget 적용 지점을 명시한다.

### M-05 — observability 계약에서 필수 retrieval sub-stage/fallback과 영구 cardinality 관리가 빠졌다

근거:

- 요구사항은 `query_embed`, `bm25`, `dense`, `rrf`, `mmr`, `reranker`, fallback을
  같은 request ID로 연결하고 route/fallback 및 단계별 error metric을 요구한다
  (`Requirement.md:121-122,133-134`).
- 설계의 제품 stage는 `routing/web_search/retrieval/generation` 네 개뿐이고,
  metric 표에 route/fallback counter와 stage error metric이 없다
  (`Design.md:329-335,353-370`).
- `qna_rag_index_version_info{version=...}`가 이전 label child를 제거하는 규칙 없이
  프로세스 수명 중 rollback 1회라는 가정으로만 상한을 계산한다
  (`Design.md:371-381`). Phase 5는 반복 activate/rollback을 지원하므로 가정이
  lifecycle 계약과 충돌한다.

영향: 운영자가 retrieval 내부 병목/오류와 fallback을 구분할 수 없고 장기 실행
시 metric series 상한 150을 넘을 수 있다.

수정안: 평가용 `RetrievalTrace`를 변경하지 않되 그 6개 이름을 bounded product
event/metric allowlist로 투영한다. fallback counter와 stage error counter를 명시하고
cardinality 예산을 다시 계산한다. index info는 old label child를 `remove()`한 후
현재 하나만 노출하거나 label 없는 hash/info 방식으로 고정한다. 100회
activate/rollback 후 series 상한 테스트를 추가한다.

### M-06 — container ignore와 CI 명령이 실제 Docker 동작과 맞지 않아 보안 gate가 통과할 수 없다

근거:

- build context는 저장소 루트(`docker build ... .`, Compose `context: ..`)인데
  ignore 파일은 `deploy/.dockerignore`로 설계했다(`Design.md:905-934,955-964`).
  Docker가 이 빌드에서 읽는 파일은 context root `.dockerignore` 또는
  `deploy/Dockerfile.dockerignore`이므로 제시한 ignore가 적용되지 않는다.
- image의 ENTRYPOINT가 `simple-qna-rag-web`인데 CI는 override 없이
  `docker run ... qna-rag:ci id -u`를 실행한다. 이는 `id -u`를 웹 CLI 인자로
  전달하므로 UID check가 실패한다(`Design.md:899-902,964-965`).
- CI는 requirement의 import/config check와 mock readiness를 실행하지 않고 live만
  확인한다(`Requirement.md:223-225`, `Design.md:966-984`).
- `SIMPLE_QNA_RAG_MOCK_ENGINE`은 Settings schema에 없으며 제품 환경변수로 엔진을
  우회하는 숨은 production backdoor가 된다(`Design.md:970,980-984`).

영향: runtime, `.env`, model/report가 daemon에 전송될 수 있고 필수 container
gate가 거짓 음성 또는 실행 실패가 된다.

수정안: ignore를 root `.dockerignore` 또는
`deploy/Dockerfile.dockerignore`로 옮긴다. UID 검사는
`docker run --entrypoint id ... -u`, import/config는 별도 `--entrypoint python`
또는 검증 entrypoint로 수행한다. test-only mock은 제품 settings/env에서 제거하고
테스트 전용 app factory/dependency injection 또는 별도 CI image stage를 사용한다.
ready 200/503 상태표와 image history/layer tar의 known-secret 검사를 실제 명령으로
완성한다.

### M-07 — `m4_gate`가 테스트 실행 증거를 소비·검증하는 기계 계약이 없다

근거:

- 설계는 gate 데이터 소스를 단순히 “pytest 결과”, “CI 로그 exit code”로
  매핑하지만 파일 형식, 경로, schema, SHA-256, candidate/profile 일치 검증을
  정의하지 않는다(`Design.md:1015-1039`).
- 최종 명령은 일반 `pytest -q`, Docker build, live 실행 후 마지막에
  `m4_gate`를 호출한다. 앞 명령의 결과를 `m4_gate`가 읽을 artifact로 저장하지
  않는다(`Design.md:1041-1059`). 별도 shell의 이전 exit code도 읽을 수 없다.
- Requirement는 미측정/schema/fingerprint mismatch가 pass가 아니고 JSON/Markdown
  판정이 같아야 한다(`Requirement.md:229-239`).

영향: 필수 gate가 실제로 수행됐는지 확인하지 못한 채 `overall_pass=true`를
만들거나, 반대로 항상 UNKNOWN이 된다.

수정안: 모든 runner가 공통 `evidence.json` schema로 command, start/end UTC,
exit code, profile, candidate ID, Git/settings/lock/index fingerprint, result artifact
SHA-256을 원자 기록하도록 한다. pytest는 JUnit XML, container job은 attestation,
live/load/index는 canonical JSON을 생성한다. `m4_gate`는 expected evidence manifest의
모든 항목·freshness·fingerprint·hash를 fail-closed 검증하고 그 단일 판정 모델에서
JSON과 Markdown을 함께 렌더링해야 한다.

## 3. MINOR 발견사항

### m-01 — graceful shutdown의 drain 알고리즘과 executor 종료 순서가 없다

`STARTING/READY/DRAINING/STOPPED` 전이는 정의했지만 inflight가 0이 될 때까지
기다리는 primitive, queue waiter 거절/깨우기, grace 만료 후
`ThreadPoolExecutor.shutdown(wait=False, cancel_futures=True)` 처리와 callback이
닫힌 loop에 접근하지 않도록 하는 순서가 없다(`Design.md:394-411,503-590`).
상태표와 shutdown race 테스트를 추가한다.

### m-02 — body 크기 제한 구현 seam이 ASGI receive 계약까지 구체화되지 않았다

`request.stream()`을 래핑한다고만 되어 있으나 FastAPI/Pydantic body parsing보다
먼저 제한하려면 raw ASGI `receive` wrapper가 필요하다(`Design.md:647-649`).
Content-Length 중복/비정수/음수, 실제 body가 header보다 큰 경우, disconnect의
판정을 포함한 순수 ASGI middleware와 테스트를 명시한다.

### m-03 — settings canonical hash와 redacted 출력 hash의 의미가 불명확하다

`canonical_json()`과 `redacted_dict()`는 제시했지만 SHA-256이 secret 포함 원본인지
redacted canonical 값인지 명시되지 않는다(`Design.md:221-226,281-287`). Secret
변경 감지는 필요하지만 secret 자체를 report로 유추할 위험도 있다. settings
fingerprint용 keyed/field-level digest 또는 redacted operational hash 정책과
stdout/stderr 계약을 명확히 한다.

## 4. TRIVIAL 발견사항

### t-01 — 존재하지 않는 의사코드 문구를 설명하는 잔여 문장이 있다

`else_done := None` 줄이 실제 코드 블록에는 없는데 정리 실수 방지 주석이 남아
있다(`Design.md:573-574`). 설계 가독성을 위해 삭제한다.

## 5. 긍정적으로 확인한 사항

- M4-REQ-001~010과 Phase 0~7의 큰 구조 및 산출물 경계가 일관된다.
- 기존 성공 `/rag` schema, CLI 이름, M3 rollback flag와 승인 baseline을 보존하는
  호환성 의도가 명시되어 있다.
- request ID 형식, payload/secret/absolute path 금지, metric label allowlist처럼
  구현자가 바로 테스트로 옮길 수 있는 bounded 계약이 다수 제시되어 있다.
- health query 경로와 query executor 분리, staging 검증 후 atomic pointer 교체,
  serving index read-only mount 방향은 적절하다.
- live 12 case ID와 concurrency tuning 하향 조건이 사전에 고정되어 선택 편향을
  줄인다.

## 6. 다음 Iteration 필수 수정 체크리스트

- [ ] exactly-once release와 absolute deadline을 갖는 명시적 FIFO executor 상태표/의사코드
- [ ] manifest 없는 legacy pickle 자동 로드 제거와 M4 fail-closed migration 계약
- [ ] 비순환 version ID, build/import 공통 staging, build+activate lock/충돌 계약
- [ ] query deadline 이하의 upstream overall timeout 예산
- [ ] retrieval 6 sub-stage/fallback observability와 반복 activation cardinality 상한
- [ ] 실제 적용되는 Docker ignore, 유효한 container 명령, mock injection 분리
- [ ] fingerprint/hash가 결합된 공통 gate evidence schema와 fail-closed aggregator
- [ ] shutdown/body-limit/settings-hash 세부 계약 보완

위 항목을 반영한 뒤 Iteration 2 독립 리뷰가 필요하다. CRITICAL/MAJOR가 모두
0이고 MINOR가 최소화되며 9.7/10 이상일 때만 구현 단계로 진행할 수 있다.
