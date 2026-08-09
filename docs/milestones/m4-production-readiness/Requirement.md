# M4 Production Readiness 요구사항 정의서

상태: **동결 — 분할 복구 결정 이전의 감사 기록**  
후속: [M4 복구 결정](Recovery_Decision.md), [M4.1 요구사항](../m4.1-configuration-observability/Requirement.md)  
기준일: 2026-08-08  
상위 문서: [Roadmap](../../Roadmap.md), [알려진 문제](../../Problem.md)  
승인 기준선: [M3 baseline](../../../evaluation/baselines/m3_initial.md),
[M3 요구사항 추적표](../m3-retrieval-domain-quality/Traceability.md)

## 1. 목적과 성공 상태

M4는 현재의 단일 프로세스·로컬 우선 FastAPI 서비스를 **새 환경에서 반복
배포하고, 장애 원인을 관측하며, 제한된 동시 요청을 데이터 손실 없이 처리할 수
있는 내부 서비스**로 만든다. M3의 검색·라우팅·답변 품질을 보존하면서 운영
계약을 추가하며, 분산 시스템이나 대규모 재작성은 하지 않는다.

M4 완료 시 다음 상태를 기계적으로 증명할 수 있어야 한다.

1. 승인된 Python/Node 의존성과 실행 설정을 동일하게 복원할 수 있다.
2. 로그와 메트릭만으로 요청, 단계, 결과, 오류 유형과 latency를 연결할 수 있다.
3. liveness는 프로세스 생존을, readiness는 실제 요청 수락 가능성을 구분한다.
4. blocking 모델 호출과 CPU 작업이 ASGI event loop를 점유하지 않으며, 동시성은
   명시적으로 제한되고 초과 부하는 빠르게 거절된다.
5. 인덱스는 provenance 검증을 통과한 완전한 버전만 원자적으로 활성화되며,
   중단·충돌·불일치가 현재 활성 인덱스를 훼손하지 않는다.
6. 컨테이너 또는 문서화된 호스트 실행 절차로 비루트·최소 권한 내부 배포가
   가능하고, 보안 경계와 롤백 절차가 검증된다.

## 2. 현재 기준과 범위 결정 근거

### 2.1 보존할 승인 기준

- Python 3.11, Ollama `gpt-oss:20b`, BGE-M3 embedding/reranker, 18개 corpus와
  현재 FAISS artifact를 기준으로 한다.
- M3 승인 fingerprint는 dataset
  `61b768ac...d1017a`, corpus manifest `5c0d648d...c82374a`,
  `index.faiss` `c52fb288...d69820`, `index.pkl`
  `3f7217a2...91bb00`이다. 전체 값의 단일 원본은 M3 baseline이다.
- M3 품질 gate 14/14, Retrieval 42건, Routing 76건×3회, Answer 29건과 public
  `/rag` 응답 필드(`answer`, `sources`, `success`, `search_type`, `intent`)를
  회귀 기준으로 보존한다.

### 2.2 현재 결함

- `config.py`에 상수와 부분적인 환경변수 parser가 섞여 있고 상호 제약,
  secret-safe 출력, 전체 설정 검증 계약이 없다.
- `requirements.txt`는 넓은 범위이며 baseline dependency snapshot과 설치 lock이
  없다. CI도 매번 범위를 새로 해석한다.
- 서버와 agent/index 경로는 `print()` 중심이며 요청 상관관계, 안정된 오류 분류,
  집계 가능한 운영 메트릭이 없다.
- `async def /rag`가 동기 Ollama, DDGS, embedding, reranker를 직접 호출한다.
  대형 singleton을 고려하지 않은 worker 증가는 메모리를 중복한다.
- `/health`는 엔진 초기화 실패 여부와 무관하게 `healthy`를 반환할 수 있고
  readiness/liveness가 분리되지 않았다.
- 인덱스 CLI가 활성 경로에 직접 `save_local()`하며 manifest, staging 검증,
  원자 교체, 이전 버전 롤백 계약이 없다.
- Dockerfile/배포 자동화가 없고 API 인증·rate limit·요청 크기 경계가 없다.

### 2.3 필수·조건부·제외 범위

| 영역 | M4 결정 | 근거 |
|---|---|---|
| typed settings와 시작 시 검증 | **필수** | 현재 환경변수·상수 분산과 조용한 오설정을 제거 |
| Python lock, Node lock 검증, dependency snapshot | **필수** | 같은 Git/artifact라도 설치 결과가 달라지는 P2 해결 |
| JSON structured logging, 프로세스 로컬 metrics | **필수** | 단일 인스턴스 장애 진단에 충분한 최소 관측성 |
| liveness/readiness 분리 | **필수** | 무거운 모델과 외부 Ollama 준비 상태를 구분 |
| bounded concurrency와 blocking offload | **필수** | async endpoint의 event-loop 점유와 무제한 대기 방지 |
| versioned index, provenance, atomic activation/rollback | **필수** | 부분 저장·설정 불일치·현재 인덱스 손상 방지 |
| 입력 크기, 타임아웃, CORS/host, 오류 정보 제한 | **필수** | 공개 인증 이전에도 필요한 기본 앱 경계 |
| OCI container와 내부 배포 runbook | **필수** | 새 환경 재현과 비루트 실행의 검증 가능한 기준 |
| TLS, 인증, 사용자별 rate limiting | **조건부** | loopback/신뢰망 밖 노출 시 reverse proxy에서 필수 |
| 외부 metrics backend/trace collector | **조건부** | 다중 replica 또는 보존·알림 요구 발생 시 도입 |
| 외부 vector DB, 작업 큐, Kubernetes, autoscaling | **M5로 제외** | 현재 문서·트래픽 규모에서 비용과 복잡도가 과도함 |
| multi-process web worker 기본화 | **제외** | worker마다 모델·인덱스 메모리 중복; M4 기본은 1 worker |

## 3. 기능 요구사항

### M4-REQ-001 — 단일 typed settings 계약

1. 모든 운영 설정은 하나의 immutable typed settings 객체에서 읽고, 제품 모듈이
   임의로 `os.environ`을 재조회하지 않는다. 기존 공개 상수 import는 한 전환
   기간 호환 facade로 유지할 수 있다.
2. 환경변수 접두사는 `SIMPLE_QNA_RAG_`로 유지한다. bool, enum, URL, 정수,
   duration, 경로를 엄격히 검증하고 알 수 없는 enum, 범위 밖 숫자, 상호 모순은
   시작 전에 설정 오류(exit code 2)로 실패한다.
3. 최소 설정에는 documents/vectorstore/model 경로, Ollama URL/model,
   retrieval flags, web search, template/routing flags, log level/format,
   요청 크기, query timeout, concurrency/queue, readiness 정책이 포함된다.
4. 상대 runtime 경로는 명시한 runtime root 기준으로 해석하고, 기존
   `environment > runtime default > legacy fallback` 및 양쪽 경로 충돌 시
   fail-closed 계약을 보존한다.
5. `--check-config` 또는 동등한 명령은 외부 모델을 로드하지 않고 설정을 검증한
   뒤 canonical JSON과 SHA-256을 출력한다. password/token 계열 값과 URL
   user-info/query는 값 대신 `<redacted>`를 출력한다.

### M4-REQ-002 — 잠긴 의존성과 실행 provenance

1. Python 3.11용 direct+transitive dependency lock을 해시 검증 가능한 표준
   형식으로 커밋한다. `requirements.txt`는 의도 선언으로 유지할 수 있으나 CI와
   container의 공식 설치는 lock만 사용하고 dependency resolver 재해석을 하지
   않는다.
2. lock 생성 도구와 갱신 절차 자체를 버전 고정한다. Linux CPU 공식 배포
   profile을 우선 지원하며 macOS 개발 profile 차이는 명시한다.
3. `pip check`, fresh locked install, import smoke test가 CI에서 통과한다.
   frontend는 기존 `package-lock.json`과 `npm ci`를 계속 사용하며 지원 Node를
   `>=22.22.2 <23`으로 검증한다.
4. build/deployment metadata에는 Git SHA/dirty, Python/Node, settings hash,
   lock SHA-256, 핵심 dependency의 정렬된 버전 목록과 canonical SHA-256,
   image digest(컨테이너 실행 시)를 기록한다.
5. M3 baseline 파일은 byte 단위로 변경하지 않는다. M4 baseline은 M3
   fingerprint를 참조하고 새 dependency/settings/index-manifest fingerprint를
   추가한다.

### M4-REQ-003 — 구조화 로그

1. 기본 운영 출력은 한 줄 JSON이며 최소 필드는 `timestamp`, `level`, `event`,
   `service`, `version`, `request_id`(요청 범위), `duration_ms`(완료 event),
   `outcome`, `error_type`이다. 개발용 text format은 선택 가능하다.
2. request ID는 유효한 inbound header가 있으면 보존하고 아니면 생성하며 응답
   header에 반환한다. 각 request의 `request_started`와 `request_completed`는
   정확히 한 쌍이어야 한다.
3. routing, web search, retrieval(`query_embed`, `bm25`, `dense`, `rrf`, `mmr`,
   `reranker`), generation, fallback 단계 완료/오류를 같은 request ID로 연결한다.
4. 질문 원문, 답변, 검색 결과 본문, 문서 chunk, 로컬 절대 경로, stack trace는
   기본 INFO 로그에 기록하지 않는다. 진단에는 길이, 안정된 오류 코드,
   비가역적 짧은 hash만 허용한다. 예상 오류 응답에는 stack trace를 노출하지 않는다.
5. logging 실패가 정상 요청을 실패시키지 않아야 하며, secret redaction 단위
   테스트가 있어야 한다.

### M4-REQ-004 — 메트릭과 운영 진단

1. dependency가 작은 프로세스 로컬 pull endpoint(`/metrics`)를 제공한다.
   기본은 loopback/신뢰망 전용이며 외부 노출 시 proxy 보호 대상이다.
2. 최소 metric은 request 총수/진행 중/거절/오류, request latency, route/fallback,
   단계별 latency/error, readiness, index version, build info다.
3. label에는 route, stage, outcome, stable error type처럼 유한 집합만 허용한다.
   request ID, 질문, source, 파일명, exception message를 label로 사용하지 않는다.
4. 테스트는 metric 이름·label allowlist, cardinality 상한과 성공/실패/거절 증가를
   검증한다. 다중 프로세스 집계와 장기 저장은 M4 필수가 아니다.

### M4-REQ-005 — liveness/readiness와 시작·종료

1. `/health/live`는 event loop가 응답 가능한 동안 외부 Ollama나 모델 상태와
   무관하게 HTTP 200을 반환한다.
2. `/health/ready`는 settings 유효, 엔진 초기화 완료, index manifest/파일 검증
   완료, 새 query 수락 가능 조건을 모두 만족할 때만 HTTP 200이다. 시작 중,
   draining, 필수 artifact 불일치는 HTTP 503과 안정된 reason code를 반환한다.
3. Ollama readiness probe는 짧은 timeout과 TTL cache를 사용한다. 일시 장애 시
   readiness 정책(`strict` 기본)을 적용하되 liveness에는 영향을 주지 않는다.
4. readiness는 무거운 query/generation을 실행하지 않으며 credential, 절대 경로,
   exception message를 노출하지 않는다. 정상 환경에서 p95 250ms 이하,
   cached probe p95 50ms 이하를 만족한다.
5. 종료 시 즉시 not-ready/draining으로 전환하고 새 요청을 거절한 뒤, 진행 중
   요청을 설정된 grace period(기본 30초)까지 기다린다.
6. 기존 `/health`는 한 release 동안 deprecated alias로 유지하고 removal 안내를
   응답 header/문서에 제공한다.

### M4-REQ-006 — blocking 경계와 bounded concurrency

1. `/rag`의 동기 routing, DDGS, embedding/reranker, Ollama 호출은 event loop 밖
   bounded executor에서 수행한다. 제품 API 전체를 async로 재작성하지 않는다.
2. 기본 단일 process/worker에서 query 동시 실행 수와 대기 수를 명시적 설정으로
   제한한다(초기 기본값 각각 2와 4). FIFO 대기, query 전체 timeout(기본 90초),
   overload 거절(HTTP 503 + `Retry-After`)을 정의한다.
3. 취소/timeout이 slot 누수나 완료 후 잘못된 응답 쓰기를 만들지 않아야 한다.
   하위 동기 호출을 강제 중단할 수 없는 한계를 문서화하고, timeout 이후에도
   실행 중인 작업이 slot을 점유하도록 정직하게 계측한다.
4. singleton 엔진 초기화는 thread-safe 해야 한다. thread-safe가 입증되지 않은
   모델 단계는 별도 lock/semaphore로 직렬화하되 event loop는 막지 않는다.
5. health/static/metrics는 query executor와 분리되어 포화 중에도 응답한다.

### M4-REQ-007 — 요청·오류·네트워크 경계

1. 질문은 공백 제거 후 1자 이상, UTF-8 4,000 bytes 이하이며 request body는
   16KiB 이하가 기본이다. 초과/잘못된 입력은 모델 호출 없이 4xx로 거절한다.
2. client에는 stable error schema(`error.code`, 안전한 `message`, `request_id`,
   `retryable`)를 제공한다. 기존 성공 response schema는 변경하지 않는다.
3. trusted host와 CORS allowlist는 명시 설정이며 wildcard credential 조합은
   시작 실패한다. 기본 bind는 하위 호환을 위해 CLI 값과 일치시키되 운영
   runbook은 loopback 또는 사설망 bind를 기본 예제로 사용한다.
4. 외부 호출에는 connect/read/overall timeout이 있어야 하며 무제한 자동
   retry를 금지한다. retry가 있다면 횟수·backoff·멱등성을 명시한다.
5. 공개망 또는 신뢰 경계 밖 노출은 M4 앱 단독으로 허용하지 않는다. TLS,
   인증, 사용자/IP rate limiting, access log 보호를 제공하는 reverse proxy를
   필수 선행조건으로 문서화하고 검증 가능한 예시 설정을 제공한다.

### M4-REQ-008 — 안전한 index lifecycle과 provenance

1. 각 index version은 불변 디렉터리에 `index.faiss`, `index.pkl`, canonical
   manifest를 가진다. manifest에는 schema/version ID, 생성 시각, corpus hash,
   source 수/chunk 수, embedding model+revision, normalize, chunk size/overlap,
   FAISS type/dimension, 생성 settings hash, lock/dependency hash, 두 파일 hash를
   포함한다.
2. pickle은 신뢰 경계임을 명시한다. 서비스는 운영자가 소유하고 쓰기 권한을
   제한한 index root 아래의 manifest가 승인한 `index.pkl`만 로드하며, 업로드나
   임의 경로 입력으로 pickle을 로드하지 않는다.
3. build는 활성 version과 다른 staging 디렉터리에서 수행한다. fsync 가능한
   파일/디렉터리를 반영하고 모든 manifest/hash/load smoke 검증 후 같은
   filesystem의 atomic rename 또는 symlink/pointer 교체로만 활성화한다.
4. 동시 build/activate는 OS-level lock으로 하나만 허용하고 이미 잠긴 경우
   예측 가능한 종료 코드로 실패한다. 중단·disk full·검증 실패·설정 불일치는
   활성 version의 byte/hash/pointer를 변경하지 않는다.
5. 서비스는 시작 시 manifest와 현재 embedding/chunk 설정을 fail-closed로
   검증한다. legacy M3 index는 읽기 전용 migration/import 명령으로 manifest를
   생성하되 원본 두 파일의 byte/hash를 보존하고 자동 덮어쓰지 않는다.
6. 직전 정상 version을 최소 1개 보존하며 명시적 rollback은 manifest 검증 후
   pointer만 교체한다. retention 삭제는 현재/직전/사용 중 version을 보호하고
   dry-run을 기본으로 한다.
7. 실행 중 hot reload는 M4 필수가 아니다. 안전한 기본 배포는 index activate 후
   단일 service restart이며, 요청 중인 process의 index를 교체하지 않는다.

### M4-REQ-009 — OCI container와 운영 runbook

1. Linux CPU용 image를 multi-stage 또는 동등한 최소 구성으로 재현 가능하게
   build하고 immutable tag/digest 사용법을 제공한다. Python lock으로 설치하고
   build context에 runtime 문서, index, `.env`, report, cache가 들어가지 않는다.
2. container는 비루트 사용자, read-only root filesystem, no-new-privileges,
   drop-all capabilities로 실행 가능해야 한다. documents는 read-only,
   versioned index root는 서비스에서 read-only mount한다. index build는 별도
   운영 명령/쓰기 volume으로 분리한다.
3. healthcheck, resource limit 예시, graceful stop, Ollama 연결, model 사전 확보,
   volume 권한, backup/restore, 로그/메트릭, index build/activate/rollback과
   incident triage를 runbook에 포함한다.
4. CI는 image build, non-root identity, import/config check, mock 기반
   liveness/readiness smoke를 수행한다. 실제 Ollama/model을 요구하는 image test는
   명시적 live job으로 분리한다.

### M4-REQ-010 — 자동 gate, 보고서와 추적성

1. M4 비교 도구는 Git SHA/dirty, settings/lock/dependency/index fingerprint,
   host CPU/RAM, worker/concurrency 설정, warm-up, test double/live profile을
   report에 기록한다.
2. 자동 gate는 원시 count와 비반올림 수치로 판정하고 JSON과 Markdown에서 같은
   결과를 낸다. 미측정, schema mismatch, fingerprint mismatch는 pass가 아니다.
3. 요구사항 ID마다 설계, 구현, 테스트, 증거와 상태를 연결하는 Traceability를
   유지한다. 상세 질문/답변과 민감 로그는 Git 제외 위치에 두고 경로+SHA-256만
   커밋 문서에서 참조한다.
4. M4 완료는 아래 §5의 모든 필수 gate가 자동 통과한 뒤에만 선언한다. 사람
   worksheet나 별도 승인 없이 판정할 수 있어야 하며, 조건부 공개 배포 기능은
   비활성/미적용으로 명시하면 완료 결격이 아니다.

## 4. 비기능 요구사항

| ID | 요구사항 |
|---|---|
| M4-NFR-001 재현성 | clean Linux 환경에서 lock 설치와 image rebuild가 성공하고 dependency/settings/artifact fingerprint가 동일하다. |
| M4-NFR-002 신뢰성 | 시작·종료·과부하·timeout·index 실패가 fail-closed이며 활성 artifact와 slot 상태를 손상하지 않는다. |
| M4-NFR-003 성능 | event-loop 응답성, bounded resource, 부하 gate를 충족하고 M3 단일 요청 품질·latency를 허용 범위 내 보존한다. |
| M4-NFR-004 보안·프라이버시 | 최소 권한, 입력 제한, pickle trust boundary, secret/질문/답변 비기록, 안전한 오류를 기본값으로 한다. |
| M4-NFR-005 호환성 | 성공 API, CLI entry point, runtime 경로 우선순위와 M3 rollback flags/artifact를 보존한다. |
| M4-NFR-006 유지보수성 | stdlib/작은 dependency와 현재 FastAPI 구조를 우선하고 새 추상화는 설정·관측·index lifecycle 경계에 한정한다. |
| M4-NFR-007 검증성 | 외부 네트워크·대형 모델 없는 결정론적 CI와 별도의 명시적 live/부하 profile을 모두 제공한다. |

## 5. 정량 수용 기준

### 5.1 필수 자동 gate

| 영역 | 자동 판정 기준 |
|---|---|
| 정적 회귀 | dataset validate, 전체 `pytest`, `npm ci && npm test`, vendor sync diff, Markdown link, `git diff --check` 모두 성공 |
| dependency | fresh Linux locked install 성공, hash 검증, `pip check` 0 오류, 동일 lock 2회 snapshot hash 동일, Node `>=22.22.2 <23` |
| settings | 유효/경계/상호모순 fixture 100% 통과, 잘못된 설정은 모델 import 전 exit 2, redaction fixture에서 secret 평문 0건 |
| logging | 성공·4xx·5xx·timeout·overload 각각 schema 100%, request start/end 쌍 누락 0, 금지 payload/secret 0건 |
| metrics | allowlist 밖 label 0, 요청 1,000개 고유 ID/질문 주입 후 time-series 수가 설계 상한(150) 이하 |
| health | live는 dependency 실패/포화 중 200; ready 상태표 전 조합 expected status 100%; cached ready p95 ≤50ms, uncached p95 ≤250ms |
| event loop | 1개 2초 blocking fake query 중 `/health/live` 20회 p95 ≤100ms, 최대 ≤250ms |
| bounded load | mock 고정 2초 query, concurrency=2/queue=4에서 동시 8요청: 실행 중 ≤2, 대기 ≤4, 나머지 2는 1초 내 503, slot 누수 0 |
| 정상 부하 | mock 200ms query 40건, concurrency=2에서 HTTP 성공률 100%, 예상 20 sequential wave 대비 wall time +20% 이내, p95 queue+service ≤650ms |
| live smoke | 기준 host에서 4 동시 사용자×각 3 query(고정 12건), 서버 5xx/timeout 0; accepted request 성공률 100%; 단일 동시성-1 warm 기준 p95 대비 accepted p95 ≤2.5배 |
| M3 품질 | Retrieval/ Routing/Answer의 기존 14 gate 전부 통과, dataset/corpus fingerprint 동일, M3 baseline 파일 byte 변경 0 |
| 단일 요청 성능 | 같은 host/profile의 M4 concurrency=1 warm 결과가 M3-compatible candidate 대비 Retrieval·Answer mean/p95 각각 +20% 이하 |
| index | 중단/disk-full/hash mismatch/동시 build/잘못된 설정 fault injection 전부에서 활성 pointer와 파일 hash 변화 0; 정상 build/rollback 100회 pointer partial state 0 |
| container | clean image build, 비루트 UID, read-only rootfs+drop capabilities smoke, health/readiness mock test, runtime/secret의 image layer 포함 검사 0건 |

부하 수치는 하드웨어 절대 성능보다 경계의 정확성과 같은 host/profile의 상대
회귀를 우선한다. live 12건은 M3 dataset에서 document/web route, 긴 answer,
abstention을 포함한 고정 case ID로 설계 단계에서 확정한다. 외부 DDGS 변동은
route 성공과 안전한 fallback을 판정하며 검색 결과 내용 자체를 gate로 삼지 않는다.

### 5.2 완료 판정

다음 조건을 모두 만족하면 human gate 없이 M4를 완료로 판정한다.

1. 필수 요구사항 M4-REQ-001~010과 NFR이 Traceability에서 모두 `PASS`다.
2. §5.1의 필수 자동 gate가 하나도 `FAIL`/`UNKNOWN`/`NOT_RUN`이 아니다.
3. 설계·코드 리뷰의 CRITICAL/MAJOR가 0이고 MINOR가 최소화되며 품질 점수가
   9.7/10 이상이다. iteration/중단 규칙은 repository의 최신
   `milestone_dev_orchestration_guide.md`를 따른다.
4. 조건부 TLS/인증/rate limit은 배포 profile이 loopback/신뢰망이면
   `NOT_APPLICABLE` 사유와 외부 노출 금지 검증으로 닫을 수 있다. 공개 profile이면
   reverse proxy integration test 없이는 완료할 수 없다.
5. 최종 M4 baseline과 rollback 증거가 생성되고 Roadmap/Problem/runbook 링크가
   깨지지 않는다.

## 6. 호환성, migration과 rollback

- public 성공 response와 CLI 이름은 유지한다. 새 오류 schema는 실패 응답에만
  적용하며 deprecated `/health`는 한 release 유지한다.
- 설정 전환은 기존 환경변수 이름을 우선 수용한다. rename이 불가피하면 한 release
  alias+warning을 제공하고 두 값 충돌 시 실패한다.
- M3 `MMR_VECTOR_SOURCE=embed`, `ROUTING_SIGNAL_OVERRIDE=0`,
  `ANSWER_TEMPLATE_MODE=intent` rollback flag와 classifier artifact를 제거하지 않는다.
- legacy M3 index import는 copy/manifest 생성만 수행하고 원본을 수정하지 않는다.
  새 lifecycle 문제 시 서비스 version rollback + 승인된 M3 index 경로 지정으로
  복구할 수 있어야 한다.
- schema/manifest/lock 변경은 버전 상승과 forward validation을 요구한다. 새 binary가
  이해하지 못하는 schema를 조용히 읽지 않는다.
- DB migration은 없다. container 도입은 host 실행을 즉시 제거하지 않으며 M4 동안
  두 실행 방식의 같은 settings/index 계약을 유지한다.

## 7. 전제와 열린 쟁점

자동 진행을 막는 human decision은 없다. 상세 설계는 아래 기본 결정을 사용하고
실측 증거가 기준을 깨뜨릴 때에만 변경 사유와 원래 기준을 기록한다.

- 공식 배포 profile은 Linux x86_64 CPU, Python 3.11, Node 22.22.2 이상,
  단일 Uvicorn worker다. GPU image와 macOS production은 범위 밖이다.
- 기본 concurrency=2/queue=4는 보수적인 시작값이며 Phase 4 부하 결과로 **더 낮출
  수는 있어도**, 메모리·thread safety 증거 없이 높이지 않는다.
- readiness의 기본은 Ollama까지 요구하는 `strict`다. 문서-only degraded readiness는
  향후 별도 profile로 명시할 때만 허용한다.
- 메트릭은 프로세스 재시작 시 초기화된다. 보존/alerting/SLO는 운영자가 실제
  다중 인스턴스 또는 장기 추세 요구를 제시할 때 조건부 도입한다.
- 앱 내 사용자 계정/권한 모델은 만들지 않는다. 신뢰망 밖 노출 시 proxy 인증을
  채택한다는 배포 경계가 M4의 결정이다.
