# M4.1 Configuration & Observability Foundation 요구사항

상태: **착수**  
상위 결정: [M4 복구 결정](../m4-production-readiness/Recovery_Decision.md)  
승인 기준선: [M3 baseline](../../../evaluation/baselines/m3_initial.md)

## 1. 목적

M4.1은 후속 serving/index/container 작업이 공통으로 사용할 재현 가능한 설정과
관측 기반을 만든다. 동시성·timeout·index lifecycle을 함께 해결하지 않고 현재
제품 동작과 M3 품질을 보존한다.

## 2. 필수 범위

### M4.1-REQ-001 — Python/Node 실행 재현성

1. 공식 profile은 Linux x86_64 CPU, Python `>=3.11,<3.12`, Node
   `>=22.22.2 <23`이다.
2. Python direct/transitive dependency는 hash 검증 가능한 lock으로 고정한다.
3. clean locked install, `pip check`, Node engine 및 `npm ci`가 CI에서 통과한다.
4. dependency snapshot과 lock/package hash를 canonical JSON으로 기록한다.

### M4.1-REQ-002 — Settings 단일 원본

1. 모든 현재 운영 설정을 field/type/default/env alias/validator/consumer로 inventory한다.
2. immutable typed `Settings`가 단일 원본이며 제품 모듈의 직접 `os.environ` 접근을
   제거한다.
3. 기존 환경변수 이름과 `config.py` 공개 상수 import는 한 release 호환한다.
4. unknown enum, 범위 오류, 상호모순은 모델·index 초기화 전에 exit code 2로 실패한다.
   **예외**: `simple-qna-rag-web`(web 서버 프로세스, `--check-config` 미사용 시)은
   PROJECT_ROOT 기반 고정 상수만으로 부팅하고, 나머지 settings 검증 실패는 exit
   대신 `/health/ready` 503(`reason=settings_invalid`)로 표현한다(REQ-005.2).
   `--check-config`·query·index CLI는 이 예외 대상이 아니며 원칙대로 exit 2를
   유지한다(Design.md §3).
5. `--check-config`는 외부 모델 없이 검증하고 secret/credential/절대 private path를
   출력하지 않는다.
6. field count와 facade mapping은 schema에서 계산하며 magic number를 사용하지 않는다.

### M4.1-REQ-003 — Payload-safe structured logging

1. JSON event는 timestamp, level, event, service, version과 요청 범위 request ID를
   포함한다.
2. request 시작/종료, routing, web, retrieval, generation, startup/readiness 오류를
   bounded event/stage/error allowlist로 기록한다.
3. 질문, 답변, 문서 내용, 검색 원문, prompt, secret, credential, 사용자 절대 경로는
   로그에 기록하지 않는다.
4. logging backend 실패는 제품 요청을 실패시키지 않는다.

### M4.1-REQ-004 — Bounded process-local metrics

1. request/response status, stage duration/error, route/fallback, readiness reason을
   process-local Prometheus metrics로 제공한다.
2. label 값은 enum/allowlist만 허용하고 request ID, 질문, source path, exception text,
   index hash를 label로 사용하지 않는다.
3. 고유 요청 1,000건 후 실제 collector sample 수가 설계 상한 150 이하이다.
4. created-series 정책은 deployment 초기화 또는 prometheus-client public API로
   설정하며 metrics 모듈이 환경변수를 직접 수정하지 않는다.

### M4.1-REQ-005 — 기본 health 계약

1. `/health/live`는 event loop가 응답 가능한 동안 dependency 상태와 무관하게 200이다.
2. `/health/ready`는 settings 유효성과 현재 engine 초기화 성공 여부를 구분해
   stable reason code로 200/503을 반환한다.
3. 기존 `/health`는 한 release deprecated alias로 유지한다. `Sunset` 헤더는
   `Fri, 06 Nov 2026 00:00:00 GMT`(RFC 8594)로, 제거 시점은 패키지 버전
   0.3.0(M4.2 릴리스)으로 고정한다(Design.md §11.2, Roadmap.md와 동기화).
4. M4.2의 queue saturation/orphan 조건은 이번 단계에 포함하지 않고 확장 seam만 둔다.

### M4.1-REQ-006 — 호환성과 자동 증거

1. `/rag` 성공 response schema와 세 CLI entry point, M3 rollback flag를 보존한다.
2. M3 baseline 파일과 runtime vectorstore를 변경하지 않는다.
3. Requirement ID별 구현·테스트·증거를 Traceability에 연결한다.
4. JSON/Markdown 결과는 같은 판정 모델에서 생성한다.

## 3. 제외 범위

- request offload/concurrency/queue/timeout/cancel/drain과 부하 tuning: M4.2
- versioned index/import/activate/rollback: M4.3
- Docker production image, deployment runbook, 전체 14-gate assembly: M4.3
- 인증, 외부 metrics backend, distributed worker/vector DB/Kubernetes: 조건부/M5

## 4. 수용 기준

| Gate | 완료 기준 |
|---|---|
| dependency | clean locked install, hash verification, `pip check`, Node engine, `npm ci` PASS |
| settings | inventory/schema/facade 1:1, valid/boundary/conflict fixtures PASS, direct env read 0, secret 평문 0 |
| logging | success/4xx/5xx/startup fixture schema 100%, start/end 누락 0, 금지 payload 0 |
| metrics | allowlist 밖 label 0, 실제 sample ≤150, 1,000 unique payload 회귀 PASS |
| health | settings/engine 상태표 100%, live dependency 독립성 PASS |
| regression | 전체 pytest/frontend/vendor/Markdown/diff PASS, M3 14 gate와 baseline bytes 보존 |
| review | CRITICAL/MAJOR 0, MINOR 최소화, 9.7/10 이상 |

모든 필수 Gate가 PASS여야 하며 `UNKNOWN`/`NOT_RUN`은 완료가 아니다.

