# M4 Production Readiness 개발 계획

상태: **동결 — 분할 복구 결정 이전의 감사 기록**  
후속: [M4 복구 결정](Recovery_Decision.md), [M4.1 계획](../m4.1-configuration-observability/Plan.md)  
요구사항: [Requirement.md](Requirement.md)

## 1. 실행 원칙과 역할

M4는 운영 기반을 한꺼번에 재작성하지 않고, 재현성 → 설정 → 관측 → 서비스
경계 → index → 배포 순으로 좁은 vertical slice를 완성한다. 각 Phase는 이전
Phase의 자동 증거를 재사용하며 CRITICAL/MAJOR 0, 품질 9.7/10 이상일 때만 다음
Phase로 진행한다.

- **Codex**: 요구사항·계획, 상세 설계/코드 리뷰와 추적성 검증
- **Claude Code**: 상세 설계, 구현·테스트·리뷰 반영, 최종 승인 뒤 Git 작업
- **프로젝트 리더**: 단계 Gate와 오케스트레이션. 자동 기준으로 판정하며 별도
  사용자 승인이나 worksheet를 기다리지 않는다.

이번 문서 단계에서는 제품 코드를 변경하지 않는다. 상세 설계는 정확한 파일,
schema, metric 이름, error code, lock 도구와 fault injection 방법을 확정하되
[요구사항](Requirement.md)의 수치·호환·실패 안전 계약을 완화하지 않는다.

## 2. 단계 흐름

```text
Phase 0 M3 기준·위험 고정
   -> Phase 1 dependency lock + typed settings
   -> Phase 2 structured logging + metrics
   -> Phase 3 health + blocking/concurrency/input boundary
   -> Phase 4 부하·장애 검증 및 tuning
   -> Phase 5 versioned index lifecycle
   -> Phase 6 container + security/deployment runbook
   -> Phase 7 clean 통합 검증 + M4 baseline/추적표
```

Phase 2의 문서/schema 설계와 Phase 5의 manifest 설계는 병렬 리뷰할 수 있으나,
같은 production 파일 구현과 공식 성능 실행은 직렬화한다. Phase 6 container는
Phase 1 lock과 Phase 3 health 계약이 고정된 뒤 시작한다. index fault injection과
부하 실행은 공유 runtime artifact를 건드리지 않는 임시 디렉터리에서 수행한다.

## 3. 공통 Gate와 증거 계약

각 Phase는 다음 순서로 닫는다.

1. Requirement ID별 설계/코드/테스트/결과를 Traceability 초안에 연결한다.
2. Phase 결정론적 테스트와 전체 정적 회귀를 실행한다.
3. report에 Git SHA/dirty, settings/lock/dependency/index fingerprint와 profile을
   기록하고 JSON/Markdown 판정 일치를 검사한다.
4. Codex 리뷰에서 CRITICAL/MAJOR 0, MINOR 최소화, 9.7/10 이상을 확인한다.
5. 프로젝트 리더가 자동 증거로 다음 Phase 진행을 결정한다.

공통 명령은 구현 중 상세 설계에 맞춰 lock 명령이 추가되더라도 아래 계약을
최소로 유지한다.

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
pytest -q
npm ci
npm test
npm run sync-vendor
git diff --exit-code -- web/static/vendor
python scripts/check_markdown_links.py
git diff --check
```

live Ollama/DDGS, 부하, container는 명시 profile로 분리한다. 환경 제약으로 실행할
수 없는 선택적 진단은 기록할 수 있지만 Requirement §5.1의 필수 gate는
`NOT_RUN`으로 완료 처리할 수 없다. 리뷰 iteration과 중단 조건은 최신
`milestone_dev_orchestration_guide.md`를 그대로 적용한다.

## 4. Phase별 계획

### Phase 0 — M3 baseline과 운영 위험 고정

목표: M4 변경 전 승인 artifact, API, dependency, event-loop/index 위험을 재현
가능한 증거로 고정한다.

작업:

1. M3 baseline 두 파일 SHA-256, dataset/corpus/index fingerprint, 14개 gate와
   현재 Git dirty 파일을 기록한다. 사용자의 guide rename은 수정하지 않는다.
2. 현재 Python/Node/pip dependency snapshot과 `requirements.txt`,
   `package-lock.json` hash를 수집하고 `pip check` 불일치를 분류한다.
3. FastAPI startup, `/rag`, `/health`, singleton 초기화, sync Ollama/DDGS/model,
   index direct-save 경계를 호출 그래프로 기록한다.
4. mock 2초 query 중 기존 `/health` latency와 1/2/4 동시 요청의 처리량·RSS·thread
   수를 측정해 “개선 전” 진단값을 남긴다. 이는 M3 품질 baseline을 대체하지 않는다.
5. 공식 Linux CPU/container profile, 고정 live 12 case ID, 오류/metric label
   allowlist 초안을 상세 설계에서 확정한다.

산출물: `Design.md` baseline/risk 절, Phase 0 report, `Traceability.md` 초안.

수용 기준:

- M3 baseline 파일 byte와 네 fingerprint가 승인값과 일치한다.
- dependency snapshot이 정렬·canonical hash로 재실행 시 동일하다.
- blocking과 index 손상 가능 경계가 누락 없이 Requirement ID에 연결된다.
- 측정 로그는 Git 제외 경로에 있고 report에는 경로+SHA-256만 남는다.

관련 요구사항: M4-REQ-002, M4-REQ-005~006, M4-REQ-008, M4-REQ-010,
M4-NFR-001~007

### Phase 1 — dependency lock과 typed settings

목표: 모델을 로드하기 전에 설치와 설정의 단일 재현 계약을 만든다.

작업:

1. Linux CPU Python 3.11 lock 형식/생성 도구를 선택해 도구 버전, 갱신/검토
   절차와 direct/transitive 분리를 설계한다. Torch CPU wheel source 등 플랫폼
   선택을 명시하고 broad resolver install을 공식 경로에서 제거한다.
2. CI에 hash-verified fresh install, `pip check`, package import를 추가한다.
   Node engine을 선언하고 `npm ci` 환경을 22.22.2 이상으로 고정한다.
3. immutable settings model, precedence(default < env < explicit CLI), validation,
   deprecated alias/conflict, canonical redacted dump/hash를 구현한다.
4. 기존 상수 consumer를 얇은 facade 또는 settings injection으로 단계 이동한다.
   import-time 대형 model/network I/O는 추가하지 않는다.
5. 유효/최솟값/최댓값/상호모순/URL redaction/legacy path conflict와 외부 cwd
   실행을 단위·통합 테스트한다.

검증 예시:

```bash
python -m pip install --require-hashes -r <linux-lock-file>
python -m pip check
simple-qna-rag-web --check-config
pytest -q tests/unit/test_settings.py tests/integration/test_cli_entrypoints.py
```

수용 기준:

- Requirement §5.1 dependency/settings gate가 전부 통과한다.
- 같은 입력의 settings JSON/hash와 dependency snapshot hash가 반복 실행에서 같다.
- 잘못된 설정은 모델/Ollama 초기화 전에 exit 2이며 secret 평문이 없다.
- 기존 runtime path 우선순위, entry point와 M3 rollback flag 테스트가 통과한다.

관련 요구사항: M4-REQ-001~002, M4-REQ-010, M4-NFR-001, M4-NFR-004~007

### Phase 2 — structured logging과 bounded metrics

목표: 요청과 모든 주요 단계를 상관관계로 추적하면서 payload와 고 cardinality를
노출하지 않는다.

작업:

1. event schema, request context propagation, stable error taxonomy와 redaction
   filter를 구현하고 기존 `print()`를 단계적으로 event로 교체한다.
2. inbound/generated request ID validation과 response header를 추가한다.
3. 기존 `RetrievalTrace` 계측을 제품 관측 경계와 연결하되 평가 trace 계약을
   깨뜨리거나 같은 latency를 중복 측정하지 않는다.
4. request/stage/route/fallback/readiness/index/build metric을 구현하고 label
   allowlist를 코드와 문서의 단일 원본으로 둔다.
5. 1,000개 고유 질문/request ID, secret/URL/path/exception fixture로 로그 및
   cardinality 검증을 수행한다.

수용 기준:

- 성공·실패·timeout·fallback request의 start/end 쌍과 stage가 같은 ID로 연결된다.
- INFO 로그의 질문/답변/chunk/절대 경로/secret 유출 0건이다.
- time-series 상한 150과 metric 증가량 gate가 통과한다.
- logging/metrics sink 오류 주입에도 요청의 본래 결과가 유지된다.

관련 요구사항: M4-REQ-003~004, M4-REQ-010, M4-NFR-003~004, M4-NFR-006~007

### Phase 3 — health, lifecycle, blocking과 입력 경계

목표: event loop와 무거운 singleton을 보호하면서 준비 상태와 실패를 예측
가능하게 만든다.

작업:

1. FastAPI lifespan으로 startup/draining/shutdown 상태 머신을 구현하고 deprecated
   `/health`, 새 `/health/live`, `/health/ready` 계약을 추가한다.
2. settings/index/engine/Ollama readiness probe를 분리하고 timeout+TTL cache와
   안전한 reason code를 적용한다.
3. sync `route_query()` 전체를 bounded executor로 offload한다. semaphore/queue
   원자성, FIFO, timeout, cancellation, shutdown drain을 상세 설계의 상태표로
   구현한다.
4. engine/router singleton의 thread-safe 초기화를 보장하고 실제 model component별
   thread-safety가 불명확하면 최소 임계구역 semaphore를 둔다.
5. request/question 크기, trusted host/CORS, stable error schema, `Retry-After`,
   외부 호출 timeout을 구현한다.
6. 느린 fake, never-return-until-released fake, cancellation/timeout/overload,
   startup failure와 dependency flap을 결정론적 ASGI 테스트로 검증한다.

수용 기준:

- Requirement §5.1 health/event-loop/bounded-load gate가 모두 통과한다.
- 최대 running=2, waiting=4가 어떤 cancellation 순서에서도 초과되지 않고
  종료 후 slot/queue가 0이다.
- health/static/metrics는 query 포화와 Ollama 장애에도 정해진 latency로 응답한다.
- 기존 성공 `/rag` schema와 agent fallback 회귀가 유지된다.

관련 요구사항: M4-REQ-005~007, M4-NFR-002~007

### Phase 4 — 부하·장애 검증과 보수적 tuning

목표: 기본 concurrency가 현재 모델/인덱스에 안전함을 mock과 실제 profile에서
입증하고 자동 gate 도구를 고정한다.

작업:

1. in-process ASGI mock load와 실제 HTTP live load를 분리한 재현 가능한 harness를
   작성한다. percentile, queue/service/total latency, RSS peak, threads, rejection,
   slot leak를 JSON/Markdown으로 출력한다.
2. mock 200ms 40건, mock 2초 8건, timeout/cancel burst를 반복해 결정론적 gate를
   검증한다.
3. 동일 host에서 concurrency=1 warm 비교 후 고정 12건을 4 client로 실행한다.
   M3 14개 품질 gate도 공식 candidate로 재실행한다.
4. concurrency 1/2를 비교한다. 기본 2가 메모리/thread safety/live 수용 기준을
   못 지키면 1로 낮추고 원인을 기록한다. 외부 queue/worker를 도입하지 않는다.
5. fault injection으로 Ollama timeout, DDGS failure, executor saturation, client
   disconnect, logging failure를 검증한다.

수용 기준:

- Requirement §5.1 정상 부하/live smoke/단일 요청/M3 품질 gate가 모두 통과한다.
- accepted/rejected/timeout 합계가 요청 수와 정확히 일치하고 report에 미분류
  outcome이 0이다.
- 설정 기본값은 검증된 가장 단순하고 보수적인 값이다.

관련 요구사항: M4-REQ-004~007, M4-REQ-010, M4-NFR-002~003, M4-NFR-007

### Phase 5 — versioned index lifecycle

목표: 기존 M3 index를 보존하면서 provenance가 검증된 새 index만 원자적으로
활성화하고 되돌릴 수 있게 한다.

작업:

1. canonical manifest schema/version ID, 디렉터리/pointer/lock/staging/retention
   layout과 trust boundary를 상세 설계한다.
2. 기존 index CLI를 build → hash/manifest → load smoke → activate 단계로 분리한다.
   build와 service가 같은 라이브 경로에 쓰지 못하게 한다.
3. M3 legacy import를 임시 복사본에서 실행해 원본 hash 보존과 새 manifest를
   확인한다. 서비스 startup validation과 actionable 오류를 추가한다.
4. atomic pointer 교체, explicit rollback, dry-run cleanup과 OS lock을 구현한다.
5. 각 write/rename/fsync/validation 지점에 crash/disk-full/corruption/concurrent
   process fault를 주입하고 활성 artifact 불변성을 검사한다.
6. `CorpusManifestError`를 index/retrieval/answer/baseline CLI에서 안정된 오류
   code/exit로 통일하고 traceback을 기본 노출하지 않는다.

수용 기준:

- manifest가 Requirement M4-REQ-008의 모든 provenance 필드를 포함하고 canonical
  재직렬화 hash가 같다.
- 정상 build/rollback 100회 partial pointer 0, 모든 실패 주입에서 활성 hash 변화
  0이며 lock 경쟁자가 예측 가능한 종료 코드를 받는다.
- M3 두 index 파일과 baseline byte가 변경되지 않는다.
- wrong embedding/chunk/schema/hash가 readiness 503과 안전한 reason code로
  fail-closed한다.

관련 요구사항: M4-REQ-008, M4-REQ-010, M4-NFR-001~002, M4-NFR-004~007

### Phase 6 — container, security와 deployment runbook

목표: 동일 lock/settings/index 계약을 최소 권한 OCI 실행으로 패키징하고 운영자가
배포·복구할 수 있게 한다.

작업:

1. Linux CPU multi-stage Dockerfile과 엄격한 `.dockerignore`를 추가한다. dependency
   layer는 lock 기반이며 runtime user는 비루트다.
2. read-only rootfs, tmpfs, drop-all capabilities, no-new-privileges, resource/stop
   설정을 가진 예시 실행/Compose profile을 작성한다. Ollama와 runtime artifact를
   image에 bake하지 않는다.
3. loopback/사설망 기본 runbook과 외부 노출용 reverse proxy profile을 구분한다.
   후자는 TLS/auth/rate limit/request body limit/forwarded header trust를 포함한다.
4. deploy, model preflight, settings check, readiness, index activate/restart,
   backup/restore, rollback, dependency/Ollama/index/overload incident triage를
   명령 단위로 문서화한다.
5. CI에 image build, layer content, UID, read-only/capability와 mock health smoke를
   추가하고 dependency action/image tag는 immutable SHA 또는 정책상 고정 버전을
   사용한다.

수용 기준:

- Requirement §5.1 container gate가 clean runner에서 통과한다.
- image/layer에 `runtime/`, `.env`, evaluation report, model cache, Git metadata와
  known secret fixture가 없다.
- service는 read-only/non-root/drop-capabilities 상태에서 mock query와 health를
  처리한다.
- 공개 profile은 proxy integration test를 통과하거나 조건부 `NOT_APPLICABLE`로
  명시되고 외부 bind 금지 검증이 통과한다.

관련 요구사항: M4-REQ-007, M4-REQ-009~010, M4-NFR-001~007

### Phase 7 — clean 통합 검증과 M4 baseline

목표: 모든 채택 변경의 상호작용을 clean 환경에서 검증하고 사람 판단 없이 M4
완료 여부를 산출한다.

작업:

1. clean locked host install과 clean image에서 공통 Gate, settings/observability,
   lifecycle/index/security test를 전부 실행한다.
2. mock load/fault suite, live 12-case load와 M3 통합 14 gate를 최종 설정으로
   재실행한다.
3. M3 baseline byte/hash, dataset/corpus/index migration 전후 hash와 public API/CLI
   호환성을 비교한다.
4. 최종 `Traceability.md`, 운영 runbook, 채택/조건부/제외 결정, 잔여 위험과
   rollback drill 결과를 완성한다.
5. machine-readable M4 baseline과 사람이 읽는 요약을 고정하고 자동 gate
   `overall_pass=true`일 때만 Roadmap을 완료로 변경한다.

최종 검증 명령군:

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
pytest -q
npm ci && npm test && npm run sync-vendor
git diff --exit-code -- web/static/vendor
python scripts/check_markdown_links.py
<locked-install-and-snapshot-check>
<settings-observability-health-load-index-fault-gate>
<container-security-smoke>
RUN_LIVE_LLM_TESTS=1 <m4-live-and-m3-quality-gate>
git diff --check
```

수용 기준:

- Requirement §5.1의 모든 필수 gate가 `PASS`이고 자동 결과가
  `overall_pass=true`다.
- Traceability에서 필수 ID의 미연결/미검증/UNKNOWN이 0이다.
- CRITICAL/MAJOR 0, 점수 9.7/10 이상이며 unresolved MINOR가 있다면 영향과
  후속 조건이 명시된다.
- 조건부 항목은 profile과 `NOT_APPLICABLE` 근거가 기계 판독 가능하다.

관련 요구사항: M4-REQ-001~010, M4-NFR-001~007

## 5. 변경 예상 영역과 소유 경계

상세 설계가 최종 파일명을 정하지만 변경은 다음 책임 경계에 한정한다.

- 설정/호환 facade: `src/simple_qna_rag/config.py`와 소형 settings 모듈
- 서비스 lifecycle/HTTP boundary: `src/simple_qna_rag/web/server.py`
- 관측 seam: agent, RAG, web search의 기존 단계 경계와 소형 observability 모듈
- index lifecycle: `cli/index_documents.py`와 독립 manifest/activation 모듈
- 검증: `tests/unit`, `tests/integration`, 별도 load/fault/container smoke
- 배포: root의 container ignore/build 파일과 `docs/operations/` runbook
- 재현성/CI: dependency lock, `pyproject.toml`, `.github/workflows/ci.yml`

평가 dataset, 승인 M2/M3 baseline, corpus 및 현재 runtime index는 수정 대상이
아니다. 외부 DB/queue, Kubernetes, 인증 사용자 DB, RAG algorithm 교체는 만들지
않는다.

## 6. 위험과 대응

| 위험 | 대응/rollback |
|---|---|
| dependency lock이 CPU/GPU/macOS를 동시에 만족하지 못함 | Linux CPU를 공식 production profile로 고정하고 dev 차이를 별도 문서화; 기존 requirements는 의도 선언으로 보존 |
| thread offload 뒤 모델 component race/RSS 증가 | component별 stress test와 최소 semaphore; 실패하면 concurrency=1로 낮추되 event loop offload는 유지 |
| timeout 뒤 sync 작업이 계속 실행 | slot을 실제 완료까지 유지하고 orphan count를 metric으로 노출; process hard timeout은 proxy/graceful restart runbook으로 처리 |
| readiness가 외부 Ollama flap으로 진동 | 짧은 TTL cache와 stable reason; liveness 분리; strict/degraded 정책을 설정 hash에 포함 |
| metric cardinality/로그 개인정보 증가 | 코드 allowlist, payload 금지 테스트, 1,000 고유 입력 cardinality gate |
| 새 index schema가 기존 artifact를 못 읽음 | read-only legacy import, 원본 hash 보존, explicit old service+index rollback |
| container가 모델/runtime를 bake해 거대·민감해짐 | strict ignore/layer scan, read-only volume, Ollama 별도 서비스 |
| 운영 기능이 품질 latency를 회귀 | 같은 host concurrency=1 상대 gate와 M3 14 gate 실패 시 기능 flag/service version rollback |

## 7. 완료 이후 조건부 후속

다음은 M4 실측 또는 운영 요구가 발생할 때만 별도 마일스톤으로 연다.

- 다중 replica metrics aggregation, tracing backend와 SLO alert
- 사용자 계정·세션·권한을 포함한 앱 내부 인증
- GPU image, multi-architecture build와 supply-chain signing/SBOM 배포 정책 확대
- 외부 vector DB, 분산 queue, autoscaling/Kubernetes
- hot index reload와 무중단 다중 replica rollout
