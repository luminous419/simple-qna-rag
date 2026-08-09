# M4 요구사항 추적표 (초안)

상태: **초안 v4 — 구현 전, [Design.md](Design.md)가 [Design Review
Iteration 1](Design_Review_Iteration_1.md) MAJOR M-01~M-07/MINOR
m-01~m-03/TRIVIAL t-01, [Design Review
Iteration 2](Design_Review_Iteration_2.md) MAJOR M2-01~M2-06/MINOR
m2-01~m2-04, [Design Review Iteration 3](Design_Review_Iteration_3.md)
MAJOR M3-01~M3-07/MINOR m3-01~m3-02를 모두 반영한 상태를 추적**
근거: [Design.md](Design.md), [Requirement.md](Requirement.md)

모든 상태는 `PLANNED`다. 구현/테스트/증거 열은 각 Phase 완료 시 채운다.
Design 절 번호는 Iteration 3 개정 이후 번호를 기준으로 한다(Design.md §14는
Iteration 1 대응, §15는 Iteration 2 대응, §16은 Iteration 3 대응 매핑).

## 1. 기능 요구사항

| ID | Design 절 | 구현 파일(예정) | 테스트(예정) | 증거 | 상태 |
|---|---|---|---|---|---|
| M4-REQ-001 typed settings | §4.3, §4.3a, §4.3b, §4.4 | `src/simple_qna_rag/settings.py` | `tests/unit/test_settings.py`(규칙 7 upstream 예산 상호제약 + M3-07 필드 수/값 이관 검증 포함), `tests/unit/test_no_direct_environ_access.py`(§4.3b, m2-04 AST 정적 gate) | Phase 1 report, `evidence/settings.json` | PLANNED |
| M4-REQ-002 dependency lock | §4.1, §4.2 | `requirements/lock-linux-py311.txt`, `.github/workflows/ci.yml` | CI install job | Phase 1 report, `evidence/dependency.json` | PLANNED |
| M4-REQ-003 구조화 로그 | §5.1-§5.3, §5.6 | `src/simple_qna_rag/observability/__init__.py`(`ObservationSink`, `safe_sink_call`), `observability/logging.py`, `observability/request_id.py` | `tests/integration/test_logging_contract.py`, `tests/integration/test_observation_sink.py`(§5.3/§5.6 M2-03/M3-03 — retrieval substage/fallback 이벤트, `safe_sink_call()`이 stage/substage/fallback 세 호출 모두를 개별 격리함을 검증, `RetrievalStageTrace` 3필드 스키마 불변) | Phase 2 report, `evidence/logging.json` | PLANNED |
| M4-REQ-004 메트릭 | §5.4-§5.6 | `src/simple_qna_rag/observability/metrics.py` | `tests/unit/test_metrics_cardinality.py`(1,000 고유 입력 + 100회 index activate/rollback cardinality 상한, m2-03 실제 Prometheus sample 계약으로 재계산한 139/150 예산, M3-07 `disable_created_metrics()` non-env 호출 검증) | Phase 2 report, `evidence/metrics.json` | PLANNED |
| M4-REQ-005 liveness/readiness | §6.1, §6.1a, §6.2 | `src/simple_qna_rag/web/server.py`, `web/concurrency.py`(`begin_drain`/`shutdown_pool`) | `tests/integration/test_health_state_table.py`, `tests/integration/test_shutdown_drain.py`(m2-01 idle 즉시 완료/public pool API 포함) | Phase 3 report, `evidence/health.json` | PLANNED |
| M4-REQ-006 blocking/concurrency | §6.4, §6.5, §6.7 | `src/simple_qna_rag/web/concurrency.py`, `net_budget.py`(`DeadlineBudget`) | `tests/integration/test_web_concurrency.py`(§6.4 M2-01/M3-01 결정론적 race 테스트 12종: 기존 9종 + `queue_grant_then_cancel_before_submit`/`cancel_during_submit_failure_finalize`/`callback_queued_then_loop_shutdown`) | Phase 3/4 report, `evidence/bounded_load.json` | PLANNED |
| M4-REQ-007 요청/오류/네트워크 경계 | §6.3, §6.6, §6.6a, §6.6b | `src/simple_qna_rag/errors.py`, `web/schemas.py`, `web/server.py`, `web/body_limit.py`, `net_budget.py`(`run_in_killable_subprocess`), `observability/request_id.py` | `tests/integration/test_error_schema.py`, `tests/integration/test_body_size_limit.py`(8 케이스, §6.6a m2-02 request-ID 공유/response-start 안전 처리 포함), `tests/integration/test_upstream_deadline.py`(§6.6b M2-02/M3-02 — remaining<=0 차단, connect/read/stream stall, budget 전파, `model_copy` 미반영 회귀 감시, router/answer 매 요청 재생성, DDGS subprocess 경계 bounded 반환) | Phase 3 report | PLANNED |
| M4-REQ-008 index lifecycle | §8 전체(§8.2 content_digest/version_id, §8.3a lock 범위, §8.7 M2-04/M3-04 trust boundary 포함) | `src/simple_qna_rag/index/manifest.py`, `index/lifecycle.py`, `cli/index_lifecycle.py` | `tests/integration/test_index_lifecycle_stress.py`, fault injection 표(§8.5, 9개 시나리오), `tests/integration/test_import_legacy_trust_boundary.py`(§8.7 M2-04/M3-04 — 승인 root/parent symlink·TOCTOU/최종 파일 symlink/owner·mode/hash 불일치 9개 시나리오, `FAISS.load_local` 0회 검증), `tests/unit/test_legacy_import_approved_hash_matches_baseline.py`(M3-04 승인 상수-baseline drift 감시) | Phase 5 report, `evidence/index.json` | PLANNED |
| M4-REQ-009 container/runbook | §9 전체(§9.0 test-only DI, §9.1 M2-05/M3-05 venv/COPY --exclude/명시적 production stage, §9.2 root `.dockerignore` `!README.md`, §9.4 layer 스캔/evidence write+upload) | `deploy/Dockerfile`(runtime/test/production 멀티스테이지), `.dockerignore`(저장소 루트), `deploy/docker-compose.yml`, `docs/operations/Runbook.md`, `src/simple_qna_rag/testing/`, `cli/web_testonly.py`, `scripts/scan_image_layers.py` | CI `container` job(§9.4 — UID/import-config/readiness smoke(trap cleanup+bounded poll)/layer scan/evidence write+upload 5단계, M2-05/M3-05 outer archive 해제+경로 canonicalization 후 layer별 스캔), `tests/unit/test_scan_image_layers.py`(M3-05 positive/negative control) | Phase 6 report, `evidence/container.json` | PLANNED |
| M4-REQ-010 gate/report/traceability | §10.1, §10.1a, §10.1b, §10.1c, 본 문서 | `evaluation/m4_fingerprint.py`, `evaluation/m4_evidence.py`(`_open_contained`, `write_result_json`, `assemble_candidate_evidence`), `evaluation/run_m4_gates.py`, `evaluation/run_static_regression_gate.py`, `evaluation/m4_gate.py`, `evaluation/run_pytest_gate.py`, `evaluation/run_compare_gate.py` | `evaluation/m4_gate.py` 자체 실행(evidence fail-closed 검증 포함), `tests/integration/test_run_m4_gates_self_test.py`(§10.1c M2-06/M3-06 — fake runner registry 기반 subprocess 없는 self-test, fresh/non-empty dir 강제, 삭제/변조/binding mismatch/container assemble/live-mode 7종) | Phase 7 report, `evaluation/baselines/m4_initial.json` | PLANNED |

## 2. 비기능 요구사항

| ID | Design 절 | 검증 방법(예정) | 상태 |
|---|---|---|---|
| M4-NFR-001 재현성 | §4.1, §10.1 | lock 2회 hash 동일성, fingerprint 재실행 비교 | PLANNED |
| M4-NFR-002 신뢰성 | §6.1a, §6.4(M2-01/M3-01), §6.6b(M2-02/M3-02), §7.5, §8.5 | shutdown drain 테스트, executor 단일 finalize race 테스트 12종, upstream deadline stall 테스트 + DDGS subprocess 경계 bounded 반환 테스트, fault injection 표 전체 통과 | PLANNED |
| M4-NFR-003 성능 | §7 전체 | `evaluation/m4_load.py` mock/live 결과 | PLANNED |
| M4-NFR-004 보안·프라이버시 | §4.3a(secret-safe hash), §5.6, §6.6a(body limit, m2-02), §8.1/§8.6/§8.7(pickle trust boundary, fail-closed, M2-04/M3-04 committed baseline hash + dir_fd openat/fstat), §9(M2-05/M3-05 layer 스캔 + 경로 canonicalization + positive/negative control) | redaction fixture, body/host 제한 테스트, `test_import_legacy_trust_boundary.py`, `test_legacy_import_approved_hash_matches_baseline.py`, 이미지 layer 바이트 scan, `test_scan_image_layers.py` | PLANNED |
| M4-NFR-005 호환성 | §12 체크리스트 | 기존 통합 테스트 회귀 + M3 14 gate | PLANNED |
| M4-NFR-006 유지보수성 | §2.2, §13-1 | 코드 리뷰(Codex) | PLANNED |
| M4-NFR-007 검증성 | §7.5, §9.4, §10.1a, §10.3 | mock-only CI + 별도 live/부하 job 분리 확인, evidence schema 일관성 | PLANNED |

## 3. §5.1 필수 자동 gate (Requirement.md) 매핑

Design.md §10.1b의 표를 그대로 참조 — `evaluation/m4_gate.py`가 유일한
판정 원본이며, 각 gate는 §10.1a의 공통 `evidence.json`(fingerprint/hash/
freshness/artifact 경로 containment fail-closed 검증 포함)을 거쳐야만
PASS/FAIL로 인정된다. `evaluation/run_m4_gates.py`(§10.1c, M2-06)가 `container`를
제외한 13개 gate를 fresh evidence 디렉터리에서 순서대로 실행해 이 evidence를
생성하는 단일 진입점이다. 이 표는 구현 시작 시 `evaluation/reports/m4/m4-final/`의
실측 `overall_pass` 값으로 갱신한다. 현재는 전부 `NOT_RUN`이다.

| gate | evidence 파일(예정) | 상태 |
|---|---|---|
| 정적 회귀 | `evidence/static_regression.json` | NOT_RUN |
| dependency | `evidence/dependency.json` | NOT_RUN |
| settings | `evidence/settings.json` | NOT_RUN |
| logging | `evidence/logging.json` | NOT_RUN |
| metrics | `evidence/metrics.json` | NOT_RUN |
| health | `evidence/health.json` | NOT_RUN |
| event loop | `evidence/event_loop.json` | NOT_RUN |
| bounded load | `evidence/bounded_load.json` | NOT_RUN |
| 정상 부하 | `evidence/normal_load.json` | NOT_RUN |
| live smoke | `evidence/live_smoke.json` | NOT_RUN |
| M3 품질 | `evidence/m3_quality.json` | NOT_RUN(재실행 필요, 최근 승인값은 `evaluation/baselines/m3_initial.md` 참조) |
| 단일 요청 성능 | `evidence/single_request_perf.json` | NOT_RUN |
| index | `evidence/index.json` | NOT_RUN |
| container | `evidence/container.json` | NOT_RUN |

## 4. Design Review Iteration 1 발견사항 대응 상태

Design.md §14의 매핑을 그대로 추적한다. 모든 항목은 문서 대응까지만
완료됐고(Design.md 반영), 코드 구현/테스트 실행은 각 Phase 진행 시
`PLANNED -> DONE`으로 갱신한다.

| 발견 | 심각도 | Design 대응 절 | 문서 반영 | 구현 검증 |
|---|---|---|---|---|
| M-01 executor FIFO/deadline/exactly-once | MAJOR | §6.4, §6.1a | DONE | PLANNED(Phase 3, `test_web_concurrency.py` — M2-01로 9종까지 확장, §5 참조) |
| M-02 legacy 자동 폴백 제거 | MAJOR | §8.1, §8.6, §8.5#7 | DONE | PLANNED(Phase 5) |
| M-03 version ID 순환/staging/lock | MAJOR | §8.2, §8.3, §8.3a, §8.4, §8.5#5,#8,#9 | DONE | PLANNED(Phase 5) |
| M-04 upstream 예산 파생 | MAJOR | §4.3(규칙7), §6.6b | DONE | PLANNED(Phase 3) |
| M-05 retrieval sub-stage/cardinality | MAJOR | §5.3, §5.4 | DONE | PLANNED(Phase 2) |
| M-06 dockerignore/entrypoint/DI | MAJOR | §9.0, §9.1, §9.2, §9.4 | DONE | PLANNED(Phase 6) |
| M-07 gate evidence schema | MAJOR | §10.1a, §10.1b, §10.3 | DONE | PLANNED(Phase 7) |
| m-01 shutdown drain | MINOR | §6.1a | DONE | PLANNED(Phase 3, `test_shutdown_drain.py`) |
| m-02 ASGI body wrapper | MINOR | §6.6a | DONE | PLANNED(Phase 3, `test_body_size_limit.py`) |
| m-03 settings hash 의미 | MINOR | §4.3a | DONE | PLANNED(Phase 1) |
| t-01 잔여 문구 | TRIVIAL | §6.4 | DONE(재작성으로 제거) | N/A |

## 5. Design Review Iteration 2 발견사항 대응 상태

Design.md §15의 매핑을 그대로 추적한다. 모든 항목은 문서 대응까지만
완료됐고(Design.md 반영), 코드 구현/테스트 실행은 각 Phase 진행 시
`PLANNED -> DONE`으로 갱신한다.

| 발견 | 심각도 | Design 대응 절 | 문서 반영 | 구현 검증 |
|---|---|---|---|---|
| M2-01 executor 단일 finalize/enum/동일 clock | MAJOR | §6.4, §6.7 | DONE | PLANNED(Phase 3, `test_web_concurrency.py` 9종) |
| M2-02 단일 DeadlineBudget/request-scoped LLM seam | MAJOR | §6.4, §6.6b, §13-5 | DONE | PLANNED(Phase 3, `test_upstream_deadline.py`) |
| M2-03 ObservationSink/event allowlist/trace 공유 | MAJOR | §5.1, §5.3, §5.6 | DONE | PLANNED(Phase 2, `test_observation_sink.py`) |
| M2-04 legacy import expected hash/경로 containment | MAJOR | §8.3, §8.7 | DONE | PLANNED(Phase 5, `test_import_legacy_trust_boundary.py`) |
| M2-05 clean Docker build/test 정상 packaging/layer 스캔 | MAJOR | §9.0, §9.1, §9.2, §9.4 | DONE | PLANNED(Phase 6) |
| M2-06 run_m4_gates.py 단일 DAG/CI attestation/self-test | MAJOR | §10.1a, §10.1b, §10.1c, §10.3 | DONE | PLANNED(Phase 7, `test_run_m4_gates_self_test.py`) |
| m2-01 begin_drain/shutdown_pool public API | MINOR | §6.1a, §6.4 | DONE | PLANNED(Phase 3, `test_shutdown_drain.py`) |
| m2-02 body limiter request ID/response-start 안전성 | MINOR | §5.2, §6.6a | DONE | PLANNED(Phase 3, `test_body_size_limit.py`) |
| m2-03 실제 Prometheus sample 예산 재계산 | MINOR | §5.4 | DONE | PLANNED(Phase 2, `test_metrics_cardinality.py`) |
| m2-04 typed settings 완전 인벤토리/os.environ 0건 gate | MINOR | §4.3, §4.3b | DONE | PLANNED(Phase 1, `test_no_direct_environ_access.py`) |

## 6. Design Review Iteration 3 발견사항 대응 상태

Design.md §16의 매핑을 그대로 추적한다. 모든 항목은 문서 대응까지만
완료됐고(Design.md 반영), 코드 구현/테스트 실행은 각 Phase 진행 시
`PLANNED -> DONE`으로 갱신한다. M3-02/M3-07은 lock된
`langchain-ollama==0.3.10`/`ollama==0.6.0`/`ddgs==9.14.4`/
`prometheus-client==0.26.0`(신규) API를 read-only executable spike로
직접 실행/확인해 반영했다.

| 발견 | 심각도 | Design 대응 절 | 문서 반영 | 구현 검증 |
|---|---|---|---|---|
| M3-01 executor ownership token/cancellation-free critical section | MAJOR | §6.1a, §6.4(핵심 재작성) | DONE | PLANNED(Phase 3, `test_web_concurrency.py` 12종) |
| M3-02 router/answer 매 요청 재생성, DDGS subprocess 경계 | MAJOR | §6.6b(전면 재작성), §13-5 | DONE | PLANNED(Phase 3, `test_upstream_deadline.py`) |
| M3-03 ObservationSink/RetrievalStageTrace 3필드 호환 | MAJOR | §5.1, §5.3(핵심 재작성), §5.6 | DONE | PLANNED(Phase 2, `test_observation_sink.py`) |
| M3-04 committed baseline 승인 hash, dir_fd openat/fstat trust boundary | MAJOR | §8.3, §8.4, §8.7(핵심 재작성) | DONE | PLANNED(Phase 5, `test_import_legacy_trust_boundary.py`, `test_legacy_import_approved_hash_matches_baseline.py`) |
| M3-05 명시적 production stage, requirements.txt COPY, CI evidence/scanner control | MAJOR | §9.1(핵심 재작성), §9.3, §9.4(핵심 재작성) | DONE | PLANNED(Phase 6, `test_scan_image_layers.py`) |
| M3-06 gate DAG 재귀 제거/fresh-container 분리/binding 검증 | MAJOR | §10.1a, §10.1b, §10.1c(핵심 재작성), §10.3(재작성) | DONE | PLANNED(Phase 7, `test_run_m4_gates_self_test.py` 7종) |
| M3-07 settings 완전 인벤토리, Prometheus non-env API | MAJOR | §4.3(전체 필드), §4.3b(재작성), §5.4 | DONE | PLANNED(Phase 1/2, `test_settings.py`, `test_metrics_cardinality.py`) |
| m3-01 Docker smoke trap cleanup/bounded poll | MINOR | §9.4 | DONE | PLANNED(Phase 6) |
| m3-02 소스 근거 module::symbol 표기 | MINOR | §6.6b, §8.7 | DONE | N/A(문서 표기 규칙) |

## 7. 다음 단계

Phase 0 착수 시 §3.1(fingerprint 재확인), §3.2(live 12 case 확정 — 본
Design.md §3.2에 이미 확정됨), §3.3(label allowlist 동결, §5.4 확장분
포함)을 실행하고 이 문서의 상태를 `PLANNED -> IN_PROGRESS`로 갱신한다.
Iteration 4 독립 리뷰가 §4/§5/§6의 모든 항목을 "문서 반영=DONE"으로
확인한 뒤에만 CRITICAL/MAJOR 0, 9.7/10 이상 기준으로 구현 Gate를 열 수
있다.
