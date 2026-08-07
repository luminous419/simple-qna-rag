# M3 요구사항 추적표

상태: **M3 완료 및 사용자 승인** — Phase 0~6 구현, Phase 4 블라인드 평가,
최종 통합 live 실행과 14개 품질 gate 승인을 완료했다.
기준일: 2026-08-07 (M3 구현 세션)

## 1. 기능 요구사항

| ID | 요구 요지 | 구현 위치 | 검증 | 상태 |
|---|---|---|---|---|
| M3-REQ-001 | 비교 가능한 실험 경계(candidate ID, SHA, 설정, fingerprint, evaluator version) | `evaluation/fingerprint.py`, `evaluation/reporting.py::build_candidate_metadata()`, candidate ID 정규식(§3.1) | `tests/unit/test_evaluation_fingerprint.py`, `tests/unit/test_config.py`(candidate ID 정규식) | ✅ 완료 |
| M3-REQ-002 | 안전한 MMR 벡터 재사용 | `src/simple_qna_rag/vector_index.py`(`StoredVectorIndex`), `rag_engine.py`(`_bump`/`_note`/`_candidate_vectors`/`_apply_mmr`) | `tests/unit/test_vector_index.py`(12), `tests/unit/test_rag_engine_trace.py`(7), `tests/integration/test_evaluation_retrieval.py`(폴백 6칸 행렬 포함 63) | ✅ 완료 + **live 42건 gate 통과**(아래 §3) |
| M3-REQ-003 | 검색 품질·성능 비교 | `evaluation/compare.py`(`compare_reports`, per-case 순위 diff) | `tests/integration/test_evaluation_compare.py` | ✅ 완료 |
| M3-REQ-004 | 라우팅 taxonomy와 precision-first 명시 우선순위 | `src/simple_qna_rag/routing_signals.py`, `agent.py`(`_decide_tool` 신호 우선 판정) | `tests/unit/test_routing_signals.py`(48, 골든 76건 exact set WEB8/DOC12/NONE56 포함), `tests/integration/test_agent_routing_policy.py`(27, 신호 stub 12칸 + S1~S12) | ✅ 완료 + **live 76×3 gate 통과**(아래 §3) |
| M3-REQ-005 | Routing 반복 평가 | `evaluation/routing.py::evaluate_routing_multi()`, `--runs` | `tests/integration/test_evaluation_routing.py`(TestEvaluateRoutingMulti 6건) | ✅ 완료 |
| M3-REQ-006 | Answer evaluator v2 | `evaluation/answer_rules.py`, `evaluation/answer_variants.json`, `evaluation/rescore.py`, `answers.py` v1/v2 병기 | `tests/unit/test_answer_rules.py`(56, 11개 FN 전수 재현 포함), `tests/integration/test_evaluation_rescore.py`(5), `tests/integration/test_evaluation_answers.py`(45, 무회귀) | ✅ 완료 — **11개 FN 전부 TP, 회귀 0**(`rescore` 실행으로 재현) |
| M3-REQ-007 | Intent 대조 실험과 결정 기록 | `evaluation/intent_ab.py`, `rag_engine.py` seam(`build_context`/`format_sources`/`generate_answer`), ADR | `tests/integration/test_intent_ab.py`(13), `tests/integration/test_rag_engine_seam.py`(8), live 29쌍 | ✅ 완료 — default 20 / intent 2 / tie 7, 사용자 승인 |
| M3-REQ-008 | 조건부 BM25 실험 | (진입 조건 미충족) | — | ⚪ **미진입**(`docs/milestones/m3-retrieval-domain-quality/Phase5_Non_Adoption.md`) — Requirement §3.2에 따라 M3 완료 결격 사유 아님 |
| M3-REQ-009 | 실패 안전성과 호환성 | 기존 `RAGEngine.query()`/`agent.route_query()`/`POST /rag` 응답 계약 무변경 | 기존 `tests/integration/test_agent.py`, `test_cli_entrypoints.py` 무수정 통과(639건 전체 회귀) | ✅ 완료 |
| M3-REQ-010 | 승인 baseline과 추적성 | `evaluation/baseline.py`(`--routing-runs`/`--warmup-cases`/`--candidate-id`/`gate_evaluation`) | `tests/integration/test_evaluation_baseline.py`, `evaluation/baselines/m3_initial.{json,md}` | ✅ 완료 — `m3-final-rerun` 14/14 gate 통과 및 사용자 승인 |

## 2. 비기능 요구사항

| ID | 요구 요지 | 구현 위치 | 검증 | 상태 |
|---|---|---|---|---|
| M3-NFR-001 | 재현성(고정 fixture, stable ordering, JSON=Markdown 동일 집계) | `check_markdown_links.py`의 stable sort, `routing_signals`/`answer_rules`의 결정론적 규칙, `intent_ab`의 seed 재현성 | `test_seed_reproducibility_of_slot_order`, `test_check_markdown_links.py::test_e6_deterministic_repeat_run` | ✅ 완료 |
| M3-NFR-002 | 성능 측정 건전성(동일 process warm-up) | `evaluation/retrieval.py`/`answers.py`의 `--warmup-cases`, `reporting.build_warmup_metadata()` | `TestEvaluateRetrievalWarmupAndMmrInstrumentation`(7), live 실행에서 `warmup.performed=true` 확인 | ✅ 완료 |
| M3-NFR-003 | 보안·프라이버시(상세 리포트 Git 제외, loopback 조건부 corpus hint) | `evaluation/reports/`는 기존 `.gitignore` 적용, `routing_signals.is_loopback_endpoint()` | `test_routing_signals.py`(loopback 8건) | ✅ 완료 |
| M3-NFR-004 | 유지보수성(evaluator에 제품 로직 복제 금지, fake 테스트 가능) | `vector_index.py`는 duck-typed(faiss 직접 import 없음), `intent_ab.py`는 `RAGEngine` seam만 사용 | 모든 신규 단위/통합 테스트가 fake/mock만 사용(live opt-in 제외) | ✅ 완료 |
| M3-NFR-005 | 회귀 방지(전체 테스트·dataset validation·Markdown link·git diff --check) | `scripts/check_markdown_links.py`(신규) | 아래 §4 정적 회귀 결과 | ✅ 완료 |

## 3. §4.1 gate 판정 (실측)

| gate | M2 기준 | M3 최소 | 실측(live) | 판정 |
|---|---:|---:|---:|:-:|
| Retrieval 평균 latency | 16,840ms | ≤8,420ms | **2,213.26ms** | ✅ |
| Retrieval p95 latency | 22,610ms | ≤13,570ms | **2,403.69ms** | ✅ |
| MMR 평균 latency | 14,349.31ms | ≤2,869.862ms | **8.38ms** | ✅ |
| Recall@10 | 97.62% | ≥95.24% | **97.62%**(M2와 완전 동일) | ✅ |
| Recall@5 | 95.24% | ≥92.86% | **95.24%** | ✅ |
| MRR@10 | 98.21% | ≥96.00% | **98.21%** | ✅ |
| nDCG@10 | 95.43% | ≥93.00% | **95.43%** | ✅ |
| Routing accuracy(중앙값, 분모 76) | 77.63%(59/76) | ≥90.79%(69/76) | **98.68%(75/76, values=[74,75,75])** | ✅ |
| Document route recall(중앙값, 분모 61) | 72.13%(44/61) | ≥88.52%(54/61) | **98.36%(60/61, values=[59,60,60])** | ✅ |
| Web search recall(각 run, 분모 15) | 100%(15/15) | ==100%(15/15) | **15/15, 15/15, 15/15**(재검증 후) | ✅ |
| Source any-hit / mean recall | 100% / 95.45% | ==100% / ≥93.00% | **100% / 95.45%** | ✅ |
| Answer E2E latency | 55.48s / 74.88s | ≤61.03s / ≤82.37s | **27.51s / 37.34s** | ✅ |

**Routing 재검증 경위**: 1차 76×3 live 실행에서 `web_search_recall`이 2개 run에서 14/15로 관측되어 gate를 충족하지 못했다. 원인은 `ws-006`("최근 발표된 삼성전자 실적을 검색해줘")처럼 명시 채널 단어가 없는(§7.2 NONE 신호) 질문에서 LLM이 3순위 판단을 일관되게 내리지 못한 것이었다 — 이 질문 자체는 결정론적 규칙의 대상이 아니라 설계상 LLM 위임 영역이므로 **결정론 규칙 자체는 정확히 동작했다**. `agent.py`의 `SYSTEM_PROMPT`(§7.3 "지금 이 순간의 값" 예시)가 "오늘의 날씨·현재 시세·실시간 지수·환율·경기 결과·속보"만 열거해 "기업의 최근 발표된 실적"류를 명시적으로 포함하지 않았던 것이 근본 원인으로 진단됐다(M2의 더 넓은 "최신 정보, 실시간 정보" 표현이 이 사례를 우연히 포괄했던 것과 대비된다). 예시를 "특정 기업의 최근 발표된 실적/뉴스"로 보강한 뒤 6/6 반복 호출로 일관성을 확인하고, 공식 76×3 재실행으로 세 gate 모두 통과를 확인했다(`evaluation/reports/m3/m3-p3a-signal-override/phase3_decision.md`). `ROUTING_SIGNAL_OVERRIDE` 기본값을 `True`로 채택했다.

## 4. 정적 회귀 (최신)

| 명령 | 결과 |
|---|---|
| `pytest -q` | 643 passed, 1 skipped |
| `npm test` | 9 passed(변경 없음) |
| `python -m evaluation.dataset validate evaluation/datasets/golden.jsonl` | 통과 |
| `python scripts/check_markdown_links.py` | 44 files, 103 links, 실패 0건 |
| `git diff --check` | 통과(공백 오류 없음) |

## 5. 최종 승인 근거

1. Phase 4 paired blind 평가: default 20, intent 2, tie 7, incomplete 0.
2. 사용자가 2026-08-08 비식별 worksheet 29건을 직접 검토해 기존 사례별
   채점을 승인했고, 결과와 `ANSWER_TEMPLATE_MODE=default` 채택을 승인했다.
3. Phase 6 통합 실행은 Retrieval 42건, Routing 76×3, Answer 29건을 모두
   성공했으며 `gate_evaluation.overall_pass=true`(14/14)를 기록했다.
4. 승인 값은 `evaluation/baselines/m3_initial.{json,md}`에 고정했다.

상세 live 원본은
`evaluation/reports/m3/m3-final-approved/baseline_20260807T155903164991Z.json`에서
식별한다. 이 디렉터리는 질문과 모델 답변을 포함할 수 있어 Git에서 제외한다.
