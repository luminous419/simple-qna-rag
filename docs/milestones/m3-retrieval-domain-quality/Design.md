# M3 Retrieval & Domain Quality 상세 설계

상태: **설계 초안 rev 3** — 리뷰·사용자 Gate 승인 전
기준일: 2026-08-05 (rev 2: 설계 리뷰 iteration 1 반영; rev 3: 2026-08-07 라우팅 단순화 사이클 1회차 — §7.2 재작성, §7.4·§11~§14 연동)
선행 문서: [M3 요구사항](Requirement.md), [M3 개발 계획](Plan.md)
비교 기준선: [M2 최초 기준선](../../../evaluation/baselines/m2_initial.md) / [기계 판독본](../../../evaluation/baselines/m2_initial.json)
구조 규칙: [Repository Structure](../../architecture/Repository_Structure.md)

이 문서는 Phase 0~6을 구현 가능한 수준까지 구체화한다. 요구사항의 비교·호환·실패 안전 계약은 변경하지 않으며, 계약을 만족하는 구현 경계(모듈, 함수 시그니처, schema, 실패 처리, 테스트, 명령)를 확정한다. **이 문서 작성 단계에서는 제품 코드를 구현하지 않는다.** §5.7과 §6.2의 "사전 검증 결과"는 저장소를 수정하지 않는 일회용 분석 스크립트(`/tmp` 실행)로 얻은 측정값이며, 그 스크립트는 산출물이 아니다.

---

## 1. 문서 사용법과 범위

| 절 | 내용 | 주 대응 요구사항 |
|---|---|---|
| §2 | 현재 컴포넌트·데이터 흐름과 변경 지점 | — |
| §3 | 공통 계약(candidate/report/version schema, 오류, 동시성, privacy) | M3-REQ-001, M3-NFR-001·003 |
| §4 | Phase 0 기준 고정(지문, warm-up 계약, link 검사기, 로그 artifact) | M3-REQ-001, M3-REQ-010, M3-NFR-002·005 |
| §5 | Phase 1 비교 하네스와 evaluator v2 | M3-REQ-006 |
| §6 | Phase 2 MMR 최적화 | M3-REQ-002, M3-REQ-003 |
| §7 | Phase 3 Routing 교정 | M3-REQ-004, M3-REQ-005 |
| §8 | Phase 4 Intent 대조 실험 | M3-REQ-007 |
| §9 | Phase 5 조건부 BM25 실험 | M3-REQ-008 |
| §10 | Phase 6 통합·승인 | M3-REQ-010 |
| §11 | 파일·모듈·API 단위 설계와 공개 계약 호환성 | M3-REQ-009 |
| §12 | 테스트 설계와 정확한 실행 명령 | M3-NFR-005 |
| §13 | Phase 순서·의존성·migration·rollback | — |
| §14 | 요구사항 추적표 | 전체 |
| §15 | 열린 쟁점과 사용자 결정 필요 항목 | — |

범위 밖(요구사항 §3.3)은 이 문서에서도 설계하지 않는다: 청킹/metadata schema 변경, 재색인, 웹 원문 수집, query rewriting, 모델 교체, ANN/vector DB, 운영 로그·dependency lock·vectorstore provenance.

---

## 2. 현재 시스템 구조와 데이터 흐름

### 2.1 컴포넌트 맵

| 계층 | 파일 | 책임 |
|---|---|---|
| Web/CLI 진입 | `src/simple_qna_rag/web/server.py`, `cli/query.py`, `cli/web.py`, `cli/index_documents.py` | HTTP `/rag`, 대화형 CLI, 색인 |
| 라우팅 | `src/simple_qna_rag/agent.py` | LLM tool calling으로 web_search/document_qa 선택 후 도구 직접 실행 |
| 라우팅 폴백 | `src/simple_qna_rag/query_router.py` | 키워드 기반 라우팅(Agent 실패 시) |
| 도구 | `src/simple_qna_rag/tools.py`, `web_search.py` | LangChain `Tool` 래퍼, DuckDuckGo 검색 |
| RAG 코어 | `src/simple_qna_rag/rag_engine.py` | vectorstore 로드, BM25/Dense/RRF/MMR/Reranker, 답변 생성 |
| 의도 분류 | `src/simple_qna_rag/intent_classifier.py`, `prompt_templates.py` | BGE-M3 임베딩 + Linear head, intent별 템플릿 |
| 설정 | `src/simple_qna_rag/config.py` | 상수와 runtime 경로 resolve |
| 평가 | `evaluation/{schema,dataset,metrics,reporting,retrieval,routing,answers,baseline}.py` | 골든셋, 지표, 리포트, 4개 evaluator |

### 2.2 질의 경로 호출 그래프 (현재)

```text
POST /rag  ─▶ agent.route_query(question)
                │  USE_WEB_SEARCH=False ─▶ rag_engine.query()                    (search_type=document_qa)
                │
                ├─ agent._decide_tool(question)                                   ① ChatOllama.bind_tools 1회
                │     예외 ─▶ query_router.route_query()                          ② 키워드 폴백
                │     (None,None) ─▶ query_router.route_query()                   ③ 도구 미선택 폴백
                │
                ├─ "web_search"  ─▶ web_search_tool.func(tool_query)
                │        success=False ─▶ rag_tool.func(원본 question)            ④ 웹 실패 폴백
                └─ "document_qa" ─▶ rag_tool.func(question)
                                        └─▶ RAGEngine.query(question)
                                              1. classify_intent()               (FileNotFoundError/Exception → "other")
                                              2. get_template_by_intent()
                                              3. _retrieve_documents()
                                              4. context = "\n\n".join(page_content)
                                              5. PromptTemplate | OllamaLLM | StrOutputParser
                                              6. sources[] = {index, source, page, content[:200]}
```

`_retrieve_documents()`의 현재 파이프라인(`USE_HYBRID_SEARCH=True`, `USE_MMR=True`, `USE_RERANKER=True`):

```text
bm25(50) ─┐
          ├─ RRF(k=60, top 50) ─▶ MMR(k=20, λ=0.5) ─▶ Reranker(top 10)
dense(50)─┘
   ▲ dense_retriever.invoke()가 내부에서 embed_query() 1회
                                   ▲ _apply_mmr()가 embed_query() 1회(질의) + 50회(후보 본문)
```

M2 기준선 단계별 평균: BM25 `0.52ms`, Dense `115.35ms`, RRF `0.04ms`, **MMR `14,349.31ms`**, Reranker `2,377.09ms`, 전체 `16.84초`. MMR이 전체의 85.2%다.

### 2.3 평가 subsystem 경로 (현재)

```text
evaluation.baseline.run_baseline()
   ├─ dataset.load_jsonl + validate_composition
   ├─ retrieval.evaluate_retrieval()  ─▶ RAGEngine._retrieve_documents(trace=RetrievalTrace())
   ├─ routing.evaluate_routing(cases, agent._decide_tool)      (live opt-in 필요)
   └─ answers.evaluate_answers()      ─▶ RAGEngine.query()
리포트: reporting.write_report() → <name>_<UTCts>.json/.md (배타 생성, 덮어쓰기 없음)
재현성: reporting.build_reproducibility_metadata(DATA_DIR, VECTORSTORE_PATH)
```

### 2.4 M3 변경 지점 요약

| # | 파일 | 변경 성격 | Phase | 공개 계약 영향 |
|---|---|---|---|---|
| 1 | `src/simple_qna_rag/vector_index.py` (신규) | FAISS row↔Document 매핑·검증·벡터 조회 | 2 | 없음(내부) |
| 2 | `src/simple_qna_rag/rag_engine.py` | query embedding 1회화, MMR 벡터원 교체, trace counters, `generate_answer()` seam 추출 | 2, 4 | 반환 dict 키 불변 |
| 3 | `src/simple_qna_rag/routing_signals.py` (신규) | 명시적 web/document 신호 판정 순수 함수 + corpus topic hint 생성 | 3 | 없음(내부) |
| 4 | `src/simple_qna_rag/agent.py` | SYSTEM_PROMPT 개정, `_decide_tool()` override 계층 | 3 | `route_query()` 반환 키 불변 |
| 5 | `src/simple_qna_rag/query_router.py` | 웹 검색어 추출 로직을 순수 함수로 추출(동작 동일) | 3 | `route_query()` 불변 |
| 6 | `src/simple_qna_rag/text_tokenizers.py` (신규) | BM25 tokenizer registry (기본 whitespace) | 5 | 없음(내부) |
| 7 | `src/simple_qna_rag/config.py` | M3 flag 추가(모두 기존 동작 유지 기본값) | 2,3,4,5 | 없음 |
| 8 | `evaluation/answer_rules.py` (신규) + `answer_variants.json` | evaluator v1/v2 순수 규칙 | 1 | 없음 |
| 9 | `evaluation/answers.py` | v1/v2 병기 채점, candidate block | 1 | 기존 JSON 키 유지(추가만) |
| 10 | `evaluation/retrieval.py` | candidate block, MMR 계측 요약, `--warmup-cases N` | 2 | 추가만 |
| 11 | `evaluation/routing.py` | `--runs N`, 다중 run 집계, router prompt fingerprint | 3 | 추가만 |
| 12 | `evaluation/answers.py`(warm-up) | `--warmup-cases N` | 4,6 | 추가만 |
| 13 | `evaluation/baseline.py` | candidate/`--routing-runs`/`--warmup-cases` 전달, gate 판정 block | 6 | 추가만 |
| 14 | `evaluation/reporting.py` | `_active_retrieval_config()` 확장, `build_candidate_metadata()`, `build_warmup_metadata()` | 1~6 | 추가만 |
| 15 | `evaluation/fingerprint.py` (신규) | Phase 0 지문 확인 CLI(기존 `reporting` 함수 래핑) | 0,6 | 신규 CLI |
| 16 | `evaluation/{rescore,compare}.py` (신규) | Phase 1/6 재채점·비교·gate 판정(gate는 `compare.py` 내부 순수 함수) | 1,6 | 신규 CLI |
| 17 | `evaluation/intent_ab.py` (신규) | Phase 4 paired blind 실험 | 4 | 신규 CLI |
| 18 | `evaluation/experiments/bm25_tokenizer.py` (신규) | Phase 5 오프라인 A/B | 5 | 신규 CLI |
| 19 | `scripts/check_markdown_links.py` (신규) | 표준 라이브러리 전용 Markdown 로컬 링크 검사 gate 도구 | 0 | 없음(회귀 도구) |

---

## 3. 공통 설계 계약

### 3.1 candidate ID와 리포트 디렉터리

candidate ID 문법은 정규식 **하나**로 확정한다(검증기와 리포트 디렉터리 naming이 같은 규칙을 쓴다).

```python
CANDIDATE_ID_RE = re.compile(r"^m3-(?:final|p[0-6][a-z]?(?:-[a-z0-9]+)+)$")
```

- 소문자와 숫자, 구분자 `-`만 쓴다. 세그먼트는 비어 있을 수 없다.
- `p<phase>` 다음의 **단일 소문자**는 그 Phase 안에서 평가한 후보의 순서(`a`가 첫 후보)를 뜻한다. "round"라는 별도 개념은 쓰지 않으며 `r2`/`r3` 같은 표기는 금지한다. 후보가 하나뿐인 Phase는 letter를 생략한다.
- `m3-final`만 예외적으로 slug 없이 쓴다.

| candidate ID | Phase | 내용 |
|---|---|---|
| `m3-p0-baseline-check` | 0 | 변경 없음, fingerprint 확인 |
| `m3-p1-evaluator-v2` | 1 | evaluator v2 규칙 |
| `m3-p2a-stored-vector` | 2 | FAISS 저장 벡터 재사용(1번째 후보) |
| `m3-p2b-embed-cache` | 2 | bounded embedding cache (예비, 2번째 후보) |
| `m3-p3a-signal-override` | 3 | 프롬프트 개정 + 명시 신호 우선 판정(1번째 후보) |
| `m3-p3b-corpus-hint` | 3 | a + corpus topic hint |
| `m3-p3c-two-stage` | 3 | b + 2단계 LLM 판정 (최후) |
| `m3-p4-intent-ab` | 4 | Intent paired blind 실험 |
| `m3-p5a-char2gram` | 5 | char 2-gram tokenizer 후보 |
| `m3-p5b-bge-subword` | 5 | BGE subword tokenizer 후보 |
| `m3-p5-bm25-offline` | 5 | tokenizer 오프라인 A/B(모든 tokenizer 동시 비교, 단일 실행) |
| `m3-final` | 6 | 채택 후보 통합 |

리포트 경로 규칙: `evaluation/reports/m3/<candidate_id>/<stage>/`
예) `evaluation/reports/m3/m3-p2a-stored-vector/retrieval/retrieval_<UTCts>.json`
Phase 3의 3회 실행은 단일 CLI 호출(`--runs 3`)로 한 디렉터리에 1개 리포트를 만든다.

`evaluation/reports/`는 이미 `.gitignore` 대상이므로 새 하위 디렉터리도 자동으로 Git에서 제외된다(§3.7 검증 명령 참고).

Plan §4의 예시 경로와의 대응(같은 실행을 가리키며, candidate ID를 경로에 넣어 후보 간 혼선을 없앤 것이 유일한 차이다):

| Plan 예시 경로 | 이 설계의 경로 |
|---|---|
| `evaluation/reports/m3/mmr` | `evaluation/reports/m3/m3-p2a-stored-vector/retrieval` |
| `evaluation/reports/m3/routing-run-1..3` | `evaluation/reports/m3/<routing candidate>/routing` (단일 `--runs 3` 리포트, run별 결과는 `per_run`에 보존) |
| `evaluation/reports/m3/intent` | `evaluation/reports/m3/m3-p4-intent-ab` |
| `evaluation/reports/m3/bm25` | `evaluation/reports/m3/m3-p5-bm25-offline/bm25_only` |
| `evaluation/reports/m3/final` | `evaluation/reports/m3/m3-final` |

warm-up은 별도 candidate ID를 만들지 않는다. 동일 process에서 실행되고 집계에서 제외되므로 후보 리포트의 `warmup` block으로만 표현된다(§4.4).


### 3.2 version 체계

| 축 | 값 | 규칙 |
|---|---|---|
| dataset schema | `evaluation/schema.py: SCHEMA_VERSION = "1.0.0"` | **변경 없음**. golden.jsonl 파일은 M3에서 byte 단위로 수정하지 않는다. |
| report schema | evaluator 모듈별 `SCHEMA_VERSION` → `1.0.0` → **`1.1.0`** | 필드 **추가만** 허용(minor). 기존 키 제거·의미 변경은 major이며 M3에서 금지. |
| assertion evaluator | `ASSERTION_EVALUATOR_VERSION` = `"v1"` 또는 `"v2"` | v1은 M2 규칙 그대로 동결. |
| abstention detector | `ABSTENTION_DETECTOR_VERSION` = `"v1"` 또는 `"v2"` | 동일. |
| 규칙 fingerprint | `rules_fingerprint` = SHA-256(canonical JSON of rule table) | 규칙이 1글자라도 바뀌면 값이 바뀐다. |

`rules_fingerprint` 계산 대상(canonical JSON, `sort_keys=True, ensure_ascii=False, separators=(",",":")`):

```json
{
  "assertion_version": "v2",
  "abstention_version": "v2",
  "normalization_steps": ["nfc","translate_lookalike","strip_markdown","casefold",
                          "strip_thousands","canonical_pp","canonical_pct",
                          "join_number_unit","split_ascii_separator","collapse_ws"],
  "abstention_scope_tokens": [...],
  "abstention_info_tokens": [...],
  "abstention_absence_tokens": [...],
  "abstention_literal_phrases": [...],
  "reviewed_variants_sha256": "<sha256 of evaluation/answer_variants.json bytes>"
}
```

### 3.3 candidate / report / version schema

모든 evaluator 리포트에 다음 block이 추가된다(값이 없으면 `null`, 기존 소비자 호환).

```jsonc
// build_candidate_metadata(candidate_id, label, baseline_ref, notes) 의 출력
"candidate": {
  "candidate_id": "m3-p2a-stored-vector",     // string | null
  "label": "MMR stored-vector reuse",          // string | null
  "phase": 2,                                  // int | null
  "baseline_ref": "evaluation/baselines/m2_initial.json",
  "baseline_dataset_sha256": "61b768…",        // 비교 기준 dataset 해시(문서 상수)
  "notes": null                                // string | null
}
```

latency를 측정하는 evaluator(`retrieval`, `answers`, `baseline`)에는 warm-up block이 추가된다(§4.4). 값은 CLI/API 인자와 실제 실행 결과로만 채우며 사람이 쓴 문자열로 대체할 수 없다.

```jsonc
// build_warmup_metadata(...) 의 출력
"warmup": {
  "requested_cases": 3,          // --warmup-cases 값 (0이면 warm-up 없음)
  "executed_cases": 3,           // 실제로 실행된 warm-up 사례 수
  "succeeded_cases": 3,
  "failed_cases": 0,
  "case_ids": ["<대상 목록의 앞 3건 id>"],   // 실제 값은 실행 시 대상 순서로 결정된다
  "same_process": true,          // 항상 true — 동일 process·동일 engine에서만 수행된다
  "engine_object_id_matches": true,  // warm-up과 측정이 같은 engine 인스턴스였는지
  "discarded_from_metrics": true, // 항상 true — warm-up 실행분은 집계에 들어가지 않는다
  "performed": true              // executed_cases > 0 and failed_cases == 0
},
"measured_case_count": 42        // 집계에 들어간 사례 수(retrieval 42 / answers 29 / 각 단계별)
```

`performed=false`이면 그 리포트로 latency gate를 판정할 수 없다(`gate_evaluation.items[].pass=null`, §3.5).

evaluator별 추가 필드:

```jsonc
// retrieval 리포트 (schema_version 1.1.0)
"retrieval_config": {                 // reporting._active_retrieval_config() 확장
  "...기존 키...": "...",
  "mmr_vector_source": "stored",      // "stored" | "embed"
  "bm25_tokenizer": "whitespace"
},
"mmr_instrumentation": {              // MMR 활성 시에만 non-null
  "query_embedding_calls_total": 42,          // == 성공 사례 수 (사례당 ≤1)
  "candidate_embedding_calls_total": 0,       // stored 경로에서 0이어야 함
  "vector_lookup_hits_total": 2100,
  "vector_lookup_misses_total": 0,
  "fallback_case_count": 0,
  "fallback_reasons": {}                      // reason -> count
},
"stage_summary": { "query_embed": {...}, "bm25": {...}, "dense": {...},
                   "rrf": {...}, "mmr": {...}, "reranker": {...}, "total": {...} }

// routing 리포트 (schema_version 1.1.0)
"run_count": 3,
"median_run_index": 1,                        // accuracy 기준 중앙 run(0-based, 동률은 최소 index)
"router_prompt_sha256": "…",                  // 최종 system prompt의 SHA-256
"routing_policy": {"signal_override": true, "corpus_topic_hint": false,
                   "signal_counts": {"web": 10, "document": 12, "none": 54},
                   "signal_conflict_count": 0,        // 웹 증거와 문서 증거가 함께 관측된 사례 수(§7.2.3)
                   "signal_suppressed_count": 0,      // 부정/인용으로 억제된 사례 수(§7.2.2)
                   "signal_error_count": 0},
"recall_denominators": {"document_qa": 61, "web_search": 15},  // expected_route 기준(§7.5)
"per_run": [ { "run_index":0, "accuracy":…, "correct_count":…, "total_cases":76,
               "document_route_recall":…, "document_route_correct":44,
               "web_search_recall":…, "web_search_correct":15, "failures":[…],
               "latency_ms":{…} }, … ],
"aggregate": {
  // 백분율 계열(표시용)
  "accuracy":             {"values":[…], "median":…, "min":…, "max":…},
  "document_route_recall":{"values":[…], "median":…, "min":…, "max":…},
  "web_search_recall":    {"values":[…], "median":…, "min":…, "max":…},
  // 분자 count 계열(gate 판정의 유일한 출처, §5.8)
  "correct_count":         {"values":[69,70,69], "median":69, "min":69, "max":70},
  "document_route_correct":{"values":[54,55,54], "median":54, "min":54, "max":55},
  "web_search_correct":    {"values":[15,15,15], "median":15, "min":15, "max":15}
},
"case_variation": [ {"id":"dq-…","expected_route":"document_qa",
                     "routes":["document_qa","web_search","document_qa"],
                     "distinct_count":2, "changed":true} ]
// runs==1이면 per_run/aggregate/case_variation은 길이 1 또는 null이고,
// 기존 최상위 키(accuracy, correct_count, failures, precision_recall_f1 …)는 그대로 유지된다.

// answers 리포트 (schema_version 1.1.0)
"evaluator_versions": {
  "assertion": "v1+v2", "abstention": "v1+v2", "rules_fingerprint": "…",
  "evaluator_profile": "v2",              // "v2" (공식) | "v2-no-variants" (실험 전용)
  "reviewed_variants_loaded": true,
  "reviewed_variants_sha256": "…",
  "official": true                        // profile=="v2" 이고 변형 표 검증 통과일 때만 true
},
"assertion":    { …M2와 동일한 v1 집계… },     // 의미 불변
"abstention":   { …M2와 동일한 v1 집계… },     // 의미 불변
"assertion_v2": { "cases_scored":…, "assertions_total":…, "assertions_passed":…,
                  "pass_rate":…, "fixed_vs_v1":[{"id":…, "index":…}],
                  "regressed_vs_v1":[] },
"abstention_v2":{ "true_positive":…, "true_negative":…, "false_positive":…,
                  "false_negative":…, "accuracy":…, "evaluated_count":…,
                  "fixed_vs_v1":[…], "regressed_vs_v1":[] },
"case_results[].assertion_passed_v2": 1,
"case_results[].predicted_abstention_v2": true,
"case_results[].abstention_match_v2": true

// baseline 리포트 (schema_version 1.1.0)
"gate_evaluation": {                          // evaluation/compare.py: evaluate_gates() 출력
  "spec_version": "m3-4.1",
  "overall_pass": false,
  "items": [ {"id":"retrieval_latency_mean", "metric":…, "threshold":…,
              "comparison":"<=", "pass":true, "source":"retrieval.latency_ms.mean",
              "note":"절대값과 감소율 모두 만족해야 함"} ]
}
```

**v1/v2 분리 원칙(요구사항 §4.2)**: 최상위 `assertion`/`abstention`은 항상 v1 의미를 유지하고, v2는 `*_v2` 별도 키에만 쓴다. Markdown 리포트도 "Assertion (v1)" / "Assertion (v2)" 두 섹션으로 분리 출력하며 한 표에 섞지 않는다. `evaluation/baselines/m2_initial.{json,md}`는 어떤 경우에도 수정하지 않는다(§13.3 검증).

### 3.4 config flag와 기본값

`src/simple_qna_rag/config.py`에 추가한다. **모든 기본값은 M2 동작을 그대로 재현한다.** 채택 결정 이후에만 기본값을 바꾸며, 각 flag는 rollback 스위치를 겸한다.

| 상수 | 기본값 | 채택 시 값 | 환경변수 override | Phase |
|---|---|---|---|---|
| `MMR_VECTOR_SOURCE` | `"embed"` | `"stored"` | `SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE` | 2 |
| `MMR_VECTOR_VALIDATION_SAMPLE` | `3` | 동일 | — | 2 |
| `MMR_VECTOR_COSINE_FLOOR` | `0.99` | 동일 | — | 2 |
| `MMR_EMBED_CACHE_MAX_ITEMS` | `2048` | 동일 | — | 2(예비) |
| `ROUTING_SIGNAL_OVERRIDE` | `False` | `True` | `SIMPLE_QNA_RAG_ROUTING_SIGNAL_OVERRIDE` | 3 |
| `ROUTING_CORPUS_TOPIC_HINT` | `False` | 결정 후 | `SIMPLE_QNA_RAG_ROUTING_CORPUS_TOPIC_HINT` | 3 |
| `ROUTING_CORPUS_TOPIC_HINT_MAX_ITEMS` | `25` | 동일 | — | 3 |
| `ANSWER_TEMPLATE_MODE` | `"intent"` | 결정 후 | `SIMPLE_QNA_RAG_ANSWER_TEMPLATE_MODE` | 4 |
| `INTENT_CONFIDENCE_FLOOR` | `0.0` (비활성) | 결정 후 | — | 4 |
| `BM25_TOKENIZER` | `"whitespace"` | 결정 후 | `SIMPLE_QNA_RAG_BM25_TOKENIZER` | 5 |

`ROUTING_CORPUS_TOPIC_HINT`는 **loopback endpoint 조건부**다. `True`여도 `OLLAMA_BASE_URL`의 host가 loopback이 아니면 hint를 생성하지 않는다(§7.3, M3-NFR-003). 현재 `config.OLLAMA_BASE_URL`은 `http://localhost:11434` 상수이므로 지금은 항상 활성 조건을 만족하지만, 향후 configurable endpoint가 생겨도 corpus 파일명이 외부로 나가지 않도록 코드 계약으로 고정한다.

환경변수 파싱 규칙: bool은 `{"1","true","yes","on"}`(casefold)만 True, 그 외 문자열은 False, 미설정은 기본값. 잘못된 열거값(예: `MMR_VECTOR_SOURCE=foo`)은 import 시 `ValueError`로 즉시 실패한다 — 조용히 기본값으로 되돌아가면 어떤 설정으로 측정했는지 리포트가 거짓이 된다. 모든 유효값은 리포트 `retrieval_config`/`routing_policy`에 기록된다.

### 3.5 error handling 정책

| 상황 | 처리 | 관찰 |
|---|---|---|
| StoredVectorIndex 구축/검증 실패 (init 시) | 경고 로그 + `mmr_vector_source="embed"`로 강등, 엔진 초기화는 성공 | `engine.mmr_vector_status = {"source":"embed","reason":…}`, 리포트 `mmr_instrumentation.fallback_reasons` |
| 질의 중 특정 후보 문서의 row 조회 실패 | 해당 질의 MMR 전체를 legacy embed 경로로 폴백(부분 혼용 금지) | `fallback_case_count += 1`, reason `lookup_miss` |
| 벡터 dimension 불일치 / non-finite | `VectorLookupError` → 위와 동일 폴백 | reason `dimension_mismatch` / `non_finite` |
| 공식 평가 실행에서 `fallback_case_count > 0` | latency gate 판정 **불가**로 표시(`gate_evaluation.items[].pass=null`) | 사용자에게 원인 보고 후 재실행 |
| warm-up 미요청(`--warmup-cases 0`) 또는 warm-up 사례 실패 | 실행은 계속하되 `warmup.performed=false` → latency gate 판정 **불가** | 리포트 `warmup` block |
| 명시 신호가 NONE인 질문에서 `_llm_decide_tool()` 예외 | 예외를 그대로 전파해 기존 `route_query()`의 `keyword_fallback_route()` 경로 유지(계약 불변) | routing 리포트 `exception` |
| 명시 신호가 WEB인 질문에서 `_llm_decide_tool()` 예외/no-tool | `("web_search", extract_web_search_query(question))`로 결정론적 보완 | routing 리포트 `signal_resolved_count.web` |
| 명시 신호가 DOCUMENT인 질문 | LLM을 호출하지 않고 `("document_qa", question)` | routing 리포트 `signal_resolved_count.document` |
| 신호 분류 함수 자체 예외 | 신호를 NONE으로 간주하고 기존 LLM→keyword fallback 계약 사용, 경고 로그 | routing 리포트 `routing_policy.signal_error_count` |
| corpus topic hint 생성 실패(디렉터리 없음/권한) | hint 없이 기본 프롬프트 사용 | `router_prompt_sha256`가 달라지므로 리포트에서 식별 가능 |
| corpus topic hint 활성인데 endpoint가 비-loopback | hint 자동 억제 | `routing_policy.corpus_topic_hint_suppressed_reason="non_loopback_endpoint"` |
| 공식 v2 실행에서 `answer_variants.json` 부재/schema 오류/fingerprint 불일치 | 사람이 읽을 오류 + exit 2 (fail-closed, §5.5) | — |
| Intent A/B에서 특정 사례 생성 실패 | 해당 사례를 `status="failure"`로 기록하고 계속, worksheet에 실패 표기 | `case_counts.failure` |
| BM25 tokenizer 미지원 값 | import 시 `ValueError`(§3.4) | — |
| rescore/compare CLI 입력 JSON schema 불일치 | 사람이 읽을 오류 + exit 2 | — |

기존 4개 폴백 경로(§2.2 ①②③④)는 **명시 신호가 NONE인 질문에서 동작·호출 순서·인자 모두 보존**하며 §12의 회귀 테스트로 고정한다. 명시 신호가 WEB/DOCUMENT인 질문에서는 요구사항 M3-REQ-004가 "모델 가용성과 무관하게 우선순위를 유지"하도록 요구하므로 `_decide_tool()`이 keyword fallback 대신 결정론적 route를 반환한다 — 이는 의도된 계약 변경이며 `route_query()`의 코드·구조는 그대로다(§7.4). 웹 검색 실패 시 원본 질문으로 document QA를 재시도하는 ④는 신호 종류와 무관하게 보존한다.

### 3.6 concurrency와 cache 경계

- `RAGEngine`은 프로세스 전역 싱글톤이고 FastAPI 동기 endpoint는 threadpool에서 실행되므로 `_retrieve_documents()`는 동시 호출될 수 있다.
- `StoredVectorIndex`는 **`initialize()` 안에서 1회 구축한 뒤 불변**이다. 이후 읽기 전용(`dict.get`, `faiss.Index.reconstruct`, numpy 읽기)만 수행하므로 lock이 필요 없다. 구축은 단일 스레드 구간에서만 일어난다. 지연 구축(lazy)은 채택하지 않는다 — 첫 동시 요청 두 개가 동시에 구축하는 경쟁을 만들기 때문이다.
- **query embedding 재사용은 호출 지역 변수로만 전달한다.** 요청 간 캐시를 두지 않으므로 질문 텍스트가 프로세스 메모리에 누적되지 않고 stale 위험도 없다.
- 예비 후보 B(bounded embedding cache)를 구현하게 되면 계약은 다음과 같다.
  - key: `(EMBEDDING_MODEL_NAME, NORMALIZE_EMBEDDINGS, sha256(page_content.encode("utf-8")))` — 내용 기반이라 재색인·문서 수정에 자동 무효화된다.
  - 자료구조: `collections.OrderedDict` LRU, 상한 `MMR_EMBED_CACHE_MAX_ITEMS`(기본 2048 ≈ 2048×1024×4B ≈ 8MB).
  - 동시성: 단일 `threading.Lock`으로 조회/삽입을 감싸고, 임베딩 계산 자체는 lock 밖에서 수행한다(중복 계산은 허용, 손상 상태는 불가).
  - 리포트에 `cache_hits/cache_misses/cache_evictions`를 기록한다.
- reranker 모델의 기존 지연 로딩(`self.reranker_model`)은 M3에서 건드리지 않는다(범위 밖, 기존 동작 유지).

### 3.7 privacy와 gitignore (M3-NFR-003)

- 질문·답변·context·검색 chunk가 포함되는 산출물은 **모두** `evaluation/reports/` 아래에만 만든다: 각 evaluator 리포트, Intent A/B worksheet, 그 blind key 파일, 고정된 context snapshot.
- Git 추적 대상이 되는 새 파일 중 답변 유래 문자열을 담는 것은 `evaluation/answer_variants.json` 하나이며, 여기에는 **골든셋 assertion과 동급의 짧은 표현 변형만** 넣는다(문장 이상, 개인정보, 문서 원문 인용 금지). 리뷰 항목이다.
- `.gitignore` 수정은 필요 없다. Phase 0에서 다음으로 확인한다.

```bash
git check-ignore -v evaluation/reports/m3/m3-p2a-stored-vector/retrieval/x.json
git check-ignore -v evaluation/reports/m3/m3-p4-intent-ab/worksheet_key.json
git status --porcelain evaluation/   # reports/ 하위가 나타나면 안 됨
```

- 외부 전송을 새로 추가하지 않는다. corpus topic hint(§7.3)는 **loopback endpoint의 로컬 Ollama 프롬프트에만** 들어간다. `is_loopback_endpoint(OLLAMA_BASE_URL)`이 False면 hint를 생성하지 않으며, 이 조건은 문구가 아니라 코드 계약이고 단위 테스트로 고정한다(§7.3, §12.1).

---

## 4. Phase 0 — 기준 상태와 실험 계약 고정

### 4.1 현재 fingerprint 확인 결과 (2026-08-05 **관측값**)

아래 표의 "현재 worktree" 열은 2026-08-05 시점의 **관측값**이며 설계가 고정하는 계약이 아니다. 계약은 "M2 승인 기준선 열의 4개 지문과 일치해야 한다"는 조건 자체다. 재현은 Phase 0의 `python -m evaluation.fingerprint`로 하며, 순간 수치(테스트 통과 수, 환경 버전)는 §4.6의 로그 artifact로 독립 재검증한다.

| 항목 | M2 승인 기준선 | 현재 worktree(관측값) | 일치 |
|---|---|---|---|
| dataset SHA-256 | `61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a` | 동일 | ✅ |
| corpus manifest SHA-256 | `5c0d648d032bc231c93cf1c545e5b3f8d26337da56677a4a0c1a66f24c82374a` (18 파일) | 동일 | ✅ |
| `index.faiss` SHA-256 | `c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820` | 동일 | ✅ |
| `index.pkl` SHA-256 | `3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00` | 동일 | ✅ |
| dataset 구성(category) | 76 / DQ 51 / WS 15 / BD 3 / UA 7, Retrieval 42, Answer 29 | 동일 | ✅ |
| dataset 구성(`expected_route`) | document_qa **61**(= DQ 51 + BD 3 + UA 7), web_search **15** | 동일 | ✅ (routing recall 분모, §7.5) |
| 정적 회귀 | — | `pytest -q` 358 passed 1 skipped, `npm test` 9 passed | ✅ |
| 기준선 파일 무결성 | — | `m2_initial.json` `e1edf23e7bc7c7584ee0e166de7ace211e02fc1d064fccbbac1d081a35faa6d5`, `m2_initial.md` `844e3c9cdc75ab244ca39fabe7693502e67a258892f25aea9d177f20025ad3f8` | 기준값으로 고정 |

Git 상태: HEAD `8771924426502b35b2f0a8b779d9f35519f02244`, dirty 파일은 사용자 소유의 `docs/Roadmap.md`, `docs/milestones/m3-retrieval-domain-quality/`, `m3_orchestration_guide.md`뿐이며 **M3 작업은 이를 수정하지 않는다**(Design.md 추가 제외).

환경: Python 3.11, `faiss` 1.10.0, FAISS index `IndexFlatIP`, `ntotal=389`, `d=1024`, 저장 벡터 L2 norm = 1.0.

**동일 조건 비교가 성립한다.** 이후 어떤 Phase에서든 위 4개 지문 중 하나라도 달라지면 공식 비교를 중단하고 "비교 불가"로 표시한 뒤 사용자 결정을 요청한다(Plan §6).

### 4.2 기존 실패 case ID 고정

**Routing 17건**(모두 `document_qa → web_search` mismatch, 원본: `evaluation/reports/m2_full/routing/routing_20260804T142931996640Z.json`) — 전체 목록과 taxonomy는 §7.1.

**Assertion false negative 8건**(6개 사례, 원본: `.../answers/answers_20260804T145621300637Z.json`):

| case_id | assertion index | 기대 any_of | 실제 답변 표면형 |
|---|---:|---|---|
| `dq-rag-001` | 0 | 정보 검색과 생성을 통합 / 검색과 생성을 통합 | “검색(Information Retrieval) + 생성(Generation)”의 **결합** |
| `dq-langgraph-reducer-001` | 0 | list에 메시지를 추가 / 메시지를 추가 | 메시지를 `left`에 순차적으로 **삽입** / 하나의 리스트에 **병합** |
| `dq-sparse-vs-dense-001` | 1 | 연속적인 벡터 / 고차원 벡터 | 연속 **고차원 임베딩 벡터** |
| `dq-econ-growth-revision-001` | 0 | 0.7% | `0.7 %` (숫자-단위 사이 공백) |
| `dq-econ-growth-revision-001` | 1 | 1.0%p / 1.0% 포인트 / 1.0%포인트 | `1.0 pp` (단위 별칭) |
| `dq-textsplit-steps-001` | 0 | 단위 크기 선정 / chunk size / 청크 크기 | `chunk_size` (underscore) |
| `dq-textsplit-steps-001` | 1 | 청크 오버랩 / chunk overlap | `chunk_overlap` (underscore) |
| `dq-realestate-procedure-001` | 0 | 구청 허가 | **구청의** 허가 승인 / 구청**에 거래** 허가 신청 |

**Abstention false negative 3건**: `ua-001`, `ua-002`, `ua-004`(모두 `expect_abstention=true`, 의미상 올바른 거절인데 v1이 공식 문구만 인식해 미탐).

이 11건은 Phase 1 fixture의 고정 입력이며, "M3에서 새로 생긴 실패"와 구분하는 기준선이다.

### 4.3 `evaluation/fingerprint.py` (신규, Phase 0/6 공용)

```python
def collect_fingerprint(data_dir: Path, vectorstore_path: Path, dataset_path: Path) -> dict:
    """{dataset_sha256, corpus_manifest_sha256, corpus_file_count,
        index_faiss_sha256, index_pkl_sha256, git_commit, git_dirty, python_version}
    기존 reporting.build_corpus_manifest()/build_vectorstore_fingerprint()만 사용하고
    모델/Ollama/FAISS 역직렬화를 하지 않는다."""

def compare_with_baseline(current: dict, baseline_json: Path) -> dict:
    """{"match": bool, "diff": [{"field","expected","actual"}], "baseline_id": …}
    baseline JSON의 reproducibility/dataset 블록에서 기대값을 읽는다."""
```

CLI: `python -m evaluation.fingerprint [--dataset …] [--baseline …] [--json]`
exit code: `0` 일치 / `3` 불일치(비교 불가 신호) / `2` 파일 없음. 모델을 전혀 로드하지 않으므로 CI에서도 실행 가능하다.

### 4.4 동일 process warm-up과 latency 측정 절차 (M3-NFR-002)

별도 프로세스 warm-up은 **채택하지 않는다.** embedding/reranker 모델 객체는 `RAGEngine` 인스턴스에 보관되고(reranker는 `_rerank_documents()`에서 lazy-load) 프로세스 종료와 함께 사라지므로, 별도 프로세스 실행은 후보 측정을 warm 상태로 만들지 못한다. 대신 evaluator 자체에 warm-up 단계를 넣는다.

#### CLI/API 계약

```text
python -m evaluation.retrieval --dataset … --output … --warmup-cases 3
python -m evaluation.answers   --dataset … --output … --warmup-cases 2
python -m evaluation.baseline  --dataset … --output … --warmup-cases 3   # 각 단계로 전달
```

```python
def evaluate_retrieval(*, dataset_path, output_dir, limit=None, tag=None,
                       warmup_cases: int = 0, candidate_id: str | None = None) -> dict: ...
```

- `--warmup-cases N`(기본 `0`, 음수는 argparse 오류): 평가 대상 목록의 **앞 N건**을 같은 `get_rag_engine()` 인스턴스로 먼저 1회 실행한다. 별도 프로세스도, 별도 engine도 만들지 않는다.
- warm-up 사례의 결과·latency·per-case 기록은 **전부 폐기**한다. 그 뒤 곧바로 같은 프로세스에서 공식 표본 전체(42건)를 처음부터 측정한다. 즉 warm-up 사례는 공식 표본에서 제외되는 것이 아니라 **먼저 한 번 더 실행되고 그 실행분만 버려진다** — 표본 구성이 42건에서 줄지 않는다.
- warm-up 중 예외는 실행을 중단시키지 않고 `warmup.failed_cases`로 집계한다. `warmup.performed`는 `executed_cases > 0 and failed_cases == 0`일 때만 True다.
- 결과는 §3.3의 `warmup` block으로 기록한다. `same_process`/`engine_object_id_matches`는 warm-up과 측정에 쓰인 engine의 `id()`를 비교해 채운다. `candidate.notes` 같은 자유 서술은 warm-up 증거로 쓰지 않는다.
- `warmup.performed=false`인 리포트는 latency gate를 `pass=null`(판정 불가)로 만든다(§3.5, §5.8).
- `--warmup-cases`는 `--limit`과 독립이며 함께 쓸 수 있다. 다만 `warmup_cases > 대상 사례 수`는 argparse 오류다.

#### 절차

1. 다른 무거운 프로세스를 종료하고 Ollama 실행을 확인한다. **모든 공식 latency 실행은 직렬**이며 병렬 실행 결과는 공식 비교에 쓰지 않는다(이 문장이 직렬 실행 규칙의 유일한 정의이고, §13.1은 Phase 의존성 관점에서 이 절을 참조한다).
2. 후보 설정으로 `--warmup-cases 3`을 붙여 evaluator를 **한 번만** 실행한다. warm-up과 42건 측정이 같은 프로세스·같은 engine에서 연속 수행된다.
3. `time.perf_counter()`는 그대로 사용한다(변경 없음).
4. 리포트의 `warmup` block과 `measured_case_count`를 확인한다.
5. 환경 변화가 의심되면 M2 commit과 후보 commit을 같은 세션에서 각 1회 재실행한 paired 비교를 우선하고, 재실행 기준값과 원래 승인 기준값을 함께 보고한다(요구사항 §4.1).

warm-up 3건의 근거: M2 단계별 수치상 첫 호출에서만 발생하는 비용은 embedding 모델 로드와 reranker lazy-load다. 두 경로를 모두 통과시키려면 reranker가 실제로 동작하는 사례가 1건 이상 필요하고, 여유를 둬 3건으로 정한다. Answer evaluator는 생성 LLM warm-up까지 필요하므로 2건으로 충분하다(생성 1회당 비용이 크다).

### 4.5 Markdown local link 검사기 (M3-NFR-005)

필수 회귀 gate가 재현 가능해야 하므로 검사기를 저장소 안에 만든다. 새 dependency는 추가하지 않는다.

**경로**: `scripts/check_markdown_links.py` (Python 3.11 표준 라이브러리만: `argparse`, `json`, `pathlib`, `re`, `subprocess`, `sys`, `unicodedata`, `urllib.parse`). 제품 패키지(`src/simple_qna_rag/`)에 넣지 않는다 — 런타임 코드가 아니라 회귀 도구다. `scripts/`는 이미 `sync-vendor.js`가 있는 도구 디렉터리다.

**대상 파일**: 기본은 **추적 파일과 미추적(non-ignored) 파일의 합집합**이다.

```bash
git ls-files -z --cached --others --exclude-standard -- '*.md' '*.markdown'
```

- `--cached`(추적)와 `--others --exclude-standard`(미추적 중 `.gitignore`에 걸리지 않은 것)를 **한 번의 호출로** 합쳐 열거한다. `--exclude-standard`가 `.gitignore`를 적용하므로 `node_modules/`, `venv/`, `build/`, `evaluation/reports/`는 그대로 제외된다.
- 결과는 **중복 제거 후 stable sort**(정규화된 repo-relative POSIX 경로의 코드포인트 오름차순)한다. index 이상(중복 entry, unmerged stage)으로 같은 경로가 두 번 나올 수 있으므로 중복 제거는 필수이며, 정렬은 출력·로그·리포트의 재현성을 위해 필수다(M3-NFR-001).
- `--cached`는 **worktree에서 삭제됐지만 index에 남은 경로**도 반환한다. 이런 경로는 `warning: <path>: 파일이 없어 건너뜀`을 출력하고 **건너뛴다**(exit 2 아님). staged deletion이 회귀 gate를 깨뜨리면 안 되기 때문이다. 반대로 존재하는데 읽지 못하는 경로(권한·디코딩 실패)는 exit 2다.
- **이 기본값이 tracked-only가 아닌 이유**: `git ls-files`만 쓰면 커밋 전 신규 Markdown이 검사 대상에서 빠져 gate가 fail-open 된다. 실제로 이 마일스톤 문서 디렉터리(`docs/milestones/m3-retrieval-domain-quality/`)는 작성 시점에 전부 미추적이었고, tracked-only 열거는 26개, 합집합 열거는 32개였다(2026-08-06 관측, 부록 A). 즉 **신규 산출물 6개가 통째로 검사되지 않는 상태**였다. M3-NFR-005는 모든 Phase gate에서 신규 문서의 깨진 링크를 잡아야 하므로 합집합이 유일한 정답이다.
- `--paths <path>…`로 부분 검사도 가능하고, `--no-git`은 지정 경로를 재귀 walk 한다(Git 없는 환경 대비). `--no-git` walk는 `.git/`, `node_modules/`, `venv/`, `.venv/`, `build/`, `dist/`, `__pycache__/`, `evaluation/reports/`를 하드코딩으로 건너뛴다(`.gitignore` 판정 없이 동작해야 하므로).

**검사 범위**:

| 대상 | 처리 |
|---|---|
| inline link `[text](target)`, image `![alt](target)` | 검사 |
| reference definition `[id]: target` | 검사 |
| angle-bracket 감싼 target `[t](<a b.md>)` | 검사(꺾쇠 제거 후) |
| fenced code block (``` 또는 ~~~, 여는 fence와 같은 문자·같은 길이 이상으로 닫힘) | **제외** |
| inline code span (`` ` `` 쌍 안) | **제외** |
| 외부 scheme(`http:`, `https:`, `mailto:`, `tel:`, `ftp:`, `data:`) 및 protocol-relative `//` | **제외**(네트워크 접근 없음) |
| HTML `<a href=…>` | **제외**(현재 문서에 없음. 발견 시 범위 확장을 별도 결정) |

**판정 규칙**:

1. `path` 또는 `path#anchor`: `path`를 링크가 있는 파일의 디렉터리 기준으로 resolve해 존재하는지 확인한다(디렉터리도 허용). URL-encoded `%20`은 디코드하고, `?query`는 잘라낸다. 존재하지 않으면 **실패**.
2. `#anchor`만 있는 링크: 같은 파일 안의 heading에서 anchor를 찾는다. 없으면 **실패**.
3. `path#anchor`에서 `path`가 Markdown 파일이면 그 파일의 heading에서 anchor를 찾는다. 없으면 **실패**. Markdown이 아닌 파일의 fragment는 검사하지 않는다(경고만).
4. anchor slug 규칙(GitHub 호환): heading 텍스트에서 Markdown 강조·링크·인라인 코드 표시를 제거 → NFC → casefold → 공백을 `-`로 → `[^\w\-]` 제거(유니코드 `\w` 허용이라 한글 heading 지원) → 같은 slug가 반복되면 `-1`, `-2` 접미사. 이 규칙은 완전한 GitHub 재현이 아니라 **저장소 내부 일관성 검사**이며, 그 한계를 `--help`와 이 절에 명시한다.
5. 저장소 밖으로 나가는 상대 경로(`..`로 repo root를 벗어남)는 **실패**로 본다.

**출력과 exit code**:

출력 예시(가상의 실패 2건):

```text
docs/example.md:7: broken link -> ../missing/Nowhere.md (파일 없음)
docs/example.md:120: broken anchor -> Requirement.md#존재하지-않는-절 (대상 파일에 anchor 없음)
검사 파일 32개(tracked 26 + untracked 6), 링크 214개, 실패 2개
```

- 마지막 요약 행은 항상 출력하며(실패 0건이어도) **`검사 파일 N개(tracked T + untracked U)` 형식으로 열거 출처를 분해**해 보여준다. 이 행이 Phase 0 로그의 fail-open 감시 지표다(§4.6).
- `--no-git` 모드에서는 분해 없이 `검사 파일 N개(walk)`로 출력한다.
- `--json` 플래그를 주면 `{"files": N, "tracked": T, "untracked": U, "links": L, "failures": [...]}`를 stdout에 출력한다. Phase 0 리포트가 사람이 읽는 요약 행을 파싱하지 않아도 되게 하려는 것이며, 집계값은 요약 행과 동일해야 한다(M3-NFR-001).

exit code:

- `0`: 실패 0건
- `1`: 깨진 링크 1건 이상(파일·줄·target·이유를 모두 출력)
- `2`: 사용법 오류, 존재하는 대상 파일 읽기 실패, `git ls-files` 실패

**테스트**: `tests/unit/test_check_markdown_links.py`. `scripts/`는 패키지가 아니므로 `importlib.util.spec_from_file_location()`으로 로드한다. `tmp_path`에 fixture Markdown을 만들어 (a) 정상 상대 링크, (b) 깨진 상대 링크, (c) 정상/깨진 anchor, (d) 코드블록·인라인 코드 안의 가짜 링크 무시, (e) 외부 URL 무시, (f) 중복 heading의 `-1` 접미사, (g) 한글 heading anchor, (h) repo 밖 경로 → 실패, (i) exit code 0/1/2를 검증한다. 이 묶음은 모델·네트워크·Git을 요구하지 않는다(`--no-git --paths`로 실행).

**열거 계약 테스트(필수, 리뷰 Iteration 2 M2)**: 위와 같은 파일에 `git`이 있을 때만 실행되는(`shutil.which("git")` 없으면 `pytest.mark.skipif`) 열거 전용 케이스를 둔다. `tmp_path`에 `git init` 한 임시 repo를 만들고 `git config user.email/user.name`을 설정한 뒤:

| 케이스 | fixture | 기대 |
|---|---|---|
| E1 | **커밋된** Markdown에 깨진 상대 링크 | exit 1 |
| E2 | **미추적(신규, `git add` 안 함)** Markdown에 깨진 상대 링크 | **exit 1** — tracked-only 열거라면 0이 되어 실패하는 회귀 방지 케이스 |
| E3 | `.gitignore`에 등록된 디렉터리 안의 깨진 Markdown | exit 0 (검사 대상 아님) |
| E4 | 정상 tracked 1개 + 정상 untracked 1개 | exit 0, `--json`의 `files==2`, `tracked==1`, `untracked==1` |
| E5 | tracked Markdown을 worktree에서 삭제(index 유지) | exit 0, warning 1행, 집계에서 제외 |
| E6 | 같은 내용으로 두 번 실행 | 파일 목록·요약 행·`--json` 출력이 완전히 동일(정렬·중복 제거 결정론) |

E2가 이 기본값 변경의 유일한 회귀 증거이므로 반드시 실제 `git init` repo에서 수행한다(`git ls-files` mock 금지).

**공통 명령**(Phase 0~6 동일):

```bash
python scripts/check_markdown_links.py
```

### 4.6 Phase 0 실행 로그 artifact

설계 본문의 순간 수치(테스트 통과 수 등)는 관측값이며, 재검증 가능한 원본은 Git 제외 경로에 남기고 설계·리포트에는 **경로와 SHA-256만** 기록한다.

| artifact | 경로(모두 `evaluation/reports/` 하위이므로 Git 제외) |
|---|---|
| dataset validation | `evaluation/reports/m3/m3-p0-baseline-check/logs/dataset_validate.log` |
| fingerprint JSON | `.../logs/fingerprint.json` |
| Python 회귀 | `.../logs/pytest.log` |
| frontend 회귀 | `.../logs/npm_test.log` |
| link 검사 (사람용) | `.../logs/markdown_links.log` |
| link 검사 (기계용) | `.../logs/markdown_links.json` |
| `git diff --check` | `.../logs/git_diff_check.log` |

Phase 0 리포트(`m3-p0-baseline-check`)에는 각 artifact의 **상대 경로, SHA-256, 캡처 UTC 시각, 종료 코드**를 표로 기록한다.

**link 검사 로그의 추가 계약(M3-NFR-005 fail-open 감시)**: `markdown_links.log`에는 §4.5의 요약 행이 그대로 남고, `markdown_links.json`에는 `files`/`tracked`/`untracked`/`links` 집계가 남는다. Phase 0 리포트에는 이 값들을 다음 형태로 **본문에 전사**한다.

| 항목 | Phase 0 기록 방식 | 2026-08-06 관측값(Phase 0에서 재측정) |
|---|---|---|
| 열거 명령 | 문자열 그대로 | `git ls-files -z --cached --others --exclude-standard -- '*.md' '*.markdown'` |
| tracked 수 | 정수 | 26 |
| untracked(non-ignored) 수 | 정수 | 6 |
| 검사 파일 수 | 정수, `tracked + untracked`와 일치해야 함 | 32 |
| 이번 마일스톤 신규 문서 포함 여부 | `docs/milestones/m3-retrieval-domain-quality/*.md` 각 파일이 검사 목록에 있는지 개별 확인 | 포함(commit 전 상태에서도) |

Phase 0 수용 기준은 **검사 파일 수 == tracked + untracked**이고 **M3 문서 디렉터리의 모든 Markdown이 목록에 있음**이다. 이 두 값이 리포트에 없으면 link 검사 gate는 통과로 인정하지 않는다. 설계 §4.1과 부록 A의 `358 passed, 1 skipped / 9 passed`는 2026-08-05 관측값이며, Phase 0에서 같은 절차를 재실행해 위 로그로 대체 기록한다. 사용자 소유 dirty 파일은 수정하지 않으므로 로그에 dirty 목록만 남기고 내용은 담지 않는다.

---

## 5. Phase 1 — 비교 하네스와 Answer evaluator v2

### 5.1 모듈 경계

```text
evaluation/answer_rules.py   (순수, 모델·파일 I/O 없음)
   normalize_text(text) -> str
   assertion_hit(answer_norm, phrase) -> bool
   assertion_coverage_v2(case_id, answer, assertions, variants) -> (passed, total, per_assertion)
   detect_abstention_v1(answer) -> bool      # M2 규칙 동결 사본
   detect_abstention_v2(answer) -> bool
   load_reviewed_variants(path=DEFAULT, *, required=True) -> VariantTable   # fail-closed
   rules_fingerprint(variants) -> str
   ASSERTION_EVALUATOR_VERSION / ABSTENTION_DETECTOR_VERSION

evaluation/answer_variants.json  (검토된 scoped 변형 표, Git 추적)
evaluation/rescore.py            (저장된 answers 리포트를 모델 없이 v1/v2 재채점)
evaluation/compare.py            (두 리포트/기준선 비교: 지표 delta, per-case 변화, fingerprint,
                                  그리고 요구사항 §4.1 gate 판정 순수 함수 M3_GATES/evaluate_gates())
```

**Phase 1 신규 모듈은 4개(+데이터 1개)로 제한한다.** 리뷰 권고(선행 코드 표면 축소)에 따라 gate 판정은 별 모듈을 만들지 않고 `compare.py` 안의 순수 함수로 둔다 — 판정 로직의 유일한 소비자가 `compare.py`와 `baseline.py`이고, 후자는 `from evaluation.compare import evaluate_gates`로 재사용한다. 재사용 지점이 셋 이상으로 늘어나면 그때 `evaluation/gates.py`로 추출한다(그 시점까지 공개 심볼 이름은 바뀌지 않는다). `evaluation/fingerprint.py`는 Phase 0 소관이며 기존 `reporting.build_corpus_manifest()`/`build_vectorstore_fingerprint()`를 감싸는 얇은 CLI 이상으로 키우지 않는다.

`evaluation/metrics.py::assertion_coverage()`(v1)는 **수정하지 않는다**. v2는 `answer_rules.py`에만 존재하며, `answers.py`가 두 함수를 각각 호출해 두 열을 만든다.

### 5.2 정규화 파이프라인 정확 명세

`normalize_text()`는 아래 순서를 **정확히** 따른다. answer와 assertion phrase 양쪽에 동일하게 적용한다(비대칭 처리 금지).

| # | 단계 | 규칙 | 근거 |
|---|---|---|---|
| 1 | `nfc` | `unicodedata.normalize("NFC", t)` | v1과 동일 |
| 2 | `translate_lookalike` | 전각 ASCII `U+FF01–U+FF5E → -0xFEE0`; `U+3000/00A0/2009/202F → " "`; 대시류 `U+2010–U+2015, U+2212 → "-"`; 인용부호 `U+2018/2019 → "'"`, `U+201C/201D → '"'` | 실제 답변에 `TF‑IDF`(U+2011), `검색‑생성`이 등장 |
| 3 | `strip_markdown` | 백틱(`` ` ``) 제거, `~` 제거, **길이 2 이상의 `*` 런만** 제거 | `**핵심 요약**`, `` `chunk_size` ``. 단일 `*`는 남겨 `2*3`의 의미를 보존 |
| 4 | `casefold` | `str.casefold()` | v1과 동일 |
| 5 | `strip_thousands` | `(?<=\d),(?=\d{3}\b)` → 삭제 | `1,000` vs `1000` |
| 6 | `canonical_pp` | 정규식 `(?:%\s*p\b` \| `%\s*포인트` \| `퍼센트\s*포인트` \| `\bpp\b` \| `%p)` (IGNORECASE) → `⟪pp⟫` | `1.0%p` ↔ `1.0 pp` |
| 7 | `canonical_pct` | 정규식 `(?:%` \| `퍼센트)` → `⟪pct⟫` (6 이후에 실행하므로 `%p`는 이미 소비됨) | `0.7%` ↔ `0.7 %` |
| 8 | `join_number_unit` | `(?<=\d)\s+(?=⟪)` → 삭제 | 숫자와 단위 사이 공백 흡수 |
| 9 | `split_ascii_separator` | 정규식 `(?<=[0-9A-Za-z])` + `[_\-]` + `(?=[0-9A-Za-z])` → `" "` | `chunk_size` ↔ `chunk size`. **앞뒤가 모두 ASCII 영숫자일 때만** 발동 |
| 10 | `collapse_ws` | `\s+` → `" "`, 양끝 `strip()` | 줄바꿈/표 정렬 공백 흡수 |

`⟪`/`⟫`(U+27EA/U+27EB)는 원문에 등장하지 않는 sentinel이며, 단계 2가 이 문자를 변형하지 않음을 단위 테스트로 고정한다.

### 5.3 안전 경계와 반례 (금지되는 정규화)

의미를 지우는 정규화는 **하지 않는다**. 아래는 모두 단위 테스트로 고정하는 반례다.

| # | 반례 입력 | 기대 | 이유 |
|---|---|---|---|
| C1 | phrase `1.0%` vs answer `1.0%p` | **불일치** | `⟪pct⟫` ≠ `⟪pp⟫`. 단계 6이 7보다 먼저라 `%p`가 절대 `%`로 붕괴하지 않는다 |
| C2 | phrase `0.7%` vs answer `10.7%` | **불일치** | 숫자 경계 lookaround(§5.4) |
| C3 | phrase `0.7%` vs answer `0.07%` | **불일치** | 문자열상 `0.7`이 `0.07`의 부분열이 아님 + 경계 규칙 |
| C4 | phrase `-1.0pp` vs answer `1.0pp` | **불일치** | 단계 9는 앞뒤가 모두 ASCII 영숫자일 때만 발동하므로 선행 부호 `-`를 절대 지우지 않는다(부정·부호 보존) |
| C5 | phrase `증가` vs answer `증가하지 않았다` | **일치(v1과 동일)** | v2는 부정 판정을 하지 않는다. 이는 v1에서도 동일한 알려진 한계이며 §5.9의 limitation note와 사람 검토로만 보완한다. v2가 이 한계를 **악화시키지 않음**을 테스트로 고정한다 |
| C6 | phrase `2*3` vs answer `23` | **불일치** | 단일 `*`를 지우지 않는다 |
| C7 | phrase `chunk size` vs answer `chunksize` | **불일치** | 구분자를 공백으로 바꿀 뿐 삭제하지 않는다 |
| C8 | phrase `비 공개` vs answer `비_공개` | **불일치** | 단계 9가 한글에는 발동하지 않는다(과도한 일반화 차단) |
| C9 | phrase `1000` vs answer `1,0000` | **불일치** | `(?=\d{3}\b)` 조건 때문에 `1,0000`의 쉼표는 제거되지 않는다 |
| C10 | 임의 phrase vs answer에 `⟪`/`⟫`가 원래 들어 있는 경우 | 양쪽 동일 처리 | sentinel 충돌 없음을 테스트로 확인 |

추가 금지 사항: 한국어 조사 제거, 어간 추출, 공백 전면 삭제, 동의어 사전의 전역 적용, LLM 기반 판정. 동의어는 §5.5의 **사례·assertion index로 범위가 고정된 검토 변형**으로만 추가한다(요구사항 §4.2).

### 5.4 assertion 매칭 규칙

```python
def assertion_hit(answer_norm: str, phrase_norm: str) -> bool:
    if not phrase_norm:
        return False
    pattern = re.escape(phrase_norm)
    if phrase_norm[0].isdigit():
        pattern = r"(?<![0-9.])" + pattern      # 10.7% 가 0.7% 로 오탐되지 않게
    if phrase_norm[-1].isdigit():
        pattern = pattern + r"(?![0-9.])"
    return re.search(pattern, answer_norm) is not None
```

숫자로 시작/끝나지 않는 phrase는 v1과 동일한 부분 문자열 의미를 유지한다. 이 lookaround는 v1 대비 **더 엄격**해질 수 있으므로, "기존 true positive 0 회귀"는 §5.7의 replay로 실증한다.

### 5.5 검토된 scoped 변형 표 (`evaluation/answer_variants.json`)

```jsonc
{
  "schema_version": "1.0.0",
  "review": { "reviewed_by": "<리뷰 담당>", "reviewed_at": "<승인일>",
              "evidence_report": "evaluation/reports/m2_full/answers/answers_20260804T145621300637Z.json" },
  "variants": [
    { "case_id": "dq-rag-001", "assertion_index": 0,
      "add_any_of": ["검색과 생성을 결합", "검색과 생성의 결합"],
      "rationale": "‘통합’과 ‘결합’은 이 문장에서 동일한 관계를 가리키며 숫자·부정·단위를 바꾸지 않는다." },
    { "case_id": "dq-langgraph-reducer-001", "assertion_index": 0,
      "add_any_of": ["메시지를 순차적으로 삽입", "메시지 리스트에 병합"],
      "rationale": "reducer 동작 서술에서 ‘추가/삽입/병합’은 동일 동작의 표면 변형이다." },
    { "case_id": "dq-sparse-vs-dense-001", "assertion_index": 1,
      "add_any_of": ["고차원 임베딩 벡터", "연속 고차원"],
      "rationale": "‘고차원 벡터’ 사이에 수식어 ‘임베딩’이 삽입된 형태로 의미가 같다." },
    { "case_id": "dq-realestate-procedure-001", "assertion_index": 0,
      "add_any_of": ["구청의 허가", "구청에 거래 허가"],
      "rationale": "조사 삽입만 다른 동일 표현이다." }
  ]
}
```

계약:
- **적용 범위는 `(case_id, assertion_index)`로 고정**된다. 전역 동의어 사전이 아니므로 다른 사례로 번지지 않는다.
- 각 항목은 `rationale`이 필수이며 리뷰 승인 없이는 추가하지 않는다.
- 추가 변형은 §5.2 정규화를 거친 뒤 `assertion_hit()`로 비교한다.
- 회귀 가드: 어떤 변형도 **다른 사례의 판정을 바꾸면 안 된다**(범위가 고정돼 구조적으로 불가능하지만, 전체 replay로 재확인).

#### fail-closed 정책 (요구사항 §4.2)

공식 v2 판정에서 변형 표는 **필수 fixture**다. 8개 assertion false negative 중 4개가 이 표에 의존하므로, 표가 없는 상태의 v2는 같은 이름의 다른 규칙이 된다. 따라서 fail-open을 금지한다.

| 상황 | 공식 profile `v2` | 실험 profile `v2-no-variants` |
|---|---|---|
| 파일 없음 | 오류 메시지 + **exit 2** | 계속(정규화만), `official=false` |
| JSON 파싱 실패, `schema_version` 미지원, 필수 키(`case_id`/`assertion_index`/`add_any_of`/`rationale`) 누락, `add_any_of` 빈 배열 | 오류 메시지 + **exit 2** | 동일하게 exit 2(깨진 파일은 어느 profile에서도 허용하지 않는다) |
| `--expect-variants-sha256 <hex>`와 파일 SHA-256 불일치 | 오류 메시지 + **exit 2** | 동일 |
| 정상 로드 | `official=true` | 해당 없음 |

- 기본 profile은 `v2`다. 변형 없는 실험은 `--evaluator-profile v2-no-variants`를 **명시적으로** 지정해야 하고, 그 리포트는 `evaluator_profile="v2-no-variants"`, `official=false`가 되어 M3 gate 판정에 쓸 수 없다(`compare.py`가 `official=false` 리포트의 v2 열 비교를 차단).
- `--expect-variants-sha256`은 Phase 1 승인 시점의 해시를 CI/재현 실행에 고정하기 위한 옵션이며, 미지정이면 해시 검사를 건너뛰고 실제 값을 `reviewed_variants_sha256`으로 기록한다. 공식 Phase 6 실행에서는 지정을 필수로 한다.
- `rules_fingerprint`에 `reviewed_variants_sha256`이 이미 포함되므로(§3.2) 표가 바뀌면 fingerprint가 바뀌고 O10 정책이 발동한다.
- `load_reviewed_variants(..., required=True)`가 위 오류를 `VariantTableError`로 올리고, CLI 계층이 이를 exit 2로 변환한다. 순수 함수 계층은 `sys.exit()`을 호출하지 않는다.

### 5.6 abstention detector v2

```python
def detect_abstention_v2(answer: str) -> bool:
    # L1: v1 공식 문구(정규화 후) — 절대 회귀하지 않도록 항상 먼저 검사
    n_full = normalize_text(answer)
    if any(normalize_text(p) in n_full for p in ABSTENTION_LITERAL_PHRASES):
        return True
    # L2: 표(table) 행 제외 — Markdown 표 셀의 "없음"은 항목 값이지 거절 선언이 아니다
    prose = "\n".join(l for l in answer.splitlines() if not l.strip().startswith("|"))
    # L3: 문장 단위로 (scope → info → absence) 순서 동시 출현
    for seg in re.split(r"[.!?\n]+", normalize_text(prose)):
        s = first_index(seg, SCOPE_TOKENS)      # 제공된 문서 / 문서 / 문맥 / 자료 / 문서 모음 / context
        i = first_index(seg, INFO_TOKENS)       # 정보 / 내용 / 언급 / 자료 / 데이터 / 근거 / 기재 / 설명 / 답변
        a = last_index(seg, ABSENCE_TOKENS)     # 찾을 수 없 / 없습니다 / 없음 / 존재하지 않 / 포함되어 있지 않 /
                                                # 포함되지 않 / 확인할 수 없 / 언급되지 않 / 나와 있지 않 /
                                                # 확인되지 않 / 제공되지 않 / 기재되어 있지 않 / 찾지 못
        if s is not None and i is not None and a is not None and s < i < a:
            return True
    return False
```

설계 근거와 반례:

| # | 입력 | 기대 | 이유 |
|---|---|---|---|
| A1 | `제공된 문서에서 관련 정보를 찾을 수 없습니다` | True | L1 (v1 호환) |
| A2 | `제공된 문서만으로는 확실한 답변이 어렵습니다` | True | L1 (yesno 템플릿 문구) |
| A3 | `문맥에 2025년 노벨 경제학상 수상자에 관한 언급이 존재하지 않습니다.` | True | L3 (`ua-002` 실제 답변) |
| A4 | `제공된 문서에서는 … 확인할 수 있는 정보가 없습니다.` | True | L3 (`ua-001`) |
| A5 | `\| **문서 길이 고려 여부** \| **없음** – 모든 단어가 … 문서를 전제로 함 \|` | **False** | L2 표 행 제외 + info 토큰 부재. 미적용 시 `dq-tfidf-vs-bm25-001`이 오탐(실측 확인) |
| A6 | `\| … \| 해당 없음 – 문서에는 다른 나라가 주최한 회담 언급이 없습니다. \|` | **False** | L2 표 행 제외. 미적용 시 `dq-apec-location-yn-001`이 오탐(2026-08-03 실행에서 실측) |
| A7 | `제공된 문서에 없는 내용은 추측하지 않습니다.` | **False** | `않습니다` 단독은 absence 토큰이 아니고 순서 조건도 불성립 |
| A8 | `관련 정보가 없는 경우에는 웹 검색이 필요합니다.` | **False** | scope 토큰 부재 |
| A9 | `2025년 성장률은 0.7%이며, 추가 통계는 문서에 정보가 없습니다.` | True(알려진 한계) | 부분 거절은 v2가 거절로 판정한다. §5.9 limitation과 worksheet 검토로 보완 |

**미채택 규칙**: 위치 기반 가중(앞/뒤 N 문장), 문장 길이 임계, 부정 극성 분석 — 실측에서 필요 없었고 근거 없는 복잡도를 늘리기 때문이다. 새 오탐이 관측되면 이 순서로 추가를 검토하고 근거를 기록한다.

### 5.7 사전 검증 결과 (설계 근거)

M2 승인 실행과 그 이전 두 실행의 저장된 답변에 위 규칙을 그대로 적용한 결과(모델 호출 없음, 저장소 미변경 스크립트):

| 리포트 | 사례 | assertion v1 | assertion v2 | abstention v1 acc | abstention v2 acc | v2 신규 오탐 | v1 TP 손실 |
|---|---:|---:|---:|---:|---:|---|---|
| `m2_full/answers_20260804T145621300637Z` (승인 기준선) | 29 | 24/32 | **32/32** | 0.8966 | **1.0000** | 0 | 0 |
| `answers/answers_20260803T163516543335Z` | 29 | — | — | 0.9655 | **1.0000** | 0 | 0 |
| `preflight/answers_20260804T140936998271Z` | 1 | — | — | 1.0000 | 1.0000 | 0 | 0 |

- 확인된 false negative 11건(assertion 8 + abstention 3)이 **모두** true positive로 바뀐다.
- 기존 true positive의 회귀는 **0건**이다.
- 8건 중 4건은 §5.2 정규화만으로(0.7 %, 1.0 pp, chunk_size, chunk_overlap), 4건은 §5.5 검토 변형으로 해결된다. 어떤 항목도 전역 동의어 규칙을 필요로 하지 않는다.
- 표 행 제외(L2)를 빼면 두 건의 오탐이 발생하므로 L2는 필수 규칙이다.

이 결과는 Phase 1 구현 후 `evaluation/rescore.py`로 **재현 가능한 산출물**로 다시 만든다.

### 5.8 rescore / compare / gates CLI

```text
python -m evaluation.rescore \
  --report evaluation/reports/m2_full/answers/answers_20260804T145621300637Z.json \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports/m3/m3-p1-evaluator-v2/rescore
```
- 저장된 리포트의 `case_results[].answer`만 읽어 v1/v2를 다시 채점한다. 모델·vectorstore·네트워크를 쓰지 않는다.
- 출력: 새 JSON/Markdown(원본 미수정) + `fixed_vs_v1`, `regressed_vs_v1`, `rules_fingerprint`.
- 원본 리포트에 `answer`가 없으면(예: 향후 마스킹된 리포트) 사람이 읽을 오류 + exit 2.

```text
python -m evaluation.compare \
  --baseline evaluation/baselines/m2_initial.json \
  --candidate evaluation/reports/m3/m3-p2a-stored-vector/retrieval/retrieval_<ts>.json \
  --kind retrieval \
  --output evaluation/reports/m3/m3-p2a-stored-vector/compare
```
- fingerprint 동일성 검사(불일치면 `comparable=false`, exit 3), 지표 delta, per-case 순위 변화(`ranked_source_ids` diff), floor 위반 사례 목록, §4.1 gate 표를 출력한다.
- `--kind {retrieval,routing,answers,baseline}`.

`evaluation/compare.py`의 gate 계층은 요구사항 §4.1을 코드로 고정한다. **count로 정의된 gate는 반올림 float 상수를 쓰지 않고 `Fraction` 또는 count 비교로 표현한다**(요구사항 §4.1 "표시 반올림이 아닌 원시 count/float 판정").

```python
from fractions import Fraction

M3_GATES = [
  Gate("retrieval_latency_mean_ms", "<=",  8420.0, also=Reduction(16840.0, 0.50)),
  Gate("retrieval_latency_p95_ms",  "<=", 13570.0, also=Reduction(22610.0, 0.40)),
  Gate("mmr_latency_mean_ms",       "<=",  2869.862, also=Reduction(14349.31, 0.80)),
  Gate("recall@10", ">=", 0.9524), Gate("recall@5", ">=", 0.9286),
  Gate("mrr@10",    ">=", 0.96),   Gate("ndcg@10",  ">=", 0.93),
  # Routing: 분모는 expected_route 기준(document 61, web 15).
  # CountGate는 float를 거치지 않고 분자 count만 비교한다 — 이것이 유일한 판정 출처다.
  CountGate("routing_accuracy_median",             ">=", 69, denominator=76),
  CountGate("routing_document_route_recall_median",">=", 54, denominator=61),
  CountGate("routing_web_recall_each_run",         "==", 15, denominator=15),
  Gate("source_any_hit", "==", 1.0), Gate("source_mean_recall", ">=", 0.93),
  Gate("answer_latency_mean_s", "<=", 61.03), Gate("answer_latency_p95_s", "<=", 82.37),
]
```

- `CountGate(name, op, required_correct, denominator)`는 리포트의 **분자 count**를 읽어 `Fraction(measured_correct, denominator)`와 `Fraction(required_correct, denominator)`를 비교한다(정수 비교와 동치). 백분율 float를 판정에 쓰지 않으므로 왕복 오차가 경계값을 바꿀 수 없다. 리포트에는 `{"correct": 54, "denominator": 61, "value": 0.885245…, "threshold_correct": 54}`를 함께 기록하고 `value`는 표시용이다.
- 분자 count가 리포트에 없는 구형 리포트(schema 1.0.0)는 `pass=null`(판정 불가)로 처리하고 백분율로 역산하지 않는다.
- `Reduction(base, ratio)`는 `value <= base * (1 - ratio)`를 뜻하며 절대 임계와 **둘 다** 만족해야 한다.
- `fallback_case_count > 0`(§3.5) 또는 `warmup.performed == false`(§4.4)이면 latency gate의 `pass`는 `null`(판정 불가)로 표시한다.
- `answers` 리포트의 `evaluator_versions.official == false`이면 v2 관련 gate/비교를 차단한다(§5.5).

count 기반 gate와 M2 기준값의 대응은 다음 한 표가 유일한 출처다.

| gate | 분모 | M2 | M3 최소 | 판정식 |
|---|---:|---|---|---|
| Routing accuracy | 76 (전체 사례) | 59/76 = 77.63% | **69/76 = 90.79%** | `Fraction(correct, 76) >= Fraction(69, 76)` |
| Document route recall | 61 (`expected_route == document_qa`) | 44/61 = 72.13% | **54/61 = 88.52%** | `Fraction(tp, 61) >= Fraction(54, 61)` |
| Web search recall (각 run) | 15 (`expected_route == web_search`) | 15/15 = 100% | **15/15 = 100%** | `tp == 15` |

category `document_qa` 51건은 이 표에 등장하지 않으며 어떤 gate의 분모도 아니다(요구사항 §4.1).

### 5.9 리포트 표기

- Markdown은 `## Assertion (v1 규칙)` / `## Assertion (v2 규칙)`, `## Abstention (v1 규칙)` / `## Abstention (v2 규칙)` 네 섹션으로 나눈다.
- v2 섹션에는 항상 다음 한계 문구를 넣는다: "v2는 표면 표현 차이를 줄여 false negative를 낮출 뿐, 의미 기반 correctness나 faithfulness를 보증하지 않는다. 부정 표현이 붙은 문장도 v1과 동일하게 일치로 판정될 수 있다."(M3-REQ-006)
- `rules_fingerprint`, `evaluator_profile`, `reviewed_variants_loaded`, `reviewed_variants_sha256`, `official`을 함께 출력한다. `official=false`인 리포트는 제목 줄에 "실험 profile — M3 gate 판정 불가"를 명시한다.

---

## 6. Phase 2 — MMR 병목 제거

### 6.1 병목 분석

`_apply_mmr()`는 질문마다 `embeddings_model.embed_query(doc.page_content)`를 후보 수(=`RRF_TOP_K`=50)만큼 호출한다. 후보 본문은 이미 색인 시 임베딩돼 FAISS에 저장돼 있으므로 전량 재계산이다. 또한 `dense_retriever.invoke()`가 질의를 1회, `_apply_mmr()`가 다시 1회 임베딩해 **질문당 query embedding이 2회**다(요구사항 M3-REQ-002는 ≤1회를 요구).

### 6.2 FAISS 매핑 실증 (2026-08-05 측정)

| 확인 항목 | 결과 |
|---|---|
| `index.pkl` 구조 | `(InMemoryDocstore, index_to_docstore_id)` 튜플 |
| `index_to_docstore_id` | `dict[int, str]`, 389개, key 0..388, 값 유일 |
| `set(i2d.values()) == set(docstore._dict)` | True |
| `index.ntotal == len(docstore._dict)` | 389 == 389 |
| index 타입/차원 | `IndexFlatIP`, `d=1024`, `metric_type=0`(inner product) |
| `index.reconstruct(i)` | 성공, 유한값, L2 norm ≈ 1.0 |
| `Document.id` | **None** — 현재 vectorstore는 `Document.id`를 채우지 않는다 |

핵심 결론 두 가지.

1. **`Document.id`에 의존하면 안 된다.** 매핑은 docstore 객체 동일성(identity)으로 만든다. `langchain_community` FAISS의 `similarity_search_with_score_by_vector()`는 `self.docstore.search(_id)`가 반환한 **저장된 객체 그 자체**를 돌려주고, BM25 retriever도 `docstore._dict.values()`를 그대로 쓴다. 따라서 RRF에 들어오는 모든 후보 `Document`는 docstore가 강한 참조로 보유한 바로 그 객체다(현재 `_reciprocal_rank_fusion()`의 `id(doc)` 중복 제거가 성립하는 것과 같은 근거).
2. **저장 벡터는 정규화된 상태로 그대로 재사용할 수 있다.** `NORMALIZE_EMBEDDINGS=True`로 임베딩 시점에 L2 정규화되어 저장됐고 실측 norm이 1.0이므로, 내적 = 코사인 유사도라는 `_apply_mmr()`의 기존 가정이 유지된다.

### 6.3 `src/simple_qna_rag/vector_index.py` (신규)

```python
class VectorLookupError(RuntimeError): ...
class VectorIndexValidationError(RuntimeError): ...

@dataclass(frozen=True)
class VectorIndexStats:
    document_count: int
    dimension: int
    validated_samples: int
    min_sample_cosine: float

class StoredVectorIndex:
    """FAISS row ↔ docstore Document 매핑을 1회 구축·검증하고, 이후 읽기 전용으로
    후보 Document의 저장 벡터를 돌려준다. faiss 모듈을 직접 import하지 않고
    vectorstore.index의 duck-typed 인터페이스(ntotal/d/reconstruct)만 사용하므로
    fake로 단위 테스트할 수 있다(M3-NFR-004)."""

    @classmethod
    def build(cls, vectorstore, *, sample_size: int, cosine_floor: float) -> "StoredVectorIndex": ...
    def row_for(self, document) -> int: ...            # 없으면 VectorLookupError
    def vectors_for(self, documents) -> "np.ndarray": ...  # shape (n, d), dtype float64
    @property
    def stats(self) -> VectorIndexStats: ...
```

`build()`의 검증 순서(모두 통과해야 `stored` 경로를 쓴다):

| 단계 | 검사 | 실패 시 |
|---|---|---|
| V1 | `index.ntotal == len(docstore._dict) == len(index_to_docstore_id)` | `VectorIndexValidationError("count_mismatch")` |
| V2 | `set(index_to_docstore_id.values()) == set(docstore._dict.keys())`, row 값 중복 없음 | `"key_mismatch"` |
| V3 | 모든 row가 `0 <= row < ntotal` 정수 | `"row_out_of_range"` |
| V4 | `index.d == len(embeddings.embed_query(_CANARY))` — 임베딩 1회로 차원 일치 확인 | `"dimension_mismatch"` |
| V5 | 결정론적 표본 `sample_size`개(문서 key 정렬 후 등간격 선택)에 대해 `cos(reconstruct(row), embed_query(page_content)) >= cosine_floor` | `"semantic_mismatch"` |
| V6 | 표본 벡터가 모두 유한(`np.isfinite`) | `"non_finite"` |

V5가 요구사항 §3.1의 "저장된 벡터와 문서의 대응 관계를 검증할 수 없으면 채택하지 않는다"에 대한 직접 증거다. 비용은 초기화 시 임베딩 `1 + sample_size`회(기본 4회)로, 질의당 50회를 없애는 대가로 1회성이다. `sample_size=0`은 허용하지 않는다(검증 없는 채택 금지).

`row_for()`의 매핑 자료구조는 `build()`에서 만든 `dict[int, int]`(`id(document) -> row`)이고, 동시에 `docstore._dict`에 대한 강한 참조를 인스턴스가 보유해 **객체가 살아 있는 동안 `id()` 재사용이 일어나지 않도록** 보장한다. 이 수명 계약을 docstring과 테스트에 명시한다.

### 6.4 `rag_engine.py` 변경 설계

```python
# 추가: RetrievalTrace 확장 (기존 필드 유지, 추가만)
@dataclass
class RetrievalTrace:
    stages: list[RetrievalStageTrace] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)      # 신규
    notes: list[str] = field(default_factory=list)              # 신규

# 신규: null-safe 계측 helper (module-level 순수 함수).
# 제품 경로 RAGEngine.query()는 trace를 전달하지 않으므로 모든 계측 접근은
# 반드시 이 두 함수를 통해서만 한다. trace.counters[...]를 직접 만지는 코드는 금지.
def _bump(trace: "RetrievalTrace | None", key: str, delta: int = 1) -> None:
    if trace is None:
        return
    trace.counters[key] = trace.counters.get(key, 0) + delta

def _note(trace: "RetrievalTrace | None", message: str) -> None:
    if trace is None:
        return
    trace.notes.append(message)

# 변경: 질의 임베딩을 한 번만 계산하고 dense/MMR이 공유
def _retrieve_documents(self, question, trace=None):
    if USE_HYBRID_SEARCH:
        query_vec = stage("query_embed", lambda: self.vectorstore.embeddings.embed_query(question))
        _bump(trace, "query_embedding_calls")          # null-safe (trace=None 허용)
        bm25_docs = stage("bm25",  lambda: self.bm25_retriever.invoke(question, top_k=BM25_TOP_K))
        dense_docs = stage("dense", lambda: self.vectorstore.similarity_search_by_vector(
                                                query_vec, k=DENSE_TOP_K))
        docs = stage("rrf", lambda: self._reciprocal_rank_fusion(bm25_docs, dense_docs,
                                                                 top_k=RRF_TOP_K, k=RRF_CONSTANT))
        if USE_MMR:
            docs = stage("mmr", lambda: self._apply_mmr(question, docs, top_k=MMR_K,
                                                        lambda_mult=MMR_LAMBDA,
                                                        query_embedding=query_vec, trace=trace))
        if USE_RERANKER:
            docs = stage("reranker", lambda: self._rerank_documents(question, docs, top_k=RERANKER_TOP_K))
    ...
```

동등성 근거: `as_retriever(search_type="similarity", search_kwargs={"k": DENSE_TOP_K}).invoke(q)`는 내부적으로 `embed_query(q)` → `similarity_search_with_score_by_vector(vec, k)`를 수행한다. FAISS 래퍼의 `_normalize_L2` 플래그는 `load_local()` 기본값 `False`이므로 벡터에 추가 변형이 없다. 따라서 위 치환은 **동일한 후보와 동일한 순서**를 낸다. 테스트에서 이 등가성을 fake vectorstore로 고정한다.

`_apply_mmr()` 변경(시그니처는 하위 호환 — 새 인자는 모두 키워드 기본값):

```python
def _apply_mmr(self, query, documents, top_k=20, lambda_mult=0.5,
               *, query_embedding=None, trace=None):
    if len(documents) <= top_k:
        return documents                      # 기존 조기 반환 유지
    q = np.asarray(query_embedding if query_embedding is not None
                   else self.vectorstore.embeddings.embed_query(query), dtype=np.float64)
    doc_matrix = self._candidate_vectors(documents, trace)     # (n, d) float64
    ... 기존 선택 루프와 동일한 수식·동일한 tie-break ...
```

`_candidate_vectors()` — **`trace=None`에서도 반드시 안전해야 한다.** 제품 `RAGEngine.query()`는 `_retrieve_documents()`를 trace 없이 호출하므로, 계측 접근을 직접 하면 폴백이 필요한 바로 그 순간에 `AttributeError`로 질의 전체가 실패한다(M3-REQ-002 위반). 따라서 모든 계측은 위 `_bump()`/`_note()`만 사용한다.

```python
def _candidate_vectors(self, documents, trace=None):
    index = self.stored_vector_index                  # 초기화 강등 시 None (mmr_vector_status)
    if MMR_VECTOR_SOURCE == "stored" and index is not None:
        try:
            vectors = index.vectors_for(documents)    # (n, d) float64
            _bump(trace, "vector_lookup_hits", len(documents))
            return vectors
        except VectorLookupError as exc:              # lookup_miss / dimension_mismatch / non_finite
            _bump(trace, "vector_lookup_misses", len(documents))
            _bump(trace, "mmr_fallback")
            _note(trace, f"fallback:{exc.reason}")
            self._log_mmr_fallback_once(exc.reason)   # 질의당 최대 1회 경고 로그
    # legacy 경로 (검증된 기존 코드와 동일)
    vectors = np.array([self.vectorstore.embeddings.embed_query(d.page_content)
                        for d in documents], dtype=np.float64)
    _bump(trace, "candidate_embedding_calls", len(documents))
    return vectors
```

- 강등 상태(초기화 시 `StoredVectorIndex.build()` 실패)는 trace와 무관한 인스턴스 속성 `self.mmr_vector_status = {"source": …, "reason": …}`로 유지하고, `stored_vector_index`는 `None`이 된다. 평가 리포트는 이 속성을 읽어 `mmr_instrumentation.fallback_reasons`에 반영하며, trace가 없어도 상태 자체는 조회 가능하다.
- 예외를 잡는 범위는 `VectorLookupError`(및 그 하위)로 한정한다. 그 밖의 예기치 않은 예외는 삼키지 않고 전파해 조용한 오답을 만들지 않는다.
- `_log_mmr_fallback_once()`는 질의 단위 경고 중복을 막는 헬퍼이며 trace와 독립이다.

수치 동등성: 저장 벡터는 float32, 기존 경로는 Python float 리스트(float64). `dtype=np.float64`로 승격해 비교하지만 마지막 비트 차이로 **완전 동률**이 갈릴 수 있다. 설계 결정: (a) 순위 동일성은 "고정 fixture에서 완전 일치"로 단위 테스트하고, (b) 실제 42건에서는 `compare.py`의 per-case 순위 diff로 변화 사례를 명시 보고한다. 순위 변화가 있으면 그 자체가 gate 위반이 아니라 **보고 대상**이며, 품질 floor(§4.1)로 판정한다.

### 6.5 실패·폴백·계측 계약 (M3-REQ-002)

| 계약 | 구현 | 검증 |
|---|---|---|
| query embedding ≤ 1회/질문 | `_retrieve_documents()`가 1회 계산해 전달 | 카운팅 fake embeddings로 `embed_query` 호출 수 == 1 assert |
| 후보 본문 재임베딩 0회 | `stored` 경로는 `embed_query`를 호출하지 않음 | 같은 fake로 후보 수만큼의 추가 호출이 **없음**을 assert |
| 대응 안정성·검증 가능성 | §6.3 V1~V6 | 각 실패 모드별 단위 테스트 |
| 실패 시 조용한 오답 금지 | 예외 → 질의 단위 전체 legacy 폴백(부분 혼용 금지) | 폴백 시 결과가 legacy와 **완전히 동일**함을 테스트 |
| cache 사용 시 key/상한/동시성 | §3.6 (후보 B 채택 시에만) | LRU 상한·동시 접근 테스트 |
| 계측이 순서를 바꾸지 않음 | 모든 counter/note 접근이 `_bump()`/`_note()` 경유(null-safe), 반환값 경로 동일 | trace 유무 두 번 호출해 결과 리스트 동일성 assert |
| **계측 비활성 경로 안전성** | `trace=None`에서도 폴백이 예외 없이 완료 | 아래 6칸 테스트 행렬 전부 |

**폴백 테스트 행렬(6칸 모두 필수)** — 각 칸에서 (a) 예외가 발생하지 않고, (b) 반환 문서 리스트와 순서가 legacy 경로와 **완전히 동일**함을 assert한다.

| 실패 모드 | `trace=None` (제품 경로) | `trace=RetrievalTrace()` (평가 경로) |
|---|---|---|
| `lookup_miss` (`row_for()` 미등록 문서) | 필수 | 필수 + `counters["mmr_fallback"]==1`, `notes==["fallback:lookup_miss"]` |
| `dimension_mismatch` | 필수 | 필수 + reason 기록 |
| `non_finite` | 필수 | 필수 + reason 기록 |

`trace=None` 칸은 `RAGEngine.query()`를 통해서도 한 번 더 검증한다(`query()`가 `_retrieve_documents()`를 trace 없이 호출하는 현재 계약을 그대로 쓰는 회귀 테스트). 초기화 강등(`build()` 실패) 경로도 같은 방식으로 `trace=None`에서 안전함을 확인한다.

계측 값은 `RetrievalTrace.counters`에 다음 key로 쌓고 `evaluation/retrieval.py`가 `mmr_instrumentation`으로 집계한다: `query_embedding_calls`, `candidate_embedding_calls`, `vector_lookup_hits`, `vector_lookup_misses`, `mmr_fallback`.

### 6.6 예비 후보 B — bounded embedding cache

후보 A(§6.3)가 V1~V6를 통과하지 못하거나 §4.1 gate를 못 넘길 때만 구현한다. 계약은 §3.6에 정의했다. A가 통과하면 **구현하지 않으며**(Plan §4 Phase 2-6 "최소 복잡도 후보만 승격"), 이 절은 설계 근거로만 남긴다.

### 6.7 예상 효과와 gate 판정

| 단계 | M2 평균 | 예상(후보 A) | 근거 |
|---|---:|---:|---|
| query_embed | (dense에 포함) | ~115ms | 기존 dense 단계 대부분이 질의 임베딩 |
| bm25 | 0.52ms | 0.52ms | 변화 없음 |
| dense | 115.35ms | ~1ms | FAISS 검색만 남음 |
| rrf | 0.04ms | 0.04ms | 변화 없음 |
| **mmr** | **14,349.31ms** | **~5–20ms** | 임베딩 0회 + reconstruct 50회(메모리 복사) + 기존 파이썬 선택 루프 |
| reranker | 2,377.09ms | 2,377.09ms | 변화 없음 |
| **total** | **16,840ms** | **≈2,500ms** | gate 8,420ms 대비 충분한 여유 |

MMR gate(≤2,869.86ms)와 total mean/p95 gate를 모두 큰 폭으로 만족할 것으로 예상되며, 실제 판정은 warm-up 후 42건 단독 실행 리포트로 한다. 미달 시 후보 A를 승격하지 않고 현행 경로를 유지한다(Plan §4 Phase 2 수용 기준).

---

## 7. Phase 3 — 문서 우선 Routing 교정

### 7.1 M2 17건 오류 taxonomy

태그: **L**=명시적 로컬 문서 신호, **Y**=연도/최신 표현, **P**=정책·시장·기업 주제, **A**=복합/모호 표현. (중복 태깅)

| # | case_id | 질문 요지 | L | Y | P | A | 판정 이유 |
|---:|---|---|:-:|:-:|:-:|:-:|---|
| 1 | `dq-tongyi-version-001` | 통이치엔원 2.0 vs 1.0 향상점 | | | ● | | 기업·제품명이 최신 정보로 보임. 실제 근거는 SPRI AI Brief |
| 2 | `dq-econ-growth-revision-001` | 2025 성장률 전망 수정치 비교 | | ● | ● | | 연도+경제 지표 → 실시간 통계로 오인 |
| 3 | `dq-realestate-procedure-001` | 2025 10·15 대책 거래 절차 | | ● | ● | | 연도+정책 |
| 4 | `dq-apec-location-yn-001` | 2025 APEC 한국 개최 여부 | | ● | ● | ● | 연도+시사 이슈+yes/no 형식 |
| 5 | `dq-chain-lcel-001` | 체인 생성 시 여러 단계 결합 문법 | | | | ● | 일반 기술 질문인데 문서 신호가 약함 |
| 6 | `dq-agent-vs-model-001` | Google AI 에이전트 **백서에서** 설명하는 차이 | ● | | ● | | 명시적 문서 신호가 있는데도 웹으로 감 |
| 7 | `dq-g7-hiroshima-001` | G7 히로시마 AI 프로세스 합의 | | | ● | | 국제 정책 주제 |
| 8 | `dq-samsung-gauss-001` | 삼성 가우스 모델 구성 | | | ● | | 기업 제품 |
| 9 | `dq-cohere-provenance-001` | 코히어 데이터 출처 탐색기 | | | ● | | 기업 제품 |
| 10 | `dq-kb-price-outlook-001` | **2025 KB 부동산 보고서에서** 매매가 전망 | ● | ● | ● | | 명시적 문서 신호 + 연도 + 시장 |
| 11 | `dq-kb-gangnam-001` | **2025 KB 부동산 보고서에서** 강남구 이슈 | ● | ● | ● | | 동일 |
| 12 | `dq-econ-growth-rate-001` | 2025 성장률 수정치 | | ● | ● | | 연도+경제 |
| 13 | `dq-econ-consumption-001` | 2025 민간소비증가율 | | ● | ● | | 연도+경제 |
| 14 | `dq-realestate-15oct-001` | 2025 10·15 대책 정책 방향 | | ● | ● | | 연도+정책 |
| 15 | `dq-realestate-ltv-001` | 10·15 대책 LTV 한도 | | ● | ● | | 정책+수치 |
| 16 | `dq-apec-theme-001` | 2025 APEC 공식 주제 | | ● | ● | | 연도+시사 |
| 17 | `bd-002` | "관련된 자료가 있으면 알려줘" | ◐ | | | ● | 대상이 생략된 모호 질문. "자료"는 약한 로컬 신호 |

집계: L 4건(+약한 신호 1), Y 11건, P 15건, A 3건. **17건 모두 로컬 corpus가 실제로 답을 가진 주제**다(§7.3 corpus 목록 대조).

정답 쪽 보호 대상: web_search 15건 중 13건은 채널 토큰(웹/인터넷/온라인/웹검색)을 명시하고, `rr-ws-seoul-temp-001`("지금 서울 기온이 몇 도야?")와 `rr-ws-exchange-rate-001`("오늘 환율이 어떻게 돼?") 2건은 **채널 명시 없이 실시간성만으로** 웹이어야 한다. 이 2건이 정책 변경의 최대 위험이며 프롬프트에 실시간 수치 예외를 반드시 유지한다.

### 7.2 `routing_signals.py` — 라우팅 단순화 사이클 1의 규범 설계

이 절이 M3-REQ-004의 유일한 구현 계약이다. 아래 §7.2-L은 이전 Iteration 1~6의 감사 기록이며 구현 입력으로 사용하지 않는다.

```python
class ExplicitSignal(str, Enum):
    WEB = "web"
    DOCUMENT = "document"
    NONE = "none"

def classify_explicit_signal(question: str) -> ExplicitSignal:
    """WEB_COMMAND → DOCUMENT_SCOPE → NONE 순서의 순수 판정."""
```

판정은 다음 세 단계뿐이다.

1. 인용 span을 마스킹하고, 웹 금지 cue가 있는 문장은 WEB 후보에서 제외한다.
2. 아래 두 WEB command grammar 중 하나가 성립하면 WEB이다.
3. 그렇지 않고 명시적 DOCUMENT scope 토큰이 있으면 DOCUMENT, 나머지는 NONE/LLM이다.

WEB command grammar:

- **직접 검색 명령**: `WEB_FUSED + 명령형`이 인접한다. 예: `웹검색해줘`, `구글링해줘`, `구글링해서 알려줘`.
- **채널 지정 검색 명령**: `CHANNEL + (에서|으로|로)`가 있고 문장 마지막 술어가 `검색|찾|조회|확인|알아보` 계열 명령형이다.

일반 응답 술어 `알려|답해|보여`는 두 번째 grammar의 검색 행위 술어가 아니다. 따라서 `웹검색으로 알려줘`, `웹검색에서 사용하는 API 구조 알려줘`, `구글에서 사용하는 검색 기술 알려줘`는 NONE이다. 명시 표현을 놓친 경우도 NONE으로 보내 LLM이 판단하며 결정론 규칙의 recall을 늘리기 위한 예외를 추가하지 않는다.

`SOURCE_PARTICLE`, `WEB_FUSED`, 최신성 표현은 단독 충분조건이 아니다. `TOPIC_HEAD`, 조사 뒤 관형절 cue, 어절 거리 상수, `has_particle` fast path는 존재하지 않는다. 왼쪽 Unicode 경계, 인용 마스킹, 부정 억제, DOCUMENT 토큰과 WEB 우선순위는 기존 외부 계약을 보존한다.

골든 76건의 결정론적 기대 집합은 다음과 같이 고정한다.

- WEB 8: `ws-000`, `ws-002`, `ws-005`, `ws-007`, `ws-009`, `rr-ws-python-version-001`, `rr-ws-bitcoin-price-001`, `rr-ws-samsung-stock-001`
- DOCUMENT 12: `bd-000`, `ua-000`~`ua-006`, `dq-agent-arch-001`, `dq-agent-vs-model-001`, `dq-kb-price-outlook-001`, `dq-kb-gangnam-001`
- NONE 56: 나머지 전부. `ws-003`, `ws-008`은 `웹검색으로 + 알려줘`이므로 LLM 위임이다.

WEB/DOCUMENT 판정의 expected route 오탐은 각각 0이어야 하며 세 집합은 exact equality로 검사한다. 별도 boundary/property table은 최소 다음을 포함한다.

| 입력 | 기대 | 경계 |
|---|---|---|
| `오늘 날씨를 웹에서 검색해줘` | WEB | 채널+조사+검색 술어 |
| `구글링해서 알려줘` | WEB | 직접 검색 명령 |
| `웹검색으로 최신 환율 알려줘` | NONE | 일반 응답 술어 → LLM |
| `웹검색에서 사용하는 API 구조 알려줘` | NONE | 관형절 우회 불가 |
| `구글에서 사용하는 검색 기술 알려줘` | NONE | 관형절 우회 불가 |
| `웹검색 방법 알려줘` | NONE | 검색 주제 질문 |
| `"웹에서 검색해줘"라는 문구를 설명해줘` | NONE | 인용 마스킹 |
| `웹 검색은 하지 말고 문서로 답해줘` | DOCUMENT | 금지 후 문서 범위 |
| `질문:웹에서 검색해줘` | WEB | Unicode 왼쪽 경계 |
| `freewebsearch 사용법 알려줘` | NONE | 복합어 내부 배제 |

구현 복잡도 gate: command grammar 정규식/판정 함수는 두 개를 넘지 않고, 허용 술어 집합 하나와 금지/인용 전처리만 둔다. 새 예외 목록이나 거리 상수가 필요하면 이 설계를 구현하지 않고 다음 재설계 iteration으로 돌린다.

### 7.2-L 이전 Iteration 1~6 설계 — 비규범 감사 기록

아래 §7.2-L.1~§7.2-L.4는 이전 STOP까지의 판단 근거를 보존한 것이다. 코드·테스트·추적표는 위 §7.2만 따라야 한다.

```python
class ExplicitSignal(str, Enum):
    WEB = "web"; DOCUMENT = "document"; NONE = "none"

def classify_explicit_signal(question: str) -> ExplicitSignal:
    """요구사항 §5 M3-REQ-004의 1·2순위를 결정론적으로 구현한 순수 함수.
    3순위(taxonomy·실제 최신성)는 LLM이 담당하며 이 함수는 NONE을 반환한다.
    LLM·네트워크·파일 I/O를 쓰지 않으므로 LLM 장애와 무관하게 항상 판정 가능하다."""

def is_loopback_endpoint(base_url: str) -> bool:
    """urllib.parse로 host를 뽑아 loopback인지 판정한다(순수 함수).
    허용: "localhost", "127.0.0.0/8" 내 IPv4, "::1". 그 외(빈 host, 파싱 실패 포함)는 False.
    ipaddress 표준 라이브러리만 사용한다."""
```

#### 7.2-L.1 어휘 집합 (폐기됨)

정규화 `norm(q)`: NFC → casefold → 연속 공백 1칸으로 축약 → 양끝 trim. `REQUEST`/`DOCUMENT`/`PROHIBITION`/`TOPIC_HEAD`는 `norm(q)` 위의 **부분 문자열 포함**이며, 한국어 조사 변형을 흡수하기 위해 어간 형태로만 적는다(예: `찾`은 `찾아줘`/`찾아봐`/`찾아주세요`를 모두 덮는다). `CHANNEL`은 아래처럼 **공백 분리 어절(語節) 전체 일치**로 매칭한다(리뷰 Iteration 3 M1 대응 — 부분 문자열 포함은 `websocket`/`webhook`/`googleapis`/`구글맵` 같은 복합어 내부까지 채널로 오인한다). `WEB_FUSED`는 순수 부분 문자열이 아니라 **왼쪽 경계가 있는** 부분 문자열/정규식이다.

**왼쪽 경계(Unicode-aware token boundary)**: `WEB_FUSED`의 왼쪽 경계는 정규식 lookbehind `(?<!\w)`(매치 직전 위치가 "단어 문자"가 아님)로 정의한다. Python `re`는 기본이 유니코드 모드이므로 `\w`는 ASCII 알파벳·숫자·밑줄뿐 아니라 한글 음절(예: `가`~`힣`)도 포함한다. 따라서 `(?<!\w)`는 문자열 시작·공백 뒤는 물론 쉼표·콜론·괄호·따옴표 같은 **정상 문장부호 뒤**도 왼쪽 경계로 인정하지만(그 문자들은 `\w`가 아니므로), 영숫자·한글이 곧바로 이어지는 **복합어 내부**는 경계로 인정하지 않는다(그 앞 문자가 `\w`이므로). 예: `질문:웹검색으로 최신 환율 알려줘`의 `:` 뒤, `(구글링해서 알려줘)`의 `(` 뒤는 경계로 인정되어 매치가 성립하고, `freewebsearch`의 `web`은 직전 문자 `e`가 `\w`라 매치가 성립하지 않는다(리뷰 Iteration 5 M1 대응 — 기존 "문장 시작 또는 공백"만 인정하던 정의가 문장부호 뒤 강한 명령을 과소탐지하던 결함을 구조적으로 제거한다). 오른쪽은 조사가 바로 이어질 수 있으므로 제약하지 않는다.

| 집합 | 원소 | 매칭 방식 | 의미 |
|---|---|---|---|
| `CHANNEL` | `웹`, `인터넷`, `온라인`, `구글`, `포털`, `검색엔진`, `web`, `google` | **어절 전체 일치**: 어절(끝의 문장부호 제거 후)이 `CHANNEL` 원소 하나와 정확히 같거나, 그 원소 + `SOURCE_PARTICLE` 하나로 끝에서 다른 문자 없이 완전히 소진돼야 한다(정규식 `^(웹|인터넷|온라인|구글|포털|검색엔진|web|google)(에서|으로|로)?$`, `web`/`google`은 casefold 후 비교) | **채널 토큰**. 정보 출처가 외부 웹임을 가리키는 명사이되, 다른 조사(`의`/`이`/`가`/`은`/`는` 등)가 붙거나 뒤에 다른 글자가 이어 붙은 복합어의 일부인 경우는 매치가 아니다 |
| `SOURCE_PARTICLE` | `에서`, `으로`, `로` | `CHANNEL` 어절 또는 `WEB_FUSED` 매치에 **직접 결합**(개입 문자 없음) | **출처·수단 조사**. `CHANNEL`·`WEB_FUSED` 어느 쪽에 결합하든 동일하게 "그 채널·융합 표현을 검색의 출처/수단으로 삼는다"는 구문적 근거이며, 이 근거가 성립하면(`has_particle=True`) 뒤따르는 목적어 절의 내용을 더 이상 스캔하지 않고 곧바로 command 증거로 인정한다(아래 "주제절 억제 규칙"). `의`/`이`/`가`/`은`/`는` 등 다른 조사는 불인정. 이 결합은 whole-token/왼쪽 경계 조건을 만족하는 **한 형태**일 뿐이며, `REQUEST_TAIL`을 면제하는 **독립 충분조건이 아니다**(리뷰 Iteration 4 M2) — `인터넷에서 최신 소식 부탁해`처럼 조사 결합이 있어도 마지막 어절이 요청 표현이 아니면(`REQUEST_TAIL` 거짓) WEB 증거로 인정하지 않는다(§7.2.4 음성 fixture로 고정) |
| `REQUEST` | `검색`, `서치`, `찾`, `알아봐`, `알아보`, `확인`, `조회`, `보여`, `알려`, `답해`, `답변`, `가져와`, `물어봐` | 부분 문자열 포함 | **요청 표현**. 사용자가 행위를 지시하는 술어 계열 |
| `WEB_FUSED` | 정규식 `(?<!\w)(웹|인터넷|온라인|구글|web)\s?(검색|서치|search)` 또는 `(?<!\w)구글링` | **Unicode-aware 왼쪽 경계**(위 문단)의 부분 문자열/정규식. 매치 직후 문자가 `SOURCE_PARTICLE`(`에서`/`으로`/`로`) 중 하나로 시작하면 그 조사까지 매치에 포함해 `has_particle=True`로 기록한다(그 외에는 `False`). 오른쪽은 조사가 없어도 매치가 성립한다(`구글링해서`처럼 조사 없이 요청 동사가 곧바로 이어져도 매치) | **융합형 웹 검색 요청 후보**. 채널과 검색이 한 어구로 붙은 신호 후보일 뿐이며, `CHANNEL`과 **동일한 단일 command-intent 판정**(아래 "주제절 억제 규칙")을 통과해야 최종 WEB 증거로 인정된다 — 융합형이라는 사실 자체가 별도의 즉시-WEB 우회 규칙이 되지 않는다(리뷰 Iteration 4 M1·Iteration 5 M1 대응) |
| `DOCUMENT` | `이 문서`, `그 문서`, `제공된 문서`, `제공된 자료`, `등록된 문서`, `문서에서`, `문서에는`, `문서로`, `문서만`, `문서들`, `문서 모음`, `문서 내`, `문서 기반`, `첨부`, `로컬 문서`, `로컬 자료`, `보고서에서`, `백서에서`, `백서에 따르면`, `자료에서`, `본문에서` | 부분 문자열 포함 | **문서 범위 토큰** |
| `PROHIBITION` | `하지 말`, `하지 마`, `말고`, `없이`, `대신`, `금지`, `아니라`, `아닌`, `제외하고`, `빼고` | 부분 문자열 포함 | **부정·금지 cue** |
| `TOPIC_HEAD` | `기능`, `방식`, `원리`, `구현`, `도구`, `모듈`, `코드`, `프롬프트`, `설정`, `옵션`, `파이프라인`, `방법`, `기술`, `구조` | 부분 문자열 포함, **아래 "주제절 억제 규칙"이 정하는 전체 절 범위**에서 스캔(고정 어절 거리 없음) | **주제어 head 명사**(웹 검색 자체를 설명·서술 대상으로 삼는 표현). `방법`·`기술`·`구조`는 리뷰 Iteration 4 M1 대응으로 추가했다 — `웹검색 방법`·`웹 검색 기술`·`web search API 구조`처럼 검색 기술·기능 자체를 묻는 표현을 포착한다. 스캔 범위가 고정 거리가 아니므로 `웹검색 관련 API 구조 알려줘`처럼 관형 표현이 끼어들어 `TOPIC_HEAD`가 여러 어절 떨어진 경우도 포착한다(리뷰 Iteration 5 M1 대응) |
| `QUOTE_SPAN` | 정규식 `"[^"]*"`, `'[^']*'`, `` `[^`]*` ``, `「[^」]*」`, `“[^”]*”` | 정규식 | **인용·인라인 코드 구간** |

`REQUEST`에 `설명`을 넣지 않는다. `설명해줘`는 문서 설명 요청에서 지배적으로 쓰여(`rr-dq-embedding-explain-001` 등) 채널 토픽 문장까지 WEB으로 끌어올리기 때문이다. 이 배제는 §7.2.4의 음성 fixture로 고정한다.

`REQUEST_TAIL`: 질문을 어절로 나눈 뒤 **마지막 어절**(끝의 문장부호 제거 후)이 `REQUEST` 원소를 부분 문자열로 포함하는지 여부. 한국어 요청문은 대개 서술어가 문장 맨 끝에 오므로(`…해줘`, `…알려줘`, `…보여줘`), 이는 "요청 표현이 실제 명령형 서술어로 쓰였는지"를 근사한다. `REQUEST` 어간이 문장 중간에서 관형형 어미(`-는`/`-은`/`-ㄴ`, 예: `찾아가는`)로 뒤 명사를 수식하거나, 복합 명사의 일부(예: `확인 방법`의 `확인`)로만 쓰이면 `REQUEST_TAIL`은 거짓이 되어 순위 3 증거에서 제외된다.

#### 주제절 억제 규칙 — 수단/행위 명령 vs. 검색 주제 명사구의 경계 (`CHANNEL`·`WEB_FUSED` 공통, 고정 어절 거리 없음)

리뷰 Iteration 5 M1은 이전 설계의 `CHANNEL_REQUEST_MAX_WORD_GAP`/`WEB_FUSED_TOPIC_GAP`(둘 다 고정값 2)이 "주제어가 채널·융합 표현으로부터 몇 어절 떨어져 있는가"라는 **위치**만으로 명령/주제를 가르다가, 주제어를 두 어절만 더 밀어 넣으면(`m+3`) 같은 의미의 주제 질문이 다시 WEB이 되는 구조적 우회를 지적했다. 이번 개정은 두 상수를 **모두 제거**하고, 억제되지 않은(부정·인용에 걸리지 않은) `CHANNEL` 어절 매치와 `WEB_FUSED` 매치 모두에 다음 **단일 절차**를 적용한다(§7.2.3 순위 1·3이 이제 같은 함수를 공유한다 — 리뷰 Iteration 5 M1이 요구한 "동일한 단일 command-intent 판정 파이프라인"):

1. **`REQUEST_TAIL`이 거짓이면 그 매치는 애초에 증거가 아니다.** (변화 없음.)
2. 매치에 `SOURCE_PARTICLE`가 직접 결합했으면(`has_particle=True`) — 즉 채널·융합 표현이 조사로 출처·수단임을 스스로 밝혔거나, 조사 없이 요청 동사가 개입 어절 없이 곧바로 이어지면 — **그 자체로 command 증거**로 인정한다. 이 경우 뒤따르는 목적어 절의 내용은 검사하지 않는다. 검색의 수단·출처로 이미 확정됐으므로, 그 목적어 안에 `TOPIC_HEAD` 단어가 우연히 포함돼 있어도(예: "이번 학기 **수업방식**") 결과에 영향을 주지 않는다 — 이것이 옛 `WEB_FUSED_TOPIC_GAP`의 "`m+3`은 억제하지 않는다"는 특수 사례를 **일반화**한 규칙이다.
3. 조사 결합이 없으면(bare) — 매치가 끝나는 어절 바로 다음부터 **마지막 어절(=`REQUEST_TAIL`이 검사한 서술어) 앞까지 이어지는 절 전체**(관형 결합 "관련"/"의" 포함, 어절 개수 무관)에서 `TOPIC_HEAD` 원소를 부분 문자열로 포함하는 어절이 하나라도 있는지 스캔한다. 있으면 그 매치는 **주제절**로 억제되어 증거에서 제외되고(부정/인용 억제와는 별개의 억제 경로), 없으면(스캔 범위가 비어 있는 경우 포함, 예: 매치 바로 다음이 마지막 어절인 `구글링해서 알려줘`) command 증거로 인정한다.

이 규칙은 다음을 가른다(§7.2.4 실측):

- **bare + 전체 절 안에 `TOPIC_HEAD` 있음 → 주제절 억제**: `웹검색 방법 알려줘`(매치 직후 `방법`), `웹 검색 기술을 보여줘`(매치 직후 `기술을`), `구글링 기능 알려줘`(매치 직후 `기능`), `web search API 구조 알려줘`(매치와 마지막 어절 사이 `API`·`구조`) — 옛 `m+1`/`m+2` 경계 안. **주제어가 더 멀리 있어도 결과는 같다**: `웹검색 관련 API 구조 알려줘`, `웹검색 관련 핵심 기능 알려줘`, `구글링 사용 관련 기술 알려줘`(모두 매치와 마지막 어절 사이가 옛 기준으로는 `m+3` 이상이지만, 고정 거리를 쓰지 않으므로 전체 절 스캔이 여전히 `TOPIC_HEAD`를 찾아 억제한다 — 리뷰 Iteration 5 M1이 요구한 반례). `Google AI 에이전트 구조를 보여줘`, `온라인 서비스 구조를 보여줘`, `웹 개발 방법 알려줘`도 같은 이유로 억제된다(옛 설계에서는 "거리 초과"로 설명했으나, 새 설계에서는 애초에 거리를 재지 않고 전체 절에서 `TOPIC_HEAD`를 찾아 억제하는 것으로 설명이 바뀐다 — 결과는 동일).
- **has_particle=True → 목적어 내용과 무관하게 command 증거 유지**: `웹검색으로 최신 환율 알려줘`, `구글링해서 알려줘`(조사 없이 요청 동사가 바로 이어짐), `웹검색으로 이번 학기 수업방식 알려줘`(목적어 `수업방식`에 `TOPIC_HEAD` `방식`이 들어 있어도 `has_particle=True`이므로 스캔 자체를 하지 않아 억제되지 않음 — 옛 `WEB_FUSED_TOPIC_GAP` 경계 사례의 일반화), `웹 기준으로 답해줘`/`인터넷에서 알려줘`/`온라인 자료를 보여줘`/`구글로 조회해줘`(모두 `CHANNEL` + `SOURCE_PARTICLE`).
- **bare + 전체 절 안에 `TOPIC_HEAD` 없음 → command 증거 유지**: 위와 같은 조사 없는 매치라도 그 뒤 절에 `TOPIC_HEAD`가 전혀 없으면 억제되지 않는다(스캔 범위가 비어 있는 경우도 포함).
- `REQUEST_TAIL` 자체가 거짓인 사례는 이 규칙 이전에 이미 걸러진다: `인터넷 회사 찾아가는 길`(마지막 어절 `길`), `온라인 게임 확인 방법`(마지막 어절 `방법`), `인터넷에서 최신 소식 부탁해`·`웹에서 확인할 수 있을까`·`구글로 좀 찾아줄 수 있어?`(조사 결합이 있어도 `REQUEST_TAIL`을 면제하지 않는다, 리뷰 Iteration 4 M2 — 변화 없음).

#### 7.2-L.2 전처리 (폐기됨)

- **인용/코드 규칙**: `QUOTE_SPAN`에 해당하는 구간은 같은 길이의 공백으로 마스킹한 `masked` 문자열을 만든다. 채널 토큰·`WEB_FUSED`·`DOCUMENT` 토큰이 인용 구간 안에만 있으면 **신호로 세지 않는다**. 예: `"웹 검색"이라는 용어의 뜻을 문서에서 알려줘` → 채널 증거 0, DOCUMENT 1.
- **부정/금지 규칙**: 어떤 웹 증거(융합형 매치 또는 어절 전체 일치하는 채널 토큰) 매치의 끝 위치 뒤 **10자 이내**에 `PROHIBITION` cue가 시작하면 그 매치는 **억제**된다(정규화 후 문자 기준, 결정론적). 채널 토큰의 "끝 위치"는 §7.2.1의 어절 매치가 끝나는 문자 오프셋이다. 예: `웹 검색은 하지 말고 문서로 답해줘` → 융합형 매치 종료 후 4자 뒤에 `하지 말` → 억제. 창 크기 10은 한국어 보조 용언 연쇄(`~은/는 하지 말고`, `~ 없이`)를 덮는 최소값이며, 값 자체를 모듈 상수 `PROHIBITION_WINDOW = 10`으로 노출해 테스트에서 경계(10자/11자)를 고정한다.
- 부정 규칙은 `DOCUMENT` 토큰에는 적용하지 않는다. `문서 없이 웹에서 찾아줘`처럼 문서를 부정하는 표현은 융합형/약한 웹 증거가 이미 WEB을 만들고, 문서 부정만 있고 웹 증거가 없는 표현(`문서 없이 답해줘`)은 3순위(LLM)로 보내는 것이 안전하기 때문이다.

#### 7.2-L.3 판정 우선순위 (폐기됨)

| 순위 | 조건 | 결과 |
|---:|---|---|
| 1 | 부정·인용으로 억제되지 않고 **주제절 억제 규칙**(위)으로도 억제되지 않은 `WEB_FUSED` 매치가 있으며, `REQUEST_TAIL`이 참이다 | **WEB** |
| 2 | `DOCUMENT` 증거가 있다 | **DOCUMENT** |
| 3 | 억제되지 않고 **주제절 억제 규칙**(위)으로도 억제되지 않은 `CHANNEL` **어절** 매치가 1개 이상 있으며, `REQUEST_TAIL`이 참이다 | **WEB** |
| 4 | 그 외 | **NONE** |

순위 1과 순위 3은 이제 **동일한 주제절 억제 규칙**을 공유한다(리뷰 Iteration 5 M1 — "`WEB_FUSED`를 별도 즉시-WEB 우회 규칙으로 유지하지 않는다"). 둘의 차이는 판정 로직이 아니라 **신호 강도에 따른 우선순위 위치**뿐이다: `WEB_FUSED`(채널+검색이 한 어구로 붙은 융합형)는 요구사항의 1순위 문언을 문자 그대로 satisfy하는 가장 강한 신호이므로 `DOCUMENT`보다 앞에 오고(순위 1), 개별 `CHANNEL` 어절 + 조사/인접 결합은 더 약한 신호이므로 `DOCUMENT`보다 뒤에 온다(순위 3). 이전 설계(리뷰 Iteration 4 이전)에서 `WEB_FUSED`와 `CHANNEL`이 서로 다른 억제 메커니즘(고정 거리 상수 2개)을 썼던 것과 달리, 이제 두 신호 유형은 판정 함수 수준에서 완전히 동일한 코드 경로를 통과하며 우선순위 표에서만 위치가 다르다.

- **최소 positive grammar를 "채널 어절 + 문장 끝 요청 표현의 결합"으로 확장했다**(순위 3). 이전 초안은 검색 동사 4개(`검색`, `찾`, `알아봐`, `확인해`)만 인정해 `웹 기준으로 답해줘`, `인터넷에서 알려줘`, `온라인 자료를 보여줘`, `구글로 조회해줘` 같은 정상 요청을 NONE으로 떨어뜨렸다(리뷰 Iteration 2 M1). `REQUEST`를 요청 술어 계열 전체로 넓히되, Iteration 2 수정안이 이를 "질문 전체에 대한 전역 부분 문자열 결합"으로 구현해 두 갈래의 정밀도 결함을 만들었다(리뷰 Iteration 3 M1): (a) `websocket 설정을 알려줘`(`web`+`알려`), `Google AI 에이전트 구조를 보여줘`(`google`+`보여`), `인터넷의 역사에서 중요한 사건을 알려줘`(`인터넷`+`알려`)처럼 채널 토큰이 복합어 내부이거나 다른 조사로 다른 명사를 수식하는 **주제 언급**, (b) `인터넷 회사 찾아가는 길`, `온라인 게임 확인 방법`처럼 채널과 `REQUEST` 어간이 가깝지만 그 어간이 실제로는 관형형 어미(`찾아가는`)나 복합 명사(`확인 방법`)로 쓰여 **명령형 서술어가 아닌** 경우. §7.2.1의 **어절 전체 일치**(복합어·다른 조사 결합 배제), **주제절 억제 규칙**(위 — bare 매치 뒤 절 전체에서 `TOPIC_HEAD` 스캔), **`REQUEST_TAIL`**(요청 표현이 문장 맨 끝의 서술어 자리에 있어야 함) 세 제약을 함께 적용해, recall을 유지하면서 두 갈래 오탐을 모두 제거한다(§7.2.4 실측).
- **`WEB_FUSED`도 `CHANNEL`과 완전히 같은 절차를 통과해야 한다**(순위 1, 위 주제절 억제 규칙). 리뷰 Iteration 4 M1은 `WEB_FUSED`가 whole-token·주제어 배제·명령성 검증을 우회해 `웹검색 방법 알려줘`, `웹 검색 기술을 보여줘`, `구글링 기능 알려줘`, `web search API 구조 알려줘` 같은 검색 기술·기능 자체를 묻는 주제 질문까지 WEB으로 오분류한다고 지적했고, 그 수정으로 도입한 고정 거리 상수(`WEB_FUSED_TOPIC_GAP=2`)가 리뷰 Iteration 5 M1에서 "주제어를 두 어절 더 밀면(`m+3`) 같은 주제 질문이 다시 WEB이 되는" 구조적 우회를 남긴다고 재지적됐다. 이번 개정은 거리 상수를 제거하고 위 주제절 억제 규칙(전체 절 스캔 + `SOURCE_PARTICLE`/명령 인접 결합에 의한 즉시 command 판정)으로 대체해, `웹검색 방법 알려줘` 네 문장과 그 `m+3` 이후 변형(`웹검색 관련 API 구조 알려줘` 등, §7.2.4)이 모두 NONE으로 3순위 LLM에 넘어가고, `웹검색으로 최신 환율 알려줘`·`구글링해서 알려줘`·`웹검색으로 이번 학기 수업방식 알려줘`처럼 실제로 검색 실행을 명령하는 강한 형태는 목적어 내용과 무관하게 그대로 WEB을 유지하게 한다(§7.2.4 실측).
- **단독 언급·비서술어 사용은 WEB이 아니다**(`REQUEST_TAIL` 요구, 순위 1·3 공통). `웹이 뭐야?`, `인터넷의 역사에 대해 설명해줘`처럼 채널 토큰이 **주제**로 쓰이거나, `인터넷 회사 찾아가는 길`/`온라인 게임 확인 방법`처럼 `REQUEST` 어간이 문장 끝이 아닌 관형어·복합 명사로 쓰인 문장은 NONE으로 남아 3순위(LLM)가 판단한다. `인터넷에서 최신 소식 부탁해`/`웹에서 확인할 수 있을까`/`구글로 좀 찾아줄 수 있어?`처럼 출처·수단 조사가 결합돼 있어도 마지막 어절이 요청 표현이 아니면 같은 이유로 NONE이다(리뷰 Iteration 4 M2 — 조사 결합은 채널의 출처 의미를 확인할 뿐 `REQUEST_TAIL`을 면제하는 독립 충분조건이 아니다).
- **맨몸 요청 표현만으로도 WEB이 아니다**(순위 3이 `CHANNEL` 어절 매치를 요구). 그렇지 않으면 `bd-000`("이 문서에서 관련 내용을 **찾아줘**", 기대 document_qa, M2에서 정답)이 WEB으로 뒤집혀 **정답이 회귀**한다. `bd-000`은 채널 토큰이 없으므로 순위 2에서 DOCUMENT다. 이 케이스는 필수 회귀 테스트다.
- **충돌 시 강도 순으로 해소한다.** 융합형(`웹검색으로 …`)은 요구사항 1순위의 문자 그대로의 근거이므로 문서 범위 토큰을 이긴다(순위 1). 반면 **약한 증거(채널 어절 + 요청 표현의 결합)는 명시적 문서 범위에 진다**(순위 2 < 3). `온라인 문서에서 알려줘`는 어절 매치("온라인", 뒤 절에 `TOPIC_HEAD` 없음)로 순위 3 증거가 생기지만, `문서에서`가 `DOCUMENT` 토큰이므로 순위 2가 먼저 걸려 DOCUMENT가 된다. `구글 백서에서 에이전트 정의를 알려줘`는 `백서에서`가 `DOCUMENT` 토큰이므로 순위 2로 직행한다(순위 3 증거 성립 여부와 무관하게 순위 2가 먼저 걸린다 — 옛 설계는 이를 "거리 초과로 순위 3 불성립"으로 따로 설명했으나, 새 설계는 순위 2가 순위 3보다 먼저 검사되므로 순위 3의 성립 여부 자체가 결과에 영향을 주지 않는다) — 두 사례 모두 "명시적 문서 범위가 있으면 그쪽이 더 강한 명시 신호"라는 동일한 우선순위 원칙을 보인다. 이 우선순위 정의는 요구사항 §5 M3-REQ-004에 같은 문단으로 기록했다(리뷰 Iteration 3 t1 대응).
- **주제절 억제(순위 1·3 공통)가 필요한 이유**: 이 저장소의 corpus와 골든셋은 RAG·에이전트·도구 사용을 다루므로 `웹검색 기능이 이 문서에 어떻게 설명돼 있나요?`, `웹검색 방법 알려줘`처럼 **웹 검색 자체가 설명·서술 대상인 주제 질문**이 현실적으로 존재하며, 주제어가 채널·융합 표현으로부터 멀리 떨어져 있어도(예: "관련"으로 연결된 명사구) 여전히 같은 주제 질문이다(리뷰 Iteration 3~5). `WEB_FUSED`·`CHANNEL`이 무조건 이기면 이런 주제 질문이 WEB으로 오분류된다. 주제절로 판정되면 매치를 증거에서 제외하므로, `DOCUMENT` 토큰이 함께 있으면 순위 2에서 DOCUMENT로, 없으면 순위 3·4를 거쳐 최종 NONE(3순위 LLM)으로 판정한다.
- **두 종류 이상의 증거가 동시에 관측되면 리포트에 남긴다.** `routing_policy.signal_conflict_count`(웹 증거와 DOCUMENT 증거가 함께 관측된 사례 수)와 `routing_policy.signal_suppressed_count`(부정/인용으로 억제된 사례 수)를 routing 리포트에 기록해, 규칙이 실제 데이터에서 어떻게 작동했는지 사후 검증할 수 있게 한다(§3.3).
- 함수는 `ExplicitSignal`만 반환한다. 위 두 카운터는 `classify_explicit_signal_detail(question) -> SignalDecision`(신호 + 증거 목록 + 억제/충돌 플래그) 형태의 **부가 함수**로 노출하고, `classify_explicit_signal()`은 그 `.signal`을 돌려주는 얇은 wrapper다. 제품 경로는 wrapper만 쓰고 evaluator만 detail을 쓴다(로직 복제 금지, M3-NFR-004).

#### 7.2-L.4 골든셋 76건 dry-run 기대값 (폐기됨)

위 규칙을 골든셋 전수에 적용한 결과는 **WEB 10 / DOCUMENT 12 / NONE 54**이며 오탐은 0이다(2026-08-07 스크립트 재확인, 부록 A — `CHANNEL_REQUEST_MAX_WORD_GAP`·`WEB_FUSED_TOPIC_GAP` 두 거리 상수를 제거하고 주제절 억제 규칙으로 교체한 뒤에도 3개 ID 집합과 카운트는 2026-08-06 확인분과 완전히 동일하다 — 골든셋 76건 중 `WEB_FUSED`가 매치하는 3건(`ws-003`, `ws-008`, `rr-ws-samsung-stock-001`)과 `CHANNEL` 순위 3으로 잡히는 나머지 WEB 건은 모두 주제절에 `TOPIC_HEAD`가 없거나 `SOURCE_PARTICLE`이 결합돼 있어 새 규칙에서도 그대로 WEB이다).

- **WEB 10건**: `ws-000`, `ws-002`, `ws-003`, `ws-005`, `ws-007`, `ws-008`, `ws-009`, `rr-ws-python-version-001`, `rr-ws-bitcoin-price-001`, `rr-ws-samsung-stock-001`. 전부 `expected_route=web_search`다.
- **DOCUMENT 12건**: `bd-000`, `ua-000`~`ua-006`, `dq-agent-arch-001`, `dq-agent-vs-model-001`, `dq-kb-price-outlook-001`, `dq-kb-gangnam-001`. 전부 `expected_route=document_qa`다.
- **NONE 54건**: 나머지 전부. 여기에는 채널 토큰이 없는 실시간 값 질문(`ws-001`, `ws-004`, `ws-006`, `rr-ws-seoul-temp-001`, `rr-ws-exchange-rate-001`)이 포함되며, 이들은 설계 의도대로 3순위(LLM + 개정 프롬프트)가 판정한다.
- 두 불변식은 **"WEB 오탐 0"**(WEB 판정 사례의 `expected_route`는 모두 `web_search`)와 **"DOCUMENT 오탐 0"**(DOCUMENT 판정 사례의 `expected_route`는 모두 `document_qa`)이다. 위 3개 ID 집합은 정확한 집합 동등성으로 assert한다(부분집합 assert 금지 — recall 결함이 다시 숨는다).

**양성 paraphrase fixture**(골든셋 밖, 순위 1·3의 recall 증거, 13건 — 리뷰 Iteration 2 M1이 지적한 표현, 리뷰 Iteration 4 M1이 명시한 융합형 강한 명령 표현, 리뷰 Iteration 5 M1이 요구한 왼쪽 경계 사례를 모두 포함):

| 입력 | 기대 | 근거 순위 |
|---|---|---|
| `웹 기준으로 답해줘` | WEB | 3 (`웹`은 bare 매치. 스캔 범위 `기준으로`에 `TOPIC_HEAD` 없음 → 억제 안 됨, `REQUEST_TAIL`=`답해줘` 참) |
| `인터넷에서 알려줘` | WEB | 3 (`인터넷에서` = `CHANNEL`+`SOURCE_PARTICLE` 결합 → `has_particle=True`, 스캔 없이 즉시 command 증거) |
| `온라인 자료를 보여줘` | WEB | 3 (`온라인`은 bare. 스캔 범위 `자료를`에 `TOPIC_HEAD` 없음 → 억제 안 됨) |
| `구글로 조회해줘` | WEB | 3 (`구글로` = `CHANNEL`+`SOURCE_PARTICLE`=`로` → `has_particle=True`) |
| `검색엔진으로 확인해줘` | WEB | 3 (`SOURCE_PARTICLE`=`으로` 결합 → `has_particle=True`) |
| `포털에서 가져와줘` | WEB | 3 (`SOURCE_PARTICLE`=`에서` 결합 → `has_particle=True`) |
| `웹서치로 최신 환율 알려줘` | WEB | 1 (`WEB_FUSED`=`웹서치로`, `SOURCE_PARTICLE`=`로` 결합 → `has_particle=True`, 스캔 없이 즉시 command 증거) |
| `웹검색으로 최신 환율 알려줘` | WEB | 1 (동일 근거 — 융합 표현이 `웹서치`→`웹검색`으로 바뀌어도 결과는 같다. 리뷰 Iteration 4 M1이 WEB 유지를 명시적으로 요구한 경계 사례) |
| `구글링해서 알려줘` | WEB | 1 (`WEB_FUSED`=`구글링`, 조사 없이 요청 동사 `알려줘`가 개입 없이 바로 이어짐 → 스캔 범위가 비어 `TOPIC_HEAD` 없음, command 증거 유지) |
| `웹검색으로 이 문서 내용을 확인해줘` | WEB | 1 (`SOURCE_PARTICLE`=`으로` 결합 → `has_particle=True`, 목적어에 `문서`가 있어도 스캔하지 않으므로 문서 범위 토큰(순위 2)보다 먼저 순위 1이 WEB으로 확정) |
| `웹검색으로 이번 학기 수업방식 알려줘` | WEB | 1 (`WEB_FUSED`=`웹검색으로`, `SOURCE_PARTICLE`=`으로` 결합 → `has_particle=True`. 목적어 `수업방식`에 `TOPIC_HEAD` `방식`이 포함돼 있어도 `has_particle=True`이면 절 내용을 스캔하지 않으므로 억제되지 않는다 — 옛 `WEB_FUSED_TOPIC_GAP` 경계 사례를 거리 없이 일반화) |
| `질문:웹검색으로 최신 환율 알려줘` | WEB | 1 (`WEB_FUSED` 왼쪽 경계가 `:` — Unicode-aware 왼쪽 경계(`(?<!\w)`)가 문장부호 뒤를 인정, `SOURCE_PARTICLE`=`으로` 결합으로 즉시 command 증거. 리뷰 Iteration 5 M1이 지적한 왼쪽 경계 과소탐지 회귀 사례) |
| `(구글링해서 알려줘)` | WEB | 1 (`WEB_FUSED` 왼쪽 경계가 `(` — 동일하게 Unicode-aware 경계로 인정, 조사 없이 요청 동사가 바로 이어져 command 증거. 마지막 어절 `알려줘)`는 끝의 문장부호 제거 후 `REQUEST_TAIL` 성립. 리뷰 Iteration 5 M1 대응) |

**부정·충돌·단독 언급 fixture**(precision 증거):

| 입력 | 기대 | 근거 순위 |
|---|---|---|
| `웹 검색은 하지 말고 문서로 답해줘` | DOCUMENT | 부정 억제 → 2 |
| `인터넷 검색 없이 제공된 문서만으로 답해줘` | DOCUMENT | 부정 억제 → 2 |
| `웹에서 찾지 말고 첨부 자료로 답해` | DOCUMENT | 부정 억제 → 2 |
| `"웹 검색"이라는 용어의 뜻을 문서에서 알려줘` | DOCUMENT | 인용 마스킹 → 2 |
| `` `웹검색`이 무슨 기능인지 문서에서 알려줘 `` | DOCUMENT | 인용 마스킹 → 2 |
| `웹검색 기능이 이 문서에 어떻게 설명돼 있는지 알려줘` | DOCUMENT | 2 (`WEB_FUSED`=`웹검색` bare 매치, 스캔 범위 `기능이`에 `TOPIC_HEAD` → 주제절 억제로 WEB 증거 없음, `DOCUMENT` 토큰 `문서에`로 순위 2 적중) |
| `구글 백서에서 에이전트 정의를 알려줘` | DOCUMENT | 2 (`백서에서`가 `DOCUMENT` 토큰이므로 순위 2가 순위 3보다 먼저 걸려 문서 범위로 직행 — `구글`이 순위 3 command 증거를 만드는지 여부와 무관하게 결과는 DOCUMENT다) |
| `웹이 뭐야?` | NONE | 4 (단독 언급) |
| `인터넷의 역사에 대해 설명해줘` | NONE | 4 (`설명`은 `REQUEST` 아님) |
| `검색해줘` / `찾아줘` / `알려줘` | NONE | 4 (맨몸 요청 표현) |
| `이 문서에서 관련 내용을 찾아줘`(`bd-000`) | DOCUMENT | 2 |
| `인터넷에서 최신 소식 부탁해` | NONE | 4 (`인터넷에서` = `CHANNEL`+`SOURCE_PARTICLE` 결합으로 어절 전체 일치는 성립하나, 마지막 어절 `부탁해`가 `REQUEST` 원소를 포함하지 않아 `REQUEST_TAIL` 거짓 → 순위 3 불성립. 출처·수단 조사 결합이 요청 종결 요건을 면제하지 않음을 보이는 사례, 리뷰 Iteration 4 M2 대응) |
| `웹에서 확인할 수 있을까` | NONE | 4 (`웹에서`는 어절 전체 일치하나, 마지막 어절 `있을까`는 `REQUEST` 원소가 없어 `REQUEST_TAIL` 거짓 — `확인`은 문장 중간에 있을 뿐 서술어 자리가 아니다. 리뷰 Iteration 4 M2 대응) |
| `구글로 좀 찾아줄 수 있어?` | NONE | 4 (`구글로`는 어절 전체 일치하나, 마지막 어절 `있어`는 `REQUEST` 원소가 없어 `REQUEST_TAIL` 거짓 — `찾아줄`이 관형형으로 `수`를 수식할 뿐 문장 끝 서술어가 아니다. 리뷰 Iteration 4 M2 대응) |

**채널 주제어 오인 방지 음성 fixture**(리뷰 Iteration 3 M1 — 확장된 recall 규칙이 만든 신규 정밀도 결함을 막는 회귀 사례. 4~11번째는 리뷰 Iteration 4 M1, 마지막 3건은 리뷰 Iteration 5 M1 — `WEB_FUSED`가 같은 종류의 결함을 형태만 바꿔 재생산하는 것을 막는 회귀 사례):

| 입력 | 기대 | 근거 순위 |
|---|---|---|
| `websocket 설정을 알려줘` | NONE | 4 (`websocket`은 `web` + `에서/으로/로`가 아니라 `web`+`socket`이므로 어절 전체 일치 자체가 실패) |
| `webhook 사용법을 알려줘` | NONE | 4 (동일 — `webhook` ≠ `web`(`에서`&#124;`으로`&#124;`로`)) |
| `googleapis 사용법을 알려줘`(참고: `구글` 임베딩) | NONE | 4 (`googleapis`는 `google`+`apis`이므로 `CHANNEL` 어절 전체 일치도 `WEB_FUSED`도 성립하지 않아 웹 증거 자체가 없다 — 리뷰 Iteration 5 M1이 명시한 복합어 반례) |
| `Google AI 에이전트 구조를 보여줘` | NONE | 4 (`Google`은 bare 매치. 스캔 범위 `AI`·`에이전트`·`구조를`에 `TOPIC_HEAD`(`구조`) 있음 → 주제절 억제) |
| `인터넷의 역사에서 중요한 사건을 알려줘` | NONE | 4 (`인터넷의`는 조사 `의`가 붙어 어절 전체 일치 자체가 실패 — `역사에서`의 `에서`는 `역사`에 결합한 것이지 `인터넷`에 결합한 것이 아니다) |
| `온라인 서비스 구조를 보여줘` | NONE | 4 (`온라인`은 bare 매치. 스캔 범위 `서비스`·`구조를`에 `TOPIC_HEAD`(`구조`) 있음 → 주제절 억제) |
| `온라인 문서에서 알려줘` | DOCUMENT | 2 (`온라인`은 bare 매치, 스캔 범위 `문서에서`에 `TOPIC_HEAD` 없어 순위 3 command 증거는 성립하나, `문서에서`가 `DOCUMENT` 토큰이라 순위 2가 먼저 걸림 — 우선순위 2 < 3을 실제로 행사하는 사례) |
| `웹 개발 방법 알려줘` | NONE | 4 (`웹`은 bare 매치. 스캔 범위 `개발`·`방법`에 `TOPIC_HEAD`(`방법`) 있음 → 주제절 억제. 옛 설계는 이를 "거리 초과"로 설명했으나 결과는 동일) |
| `인터넷 회사 찾아가는 길` | NONE | 4 (마지막 어절은 `길`이지 `찾아가는`이 아니므로 `REQUEST_TAIL` 거짓 — `찾`이 관형형 어미로 뒤 명사 `길`을 수식할 뿐 명령형 서술어가 아니다. `REQUEST_TAIL`이 거짓이므로 주제절 스캔 이전에 이미 증거가 아니다) |
| `온라인 게임 확인 방법` | NONE | 4 (마지막 어절은 `방법`이지 `확인`이 아니므로 `REQUEST_TAIL` 거짓 — `확인`이 복합 명사 `확인 방법`의 일부일 뿐 명령형 서술어가 아니다) |
| `웹검색 방법 알려줘` | NONE | 4 (`WEB_FUSED`=`웹검색` bare 매치. 스캔 범위 `방법`에 `TOPIC_HEAD` 있음 → 주제절 억제. `웹검색`은 `CHANNEL` 원소+조사 형태가 아니므로 순위 3 증거도 없어 NONE — 리뷰 Iteration 4 M1) |
| `웹 검색 기술을 보여줘` | NONE | 4 (`WEB_FUSED`=`웹 검색` bare 매치. 스캔 범위 `기술을`에 `TOPIC_HEAD` 있음 → 주제절 억제. `웹`은 `CHANNEL` 어절 전체 일치이나 그 뒤 절(`검색`·`기술을`)에도 `TOPIC_HEAD`가 있어 순위 3도 억제 → NONE — 리뷰 Iteration 4 M1) |
| `구글링 기능 알려줘` | NONE | 4 (`WEB_FUSED`=`구글링` bare 매치. 스캔 범위 `기능`에 `TOPIC_HEAD` 있음 → 주제절 억제. `구글링`은 `CHANNEL` 원소+조사 형태가 아니므로 순위 3 증거도 없어 NONE — 리뷰 Iteration 4 M1) |
| `web search API 구조 알려줘` | NONE | 4 (`WEB_FUSED`=`web search` bare 매치. 스캔 범위 `API`·`구조`에 `TOPIC_HEAD`(`구조`) 있음 → 주제절 억제. `web`은 `CHANNEL` 어절 전체 일치이나 그 뒤 절에도 `TOPIC_HEAD`가 있어 순위 3도 억제 → NONE — 리뷰 Iteration 4 M1) |
| `웹검색 관련 API 구조 알려줘` | NONE | 4 (`WEB_FUSED`=`웹검색` bare 매치. **옛 `WEB_FUSED_TOPIC_GAP=2` 기준으로는 `TOPIC_HEAD`(`구조`)가 매치로부터 3어절 떨어져 있어 억제되지 않고 WEB으로 오분류됐을 m+3 우회 사례** — 새 규칙은 고정 거리를 재지 않고 매치 다음부터 마지막 어절 앞까지 전체 절(`관련`·`API`·`구조`)을 스캔하므로 `구조`를 찾아 그대로 억제한다. `웹검색`은 조사 없는 bare 매치이므로 순위 3 `CHANNEL` 증거도 없어 최종 NONE(리뷰 Iteration 5 M1 필수 반례) |
| `웹검색 관련 핵심 기능 알려줘` | NONE | 4 (동일 근거 — 스캔 범위 `관련`·`핵심`·`기능`에 `TOPIC_HEAD`(`기능`) 있음 → 주제절 억제. m+3 우회 반례, 리뷰 Iteration 5 M1) |
| `구글링 사용 관련 기술 알려줘` | NONE | 4 (`WEB_FUSED`=`구글링` bare 매치. 스캔 범위 `사용`·`관련`·`기술`에 `TOPIC_HEAD`(`기술`) 있음 → 주제절 억제. m+3 우회 반례, 리뷰 Iteration 5 M1) |

이 네 묶음(골든셋 전수 집합 동등성, 양성 paraphrase, 부정·충돌·단독 언급, 채널 주제어 오인 방지 음성 fixture)을 검사하는 단위 테스트가 M3-REQ-004의 "외부에서 검증 가능"에 대한 증거다. 어휘 집합에 원소를 추가·삭제하거나 `REQUEST_TAIL`·`SOURCE_PARTICLE`·주제절 억제 규칙을 바꾸는 변경은 이 네 묶음을 모두 재실행해야 한다. 위 fixture 확장(리뷰 Iteration 4 M1·M2, Iteration 5 M1 대응)은 §7.2.4 골든셋 76건 dry-run 결과(WEB 10 / DOCUMENT 12 / NONE 54, 오탐 0)를 바꾸지 않는다 — 골든셋 76건 중 `WEB_FUSED`가 매치하는 3건(`ws-003`, `ws-008`, `rr-ws-samsung-stock-001`)은 모두 주제절에 `TOPIC_HEAD`가 없거나 `SOURCE_PARTICLE`이 결합돼 있어 새 조건에서도 그대로 WEB이다(2026-08-07 재확인).

**property-style 왼쪽 경계 표**(리뷰 Iteration 5 M1 — 왼쪽 경계 변경이 정밀도를 깨지 않음을 진리표로 고정, `test_routing_signals.py`에 파라미터화 테스트로 반영):

| 매치 직전 문자 | 예시 | `(?<!\w)` 판정 | 근거 |
|---|---|---|---|
| 문자열 시작 | `웹검색으로 확인해줘` | 경계 성립 | 변화 없음(기존과 동일) |
| 공백 | `내용을 웹검색으로 확인해줘` | 경계 성립 | 변화 없음(기존과 동일) |
| 쉼표 | `그리고, 웹검색으로 확인해줘` | 경계 성립(신규) | 옛 정의(시작/공백만)는 실패했던 사례 |
| 콜론 | `질문:웹검색으로 최신 환율 알려줘` | 경계 성립(신규) | §7.2.4 양성 fixture로 고정 |
| 여는 괄호 `(`/`[` | `(구글링해서 알려줘)`, `[구글링해서 알려줘]` | 경계 성립(신규) | §7.2.4 양성 fixture로 고정 |
| 여는 따옴표 뒤(따옴표 자체는 `QUOTE_SPAN`으로 별도 마스킹) | — | 해당 없음 | `QUOTE_SPAN` 내부는 §7.2.2 인용 규칙으로 먼저 마스킹되므로 왼쪽 경계 판정 이전에 이미 공백 처리됨 |
| 한글 음절(`\w`) | `무료웹검색사이트`의 `웹검색` | 경계 불성립(변화 없음) | `료`가 `\w`이므로 매치 실패 — 복합어 내부 임베딩 배제 유지 |
| ASCII 알파벳(`\w`) | `freewebsearch`의 `websearch` | 경계 불성립(변화 없음) | `e`가 `\w`이므로 매치 실패 |
| ASCII 알파벳(`\w`, `구글` 임베딩) | `googleapis`의 `google` | 왼쪽 경계는 성립하나 오른쪽에서 `\s?(검색\|서치\|search)` 불성립 | `apis`가 검색 접미사가 아니므로 `WEB_FUSED` 자체가 매치하지 않는다(경계 문제가 아니라 접미사 문제) |

### 7.3 SYSTEM_PROMPT 개정과 corpus topic hint

개정 방향(현행 문구의 도구 계약 부분은 그대로 유지):

1. 기본값을 문서 우선으로 명시: "판단이 애매하면 document_qa를 선택한다."
2. 오작동 유발 신호를 명시적으로 무효화: "질문에 연도(2024/2025), 정책·법령·기업·제품·지수 이름이 들어 있다는 이유만으로 web_search를 선택하지 않는다. 등록 문서에는 최근 연도의 보고서와 정책 자료가 포함돼 있다."
3. web_search 조건을 좁힘: "① 사용자가 웹/인터넷/온라인 검색을 명시했거나, ② 오늘의 날씨·현재 시세·실시간 지수·환율·경기 결과·속보처럼 **지금 이 순간의 값**이 필요한 질문일 때만 web_search를 선택한다."
4. tool query 계약(웹은 키워드 추출, 문서는 원문 유지)은 문구 그대로 보존한다.

`build_corpus_topic_hint(file_names, max_items) -> str`(순수 함수, `routing_signals.py`):
- 입력은 `Path(config.DATA_DIR)`의 파일명 목록(정렬). 확장자 제거, `복사본`/연속 공백/언더스코어 정리, 중복 제거, 최대 `max_items`개, 총 길이 1,200자 상한.
- 출력 예: `등록 문서 주제: 2025 KB 부동산 보고서, 2025 APEC 정상회담 주요 협의사항, 2025 한국정부 부동산정책 정리, 2025년 한국 경제 전망(현대경제연구원), google-ai-agents-whitepaper, LangGraph 개요, …, SPRI AI Brief 2023년12월호, 체인(Chain) 생성, 텍스트 분할, 프롬프트`.
- 현재 corpus 18개 파일명은 실패 17건의 주제(부동산 정책·KB 보고서·경제 전망·APEC·AI Brief·에이전트 백서·체인 생성)를 **모두** 포함하므로 hint의 기대 효과가 크다.
- corpus가 바뀌면 hint도 자동으로 바뀐다(하드코딩 금지). 디렉터리 접근 실패 시 빈 문자열을 반환하고 프롬프트는 hint 없이 구성된다.
- **loopback 제한(M3-NFR-003, t2)**: hint 조립은 `ROUTING_CORPUS_TOPIC_HINT and is_loopback_endpoint(OLLAMA_BASE_URL)`일 때만 수행한다. 비-loopback이면 hint를 만들지 않고 경고 로그를 남기며 리포트 `routing_policy.corpus_topic_hint_suppressed_reason="non_loopback_endpoint"`를 채운다. 현재 `OLLAMA_BASE_URL`은 `http://localhost:11434` 상수이므로 지금은 항상 활성이지만, 향후 endpoint가 설정 가능해져도 corpus 파일명이 외부로 전송될 수 없다. 마스킹이나 opt-in 대신 **자동 억제**를 택한 이유는, 파일명이 사용자 가치에 필수가 아니고(hint 없이도 프롬프트 개정만으로 후보 1이 성립) 억제가 가장 단순한 안전 기본값이기 때문이다.
- 프롬프트가 corpus에 의존하게 되므로 routing 리포트에 `router_prompt_sha256`를 항상 기록한다(§3.3). Routing 리포트의 `reproducibility_note`는 hint 활성 시 "corpus 파일명이 router 프롬프트에 포함됨"으로 갱신한다.

### 7.4 `_decide_tool()` 명시 신호 우선 판정 설계

요구사항 M3-REQ-004는 1·2순위를 "**LLM 호출 이전에** 결정론적으로 판정"하고 "LLM 예외·no-tool에서도 우선순위가 유지"되도록 요구한다. 따라서 신호 분류는 LLM 호출보다 **먼저** 실행한다.

```python
def _decide_tool(question: str) -> tuple[str | None, str | None]:
    # 1) 명시 신호를 LLM 호출 전에 결정론적으로 판정한다.
    signal = ExplicitSignal.NONE
    if ROUTING_SIGNAL_OVERRIDE:
        try:
            signal = classify_explicit_signal(question)
        except Exception:                      # 신호 분류 자체의 결함이 라우팅을 막지 않게
            _log_signal_error()                # routing_policy.signal_error_count
            signal = ExplicitSignal.NONE

    # 2) DOCUMENT: LLM을 호출하지 않는다. 문서 경로의 query 계약은 "원본 질문"이므로
    #    LLM이 기여할 정제 여지가 없고, 모델 장애와 무관하게 사용자 의도가 보존된다.
    if signal is ExplicitSignal.DOCUMENT:
        return "document_qa", question

    # 3) WEB: 검색어 품질을 위해 LLM을 시도하되, 실패해도 route는 web을 유지한다.
    if signal is ExplicitSignal.WEB:
        try:
            llm_name, llm_query = _llm_decide_tool(question)
        except Exception:                      # 예외를 keyword fallback으로 흘리지 않는다
            _log_llm_error()
            return "web_search", extract_web_search_query(question)
        if llm_name == "web_search" and llm_query:
            return "web_search", llm_query     # LLM이 정제한 검색어를 우선 사용
        # no-tool((None, None)), 다른 도구 선택, 빈 query → 결정론적으로 보완
        return "web_search", extract_web_search_query(question)

    # 4) NONE: 기존 계약을 그대로 유지한다(예외 전파, (None, None) 반환 포함).
    return _llm_decide_tool(question)
```

설계 결정과 근거:

- **신호 판정을 LLM보다 먼저 둔다.** 이전 초안은 `_llm_decide_tool()`을 항상 먼저 호출한 뒤 override했는데, LLM이 예외를 던지면 신호 분류가 아예 실행되지 않아 "가장 강한 사용자 의도"가 모델 가용성에 종속됐다. 순서를 바꾸면 이 결함이 구조적으로 사라진다.
- **tool query 계약은 그대로 지킨다.** DOCUMENT는 정의상 원본 질문이므로 LLM이 필요 없다(부수 효과로 라우팅 latency도 줄지만, 이는 gate 대상이 아니라 부산물이다). WEB은 LLM 정제 검색어를 **우선** 쓰고 예외·no-tool·빈 값일 때만 검증된 `extract_web_search_query()`로 대체하므로 검색어 품질이 정상 경로에서 저하되지 않는다.
- **NONE 경로는 완전히 무회귀다.** 신호가 없으면 `_llm_decide_tool()`의 반환·예외 계약이 호출자에게 그대로 노출되고, `route_query()`의 ①②③ 폴백이 M2와 동일하게 동작한다.
- **WEB/DOCUMENT에서 keyword fallback을 타지 않는 것은 의도된 계약 변경이다**(§3.5). 명시 신호가 있는 질문에서 keyword fallback이 다른 route를 고르면 요구사항의 우선순위가 깨지기 때문이다. 웹 검색 실행 실패 시 원본 질문으로 document QA를 재시도하는 ④는 그대로 보존된다.
- **`_decide_tool()` 안에 둔다.** `evaluation.routing --mode live`가 이 함수를 직접 호출하므로 평가와 제품이 같은 정책을 본다(M3-NFR-004, evaluator에 로직 복제 금지).
- `ROUTING_SIGNAL_OVERRIDE=False`(기본값)이면 함수는 M2와 완전히 동일하게 `_llm_decide_tool(question)`만 호출한다 — rollback 스위치(§13.3).
- 시그니처와 반환 타입은 불변이다. NONE 경로에서만 `(None, None)`과 예외가 관찰될 수 있다.

`extract_web_search_query(question) -> str`는 `query_router.py`의 기존 키워드 제거 로직을 **동작 변경 없이** 순수 함수로 추출한 것이다. `query_router.route_query()`는 이 함수를 호출하도록 바꾸며, 기존 단위 테스트(`tests/unit/test_query_router.py`)가 그대로 통과해야 한다. 빈 문자열을 반환할 수 있는 입력에 대해서는 원본 질문으로 되돌리는 기존 동작을 유지한다.

#### 필수 테스트 행렬 (`tests/integration/test_agent_routing_policy.py`, 모델 불필요)

`_llm_decide_tool`을 stub으로 두고 신호 × LLM 결과의 모든 조합을 고정한다.

| # | 신호 | `_llm_decide_tool` stub | 기대 결과 | 비고 |
|---:|---|---|---|---|
| 1 | WEB | `("web_search","환율")` | `("web_search","환율")` | LLM 검색어 우선 |
| 2 | WEB | `("document_qa", q)` | `("web_search", extract(q))` | route 교정 |
| 3 | WEB | `(None, None)` | `("web_search", extract(q))` | **no-tool에서도 유지** |
| 4 | WEB | `("web_search", "")` | `("web_search", extract(q))` | 빈 query 보완 |
| 5 | WEB | `raise RuntimeError` | `("web_search", extract(q))` | **예외에서도 유지** |
| 6 | DOCUMENT | 호출되지 않아야 함 | `("document_qa", q)` | stub 호출 수 0 assert |
| 7 | DOCUMENT | `raise RuntimeError` | `("document_qa", q)` | 호출 자체가 없으므로 무관 |
| 8 | NONE | `("document_qa", q)` | `("document_qa", q)` | 통과 |
| 9 | NONE | `(None, None)` | `(None, None)` | 기존 ③ 폴백 계약 |
| 10 | NONE | `raise RuntimeError` | 예외 전파 | 기존 ① 폴백 계약 |
| 11 | 임의 | flag off | LLM 결정 그대로 | rollback 검증 |
| 12 | 신호 분류 함수가 예외 | `("document_qa", q)` | `("document_qa", q)` + `routing_policy.signal_error_count==1` | 신호 결함 격리 |

추가로 `route_query()` 수준에서 ①②③④ 폴백 4경로가 NONE 신호에서 M2와 동일하게 동작하는지, WEB 신호 + 웹 검색 실행 실패에서 ④(원본 질문 document QA 재시도)가 유지되는지 확인한다.

#### 필수 통합 계약 — 단순화 Cycle 1 실제 classifier

아래 12개가 현재 규범 행렬이다. classifier는 실제 문자열을 사용하고 `_llm_decide_tool`만 stub한다.

| # | 입력 | LLM stub | 기대 | 계약 |
|---:|---|---|---|---|
| S1 | `웹에서 검색해줘` | 예외 | WEB + fallback query | 채널 지정 검색 명령 |
| S2 | `구글링해서 알려줘` | no-tool | WEB + fallback query | 직접 검색 명령 |
| S3 | `질문:웹에서 검색해줘` | 예외 | WEB + fallback query | Unicode 경계 |
| S4 | `웹 검색은 하지 말고 문서로 답해줘` | 미호출 | DOCUMENT | 부정 + 문서 범위 |
| S5 | `"웹 검색"의 뜻을 문서에서 알려줘` | 미호출 | DOCUMENT | 인용 + 문서 범위 |
| S6 | `이 문서에서 관련 내용을 찾아줘` | 미호출 | DOCUMENT | 문서 범위 |
| S7 | `웹검색으로 최신 환율 알려줘` | `document_qa` | LLM 결과 통과 | 일반 응답 술어 → NONE |
| S8 | `웹검색에서 사용하는 API 구조 알려줘` | 예외 | 예외 전파 | 관형절 최소쌍 1 → NONE |
| S9 | `구글에서 사용하는 검색 기술 알려줘` | no-tool | no-tool 통과 | 관형절 최소쌍 2 → NONE |
| S10 | `웹검색 방법 알려줘` | `document_qa` | LLM 결과 통과 | 검색 주제 → NONE |
| S11 | `freewebsearch 사용법 알려줘` | 예외 | 예외 전파 | 복합어 내부 → NONE |
| S12 | 임의 입력, flag off | 임의 결과 | LLM 결과 그대로 | rollback |

WEB/DOCUMENT에서는 LLM 장애와 무관하게 결정론 route를 유지하고, NONE에서는 기존 반환·예외·fallback 계약을 그대로 관찰한다. 이 행렬과 §7.2의 골든 8/12/56 exact set만 현재 구현·인수 테스트에 사용한다.

#### 이전 R1~R36 행렬 — 비규범 감사 기록

아래 행렬은 Iteration 1~6의 기록이며 테스트로 구현하지 않는다.

위 12칸은 `classify_explicit_signal()`을 **stub**으로 두므로 §7.2 어휘 집합의 recall/precision 결함을 잡지 못한다(리뷰 Iteration 2 M1, Iteration 3 M1). 따라서 같은 파일에 **classifier를 stub하지 않고 실제 문자열을 넣는** 두 번째 행렬을 둔다. 여기서는 `_llm_decide_tool`만 stub하며, 신호는 실제 `routing_signals.classify_explicit_signal()`이 계산한다(Iteration 4 리비전에서 리뷰 M1·M2 대응 행 R21~R30을 추가했다).

| # | 실제 입력 문자열 | `_llm_decide_tool` stub | 기대 결과 | 검증 대상 |
|---:|---|---|---|---|
| R1 | `웹 기준으로 답해줘` | `raise RuntimeError` | `("web_search", extract(q))` | 확장된 positive grammar가 **LLM 예외**에서도 web을 유지 |
| R2 | `인터넷에서 알려줘` | `(None, None)` | `("web_search", extract(q))` | **no-tool**에서도 web 유지 |
| R3 | `온라인 자료를 보여줘` | `("document_qa", q)` | `("web_search", extract(q))` | route 교정 |
| R4 | `구글로 조회해줘` | `("web_search", "")` | `("web_search", extract(q))` | 빈 query 보완 |
| R5 | `구글로 조회해줘` | `("web_search", "구글 조회")` | `("web_search", "구글 조회")` | LLM 정제 검색어 우선 |
| R6 | `웹 검색은 하지 말고 문서로 답해줘` | `raise RuntimeError` | `("document_qa", q)` | 부정 규칙 → DOCUMENT, LLM 미호출(호출 수 0 assert) |
| R7 | `"웹 검색"이라는 용어의 뜻을 문서에서 알려줘` | `raise RuntimeError` | `("document_qa", q)` | 인용 규칙 → DOCUMENT, LLM 미호출 |
| R8 | `웹검색 기능이 이 문서에 어떻게 설명돼 있는지 알려줘` | `raise RuntimeError` | `("document_qa", q)` | `WEB_FUSED` 주제어 인접 억제 → DOCUMENT, LLM 미호출 |
| R9 | `웹검색으로 이 문서 내용을 확인해줘` | `(None, None)` | `("web_search", extract(q))` | 융합형이 문서 범위를 이김 |
| R10 | `웹이 뭐야?` | `(None, None)` | `(None, None)` | 단독 언급 → NONE 경로의 기존 ③ 계약 보존 |
| R11 | `인터넷의 역사에 대해 설명해줘` | `raise RuntimeError` | 예외 전파 | 단독 언급 → NONE 경로의 기존 ① 계약 보존 |
| R12 | `이 문서에서 관련 내용을 찾아줘`(`bd-000`) | `("web_search", "관련 내용")` | `("document_qa", q)` | `bd-000` 무회귀 + LLM 미호출 |
| R13 | `websocket 설정을 알려줘` | `raise RuntimeError` | 예외 전파 | 어절 전체 일치 실패 → NONE 경로의 기존 ① 계약 보존(리뷰 Iteration 3 M1) |
| R14 | `Google AI 에이전트 구조를 보여줘` | `(None, None)` | `(None, None)` | `Google` bare 매치, 스캔 범위(`AI`·`에이전트`·`구조를`)에 `TOPIC_HEAD`(`구조`) 있음 → 주제절 억제 → NONE 경로의 기존 ③ 계약 보존(리뷰 Iteration 3 M1) |
| R15 | `인터넷의 역사에서 중요한 사건을 알려줘` | `raise RuntimeError` | 예외 전파 | 조사 `의` 결합으로 어절 일치 실패 → NONE 경로의 기존 ① 계약 보존(리뷰 Iteration 3 M1) |
| R16 | `온라인 서비스 구조를 보여줘` | `("web_search", "온라인 서비스")` | `("web_search", "온라인 서비스")` | `온라인` bare 매치, 스캔 범위(`서비스`·`구조를`)에 `TOPIC_HEAD`(`구조`) 있음 → 주제절 억제 → NONE, LLM이 자유롭게 결정(강제 override 없음, 리뷰 Iteration 3 M1) |
| R17 | `온라인 문서에서 알려줘` | `("web_search", "온라인")` | `("document_qa", q)` | 순위 3 증거(스캔 범위 `문서에서`에 `TOPIC_HEAD` 없음) 성립하나 `문서에서`가 `DOCUMENT` 토큰이라 순위 2가 먼저 걸림 → DOCUMENT, LLM 미호출(호출 수 0 assert) — 우선순위 2 < 3의 실제 conflict 사례 |
| R18 | `웹 개발 방법 알려줘` | `raise RuntimeError` | 예외 전파 | `웹` bare 매치, 스캔 범위(`개발`·`방법`)에 `TOPIC_HEAD`(`방법`) 있음 → 주제절 억제 → NONE 경로의 기존 ① 계약 보존(리뷰 Iteration 3 M1) |
| R19 | `인터넷 회사 찾아가는 길` | `("web_search", "인터넷 회사")` | `("web_search", "인터넷 회사")` | `REQUEST_TAIL` 거짓(마지막 어절 `길`) → NONE, LLM이 자유롭게 결정 |
| R20 | `온라인 게임 확인 방법` | `(None, None)` | `(None, None)` | `REQUEST_TAIL` 거짓(마지막 어절 `방법`) → NONE 경로의 기존 ③ 계약 보존 |
| R21 | `웹검색 방법 알려줘` | `raise RuntimeError` | 예외 전파 | `WEB_FUSED` bare 매치, 스캔 범위(`방법`)에 `TOPIC_HEAD` 있음 → 주제절 억제 + `CHANNEL` 어절 불일치 → NONE 경로의 기존 ① 계약 보존(리뷰 Iteration 4 M1) |
| R22 | `웹 검색 기술을 보여줘` | `(None, None)` | `(None, None)` | `WEB_FUSED` bare 매치, 스캔 범위(`기술을`)에 `TOPIC_HEAD` 있음 → 주제절 억제, `CHANNEL`=`웹`도 같은 절에 `TOPIC_HEAD`가 있어 억제 → NONE 경로의 기존 ③ 계약 보존(리뷰 Iteration 4 M1) |
| R23 | `구글링 기능 알려줘` | `raise RuntimeError` | 예외 전파 | `WEB_FUSED` bare 매치, 스캔 범위(`기능`)에 `TOPIC_HEAD` 있음 → 주제절 억제 + `CHANNEL` 어절 불일치 → NONE 경로의 기존 ① 계약 보존(리뷰 Iteration 4 M1) |
| R24 | `web search API 구조 알려줘` | `(None, None)` | `(None, None)` | `WEB_FUSED` bare 매치, 스캔 범위(`API`·`구조`)에 `TOPIC_HEAD`(`구조`) 있음 → 주제절 억제, `CHANNEL`=`web`도 같은 절에 `TOPIC_HEAD`가 있어 억제 → NONE 경로의 기존 ③ 계약 보존(리뷰 Iteration 4 M1) |
| R25 | `인터넷에서 최신 소식 부탁해` | `raise RuntimeError` | 예외 전파 | `SOURCE_PARTICLE` 결합이 있어도 `REQUEST_TAIL` 거짓(`부탁해`) → NONE 경로의 기존 ① 계약 보존(리뷰 Iteration 4 M2 — 조사 결합은 독립 충분조건이 아님) |
| R26 | `웹에서 확인할 수 있을까` | `(None, None)` | `(None, None)` | `SOURCE_PARTICLE` 결합이 있어도 `REQUEST_TAIL` 거짓(`있을까`) → NONE 경로의 기존 ③ 계약 보존(리뷰 Iteration 4 M2) |
| R27 | `구글로 좀 찾아줄 수 있어?` | `("web_search", "구글 검색")` | `("web_search", "구글 검색")` | `REQUEST_TAIL` 거짓(`있어`) → NONE, LLM이 자유롭게 결정(강제 override 없음, 리뷰 Iteration 4 M2) |
| R28 | `웹검색으로 최신 환율 알려줘` | `raise RuntimeError` | `("web_search", extract(q))` | `WEB_FUSED`+`SOURCE_PARTICLE`=`으로` 결합 → `has_particle=True`로 스캔 없이 즉시 command 증거, 부정·인용에도 해당하지 않고 `REQUEST_TAIL` 참 → **LLM 예외에서도** web 유지(리뷰 Iteration 4 M1 — 강한 검색 명령은 그대로 WEB) |
| R29 | `구글링해서 알려줘` | `(None, None)` | `("web_search", extract(q))` | `WEB_FUSED`=`구글링`, 조사 없이 요청 동사가 바로 이어져 스캔 범위가 비어 `TOPIC_HEAD` 없음 → 동일 근거로 **no-tool**에서도 web 유지 |
| R30 | `웹검색으로 이번 학기 수업방식 알려줘` | `raise RuntimeError` | `("web_search", extract(q))` | `WEB_FUSED`+`SOURCE_PARTICLE`=`으로` 결합 → `has_particle=True`. 목적어 `수업방식`에 `TOPIC_HEAD`(`방식`)가 있어도 스캔하지 않으므로 억제되지 않음, `REQUEST_TAIL` 참 → **LLM 예외에서도** web 유지(리뷰 Iteration 4 M1 — 옛 `WEB_FUSED_TOPIC_GAP` 경계 사례를 거리 없는 규칙으로 재확인) |
| R31 | `웹검색 관련 API 구조 알려줘` | `raise RuntimeError` | 예외 전파 | `WEB_FUSED`=`웹검색` bare 매치, 전체 절 스캔(`관련`·`API`·`구조`)에서 `TOPIC_HEAD`(`구조`) 발견 → 주제절 억제. **옛 `WEB_FUSED_TOPIC_GAP=2`라면 이 `TOPIC_HEAD`는 매치로부터 3어절째라 억제되지 않고 WEB으로 오분류됐을 m+3 우회 사례** → NONE 경로의 기존 ① 계약 보존(리뷰 Iteration 5 M1 필수 반례) |
| R32 | `웹검색 관련 핵심 기능 알려줘` | `(None, None)` | `(None, None)` | 동일 근거(전체 절 스캔에서 `기능` 발견, m+3 우회 반례) → NONE 경로의 기존 ③ 계약 보존(리뷰 Iteration 5 M1) |
| R33 | `구글링 사용 관련 기술 알려줘` | `raise RuntimeError` | 예외 전파 | `WEB_FUSED`=`구글링` bare 매치, 전체 절 스캔(`사용`·`관련`·`기술`)에서 `TOPIC_HEAD`(`기술`) 발견 → 주제절 억제, m+3 우회 반례 → NONE 경로의 기존 ① 계약 보존(리뷰 Iteration 5 M1) |
| R34 | `질문:웹검색으로 최신 환율 알려줘` | `raise RuntimeError` | `("web_search", extract(q))` | `WEB_FUSED` 왼쪽 경계가 `:` — Unicode-aware 경계(`(?<!\w)`)로 인정, `SOURCE_PARTICLE`=`으로` 결합 → **LLM 예외에서도** web 유지(리뷰 Iteration 5 M1 — 왼쪽 경계 과소탐지 회귀 사례) |
| R35 | `(구글링해서 알려줘)` | `(None, None)` | `("web_search", extract(q))` | `WEB_FUSED` 왼쪽 경계가 `(` — Unicode-aware 경계로 인정, 조사 없이 요청 동사가 바로 이어져 command 증거 → **no-tool**에서도 web 유지(리뷰 Iteration 5 M1) |
| R36 | `googleapis 사용법을 알려줘` | `raise RuntimeError` | 예외 전파 | `googleapis`는 `google`+`apis`로 `WEB_FUSED`(접미사 `검색`/`서치`/`search` 없음)도 `CHANNEL` 어절 전체 일치도 성립하지 않아 웹 증거 자체가 없음 → NONE 경로의 기존 ① 계약 보존(리뷰 Iteration 5 M1 복합어 반례) |

- R1~R5, R9, R28~R30, R34~R35는 **LLM 실패(예외·no-tool·빈 query)에서도 최종 route가 `web_search`**임을 확인한다 = M3-REQ-004의 "명시 신호가 있는 질문의 최종 route는 모델 가용성과 무관"에 대한 통합 증거다.
- R6~R8, R12, R17은 stub의 반환값과 무관하게 `document_qa`가 나오고 `_llm_decide_tool` 호출 수가 **0**임을 확인한다.
- R10~R11, R13, R15, R18, R21, R23, R31, R33, R36은 NONE 경로에서 M2 계약(예외 전파)이 그대로 관찰됨을 확인한다. R14, R16, R19, R22, R24, R27, R32는 NONE 경로에서 LLM이 반환한 값이 그대로 통과함을 확인한다 — 확장된 채널+요청 규칙이 "이 질문은 절대 web이 될 수 없다"를 강제하는 것이 아니라 단지 **결정론적 override를 만들지 않을 뿐**임을 보여준다(모델이 실제로 web을 고르면 그 판단은 존중된다). R20, R25~R26은 NONE 경로에서 `(None, None)`/예외가 그대로 통과함을 확인한다.
- R13~R20은 §7.2.1/§7.2.3에서 새로 도입한 어절 전체 일치·주제절 억제 규칙·`REQUEST_TAIL`이 stub 없이 실제 classifier 경로에서도 작동함을 보이는 리뷰 Iteration 3 M1의 필수 증거다(R18은 주제절 억제, R19~R20은 `REQUEST_TAIL` 미충족이 원인이며 서로 다른 사례다). R21~R27은 리뷰 Iteration 4 M1(`WEB_FUSED` 주제절 억제)·M2(조사 결합의 비독립성)가 stub 없이 실제 classifier 경로에서도 NONE을 유지함을 보이는 필수 증거이고, R28~R30은 그 정밀도 강화가 강한 검색 명령의 recall을 깎지 않았음을 보인다. R31~R33은 리뷰 Iteration 5 M1이 지적한 **m+3 우회가 새 규칙(고정 거리 제거 + 전체 절 스캔)에서 실제로 닫혔음**을 stub 없이 실제 classifier 경로에서 확인하는 필수 증거이고, R34~R35는 **Unicode-aware 왼쪽 경계**가 문장부호·괄호 뒤 강한 명령을 정확히 포착함을, R36은 `googleapis` 같은 복합어 내부가 여전히 오탐되지 않음을 확인한다(리뷰 Iteration 5 M1).
- 이 행렬은 `ROUTING_SIGNAL_OVERRIDE=True`에서 실행하며, 같은 36건을 `False`로 한 번 더 돌려 **전부 LLM 결정 그대로**가 되는지도 확인한다(rollback 무회귀).
- 모델·네트워크를 쓰지 않으므로 CI에 포함한다(M3-REQ-009).

### 7.5 3회 반복 평가 (M3-REQ-005)

`evaluation/routing.py`에 `--runs N`(기본 1)을 추가한다. 단일 CLI 방식을 택한 이유: 세 실행의 모델 warm 상태가 같고, 사례별 변동을 같은 프로세스에서 정확히 대응시킬 수 있으며, 리포트 파일이 하나로 남아 비교가 단순하다. 실행은 **순차**다(NFR-002).

```python
def evaluate_routing_multi(cases, decide_tool, *, runs: int, measure_latency=True) -> dict:
    """runs번 evaluate_routing()을 순차 호출하고 per_run/aggregate/case_variation을 만든다.
    runs < 1이면 ValueError. runs == 1이면 기존 반환 dict에 run_count=1만 더한다."""
```

#### recall 분모 계약 (요구사항 §4.1과 동일한 단일 정의)

- 분모는 **`expected_route` 기준**이다: `document_qa` **61**건(category document_qa 51 + boundary 3 + unanswerable 7), `web_search` **15**건. 이는 현재 `evaluation/routing.py`가 `expected = case.expected_route.value`를 actual label로 `precision_recall_f1()`에 넘기는 구현과 같고, M2 승인값 `72.13% = 44/61`을 그대로 재현한다(§4.1 실측 확인).
- category `document_qa` 51은 **어떤 recall의 분모도 아니다.** 51을 쓰면 M2 승인값이 재현되지 않는다.
- per-run 지표: `accuracy`, `correct_count`, `document_route_recall`(= `confusion[document_qa][document_qa] / 61`), `document_route_correct`(분자 count), `web_search_recall`(= `confusion[web_search][web_search] / 15`), `web_search_correct`, `failures`, `latency_ms`. 리포트에는 `recall_denominators = {"document_qa": 61, "web_search": 15}`를 항상 명시해 소비자가 분모를 추측하지 않게 한다.

#### 집계와 gate

- `aggregate.<metric>.median`: 값 3개의 `statistics.median`. **지표별로 독립 계산**한다(요구사항 §4.1의 "중앙값이 accuracy 및 document route recall 기준을 만족"에 대응). recall/accuracy는 분자 count의 median도 함께 계산한다.
- `median_run_index`: accuracy 기준 중앙 run의 index(동률이면 최소 index). 사람 판독용이며 gate는 지표별 median으로 판정한다.
- `case_variation`: 사례별 3개 route와 `distinct_count`, `changed`. 변동 사례는 Markdown 표로도 출력한다.
- gate(§5.8의 상수와 동일):

| 조건 | 판정식 (분자 count 비교) | 근거 |
|---|---|---|
| 각 run의 web search recall | `web_search_correct == 15` (모든 run) | 요구사항 §4.1 |
| accuracy 중앙값 | `median(correct_count) >= 69` (분모 76) | 오류 ≤ 7건 |
| document route recall 중앙값 | `median(document_route_correct) >= 54` (분모 61) | 요구사항 §4.1 |

세 조건은 서로 모순되지 않는다: 오류가 최대 7건이고 web recall 보존 요구상 모든 오류가 `document_qa → web_search`이므로, accuracy 69/76을 만족하면 document route 분자는 최소 `61 - 7 = 54`가 된다. 즉 `54/61 (88.52%)`은 다른 두 gate에서 유도되는 값이며 독립 확인용 하한으로도 남겨 둔다.

판정은 반올림 백분율 상수(`0.8627` 등)를 쓰지 않는다. count 또는 `Fraction`으로만 비교하고 백분율은 표시용으로만 계산한다.

- 기존 `--runs` 미지정 호출의 리포트 키 집합과 의미는 완전히 유지된다. 기존 키 `precision_recall_f1.document_qa.recall`은 값·의미 모두 그대로이며(같은 분모 61), `document_route_recall`은 gate 판정용으로 이름을 명확히 한 **추가** 키다.

### 7.6 후보 ladder와 중단 조건

| 순서 | 후보 | 내용 | 비용 |
|---|---|---|---|
| 1 | `m3-p3a-signal-override` | 프롬프트 개정 + 명시 신호 우선 판정 | 프롬프트 + 순수 함수. **최소 진입 후보**(요구사항이 1·2순위의 외부 검증을 요구하므로 프롬프트 단독 변경은 단독 후보가 될 수 없다) |
| 2 | `m3-p3b-corpus-hint` | a + corpus topic hint | 함수 1개 추가, 프롬프트 길이 증가(라우팅 latency 소폭 증가 가능) |
| 3 | `m3-p3c-two-stage` | b + 2단계 LLM 판정(1차 후보 판단 → 2차 검증) | LLM 호출 2배. **a·b가 gate를 못 넘고 추가 latency가 정당화될 때만** |

앞 후보가 §4.1 gate를 만족하면 뒤 후보는 구현하지 않는다(Plan §4 Phase 3-3). 3회 실행 비용이 크므로 후보 선별은 먼저 `--runs 1`로 스크리닝하고, gate 판정용 공식 실행만 `--runs 3`으로 수행한다. 4회 iteration 후에도 미달이면 중단 보고서를 쓴다(Plan §3).

---

## 8. Phase 4 — Intent Classifier 효용 실험과 경로 결정

### 8.1 데이터 흐름 (paired blind)

```text
[1] 대상 선정: is_answer_eval_eligible()==True 인 29건 (골든셋 순서 유지)
[2] context 고정: 사례별 engine._retrieve_documents(question) 1회
        → context_text = "\n\n".join(doc.page_content)   ← RAGEngine.query()와 동일 규칙
        → sources = engine.format_sources(docs)
        → evaluation/reports/m3/m3-p4-intent-ab/context_snapshot.json  (Git 제외)
[3] variant A(intent): classify_intent(question) → get_template_by_intent(intent)
    variant B(default): DEFAULT_TEMPLATE 고정
    두 variant 모두 [2]의 동일 context로 engine.generate_answer(question, context, template)
[4] 자동 채점: v1/v2 assertion, v1/v2 abstention, source any-hit/recall  (variant별)
[5] blind worksheet: 사례마다 두 출력을 seed 기반으로 섞어 "출력 1 / 출력 2"로 제시
    정답 매핑은 별도 key 파일에 저장(worksheet에는 없음)
[6] 사람 검토 → worksheet 채워서 저장
[7] 집계: python -m evaluation.intent_ab aggregate → 결정 근거 리포트
```

`[2]`의 검색은 Phase 2에서 채택된 경로를 그대로 쓴다(따라서 Phase 4는 Phase 2 결정 이후 실행). 검색은 사례당 1회뿐이므로 두 variant가 **완전히 동일한 context**를 본다.

### 8.2 production seam 리팩터링 (로직 복제 금지)

`RAGEngine.query()`에서 다음 세 조각을 공개 메서드로 추출한다. 추출 후 `query()`는 이들을 호출만 하며 **반환 dict의 키·값·예외 처리는 그대로**다.

```python
def build_context(self, documents) -> str:            # "\n\n".join(page_content)
def format_sources(self, documents) -> list[dict]:    # {index, source, page, content[:200]}
def generate_answer(self, question: str, context: str, template_str: str) -> str:
    """PromptTemplate → llm → StrOutputParser 체인을 구성해 답변만 반환한다."""
```

`evaluation/intent_ab.py`는 이 세 메서드와 `_retrieve_documents()`만 사용한다. 프롬프트 조립·context 결합 규칙을 evaluator에 복제하지 않는다(M3-NFR-004). 회귀 안전장치로 "`query()`의 결과가 `generate_answer(build_context(...))` 조합과 동일"함을 fake LLM 통합 테스트로 고정한다.

### 8.3 worksheet 포맷과 blind key

- 순서 무작위화: `random.Random(f"{seed}:{case_id}")`로 variant 순서를 결정한다. `seed`(기본 `m3-intent-ab`)는 key 파일에 기록되어 완전 재현 가능하다(M3-NFR-001).
- worksheet(`intent_ab_<ts>_worksheet.md`)에는 variant 이름·intent 라벨·템플릿 종류를 **일절 표시하지 않는다**. 자동 점수도 "출력 1 / 출력 2" 라벨로만 표기해 정체를 노출하지 않는다.
- 답변 본문은 `evaluation/answers.py::_fence_for()`와 동일한 동적 fence 규칙으로 감싼다(코드블록 포함 답변 대응).
- 사례 블록의 기계 판독 가능한 채점 슬롯:

````markdown
## dq-rag-001
**질문**: …
**출력 1**:
```
…
```
**출력 2**:
```
…
```
<!-- 아래 5줄만 수정하세요. 값은 0 또는 1, 선호는 1/2/tie -->
- 출력1_형식적합성: _
- 출력1_핵심사실보존: _
- 출력2_형식적합성: _
- 출력2_핵심사실보존: _
- 선호: _
- 검토메모:
````

- key 파일(`intent_ab_<ts>_key.json`, Git 제외): `{"seed":…, "cases":[{"id":…, "slot1":"intent", "slot2":"default", "intent_label":"yesno"}]}`.
- 파서는 값이 비었거나(`_`) 허용값이 아니면 그 사례를 `incomplete`로 표시하고 집계에서 제외하되 **개수를 보고**한다. 부분 채점을 성공으로 취급하지 않는다.

### 8.4 집계와 결정 규칙 (요구사항 §4.2)

- `preferred_intent`, `preferred_default`, `tie`, `incomplete` 카운트.
- `margin_pp = (preferred_intent - preferred_default) / N_scored * 100`.
- 축별 합계: `형식적합성_intent/ default`, `핵심사실보존_intent/ default`.
- 자동 지표 병기: variant별 assertion(v1/v2), abstention(v1/v2), source any-hit/recall.
- 결정:

| 조건 | 결론 |
|---|---|
| `margin_pp >= 20.0` **AND** intent variant에서 assertion/abstention/source gate 회귀 없음 | **유지 가능** → §8.5 |
| 그 외(동률, 미달, `incomplete > 0`으로 판단 불가) | **입증되지 않음** → §8.6 (보수적 처리) |

N=29 기준 `margin_pp >= 20.0`은 순선호 차 ≥ 5.8, 즉 **≥6건**을 뜻한다. 집계는 원시 count로 계산하고 표시만 반올림한다.

### 8.5 유지 분기 설계

유지하려면 intent accuracy **22/29 이상**을 달성해야 한다. 재학습 없이 가능한 순서로 실험한다.

1. **confidence floor**: `classify_intent_with_confidence()`의 confidence가 `INTENT_CONFIDENCE_FLOOR` 미만이면 `other`로 강등. `other`/`uncertain`이 같은 DEFAULT 템플릿을 쓰므로 답변 경로 위험이 낮다. 임계값은 골든 29건이 아니라 `training/intent_classifier/datasets/dev.jsonl`로 정한다(평가셋 누수 금지).
2. **yesno 보강**: M2 검토상 `yesno`가 comparison/explanation/other로 분산됐다. 학습 데이터에 yesno/uncertain 도메인 예시를 추가해 재학습하되, **골든셋 29건의 질문 문자열을 그대로 넣지 않는다**(패러프레이즈만). 누수 위험을 결정 기록에 명시한다.
3. gate 달성 후 추가 학습 확대는 하지 않는다(Plan §4 Phase 4-5).

`training/intent_classifier/`의 산출물 경로·config schema는 변경하지 않는다.

### 8.6 단순화 분기 설계 (공개 계약 보존)

- `ANSWER_TEMPLATE_MODE="default"`이면 `RAGEngine.query()`가 `classify_intent()`를 호출하지 않고 `DEFAULT_TEMPLATE`을 쓴다.
- **응답 계약(M3-REQ-009) 보존**: `query()`/`/rag` 응답은 계속 `intent` 키를 포함하고 값은 문자열 `"other"`다. 새 값(`"disabled"` 등)을 만들지 않는다 — `Intent` enum과 골든셋 라벨 집합에 이미 존재하는 값이어야 소비자와 evaluator가 깨지지 않는다. `QueryResponse.intent` 필드도 그대로 둔다. 프런트엔드는 `intent`를 사용하지 않음을 확인했다(`web/static`, `web/templates` 전수 grep 결과 참조 없음).
- Answer 리포트의 `intent.accuracy`는 0%로 보고하지 않고 `null` + `intent_excluded_reason="ANSWER_TEMPLATE_MODE=default (classifier 비활성)"`로 표기한다. 잘못된 0%가 회귀로 오독되는 것을 막는다.
- 모델 artifact(`models/intent_classifier/`)와 `training/`은 **삭제하지 않는다**. 정리는 후속 승인 범위(요구사항 §4.2)이며 이 마일스톤에서는 "비활성 + 보존"이 결정 사항이다.
- `intent_classifier.py`는 import 가능 상태를 유지한다(다른 코드가 import만 해도 실패하면 안 됨, M3-NFR-004).

### 8.7 실행 비용 추정 (사용자 Gate 입력)

29건 × 2 variant 생성 + 29회 검색. M2 Answer 평균 55.5초(검색 포함)이고 Phase 2 이후 검색이 ~2.5초로 줄면 생성만 ≈ 43초/건으로 추정. 총 ≈ 29 × 2 × 43초 + 29 × 2.5초 ≈ **43분**. 사용자 승인 항목이다.

---

## 9. Phase 5 — 조건부 한국어 BM25 tokenizer 실험

진입 조건: Phase 2 이후 Retrieval floor 안정 + 사용자의 추가 실험 비용 승인. 진입하지 않아도 M3는 완료할 수 있다.

### 9.1 tokenizer 경계

```python
# src/simple_qna_rag/text_tokenizers.py
Tokenizer = Callable[[str], list[str]]
def whitespace_tokenizer(text: str) -> list[str]: return text.split()      # 현행과 완전 동일
def char_ngram_tokenizer(text: str, n: int = 2) -> list[str]: ...          # 표준 라이브러리만
def bge_subword_tokenizer(text: str) -> list[str]: ...                     # 이미 설치된 transformers 사용
def get_tokenizer(name: str) -> Tokenizer: ...                             # 미지원 이름은 ValueError
```

- 기본값은 `whitespace`이며 `_create_bm25_retriever()`는 `get_tokenizer(BM25_TOKENIZER)`를 문서·질의 양쪽에 동일하게 적용한다. **기본 경로의 동작은 현행과 byte 단위로 같다.**
- `bge_subword_tokenizer`는 `transformers`를 **함수 내부에서 지연 import**하고, 실패하면 명확한 `RuntimeError`를 던진다. 모듈 import 자체는 어떤 경우에도 실패하지 않는다(M3-NFR-004). 새 필수 dependency는 추가하지 않는다(`transformers`는 이미 `requirements.txt`에 있고 BGE-M3 모델은 이미 로컬 캐시에 있다).
- 미채택 시 `text_tokenizers.py`에서 실험 tokenizer를 제거하거나, 유지하더라도 기본 경로·필수 dependency에 어떤 흔적도 남기지 않는다(M3-REQ-008). 판단 기준: 채택 실패 시 **모듈 자체를 삭제하고 `_create_bm25_retriever()`를 원상 복구**한다.

### 9.2 오프라인 A/B 하네스

`evaluation/experiments/bm25_tokenizer.py`:

```text
python -m evaluation.experiments.bm25_tokenizer \
  --dataset evaluation/datasets/golden.jsonl \
  --tokenizers whitespace,char2gram,bge-subword \
  --repeats 3 \
  --output evaluation/reports/m3/m3-p5-bm25-offline/bm25_only
```

- `index.pkl`의 docstore만 로드해 BM25 순위를 만든다. **임베딩 모델도 Ollama도 필요 없다** → 빠르고 결정론적이며 CI에서도 (index가 있으면) 실행 가능하다.
- 지표는 `evaluation/metrics.py`를 재사용해 Recall@1/3/5/10, MRR@10, nDCG@10을 42건에 대해 계산한다.
- 시간 측정: `_create_bm25_retriever()` 상당 구간을 `time.perf_counter()`로 3회 측정한 median. 초기화 시간은 (a) BM25 빌드 시간, (b) `RAGEngine.initialize()` 전체 시간 두 값을 모두 보고하고 gate는 (b)에 적용한다.
- **메모리 측정(m2 반영)**: `tracemalloc`은 Python allocator만 추적해 tokenizer/`transformers`/native 배열의 실제 사용량을 놓치므로 **주 판정값으로 쓰지 않는다.** 주 판정값은 동일 프로세스의 **RSS peak**다.

| 항목 | 방법 |
|---|---|
| 주 판정값 | `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` 기반 **RSS peak**. Linux는 KiB, macOS는 byte 단위이므로 `sys.platform`으로 정규화해 바이트로 환산하고 리포트에 `rss_unit_source`를 남긴다. 표준 라이브러리만 쓰며 `psutil` 같은 새 dependency를 추가하지 않는다. |
| 측정 창 | tokenizer별로 **새 subprocess 1개**를 띄워 (i) import 직후, (ii) docstore 로드 후, (iii) BM25 빌드 후, (iv) `initialize()` 완료 후 4개 지점에서 `ru_maxrss`를 샘플링한다. peak는 단조 증가하므로 subprocess를 분리해야 tokenizer별 값이 서로 오염되지 않는다. |
| 반복·집계 | tokenizer별 subprocess 3회 실행, 각 실행의 (iv) 시점 peak를 취해 **median**을 판정값으로 한다. |
| 판정식 | `median_peak(candidate) <= median_peak(whitespace) * 1.20` (요구사항 §3.2의 20%) |
| 진단값 | `tracemalloc` peak(같은 구간)와 `sys.getsizeof` 기반 BM25 자료구조 크기를 **참고 수치로 병기**한다. gate 판정에는 쓰지 않는다. |
| baseline | 같은 세션·같은 host에서 `whitespace` tokenizer로 동일 절차를 실행한 값을 baseline으로 쓴다. M2 리포트의 값을 재활용하지 않는다(측정 방법이 다르다). |

RSS는 OS/할당자 특성상 소량의 실행 간 변동이 있으므로 3회 median과 함께 min/max를 모두 리포트에 남기고, 판정이 20% 경계에서 갈리면 "판정 불가"로 표시해 사용자 결정을 요청한다.
- 전체 hybrid 비교는 `SIMPLE_QNA_RAG_BM25_TOKENIZER=<name> python -m evaluation.retrieval …`로 별도 실행한다(BM25-only와 분리 보고, Plan §4 Phase 5-3).

### 9.3 채택 gate (요구사항 §3.2)

| 조건 | 임계 |
|---|---|
| Recall@10 **또는** nDCG@10 개선 | ≥ +1.00%p (Recall@10 ≥ 98.62%, 또는 nDCG@10 ≥ 96.43%) |
| Retrieval floor 전체 | §4.1 모두 만족 |
| 새 필수 native runtime / 외부 서비스 / 배포 크기 급증 | 없음 |
| 초기화 시간 증가(`initialize()` 전체) | ≤ 20% (3회 median) |
| 메모리 증가(RSS peak 주 판정, `tracemalloc` 병기) | ≤ 20% (3회 median, §9.2) |

Recall@10이 이미 97.62%로 천장에 가까우므로 실질적 레버는 nDCG@10이다. 미달 시 실험 결과만 기록하고 현행 tokenizer를 유지한다.

### 9.4 fixture

조사·어미(`부동산의`, `부동산에서`, `부동산을`), 복합명사(`주택담보대출`, `벡터스토어`), 영문/숫자 혼합(`LTV 40%`, `bge-m3`, `chunk_size=1000`), 한영 혼용(`RAG 시스템`)을 포함한 결정론적 tokenizer 단위 테스트를 만든다. 모델 없이 실행되어야 한다.

---

## 10. Phase 6 — 통합 회귀, live 평가와 최종 승인

1. clean 환경에서 정적 회귀 전체 실행(§12.4).
2. 채택된 변경만 활성화한 `m3-final` 설정으로 통합 실행:
   - Retrieval 42건 (동일 process warm-up `--warmup-cases 3` 후 단독)
   - Routing 76건 × 3회 (`--runs 3`)
   - Answer 29건 (v1/v2 병기, 공식 profile `v2` + `--expect-variants-sha256`)
   - 통합 baseline
3. `evaluation.compare`의 gate 판정 + 같은 모듈로 M2 기준선 대비 per-case 변화 산출.
4. 사람 검토: source relevance, assertion/abstention worksheet, Intent A/B worksheet.
5. 요구사항 추적표(§14) 최종본, 채택/기각 후보, 알려진 한계, 실행 비용, 잔여 위험 요약.
6. **사용자 승인 후에만** `evaluation/baselines/m3_initial.{json,md}`를 새로 만들고 `docs/Roadmap.md` 상태를 갱신한다. 승인 전에는 baseline 파일을 만들지 않는다(M3-REQ-010).

M3 baseline에 포함하는 것: 집계 수치, 비민감 실패 taxonomy, 실행 metadata, fingerprint, 승인 시각, 원본 local report 식별 경로. 포함하지 않는 것: 질문·답변 원문, context.

`baseline.py`에 `--routing-runs N`(기본 1), `--warmup-cases N`(기본 0), `--candidate-id`를 추가하고, Routing 단계가 `evaluate_routing_multi()`를 호출하도록 한다. `--warmup-cases`는 Retrieval·Answer 단계에 전달되며 baseline은 단일 프로세스에서 모든 단계를 실행하므로 warm-up 효과가 그대로 이어진다. 기존 옵션·단계 순서·실패 격리 정책은 변경하지 않는다.

---

## 11. 파일·모듈·API 단위 설계와 공개 계약 호환성

### 11.1 신규 파일

| 경로 | 성격 | 주요 공개 심볼 | 모델 의존 |
|---|---|---|---|
| `src/simple_qna_rag/vector_index.py` | 제품 | `StoredVectorIndex`, `VectorLookupError`, `VectorIndexValidationError`, `VectorIndexStats` | 없음(duck-typed) |
| `src/simple_qna_rag/routing_signals.py` | 제품 | `ExplicitSignal`, `SignalDecision`, `classify_explicit_signal()`, `classify_explicit_signal_detail()`, `build_corpus_topic_hint()`, `is_loopback_endpoint()`, 두 command grammar, 검색 행위 술어 집합, 인용·부정 전처리, DOCUMENT 신호 | 없음 |
| `src/simple_qna_rag/text_tokenizers.py` | 제품(Phase 5 조건부) | `get_tokenizer()`, 3개 tokenizer | 지연 import |
| `evaluation/answer_rules.py` | 평가(순수) | §5.1 목록, `VariantTableError` | 없음 |
| `evaluation/answer_variants.json` | 평가 데이터 | — | — |
| `evaluation/fingerprint.py` | 평가 CLI(Phase 0) | `collect_fingerprint()`, `compare_with_baseline()`, `main()` | 없음 |
| `evaluation/rescore.py` | 평가 CLI | `rescore_report()`, `main()` | 없음 |
| `evaluation/compare.py` | 평가 CLI + gate 순수 함수 | `compare_reports()`, `M3_GATES`, `evaluate_gates(payload) -> dict`, `main()` | 없음 |
| `evaluation/intent_ab.py` | 평가 CLI | `run_experiment()`, `write_blind_worksheet()`, `aggregate_worksheet()`, `main()` | 실행 시에만 |
| `evaluation/experiments/__init__.py`, `bm25_tokenizer.py` | 평가 CLI(조건부) | `run_bm25_ab()`, `main()` | 없음 |
| `scripts/check_markdown_links.py` | 회귀 gate 도구(§4.5) | `enumerate_markdown_files()`(tracked ∪ untracked-nonignored, 중복 제거·stable sort), `collect_links()`, `check_paths()`, `main()` | 없음 |

모든 평가 모듈은 **import 시점에 `get_rag_engine()`을 호출하지 않는다**(기존 M2-NFR-003 규칙 유지).

`evaluation/experiments/`는 `evaluation/` 아래 새 하위 디렉터리다. Phase 5에 실제로 진입해 채택하는 경우에만 [Repository Structure](../../architecture/Repository_Structure.md)의 디렉터리 표에 한 줄("실험용 오프라인 A/B 하네스")을 추가한다. 기각하면 디렉터리째 삭제하므로 문서 변경도 없다.

### 11.2 변경 파일의 함수 단위 diff 요약

| 파일 | 함수 | 변경 |
|---|---|---|
| `rag_engine.py` | `RetrievalTrace` | 필드 `counters`, `notes` 추가(기본 빈 컨테이너) |
| | `_bump()`, `_note()` | 신규 module-level null-safe 계측 helper(§6.4) |
| | `_log_mmr_fallback_once()` | 신규(private), trace와 독립인 질의 단위 경고 억제 |
| | `initialize()` | `MMR_VECTOR_SOURCE=="stored"`일 때 `StoredVectorIndex.build()` 호출, 실패 시 강등(`mmr_vector_status`) |
| | `_retrieve_documents()` | `query_embed` 단계 추가, dense를 `similarity_search_by_vector`로 치환, MMR에 query 벡터·trace 전달 |
| | `_apply_mmr()` | 키워드 인자 `query_embedding`, `trace` 추가, 후보 벡터 소스 분기 |
| | `_candidate_vectors()` | 신규(private) |
| | `build_context()`, `format_sources()`, `generate_answer()` | `query()`에서 추출한 공개 메서드(신규) |
| | `query()` | 위 세 메서드 호출로 재구성, `ANSWER_TEMPLATE_MODE` 분기. **반환 dict 불변** |
| | `_create_bm25_retriever()` | tokenizer 주입(Phase 5) |
| `agent.py` | `SYSTEM_PROMPT` | 개정(§7.3) |
| | `_llm_decide_tool()` | 기존 `_decide_tool()` 본문을 이름만 바꿔 이동 |
| | `_decide_tool()` | 명시 신호 우선 판정(§7.4). 시그니처 불변, NONE 경로의 예외·`(None,None)` 계약 불변 |
| | `route_query()` | **변경 없음** |
| `query_router.py` | `extract_web_search_query()` | 신규 순수 함수(기존 로직 이동) |
| | `route_query()` | 위 함수 호출로 치환. 동작 동일 |
| `routing_signals.py` | `is_loopback_endpoint()` | 신규 순수 함수(§7.2) |
| `evaluation/answers.py` | `evaluate_answers()` | v2 채점 병기, `--candidate-id`/`--warmup-cases`/`--evaluator-profile`/`--expect-variants-sha256` 전달, 새 필드 |
| | `_render_answers_markdown()` | v1/v2 섹션 분리 |
| `evaluation/retrieval.py` | `evaluate_retrieval()` | `warmup_cases` 파라미터, `mmr_instrumentation` 집계, candidate/warmup block |
| | `main()` | `--warmup-cases`, `--candidate-id` |
| `evaluation/routing.py` | `evaluate_routing_multi()` | 신규 |
| | `main()` | `--runs`, `--candidate-id` |
| `evaluation/baseline.py` | `run_baseline()` | `routing_runs`, `warmup_cases`, `candidate_id`, `gate_evaluation` |
| `evaluation/reporting.py` | `_active_retrieval_config()` | `mmr_vector_source`, `bm25_tokenizer` 추가 |
| | `build_candidate_metadata()`, `build_warmup_metadata()` | 신규 |
| `config.py` | — | §3.4의 상수 추가 |

### 11.3 공개 계약 호환성 (M3-REQ-009)

| 계약 | M2 | M3 | 보증 방법 |
|---|---|---|---|
| `RAGEngine.query()` 반환 | `{answer, sources, success, intent}` | 동일(키·타입·에러 시 `intent="error"`) | 기존 통합 테스트 유지 + 신규 characterization 테스트 |
| `agent.route_query()` 반환 | `{answer, sources, success, search_type[, intent]}` | 동일 | `tests/integration/test_agent.py` 무수정 통과 |
| `POST /rag` 응답 | `QueryResponse{answer, sources, success, search_type, intent?}` | 동일(모델 정의 무변경) | FastAPI 스키마 테스트 |
| `sources[]` 항목 | `{index, source, page, content}` | 동일 | `format_sources()` 추출 시 동작 동일 |
| CLI entry points | `simple-qna-rag-{web,query,index}` | 동일 | `tests/integration/test_cli_entrypoints.py` |
| evaluation CLI | 기존 옵션 | **추가만**(`--candidate-id`, `--runs`) | `--help` 스냅샷 테스트 |
| 리포트 JSON | M2 키 집합 | **상위집합** | 기존 evaluator 테스트 무수정 통과 |
| vectorstore | `runtime/vectorstore/` | **읽기 전용**, 재생성·덮어쓰기 없음 | 통합 테스트에서 index 파일 mtime/해시 불변 확인 |
| CI | live 미요구 | 동일 | 새 테스트는 모두 모델/네트워크 없이 실행 |

---

## 12. 테스트 설계와 실행 명령

### 12.1 신규/확장 단위 테스트 (`tests/unit/`)

| 파일 | 대상 | 핵심 케이스 |
|---|---|---|
| `test_answer_rules.py` | §5.2~5.6 | 정규화 10단계 각각, 반례 C1~C10, 8개 FN 표면형 → 일치, 3개 abstention → True, 반례 A5~A8 → False, `rules_fingerprint` 안정성, 변형 표 fail-closed(부재·schema 오류·해시 불일치 → `VariantTableError`) |
| `test_vector_index.py` | `StoredVectorIndex` | fake vectorstore로 V1~V6 실패 모드 각각, `row_for()` 미등록 문서 → `VectorLookupError`, `vectors_for()` 순서 보존, dtype float64 |
| `test_routing_signals.py` | `classify_explicit_signal()`, `is_loopback_endpoint()` | 규범 §7.2의 골든 76건 exact set **WEB 8 / DOCUMENT 12 / NONE 56**, WEB·DOCUMENT 오탐 0, 두 command grammar 양성, 일반 응답 술어·관형절·검색 주제·인용·부정·Unicode 경계 표. `TOPIC_HEAD`, 거리 상수, `has_particle` fast path가 구현에 없음을 검증한다. |
| `test_evaluation_gates.py` | `evaluation.compare.evaluate_gates()` | count 경계값(accuracy 68/69/70 of 76, document route 53/54/55 of 61, web 14/15 of 15), `Fraction` 비교가 반올림 float와 달라지는 경계, 절대+감소율 동시 조건, `fallback_case_count>0` → `pass=null`, `warmup.performed=false` → `pass=null`, `official=false` → v2 gate 차단 |
| `test_rag_engine_trace.py` | §6.4 null-safe helper | `_bump()`/`_note()`에 `trace=None` 전달 시 무동작·무예외, trace 제공 시 누적, counters 초기 상태 |
| `test_text_tokenizers.py` | Phase 5 | §9.4 fixture, 미지원 이름 → `ValueError`, 지연 import 실패 메시지 |
| `test_config.py`(확장) | §3.4 | 환경변수 파싱, 잘못된 열거값 → `ValueError`, candidate ID 정규식(§3.1) 양·음성 사례 |
| `test_check_markdown_links.py` | §4.5 | 정상/깨진 상대 링크, 정상/깨진 anchor, 코드블록·인라인 코드 무시, 외부 URL 무시, 중복 heading `-1` 접미사, 한글 heading anchor, repo 밖 경로 → 실패, exit code 0/1/2 + **열거 계약 E1~E6**(임시 `git init` repo에서 tracked 깨짐 → exit 1, **미추적 신규 Markdown 깨짐 → exit 1**, gitignore 대상 → exit 0, `--json`의 `files/tracked/untracked`, index-only 삭제 파일 skip + warning, 재실행 결정론) |

### 12.2 신규/확장 통합 테스트 (`tests/integration/`) — 규범

라우팅 통합 테스트는 §7.4의 신호 stub 12칸, 단순화 Cycle 1 실제 classifier S1~S12, NONE의 기존 `route_query()` 폴백 4경로, WEB 실행 실패 시 document QA 재시도만 구현한다. 이전 R1~R36, `TOPIC_HEAD`, `REQUEST_TAIL`, `has_particle` 관련 테스트는 구현하지 않는다. 그 밖의 통합 테스트 계약은 아래 표의 라우팅 행을 제외하고 유지한다.

| 파일 | 검증 |
|---|---|
| `test_agent_routing_policy.py` | 신호 stub 12칸 + S1~S12 + NONE 폴백 4경로 + WEB 실패 재시도 |

### 12.2-L 이전 Iteration 1~6 통합 테스트 표 — 라우팅 행은 비규범

아래 표에서 `test_agent_routing_policy.py` 행은 감사 기록일 뿐이다. 다른 행은 현재 계약을 유지한다.

| 파일 | 검증 |
|---|---|
| `test_evaluation_retrieval.py`(확장) | ① 카운팅 fake embeddings로 **질문당 `embed_query` 정확히 1회**, 후보 본문 임베딩 **0회** ② trace 유무에 따른 결과 리스트 동일성 ③ stored/legacy 두 경로의 문서 순서 동일 ④ §6.5 폴백 6칸 행렬(`trace=None`/제공 × miss·dimension·non-finite)에서 결과가 legacy와 동일하고 예외가 없음 ⑤ `RAGEngine.query()`(trace 미전달)에서 폴백 시 질의가 성공함 ⑥ 초기화 강등(`build()` 실패) 후 `trace=None` 안전성 ⑦ `mmr_instrumentation` 집계 ⑧ `--warmup-cases 2`에서 warm-up 사례가 집계에서 제외되고 `measured_case_count`가 유지되며 같은 engine 인스턴스가 쓰였음 |
| `test_agent_routing_policy.py`(신규, 모델 불필요) | ① §7.4의 stub 신호 12칸 행렬 전부(WEB/DOCUMENT/NONE × LLM 정상·no-tool·빈 query·예외, flag off, 신호 함수 예외), DOCUMENT에서 `_llm_decide_tool` 호출 횟수 0 ② §7.4의 **실제 classifier 행렬 R1~R36**(classifier stub 없이 실제 한국어 문자열 → `classify_explicit_signal()` → `_decide_tool()`, `_llm_decide_tool`만 stub): 확장 positive grammar가 LLM 예외/no-tool/빈 query에서도 `web_search` 유지, 부정·인용·주제절 입력은 `document_qa` + LLM 호출 0, 단독 언급은 NONE 경로의 `(None, None)`·예외 전파 보존, **`websocket`/`Google AI`/`인터넷의 역사`/`온라인 서비스 구조` 등 어절 일치·주제절 억제 기반 채널 주제어 오인 방지 fixture(R13~R16)가 NONE 경로 계약을 보존**, **`온라인 문서에서 알려줘`(R17)가 순위 3 증거에도 불구하고 DOCUMENT + LLM 호출 0**, **`웹 개발 방법 알려줘`(R18)가 주제절 억제로, `인터넷 회사 찾아가는 길`/`온라인 게임 확인 방법`(R19~R20)이 `REQUEST_TAIL` 미충족으로 NONE 경로 계약을 보존**, **`웹검색 방법 알려줘`/`웹 검색 기술을 보여줘`/`구글링 기능 알려줘`/`web search API 구조 알려줘`(R21~R24)가 `WEB_FUSED` 주제절 억제로 NONE 경로 계약을 보존**, **`인터넷에서 최신 소식 부탁해`/`웹에서 확인할 수 있을까`/`구글로 좀 찾아줄 수 있어?`(R25~R27)가 조사 결합만으로는 WEB이 되지 않고 `REQUEST_TAIL` 미충족으로 NONE 경로 계약을 보존**(리뷰 Iteration 4 M1·M2), **`웹검색으로 최신 환율 알려줘`/`구글링해서 알려줘`/`웹검색으로 이번 학기 수업방식 알려줘`(R28~R30)가 LLM 예외·no-tool에서도 `web_search`를 유지**, **`웹검색 관련 API 구조 알려줘`/`웹검색 관련 핵심 기능 알려줘`/`구글링 사용 관련 기술 알려줘`(R31~R33)가 옛 고정 거리 상수라면 놓쳤을 m+3 우회를 새 전체 절 스캔으로 여전히 NONE 유지함을 실측**, **`질문:웹검색으로 최신 환율 알려줘`/`(구글링해서 알려줘)`(R34~R35)가 Unicode-aware 왼쪽 경계로 문장부호·괄호 뒤 강한 명령에서도 `web_search`를 유지**, **`googleapis 사용법을 알려줘`(R36)가 복합어 내부 오탐 없이 NONE을 유지**(리뷰 Iteration 5 M1), 같은 36건을 flag off로 재실행 시 전부 LLM 결정 그대로 ③ `route_query()` 4개 폴백 경로 무회귀(NONE) ④ WEB + 웹 검색 실행 실패 시 ④ 재시도 보존 |
| `test_evaluation_routing.py`(확장) | `--runs 3` 집계, per-metric median, 분자 count median, `recall_denominators == {"document_qa":61,"web_search":15}`, `case_variation`, runs=1 하위 호환, 기존 `precision_recall_f1` 값 불변 |
| `test_evaluation_answers.py`(확장) | v1/v2 병기 필드, 최상위 v1 의미 불변, worksheet 구조, 공식 profile에서 변형 표 부재/해시 불일치 → exit 2, `--evaluator-profile v2-no-variants` → `official=false` |
| `test_evaluation_rescore.py`(신규) | fixture 리포트 → v1/v2 재채점, 원본 파일 미수정, `answer` 없는 리포트 → exit 2 |
| `test_evaluation_compare.py`(신규) | fingerprint 불일치 → `comparable=false`/exit 3, per-case 순위 diff |
| `test_evaluation_fingerprint.py`(신규) | 기준선 비교 일치/불일치, exit code |
| `test_intent_ab.py`(신규, fake engine) | context 1회 고정, 두 variant 동일 context, seed 재현성, worksheet에 variant 정체 미노출, 파서의 `incomplete` 처리 |
| `test_rag_engine_seam.py`(신규, fake LLM) | `query()` == `generate_answer(build_context(...))` 조합, `ANSWER_TEMPLATE_MODE=default`에서 `intent=="other"` 유지 |

### 12.3 live 테스트 (opt-in, CI 제외)

- 기존 `tests/integration/test_agent_routing.py`(RUN_LIVE_LLM_TESTS=1)는 유지하되, 정책 변경 후 `MIN_ACCURACY`는 그대로 두고 실패 목록만 확인한다(임계 하향 금지).
- 새 live 테스트는 추가하지 않는다. live 측정은 evaluator CLI로만 수행한다(M3-REQ-009).

### 12.4 정확한 실행 명령

**Phase 0~6 공통 정적 회귀(모든 Phase gate에서 동일하게 실행)**:

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
pytest -q
npm test
python scripts/check_markdown_links.py
git diff --check
```

Phase 0 추가:

```bash
python -m evaluation.fingerprint --dataset evaluation/datasets/golden.jsonl \
  --baseline evaluation/baselines/m2_initial.json
python scripts/check_markdown_links.py --help
pytest -q tests/unit/test_check_markdown_links.py

# link 검사 대상 파일 열거의 fail-open 감시(§4.5, §4.6)
git ls-files -- '*.md' '*.markdown' | wc -l                                  # tracked 수
git ls-files --others --exclude-standard -- '*.md' '*.markdown' | wc -l      # untracked(non-ignored) 수
python scripts/check_markdown_links.py --json                                # files/tracked/untracked/links 집계
# 위 명령들의 출력을 §4.6의 로그 경로에 저장하고 SHA-256을 Phase 0 리포트에 기록한다.
# 검사 파일 수 == tracked + untracked 이고 M3 문서 디렉터리의 모든 Markdown이 목록에 있어야 한다.
```

Phase 1:

```bash
pytest -q tests/unit/test_answer_rules.py tests/unit/test_evaluation_gates.py \
         tests/unit/test_evaluation_metrics.py \
         tests/integration/test_evaluation_answers.py \
         tests/integration/test_evaluation_rescore.py \
         tests/integration/test_evaluation_compare.py
python -m evaluation.answers --help
python -m evaluation.rescore --help
python -m evaluation.rescore \
  --report evaluation/reports/m2_full/answers/answers_20260804T145621300637Z.json \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports/m3/m3-p1-evaluator-v2/rescore
pytest -q
python scripts/check_markdown_links.py
```

Phase 2 (동일 process warm-up 포함 단독 실행 — 별도 warm-up 프로세스를 띄우지 않는다):

```bash
pytest -q tests/unit tests/integration/test_evaluation_retrieval.py
python -m evaluation.retrieval --help
SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE=stored python -m evaluation.retrieval \
  --dataset evaluation/datasets/golden.jsonl \
  --warmup-cases 3 \
  --output evaluation/reports/m3/m3-p2a-stored-vector/retrieval \
  --candidate-id m3-p2a-stored-vector
python -m evaluation.compare --kind retrieval \
  --baseline evaluation/baselines/m2_initial.json \
  --candidate evaluation/reports/m3/m3-p2a-stored-vector/retrieval/retrieval_<ts>.json \
  --output evaluation/reports/m3/m3-p2a-stored-vector/compare
python scripts/check_markdown_links.py
```

Phase 3:

```bash
pytest -q tests/unit/test_query_router.py tests/unit/test_routing_signals.py \
         tests/integration/test_agent_routing_policy.py \
         tests/integration/test_evaluation_routing.py
python -m evaluation.routing --help
SIMPLE_QNA_RAG_ROUTING_SIGNAL_OVERRIDE=1 RUN_LIVE_LLM_TESTS=1 \
python -m evaluation.routing --dataset evaluation/datasets/golden.jsonl --mode live \
  --runs 3 --candidate-id m3-p3a-signal-override \
  --output evaluation/reports/m3/m3-p3a-signal-override/routing
pytest -q
python scripts/check_markdown_links.py
```

Phase 4:

```bash
pytest -q tests/unit tests/integration/test_intent_ab.py \
         tests/integration/test_rag_engine_seam.py \
         tests/integration/test_evaluation_answers.py tests/integration/test_agent.py
python -m evaluation.intent_ab --help
RUN_LIVE_LLM_TESTS=1 python -m evaluation.intent_ab run \
  --dataset evaluation/datasets/golden.jsonl \
  --warmup-cases 2 \
  --output evaluation/reports/m3/m3-p4-intent-ab
# 사람 검토 후
python -m evaluation.intent_ab aggregate \
  --worksheet evaluation/reports/m3/m3-p4-intent-ab/intent_ab_<ts>_worksheet.md \
  --key evaluation/reports/m3/m3-p4-intent-ab/intent_ab_<ts>_key.json \
  --output evaluation/reports/m3/m3-p4-intent-ab
```

Phase 5:

```bash
pytest -q tests/unit tests/integration/test_evaluation_retrieval.py
python -m evaluation.experiments.bm25_tokenizer \
  --dataset evaluation/datasets/golden.jsonl \
  --tokenizers whitespace,char2gram,bge-subword \
  --repeats 3 \
  --output evaluation/reports/m3/m3-p5-bm25-offline/bm25_only
SIMPLE_QNA_RAG_BM25_TOKENIZER=char2gram python -m evaluation.retrieval \
  --dataset evaluation/datasets/golden.jsonl --warmup-cases 3 \
  --output evaluation/reports/m3/m3-p5a-char2gram/retrieval --candidate-id m3-p5a-char2gram
python scripts/check_markdown_links.py
```

Phase 6:

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
pytest -q
npm ci
npm test
npm run sync-vendor
git diff --exit-code -- web/static/vendor
python scripts/check_markdown_links.py
RUN_LIVE_LLM_TESTS=1 python -m evaluation.baseline \
  --dataset evaluation/datasets/golden.jsonl \
  --routing-runs 3 --warmup-cases 3 --candidate-id m3-final \
  --expect-variants-sha256 "<Phase 1 승인 해시>" \
  --output evaluation/reports/m3/m3-final
python -m evaluation.compare --kind baseline \
  --baseline evaluation/baselines/m2_initial.json \
  --candidate evaluation/reports/m3/m3-final/baseline_<ts>.json \
  --output evaluation/reports/m3/m3-final/compare
git diff --check
python - <<'PY'
import hashlib, pathlib
for p, want in [("evaluation/baselines/m2_initial.json",
                 "e1edf23e7bc7c7584ee0e166de7ace211e02fc1d064fccbbac1d081a35faa6d5"),
                ("evaluation/baselines/m2_initial.md",
                 "844e3c9cdc75ab244ca39fabe7693502e67a258892f25aea9d177f20025ad3f8")]:
    got = hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
    assert got == want, (p, got, want)
print("m2_initial 무결성 OK")
PY
```

---

## 13. Phase 순서·의존성·migration·rollback

### 13.1 순서와 의존성

```text
Phase 0 (기준 고정, 제품 코드 변경 없음 — 회귀 도구만 추가)
   └─ Phase 1 (evaluator v2 · 비교 도구) ── 이후 모든 Phase의 판정 기반
        ├─ Phase 2 (MMR) ── Phase 4의 context 고정과 Phase 5의 hybrid 비교 전제
        ├─ Phase 3 (Routing) ── Phase 2와 독립(검색 경로 미사용)
        ├─ Phase 4 (Intent) ── Phase 2 채택 결정 이후 실행
        └─ Phase 5 (BM25) ── Phase 2 이후, 조건부
              └─ Phase 6 (통합 live · 승인)
```

- 설계 검토는 병렬 가능하지만 같은 worktree의 제품 파일 편집은 직렬화한다(Plan §2). 공식 latency 실행의 직렬 규칙은 §4.4 절차 1번이 유일한 정의이며 여기서 다시 서술하지 않는다.
- Phase 3은 `rag_engine.py`를 건드리지 않고 Phase 2는 `agent.py`를 건드리지 않아 파일 충돌이 없다. 공통 편집 파일은 `config.py`와 `evaluation/reporting.py`뿐이며 서로 다른 상수/키만 추가한다.

### 13.2 migration

- **데이터 migration 없음**: golden.jsonl, corpus, vectorstore 모두 읽기 전용이며 재생성하지 않는다.
- **리포트 schema migration**: 1.0.0 → 1.1.0은 추가 전용이므로 기존 리포트를 변환할 필요가 없다. `compare.py`는 `schema_version`이 `1.0.0`인 리포트도 읽을 수 있어야 하며(누락 키는 `null` 취급) 이를 테스트로 고정한다.
- **evaluator version migration**: M2 리포트는 v1만 갖고 있고 v2 값은 `rescore.py`로 사후 생성한다. M2 리포트를 수정하지 않는다.

### 13.3 rollback

| 변경 | rollback 방법 | 확인 |
|---|---|---|
| MMR 저장 벡터 재사용 | `MMR_VECTOR_SOURCE="embed"` | 기존 42건 지표·순위가 M2와 동일 |
| Routing override | `ROUTING_SIGNAL_OVERRIDE=False` | `_decide_tool()`이 LLM 결정만 반환 |
| corpus topic hint | `ROUTING_CORPUS_TOPIC_HINT=False` | `router_prompt_sha256`가 개정 프롬프트 값으로 복귀 |
| Intent 단순화 | `ANSWER_TEMPLATE_MODE="intent"` | intent별 템플릿 경로 복귀 |
| BM25 tokenizer | `BM25_TOKENIZER="whitespace"` (미채택 시 모듈 삭제) | BM25 순위가 현행과 동일 |
| evaluator v2 | 리포트의 v1 열만 읽음 | v1 값이 M2와 재현 일치 |

Plan §6의 즉시 기각 조건을 그대로 따른다: 품질 floor 하나라도 위반, 매핑 정확성 미증명, web recall 손실, evaluator의 의미 합침. 환경 지문 불일치·모델 부재는 "실패"가 아니라 **"비교 불가"**로 표시하고 사용자 결정을 요청한다. warm-up 미수행·MMR 폴백 발생·evaluator 변형 표 불일치도 같은 "판정 불가" 처리 대상이다(§3.5, §4.4, §5.5).

### 13.4 이 문서의 계약 범위와 Phase별 상세의 지위

리뷰 권고(선행 코드 표면 축소)를 반영해 무엇이 **불변 계약**이고 무엇이 **구현 지침**인지 구분한다.

| 지위 | 해당 절 | 변경 조건 |
|---|---|---|
| 불변 계약 (사용자/리뷰 승인 필요) | §3.1~3.7 공통 계약, §4.4 warm-up, §4.5 link 검사 범위, §5.5 fail-closed, §5.8 gate 상수와 분모, §6.5 실패·폴백 계약, §7.2 신호 정의와 §7.4 우선순위, §7.5 분모·집계, §9.3 채택 gate, §11.3 공개 계약, §13.3 rollback | 요구사항 정정 또는 사용자 승인 |
| 구현 지침 (Phase 진입 시 세부 조정 가능) | §5.2~5.6 규칙 세부, §6.3~6.4 내부 시그니처, §8.3 worksheet 세부 포맷, §9.1~9.2 실험 하네스 내부 | 위 계약을 바꾸지 않는 범위에서 Phase 진입 시점의 implementation note로 조정하고 그 사실을 Phase 리포트에 남긴다 |

Phase 2~5의 내부 시그니처를 이 문서에서 지우지 않는 이유는, 이미 실측(§6.2, §5.7)으로 뒷받침돼 구현 위험을 낮추는 정보이고 삭제하면 같은 분석을 다시 해야 하기 때문이다. 대신 **동시 변경량**은 신규 모듈 축소(§5.1)와 Phase별 조건부 구현 중단 조건(§6.6, §7.6, §9 진입 조건)으로 제한한다.

---

## 14. 요구사항 추적표

### 14.1 기능 요구사항

| ID | 요구 요지 | 설계 대응 | 산출물 | 검증 |
|---|---|---|---|---|
| M3-REQ-001 | 비교 가능한 실험 경계(candidate ID, SHA, 설정, fingerprint, evaluator version) | §3.1~3.3, §4.1, §4.3, §4.6 | `candidate`/`warmup` block, `evaluation/fingerprint.py`, `retrieval_config` 확장, candidate ID 정규식, Phase 0 로그 해시 | `test_evaluation_fingerprint.py`, `test_config.py`(ID 정규식), 리포트 키 테스트 |
| M3-REQ-002 | 안전한 MMR 벡터 재사용(질의 임베딩 ≤1회, 대응 검증, 실패 시 명확 처리, cache 계약, 계측 무영향·계측 부재 안전) | §6.3~6.5, §3.6 | `vector_index.py`, `_bump()`/`_note()` null-safe helper, `_candidate_vectors()`, `mmr_instrumentation` | `test_vector_index.py`, `test_rag_engine_trace.py`, §6.5 폴백 6칸 행렬 |
| M3-REQ-003 | 검색 품질·성능 비교(per-case 순위 변화, 단계별 latency, floor 위반 사례) | §5.8, §6.7, §9.2 | `evaluation/compare.py`(gate 포함), `stage_summary` | `test_evaluation_compare.py`, `test_evaluation_gates.py` |
| M3-REQ-004 | 라우팅 taxonomy와 precision-first 명시 우선순위, 기존 fallback 보존 | §7.1, 규범 §7.2, §7.4 | 두 command grammar의 `routing_signals.py`, 신호 우선 `_decide_tool()` | `test_routing_signals.py`(76건 exact 8/12/56 + boundary/property 표), `test_agent_routing_policy.py`(WEB/DOCUMENT/NONE 및 관형절 최소쌍) |
| M3-REQ-005 | Routing 3회 집계(run별 지표·중앙값·변동 횟수, 단일 recall 분모) | §7.5 | `evaluate_routing_multi()`, `--runs`, `recall_denominators` | `test_evaluation_routing.py` |
| M3-REQ-006 | Answer evaluator v2(순수 함수, v1/v2 분리, 과대 주장 금지, 공식 실행 fail-closed) | §5.1~5.9 | `answer_rules.py`, `answer_variants.json`, `rescore.py`, `evaluator_profile`/`official` | `test_answer_rules.py`, `test_evaluation_answers.py`, replay |
| M3-REQ-007 | Intent 대조 실험(동일 context, 순서 무작위/정체 은닉, 결정 기록) | §8.1~8.6 | `intent_ab.py`, worksheet/key, ADR | `test_intent_ab.py` |
| M3-REQ-008 | 조건부 BM25(인터페이스+fixture 비교, 미채택 시 잔재 없음) | §9.1~9.4 | `text_tokenizers.py`, `experiments/bm25_tokenizer.py` | `test_text_tokenizers.py`, 기본값 무변경 테스트 |
| M3-REQ-009 | 공개 CLI/API 계약 보존, vectorstore 불변, CI에 live 미추가 | §11.3, §8.6, §12.3 | 계약 표, 기존 테스트 무수정 통과 | `test_agent.py`, `test_cli_entrypoints.py`, index 해시 불변 확인 |
| M3-REQ-010 | 승인 baseline과 추적성 | §10, §14 | 통합 리포트, worksheet, 이 추적표 | Phase 6 체크리스트 |

### 14.2 비기능 요구사항

| ID | 요구 요지 | 설계 대응 | 검증 |
|---|---|---|---|
| M3-NFR-001 | 재현성(고정 fixture, stable ordering, JSON=Markdown 동일 집계) | 정렬된 표본 선택(§6.3 V5), seed 기반 blind 순서(§8.3), Markdown 렌더러가 JSON 값만 사용 | 결정론 테스트, JSON/Markdown 일치 테스트 |
| M3-NFR-002 | 성능 측정 건전성(`perf_counter` 유지, 동일 process warm-up, 병렬 금지, 측정 제외와 구조화 metadata) | §4.4, §7.5(순차 runs), §9.2(RSS) | 리포트 `warmup` block과 `measured_case_count`, `test_evaluation_retrieval.py` ⑧, `warmup.performed=false` → gate `pass=null` |
| M3-NFR-003 | 보안·프라이버시(상세 리포트 Git 제외, 외부 전송 추가 금지, loopback 한정) | §3.7, §7.3 | `git check-ignore` 확인, `is_loopback_endpoint()` 단위 테스트, 비-loopback에서 hint 미생성 |
| M3-NFR-004 | 유지보수성(evaluator에 제품 로직 복제 금지, fake 테스트 가능, optional dependency import 안전, 신규 모듈 최소화) | §8.2 seam, §6.3 duck-typed, §9.1 지연 import, §5.1 모듈 축소 | fake 기반 테스트 전부, import-only 테스트 |
| M3-NFR-005 | 회귀 방지(전체 테스트·dataset validation·Markdown local link 검사·`git diff --check`) | §4.5 검사기 설계, §12.4 공통 명령 | `python scripts/check_markdown_links.py` (exit 0 필수), `tests/unit/test_check_markdown_links.py`, Phase별 gate 절차 |

### 14.3 §4.1 gate 대응표

| gate | 판정 소스 | 임계 | 설계 위치 |
|---|---|---|---|
| Retrieval 평균/p95 latency | `retrieval.latency_ms.{mean,p95}` (+ `retrieval.warmup.performed`) | ≤ 8,420ms / ≤ 13,570ms + 감소율 | §4.4, §6.7, §5.8 |
| MMR 평균 latency | `retrieval.stage_summary.mmr.latency_ms_mean` | ≤ 2,869.862ms + 80% 감소 | §6.7 |
| Recall@10/@5, MRR@10, nDCG@10 | `retrieval.metrics` | ≥ 0.9524 / 0.9286 / 0.96 / 0.93 | §6.7 |
| Routing accuracy (중앙값) | `routing.aggregate.correct_count.median` (분모 **76**) | `>= 69` | §7.5, §5.8 |
| Document route recall (중앙값) | `routing.aggregate.document_route_correct.median` (분모 **61**) | `>= 54` | §7.5, §5.8 |
| Web search recall (각 run) | `routing.per_run[].web_search_correct` (분모 **15**) | `== 15` (모든 run) | §7.5, §5.8 |
| Source any-hit / mean recall | `answers.source.*` (v1 정의 유지) | `== 1.0` / ≥ 0.93 | §5.9 |
| Answer E2E 평균/p95 | `answers.latency_ms` (+ `answers.warmup.performed`) | ≤ 61.03s / ≤ 82.37s | §4.4, §10 |
| BM25 tokenizer 자원(조건부) | `bm25_ab.resources.rss_peak_bytes.median`, `initialize_ms.median` | ≤ baseline × 1.20 | §9.2, §9.3 |

---

## 15. 열린 쟁점과 사용자 결정 필요 항목

| # | 쟁점 | 설계 기본안 | 사용자/리뷰 결정 필요 |
|---|---|---|---|
| O1 | 검토된 scoped 변형(§5.5) 4건이 evaluator를 특정 답변에 맞춘 과적합인가 | 범위를 `(case_id, assertion_index)`로 고정하고 `rationale` 필수, 전역 동의어 금지 | 4건 각각의 승인 |
| O2 | 부분 거절(§5.6 A9)을 abstention으로 볼지 | v2는 거절로 판정하고 한계로 명시 | 정책 확인 |
| O3 | corpus topic hint가 프롬프트에 문서 파일명을 넣는 것 | `is_loopback_endpoint(OLLAMA_BASE_URL)`가 True일 때만 활성, 비-loopback이면 자동 억제(§7.3), 파일명만 사용 | 허용 여부 |
| O4 | Routing 3회 live 실행 비용(76건 × 3, M2 평균 5.44초/건 기준 ≈ 21분/후보) | 후보 스크리닝은 `--runs 1`, 공식 판정만 3회 | 비용 승인 |
| O5 | Intent A/B 실행 비용 ≈ 43분(§8.7)과 사람 검토 29×2 | 단일 실행 후 worksheet 검토 | 비용·검토자 승인 |
| O6 | Phase 5 진입 여부 | 진입하지 않아도 M3 완료 가능 | 진입 승인 |
| O7 | `dense` 단계 정의 변경(질의 임베딩이 `query_embed`로 분리)으로 M2의 단계별 수치와 직접 비교 불가 | 전체 latency와 MMR 단계로 판정하고 단계 정의 변경을 리포트에 명시 | 비교 방식 확인 |
| O8 | Intent 유지 시 재학습에서 골든셋 누수 위험(§8.5) | 골든 질문 문자열 사용 금지, dev.jsonl로 임계 결정 | 방침 확인 |
| O9 | `bd-002`("관련된 자료가 있으면 알려줘")는 결정론 규칙으로 잡지 않음 | 약한 신호 `자료`를 하드 규칙에 넣으면 web 질문 오탐 위험 → LLM+프롬프트에 위임 | 잔여 위험 수용 여부 |
| O10 | evaluator v2 규칙 변경 시 이미 발행한 M3 후보 리포트와의 비교 | `rules_fingerprint`가 다르면 `compare.py`가 경고하고 v2 열 비교를 차단 | 정책 확인 |
| O11 | **[해소됨 — 2026-08-05 iteration 1]** Document QA recall 기준의 초기 표기 `86.27% 이상 (44/51)`이 서로 다른 모집단(category 51 vs `expected_route` 61)을 섞어 단일 metric으로 성립하지 않았다 | 요구사항 §4.1을 **단일 metric 계약**으로 정정했다: 분모 61(`expected_route == document_qa`), M2 `44/61 (72.13%)`, M3 최소 **`54/61 (88.52%)`**. category 51은 어떤 recall의 분모도 아니다. Plan §4 Phase 3, Design §5.8·§7.5·§12.1·§14.3의 count·백분율·판정식을 모두 같은 정의로 맞췄고 판정은 `Fraction`/count로 한다 | 정정 자체에 대한 사용자 승인(요구사항 §8 정정 기록) |
| O12 | 명시 신호(WEB/DOCUMENT)가 있는 질문에서 LLM 예외 시 기존 keyword fallback을 타지 않게 된 것 | 요구사항 M3-REQ-004가 "모델 가용성과 무관한 우선순위"를 요구하므로 결정론적 route를 반환한다(§7.4). NONE 신호에서는 기존 4개 폴백이 완전히 보존된다 | 계약 변경 승인 |
| O13 | `--warmup-cases` 기본값 0이면 latency gate가 "판정 불가"가 되므로, 공식 실행에서 옵션 누락이 실패로 보일 수 있다 | 기본값을 0으로 두고(기존 동작 보존) 공식 실행 명령에 항상 `--warmup-cases 3`을 포함한다(§12.4). 기본값을 3으로 바꾸면 기존 호출의 의미가 조용히 바뀌므로 채택하지 않았다 | 정책 확인 |
| O14 | Markdown link 검사기의 anchor slug 규칙이 GitHub 렌더러와 완전히 동일하지 않을 수 있다 | 저장소 내부 일관성 검사로 한정하고 한계를 §4.5와 `--help`에 명시한다. 오탐이 나오면 규칙을 좁히되 새 dependency는 추가하지 않는다 | 범위 확인 |
| O15 | **[해소됨 — 2026-08-07 Iteration 6]** 리뷰 Iteration 5 M1: `WEB_FUSED_TOPIC_GAP`/`CHANNEL_REQUEST_MAX_WORD_GAP`(둘 다 고정값 2)이 주제어를 매치로부터 `m+3` 이상 밀어 넣는 것만으로 회피되고(`웹검색 관련 API 구조 알려줘` 등), 왼쪽 경계를 문장 시작·공백으로만 인정해 문장부호·괄호 뒤 강한 검색 명령을 과소탐지했다 | 두 고정 거리 상수를 **모두 제거**하고 `CHANNEL`·`WEB_FUSED` 공통의 **주제절 억제 규칙**(§7.2.1 — `SOURCE_PARTICLE`/조사 없는 명령 인접이면 목적어 내용과 무관하게 즉시 command 증거, 아니면 매치~마지막 어절 앞까지 전체 절에서 `TOPIC_HEAD` 스캔)으로 교체했다. 왼쪽 경계는 정규식 lookbehind `(?<!\w)`(Unicode-aware)로 재정의해 문장부호·괄호 뒤는 인정하고 복합어 내부는 배제한다. `WEB_FUSED`는 `CHANNEL`과 동일한 판정 함수를 공유하며 더 이상 별도의 즉시-WEB 우회 경로가 아니다(§7.2.3). 골든 76건 WEB 10/DOCUMENT 12/NONE 54 exact set은 변화 없이 재현되고(2026-08-07 스크립트 재확인), R31~R36·17건 채널 fixture·왼쪽 경계 property 표로 반례를 고정했다 | 이번 Iteration 6 승인 — 리뷰 Iteration 5는 동일 근본 문제 2회 연속 재발을 이유로 STOP과 Iteration 6 불허를 권고했으나, 사용자가 미션 완료까지 진행을 명시적으로 지시하여 6회 상한 내 최종 Iteration 6으로 진행했다. 최종 Gate 판정은 다음 리뷰가 확정한다 |
| O16 | **[단순화 Cycle 1]** Iteration 6의 `SOURCE_PARTICLE` 관형절 우회 | 규범 §7.2에서 `has_particle` fast path와 `TOPIC_HEAD` 보정을 모두 폐기하고 두 command grammar만 사용한다. 일반 응답 술어와 관형절은 NONE/LLM이다. | `Routing_Simplification_Review_1.md`에서 Gate 판정 |

---

## 부록 A. 설계 검증에 사용한 측정 (재현 방법)

이 문서의 수치는 저장소를 수정하지 않는 일회용 스크립트로 얻은 관측값(표기가 없으면 **2026-08-05**, 명시된 행은 해당 날짜)이며 설계가 고정하는 계약이 아니다. Phase 0/1 구현 후 동일 결과를 `evaluation/fingerprint.py`와 `evaluation/rescore.py`로 재생산하고, 명령 출력은 §4.6의 Git 제외 로그 경로에 저장해 경로와 SHA-256만 Phase 0 리포트에 남긴다.

| 항목 | 방법 | 결과(관측값) | Phase 0 재생산 |
|---|---|---|---|
| fingerprint 4종 일치 | `hashlib.sha256` + `reporting.build_corpus_manifest()` | §4.1 | `logs/fingerprint.json` |
| dataset 구성(category / `expected_route`) | `golden.jsonl` 직접 집계 | 76 / 51·15·3·7 / document_qa 61·web_search 15 | `logs/dataset_validate.log` |
| FAISS 구조·매핑 | `faiss.read_index` + `pickle.load(index.pkl)` | §6.2 | Phase 2 `test_vector_index.py` |
| evaluator v2 replay | 저장된 3개 answers 리포트에 §5.2~5.6 규칙 적용 | §5.7 | Phase 1 `evaluation/rescore.py` 산출물 |
| 정적 회귀 | `pytest -q`, `npm test` | 358 passed / 1 skipped, 9 passed | `logs/pytest.log`, `logs/npm_test.log` |
| Markdown local link | (Phase 0에서 신규 구현) | — | `logs/markdown_links.log` |
| 규범 §7.2 신호 분류 기대값(**2026-08-07**, 단순화 Cycle 1) | 두 command grammar로 76건 exact set과 boundary/property 표를 구현 시 검증 | WEB 8 / DOCUMENT 12 / NONE 56, WEB·DOCUMENT 오탐 0 | Phase 3 `test_routing_signals.py`, `test_agent_routing_policy.py` |
| Markdown 열거 tracked vs 합집합 (**2026-08-06**) | `git ls-files -- '*.md' '*.markdown'` 대 `git ls-files --cached --others --exclude-standard -- '*.md' '*.markdown'` | tracked 26 / 합집합 32(신규 미추적 6건, M3 문서 디렉터리 전체 포함) | Phase 0 `logs/markdown_links.json` |

## 부록 B. 용어

| 용어 | 정의 |
|---|---|
| candidate | 한 Phase에서 평가하는 하나의 구현 후보. §3.1의 ID로 식별 |
| floor | 품질 하한. 위반 시 후보를 기본 경로로 승격하지 않음 |
| stored 경로 | MMR이 FAISS에 저장된 벡터를 재사용하는 경로 |
| legacy 경로 | MMR이 후보 본문을 매번 임베딩하는 M2 경로 |
| replay | 저장된 답변 리포트를 모델 없이 다시 채점하는 것 |
| blind worksheet | variant 정체와 순서를 가린 사람 검토 문서 |
