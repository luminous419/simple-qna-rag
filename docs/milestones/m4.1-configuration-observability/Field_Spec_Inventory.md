# M4.1 FieldSpec 전수 Inventory (generated, 설계 단계 preview, 설계 재개 사이클 2)

상태: 41-행 `FIELD_SPECS` 표는 Iteration 2([Design_Review_Iteration_2.md](Design_Review_Iteration_2.md))
이후 변경 없이 보존한다. 설계 재개 사이클 1이
[Design_Review_Iteration_5.md](Design_Review_Iteration_5.md) MINOR `m5-01`
(header 5열/데이터 7열 불일치)을 5열로 정정했고, 이번 재개 사이클 2는
[Design_Review_Resume_Cycle_1.md](Design_Review_Resume_Cycle_1.md) MAJOR
`R1-MAJ-02`(5열 표를 재생성할 입력/알고리즘 계약 부재)를 닫기 위해 아래
`MODEL_VALIDATORS` 표의 근거 서술만 [Design.md](Design.md) §4.1.1의
`ModelValidatorSpec` 선언형 단일 원본을 참조하도록 갱신한다 — 표 자체의 행/열
값은 재개 사이클 1과 동일하다.

이 파일은 [Design.md](Design.md) §4.1 `FIELD_SPECS`의 41개 필드 전수를
미리 생성한 설계 증거다. 값은 실제 `src/simple_qna_rag/config.py`(2026-08-08
기준)의 상수명/기본값을 그대로 반영했다. 구현 단계에서
`scripts/generate_field_spec.py --check`가 실제 코드에서 재생성한 표와
diff 0임을 최종 확인한다(Design §4.1). `default_pass` 컬럼은 해당 필드의
default 값이 자신의 `validators`와 관련 `MODEL_VALIDATORS`(교차 필드 제약)를
모두 통과하는지를 나타낸다 — 41행 전부 PASS이므로 `Settings.from_sources()`를
인자 없이 호출해도(모든 default) 예외 없이 생성됨을 이 표가 증명한다
(§4.5 `test_settings.py`가 실행으로 재확인). `annotation` 컬럼은 Settings
**내부** 정규 타입이다(M3-01). `facade_type`/`facade_adapter`가 `None`이면
`config.py` facade는 annotation과 동일한 값을 그대로 노출하고, `str`이면
Settings는 `Path`를 쓰되 facade는 `str(...)`로 투영해 기존 공개 타입을
보존한다(Design §4.2/§4.4) — `#2~#6`(`STATIC_DIR`/`TEMPLATES_DIR`/
`VECTORSTORE_PATH`/`DATA_DIR`/`INTENT_MODEL_PATH`)가 이 경우이며 `#1`
(`PROJECT_ROOT`)만 무변환이다. `tests/unit/test_settings_facade_compat.py`
(Design §4.5)가 41개 필드의 runtime `type()`/값을 이전 `config.py` snapshot과
비교해 이 표의 facade 컬럼을 재확인하고, 42번째 심볼 `resolve_runtime_path`는
값 대신 `inspect.signature()` 일치로 검증한다(m4-01).

## FIELD_SPECS (41)

| # | name | annotation | default | default_factory | env_alias | parser | derive/derived_from | validators | consumers | default_pass | facade_type | facade_adapter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `PROJECT_ROOT` | `Path` | — | `Path(__file__).resolve().parents[2]` | `None` | — | — | 없음 | config.py 내부 파생 필드, `test_cli_entrypoints.py` | PASS | `None` | `None` |
| 2 | `STATIC_DIR` | `Path` | — | — | `None` | — | `derive=lambda v: v["PROJECT_ROOT"]/"web"/"static"`, `derived_from=("PROJECT_ROOT",)` | 없음 | `web/server.py` | PASS | `str` | `str` |
| 3 | `TEMPLATES_DIR` | `Path` | — | — | `None` | — | `derive=lambda v: v["PROJECT_ROOT"]/"web"/"templates"`, `derived_from=("PROJECT_ROOT",)` | 없음 | `web/server.py` | PASS | `str` | `str` |
| 4 | `VECTORSTORE_PATH` | `Path` | — | `PROJECT_ROOT/"runtime"/"vectorstore"` | `SIMPLE_QNA_RAG_VECTORSTORE_DIR`(기존 유지) | path resolver(`_parse_runtime_path`, m4-01) | — | 없음 | `cli/index_documents.py`, `rag_engine.py` | PASS | `str` | `str` |
| 5 | `DATA_DIR` | `Path` | — | `PROJECT_ROOT/"runtime"/"documents"` | `SIMPLE_QNA_RAG_DOCUMENTS_DIR`(기존 유지) | path resolver | — | 없음 | `agent.py`, `cli/index_documents.py` | PASS | `str` | `str` |
| 6 | `INTENT_MODEL_PATH` | `Path` | — | `PROJECT_ROOT/"models"/"intent_classifier"` | `SIMPLE_QNA_RAG_MODEL_DIR`(기존 유지) | path expand+resolve | — | 없음 | `intent_classifier.py` | PASS | `str` | `str` |
| 7 | `COLLECTION_NAME` | `str` | `"document_collection"` | — | `SIMPLE_QNA_RAG_COLLECTION_NAME`(신규) | `str` | — | 없음 | — (facade-only, 현재 무consumer) | PASS | `None` | `None` |
| 8 | `EMBEDDING_MODEL_NAME` | `str` | `"BAAI/bge-m3"` | — | `SIMPLE_QNA_RAG_EMBEDDING_MODEL_NAME`(신규) | `str` | — | 없음 | `cli/index_documents.py`, `rag_engine.py` | PASS | `None` | `None` |
| 9 | `CHUNK_SIZE` | `int` | `1000` | — | `SIMPLE_QNA_RAG_CHUNK_SIZE`(신규) | `int` | — | `>0` | `cli/index_documents.py` | PASS(1000>0) | `None` | `None` |
| 10 | `CHUNK_OVERLAP` | `int` | `200` | — | `SIMPLE_QNA_RAG_CHUNK_OVERLAP`(신규) | `int` | — | model: `CHUNK_OVERLAP<CHUNK_SIZE` | `cli/index_documents.py` | PASS(200<1000) | `None` | `None` |
| 11 | `OLLAMA_BASE_URL` | `str` | `"http://localhost:11434"` | — | `SIMPLE_QNA_RAG_OLLAMA_BASE_URL`(신규) | `str` | — | 없음 | `agent.py`, `rag_engine.py` | PASS | `None` | `None` |
| 12 | `OLLAMA_MODEL` | `str` | `"gpt-oss:20b"` | — | `SIMPLE_QNA_RAG_OLLAMA_MODEL`(신규) | `str` | — | 없음 | `agent.py`, `rag_engine.py` | PASS | `None` | `None` |
| 13 | `RETRIEVAL_K` | `int` | `4` | — | `SIMPLE_QNA_RAG_RETRIEVAL_K`(신규) | `int` | — | `>0` | `rag_engine.py` | PASS | `None` | `None` |
| 14 | `USE_MMR` | `bool` | `True` | — | `SIMPLE_QNA_RAG_USE_MMR`(신규) | bool parser | — | 없음 | `rag_engine.py` | PASS | `None` | `None` |
| 15 | `MMR_FETCH_K` | `int` | `100` | — | `SIMPLE_QNA_RAG_MMR_FETCH_K`(신규) | `int` | — | model: `MMR_K<=MMR_FETCH_K` | `rag_engine.py` | PASS(20<=100) | `None` | `None` |
| 16 | `MMR_K` | `int` | `20` | — | `SIMPLE_QNA_RAG_MMR_K`(신규) | `int` | — | 없음 | `rag_engine.py` | PASS | `None` | `None` |
| 17 | `MMR_LAMBDA` | `float` | `0.5` | — | `SIMPLE_QNA_RAG_MMR_LAMBDA`(신규) | `float` | — | `0<=x<=1` | `rag_engine.py` | PASS | `None` | `None` |
| 18 | `NORMALIZE_EMBEDDINGS` | `bool` | `True` | — | `SIMPLE_QNA_RAG_NORMALIZE_EMBEDDINGS`(신규) | bool parser | — | 없음 | `cli/index_documents.py`, `rag_engine.py` | PASS | `None` | `None` |
| 19 | `USE_HYBRID_SEARCH` | `bool` | `True` | — | `SIMPLE_QNA_RAG_USE_HYBRID_SEARCH`(신규) | bool parser | — | 없음 | `rag_engine.py` | PASS | `None` | `None` |
| 20 | `BM25_TOP_K` | `int` | `50` | — | `SIMPLE_QNA_RAG_BM25_TOP_K`(신규) | `int` | — | `>0` | `rag_engine.py` | PASS | `None` | `None` |
| 21 | `DENSE_TOP_K` | `int` | `50` | — | `SIMPLE_QNA_RAG_DENSE_TOP_K`(신규) | `int` | — | `>0` | `rag_engine.py` | PASS | `None` | `None` |
| 22 | `RRF_TOP_K` | `int` | `50` | — | `SIMPLE_QNA_RAG_RRF_TOP_K`(신규) | `int` | — | model: `RERANKER_TOP_K<=RRF_TOP_K` | `rag_engine.py` | PASS(10<=50) | `None` | `None` |
| 23 | `RRF_CONSTANT` | `int` | `60` | — | `SIMPLE_QNA_RAG_RRF_CONSTANT`(신규) | `int` | — | `>0` | `rag_engine.py` | PASS | `None` | `None` |
| 24 | `USE_RERANKER` | `bool` | `True` | — | `SIMPLE_QNA_RAG_USE_RERANKER`(신규) | bool parser | — | 없음 | `rag_engine.py` | PASS | `None` | `None` |
| 25 | `RERANKER_MODEL` | `str` | `"BAAI/bge-reranker-v2-m3"` | — | `SIMPLE_QNA_RAG_RERANKER_MODEL`(신규) | `str` | — | 없음 | `rag_engine.py` | PASS | `None` | `None` |
| 26 | `RERANKER_TOP_K` | `int` | `10` | — | `SIMPLE_QNA_RAG_RERANKER_TOP_K`(신규) | `int` | — | `>0` | `rag_engine.py` | PASS | `None` | `None` |
| 27 | `USE_WEB_SEARCH` | `bool` | `True` | — | `SIMPLE_QNA_RAG_USE_WEB_SEARCH`(신규) | bool parser | — | 없음 | `agent.py`, `query_router.py` | PASS | `None` | `None` |
| 28 | `WEB_SEARCH_MAX_RESULTS` | `int` | `3` | — | `SIMPLE_QNA_RAG_WEB_SEARCH_MAX_RESULTS`(신규) | `int` | — | `>0` | `web_search.py` | PASS | `None` | `None` |
| 29 | `WEB_SEARCH_TIMEOUT` | `int` | `10` | — | `SIMPLE_QNA_RAG_WEB_SEARCH_TIMEOUT`(신규) | `int` | — | `>0` | `web_search.py` | PASS | `None` | `None` |
| 30 | `WEB_SEARCH_REGION` | `str` | `"kr-kr"` | — | `SIMPLE_QNA_RAG_WEB_SEARCH_REGION`(신규) | `str` | — | 없음 | `web_search.py` | PASS | `None` | `None` |
| 31 | `PROMPT_TEMPLATE` | `str` | (기존 다국어 템플릿 문자열, config.py 원문 유지) | — | `SIMPLE_QNA_RAG_PROMPT_TEMPLATE`(신규) | `str` | — | 없음 | — (facade-only, `prompt_templates.py`로 대체됨) | PASS | `None` | `None` |
| 32 | `MMR_VECTOR_SOURCE` | `Literal["embed","stored"]` | `"stored"` | — | `SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE`(기존 유지, **M3 rollback flag**) | enum | — | `Literal[...]` | `rag_engine.py` | PASS | `None` | `None` |
| 33 | `MMR_VECTOR_VALIDATION_SAMPLE` | `int` | `3` | — | `SIMPLE_QNA_RAG_MMR_VECTOR_VALIDATION_SAMPLE`(신규) | `int` | — | `>0` | `rag_engine.py` | PASS | `None` | `None` |
| 34 | `MMR_VECTOR_COSINE_FLOOR` | `float` | `0.99` | — | `SIMPLE_QNA_RAG_MMR_VECTOR_COSINE_FLOOR`(신규) | `float` | — | `0<=x<=1` | `rag_engine.py` | PASS | `None` | `None` |
| 35 | `MMR_EMBED_CACHE_MAX_ITEMS` | `int` | `2048` | — | `SIMPLE_QNA_RAG_MMR_EMBED_CACHE_MAX_ITEMS`(신규) | `int` | — | `>0` | — (facade-only, 현재 무consumer) | PASS | `None` | `None` |
| 36 | `ROUTING_SIGNAL_OVERRIDE` | `bool` | `True` | — | `SIMPLE_QNA_RAG_ROUTING_SIGNAL_OVERRIDE`(기존 유지, **M3 rollback flag**) | bool parser | — | 없음 | `agent.py` | PASS | `None` | `None` |
| 37 | `ROUTING_CORPUS_TOPIC_HINT` | `bool` | `False` | — | `SIMPLE_QNA_RAG_ROUTING_CORPUS_TOPIC_HINT`(기존 유지, **M3 rollback flag**) | bool parser | — | 없음 | `agent.py` | PASS | `None` | `None` |
| 38 | `ROUTING_CORPUS_TOPIC_HINT_MAX_ITEMS` | `int` | `25` | — | `SIMPLE_QNA_RAG_ROUTING_CORPUS_TOPIC_HINT_MAX_ITEMS`(신규) | `int` | — | `>0` | `agent.py` | PASS | `None` | `None` |
| 39 | `ANSWER_TEMPLATE_MODE` | `Literal["intent","default"]` | `"default"` | — | `SIMPLE_QNA_RAG_ANSWER_TEMPLATE_MODE`(기존 유지, **M3 rollback flag**) | enum | — | `Literal[...]` | `rag_engine.py` | PASS | `None` | `None` |
| 40 | `INTENT_CONFIDENCE_FLOOR` | `float` | `0.0` | — | `SIMPLE_QNA_RAG_INTENT_CONFIDENCE_FLOOR`(신규) | `float` | — | `0<=x<=1` | — (facade-only, 조건부 기능·현재 무consumer) | PASS | `None` | `None` |
| 41 | `BM25_TOKENIZER` | `Literal["whitespace","char2gram","bge-subword"]` | `"whitespace"` | — | `SIMPLE_QNA_RAG_BM25_TOKENIZER`(기존 유지) | enum | — | `Literal[...]` | — (조건부, `tests/unit/test_config.py`만 참조) | PASS | `None` | `None` |

`resolve_runtime_path`(41+1=42번째 공개 심볼)는 `FieldSpec` 행이 아니라
`config.py`에 기존 시그니처(`env_name, default_path, legacy_path, *,
environ`)와 legacy fallback 동작을 보존한 채 남는 공개 호환 wrapper다
(Design §4.3/§4.4, m4-01). `FIELD_SPECS`의 path parser 역할은 별도 private
`_parse_runtime_path(raw, project_root)`가 담당한다.

## MODEL_VALIDATORS(교차 필드, 3건 — 전부 default 조합에서 PASS, R1-MAJ-02 근거 갱신)

이 표는 `Design.md` §4.1/§4.1.1의 `MODEL_VALIDATORS: tuple[ModelValidatorSpec,
...]`(교차 필드 제약, `FIELD_SPECS` 행과 분리된 별도 목록)의 증거이며 정확히
**5열**이다(`#`,제약,관련 필드,default 값,판정). 각 열은 `ModelValidatorSpec`의
선언형 필드에서 기계적으로 파생된다 — 열 순서대로 `enumerate(MODEL_VALIDATORS,
start=1)`의 `#`, `mv.constraint`, `mv.related_fields`를 41-행 표 `#`번호로
투영한 값, `mv.default_rendering(defaults)`, `mv.callable(namespace)`를
`SimpleNamespace(**defaults)`에 실행해 얻은 PASS/FAIL(§4.1.1
`render_model_validators_table`)이다. `facade_type`/`facade_adapter`는 개별
`FieldSpec` 행(위 41-행 표 11/12번째 열)에만 존재하는 per-field 속성이고 필드
쌍을 비교하는 `ModelValidatorSpec`에는 해당 개념이 없으므로 이 표에는 나타나지
않는다(Iteration 5 리뷰 `m5-01`이 지적한 혼동의 재발 방지).

`scripts/generate_field_spec.py --check`는 41-field 표와 이 표를 모두
`ModelValidatorSpec`/`FIELD_SPECS` 하나의 source에서 재생성해 checked-in
`docs/generated/settings_field_spec.md`와 diff 0임을 검증한다(Design §4.1.1,
`tests/unit/test_settings_inventory.py`에 연결, R1-MAJ-02 폐쇄) — runtime
validator 실행(`_validator_namespace`가 `mv.callable`을 `model_validator`에
등록)과 generator 판정이 동일 `mv.callable` 객체를 호출하므로 두 표 사이
drift가 구조적으로 불가능하다.

| # | 제약 | 관련 필드 | default 값 | 판정 |
|---|---|---|---|---|
| 1 | `CHUNK_OVERLAP < CHUNK_SIZE` | #10, #9 | `200 < 1000` | PASS |
| 2 | `MMR_K <= MMR_FETCH_K` | #16, #15 | `20 <= 100` | PASS |
| 3 | `RERANKER_TOP_K <= RRF_TOP_K` | #26, #22 | `10 <= 50` | PASS |

## 요약

- 필드 41개 = `env_alias is None` 3개(#1~#3, bootstrap path — package/repo
  고정) + `env_alias is not None` 38개. 38개 중 기존 이름을 그대로 유지하는
  8개(path 3: #4~#6 + M3 rollback flag 4: #32,#36,#37,#39 + `BM25_TOKENIZER`
  #41)를 제외한 나머지 30개는 이번 개정에서 `SIMPLE_QNA_RAG_<NAME>` 규칙으로
  신규 부여한다(Design §4.2). 3+8+30=41.
- default_pass 41/41 PASS — `Settings.from_sources()`(인자 없음)가 예외 없이
  생성됨을 이 표가 보증하고, §4.5 `test_settings.py`가 실행으로 재확인한다.
- M3 rollback flag는 4개(#32, #36, #37, #39) — 값·이름이 §5.3 보존 매트릭스
  대상이다.
