# Settings Field Spec (generated)

이 파일은 `scripts/generate_field_spec.py`가 `simple_qna_rag.settings.FIELD_SPECS`/`MODEL_VALIDATORS`에서 재생성한다. 직접 편집하지 않는다.

## FIELD_SPECS

| # | name | annotation | default | default_factory | env_alias | parser | derive/derived_from | validators | consumers | default_pass | facade_type | facade_adapter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `PROJECT_ROOT` | Path | (factory) | 있음 | None | — | — | 없음 | config.py 내부 파생 필드, test_cli_entrypoints.py | PASS | None | None |
| 2 | `STATIC_DIR` | Path | — | — | None | — | derived_from=('PROJECT_ROOT',) | 없음 | web/server.py | PASS | str | str |
| 3 | `TEMPLATES_DIR` | Path | — | — | None | — | derived_from=('PROJECT_ROOT',) | 없음 | web/server.py | PASS | str | str |
| 4 | `VECTORSTORE_PATH` | Path | (factory) | 있음 | SIMPLE_QNA_RAG_VECTORSTORE_DIR | _parse | — | 없음 | cli/index_documents.py, rag_engine.py | PASS | str | str |
| 5 | `DATA_DIR` | Path | (factory) | 있음 | SIMPLE_QNA_RAG_DOCUMENTS_DIR | _parse | — | 없음 | agent.py, cli/index_documents.py | PASS | str | str |
| 6 | `INTENT_MODEL_PATH` | Path | (factory) | 있음 | SIMPLE_QNA_RAG_MODEL_DIR | _parse | — | 없음 | intent_classifier.py | PASS | str | str |
| 7 | `COLLECTION_NAME` | str | 'document_collection' | — | SIMPLE_QNA_RAG_COLLECTION_NAME | str | — | 없음 | — | PASS | None | None |
| 8 | `EMBEDDING_MODEL_NAME` | str | 'BAAI/bge-m3' | — | SIMPLE_QNA_RAG_EMBEDDING_MODEL_NAME | str | — | 없음 | cli/index_documents.py, rag_engine.py | PASS | None | None |
| 9 | `CHUNK_SIZE` | int | 1000 | — | SIMPLE_QNA_RAG_CHUNK_SIZE | int | — | >0, model: CHUNK_OVERLAP < CHUNK_SIZE | cli/index_documents.py | PASS | None | None |
| 10 | `CHUNK_OVERLAP` | int | 200 | — | SIMPLE_QNA_RAG_CHUNK_OVERLAP | int | — | model: CHUNK_OVERLAP < CHUNK_SIZE | cli/index_documents.py | PASS | None | None |
| 11 | `OLLAMA_BASE_URL` | str | 'http://localhost:11434' | — | SIMPLE_QNA_RAG_OLLAMA_BASE_URL | str | — | 없음 | agent.py, rag_engine.py | PASS | None | None |
| 12 | `OLLAMA_MODEL` | str | 'gpt-oss:20b' | — | SIMPLE_QNA_RAG_OLLAMA_MODEL | str | — | 없음 | agent.py, rag_engine.py | PASS | None | None |
| 13 | `RETRIEVAL_K` | int | 4 | — | SIMPLE_QNA_RAG_RETRIEVAL_K | int | — | >0 | rag_engine.py | PASS | None | None |
| 14 | `USE_MMR` | bool | True | — | SIMPLE_QNA_RAG_USE_MMR | _parse_bool | — | 없음 | rag_engine.py | PASS | None | None |
| 15 | `MMR_FETCH_K` | int | 100 | — | SIMPLE_QNA_RAG_MMR_FETCH_K | int | — | model: MMR_K <= MMR_FETCH_K | rag_engine.py | PASS | None | None |
| 16 | `MMR_K` | int | 20 | — | SIMPLE_QNA_RAG_MMR_K | int | — | model: MMR_K <= MMR_FETCH_K | rag_engine.py | PASS | None | None |
| 17 | `MMR_LAMBDA` | float | 0.5 | — | SIMPLE_QNA_RAG_MMR_LAMBDA | float | — | 0<=x<=1 | rag_engine.py | PASS | None | None |
| 18 | `NORMALIZE_EMBEDDINGS` | bool | True | — | SIMPLE_QNA_RAG_NORMALIZE_EMBEDDINGS | _parse_bool | — | 없음 | cli/index_documents.py, rag_engine.py | PASS | None | None |
| 19 | `USE_HYBRID_SEARCH` | bool | True | — | SIMPLE_QNA_RAG_USE_HYBRID_SEARCH | _parse_bool | — | 없음 | rag_engine.py | PASS | None | None |
| 20 | `BM25_TOP_K` | int | 50 | — | SIMPLE_QNA_RAG_BM25_TOP_K | int | — | >0 | rag_engine.py | PASS | None | None |
| 21 | `DENSE_TOP_K` | int | 50 | — | SIMPLE_QNA_RAG_DENSE_TOP_K | int | — | >0 | rag_engine.py | PASS | None | None |
| 22 | `RRF_TOP_K` | int | 50 | — | SIMPLE_QNA_RAG_RRF_TOP_K | int | — | model: RERANKER_TOP_K <= RRF_TOP_K | rag_engine.py | PASS | None | None |
| 23 | `RRF_CONSTANT` | int | 60 | — | SIMPLE_QNA_RAG_RRF_CONSTANT | int | — | >0 | rag_engine.py | PASS | None | None |
| 24 | `USE_RERANKER` | bool | True | — | SIMPLE_QNA_RAG_USE_RERANKER | _parse_bool | — | 없음 | rag_engine.py | PASS | None | None |
| 25 | `RERANKER_MODEL` | str | 'BAAI/bge-reranker-v2-m3' | — | SIMPLE_QNA_RAG_RERANKER_MODEL | str | — | 없음 | rag_engine.py | PASS | None | None |
| 26 | `RERANKER_TOP_K` | int | 10 | — | SIMPLE_QNA_RAG_RERANKER_TOP_K | int | — | >0, model: RERANKER_TOP_K <= RRF_TOP_K | rag_engine.py | PASS | None | None |
| 27 | `USE_WEB_SEARCH` | bool | True | — | SIMPLE_QNA_RAG_USE_WEB_SEARCH | _parse_bool | — | 없음 | agent.py, query_router.py | PASS | None | None |
| 28 | `WEB_SEARCH_MAX_RESULTS` | int | 3 | — | SIMPLE_QNA_RAG_WEB_SEARCH_MAX_RESULTS | int | — | >0 | web_search.py | PASS | None | None |
| 29 | `WEB_SEARCH_TIMEOUT` | int | 10 | — | SIMPLE_QNA_RAG_WEB_SEARCH_TIMEOUT | int | — | >0 | web_search.py | PASS | None | None |
| 30 | `WEB_SEARCH_REGION` | str | 'kr-kr' | — | SIMPLE_QNA_RAG_WEB_SEARCH_REGION | str | — | 없음 | web_search.py | PASS | None | None |
| 31 | `PROMPT_TEMPLATE` | str | "당신은 문서 기반 RAG 시스템의 답변을 생성하는 AI 어시스턴트입니다.\n\n [역할 및 규칙]\n\n 1. 먼저 질문과 제공된 문맥(Context)을 분석합니다.\n\n 2. 주어진 문맥(Context)의 정보만을 사용하여 답변합니다.\n  - 답변은 명확하고 구체적이어야 합니다.\n  - 답변의 양식은 표로 정리하기 적합한지 여부에 따라 아래 규칙을 따릅니다.\n\n 3. 답변 내용을 표로 정리하기 적합한지 판단합니다.\n  - 표로 정리하기 적합한지 판단하근 기준은, 여러 개의 항목을 공통된 기준(열/컬럼)으로 비교 및 정리하기에 적합한지 여부입니다.\n  - 답변 본문이 한 줄 이하 이거나 혹은 데이터 행이 1개 이하일 때에는 표로 정리하지 않습니다.\n  - 이 판단 과정은 답변에 드러내지 마세요. (생각을 말로 설명하지 마세요.)\n\n 4. 표로 정리하기에 적합한 답변 내용은 다음 형식의 Markdown 표로 본문을 작성합니다.\n  - 첫 줄: 한 문장 핵심 요약. 만약 사용자가 '예', '아니오'로 답변할 수 있는 질문을 했다면 '예' 또는 '아니오' 형태의 답변을 제일 서두에 표현할 것.\n  - 그 아래: 표 (헤더 1행 + 데이터 행 2개 이상)\n  - 예시 해더: | 항목 | 설명 | 참고 | 와 같은 형태\n  - 불필요한 서론 없이 간결하게 작성\n  - 표 내용 후 맨 아래에 결론으로 내용을 요약\n\n 5. 표로 정리하기에 부적합한 답변 내용은 억지로 표를 만들지 말고, 자연스러운 문단 또는 bullet/번호 리스트 형식으로 답변합니다.\n  - 추상적인 개념 설명, 스토리형 질문은 표로 정리하지 말고 자연스러운 평문으로 답합니다.\n  - '표로 답변하기 어렵다'와 같은 메타 설명은 하지 않습니다.\n\n 6. 항상 제공된 컨텍스트를 우선적으로 사용하고, 근거 없는 내용을 상상하여 추가하지 않습니다.\n\n 7. 문맥에 답이 없으면 '제공된 문서에서 관련 정보를 찾을 수 없습니다' 라고 답변합니다.\n\n 8. 가능한 경우 출처를 인용합니다.\n\n 문맥 (Context):\n {context}\n\n 질문 (Question): {question}\n\n 답변 (Answer):" | — | SIMPLE_QNA_RAG_PROMPT_TEMPLATE | str | — | 없음 | — | PASS | None | None |
| 32 | `MMR_VECTOR_SOURCE` | Literal | 'stored' | — | SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE | str | — | 없음 | rag_engine.py | PASS | None | None |
| 33 | `MMR_VECTOR_VALIDATION_SAMPLE` | int | 3 | — | SIMPLE_QNA_RAG_MMR_VECTOR_VALIDATION_SAMPLE | int | — | >0 | rag_engine.py | PASS | None | None |
| 34 | `MMR_VECTOR_COSINE_FLOOR` | float | 0.99 | — | SIMPLE_QNA_RAG_MMR_VECTOR_COSINE_FLOOR | float | — | 0<=x<=1 | rag_engine.py | PASS | None | None |
| 35 | `MMR_EMBED_CACHE_MAX_ITEMS` | int | 2048 | — | SIMPLE_QNA_RAG_MMR_EMBED_CACHE_MAX_ITEMS | int | — | >0 | — | PASS | None | None |
| 36 | `ROUTING_SIGNAL_OVERRIDE` | bool | True | — | SIMPLE_QNA_RAG_ROUTING_SIGNAL_OVERRIDE | _parse_bool | — | 없음 | agent.py | PASS | None | None |
| 37 | `ROUTING_CORPUS_TOPIC_HINT` | bool | False | — | SIMPLE_QNA_RAG_ROUTING_CORPUS_TOPIC_HINT | _parse_bool | — | 없음 | agent.py | PASS | None | None |
| 38 | `ROUTING_CORPUS_TOPIC_HINT_MAX_ITEMS` | int | 25 | — | SIMPLE_QNA_RAG_ROUTING_CORPUS_TOPIC_HINT_MAX_ITEMS | int | — | >0 | agent.py | PASS | None | None |
| 39 | `ANSWER_TEMPLATE_MODE` | Literal | 'default' | — | SIMPLE_QNA_RAG_ANSWER_TEMPLATE_MODE | str | — | 없음 | rag_engine.py | PASS | None | None |
| 40 | `INTENT_CONFIDENCE_FLOOR` | float | 0.0 | — | SIMPLE_QNA_RAG_INTENT_CONFIDENCE_FLOOR | float | — | 0<=x<=1 | — | PASS | None | None |
| 41 | `BM25_TOKENIZER` | Literal | 'whitespace' | — | SIMPLE_QNA_RAG_BM25_TOKENIZER | str | — | 없음 | tests/unit/test_config.py | PASS | None | None |
| 42 | `QUERY_CONCURRENCY_LIMIT` | int | 1 | — | SIMPLE_QNA_RAG_QUERY_CONCURRENCY_LIMIT | int | — | _check | web/concurrency.py | PASS | None | None |
| 43 | `QUERY_QUEUE_LIMIT` | int | 4 | — | SIMPLE_QNA_RAG_QUERY_QUEUE_LIMIT | int | — | _check | web/concurrency.py | PASS | None | None |
| 44 | `QUERY_QUEUE_TIMEOUT_SECONDS` | float | 5.0 | — | SIMPLE_QNA_RAG_QUERY_QUEUE_TIMEOUT_SECONDS | float | — | _check, model: QUERY_QUEUE_TIMEOUT_SECONDS < QUERY_EXECUTION_TIMEOUT_SECONDS | web/concurrency.py | PASS | None | None |
| 45 | `QUERY_EXECUTION_TIMEOUT_SECONDS` | float | 90.0 | — | SIMPLE_QNA_RAG_QUERY_EXECUTION_TIMEOUT_SECONDS | float | — | _check, model: QUERY_QUEUE_TIMEOUT_SECONDS < QUERY_EXECUTION_TIMEOUT_SECONDS | web/concurrency.py | PASS | None | None |
| 46 | `SHUTDOWN_GRACE_SECONDS` | float | 30.0 | — | SIMPLE_QNA_RAG_SHUTDOWN_GRACE_SECONDS | float | — | _check | web/server.py | PASS | None | None |
| 47 | `MAX_REQUEST_BODY_BYTES` | int | 16384 | — | SIMPLE_QNA_RAG_MAX_REQUEST_BODY_BYTES | int | — | _check | web/body_limit.py | PASS | None | None |
| 48 | `MAX_QUESTION_CHARS` | int | 4000 | — | SIMPLE_QNA_RAG_MAX_QUESTION_CHARS | int | — | _check | web/server.py | PASS | None | None |
| 49 | `UPSTREAM_CONNECT_TIMEOUT_SECONDS` | float | 5.0 | — | SIMPLE_QNA_RAG_UPSTREAM_CONNECT_TIMEOUT_SECONDS | float | — | _check | observability/deadline.py | PASS | None | None |

## MODEL_VALIDATORS

| # | 제약 | 관련 필드 | default 값 | 판정 |
|---|---|---|---|---|
| 1 | CHUNK_OVERLAP < CHUNK_SIZE | #10, #9 | 200 < 1000 | PASS |
| 2 | MMR_K <= MMR_FETCH_K | #16, #15 | 20 <= 100 | PASS |
| 3 | RERANKER_TOP_K <= RRF_TOP_K | #26, #22 | 10 <= 50 | PASS |
| 4 | QUERY_QUEUE_TIMEOUT_SECONDS < QUERY_EXECUTION_TIMEOUT_SECONDS | #44, #45 | 5.0 < 90.0 | PASS |
