# 개선 계획: Agent 통합 및 완성도 향상

작성일: 2026-08-01
방향: LLM 기반 Agent로 Query Router 완성 (`tools.py`의 Tool 정의를 실제 AgentExecutor에 연결)

---

## Phase 0. 사전 정비 (착수 전 필수, 반나절)

Agent 작업을 시작하기 전에 재현성 문제부터 해결합니다. 이걸 미루면 나중에 "내 로컬에서는 되는데 팀원 환경에서는 안 된다" 류의 문제가 생깁니다.

- [ ] **Intent Classifier 가중치 커밋**: `intent-bge-m3-softmax/classifier_head.pt`(~26KB)가 git에 추적되지 않고 있음. 파일 크기가 작으므로 바로 커밋. `config.json`만 있고 `.pt`가 없으면 fresh clone 시 조용히 `other` 인텐트로 폴백되는데, 에러가 안 나서 문제를 알아차리기 어려움.
- [ ] **LICENSE 파일 추가**: README에 "MIT License"라고 명시되어 있지만 실제 `LICENSE` 파일이 없음. MIT 텍스트 추가.
- [ ] **API 응답에 라우팅 메타데이터 노출**: `web_server.py`의 `QueryResponse`에 `search_type`(웹검색/문서QA/에이전트 선택 도구)과 `intent` 필드 추가. 현재는 Pydantic이 조용히 버리고 있어 프론트엔드가 어떤 경로로 답변이 나왔는지 알 방법이 없음. Agent 도입 시 "어떤 도구를 왜 선택했는지"가 디버깅에 필수적이므로 이 단계에서 같이 처리.

**완료 기준**: fresh clone 후 `train_intent_classifier.py` 없이 바로 Intent 분류 동작, `/rag` 응답에서 `search_type`/`intent` 확인 가능.

---

## Phase 1. Agent 통합 (핵심 작업)

### 1.1 기술 검증 (완료 ✅)

- ✅ **`gpt-oss:20b`가 Ollama tool calling을 네이티브로 지원**함을 확인 (`ollama show`/`/api/tags`의 `capabilities`에 `tools` 포함). 모델 교체 불필요.
- ✅ **`ChatOllama` + `bind_tools()` 스모크 테스트 통과**: 단순 계산기 tool로 실제 tool call 발생 확인 (`{'name': 'add', 'args': {'a': 37, 'b': 58}, ...}`).
- ⚠️ **중요 발견 — tool_choice를 강제하지 않으면 도구를 건너뜀**: `web_search`/`document_qa` 두 tool을 바인딩하고 별도 시스템 프롬프트 없이 질문을 던지면, 웹검색 키워드가 있는 질문은 `web_search`를 잘 호출하지만, 순수 문서 QA성 질문("RAG에서 MMR이 뭐야?", "FAISS와 Elasticsearch를 비교해줘")에는 **도구를 호출하지 않고 모델이 자기 파라메트릭 지식으로 직접 답하려 함** (`NO_TOOL_CALL`). 이 상태로 배포하면 RAG 문서 근거 없이 LLM이 그냥 아는 척 답변하게 되어 매우 위험함.
- ✅ **해결책 검증 완료**: "반드시 둘 중 하나의 도구를 호출해야 하며 스스로의 지식으로 답하지 말라"는 시스템 프롬프트를 추가하자 4/4 질문 모두 올바른 도구로 라우팅됨. → **1.2에서 시스템 프롬프트에 이 지침을 필수로 포함**해야 함. (참고: LangChain의 `bind_tools(tools, tool_choice="any")`로 강제하는 방법도 있으나, Ollama 백엔드가 `tool_choice` 파라미터를 지원하지 않아 이번엔 시스템 프롬프트 방식을 채택함.)

**완료 기준**: ✅ 충족. `ChatOllama(model="gpt-oss:20b")` + 강제 시스템 프롬프트 조합으로 4개 샘플 질문 모두 기대한 도구로 라우팅됨.

### 1.2 Agent 라우터 연결 (완료 ✅)

- [x] `agent.py` 신규 작성. `tools.py`의 `web_search_tool`/`rag_tool`을 `ChatOllama.bind_tools()`로 바인딩해 LLM이 도구를 선택하게 하되, 표준 `AgentExecutor`(ReAct 루프)는 채택하지 않음 — 두 도구가 이미 완결된 최종 답변(sources 포함)을 반환하므로, AgentExecutor가 도구 결과를 다시 LLM으로 요약하면 포맷이 깨지고 LLM 호출이 중복됨. 대신 LLM에게는 "어느 도구를 쓸지 + (웹검색이면) 정제된 검색어"만 맡기고, 실제 실행은 해당 함수를 직접 호출해 구조화된 결과를 그대로 전달하는 "단발성 라우팅" 방식 채택.
- [x] 시스템 프롬프트 작성 (`agent.py`의 `SYSTEM_PROMPT`) — 도구 선택 기준 + 반드시 하나를 호출할 것(자체 지식으로 답하지 말 것) + web_search는 정제된 키워드 검색어를, document_qa는 원본 질문을 그대로 전달할 것을 명시.
- [x] 도구 선택 결과를 로그(`🤖 Agent 선택: ...`)와 API 응답(`search_type`)에 반영.
- [x] **폴백 전략**: `_decide_tool()` 호출 자체가 예외를 던지거나 도구를 선택 못하면(`tool_calls`가 빈 경우) `query_router.py`의 키워드 라우터로 폴백. `query_router.py`는 삭제하지 않고 `agent.py`가 import해서 안전망으로 사용.
- [x] 웹검색 실행 결과가 `success=False`이면 `document_qa`로 재시도하는 로직 추가 (이때는 LLM이 재작성한 검색어가 아니라 원본 질문을 사용 — RAG는 자연어 질문 형태로 학습/튜닝되어 있어 재작성된 키워드를 넣으면 오히려 Intent 분류가 왜곡될 수 있음).

**실제 테스트 결과 (gpt-oss:20b, 4개 질문)**:

| 질문 | 키워드 존재 여부 | Agent 선택 | 결과 |
|---|---|---|---|
| "오늘 서울 날씨 좀 웹에서 검색해줘" | O ("웹에서") | web_search (검색어: "서울 오늘 날씨") | ✅ 정확 (AccuWeather/기상청/Weather.com) |
| "RAG에서 MMR이 뭐야?" | X | document_qa | ✅ 정확 |
| "최신 파이썬 버전이 몇이야? 인터넷에서 찾아줘" | O ("인터넷에서") | web_search (검색어: "최신 파이썬 버전") | ⚠️ 라우팅은 정확하나 DDG 검색 결과 자체가 부정확(뉴스/스팸성 사이트) — Agent 로직과 무관한 DuckDuckGo 백엔드 품질 문제로 별도 이슈로 분리 |
| "FAISS와 Elasticsearch를 비교해줘" | X | document_qa | ✅ 정확 |

키워드가 전혀 없는 순수 의미 기반 질문(2번, 4번)도 올바르게 라우팅되어, 기존 키워드 매칭 방식의 한계(오탐/누락)를 실제로 개선함을 확인.

**개발 중 발견 및 수정한 버그**: 최초 구현에서는 LLM이 tool call 인자로 생성한 정제된 검색어(`{'__arg1': '서울 오늘 날씨'}`)를 무시하고 원본 질문 전체("오늘 서울 날씨 좀 웹에서 검색해줘")를 그대로 DuckDuckGo에 넘기고 있었음 — 이로 인해 "오늘의집" 같은 무관한 결과가 섞여 나왔음. `_decide_tool()`이 tool call의 `args`에서 실제 검색어를 추출해 사용하도록 수정 후 검색 품질이 개선됨 (수정 전/후 비교로 검증).

**알려진 한계 (이번 범위 밖)**: DuckDuckGo 검색 결과 자체의 관련성은 쿼리에 따라 편차가 큼(위 표 3번 케이스). `duckduckgo-search` 패키지가 `ddgs`로 개명되었다는 Deprecation 경고도 발생 중 — 패키지 마이그레이션은 별도 항목으로 분리.

### 1.3 `document_query.py` 정리 (완료 ✅)

- [x] `document_query.py` 삭제. `rag_engine.py`와 검색/RRF/재랭킹 로직이 완전히 중복되는 구버전이었고, `document_query_cli.py`(`rag_engine.get_rag_engine()` 사용)가 이미 대체하고 있어 남겨둘 이유가 없었음. 다른 코드에서의 import 없음을 확인 후 삭제.
- [x] `document_register.py`의 안내 메시지("이제 document_query.py를 실행하여...")를 `document_query_cli.py`/`web_server.py` 안내로 수정
- [x] `README.md`의 "또는 python document_query.py" 대안 실행법 및 프로젝트 구조 표에서 해당 항목 제거

---

## Phase 2. 테스트 자동화 (완료 ✅)

Agent는 키워드 라우터보다 예측 불가능성이 크므로, 이 시점부터는 print 기반 수동 테스트로는 부족합니다.

- [x] `test_web_search_simple.py`/`test_web_search_integration.py`(print 전용, assert 없음) 삭제하고 3개 pytest 파일로 재구성:
  - `test_web_search.py`: `web_search.py`의 포맷팅/실패 처리 로직. `DDGS`를 mock으로 대체해 네트워크 호출 없음.
  - `test_query_router.py`: 키워드 라우터의 분기/검색어 정제 로직. `get_rag_engine`/`search_and_format`을 mock으로 대체. 키워드 라우터의 알려진 한계("검색해줘"만으로 문서 QA 질문도 웹검색으로 오분류되는 케이스)를 회귀 테스트로 명시적으로 남겨둠.
  - `test_agent_routing.py`: `agent.py`의 실제 LLM 라우팅 정확도 회귀 테스트. `RUN_LIVE_LLM_TESTS=1` 환경변수가 없으면 자동 스킵되어 CI에서 Ollama 없이도 안전.
- [x] **라우팅 회귀 테스트셋 구축**: 원래 목표(30~50개)보다 작은 16개 질문 세트로 시작 (웹검색 키워드형 4개, 키워드 없는 실시간성 질문 2개, 문서QA - 설명/비교/절차/예아니오 각 유형 2개씩, 키워드 라우터가 오분류하는 엣지 케이스 1개). LLM 호출 비용(질문당 수 초)을 고려해 실용적인 크기로 시작하고 필요시 확장하는 방향으로 결정.
- [x] mock 기반 순수 로직 테스트(`test_web_search.py`, `test_query_router.py`, 12개)와 실제 LLM 필요한 테스트(`test_agent_routing.py`)를 완전히 분리.
- [x] `requirements.txt`에 `pytest>=8.0.0` 추가.

**검증 결과**:
- Mock 기반 테스트 12개 전부 통과, 3.37초 (네트워크/Ollama 불필요 — CI에 바로 태울 수 있음)
- 라이브 라우팅 회귀 테스트: **16/16 (100%)** 통과, 61초 (`RUN_LIVE_LLM_TESTS=1 pytest test_agent_routing.py -v -s`)

**완료 기준**: `pytest`가 최소한 폴백 로직과 포맷팅 로직에 대해 mock 기반으로 통과.

### Phase 0~2 완료 후 재검토에서 발견/수정한 항목 (2026-08-01, 커밋 전 리뷰)

- **`WEB_SEARCH_TIMEOUT` 설정이 죽어있던 문제 수정**: `web_search.py`가 `config.WEB_SEARCH_TIMEOUT`을 import만 하고 실제로는 어디에도 쓰지 않았음(pyflakes로 발견). 원인은 `DDGS.text()`의 `timelimit` 파라미터와 혼동한 것 — `timelimit`은 검색 결과의 최신성 필터(일/주/월/년)이지 요청 타임아웃이 아님. 실제 HTTP 요청 타임아웃은 `DDGS()` 생성자의 `timeout` 인자이며, 여기에 `WEB_SEARCH_TIMEOUT`을 연결해 수정함. 수정 전에는 DDGS가 응답을 지연시키는 상황에서 Agent 라우팅 호출 전체가 무한정 대기할 수 있었음.
- **`web_server.py`의 stale 주석 수정**: `/rag` 핸들러의 주석이 여전히 "Query Router를 통한 라우팅"으로 남아있었는데, 실제로는 `agent.route_query()`(LLM 기반)를 호출함. 주석을 코드와 일치시킴.
- **`test_agent.py` 신규 추가 — 가장 중요한 커버리지 공백 해소**: 기존 테스트는 `_decide_tool()`(도구 선택)과 키워드 라우터 자체만 검증했고, 정작 Phase 1.2의 핵심 산출물인 `route_query()`의 폴백/재시도 오케스트레이션(Agent 예외 시 키워드 라우터 폴백, 도구 미선택 시 폴백, 웹검색 실패 시 문서QA 재시도)은 어떤 테스트로도 보장되지 않았음. `_decide_tool`과 도구 실행 함수를 모두 mock으로 대체한 6개 테스트를 추가해 이 공백을 메움. 전체 mock 기반 테스트 12개→18개, 3.45초, 여전히 외부 의존성 없음.

**모든 mock 기반 테스트(18개) 및 pyflakes 검사 통과 확인.**

---

## Phase 3. 기존 로드맵과의 연결

Phase 0~2를 마치면 이번 브랜치("Agent 적용")의 본래 목표가 완성됩니다. 그 이후 개선 방향은 이미 리포지토리에 있는 두 문서에 정리되어 있으므로 중복 작성하지 않고 우선순위만 다시 정리합니다.

- `improvement_suggestion.claudecode.EVALUATION_AND_ROADMAP.md`의 **Phase 1 (Quick Wins)**: 한국어 BM25 형태소 분석기 적용, MMR 임베딩 캐싱 — 이번 Agent 작업과 무관하게 언제든 착수 가능
- 같은 문서의 **Phase 3 (확장성)**: 문서 수가 실제로 늘어나기 전까지는 우선순위 낮음 (현재 데이터 규모 기준 FAISS IndexFlatIP로 충분)
- 운영 모니터링(로깅/메트릭)은 Agent의 도구 선택 로그가 쌓이기 시작하면 자연스럽게 필요해지므로 Phase 2 테스트 자동화 다음 순번으로 배치

---

## 요약 타임라인 (전체 완료 ✅ — 2026-08-01)

| Phase | 내용 | 상태 |
|---|---|---|
| 0 | 재현성 정비 (가중치 커밋, LICENSE, API 필드) | ✅ 완료 |
| 1.1 | Agent 기술 검증 (ChatOllama, tool calling 지원 확인) | ✅ 완료 — gpt-oss:20b 지원 확인 |
| 1.2 | Agent 라우터 연결 및 폴백 전략 | ✅ 완료 — `agent.py` 신규 작성 |
| 1.3 | 레거시(`document_query.py`) 정리 | ✅ 완료 — 삭제 |
| 2 | 테스트 자동화 + 라우팅 회귀 세트 | ✅ 완료 — 16/16(100%) 통과 |

**최종 산출물**: `agent.py`(신규), `tools.py`(수정 — dict 반환), `web_server.py`(agent.py 사용 + API 필드 추가), `query_router.py`(폴백으로 격하, 그대로 유지), `document_query.py`(삭제), `LICENSE`(신규), `intent-bge-m3-softmax/classifier_head.pt`(커밋), `test_web_search.py`/`test_query_router.py`/`test_agent_routing.py`(신규 pytest 스위트, 기존 print 테스트 대체).

**남은 후속 작업 (이번 범위 밖, Phase 3 로드맵 참고)**:
- `duckduckgo-search` → `ddgs` 패키지 마이그레이션 (Deprecation 경고 발생 중)
- DuckDuckGo 검색 결과 자체의 관련성 편차 (일부 쿼리에서 뉴스/스팸성 사이트 반환) — Agent 라우팅과 무관한 별도 이슈
- Intent Classifier가 "RAG에서 MMR이 뭐야?", "FAISS와 Elasticsearch를 비교해줘" 같은 명확한 질문에도 `uncertain`으로 분류하는 것을 실측으로 확인 — README가 기대하는 explanation/comparison 분류와 어긋남. 기존 로드맵의 "Intent Classifier Fine-tuning" 항목으로 연결
- 라우팅 회귀 테스트셋을 16개에서 필요에 따라 확장
