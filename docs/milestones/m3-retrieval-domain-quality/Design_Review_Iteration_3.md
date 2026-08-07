# M3 Retrieval & Domain Quality 상세 설계 독립 재리뷰 — Iteration 3

- 검토일: 2026-08-06
- 대상: 최신 `Requirement.md`, `Plan.md`, `Design.md`, `Design_Review.md`, `Design_Review_Iteration_2.md`, 현재 제품·평가 코드, `golden.jsonl`, 승인 M2 baseline
- 결론: **REVISE**
- 점수: **9.5 / 10**
- 발견사항: **CRITICAL 0, MAJOR 1, MINOR 0, TRIVIAL 1**

## 1. Gate 재판정

Iteration 2의 MAJOR 2건은 설계 수준에서 모두 해소됐다. WEB positive grammar는 채널+요청 표현으로 확대됐고, 단독 언급·인용·부정·충돌 fixture 및 실제 classifier → `_decide_tool()` → LLM 예외/no-tool 통합 행렬까지 연결됐다. Markdown 검사기도 tracked와 untracked-nonignored를 한 Git 호출로 합치고 중복 제거·stable sort하며, 실제 임시 Git 저장소에서 신규 미추적 broken Markdown이 exit 1인지 확인하도록 바뀌었다.

그러나 확대된 WEB 규칙은 채널 토큰을 독립 단어/출처 지시로 확인하지 않고 질문 전체의 임의 요청 표현과 결합한다. 그 결과 `websocket 설정을 알려줘`, `Google AI 에이전트 구조를 보여줘`처럼 웹 채널 사용 요청이 아닌 기술·회사 주제 질문도 결정론적으로 WEB이 되며 LLM이 교정할 기회까지 사라진다. 이는 M3의 직접 목표인 document 질문의 과다 web routing 교정과 충돌하는 MAJOR이므로 최종 설계 Gate 조건(CRITICAL/MAJOR 0, score 9.7 이상)을 충족하지 못한다.

## 2. 발견사항

### MAJOR

#### M1. WEB의 “채널+요청 표현”이 구문적 결합 없이 전역 substring으로 평가돼 주제 언급을 채널 요청으로 오인한다

- severity: **MAJOR**
- 위치: `Design.md` §7.2.1 `CHANNEL`/`REQUEST`, §7.2.3 판정 순위 4, §7.2.4 fixture, §7.4 실제 classifier 행렬, §12.1 `test_routing_signals.py`/§12.2 `test_agent_routing_policy.py`
- 근거: 설계는 모든 토큰을 정규화 문자열의 부분 문자열로 찾고, 억제되지 않은 `CHANNEL` 하나와 질문 어디든 존재하는 `REQUEST` 하나면 WEB으로 판정한다. 따라서 `websocket 설정을 알려줘`는 `web`+`알려`, `Google AI 에이전트 구조를 보여줘`는 `google`+`보여`, `인터넷의 역사에서 중요한 사건을 알려줘`는 `인터넷`+`알려`로 WEB이 된다. 기존 단독 언급 음성 fixture인 `인터넷의 역사에 대해 설명해줘`는 `설명`을 REQUEST에서 의도적으로 제외했기 때문에 통과할 뿐이며, 같은 주제를 `알려줘`로 표현하는 정상 paraphrase는 실패한다. `구글 백서에서 ... 알려줘`는 DOCUMENT 토큰 덕분에 보호되지만 문서 범위 표현이 없는 Google 제품 질문은 보호되지 않는다. 또한 Latin `web`/`google`은 단어 경계가 없어 `websocket`, `webhook`, `googleapis` 내부에서도 매치된다.
- 영향: 명시적 웹 요청이 아닌 문서·기술 질문을 LLM 호출 전에 강제로 web_search로 보내므로, 현재 17개 오류가 모두 document→web인 문제를 새로운 규칙으로 재생산할 수 있다. 결정론적 WEB은 `_decide_tool()`의 예외/no-tool 보장뿐 아니라 정상 LLM의 document 선택도 덮어쓰므로 오탐의 복구 경로가 없다. 골든셋 76건의 오탐 0은 이 paraphrase/제품명 공간을 포함하지 않아 안전성 증거가 되지 못한다.
- 수정안: `CHANNEL`을 단순 명사 목록이 아니라 **외부 출처를 지시하는 channel phrase**로 정의한다. 최소한 Latin 토큰에는 Unicode-aware 단어 경계를 적용하고, `웹/인터넷/온라인/구글/포털/검색엔진`은 `에서/으로/로/기준/통해` 같은 출처·수단 표지 또는 인접 검색 요청과의 제한된 거리/구문 결합을 요구한다. `WEB_FUSED`는 계속 강한 신호로 유지하되 `websocket`, `webhook`, `Google AI`, `인터넷의 역사 ... 알려줘`, `온라인 서비스 구조를 보여줘`를 음성 fixture에 추가한다. 수정된 실제 문자열을 classifier를 stub하지 않는 `_decide_tool()` 정상·예외·no-tool 행렬로 다시 검증하고, 76건 exact-set/WEB·DOCUMENT 오탐 0을 함께 재실행한다.

### TRIVIAL

#### t1. 충돌 우선순위가 Requirement에 기록됐다는 Design의 설명이 실제 Requirement 문구와 일치하지 않는다

- severity: **TRIVIAL**
- 위치: `Requirement.md` M3-REQ-004, `Design.md` §7.2.3 “충돌 시 강도 순”, §14.5 O12
- 근거: Design은 약한 WEB 증거보다 DOCUMENT가 우선하고 주제어 예외가 있다는 구체화를 “Requirement에 같은 문장으로 기록”했다고 말하지만, Requirement는 여전히 “웹 명시 → 문서 명시 → 나머지”의 세 줄만 두고 약한 증거·융합형·주제어 예외를 정의하지 않는다.
- 영향: 구현 자체는 Design의 표와 fixture로 가능하지만, 사용자 승인 대상인 상위 계약과 상세 충돌 규칙의 출처가 모호하다.
- 수정안: M1의 문법 수정과 함께 Requirement M3-REQ-004에 “명시적 웹 **사용 요청**”의 의미와 부정·인용·주제 언급 제외, 동시 신호의 강도 우선순위를 한 단락으로 올리거나, Design의 “같은 문장으로 기록” 주장을 삭제하고 해당 상세 규칙을 별도 승인 항목으로 명시한다.

## 3. Iteration 2 finding 재검증

| 2차 finding | Iteration 3 판정 | 현재 근거 |
|---|---|---|
| M1 — explicit WEB 정상 표현 누락 | **recall 결함은 해소, 신규 precision 결함 M1** | `CHANNEL`+확장 `REQUEST`, 양성 paraphrase 9건, 부정·인용·충돌·단독 언급 fixture와 실제 classifier R1~R12 행렬이 추가됐다. `웹 기준으로 답해줘`, `인터넷에서 알려줘`, `온라인 자료를 보여줘`, `구글로 조회해줘`는 LLM 예외/no-tool에서도 WEB을 유지한다. 다만 전역 substring 결합이 회사·기술 주제를 WEB으로 오인한다. |
| M2 — tracked-only Markdown 열거 | **해소** | 기본 명령이 `git ls-files -z --cached --others --exclude-standard -- '*.md' '*.markdown'`로 바뀌었고, repo-relative POSIX 경로 중복 제거·stable sort가 명시됐다. 현재 worktree에서 이 명령은 tracked 26 + untracked-nonignored 6 = 32개를 실제 반환했다. E2는 미추적 broken Markdown의 exit 1을 실제 `git init` 저장소에서 검증한다. |
| t1 — Phase 0 “코드 변경 없음” | **해소** | §13.1이 “제품 코드 변경 없음 — 회귀 도구만 추가”로 정정됐다. |
| t2 — warm-up/V5 중복 | **해소** | 해당 절차·검증 항목이 각각 단일 서술로 정리됐다. |

## 4. 필수 확인 항목별 결론

1. **explicit WEB grammar와 실제 실행 계약:** 채널+요청 양성 recall, 단독 언급, 인용, 후행 부정, WEB/DOCUMENT 충돌, classifier→`_decide_tool()`→LLM exception/no-tool 연결은 구현 가능한 수준으로 구체화됐다. `_decide_tool()`은 DOCUMENT를 즉시 반환하고 WEB은 정상 LLM 검색어를 우선하되 예외/no-tool/빈 query에서 결정론적 추출로 보완하며, NONE만 기존 keyword fallback/예외 계약을 보존한다. 다만 채널의 주제 사용과 사용 요청을 구분하지 못하는 M1이 남는다.
2. **Markdown 열거:** tracked ∪ untracked-nonignored, ignored 제외, 중복 제거, stable sort, index-only 삭제 warning/skip, 신규 미추적 broken 문서 exit 1이 모두 명시됐다. 공통 명령 `python scripts/check_markdown_links.py`와 E1~E6 테스트는 구현 가능하다.
3. **61/44/54 metric:** `golden.jsonl`을 직접 집계하면 expected route가 document 61/web 15이고, 승인 baseline은 correct 59, document TP 44, web TP 15다. Requirement/Plan/Design의 M3 gate는 accuracy 69/76, document 54/61, 각 run web 15/15와 count/`Fraction` 비교로 일치한다.
4. **warm-up:** evaluator 내부의 동일 process·동일 engine, warm-up 결과 완전 폐기, Retrieval 42/Answer 29 measured count 유지, 구조화 metadata와 실패 시 latency `pass=null` 계약이 일치한다.
5. **null-safe MMR:** `_bump()`/`_note()` null-safe helper, `trace=None`/제공 × lookup miss/dimension/non-finite 6칸, `RAGEngine.query()` 제품 경로와 초기화 강등 테스트가 설계돼 있다. 예상 밖 예외는 전파하고 검증된 실패만 질의 단위 전체 legacy 폴백하는 경계도 타당하다.
6. **evaluator fail-closed:** 공식 v2는 variants 부재/schema 오류/기대 SHA 불일치 시 exit 2이고, 무변형은 `v2-no-variants`, `official=false`로 분리돼 공식 gate에 사용할 수 없다.
7. **정확한 명령과 Phase 구현 가능성:** 공통 정적 gate, Phase별 pytest/CLI/live opt-in 명령은 현재 패키지·entry point 구조와 호환된다. Phase 0은 회귀 도구만 추가하고, Phase 1 비교 기반 후 Phase 2/3을 독립 진행하며 Phase 4/5가 Phase 2 결정에 의존하는 순서도 구현 가능하다. 후보 A 성공 시 cache, 단순 routing 성공 시 2단계 router, BM25 gate 미달 시 production 반영을 중단하는 조건이 과설계를 제한한다.

## 5. 최종 결론

Iteration 2의 필수 두 수정과 Iteration 1의 CRITICAL/MAJOR/MINOR 항목은 모두 설계에 반영됐다. 다만 WEB positive grammar의 recall 확대가 정상 기술·회사 주제 질문에 대한 결정론적 오탐을 새로 만들 수 있어, 문서 우선 교정 정책을 구현 승인하기 전 경계 규칙과 음성 fixture를 한 차례 더 좁혀야 한다.

- **Gate: REVISE**
- **Score: 9.5 / 10**
- **Counts: CRITICAL 0, MAJOR 1, MINOR 0, TRIVIAL 1**
- 다음 Gate 최소 조건: M1 해소, 상위 Requirement와 충돌 계약 동기화, CRITICAL/MAJOR 0, MINOR 최소화, score 9.7 이상

## 6. 검증 기록

- `evaluation/datasets/golden.jsonl` 직접 집계: 총 76건; category `document_qa=51`, `web_search=15`, `boundary=3`, `unanswerable=7`; expected route `document_qa=61`, `web_search=15`.
- `evaluation/baselines/m2_initial.json` 직접 확인: routing `correct_count=59`, confusion matrix document TP `44`, web TP `15`; 승인 수치 59/76, 44/61, 15/15와 일치.
- 현재 제품 코드 확인: `agent._decide_tool()`은 LLM 예외를 전파하고 no-tool이면 `(None, None)`을 반환하며, `route_query()`가 예외/no-tool을 keyword fallback하고 web 실행 실패를 원본 질문 document QA로 재시도한다. 상세 설계의 WEB/DOCUMENT override는 이 경계에 삽입 가능하다.
- Git 명령 직접 실행: 합집합 열거는 현재 리뷰 파일 생성 전 기준 32개(tracked 26 + untracked-nonignored 6)를 반환했고 ignored 파일은 포함하지 않았다.
- 원문 문서·제품 코드·dataset·baseline은 수정하지 않았고, 이 리뷰 파일만 추가했다. commit/push는 수행하지 않았다.
