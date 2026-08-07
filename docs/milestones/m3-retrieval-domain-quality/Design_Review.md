# M3 Retrieval & Domain Quality 상세 설계 독립 리뷰

- 검토일: 2026-08-05
- 대상: `Requirement.md`, `Plan.md`, `Design.md`, 현재 제품·평가 코드와 테스트
- 결론: **STOP**
- 점수: **8.1 / 10**
- 발견사항: **CRITICAL 1, MAJOR 4, MINOR 4, TRIVIAL 2**

## 1. Gate 판정

현재 설계는 구현 가능한 수준의 함수 경계, 실패 모드, 테스트 후보를 상당히 구체적으로 제시한다. 특히 evaluator v1/v2 분리, FAISS row와 docstore의 구조 검증, 요청 지역 query embedding, Intent blind key 분리, 조건부 BM25 중단 기준은 좋은 기반이다.

그러나 승인 기준 자체가 모순된 O11을 설계가 임의 해석해 코드 상수로 고정하려 하고, 명시적 routing 우선순위가 LLM 실패 시 실행되지 않으며, MMR 폴백 의사코드는 일반 제품 호출에서 `trace=None`을 안전하게 처리하지 못한다. 공식 latency warm-up 절차도 별도 프로세스 실행이라 모델 warm 상태를 후보 실행에 전달하지 못한다. 따라서 Gate 조건(CRITICAL/MAJOR 0, score 9.7 이상)을 충족하지 못하며, **요구사항 정정 및 설계 revision 전 Phase 구현을 시작하면 안 된다.**

## 2. 발견사항

### CRITICAL

#### C1. Document QA recall의 분모·count·백분율 계약이 서로 모순되며 설계가 승인 없이 한 해석을 고정한다

- 위치: `Requirement.md` §2 표, §4.1(77), `Plan.md` Phase 3 수용 기준(190), `Design.md` §5.8(594), §7.5(872~876), §15 O11(1381)
- 근거: 골든셋을 직접 집계하면 category는 document_qa 51, web_search 15, boundary 3, unanswerable 7이고, `expected_route`는 document_qa **61**, web_search 15다. 승인 baseline의 confusion matrix는 document_qa 행 TP가 44이며 보고값 `72.13%`는 `44/61 = 0.721311...`이다. `44/51 = 86.2745...`는 M3 목표가 아니라 서로 다른 모집단을 섞은 값이다. 현재 evaluator도 `expected_route`를 actual label로 사용해 recall 분모 61을 만든다(`evaluation/routing.py`의 `expected = case.expected_route.value`, `precision_recall_f1`).
- 영향: `86.27% (44/51)`는 문자 그대로 동시에 만족시킬 수 있는 단일 metric 정의가 아니다. 설계의 `>= 0.8627`/분모 61 해석은 `>= 53/61`을 허용하지만 표시된 count `44`와 무관하고, accuracy 69/76 + 매 run web recall 15/15는 실제로 `54/61`을 강제한다. 구현자가 어느 수치를 authoritative하게 보는지에 따라 gate 결과가 달라진다.
- 수정안: Requirement와 Plan을 먼저 정정하고 사용자 승인을 받는다. 권고 계약은 기존 evaluator와 연속성을 유지해 **`expected_route=document_qa` 61건, M2 44/61 (72.13%), M3 최소 54/61 (88.52%)`**로 명시하는 것이다. 그러면 accuracy 69/76 및 web 15/15와 count가 일관된다. 만약 category 51건만 별도 측정하려면 이름을 `document_qa_category_recall`로 새로 만들고 M2 분자와 M3 목표 count를 다시 산출해 기존 route recall과 병기해야 한다. 정정 전에는 gates 상수를 구현하지 않는다.

### MAJOR

#### M1. 별도 프로세스 warm-up은 공식 Retrieval 실행을 warm-up하지 않는다

- 위치: `Design.md` §4.4(391~394), §12.4 Phase 2(1200~1205), `Requirement.md` §4.1(82), M3-NFR-002
- 근거: 설계는 `python -m evaluation.retrieval --limit 3` 종료 후 새 `python -m evaluation.retrieval` 프로세스를 시작한다. embedding/reranker Python 객체와 메모리는 프로세스 종료와 함께 사라진다. “같은 프로세스가 아니어도 된다”는 문구는 “모델 warm-up 후”라는 요구와 맞지 않는다. 현재 reranker는 `RAGEngine._rerank_documents()`에서 인스턴스에 lazy-load된다.
- 영향: 후보 report가 cold model load 또는 첫 호출 효과를 포함하면서도 `warmup_performed=true`라고 기록할 수 있어 latency gate의 신뢰성이 깨진다.
- 수정안: evaluator에 `--warmup-cases N`을 추가해 **동일 프로세스·동일 engine**에서 N건을 실행하되 공식 표본/집계에서는 버리고 즉시 42건을 측정한다. 또는 evaluation API를 호출하는 단일 driver를 둔다. report에는 warm-up case 수, 성공 여부와 측정 제외를 구조화 필드로 기록하고, 수동 `candidate.notes` 진술만 신뢰하지 않는다.

#### M2. 명시적 routing 우선순위가 LLM 예외보다 뒤에 있어 요구된 정책을 보장하지 못한다

- 위치: `Requirement.md` M3-REQ-004(126~130), `Design.md` §7.4(840~858), 현재 `agent.py` `_decide_tool()`/`route_query()`
- 근거: 설계 의사코드는 먼저 `_llm_decide_tool(question)`을 호출하고 그 뒤에 explicit WEB/DOCUMENT override를 적용한다. LLM 호출이 예외를 내면 explicit signal 분류 자체가 실행되지 않고 기존 keyword fallback으로 빠진다. 따라서 “사용자가 웹을 명시하면 web, 로컬 문서를 명시하면 document”라는 우선순위가 외부 상태에 따라 무효화된다.
- 영향: 가장 강한 사용자 의도가 모델 장애 시 보존되지 않으며, 설계가 약속한 8개 override 조합 테스트만으로는 LLM exception + explicit signal 조합을 검증할 수 없다.
- 수정안: explicit signal은 LLM 호출 전에 항상 계산한다. DOCUMENT는 즉시 `(document_qa, 원문)`으로 결정할 수 있다. WEB은 기존 query 품질을 지키기 위해 LLM을 시도하되 예외/no-tool이면 검증된 `extract_web_search_query()`로 결정론적으로 보완한다. 신호 NONE에서만 기존 LLM→keyword fallback 계약을 그대로 쓴다. WEB/DOCUMENT 각각에 LLM exception/no-tool 테스트를 추가한다.

#### M3. MMR lookup 실패 폴백 의사코드가 `trace=None`인 제품 경로에서 다시 실패한다

- 위치: `Design.md` §6.4(718~728), §6.5(741), 현재 `rag_engine.py` `query()`(trace 없이 `_retrieve_documents()` 호출)
- 근거: `_candidate_vectors()` 예외 처리에서 `trace.counters[...]`와 `trace.notes.append(...)`를 무조건 호출하지만, 같은 설계는 계측을 opt-in으로 유지하고 현재 제품 `query()`는 trace를 전달하지 않는다. §6.5의 “trace is not None일 때만 기록”과 의사코드가 모순된다.
- 영향: mapping miss/dimension/non-finite 시 “검증된 legacy 폴백” 대신 `AttributeError`로 전체 질의가 실패할 수 있다. 핵심 실패 안전 요구(M3-REQ-002)를 정면으로 훼손한다.
- 수정안: `_increment_trace(trace, ...)`처럼 null-safe helper를 명세하거나 모든 counter/note 접근을 `if trace is not None`으로 감싼다. `trace=None`과 trace 제공 두 경우 모두 lookup miss/dimension/non-finite가 legacy와 같은 결과를 내는 테스트를 요구한다. 초기화 강등 상태는 trace와 별개인 `mmr_vector_status`로 유지한다.

#### M4. NFR-005가 요구하는 Markdown local link 검사의 실제 명령·도구·테스트가 없다

- 위치: `Requirement.md` M3-NFR-005(176), `Plan.md` 공통 Gate(38), `Design.md` §12.4(1165~1284), 추적표 §14.2(1351)
- 근거: 추적표에는 link 검사를 충족한다고 쓰지만 정확한 실행 명령에는 dataset/pytest/npm/git 명령만 있고 link checker가 없다. 저장소의 `pyproject.toml`과 `package.json`에도 명시된 local Markdown link checker entry가 없다.
- 영향: 필수 회귀 gate가 재현 불가능하고, 설계가 추가하는 다수의 상대 링크 및 향후 산출물 링크가 깨져도 Phase를 닫을 수 있다.
- 수정안: dependency를 늘리지 않는 저장소 내 검사 스크립트 또는 승인된 도구와 버전을 확정하고, 지원 범위(상대 파일 링크, anchor, 코드블록 제외)를 명시한다. Phase 0~6 공통 명령과 NFR-005 추적표에 정확한 명령을 추가한다.

### MINOR

#### m1. Routing gate의 백분율을 반올림 float로 코드화한다

- 위치: `Design.md` §5.8(594), §7.5(872~876)
- 근거: 요구사항은 표시 반올림이 아닌 raw count/float 판정을 요구하지만 `0.8627`을 상수로 쓴다.
- 영향: O11 정정 뒤에도 경계값의 의미가 count와 분리될 수 있다.
- 수정안: O11 결정 후 `Fraction(required_correct, 61)` 또는 명시적 count 비교를 사용한다. web recall도 `15/15` count를 함께 검사한다.

#### m2. BM25 memory gate에 `tracemalloc`만 사용하면 native/모델 메모리를 놓칠 수 있다

- 위치: `Requirement.md` §3.2(45), `Design.md` §9.2(1027)
- 근거: `tracemalloc`은 Python allocator 중심이며 tokenizer/transformers/native 배열의 전체 RSS를 보장하지 않는다.
- 영향: 실제 배포 메모리 증가가 20%를 넘는데도 채택될 수 있다.
- 수정안: 동일 프로세스의 RSS peak/delta(플랫폼 지원 도구 명시)를 주 판정값으로 하고 `tracemalloc`은 Python allocation 진단값으로 병기한다. baseline/candidate 측정 범위와 3회 집계 규칙을 고정한다.

#### m3. evaluator v2의 검토 변형 파일 손상 시 fail-open 정책이 실험 계약을 약화한다

- 위치: `Design.md` §5.5(500~505), §5.7(555~557)
- 근거: 8개 assertion false negative 중 4개는 scoped variants에 의존하지만 파일이 없거나 깨지면 정규화만으로 계속 실행한다.
- 영향: 공식 v2 report가 필수 fixture를 만족하지 않는 규칙으로 생성될 수 있고, 사용자가 `reviewed_variants_loaded`를 놓치면 같은 “v2” 이름 아래 다른 의미가 생긴다.
- 수정안: 공식 `evaluator_version=v2`에서는 variants schema/fingerprint 불일치를 exit 2 또는 gate 판정 불가로 처리한다. 변형 없는 실험이 필요하면 별도 candidate/rule profile로 명시한다.

#### m4. Phase 0의 정적 회귀 수치가 재현 로그 경로 없이 설계 본문에 고정돼 있다

- 위치: `Design.md` §4.1(341~346), 부록 A(1394)
- 근거: `358 passed, 1 skipped / 9 passed`와 환경 측정은 적혀 있으나 명령 출력 artifact 식별 경로가 없다.
- 영향: 설계 검토자가 당시 결과와 현재 사용자 dirty state를 독립 재검증하기 어렵다.
- 수정안: 상세 로그는 ignored report 경로에 두고 해시/경로만 설계 또는 Phase 0 report에 기록한다. 설계 문서의 순간 수치는 “관측값”으로 명확히 구분한다.

### TRIVIAL

#### t1. candidate ID 문법과 예시가 완전히 정규화되지 않았다

- 위치: `Design.md` §3.1(124~136)
- 근거: 형식은 `m3-p<phase><letter>-<slug>`라고 하지만 `m3-p3r2-*`, `m3-p5-<tokenizer>` 등은 letter/round 표현이 서로 다르다.
- 영향: validator나 report directory naming 구현 시 불필요한 예외가 생긴다.
- 수정안: 정규식 하나(예: `^m3-p[0-6](?:[a-z][0-9]?)?-[a-z0-9-]+$`)와 round 의미를 확정한다.

#### t2. privacy 문구가 “로컬 Ollama” 전제를 코드 계약으로 검증하지 않는다

- 위치: `Design.md` §3.7(326), M3-NFR-003
- 근거: 현재 `OLLAMA_BASE_URL`은 localhost 상수이므로 현재 상태는 안전하지만, 설계는 report에 endpoint locality 검증을 요구하지 않는다.
- 영향: 향후 configurable endpoint가 생기면 corpus filename hint가 외부 endpoint로 전송될 수 있다.
- 수정안: hint 활성화는 loopback endpoint일 때만 허용하거나, 비-loopback이면 명시적 opt-in/마스킹/리포트 경고를 요구한다.

## 3. 요구사항 추적성 검토

| ID | 판정 | 리뷰 결과 |
|---|---|---|
| M3-REQ-001 | 부분 충족 | candidate/report/fingerprint 구조는 구체적이다. Phase 0 실행 증거 경로는 보강 필요(m4). |
| M3-REQ-002 | 미충족 | FAISS 구조 검증은 타당하나 제품 `trace=None` 폴백 안전성이 깨진다(M3). |
| M3-REQ-003 | 충족 가능 | per-case rank diff, 단계 latency, floor 위반 report가 설계돼 있다. warm 측정 방식은 M1 수정 필요. |
| M3-REQ-004 | 미충족 | taxonomy와 query 계약은 좋지만 explicit 우선순위가 LLM 예외보다 뒤다(M2). |
| M3-REQ-005 | 충족 가능 | 순차 3회, per-run/median/case variation이 구체적이다. 분모 gate는 C1 선결. |
| M3-REQ-006 | 부분 충족 | v1/v2 분리와 의미 한계 표기는 우수하다. 공식 rules fail-closed가 필요하다(m3). |
| M3-REQ-007 | 충족 가능 | 동일 context, seed 기반 순서 은닉, 별도 key와 두 검토 축이 명확하다. |
| M3-REQ-008 | 부분 충족 | conditional adoption/cleanup 경계는 명확하다. 메모리 측정법을 보강해야 한다(m2). |
| M3-REQ-009 | 충족 가능 | API 키, CLI entry point, vectorstore hash, live opt-in을 보존한다. failure-path 테스트를 M3 수정안대로 확장해야 한다. |
| M3-REQ-010 | 충족 가능 | 승인 전 baseline/Roadmap 변경 금지와 최종 추적표가 명확하다. |
| M3-NFR-001 | 충족 가능 | stable ordering, canonical fingerprint, deterministic seed가 명세돼 있다. |
| M3-NFR-002 | 미충족 | 별도 프로세스 warm-up이 유효하지 않다(M1). |
| M3-NFR-003 | 충족 가능 | ignored report 경계와 git 확인 명령이 있다. endpoint locality는 t2 권고. |
| M3-NFR-004 | 충족 가능 | 제품 seam 재사용, fake 경계, lazy optional import가 구체적이다. |
| M3-NFR-005 | 미충족 | Markdown link 검사 명령/도구가 없다(M4). |

## 4. 핵심 위험별 결론

- **Evaluator v2 의미:** 숫자 경계, `%`/`pp`, 부호, underscore 범위와 v1/v2 분리는 대체로 안전하다. 부정 문장 substring 일치는 새로 악화시키지 않고 명시적 한계로 남긴 판단도 요구 범위와 양립한다. 다만 공식 variants는 fail-closed로 바꿔야 동일 v2 의미가 보장된다.
- **MMR FAISS mapping/폴백:** 현재 artifact의 `index_to_docstore_id`, docstore 객체 identity, `IndexFlatIP.reconstruct()` 근거는 구현 가능하다. V1~V6와 query 단위 전체 폴백도 적절하지만 `trace=None` 결함을 먼저 고쳐야 한다. index 원본 불변 해시 확인은 유지한다.
- **Routing 분모·정책:** 실제 분모는 61로 독립 확인됐다. O11은 단순 오탈자 수준이 아니라 acceptance contract 모순이므로 CRITICAL이다. explicit signal은 LLM 성공 여부와 독립적으로 우선해야 한다.
- **Intent A/B 공정성:** retrieval 1회 후 같은 context 공유, slot randomization, identity key 분리는 공정하다. 자동 지표에서 sources가 두 variant에 동일하므로 source 지표는 실질적으로 retrieval 회귀 확인값이라는 점을 report에 명시하면 더 명료하다.
- **BM25 조건부 경계:** 미달 시 production 모듈과 dependency 잔재 제거, full hybrid와 BM25-only 분리, 1%p/20% gate는 요구와 맞는다. memory 판정은 RSS 기반으로 보강해야 한다.
- **공개 계약·privacy·rollback·concurrency:** public dict/API 키와 CLI entry point 보존, report gitignore, immutable stored index, bounded cache lock, feature rollback은 전반적으로 구체적이다. 폴백 null-trace와 endpoint locality가 남은 위험이다.
- **테스트 명령:** Python/frontend/dataset/vendor/diff 명령은 실제 저장소 구조와 호환된다. 신규 파일을 구현한 뒤의 명령도 형태상 가능하지만 필수 Markdown link 검사만 누락됐다.

## 5. 설계 규모와 축소 제안

`Design.md`는 1,405줄이며 신규 모듈 약 10개와 기존 모듈 다수 변경을 한 번에 규정한다. Phase별 조건부 구현을 명시해 무분별한 알고리즘 확대는 막았지만, `fingerprint`, `rescore`, `compare`, `gates`를 각각 새 CLI/모듈로 만들고 report schema를 모든 evaluator에 동시에 확장하는 것은 Phase 1의 선행 작업을 크게 만든다.

구현 방해를 줄이기 위한 권고 축소는 다음과 같다.

1. 먼저 C1을 Requirement/Plan에서 결정하고, Phase 0~1에는 기존 `evaluation.reporting`의 metadata builder 확장 + `answer_rules` + rescore만 구현한다.
2. gate 계산은 초기에는 `baseline.py` 또는 하나의 `evaluation.compare` 모듈 안의 순수 함수로 두고, 재사용 필요가 실제로 생길 때 `gates.py`로 추출한다.
3. `fingerprint.py`는 기존 `reporting.build_reproducibility_metadata()`를 얇게 호출하는 CLI 이상으로 확장하지 않는다.
4. Phase 2~5 상세 API는 각 Phase 진입 직전 reviewable implementation note로 분리하되, 이 문서에는 불변 계약·gate·rollback만 유지한다.
5. 후보 B cache, corpus hint, two-stage router, Intent 재학습, BM25 tokenizer 모듈은 선행 후보가 실패했을 때만 구체 구현한다는 현재 중단 조건을 엄격히 적용한다.

이 축소는 요구사항을 줄이지 않고 선행 코드 표면과 동시 변경량만 줄인다.

## 6. 다음 iteration 수정 목록

### 필수

1. O11을 Requirement/Plan에서 승인 가능한 단일 metric/count로 정정하고 gates/추적표/문구를 모두 같은 정의로 맞춘다.
2. 동일 프로세스 warm-up 계약과 CLI/API를 설계한다.
3. explicit routing signal을 LLM 예외와 독립적으로 우선 적용하도록 흐름과 테스트를 수정한다.
4. MMR fallback의 `trace=None` 안전성을 의사코드와 테스트 행렬에 반영한다.
5. Markdown local link checker의 실제 도구·범위·정확한 명령을 추가한다.
6. 공식 evaluator v2에서 variants schema/fingerprint 누락을 판정 불가 또는 실패로 처리한다.

### 권고

1. BM25 memory gate를 RSS 기반으로 보강한다.
2. Phase 0 측정 로그의 ignored artifact 경로와 해시를 남긴다.
3. routing threshold를 반올림 float가 아닌 count/Fraction으로 코드화한다.
4. candidate ID 정규식을 하나로 확정한다.
5. corpus topic hint의 loopback endpoint 제한을 명문화한다.
6. Phase 1 신규 모듈 수를 줄이고 phase-specific 상세를 점진적으로 확정한다.

## 7. 검증 기록

- `evaluation/datasets/golden.jsonl` 직접 집계: 76건; category `51/15/3/7`; expected route `document_qa=61`, `web_search=15`.
- `evaluation/baselines/m2_initial.json` 확인: routing confusion matrix document_qa→document_qa 44; 승인 Markdown의 document recall 72.13%와 일치(`44/61`).
- 현재 코드 확인: `evaluation.routing.evaluate_routing()`은 `expected_route`를 confusion matrix actual label로 사용한다; `RAGEngine.query()`는 trace 없이 retrieval을 호출한다; reranker는 engine 인스턴스에서 lazy-load된다.
- `.gitignore` 확인: `evaluation/reports/` 제외 규칙 존재.
- 제품 공개 계약 확인: `RAGEngine.query()`, `agent.route_query()`, FastAPI `QueryResponse`의 `answer/sources/success/search_type/intent` 경계와 설계 표를 대조했다.
- 링크는 본문에서 참조한 실제 로컬 파일/절 제목을 대조했다. 다만 저장소 전체 Markdown local link 자동 검사기는 현재 설계·도구에 없어 M4로 판정했다.

