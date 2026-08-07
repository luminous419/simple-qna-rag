# M3 Retrieval & Domain Quality 개발 계획

상태: **착수 제안** — 사용자 Gate 승인 전
요구사항: [Requirement.md](Requirement.md)

## 1. 실행 원칙

이 계획은 M2 기준선의 네 문제를 독립된 실험 축으로 다룬다. 한 Phase에서 여러 알고리즘을 동시에 바꾸지 않으며, 측정에서 가치가 입증된 후보만 다음 Phase의 production 경로로 승격한다. 상세 클래스명이나 저장 형식은 설계 담당자가 정하되 [요구사항](Requirement.md)의 비교·호환·실패 안전 계약은 변경하지 않는다.

담당 역할은 다음과 같다.

- **프로젝트 리더/사용자**: 범위, 비용이 드는 live 실행, Intent 결정과 최종 M3 baseline 승인
- **설계·구현 담당**: 상세 설계, 코드·테스트·report 작성, Phase 증거 제출
- **리뷰 담당**: 설계/코드/결과를 CRITICAL·MAJOR·MINOR로 검토하고 요구사항 추적성 확인

각 Gate는 CRITICAL/MAJOR 0을 요구한다. MINOR는 가능한 한 해소하고, 문서·코드 자체 평가가 9.7/10 미만이면 다음 Phase로 진행하지 않는다.

## 2. 의존성 및 실행 흐름

```text
Phase 0 기준 고정
   ├─> Phase 1 비교 하네스·evaluator v2
   │      ├─> Phase 2 MMR 최적화 ─┐
   │      ├─> Phase 3 Routing ────┼─> Phase 6 통합 live 평가·승인
   │      ├─> Phase 4 Intent 결정 ┤
   │      └─> Phase 5 BM25 실험 ─┘ (조건부, 미채택 가능)
   └───────────────────────────────> 회귀 기준
```

Phase 2~5의 설계는 병렬 검토할 수 있지만 같은 worktree의 production 파일 편집과 공식 latency 실행은 직렬화한다. Phase 4는 Phase 2의 검색 결과 고정 기능을 재사용할 수 있다. Phase 6은 모든 채택 결정 이후에만 실행한다.

## 3. 공통 Gate 절차

각 Phase는 다음 순서로 닫는다.

1. 산출물과 요구사항 ID의 추적표 작성
2. 해당 Phase의 결정론적 테스트와 전체 회귀 실행
3. 아래 공통 명령 4개 전부 통과 확인
4. 리뷰 담당의 CRITICAL/MAJOR/MINOR 판정
5. CRITICAL/MAJOR 0, 점수 9.7/10 이상 확인
6. 프로젝트 리더가 다음 Phase 진행 승인

Phase 0~6이 공유하는 공통 명령은 다음 4개이며, Phase별 검증 절에도 같은 형태로 반복한다.

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
pytest -q
npm test
python scripts/check_markdown_links.py
git diff --check
```

`scripts/check_markdown_links.py`는 표준 라이브러리만 쓰는 저장소 내 검사기이며 Phase 0 산출물이다(상세 범위는 상세 설계 §4.5).

리뷰 iteration은 Phase/산출물별 최대 4회다. 4회 후에도 gate를 충족하지 못하면 구현을 확장하지 않고 미충족 요구사항, 시도, 증거와 복구 선택지를 별도 중단 기록으로 남긴다.

## 4. Phase별 계획

### Phase 0 — 기준 상태와 실험 계약 고정

목표: M3 변경 전 코드·artifact·테스트 상태와 비교 방법을 고정한다.

작업:

1. Git SHA/dirty 파일을 기록하되 기존 사용자 변경을 수정하거나 포함하지 않는다.
2. M2 baseline의 dataset/corpus/vectorstore fingerprint와 현재 값을 비교한다.
3. Python/Node/Ollama/model, retrieval 설정과 동일 process warm-up 절차를 기록한다.
4. 기존 17 routing 실패, 8 assertion false negative, 3 abstention false negative의 case ID를 원본 report에서 추출한다.
5. 공식 report 디렉터리, candidate ID 정규식, v1/v2 evaluator version 규칙을 상세 설계에 확정한다.
6. 전체 정적 회귀를 실행하고 환경 기인 실패를 M3 변경과 분리한다. 명령 출력은 Git 제외 디렉터리에 로그로 저장하고 **경로와 SHA-256만** Phase 0 report에 남긴다.
7. 표준 라이브러리 전용 `scripts/check_markdown_links.py`를 구현하고 단위 테스트를 추가한다(상세 설계 §4.5). 이 스크립트는 제품 런타임 코드가 아니라 회귀 gate 도구다.

산출물:

- 상세 설계의 baseline/experiment contract
- `scripts/check_markdown_links.py`와 `tests/unit/test_check_markdown_links.py`
- Phase 0 report(로그 경로 + SHA-256 포함)와 Git 제외 실행 로그
- 요구사항 추적표 초안

검증:

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
python -m evaluation.fingerprint --dataset evaluation/datasets/golden.jsonl \
  --baseline evaluation/baselines/m2_initial.json
pytest -q
npm test
python scripts/check_markdown_links.py
git diff --check
```

수용 기준:

- 승인 baseline의 76/42/29 사례 구성과 모든 fingerprint가 확인된다.
- `expected_route` 기준 사례 수가 document_qa 61 / web_search 15로 확인되고, Requirement §4.1의 recall 분모와 일치한다.
- fingerprint 불일치가 있으면 공식 비교를 중단하고 원인을 해결하거나 사용자에게 새 비교 기준 승인을 요청한다.
- 기존 실패와 M3에서 새로 생긴 실패를 구분할 수 있다.
- 정적 회귀 수치가 재현 가능한 로그 artifact(경로 + SHA-256)로 참조된다.
- link checker가 저장소 전체 Markdown에서 깨진 로컬 링크 0을 보고한다.

관련 요구사항: M3-REQ-001, M3-REQ-010, M3-NFR-001~005

### Phase 1 — 비교 하네스와 Answer evaluator v2

목표: 제품 최적화 전에 비교와 알려진 evaluator 오판을 결정론적으로 검증한다.

작업:

1. M2 report를 읽어 후보 report와 지표·case 변화·fingerprint를 비교하는 경계를 설계한다.
2. assertion 정규화와 abstention detector를 versioned 순수 함수로 분리한다.
3. M2의 8+3 false negative, 기존 true positive, 숫자·부정·단위 반례를 fixture로 만든다.
4. v1 호환 결과와 v2 결과가 명시적으로 분리된 JSON/Markdown schema를 정의한다.
5. 공식 v2 실행에서 검토 변형 표의 schema/fingerprint 불일치를 fail-closed로 처리한다.
6. 상세 report/worksheet가 계속 Git 제외인지 확인한다.

신규 모듈은 선행 표면을 줄이기 위해 최소로 유지한다: `evaluation/answer_rules.py`, `evaluation/answer_variants.json`, `evaluation/rescore.py`, `evaluation/compare.py`(gate 판정 순수 함수 포함). gate 로직은 재사용 필요가 실제로 생길 때만 별 모듈로 추출한다. `evaluation/fingerprint.py`는 Phase 0 소관이며 기존 `reporting` 함수를 감싸는 얇은 CLI 이상으로 키우지 않는다.

산출물:

- evaluator v2와 단위/통합 테스트
- versioned report schema와 비교/재채점 도구
- 규칙 변경 근거 문서

검증:

```bash
pytest -q tests/unit/test_answer_rules.py tests/unit/test_evaluation_gates.py \
         tests/unit/test_evaluation_metrics.py \
         tests/integration/test_evaluation_answers.py \
         tests/integration/test_evaluation_rescore.py \
         tests/integration/test_evaluation_compare.py
python -m evaluation.answers --help
python -m evaluation.rescore --help
pytest -q
```

수용 기준:

- 확인된 11개 false negative를 모두 올바르게 판정한다.
- 기존 true positive 회귀가 0이고 의미가 다른 숫자/부정 반례를 합치지 않는다.
- 공식 v2 실행이 변형 파일 부재·schema 오류·fingerprint 불일치에서 exit 2로 실패하고, 변형 없는 실행은 별도 profile로만 가능하다.
- 기존 `m2_initial` 파일은 byte 수준으로 변경되지 않는다.
- live model 없이 모든 새 규칙을 검증할 수 있다.

관련 요구사항: M3-REQ-001, M3-REQ-006, M3-REQ-009, M3-NFR-001, M3-NFR-003~005

### Phase 2 — MMR 병목 제거

목표: 검색 품질 floor를 지키며 MMR 반복 임베딩 비용을 제거한다.

작업:

1. FAISS index/docstore ID mapping과 후보 `Document` 대응 계약을 상세 설계로 증명한다.
2. 최소 두 후보를 비용 순으로 평가한다: (a) 저장 vector 직접 재사용, (b) 안정 key의 bounded embedding cache. 대응을 증명할 수 없는 후보는 구현하지 않는다.
3. query embedding 횟수, candidate embedding 횟수, cache hit/miss 또는 vector lookup 실패를 테스트 가능한 관측값으로 만든다. 계측 객체(trace) 없이 호출되는 제품 경로에서도 폴백이 동일하게 동작하게 한다.
4. 동일 입력에서 기존/후보 문서 순위와 지표를 비교한다.
5. **동일 process 내 warm-up**(`--warmup-cases N`) 후 전체 Retrieval 42건 latency를 단독 실행한다. warm-up 사례는 집계에서 제외되고 report metadata로 기록된다.
6. §4.1 gate를 만족하는 최소 복잡도 후보만 기본 경로로 승격한다.

산출물:

- MMR 최적화와 단위/통합 테스트
- 후보별 Retrieval JSON/Markdown report
- 채택/기각 결정 기록

검증:

```bash
pytest -q tests/unit tests/integration/test_evaluation_retrieval.py
python -m evaluation.retrieval --help
SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE=stored python -m evaluation.retrieval \
  --dataset evaluation/datasets/golden.jsonl \
  --warmup-cases 3 --candidate-id m3-p2a-stored-vector \
  --output evaluation/reports/m3/m3-p2a-stored-vector/retrieval
python scripts/check_markdown_links.py
```

수용 기준:

- 후보 본문 embedding을 매 질문 반복하지 않는다는 테스트 증거가 있다.
- 잘못된 mapping/dimension/non-finite 입력이 명시적으로 처리되고, `trace` 제공/미제공 두 경로 모두에서 legacy와 동일한 결과로 폴백한다.
- report의 warm-up metadata가 동일 process 실행과 측정 제외를 증명한다.
- Requirement §4.1 Retrieval latency와 품질 floor를 모두 만족한다.
- 미달 시 현행 경로를 유지하고 Phase 6 후보에 포함하지 않는다.

관련 요구사항: M3-REQ-002~003, M3-REQ-009, M3-NFR-002, M3-NFR-004~005

### Phase 3 — 문서 우선 Routing 교정

목표: web recall 100%를 보존하면서 document QA 과다 웹 라우팅을 줄인다.

라우팅 단순화 사이클 1에서는 `Design.md` §7.2의 두 command grammar만 구현한다. 이전 Iteration 1~6의 `SOURCE_PARTICLE` fast path, `TOPIC_HEAD`, 조사 뒤 관형절 cue, 어절 거리 예외는 구현하지 않는다. 결정론 규칙이 잡지 못한 명시 표현은 LLM에 위임하며, 규칙 recall보다 결정론적 WEB precision 100%를 우선한다.

작업:

1. M2의 17개 실패에 Requirement §5의 taxonomy tag와 판정 이유를 부여한다.
2. 비용 순으로 prompt 수정, 명시적 신호 경계 규칙, 2단계 판정을 실험한다.
3. 첫 후보가 gate를 만족하면 더 복잡한 후보를 구현하지 않는다.
4. 명시 신호 판정을 **LLM 호출 이전**에 수행해 LLM 예외/no-tool에서도 WEB/DOCUMENT 우선순위와 tool query 계약이 보존되게 한다.
5. tool query 원형 보존, Agent/no-tool/exception fallback과 웹 실패 fallback 회귀 테스트를 추가한다. WEB·DOCUMENT 각각에 LLM 예외·no-tool 조합 테스트를 포함한다.
6. 전체 76건을 3회 live 실행해 중앙값과 변동 사례를 집계한다.
7. 모델 없는 분류 테스트에서 WEB 8 / DOCUMENT 12 / NONE 56 exact set과 WEB·DOCUMENT 오탐 0을 검증하고, 관형절 최소쌍 두 건을 필수 회귀 테스트로 고정한다.

산출물:

- routing taxonomy
- 선택된 routing 정책과 회귀 테스트
- 3회 live report 및 채택 결정

검증:

```bash
pytest -q tests/unit/test_query_router.py tests/unit/test_routing_signals.py \
         tests/integration/test_agent_routing_policy.py \
         tests/integration/test_evaluation_routing.py
python -m evaluation.routing --help
SIMPLE_QNA_RAG_ROUTING_SIGNAL_OVERRIDE=1 RUN_LIVE_LLM_TESTS=1 \
python -m evaluation.routing \
  --dataset evaluation/datasets/golden.jsonl \
  --mode live --runs 3 --candidate-id m3-p3a-signal-override \
  --output evaluation/reports/m3/m3-p3a-signal-override/routing
pytest -q
```

수용 기준:

- 세 run 각각 web search recall이 15/15다.
- 지표별 중앙값이 accuracy `69/76` 이상, document route recall `54/61` 이상이다(분모 61, Requirement §4.1의 단일 metric 계약).
- 판정은 반올림 백분율이 아니라 count/`Fraction` 비교로 수행한다.
- 명시 신호가 있는 질문이 LLM 예외·no-tool 상황에서도 요구된 route를 유지한다는 테스트 증거가 있다.
- 결정론적 WEB 판정 8건의 precision이 100%이고, `웹검색에서 사용하는 API 구조 알려줘` 및 `구글에서 사용하는 검색 기술 알려줘`가 NONE이다.
- 변동 사례와 17개 기존 실패의 개선/잔존/신규 회귀가 식별된다.
- 더 복잡한 2단계 LLM 판정은 단순 변경이 실패하고 추가 latency가 정당화될 때만 채택한다.

관련 요구사항: M3-REQ-004~005, M3-REQ-009, M3-NFR-001~002, M3-NFR-005

### Phase 4 — Intent Classifier 효용 실험과 경로 결정

목표: 분류 정확도 자체가 아니라 실제 답변 효용을 기준으로 별도 classifier의 존속을 결정한다.

작업:

1. Answer 29건의 retrieved context를 한 번 고정하고 두 template variant가 공유하게 한다.
2. 현행 intent별 template과 기본 template의 출력 worksheet에서 variant 정체와 순서를 가린다.
3. 질문 형식 적합성·핵심 사실 보존을 사례별 검토하고 자동 assertion/abstention/source 결과도 병기한다.
4. Requirement §4.2에 따라 유지/개선 또는 production 경로 단순화를 결정한다.
5. 유지 시에만 도메인 실패 사례 보강, confidence 사용 여부와 intent accuracy 개선을 설계한다. 유지 gate를 달성한 뒤 추가 학습 확대는 하지 않는다.
6. 단순화 시 public 응답의 `intent` 호환값과 모델 artifact 처리 방침을 설계 리뷰에서 확정한다.

산출물:

- blind paired worksheet와 집계 report
- Intent Architecture Decision Record
- 선택된 경로의 코드·테스트(결정상 필요할 때)

검증:

```bash
pytest -q tests/unit tests/integration/test_evaluation_answers.py tests/integration/test_agent.py
python -m evaluation.answers --help
python -m evaluation.answers \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports/m3/intent
```

수용 기준:

- 같은 question/context를 사용한 29개 paired 결과와 완성된 사람 검토가 있다.
- 결정이 Requirement §4.2의 사전 기준과 일치한다.
- 유지 시 intent 22/29 이상이며, 단순화 시 public 응답 계약과 fallback이 보존된다.
- 사용자 Gate가 결론과 호환성 방침을 승인한다.

관련 요구사항: M3-REQ-006~007, M3-REQ-009~010, M3-NFR-001, M3-NFR-003~005

### Phase 5 — 조건부 한국어 BM25 tokenizer 실험

목표: 낮은 비용으로 sparse 검색 개선 가능성을 확인하되 기술 도입을 완료 조건으로 만들지 않는다.

진입 조건: Phase 2 이후 Retrieval floor가 안정적이고, 사용자가 추가 실험 비용을 승인한다. 진입하지 않아도 M3는 완료할 수 있다.

작업:

1. 현행 whitespace tokenizer와 최대 두 개의 경량 후보를 선정한다.
2. 조사·어미·복합명사와 영문/숫자 혼합 fixture를 만든다.
3. BM25-only 순위와 전체 hybrid 42건을 분리 비교한다.
4. 초기화 시간, 메모리, dependency/배포 영향을 측정한다. 메모리는 동일 process RSS peak를 주 판정값으로 하고 `tracemalloc`은 진단값으로 병기한다.
5. Requirement §3.2를 모두 만족하는 경우에만 production 기본값 후보로 제안한다.

산출물:

- tokenizer A/B report
- 채택 또는 기각 결정
- 채택 시 tokenizer 경계와 테스트

검증:

```bash
pytest -q tests/unit/test_text_tokenizers.py tests/integration/test_evaluation_retrieval.py
python -m evaluation.experiments.bm25_tokenizer \
  --dataset evaluation/datasets/golden.jsonl \
  --tokenizers whitespace,char2gram,bge-subword \
  --output evaluation/reports/m3/m3-p5-bm25-offline/bm25_only
```

수용 기준:

- Recall@10 또는 nDCG@10이 1.00%p 이상 개선되고 모든 Retrieval floor를 지킨 경우만 채택한다.
- 초기화 시간 증가와 RSS peak 증가가 각각 20% 이하이며 필수 native dependency가 없다.
- 미달 시 production 코드/필수 dependency에 실험 잔재를 남기지 않고 기각 근거만 보존한다.

관련 요구사항: M3-REQ-003, M3-REQ-008~009, M3-NFR-002, M3-NFR-004~005

### Phase 6 — 통합 회귀, live 평가와 최종 승인

목표: 독립적으로 채택된 변경을 함께 실행해 상호작용 회귀가 없는지 확인하고 M3를 승인 가능한 상태로 만든다.

작업:

1. dataset validation과 전체 Python/frontend 회귀를 clean 환경에서 실행한다.
2. 최종 후보로 Retrieval 42건, Routing 76건×3회, Answer 29건과 통합 baseline을 실행한다.
3. Requirement §4의 모든 gate와 per-case 신규 회귀를 판정한다.
4. source/assertion/abstention 및 Intent worksheet 사람 검토를 완료한다.
5. 요구사항 추적표, 채택/기각 후보, 알려진 한계, 실행 비용과 잔여 문제를 요약한다.
6. 사용자 승인 후에만 immutable M3 baseline을 고정하고 Roadmap을 완료로 갱신한다.

검증:

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
  --output evaluation/reports/m3/m3-final
git diff --check
```

수용 기준:

- 모든 정적 테스트와 live 단계가 성공한다.
- Retrieval, Routing, Answer, source와 Intent 결정 gate가 모두 충족된다.
- dataset/corpus/vectorstore fingerprint가 승인 비교 조건과 일치한다.
- CRITICAL/MAJOR 0, MINOR 최소화, 리뷰 점수 9.7/10 이상이다.
- 사용자가 최종 결과와 baseline 고정을 명시적으로 승인한다.

관련 요구사항: 전체

## 5. Live 평가와 승인 절차

live 평가는 비용과 시간이 드는 외부 상태 의존 작업이므로 다음 Gate를 따른다.

1. 구현 담당자가 정적 테스트와 작은 opt-in preflight 결과를 제출한다.
2. 프로젝트 리더가 full Retrieval, Routing 3회, paired Answer 실행 범위와 예상 시간을 확인한다.
3. 사용자가 live 실행을 승인한다.
4. 실행 중 일부 단계가 실패해도 성공 report를 덮어쓰지 않고 실패 원인과 partial 결과를 보존한다.
5. reviewer가 fingerprint, run count, warm-up, 오류 수와 metric gate를 확인한다.
6. 사람이 worksheet를 검토하고 Intent/BM25 결정에 동의한다.
7. 사용자가 최종 승인한 뒤 `evaluation/baselines/`에 M3 baseline을 고정한다.

승인 baseline에는 상세 질문·답변 대신 집계, 비민감 실패 taxonomy, 실행 metadata, fingerprint, 승인 시각과 원본 local report 식별 경로만 포함한다.

## 6. Rollback 및 중단 기준

- 품질 floor 하나라도 깨진 후보는 기본 경로로 승격하지 않는다.
- mapping 정확성을 증명할 수 없는 vector 재사용, web recall 손실, evaluator의 의미 합침은 즉시 기각한다.
- 새 변경이 기존 public 계약 또는 runtime artifact를 깨면 feature flag/이전 경로로 되돌리고 원인을 기록한다.
- 환경 fingerprint 불일치, Ollama/model 부재 또는 hardware 변동으로 공식 비교가 불가능하면 결과를 “실패”가 아닌 “비교 불가”로 표시하고 사용자 결정을 요청한다. warm-up 미실행·실패, MMR 폴백 발생, evaluator v2 변형 표 불일치도 같은 “판정 불가” 처리 대상이다.
- 최대 4회 리뷰 후 품질 gate 미달이면 무리하게 범위를 넓히지 않고 중단 보고서를 작성한다.

## 7. 최종 산출물 체크리스트

- [ ] 상세 설계와 요구사항 추적표
- [ ] Markdown local link 검사기(`scripts/check_markdown_links.py`)와 Phase 0 로그 artifact 경로·SHA-256
- [ ] Answer evaluator v2 및 회귀 fixture
- [ ] 채택된 MMR 최적화와 Retrieval 비교 report
- [ ] Routing taxonomy, 정책과 3회 live report
- [ ] Intent paired worksheet와 결정 기록
- [ ] BM25 실험 결과(진입한 경우)와 채택/기각 근거
- [ ] 전체 Python/frontend 테스트 결과
- [ ] 최종 live baseline report와 사람 검토 결과
- [ ] 사용자 승인된 M3 immutable baseline
- [ ] 승인 이후 Roadmap/Problem 상태 정리
