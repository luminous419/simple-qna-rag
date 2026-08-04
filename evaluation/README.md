# evaluation/ — M2 Quality Baseline

`evaluation/`은 골든 평가셋 검증, Retrieval·Routing·Answer 평가, 통합 live baseline과 결과 리포팅을 제공한다. 이 문서는 평가셋 작성법, source ID, metric 정의, 실행 방법과 결과 해석의 한계를 정리한다.

## 골든 사례 스키마

각 사례는 `evaluation/schema.py`의 `GoldenCase`로 검증된다. 필드 의미는
[Development_M2_Quality_Baseline_Requirement.md](../Development_M2_Quality_Baseline_Requirement.md)
M2-REQ-003에 정의돼 있다. 요약:

- `id`, `question`, `category`, `expected_route`, `tags`: 필수.
- `category`: `document_qa` / `web_search` / `boundary` / `unanswerable`.
- `relevant_sources`: Retrieval 정답 source ID 목록 (아래 "source ID 규칙" 참고).
- `relevance_grades`: nDCG용 0~3 등급. 값이 있으면 최소 1개는 양수(1~3)여야 하고, 양수 등급을 받은
  source는 정규화 후 `relevant_sources`에도 반드시 포함돼야 한다(grade 0은 예외).
- `answer_assertions`: Answer 평가용 핵심 사실 목록. 아래 "assertion 작성법" 참고.
- `expect_abstention`: 모델이 답변을 거절해야 하는 사례에만 `true`. `answer_assertions`와
  동시에 설정할 수 없다(schema에서 거부됨) — 하나의 답변이 "핵심 사실을 포함"과 "답을 거절" 둘 다를
  동시에 만족할 수 없기 때문이다.
- `expected_intent`: `explanation` / `comparison` / `procedure` / `yesno` / `other` / `uncertain` 중 하나.

## source ID 규칙

`document_register.py`의 `DirectoryLoader`가 `data/**/*.pdf`, `data/**/*.txt`만 수집한다 —
**확장자가 없는 파일은 인덱싱되지 않으므로 골든셋의 정답 source가 될 수 없다.** 새 문서를
`data/`에 추가할 때는 반드시 `.pdf` 또는 `.txt` 확장자를 붙인다.

`relevant_sources`/`relevance_grades`의 key는 `data/` 파일의 **basename**(전체 경로 아님)을
그대로 적는다. 비교는 `evaluation/schema.py`의 `normalize_source_id()`(NFC 정규화 → 경로 구분자
통일 → basename 추출 → casefold) 기준으로 이뤄지므로, 대소문자·Unicode 정규화 형태·경로 구분자
차이는 같은 source로 취급된다. 다만 이는 사후 검증일 뿐 — 골든셋 저작 시점에는 사람이 `data/`의
정확한 파일명을 그대로 옮겨 적어야 한다.

**정답 근거를 실제로 검색할 수 있는지 항상 확인한다.** `python -c` 등으로 파일이
`PyPDFLoader`/`TextLoader`가 지원하는 확장자를 갖는지, 그리고 이상적으로는 vectorstore 재생성 후
실제 후보로 검색되는지 확인한다. PDF는 슬라이드형 문서가 많아 다이어그램에 박힌 텍스트는
`pypdf`가 추출하지 못한다 — 아래 명령으로 실제 추출 텍스트를 먼저 확인한 뒤에만 그 내용을 근거로
삼는다.

```bash
python -c "
from pypdf import PdfReader
r = PdfReader('data/파일명.pdf')
for i, page in enumerate(r.pages):
    print(f'--- p{i+1} ---')
    print(page.extract_text())
"
```

## category/intent 구성 원칙

- 개별 `GoldenCase`는 "이 사례가 어떤 평가에 쓰일지"를 강제하지 않는다. `answer_assertions`도
  `expect_abstention`도 없는 `document_qa` 사례는 Retrieval 평가 전용 사례로서 그 자체로 유효하다.
- 데이터셋 전체 구성 규칙(M2-REQ-002, 9개)은 `evaluation/dataset.py`의 `validate_composition()`이
  검사한다 — 사례 하나만 봐서는 이 사례가 그 집계 안에서 어떤 역할을 하는지 알 수 없기 때문이다.
- `explanation`/`comparison`/`procedure`/`yesno` intent 최소 수량(각 5개 이상)은 **Answer 평가
  대상 사례에서만** 집계된다 — `evaluation/schema.py`의 `is_answer_eval_eligible()`
  (`answer_assertions`가 있거나 `expect_abstention=true`)을 만족하는 사례만 해당한다.
  `web_search`나 Retrieval 전용 `document_qa` 사례에 intent만 붙여도 이 최소치를 채울 수 없다.

## assertion 작성법

`answer_assertions`는 `AnswerAssertion` 객체의 리스트다. 각 객체의 `any_of`는 **같은 필수 사실의
동의어·표기 변형**만 나열한다(하나라도 포함되면 그 객체는 통과). 서로 **독립적으로 둘 다 필요한
사실**은 절대 하나의 `any_of`에 함께 넣지 말고 별도의 `AnswerAssertion` 객체로 분리한다 — 평가기가
객체 단위로 "통과한 assertion 수 / 전체 assertion 수"를 세므로, 독립 사실을 한 객체에 합치면 절반만
맞는 답도 그 객체 기준으로는 100% 통과한 것처럼 보인다.

```python
# 잘못된 예 — "단방향"과 "순환"은 서로 다른 두 시스템의 서로 다른 특징이라 하나만 있어도
# 이 assertion은 통과해버린다.
{"any_of": ["단방향", "순환"]}

# 올바른 예 — 두 특징을 각각 별도 assertion으로 분리해 둘 다 있어야 만점이 되도록 한다.
[{"any_of": ["단방향"]}, {"any_of": ["순환", "Cycle", "사이클"]}]
```

비교(`comparison`)·절차(`procedure`) 유형 질문은 특히 주의한다 — 비교는 보통 양쪽 특징을 모두
요구하고, 절차는 여러 단계를 모두 요구하는 경우가 많다. 다만 원문이 명시적으로 보장하지 않는
세부 단계까지 assertion으로 과도하게 강제하지는 않는다(문서에 없는 사실을 강요하면 실제로는 맞는
답도 감점된다).

## abstention 작성법

`category=unanswerable`이면서 `expect_abstention=true`인 사례는 **현재 corpus(`data/`)에서
실제로 답을 찾을 수 없는 질문**이어야 한다. 저작 시 다음을 확인한다.

1. corpus 18개 문서 어디에도 해당 사실이 없는지 실제로 확인한다(추측 금지).
2. `answer_assertions`는 비워 둔다 — `expect_abstention=true`와 동시에 설정할 수 없다(schema가
   거부한다).
3. 질문 주제가 corpus의 도메인과 이름만 비슷한 다른 대상(예: 국내 기업 대신 해외 기업, 다른 연도)을
   가리키도록 만들면 "모델이 착각해 사전지식으로 답할" 위험을 줄일 수 있다.

`category=boundary`는 라우팅 경계 사례(예: "이 문서에서 관련 내용을 찾아줘")로, abstention을
요구하지 않는다.

## validator 실행법

```bash
# schema/구성 규칙 검증 (M2-REQ-004)
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl

# import 자체가 깨지지 않는지, CLI 진입점이 살아있는지 빠르게 확인
python -c "import evaluation.schema, evaluation.dataset"
python -m evaluation.dataset --help

# Phase 1 전용 테스트 + 전체 회귀
pytest -q test_evaluation_schema.py test_evaluation_dataset.py
pytest -q
```

`validate` 명령은 성공 시 종료 코드 0과 함께 `total`/`by_category`/`document_qa_with_*`/
`answer_eval_case_count`/`intent_counts`/`korean_ratio` 등을 담은 JSON 리포트를 stdout에,
실패 시 각 미달 규칙을 stderr에 사람이 읽을 수 있는 문장으로 출력한다.

## 사람 검토 게이트

Development_M2_Quality_Baseline_Development_Plan.md §5.3에 따라 골든셋 변경 후에는 다음 두 검토를
받아야 Phase 2를 완료 처리한다.

1. **source relevance 검토**: 각 `relevant_sources`/`relevance_grades`가 실제 문서 내용과 맞는지.
2. **answer assertion 검토**: 각 `answer_assertions.any_of`가 실제 문서 내용과 일치하고, 독립적인
   여러 사실을 하나의 `any_of`에 합치지 않았는지, 표현이 지나치게 좁지 않은지.

## 알려진 한계

- assertion coverage는 답변에 핵심 문구가 "포함되는지"만 확인하는 보조 지표이며, 답변의 진실성이나
  문맥적 정확성을 완전히 대체하지 않는다(M2-REQ-008).
- 현재 이 프로젝트에는 LLM judge 기반 평가가 포함되어 있지 않다 — assertion coverage와
  abstention 정확도 같은 규칙 기반 지표만 사용한다.
- corpus manifest와 vectorstore fingerprint는 동일한 파일을 사용했는지는 식별하지만, 인덱스가 현재 embedding model과 chunk 설정으로 생성됐다는 provenance는 보증하지 않는다.
- dependency version snapshot은 현재 baseline metadata에 없으므로 같은 Git과 fingerprint라도 설치된 라이브러리 차이가 남을 수 있다.

## 평가 대상 선택 규칙

| Evaluator | 대상 사례 |
|---|---|
| Retrieval | `relevant_sources`가 하나 이상인 사례 |
| Routing | 필터 후 남은 모든 사례 |
| Answer | `answer_assertions`가 있거나 `expect_abstention=true`인 사례 |

`--tag`는 해당 tag를 가진 사례만 남기고 `--limit`은 필터 결과의 원래 순서 앞에서부터 양의 개수만 선택한다. `0`과 음수 limit은 CLI와 공개 API 모두 거부한다.

## Metric 정의

### Retrieval

- **Recall@K**: 정답 source 중 top-K 고유 검색 source에 포함된 비율을 사례별로 계산한 뒤 macro average한다.
- **MRR@10**: 첫 번째 정답 source의 reciprocal rank를 사례별로 계산한 뒤 평균한다.
- **nDCG@10**: `relevance_grades`가 있는 사례만 대상으로 `gain = 2**grade - 1`, `discount = log2(rank + 1)`을 사용해 사례별 nDCG를 계산한 뒤 macro average한다.
- 세 metric 모두 source ID를 정규화하고 중복 source는 최초 등장만 유지한다. K는 chunk 수가 아니라 top-K 고유 source 수다.
- latency는 실제 production Retrieval 전체와 BM25, Dense, RRF, MMR, reranker 단계별 mean/median/p95 및 후보 수를 기록한다.

### Routing

- 전체 accuracy와 `document_qa`/`web_search`별 precision, recall, F1을 계산한다.
- no-tool, unknown route와 exception은 제외하지 않고 각각 별도 prediction 열로 confusion matrix에 포함한다. 따라서 기대 route의 false negative로 반영된다.
- live mode는 실제 `agent._decide_tool()`을 사용하고 offline mode는 외부 LLM 없이 parsing·집계·리포팅 계약을 검증한다.

### Answer

- **Assertion coverage**: 각 assertion의 `any_of` 중 하나가 정규화된 답변 문자열에 포함되면 통과한다. 전체 통과 assertion 수를 전체 assertion 수로 나눈다.
- **Abstention accuracy**: 기대 abstention과 규칙 기반 detector가 판정한 거절 여부의 일치율이다.
- **Source any-hit/recall**: 반환 source와 `relevant_sources`를 정규화해 하나 이상 맞았는지와 정답 source 회수 비율을 계산한다.
- **Intent accuracy**: 반환 intent와 `expected_intent`가 같은 비율이다.
- End-to-End latency는 Retrieval, intent 분류와 LLM 답변 생성을 포함한다.

Assertion과 abstention은 문자열 규칙이라 표기·띄어쓰기·동의 표현에 민감하다. M2 최초 baseline에서도 의미상 맞는 답변을 자동 실패로 판정한 사례가 있었으므로 worksheet 사람 검토와 함께 해석해야 한다.

## 실행 방법

모든 명령은 저장소 루트에서 실행한다.

### Dataset validation

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
```

### Retrieval

```bash
python -m evaluation.retrieval \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports/retrieval
```

실제 `data/`, `vectorstore/`, embedding과 reranker가 필요하다.

### Routing

```bash
# 외부 모델 없는 집계·리포트 확인
python -m evaluation.routing \
  --dataset evaluation/datasets/golden.jsonl \
  --mode offline \
  --output evaluation/reports/routing-offline

# 실제 Ollama tool decision
RUN_LIVE_LLM_TESTS=1 python -m evaluation.routing \
  --dataset evaluation/datasets/golden.jsonl \
  --mode live \
  --output evaluation/reports/routing
```

### Answer

```bash
python -m evaluation.answers \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports/answers
```

JSON/Markdown 집계 report와 사례별 사람 검토 worksheet를 만든다. worksheet에는 답변과 출처가 포함될 수 있으므로 Git에 commit하지 않는다.

### 통합 baseline

```bash
RUN_LIVE_LLM_TESTS=1 python -m evaluation.baseline \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports
```

실행 순서는 validation → Retrieval → live Routing → Answer다. 한 단계가 실패해도 가능한 다음 단계를 실행하고 완료된 결과와 실패 원인을 보존한 뒤 전체 명령은 non-zero로 끝난다.

Retrieval의 corpus/vectorstore fingerprint를 top-level에 기록하고 Answer가 독립 계산한 값과 비교한다. 둘이 다르면 실행 중 artifact가 변경된 것으로 보고 전체 baseline을 실패 처리한다.

`--limit 1`은 환경 사전 점검에만 사용한다. 제한 실행, skip 실행이나 일부 tag 실행을 정식 최초 baseline으로 고정하지 않는다.

## 리포트와 기준선

- `evaluation/reports/`: timestamped 상세 JSON/Markdown/worksheet. Git 제외 대상
- `evaluation/baselines/m2_initial.json`: 사용자가 승인한 기계 판독용 최초 기준선
- `evaluation/baselines/m2_initial.md`: 사람 판독용 최초 기준선과 해석

상세 report는 질문과 모델 답변을 포함할 수 있다. 승인 기준선에는 집계 수치, 실행 metadata, 전체 corpus manifest, fingerprint와 비민감 실패 패턴만 기록한다.

비교 실행에서는 최소한 다음이 같은지 확인한다.

1. dataset SHA-256
2. corpus manifest SHA-256
3. `index.faiss`와 `index.pkl` SHA-256
4. Git revision 또는 비교 대상 변경 범위
5. 모델 이름과 Retrieval 설정

fingerprint가 다르면 동일 조건의 전후 비교로 간주하지 않는다.

## CI와 로컬 live 실행 차이

GitHub Actions는 Pull Request와 `master` push에서 다음 두 job을 실행한다.

- `python-tests`: Python 3.11, dependency check, Web import, `pytest -q`, dataset validation
- `frontend-tests`: Node 22, `npm ci`, `npm test`, vendor sync 및 diff 확인

CI는 Ollama, DDGS, Hugging Face 모델 가중치, `data/`, `vectorstore/`와 secret을 요구하지 않는다. 실제 품질 및 latency baseline은 준비된 로컬 환경에서 명시적 opt-in으로만 실행하고 사용자 검토 후 고정한다.
