# evaluation/ — M2 Quality Baseline

`evaluation/datasets/golden.jsonl` 골든 평가셋을 작성·검증하는 방법을 정리한다. 이 문서는 Phase 2
산출물로 시작했으며(Development_M2_Quality_Baseline_Development_Plan.md §5), Phase 3 이후 metric·CI
내용이 추가되면 갱신한다.

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
- Metric(Recall/MRR/nDCG 등) 정의와 계산 방식은 Phase 3(`evaluation/metrics.py`) 구현 시 이
  문서에 추가한다.
