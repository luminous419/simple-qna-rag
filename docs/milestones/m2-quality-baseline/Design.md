# M2 Phase 1 설계 — 평가 schema와 dataset validator

## 0. 문서 관계

- 상위 계획: [Development_M2_Quality_Baseline_Development_Plan.md](Development_Plan.md) §3.1~3.3, §4 Phase 1
- 요구사항: [Development_M2_Quality_Baseline_Requirement.md](Requirement.md) M2-REQ-001~004, M2-REQ-011(경계 테스트 일부), M2-NFR-001/003/005
- 검토: [design_review.md](../../reviews/m2-quality-baseline/Design_Review.md) — 이 설계 문서에 대한 외부 검토. 1차(종합 평가 8.2/10, P1 3건·P2 3건·P3 1건), 2차(종합 평가 9.0/10, P1 2건·P2 2건·P3 1건), 3차(종합 평가 9.2/10, P1 1건·P2 2건·P3 1건) 모두 검토 완료(§2.1, §3.1, §9.1/§9.2/§9.3 참고). 3차 P1("model_validator import 누락")은 검증 결과 이미 2차 반영 시점에 수정돼 있어 현재 파일에는 해당하지 않았다(§9.3).
- 코드 리뷰: [M2_Phase1_code_review_result.md](../../reviews/m2-quality-baseline/Phase_1_Code_Review.md) — 실제로 작성된 `evaluation/schema.py`/`evaluation/dataset.py`/테스트에 대한 리뷰(종합 평가 8.8/10, P1 1건·P2 1건·P3 2건). 모두 반영 완료(§2.1, §3.1, §9.4 참고).
- 범위: **Phase 1만** 다룬다 — `evaluation/schema.py`, `evaluation/dataset.py`, 관련 테스트, `requirements.txt` 한 줄. Phase 3(`metrics.py`/`reporting.py`), Phase 4 이후(`build_corpus_manifest()` 등)는 이 문서의 범위 밖이며 상위 계획에만 정의되어 있다.
- 이 문서는 상위 계획의 설계를 그대로 반복하지 않고, 실제로 작성할 코드 수준까지 내려간다. 아래 코드는 구현 시 그대로 사용 가능한 수준을 목표로 하며, 실제 작성 중 세부 조정이 있으면 이 문서를 갱신한다.

## 1. 파일 구조

```text
evaluation/
├── __init__.py       # 빈 파일
├── schema.py          # 이 문서에서 설계
└── dataset.py          # 이 문서에서 설계

test_evaluation_schema.py   # 이 문서에서 설계
test_evaluation_dataset.py  # 이 문서에서 설계
```

`evaluation/datasets/golden.jsonl`은 Phase 2 산출물이므로 이 문서에서 다루지 않는다. `evaluation/dataset.py`는 이 파일이 아직 없어도 import/실행 가능해야 한다(빈 인자로 CLI를 부르면 "파일 없음" 오류를 내야지, import 자체가 깨지면 안 됨).

## 2. `evaluation/schema.py`

### 2.1 설계 원칙

- `data/`, `vectorstore/`, 모델을 전혀 건드리지 않는다(M2-NFR-003) — import는 순수 Python/Pydantic만으로 끝나야 한다.
- **개별 `GoldenCase`는 "이 사례가 어떤 평가에 쓰일지"를 강제하지 않는다.** 예를 들어 `answer_assertions`도 `expect_abstention`도 없는 `document_qa` 사례는 그 자체로 유효하다(Retrieval 평가 전용 사례). 이 규칙은 상위 계획 §3.2에서 이미 정정한 사항이며, 집계 규칙(전체 구성에서 20개 이상 등)만 `dataset.py`의 `validate_composition()`이 검사한다.
- `relevant_sources` 리스트 **내부**의 정규화 중복만 `GoldenCase` 필드 검증에서 잡는다(순수 리스트 검사, I/O 없음). `data/`의 실제 파일끼리 충돌하는지는 Phase 3/4의 `build_corpus_manifest()`가 담당한다(상위 계획 §3.3) — 이 문서에서 구현하지 않는다.
- **(design_review.md 반영) `tags`는 M2-REQ-003이 명시한 필수 필드다.** `Field(default_factory=list)`로 두면 필드 자체가 누락돼도 통과해버려 요구사항과 어긋난다. `tags: list[str]`(기본값 없음)로 선언해 필드 존재는 강제하되, Requirement가 "비어 있지 않음"까지 요구하지는 않으므로 **빈 배열(`[]`)은 허용**한다.
- **(design_review.md 반영) source/식별자 문자열은 "존재하지만 의미 없는 값"을 별도로 차단한다.** 빈 문자열, 공백만 있는 문자열, `normalize_source_id()` 결과가 빈 문자열이 되는 값(예: 경로 구분자로만 이뤄진 `"a/b/"`)은 `relevant_sources`와 `relevance_grades` key 양쪽에서 거부한다 — 그렇지 않으면 실제로는 정답 source가 없는 사례가 `MIN_DOCUMENT_QA_WITH_SOURCES` 같은 구성 최소치를 허위로 충족시킬 수 있다.
- **(design_review.md 반영) 정규화 후 중복 검사는 완전히 동일한 표기의 중복도 포함한다.** 이전 버전은 `["x.pdf", "x.pdf"]`처럼 raw 문자열이 완전히 같으면 통과시키는 결함이 있었다 — `norm in seen`만으로 판단하도록 고쳤다. `relevant_sources`와 `relevance_grades` key 양쪽에 동일한 `_normalize_dedupe()` 헬퍼를 적용해 규칙을 하나로 통일한다.
- **(design_review.md 1차 반영) `relevance_grades`의 key 정규화 계약을 명시한다.** schema는 저장된 key의 원본 표기(대소문자, 경로 형태)를 그대로 보존하며 자동 변환하지 않는다 — 대신 Phase 3/4 evaluator가 retrieval 결과 doc_id와 `relevance_grades` key 양쪽을 **반드시 `normalize_source_id()`로 정규화한 뒤 비교**해야 한다는 계약을 `normalize_source_id()` docstring에 명시한다.
- **(design_review.md 1차 반영) `extra="forbid"`와 `expect_abstention`의 엄격한 bool 타입을 강제한다.** 골든셋은 사람이 JSON을 직접 작성하는 장기 기준선 자료이므로, 오타로 추가된 알 수 없는 필드나 `"true"`/`"yes"`/`1` 같은 문자열·숫자가 `expect_abstention`으로 조용히 강제 변환되는 것보다 저작 오류를 즉시 드러내는 편이 안전하다. `pydantic`의 기본 lax 모드는 이런 값을 모두 `bool`로 자동 변환하므로(직접 실행 검증으로 확인, §9), `GoldenCase`/`AnswerAssertion`에 `ConfigDict(extra="forbid")`를 적용하고 `expect_abstention`에 `mode="before"` strict 검사를 추가한다.
- **(design_review.md 2차 P1 반영) `relevance_grades`의 양수 등급(1~3) source는 `relevant_sources`에도 반드시 포함되어야 한다.** 1차 리뷰 직후에는 두 필드를 완전히 독립적으로 뒀으나(정규화 후 `relevant_sources`와 `relevance_grades` key 집합이 서로 부분집합일 필요 없음), 2차 리뷰에서 `relevance_grades={"a.pdf": 3}`만 있고 `relevant_sources=[]`인 사례가 schema는 통과하지만 상위 계획 §3.2가 정의한 Retrieval evaluator eligibility(`relevant_sources`가 비어 있지 않은 사례만 대상)에서는 제외돼 "nDCG 정답은 있는데 Recall/MRR 대상에서는 빠지는" 모순을 지적했다. **오직 grade 0(비관련)인 source만 `relevant_sources` 밖에 존재할 수 있다** — grade 0은 nDCG 계산에서 "이 문서는 검색됐지만 무관하다"는 정보로 쓰이므로 relevant_sources에 없는 게 자연스럽지만, grade 1~3(관련)인 source가 relevant_sources 밖에 있으면 저작 실수로 간주한다. 이 규칙은 두 필드 모두 개별 검증이 끝난 뒤에만 비교할 수 있으므로 `field_validator`가 아니라 `model_validator(mode="after")`로 구현한다.
- **(design_review.md 2차 P2 반영) `_normalize_dedupe()`와 `grades_in_range`는 비문자열 값에 대해 방어적이다.** `relevant_sources`/`relevance_grades`는 JSON을 통해서만 채워지면 항상 문자열 key/원소이지만, `GoldenCase.model_validate()`를 Python 코드(예: Phase 3/4 evaluator나 테스트 코드)에서 직접 호출하며 `{1: 2}`처럼 비문자열 key를 담은 dict를 넘기면 `raw.strip()` 호출이 `AttributeError`로 죽어 pydantic이 아닌 raw traceback이 새어나갈 수 있음을 실행 검증으로 확인했다(§9.2). `isinstance(source, str)` 검사를 먼저 수행해 항상 깔끔한 `ValidationError`로 변환한다.
- **(design_review.md 3차 P2 반영) Answer 평가 eligibility 판정 함수를 이 모듈에 공개 함수로 한 번만 정의한다.** 이전에는 `dataset.py`(Phase 1, intent 최소 수량 집계용)와 상위 계획의 `answers.py`(Phase 6, `evaluate_answers()` 채점 대상 판정용)에 이름과 로직이 동일한 `_is_answer_eval_eligible()`이 각각 독립적으로 정의돼 있었다 — 한쪽만 바뀌면 두 Phase의 "Answer 평가 대상" 정의가 다시 갈라질 위험이 있었다. `normalize_source_id()`와 같은 이유로 `evaluation/schema.py`에 공개 함수 `is_answer_eval_eligible()`로 한 번만 정의하고 `dataset.py`/`answers.py` 둘 다 여기서 import해 쓴다.
- **(M2_Phase1_code_review_result.md P1 반영) `answer_assertions`와 `expect_abstention=true`는 동시에 설정할 수 없다.** assertion은 "모델이 답하면 이 핵심 사실을 포함해야 한다"는 기대이고 abstention은 "모델이 답을 거절해야 한다"는 기대라서, 하나의 답변이 두 기대를 동시에 만족할 수 없다 — 정상적으로 거절하면 assertion coverage가 실패하고 assertion을 포함해 답하면 abstention 정확도가 실패해, 모델 품질이 아니라 정답 schema 자체의 모순으로 Phase 6 baseline 점수가 왜곡될 수 있다. 실제 구현 코드를 검토하며 이 조합이 조용히 통과하고 `is_answer_eval_eligible()`이 이를 eligible로 판정한다는 사실을 실행으로 확인했다(§9.4). `model_validator(mode="after")`로 거부하며, `is_answer_eval_eligible()`은 이 validator를 통과한 `GoldenCase`만 받으므로 기존 OR 판정식을 그대로 유지한다.
- **(M2_Phase1_code_review_result.md P2 반영) `relevance_grades`가 비어 있지 않다면 최소 1개는 양수(1~3) 등급이어야 한다.** 전부 0(비관련)인 grade mapping은 이전 validator(양수 등급-`relevant_sources` 일치 검사)를 그냥 통과했는데, nDCG의 IDCG는 등급을 내림차순 정렬한 뒤 계산하므로 모든 등급이 0이면 IDCG도 항상 0이 되어 어떤 검색 결과를 넣어도 nDCG가 무의미하다(상위 계획 §3.5 `ndcg_at_k()` 공식 참고). 골든셋 저자가 grade mapping을 채워 넣고도 그 사례가 사실상 평가되지 않는다는 사실을 놓치기 쉬우므로, `relevance_grades` 필드에 별도 `field_validator`(mode="after", 기본)를 추가해 저작 시점에 차단한다. 등급이 필요 없는 사례는 `relevance_grades` 자체를 생략하면 되므로 빈 `dict`는 이 검사에 걸리지 않는다.

### 2.2 전체 코드

```python
"""
평가 골든 케이스 데이터 모델 (M2-REQ-003).

이 모듈은 data/, vectorstore/, 모델을 전혀 사용하지 않는다. import 시점에
부수효과가 없어야 하며(M2-NFR-003), `python -m evaluation.dataset validate`가
Ollama나 벡터스토어 없이도 항상 동작해야 하기 때문이다.
"""

from __future__ import annotations

import os
import unicodedata
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"


class Category(str, Enum):
    DOCUMENT_QA = "document_qa"
    WEB_SEARCH = "web_search"
    BOUNDARY = "boundary"
    UNANSWERABLE = "unanswerable"


class Route(str, Enum):
    DOCUMENT_QA = "document_qa"
    WEB_SEARCH = "web_search"


class Intent(str, Enum):
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    PROCEDURE = "procedure"
    YESNO = "yesno"
    OTHER = "other"
    UNCERTAIN = "uncertain"


def normalize_source_id(raw: str) -> str:
    """source 비교용 정규화: NFC 정규화 -> 경로 구분자 통일 -> basename -> casefold.

    Retrieval/Answer evaluator(Phase 4/6)도 이 함수 하나만 사용해 source를
    비교한다 — 정규화 규칙이 여러 곳에 흩어지면 미묘하게 갈라질 위험이 있다.
    `relevance_grades`의 key를 조회할 때도 evaluator는 반드시 이 함수로 두
    값(retrieval 결과 doc_id, grade key)을 모두 정규화한 뒤 비교해야 한다 —
    schema는 저장된 key의 표기(대소문자, 경로 형태)를 그대로 보존하며 자동으로
    바꾸지 않는다(사람이 읽기 쉬운 원본 표기를 유지하기 위함, design_review.md P2).
    """
    value = unicodedata.normalize("NFC", raw)
    value = value.replace("\\", "/")
    value = os.path.basename(value)
    return value.casefold()


def _normalize_dedupe(raw_values: list[str], *, field_name: str) -> None:
    """raw_values 내부에서 빈 값, 정규화 후 빈 값, 정규화 후 중복(완전히 동일한
    표기 포함)이 있으면 ValueError를 던진다. relevant_sources와
    relevance_grades key 양쪽이 이 헬퍼 하나를 공유해 규칙을 통일한다
    (design_review.md 1차 P1). `isinstance` 검사는 `GoldenCase.model_validate()`를
    Python에서 직접 호출하며 비문자열 값을 넘기는 경로를 방어한다
    (design_review.md 2차 P2)."""
    seen: dict[str, str] = {}
    for raw in raw_values:
        if not isinstance(raw, str):
            raise ValueError(f"{field_name}의 항목은 문자열이어야 합니다: {raw!r}")
        if not raw.strip():
            raise ValueError(f"{field_name}에 빈 문자열/공백 항목이 있습니다: {raw!r}")
        norm = normalize_source_id(raw)
        if not norm:
            raise ValueError(
                f"{field_name}의 {raw!r}가 정규화 후 빈 문자열이 됩니다 "
                f"(경로 구분자로만 이뤄진 값 등). 유효한 파일명을 사용하세요."
            )
        if norm in seen:
            raise ValueError(
                f"{field_name}에 정규화 후 중복되는 항목이 있습니다: "
                f"{seen[norm]!r}와 {raw!r}가 모두 {norm!r}로 정규화됩니다"
            )
        seen[norm] = raw


class AnswerAssertion(BaseModel):
    """답변 핵심 사실 규칙 하나. any_of 중 하나 이상이 답변에 포함되면 통과."""

    model_config = ConfigDict(extra="forbid")

    any_of: list[str]

    @field_validator("any_of")
    @classmethod
    def non_empty(cls, v: list[str]) -> list[str]:
        if not v or not all(s.strip() for s in v):
            raise ValueError(
                "any_of는 비어 있지 않은 문자열을 하나 이상 포함해야 합니다"
            )
        return v


class GoldenCase(BaseModel):
    """골든 평가셋의 사례 하나 (M2-REQ-003).

    이 클래스는 필드 형식만 검증한다. "document_qa 20개 이상 answer_assertions
    보유" 같은 데이터셋 전체 집계 규칙은 여기서 검사하지 않는다
    (dataset.validate_composition 참고, §3.2 설계 정정 사항).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    category: Category
    expected_route: Route
    expected_intent: Optional[Intent] = None
    relevant_sources: list[str] = Field(default_factory=list)
    relevance_grades: dict[str, int] = Field(default_factory=dict)
    answer_assertions: list[AnswerAssertion] = Field(default_factory=list)
    expect_abstention: bool = False
    tags: list[str]
    notes: Optional[str] = None

    @field_validator("id")
    @classmethod
    def id_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("id는 비어 있을 수 없습니다")
        return v

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question은 비어 있을 수 없습니다")
        return v

    @field_validator("expect_abstention", mode="before")
    @classmethod
    def expect_abstention_strict_bool(cls, v: object) -> object:
        """pydantic 기본(lax) 모드는 "true"/"yes"/1/"1" 같은 값도 bool로
        강제 변환한다(직접 실행 검증으로 확인, §9). 골든셋은 사람이 JSON을
        직접 작성하므로 따옴표 붙은 "true" 같은 저작 실수를 조용히 통과시키지
        않기 위해 실제 bool 타입만 허용한다(design_review.md P2)."""
        if v is not None and not isinstance(v, bool):
            raise ValueError(
                f"expect_abstention={v!r}은(는) JSON true/false여야 합니다 "
                f"(문자열이나 숫자는 허용되지 않습니다)"
            )
        return v

    @field_validator("relevance_grades", mode="before")
    @classmethod
    def grades_in_range(cls, v: object) -> object:
        if not isinstance(v, dict):
            return v
        for source, grade in v.items():
            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    f"relevance_grades key는 비어 있지 않은 문자열이어야 합니다: {source!r}"
                )
            if not isinstance(grade, int) or isinstance(grade, bool) or not (0 <= grade <= 3):
                raise ValueError(
                    f"relevance_grades[{source!r}]={grade!r}는 0~3 정수여야 합니다"
                )
        _normalize_dedupe(list(v.keys()), field_name="relevance_grades")
        return v

    @field_validator("relevance_grades")
    @classmethod
    def relevance_grades_have_a_positive_grade(cls, v: dict[str, int]) -> dict[str, int]:
        """relevance_grades가 비어 있지 않다면 최소 1개는 양수(1~3) 등급이어야
        한다(M2_Phase1_code_review_result.md P2). 전부 0(비관련)이면 nDCG의
        IDCG가 항상 0이 돼 어떤 검색 결과를 넣어도 지표가 무의미해지고, 골든셋
        저자가 등급을 채워 넣고도 그 사례가 사실상 평가되지 않는다는 사실을
        놓치기 쉽다. 등급이 아예 필요 없는 사례는 relevance_grades를 생략하면
        된다(빈 dict는 여기서 걸리지 않음)."""
        if v and all(grade == 0 for grade in v.values()):
            raise ValueError(
                "relevance_grades의 모든 등급이 0입니다. 최소 1개는 1~3 등급이어야 "
                "nDCG가 의미 있게 계산됩니다. 등급이 필요 없으면 relevance_grades "
                "자체를 생략하세요."
            )
        return v

    @field_validator("relevant_sources")
    @classmethod
    def relevant_sources_no_duplicates(cls, v: list[str]) -> list[str]:
        """빈 문자열, 정규화 후 빈 문자열, 정규화 후 중복(완전 동일 표기 포함)을
        모두 거부한다(Problem.md 5차 리뷰 P2, design_review.md 1차 P1). data/
        파일시스템은 건드리지 않고 이 리스트 내부만 본다 — 실제 파일끼리의
        충돌은 Phase 3/4의 build_corpus_manifest()가 별도로 검사한다
        (상위 계획 §3.3)."""
        _normalize_dedupe(v, field_name="relevant_sources")
        return v

    @model_validator(mode="after")
    def positive_grades_are_in_relevant_sources(self) -> "GoldenCase":
        """relevance_grades의 양수 등급(1~3) source는 relevant_sources에도
        정규화 후 존재해야 한다 — 그렇지 않으면 nDCG 정답은 있는데 Retrieval
        evaluator(relevant_sources 기준 eligibility)에서는 제외되는 모순이
        생긴다(design_review.md 2차 P1). grade 0(비관련) source는 여전히
        relevant_sources 밖에 있을 수 있다. relevant_sources/relevance_grades
        각각의 개별 검증이 끝난 뒤에만 비교할 수 있으므로 model_validator로
        구현한다.

        책임 분리(design_review.md 3차 P2): 이 validator는 골든셋 저작
        시점의 **무결성 검증**(불리언 True/False)만 담당하며 정규화된 key로
        치환된 dict를 반환하지 않는다 — relevance_grades는 원본 표기 그대로
        저장된다. 실제 nDCG 계산에 쓰이는 **정규화된 조회용 mapping**은
        상위 계획 §3.5의 evaluation/metrics.py::normalize_relevance_grades()가
        별도로 만든다. 두 곳 모두 정규화 알고리즘은 normalize_source_id()
        하나만 쓰므로 결과는 항상 일치하지만, "저작 오류를 조기에 막는 것"과
        "계산용 mapping을 만드는 것"은 서로 다른 책임이라 의도적으로 나눴다."""
        if not self.relevance_grades:
            return self
        normalized_relevant = {normalize_source_id(s) for s in self.relevant_sources}
        for source, grade in self.relevance_grades.items():
            if grade > 0 and normalize_source_id(source) not in normalized_relevant:
                raise ValueError(
                    f"relevance_grades[{source!r}]={grade}는 양수 등급이지만 "
                    f"relevant_sources에 없습니다. 관련 있는 source는 relevant_sources에도 "
                    f"포함해야 합니다(grade 0인 비관련 source만 relevant_sources 밖에 있을 "
                    f"수 있습니다)."
                )
        return self

    @model_validator(mode="after")
    def assertions_and_abstention_are_mutually_exclusive(self) -> "GoldenCase":
        """answer_assertions와 expect_abstention=true를 동시에 설정할 수 없다
        (M2_Phase1_code_review_result.md P1). assertion은 "모델이 답하면 이
        핵심 사실을 포함해야 한다"는 기대이고 abstention은 "모델이 답을
        거절해야 한다"는 기대라서, 하나의 답변이 두 기대를 동시에 만족할 수
        없다 — 정상적으로 거절하면 assertion coverage가 실패하고 assertion을
        포함해 답하면 abstention 정확도가 실패한다. is_answer_eval_eligible()은
        이 validator를 통과한 GoldenCase만 받으므로 기존 OR 판정식을 그대로
        유지해도 된다."""
        if self.answer_assertions and self.expect_abstention:
            raise ValueError(
                "answer_assertions와 expect_abstention=true를 동시에 설정할 수 "
                "없습니다. 이 질문이 코퍼스로 답변 가능하다면 answer_assertions만, "
                "답변 불가(거절 기대)라면 expect_abstention=true만 사용하세요."
            )
        return self


def is_answer_eval_eligible(case: GoldenCase) -> bool:
    """category와 무관하게 필드 존재로만 판단한다: answer_assertions가
    하나 이상 있거나 expect_abstention=true인 사례가 대상이다(M2-REQ-008).
    Phase 1 dataset.py의 구성 검증(intent 최소 수량 집계)과 Phase 6
    answers.py의 evaluate_answers() 대상 판정이 반드시 동일한 정의를
    써야 하므로(design_review.md 3차 P2), 이 함수 하나를 두 곳에서
    import해 쓴다."""
    return bool(case.answer_assertions) or case.expect_abstention
```

`isinstance(grade, bool)` 배제는 Python에서 `bool`이 `int`의 서브클래스라 `True`/`False`가 `grades_in_range`를 그냥 통과해버리는 것을 막기 위함이다(예: `{"a.pdf": true}`가 JSON에서 파싱되면 Python `True`가 되고 `0 <= True <= 3`은 `True`라서 통과해버림 — 명백한 저작 실수인데 조용히 넘어가면 안 됨).

**`mode="before"`가 필수인 이유**: pydantic v2의 기본(lax) 모드는 `dict[str, int]` 필드를 검증할 때 `bool` 값을 먼저 `int`로 강제 변환(coercion)한 뒤 validator에 넘긴다. 기본값인 `mode="after"`로 두면 `grades_in_range`가 받는 시점에는 이미 `True`가 `1`로 바뀐 뒤라 `isinstance(grade, bool)` 검사가 항상 `False`가 되어 무력화된다. 이 문제는 코드를 스캐폴딩해 직접 실행 검증하는 과정에서 실측으로 발견했다(`GoldenCase(..., relevance_grades={"x.pdf": True})`가 예외 없이 통과함) — `mode="before"`로 원본 raw dict(여기서는 `True`/`False`가 아직 실제 bool 객체)를 검사하도록 바꿔서 고쳤다.

### 2.3 pydantic 버전 전제

`field_validator`는 Pydantic 2 전용 API다. Phase 1 커밋에 `requirements.txt` 한 줄을 추가한다 (§5).

## 3. `evaluation/dataset.py`

### 3.1 설계 원칙

- `load_jsonl()`은 **줄 단위로** 오류를 잡는다 — 한 줄이 깨졌다고 전체 파싱을 조용히 포기하지 않고, 정확히 몇 번째 줄/어떤 id가 문제인지 담은 `DatasetError`를 즉시 던진다(M2-REQ-004, M2-REQ-016).
- `validate_composition()`은 `load_jsonl()`이 이미 통과한 `GoldenCase` 리스트를 받는다 — 즉 개별 사례의 필드 유효성은 이미 보장된 상태에서 시작한다. 이 함수는 오직 M2-REQ-002의 9개 **구성** 규칙만 본다.
- CLI(`main()`)는 오류 시 반드시 0이 아닌 종료 코드를 반환한다(M2-REQ-004, M2-REQ-016). 성공 시에도 사람이 읽을 수 있는 요약을 stderr에 남긴다.
- **(design_review.md 1차 P2 반영) 파일 열기/읽기/인코딩 오류도 모두 `DatasetError`로 통일한다.** 이전 버전은 JSON 파싱과 스키마 검증 오류만 `DatasetError`로 감쌌고, `path.open()`의 `PermissionError`/`IsADirectoryError`(둘 다 `OSError`) 및 UTF-8이 아닌 파일을 읽을 때의 `UnicodeDecodeError`는 그대로 전파돼 CLI가 raw traceback을 노출했다. `with path.open(...)`부터 내부 for 루프 전체를 감싸 `except UnicodeDecodeError`/`except OSError`를 추가했다 — 내부에서 이미 던져진 `DatasetError`는 이 두 타입에 해당하지 않으므로 그대로 통과한다.
- **(design_review.md 1차 P3 반영) CLI가 인자 없이 호출돼도 `SystemExit`을 던지지 않고 정수를 반환한다.** `argparse.add_subparsers(..., required=True)`를 쓰면 `main([])`이 `args.command`를 검사하기도 전에 argparse 자체가 `SystemExit(2)`를 던져버려, 함수 시그니처(`-> int`)와 실제 동작이 어긋나고 `parser.print_help(sys.stderr); return 2` 폴백 코드가 죽은 코드가 된다(직접 실행 검증으로 확인 — `parser.parse_args([])`가 `required=True` 없이는 `SystemExit` 없이 `args.command=None`을 반환함을 확인). `required=True`를 제거했다.
- **(design_review.md 2차 P3 반영) 다만 `main()`이 "모든 경로에서" `int`를 반환한다는 설명은 부정확했다.** no-argv 경로는 이제 `int`를 반환하지만, `main(["bogus"])`처럼 존재하지 않는 subcommand나 잘못된 옵션을 주면 argparse 자체의 choice 검증이 여전히 표준 `SystemExit(2)`를 던진다(실행 검증으로 확인, §9.2). 정확한 설명은 "유효한 subcommand 또는 no-argv 경로에서는 `int`를 반환하며, argparse 문법 오류는 표준 `SystemExit(2)`를 그대로 사용한다"이며, `main()` docstring도 이렇게 수정했다(§3.3).
- **(design_review.md 2차 P1 반영) intent 최소 수량은 Answer 평가 대상 사례에서만 집계한다.** 이전 버전은 `expected_intent`를 category나 evaluator eligibility와 무관하게 전체 사례에서 셌기 때문에, `web_search`나 Retrieval 전용 사례에만 각 intent를 5개씩 배치해도 구성 검증을 통과할 수 있었다 — 그런데 상위 계획 Phase 6의 intent 정확도는 실제로 `answer_assertions`가 있거나 `expect_abstention=true`인(즉 `RAGEngine.query()`를 실제로 호출하는) Answer 평가 대상 사례에서만 측정된다. `intent_counts`는 `evaluation/schema.py`의 `is_answer_eval_eligible()`(design_review.md 3차 P2 반영으로 공유 함수화, §2.2)을 만족하는 사례에서만 집계하도록 고쳤다(§3.2, §3.3). `expected_intent=None`인 Answer 평가 대상 사례는 허용하되 최소 수량 집계에서는 제외하고, `answer_eval_cases_without_intent`로 별도 보고한다.
- **(M2_Phase1_code_review_result.md P3 반영) `DatasetError`에 `kind`를 추가해 CLI의 "다음 조치" 안내를 오류 종류에 맞게 고른다.** 실제 구현을 리뷰하며 파일 없음/디렉터리/권한/인코딩 오류(`load_jsonl()`이 파일 자체를 열지 못하는 경우)에도 "위 줄/사례를 golden.jsonl에서 수정한 뒤 다시 실행하세요"라는 content 전용 안내가 그대로 나가 오해를 준다는 지적을 확인했다. `kind="content"`(기본값, JSON 파싱/스키마 위반/중복 id)와 `kind="io"`(파일 없음/`OSError`/`UnicodeDecodeError`)로 구분해, `_run_validate()`가 `kind`에 따라 서로 다른 다음 조치 문구를 출력하도록 고쳤다(§3.3).

### 3.2 M2-REQ-002 구성 규칙 → 코드 상수 매핑

| 요구사항 문구 | 상수 | 검사 대상 |
|---|---|---|
| 고유 사례 최소 60개 | `MIN_TOTAL_CASES = 60` | 전체 |
| 문서 QA 40개 이상 | `MIN_DOCUMENT_QA = 40` | `category == document_qa` |
| 웹 검색 10개 이상 | `MIN_WEB_SEARCH = 10` | `category == web_search` |
| 문서/웹 경계 또는 답변 불가 10개 이상 | `MIN_BOUNDARY_OR_UNANSWERABLE = 10` | `category in {boundary, unanswerable}` 합계 |
| 모든 60개 사례에 기대 라우팅 결과 포함 | (검사 불필요) | `expected_route`가 필수 필드라 `load_jsonl()` 단계에서 이미 보장됨 |
| 문서 QA 중 30개 이상 Retrieval 정답 문서 포함 | `MIN_DOCUMENT_QA_WITH_SOURCES = 30` | `category == document_qa AND relevant_sources` 비어있지 않음 |
| 문서 QA 중 20개 이상 답변 핵심 사실 규칙 포함 | `MIN_DOCUMENT_QA_WITH_ASSERTIONS = 20` | `category == document_qa AND answer_assertions` 비어있지 않음 |
| 답변 불가 사례 5개 이상 | `MIN_EXPECT_ABSTENTION = 5` | `expect_abstention == True` (category 무관, §3.2 정정) |
| 한국어 질문 80% 이상 | `MIN_KOREAN_RATIO = 0.8` | `re.search(r"[가-힣]", question)` |
| explanation/comparison/procedure/yesno 각 5개 이상 | `MIN_PER_INTENT = 5` | Answer 평가 대상(`answer_assertions` 또는 `expect_abstention=true`) 사례 중 `expected_intent`별 카운트 (4개 라벨만, other/uncertain은 최소치 없음, design_review.md 2차 P1) |

### 3.3 전체 코드

```python
"""
골든 데이터셋 로더 및 검증기 (M2-REQ-002, M2-REQ-004).

data/, vectorstore/, 모델을 전혀 사용하지 않는다 — import나
`python -m evaluation.dataset validate` 실행에 Ollama나 벡터스토어가
필요해서는 안 된다(M2-NFR-003). CI가 이 전제로 돌아간다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from evaluation.schema import Category, GoldenCase, Intent, is_answer_eval_eligible

KOREAN_PATTERN = re.compile(r"[가-힣]")

MIN_TOTAL_CASES = 60
MIN_DOCUMENT_QA = 40
MIN_WEB_SEARCH = 10
MIN_BOUNDARY_OR_UNANSWERABLE = 10
MIN_DOCUMENT_QA_WITH_SOURCES = 30
MIN_DOCUMENT_QA_WITH_ASSERTIONS = 20
MIN_EXPECT_ABSTENTION = 5
MIN_KOREAN_RATIO = 0.8
MIN_PER_INTENT = 5
REQUIRED_INTENTS = (Intent.EXPLANATION, Intent.COMPARISON, Intent.PROCEDURE, Intent.YESNO)


class DatasetError(Exception):
    """dataset.py 전용 오류. case_id/line_number가 있으면 메시지 앞에 위치를 붙인다.

    kind는 CLI가 "다음 조치" 안내를 오류 종류에 맞게 고르는 데 쓰인다
    (M2_Phase1_code_review_result.md P3) — "content"(기본값)는 golden.jsonl의
    특정 줄/사례를 고쳐야 하는 경우(JSON 문법, 스키마 위반, 중복 id)이고,
    "io"는 파일 자체를 찾거나 읽을 수 없는 경우(파일 없음, 디렉터리, 권한,
    인코딩)라 "줄/사례 수정" 안내가 맞지 않는다."""

    def __init__(
        self,
        message: str,
        *,
        case_id: str | None = None,
        line_number: int | None = None,
        kind: str = "content",
    ) -> None:
        self.case_id = case_id
        self.line_number = line_number
        self.kind = kind
        location = []
        if line_number is not None:
            location.append(f"line {line_number}")
        if case_id is not None:
            location.append(f"id={case_id}")
        prefix = f"[{', '.join(location)}] " if location else ""
        super().__init__(f"{prefix}{message}")


def load_jsonl(path: Path) -> list[GoldenCase]:
    """UTF-8 JSON Lines 파일을 GoldenCase 리스트로 로드한다.

    빈 줄은 건너뛴다. 파싱/스키마/중복 id 오류는 물론 파일 열기·읽기·인코딩
    오류까지 모두 DatasetError로 변환되어 CLI가 항상 사람이 읽을 수 있는
    오류만 노출한다(M2-REQ-016, design_review.md 1차 P2).
    """
    if not path.exists():
        raise DatasetError(
            f"데이터셋 파일을 찾을 수 없습니다: {path}. "
            f"경로를 확인하거나 Phase 2 골든셋 작성을 먼저 완료하세요.",
            kind="io",
        )

    cases: list[GoldenCase] = []
    seen_ids: dict[str, int] = {}

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue

                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise DatasetError(
                        f"JSON 파싱 실패: {exc.msg}. 이 줄의 JSON 문법을 확인하세요.",
                        line_number=line_number,
                    ) from exc

                try:
                    case = GoldenCase.model_validate(payload)
                except ValidationError as exc:
                    detail = "; ".join(
                        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                        for err in exc.errors()
                    )
                    raise DatasetError(
                        f"스키마 검증 실패: {detail}",
                        case_id=payload.get("id") if isinstance(payload, dict) else None,
                        line_number=line_number,
                    ) from exc

                if case.id in seen_ids:
                    raise DatasetError(
                        f"중복된 id입니다. 이전 등장 위치: line {seen_ids[case.id]}. "
                        f"모든 사례는 파일 전체에서 고유한 id를 가져야 합니다.",
                        case_id=case.id,
                        line_number=line_number,
                    )
                seen_ids[case.id] = line_number

                cases.append(case)
    except UnicodeDecodeError as exc:
        raise DatasetError(
            f"UTF-8로 디코딩할 수 없습니다: {exc}. 파일이 UTF-8로 저장됐는지 확인하세요.",
            kind="io",
        ) from exc
    except OSError as exc:
        raise DatasetError(
            f"파일을 열거나 읽을 수 없습니다: {exc}. 경로와 권한을 확인하세요.",
            kind="io",
        ) from exc

    return cases


@dataclass
class ValidationReport:
    total: int
    by_category: dict[str, int] = field(default_factory=dict)
    document_qa_with_relevant_sources: int = 0
    document_qa_with_answer_assertions: int = 0
    total_with_expect_abstention: int = 0
    korean_ratio: float = 0.0
    answer_eval_case_count: int = 0
    answer_eval_cases_without_intent: int = 0
    intent_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_category": self.by_category,
            "document_qa_with_relevant_sources": self.document_qa_with_relevant_sources,
            "document_qa_with_answer_assertions": self.document_qa_with_answer_assertions,
            "total_with_expect_abstention": self.total_with_expect_abstention,
            "korean_ratio": round(self.korean_ratio, 4),
            "answer_eval_case_count": self.answer_eval_case_count,
            "answer_eval_cases_without_intent": self.answer_eval_cases_without_intent,
            "intent_counts": self.intent_counts,
            "valid": self.is_valid,
            "errors": self.errors,
        }


def _is_korean(question: str) -> bool:
    return bool(KOREAN_PATTERN.search(question))


def validate_composition(cases: list[GoldenCase]) -> ValidationReport:
    """M2-REQ-002의 9개 구성 규칙을 검사한다. cases는 이미 load_jsonl()을
    통과해 필드 유효성이 보장된 상태여야 한다."""

    report = ValidationReport(total=len(cases))
    errors: list[str] = []

    for cat in Category:
        report.by_category[cat.value] = sum(1 for c in cases if c.category == cat)

    # intent 최소 수량은 실제로 intent 정확도가 측정되는 Answer 평가 대상
    # 사례에서만 집계한다(design_review.md 2차 P1) — web_search나 Retrieval
    # 전용 사례에 intent만 붙여도 구성 검증이 통과해버리는 문제를 막는다.
    answer_eval_cases = [c for c in cases if is_answer_eval_eligible(c)]
    report.answer_eval_case_count = len(answer_eval_cases)
    report.answer_eval_cases_without_intent = sum(
        1 for c in answer_eval_cases if c.expected_intent is None
    )
    for intent in Intent:
        report.intent_counts[intent.value] = sum(
            1 for c in answer_eval_cases if c.expected_intent == intent
        )

    document_qa_cases = [c for c in cases if c.category == Category.DOCUMENT_QA]
    report.document_qa_with_relevant_sources = sum(
        1 for c in document_qa_cases if c.relevant_sources
    )
    report.document_qa_with_answer_assertions = sum(
        1 for c in document_qa_cases if c.answer_assertions
    )
    report.total_with_expect_abstention = sum(1 for c in cases if c.expect_abstention)

    korean_count = sum(1 for c in cases if _is_korean(c.question))
    report.korean_ratio = korean_count / len(cases) if cases else 0.0

    if report.total < MIN_TOTAL_CASES:
        errors.append(
            f"전체 사례 수가 {report.total}개로 최소 {MIN_TOTAL_CASES}개에 미달합니다."
        )
    if report.by_category[Category.DOCUMENT_QA.value] < MIN_DOCUMENT_QA:
        errors.append(
            f"document_qa 사례가 {report.by_category[Category.DOCUMENT_QA.value]}개로 "
            f"최소 {MIN_DOCUMENT_QA}개에 미달합니다."
        )
    if report.by_category[Category.WEB_SEARCH.value] < MIN_WEB_SEARCH:
        errors.append(
            f"web_search 사례가 {report.by_category[Category.WEB_SEARCH.value]}개로 "
            f"최소 {MIN_WEB_SEARCH}개에 미달합니다."
        )
    boundary_or_unanswerable = (
        report.by_category[Category.BOUNDARY.value]
        + report.by_category[Category.UNANSWERABLE.value]
    )
    if boundary_or_unanswerable < MIN_BOUNDARY_OR_UNANSWERABLE:
        errors.append(
            f"boundary+unanswerable 사례가 {boundary_or_unanswerable}개로 "
            f"최소 {MIN_BOUNDARY_OR_UNANSWERABLE}개에 미달합니다."
        )
    if report.document_qa_with_relevant_sources < MIN_DOCUMENT_QA_WITH_SOURCES:
        errors.append(
            f"relevant_sources가 있는 document_qa 사례가 "
            f"{report.document_qa_with_relevant_sources}개로 최소 "
            f"{MIN_DOCUMENT_QA_WITH_SOURCES}개에 미달합니다."
        )
    if report.document_qa_with_answer_assertions < MIN_DOCUMENT_QA_WITH_ASSERTIONS:
        errors.append(
            f"answer_assertions가 있는 document_qa 사례가 "
            f"{report.document_qa_with_answer_assertions}개로 최소 "
            f"{MIN_DOCUMENT_QA_WITH_ASSERTIONS}개에 미달합니다."
        )
    if report.total_with_expect_abstention < MIN_EXPECT_ABSTENTION:
        errors.append(
            f"expect_abstention=true 사례가 {report.total_with_expect_abstention}개로 "
            f"최소 {MIN_EXPECT_ABSTENTION}개에 미달합니다."
        )
    if report.korean_ratio < MIN_KOREAN_RATIO:
        errors.append(
            f"한국어 질문 비율이 {report.korean_ratio:.1%}로 최소 "
            f"{MIN_KOREAN_RATIO:.0%}에 미달합니다."
        )
    for intent in REQUIRED_INTENTS:
        count = report.intent_counts[intent.value]
        if count < MIN_PER_INTENT:
            errors.append(
                f"Answer 평가 대상(answer_assertions 또는 expect_abstention=true) 중 "
                f"intent={intent.value} 사례가 {count}개로 최소 {MIN_PER_INTENT}개에 "
                f"미달합니다."
            )

    report.errors = errors
    return report


def main(argv: list[str] | None = None) -> int:
    """유효한 subcommand 또는 인자 없음(no-argv) 경로에서는 항상 int를
    반환한다(design_review.md 1차 P3) — 그래야 테스트가
    pytest.raises(SystemExit) 없이 반환값만으로 종료 코드를 검증할 수 있다.
    다만 존재하지 않는 subcommand나 잘못된 옵션(예: main(["bogus"]))은
    argparse 자체의 문법 오류 처리 경로를 타므로 표준 SystemExit(2)를 그대로
    던진다 — "모든 경로에서 int를 반환한다"는 이전 설명은 부정확했다
    (design_review.md 2차 P3)."""
    parser = argparse.ArgumentParser(prog="python -m evaluation.dataset")
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser(
        "validate", help="골든 데이터셋 schema/구성 검증"
    )
    validate_parser.add_argument("path", type=Path, help="golden.jsonl 경로")

    args = parser.parse_args(argv)

    if args.command == "validate":
        return _run_validate(args.path)

    parser.print_help(sys.stderr)
    return 2


def _run_validate(path: Path) -> int:
    try:
        cases = load_jsonl(path)
    except DatasetError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        if exc.kind == "io":
            print(
                "다음 조치: 파일 경로가 올바른지, 접근 권한이 있는지, UTF-8로 "
                "저장됐는지 확인한 뒤 다시 실행하세요.",
                file=sys.stderr,
            )
        else:
            print(
                "다음 조치: 위 줄/사례를 golden.jsonl에서 수정한 뒤 다시 실행하세요.",
                file=sys.stderr,
            )
        return 1

    report = validate_composition(cases)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))

    if not report.is_valid:
        print(f"\n검증 실패: {len(report.errors)}건", file=sys.stderr)
        for err in report.errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            "다음 조치: evaluation/datasets/golden.jsonl에 부족한 카테고리/필드를 "
            "가진 사례를 추가하거나 기존 사례를 수정한 뒤 다시 실행하세요.",
            file=sys.stderr,
        )
        return 1

    print("\n검증 통과.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 4. 사례 예시

### 4.1 유효한 `document_qa` 사례 (Retrieval + Answer 평가 대상)

```json
{"id": "rag-001", "question": "RAG에서 MMR은 어떤 역할을 하나요?", "category": "document_qa", "expected_route": "document_qa", "expected_intent": "explanation", "relevant_sources": ["Retrieval-Augmented Generation (RAG) 복사본.pdf"], "relevance_grades": {"Retrieval-Augmented Generation (RAG) 복사본.pdf": 3}, "answer_assertions": [{"any_of": ["다양성", "중복을 줄"]}], "expect_abstention": false, "tags": ["korean", "explanation", "retrieval"]}
```

### 4.2 유효한 `document_qa` 사례 (Retrieval 평가 전용 — assertion/abstention 없음, §3.2 정정으로 통과해야 함)

```json
{"id": "rag-015", "question": "리랭커 문서에서 Cross-Encoder를 어떻게 설명하나요?", "category": "document_qa", "expected_route": "document_qa", "expected_intent": "explanation", "relevant_sources": ["리랭커(Reranker) 복사본.pdf"], "tags": ["korean", "retrieval-only"]}
```

### 4.3 `web_search` 사례 (assertion/relevant_sources 없음, M2-REQ-015)

```json
{"id": "web-001", "question": "오늘 서울 날씨 좀 웹에서 검색해줘", "category": "web_search", "expected_route": "web_search", "tags": ["korean", "routing_regression"]}
```

### 4.4 `unanswerable` 사례 (abstention 기대)

```json
{"id": "unans-001", "question": "이 문서 모음에 나온 2024년 서울시 지하철 요금 인상률이 얼마인가요?", "category": "unanswerable", "expected_route": "document_qa", "expect_abstention": true, "tags": ["korean", "abstention"]}
```

## 5. `requirements.txt` 변경

```diff
 # 웹 프레임워크
 fastapi>=0.104.0
 uvicorn>=0.24.0
 jinja2>=3.1.2
 python-multipart>=0.0.6
+pydantic>=2,<3
```

Phase 0 clean-venv 검증(상위 계획 §1)에서 `pydantic-2.13.4`가 fastapi의 전이 의존성으로 이미 설치됨을 확인했으므로 이 핀 고정이 기존 설치를 깨뜨리지 않는다.

## 6. 테스트 설계

### 6.1 `test_evaluation_schema.py`

```python
def test_module_imports_without_error():
    """design_review.md 3차 P1 반영: import 자체가 module-level 코드(예:
    빠뜨린 pydantic import로 인한 class 정의 시점 NameError)로 깨지지
    않는지 가장 먼저 확인하는 smoke test. 이 파일의 다른 모든 테스트가
    이미 암묵적으로 이 조건에 의존하지만, 실패 시 원인을 바로 알 수 있도록
    명시적으로 둔다."""
    import importlib
    importlib.import_module("evaluation.schema")


class TestNormalizeSourceId:
    def test_nfc_nfd_equivalent(self): ...        # 동일 한글, 다른 정규화 형태 -> 같은 결과
    def test_backslash_and_forward_slash_equivalent(self): ...
    def test_extracts_basename_from_nested_path(self): ...
    def test_case_insensitive(self): ...

class TestAnswerAssertion:
    def test_empty_any_of_raises(self): ...
    def test_whitespace_only_entries_raise(self): ...
    def test_valid_any_of_passes(self): ...

class TestGoldenCaseValid:
    def test_minimal_required_fields_passes(self): ...    # tags=[] 포함, 필수지만 빈 배열 허용
    def test_full_fields_passes(self): ...
    def test_default_factories_do_not_share_state(self):
        """pydantic mutable-default 버그 회귀 방지: 두 인스턴스의
        relevant_sources가 같은 리스트 객체를 참조하지 않아야 함
        (tags는 design_review.md 1차 반영으로 더 이상 default_factory가
        아니므로 relevant_sources/answer_assertions/relevance_grades 중
        하나로 검증)."""
        ...

class TestGoldenCaseInvalid:
    @pytest.mark.parametrize(
        "missing_field", ["id", "question", "category", "expected_route", "tags"]
    )
    def test_missing_required_field_raises(self, missing_field): ...  # tags 포함 (design_review.md 1차 P1)
    def test_invalid_category_enum_raises(self): ...
    def test_invalid_route_enum_raises(self): ...
    def test_blank_id_raises(self): ...
    def test_blank_question_raises(self): ...
    def test_relevance_grade_above_range_raises(self): ...   # grade=4
    def test_relevance_grade_below_range_raises(self): ...   # grade=-1
    def test_relevance_grade_bool_raises(self): ...           # grade=true (§2.2 bool 배제 회귀 방지)
    def test_duplicate_normalized_relevant_sources_raises(self): ...  # "X.pdf" vs "x.pdf"
    def test_exact_duplicate_relevant_sources_raises(self): ...  # "x.pdf" vs "x.pdf" (design_review.md 1차 P1)
    def test_blank_relevant_source_element_raises(self): ...  # "" (design_review.md 1차 P1)
    def test_whitespace_only_relevant_source_element_raises(self): ...  # "   "
    def test_relevant_source_normalizing_to_empty_raises(self): ...  # "a/b/" -> basename() == ""
    def test_blank_relevance_grade_key_raises(self): ...  # {"": 1} (design_review.md 1차 P2)
    def test_relevance_grade_key_normalized_collision_raises(self): ...  # {"X.pdf": 1, "x.pdf": 2}
    def test_unknown_top_level_field_raises(self): ...  # extra="forbid" (design_review.md 1차 P2)
    def test_answer_assertion_unknown_field_raises(self): ...
    def test_expect_abstention_quoted_string_true_raises(self): ...  # "true" (design_review.md 1차 P2)
    def test_expect_abstention_int_one_raises(self): ...  # 1
    def test_grades_only_case_with_empty_relevant_sources_raises(self): ...
        # relevance_grades={"a.pdf": 3}, relevant_sources=[] (design_review.md 2차 P1)
    def test_positive_grade_source_missing_from_relevant_sources_raises(self): ...
        # relevant_sources=["b.pdf"], relevance_grades={"a.pdf": 3}
    def test_relevance_grades_non_str_key_raises_validation_error_not_attributeerror(self): ...
        # GoldenCase.model_validate({..., "relevance_grades": {1: 2}}) (design_review.md 2차 P2)
    def test_relevant_sources_non_str_element_raises_validation_error_not_attributeerror(self): ...
        # GoldenCase.model_validate({..., "relevant_sources": [1]})
    def test_assertions_and_abstention_together_raises(self): ...
        # answer_assertions와 expect_abstention=true 동시 설정 (M2_Phase1_code_review_result.md P1)
    def test_all_zero_relevance_grades_raises(self): ...
        # relevance_grades={"irrelevant.pdf": 0} (M2_Phase1_code_review_result.md P2)
    def test_multiple_zero_relevance_grades_raises(self): ...
        # relevance_grades={"a.pdf": 0, "b.pdf": 0}

class TestGoldenCaseValidCombinations:
    def test_document_qa_without_assertions_or_abstention_is_valid(self):
        """§3.2 설계 정정의 핵심 회귀 테스트: assertion도 abstention도 없는
        document_qa(Retrieval 전용 사례)가 스키마 단에서 거부되면 안 됨."""
        ...
    def test_unanswerable_with_expect_abstention_and_no_assertions_is_valid(self): ...
    def test_distinct_relevant_sources_and_grade_keys_are_valid(self): ...
    def test_grade_zero_source_outside_relevant_sources_is_valid(self): ...
        # relevant_sources=["a.pdf"], relevance_grades={"a.pdf": 3, "irrelevant.pdf": 0}
        # (design_review.md 2차 P1 — grade 0은 relevant_sources 밖에 있어도 됨)
    def test_positive_grade_matching_via_normalization_is_valid(self): ...
        # relevant_sources=["A.PDF"], relevance_grades={"a.pdf": 2} — 정규화 후 동일 source
    def test_relevant_sources_without_any_relevance_grades_is_valid(self): ...
        # relevance_grades를 아예 안 주는 Retrieval 전용 사례는 여전히 유효

class TestIsAnswerEvalEligible:
    """design_review.md 3차 P2: dataset.py와 상위 계획의 answers.py(Phase 6)가
    이 함수 하나를 공유하므로, 조합을 여기서 한 번만 검증한다.
    answer_assertions와 expect_abstention을 동시에 쓰는 조합은
    M2_Phase1_code_review_result.md P1 반영으로 GoldenCase 자체가 거부하므로
    (TestGoldenCaseInvalid.test_assertions_and_abstention_together_raises)
    is_answer_eval_eligible()에 도달할 수 없다 — 남은 3가지 조합만 검증한다."""
    def test_assertions_only_is_eligible(self): ...
    def test_expect_abstention_only_is_eligible(self): ...
    def test_neither_is_not_eligible(self): ...
```

### 6.2 `test_evaluation_dataset.py`

```python
class TestLoadJsonl:
    def test_valid_file_parses_all_lines(self): ...
    def test_blank_lines_are_skipped(self): ...
    def test_malformed_json_raises_with_line_number(self): ...
    def test_schema_violation_raises_with_case_id_and_line_number(self): ...
    def test_duplicate_id_raises_referencing_first_occurrence_line(self): ...
    def test_missing_file_raises_dataset_error(self): ...
    def test_non_utf8_file_raises_dataset_error(self, tmp_path):
        """design_review.md P2: UnicodeDecodeError가 raw traceback으로 새지
        않고 DatasetError로 변환되는지 확인 (예: 잘못된 UTF-8 바이트 포함 파일)."""
        ...
    def test_directory_path_raises_dataset_error(self, tmp_path):
        """design_review.md 1차 P2: OSError(IsADirectoryError 등)도 DatasetError로
        변환되는지 확인 — path.exists()는 True이지만 open()이 실패하는 경우."""
        ...

class TestValidateComposition:
    def test_minimum_valid_dataset_has_no_errors(self):
        """9개 규칙을 정확히 최소치로 충족하는 62개(또는 60개) 사례 fixture.
        intent 최소치는 이제 Answer 평가 대상 사례에서만 채워야 통과한다
        (design_review.md 2차 P1)."""
        ...
    def test_total_below_minimum_flagged(self): ...
    def test_document_qa_below_minimum_flagged(self): ...
    def test_web_search_below_minimum_flagged(self): ...
    def test_boundary_plus_unanswerable_below_minimum_flagged(self): ...
    def test_document_qa_with_sources_below_minimum_flagged(self): ...
    def test_document_qa_with_assertions_below_minimum_flagged(self): ...
    def test_expect_abstention_below_minimum_flagged(self): ...
    def test_korean_ratio_below_minimum_flagged(self): ...
    @pytest.mark.parametrize("intent", ["explanation", "comparison", "procedure", "yesno"])
    def test_each_required_intent_below_minimum_flagged(self, intent): ...
    def test_multiple_violations_all_reported_together(self): ...
    def test_intent_only_on_non_answer_eval_cases_is_flagged(self):
        """design_review.md 2차 P1의 핵심 회귀 테스트: web_search/Retrieval
        전용 사례에만 각 intent를 5개씩 배치하면(answer_assertions도
        expect_abstention도 없음) 전체 개수는 충족해 보여도 구성 검증이
        반드시 실패해야 한다 — intent_counts가 Answer 평가 대상만 센다."""
        ...
    def test_answer_eval_case_count_and_without_intent_reported(self):
        """expected_intent=None인 Answer 평가 대상 사례가 있으면 구성
        검증에서 거부하지 않되 answer_eval_cases_without_intent로
        별도 집계돼야 한다."""
        ...

class TestCli:
    def test_validate_valid_dataset_returns_zero(self, tmp_path): ...
    def test_validate_invalid_schema_returns_one_with_stderr_detail(self, tmp_path, capsys):
        """M2_Phase1_code_review_result.md P3: stderr에 오류 원인·case id·
        content 전용 안내("golden.jsonl에서 수정한 뒤")가 있고, report가
        계산되지 않았으므로 stdout이 비어 있는지까지 확인한다."""
        ...
    def test_validate_missing_file_shows_io_advice_not_content_advice(self, tmp_path, capsys):
        """M2_Phase1_code_review_result.md P3: 파일 없음은 kind="io" 오류이므로
        content 전용 안내("golden.jsonl에서 수정한 뒤")가 아니라 경로/권한/
        인코딩 확인 안내가 나와야 하고, stdout은 비어 있어야 한다."""
        ...
    def test_validate_directory_path_shows_io_advice(self, tmp_path, capsys):
        """디렉터리 경로도 kind="io"로 분류돼 같은 안내를 받는지 확인."""
        ...
    def test_validate_prints_valid_json_report_to_stdout(self, tmp_path, capsys):
        """stdout 전체를 json.loads()로 되읽어 유효한 JSON인지 확인."""
        ...
    def test_no_argv_prints_usage_and_returns_nonzero(self, capsys):
        """design_review.md 1차 P3: main([])이 SystemExit을 던지지 않고 정수
        2를 반환하는지 직접 assert로 확인한다(§3.1 반영 — add_subparsers에서
        required=True를 제거했으므로 pytest.raises(SystemExit)이 필요 없다).
        M2_Phase1_code_review_result.md P3: `!= 0`이 아니라 계약값 `== 2`와
        stderr의 usage 출력까지 정확히 확인한다."""
        ...
    def test_invalid_subcommand_raises_systemexit(self):
        """argparse 자체의 choice 검증은 여전히 SystemExit(2)를 던진다
        (예: main(["bogus"])) — 이는 표준 argparse 동작이며 별도로 방지하지
        않는다. M2_Phase1_code_review_result.md P3: `exc.value.code == 2`까지
        정확히 확인한다."""
        ...
```

모든 `TestValidateComposition`/`TestCli` fixture 사례는 `tags`가 이제 필수 필드이므로 헬퍼에서 `tags`를 항상 채워 넣어야 한다(예: `tags=["fixture"]`) — 그렇지 않으면 `load_jsonl()`/`GoldenCase` 생성 자체가 스키마 오류로 실패해 구성 규칙 테스트의 의도와 무관한 실패가 발생한다.

모든 테스트는 `tmp_path`로 임시 JSONL 파일을 만들어 사용하며, `data/`/`vectorstore/`/네트워크/모델을 전혀 사용하지 않는다(M2-NFR-003과 동일 원칙을 테스트에도 적용).

## 7. 검증 절차

```bash
pip install -r requirements.txt   # pydantic>=2,<3 반영 확인
python -c "import evaluation.schema, evaluation.dataset"   # import 자체가 깨지지 않는지 가장 먼저 확인(design_review.md 3차 P1)
python -m evaluation.dataset --help   # schema import부터 argparse 구성까지 CLI 진입 경로 확인(design_review.md 3차 P1)
pytest -q test_evaluation_schema.py test_evaluation_dataset.py -v
pytest -q   # 전체 스위트에서 기존 실패·새 실패 없음을 확인(구체적인 pass/skip 총계는 Phase 1에서 테스트가 추가되며 매번 달라지므로 고정 숫자로 비교하지 않는다 — design_review.md 3차 P3. Phase 0 착수 전 기준선은 21 passed, 1 skipped였다, §1)
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl   # Phase 2 완료 전까지는 "파일 없음" 오류가 정상
```

## 8. Phase 1 완료 조건

- [ ] `evaluation/__init__.py`, `evaluation/schema.py`, `evaluation/dataset.py` 작성
- [ ] §6의 모든 테스트 통과
- [ ] `requirements.txt`에 `pydantic>=2,<3` 추가, 기존 `pip install` 결과와 충돌 없음 확인
- [ ] `python -m evaluation.dataset validate <없는 파일>`이 0이 아닌 종료 코드와 사람이 읽을 오류를 반환
- [ ] `python -c "import evaluation.schema, evaluation.dataset"`과 `python -m evaluation.dataset --help`가 모두 예외 없이 동작 (design_review.md 3차 P1)
- [ ] 기존 `pytest -q`에서 새 실패 없음 — Phase 0 착수 전 기준선(21 passed, 1 skipped, §1)과 비교해 실패가 늘지 않았는지만 확인하고, Phase 1에서 추가된 테스트만큼 pass 총계가 늘어나는 것은 정상(design_review.md 3차 P3, 고정 숫자로 비교하지 않음)
- [ ] design_review.md 1차(P1 3건·P2 3건·P3 1건), 2차(P1 2건·P2 2건·P3 1건), 3차(P1 1건·P2 2건·P3 1건)가 모두 §2.2/§3.3 코드와 §6 테스트 목록에 반영됨 (§9.1/§9.2/§9.3 참고)
- [ ] M2_Phase1_code_review_result.md(P1 1건·P2 1건·P3 2건)가 모두 §2.2/§3.3 코드와 §6 테스트 목록에 반영됨 (§9.4 참고)
- [ ] 커밋: `evaluation schema 및 dataset validator 추가` (상위 계획 §5 권장 커밋 단위와 동일)

Phase 2(골든셋 작성)는 이 설계가 구현되고 `evaluation.dataset validate`가 실제로 동작한 뒤에 시작한다.

## 9. 설계 자체 검증 (스캐폴딩 실행 결과)

이 설계 문서의 §2.2/§3.3 코드는 문서 작성 후 스크래치패드에 그대로 옮겨 실제로 실행 검증했다(코드가 "그럴듯해 보이는" 수준을 넘어 실제로 동작함을 확인하기 위함). 결과:

- `GoldenCase` 기본 생성, retrieval-only `document_qa` 사례(assertion/abstention 없이) 유효성, 기본값 리스트 필드가 인스턴스 간 공유되지 않음, grade 범위 초과(4) 거부, `relevant_sources` 정규화 중복 거부, `normalize_source_id()`의 NFC/NFD·역슬래시·casefold 동치성 — 모두 설계대로 동작 확인.
- **버그 발견 및 수정**: `relevance_grades={"x.pdf": True}` 같은 bool 값이 애초 설계(기본 `mode="after"` validator)에서는 거부되지 않았다 — pydantic이 `dict[str, int]` 필드의 타입 강제변환을 validator 실행보다 먼저 수행해 `True`를 `1`로 바꿔버리기 때문. `mode="before"`로 바꿔 원본 raw 값을 검사하도록 수정했고(§2.2에 반영 완료), 재검증 결과 `True`/`False` 모두 정상적으로 거부되고 일반 정수 grade는 정상 통과함을 확인했다.
- `load_jsonl()`: 파일 없음, JSON 파싱 실패, 스키마 위반, 중복 id, 정상 파일 로드 — 5가지 경로 모두 설계대로 `DatasetError`(위치 정보 포함) 또는 정상 반환.
- `validate_composition()`: 9개 구성 규칙을 정확히 충족하는 60개 사례 fixture(document_qa 40/web_search 10/boundary 5/unanswerable 5, 각 intent 10개, 한국어 100%, sources 30개, assertions 20개, abstention 5개)로 `is_valid=True, errors=[]` 확인. 최소치 미달 fixture로는 해당 오류 메시지가 정확히 발생함을 확인.
- CLI(`python -m evaluation.dataset validate`): 유효한 파일에서 종료 코드 0, 구성 미달 파일과 존재하지 않는 파일에서 각각 종료 코드 1 확인.

이 검증은 임시 스크래치패드 코드 기준이며 §6에 정의된 정식 pytest 테스트를 대체하지 않는다. Phase 1 구현 시 §6 테스트 스위트를 그대로 작성해야 한다.

### 9.1 design_review.md 반영 후 재검증

[design_review.md](../../reviews/m2-quality-baseline/Design_Review.md) 검토(종합 평가 8.2/10, P1 3건·P2 3건·P3 1건)를 반영해 §2.2/§3.1/§3.3 코드를 수정한 뒤, 다시 스크래치패드에 옮겨 각 지적 사항이 실제로 고쳐졌는지 하나씩 실행 검증했다.

- **P1 (`tags` 필수 누락 통과)**: `tags` 없이 `GoldenCase`를 생성하면 이제 `ValidationError`가 발생하고, `tags: []`(빈 배열)는 여전히 통과함을 확인 — "필드는 필수, 빈 배열은 허용"이라는 §2.1의 결정대로 동작한다.
- **P1 (빈 source가 구성 최소치를 허위 충족)**: `relevant_sources=[""]`, `["   "]`, `["a/b/"]`(정규화 결과가 빈 문자열) 모두 거부되는지 확인. `relevance_grades`의 빈/공백 key(`{"": 1}`)도 거부됨을 확인.
- **P1 (완전 동일 표기 중복 미검출)**: `["x.pdf", "x.pdf"]`(표기까지 동일)와 `["X.pdf", "x.pdf"]`(대소문자만 다름) 둘 다 거부되고, 서로 다른 두 정상 source(`["a.pdf", "b.pdf"]`)는 통과함을 확인 — `if norm in seen and seen[norm] != raw`를 `if norm in seen`으로 바꾼 수정이 의도대로 동작한다.
- **P2 (`relevance_grades` key 정규화 계약)**: `{"X.pdf": 1, "x.pdf": 2}`처럼 정규화 후 충돌하는 key 조합이 거부되고, 서로 다른 key(`{"a.pdf": 1, "b.pdf": 2}`)는 통과함을 확인. key 자체를 자동으로 정규화하지는 않는다는 설계(원본 표기 보존, evaluator가 조회 시 정규화)를 그대로 유지했다.
- **P2 (coercion/`extra` 정책)**: `GoldenCase(..., not_a_field=1)`과 `AnswerAssertion(any_of=["a"], bogus=1)` 둘 다 `extra="forbid"` 적용 후 거부됨을 확인. `expect_abstention="true"`(따옴표 붙은 문자열), `expect_abstention=1`(정수) 모두 새로 추가한 `mode="before"` strict 검사로 거부되고, 실제 `bool` 값(`True`/`False`, 필드 생략 시 기본값 `False`)은 정상 통과함을 확인.
- **P2 (파일 I/O/인코딩 오류가 `DatasetError`로 통일되지 않음)**: 디렉터리 경로를 `load_jsonl()`에 넘기면 `IsADirectoryError`(`OSError`)가 `DatasetError`로 변환됨을 확인(`파일을 열거나 읽을 수 없습니다: ...`). UTF-8이 아닌 바이트가 섞인 파일도 `UnicodeDecodeError`가 `DatasetError`로 변환됨을 확인(`UTF-8로 디코딩할 수 없습니다: ...`).
- **P3 (`main([])`이 실제로는 `SystemExit`을 던짐)**: `add_subparsers(..., required=True)`를 제거한 뒤 `main([])`을 직접 호출해 예외 없이 정수 `2`가 반환됨을 확인 — `test_no_argv_prints_usage_and_returns_nonzero`가 이름 그대로 `assert main([]) != 0` 형태로 구현 가능해졌다. `parser.print_help(sys.stderr); return 2` 폴백도 이제 실제로 도달 가능한 코드임을 확인했다.
- **회귀 확인**: 이전 라운드에서 검증한 60개 사례 구성 fixture에 `tags` 필드를 채워 넣고(각 사례 `"tags": ["g"]`) 다시 `validate_composition()`과 CLI를 실행한 결과 `is_valid=True`, 종료 코드 0으로 이전과 동일하게 통과함을 확인 — `tags` 필수화가 기존 통과 경로를 깨뜨리지 않는다.

design_review.md의 P1 3건은 모두 이번 수정에 반영됐다. P2 3건(grade key 정규화 계약 명시, extra=forbid/strict bool, I/O 오류 통일)과 P3(argparse 구조 변경)도 모두 반영했으므로 "구현 전 필수 수정 순서" 6개 항목이 모두 설계 문서에 적용된 상태다.

### 9.2 design_review.md 2차 검토 반영 후 재검증

2차 검토(종합 평가 9.0/10, P1 2건·P2 2건·P3 1건)를 반영해 §2.1/§2.2/§3.1/§3.2/§3.3을 다시 수정한 뒤, 다시 스크래치패드에 옮겨 하나씩 실행 검증했다. 1차 검증에서 확인한 19개 회귀 시나리오(§9.1 대상)도 함께 재실행해 이번 수정이 기존 동작을 깨뜨리지 않았음을 확인했다.

- **P1 (`relevance_grades`만 있고 `relevant_sources`가 빈 사례가 schema를 통과)**: `relevance_grades={"a.pdf": 3}`, `relevant_sources=[]`인 사례가 이제 거부됨을 확인. `relevant_sources=["b.pdf"]`처럼 다른 source만 있고 정작 양수 등급을 받은 `"a.pdf"`가 빠진 경우도 거부됨을 확인. 반대로 grade 0인 source(`"irrelevant.pdf"`)는 `relevant_sources` 밖에 있어도 통과함을 확인(비관련 문서를 nDCG 계산에 포함시키는 정상 시나리오). 정규화 후 일치하는 경우(`relevant_sources=["A.PDF"]`, `relevance_grades={"a.pdf": 2}`)도 정상 통과함을 확인 — `model_validator(mode="after")`가 두 필드 모두 개별 검증이 끝난 뒤 정규화 비교를 정확히 수행한다.
- **P1 (intent 최소 수량이 Answer 평가 대상이 아닌 전체 사례에서 집계됨)**: 40개 document_qa 전부를 Retrieval 전용(assertion/abstention 없음)으로 두고 그중 20개에만 4개 필수 intent를 5개씩 배치한 fixture로 `validate_composition()`을 실행한 결과, `is_valid=False`이고 4개 intent 모두 "Answer 평가 대상 중 ... 미달합니다" 오류가 발생함을 확인 — 이전 로직이라면 이 fixture는 통과했어야 하는데(전체 사례 기준으로는 각 intent가 5개씩 있었으므로), 새 로직은 `answer_eval_case_count=10`(boundary/unanswerable의 abstention 사례만 해당)으로 정확히 집계하고 4개 intent 모두 0개로 보고했다. 기존 60개 사례 회귀 fixture(document_qa 20개가 assertion을 가지며 그 20개 안에 4개 intent가 정확히 5개씩 분포)는 여전히 `is_valid=True`로 통과함을 확인 — `answer_eval_case_count=25`, `intent_counts`가 각 5씩 정확히 나옴.
- **P2 (`_normalize_dedupe()`가 비문자열 key에서 raw `AttributeError`를 냄)**: `GoldenCase.model_validate({..., "relevance_grades": {1: 2}})`와 `GoldenCase.model_validate({..., "relevant_sources": [1]})` 둘 다 이제 `AttributeError`가 아니라 깔끔한 `pydantic.ValidationError`로 거부됨을 확인.
- **P2 (grade key 정규화 책임이 Phase 3/4 설계로 추적되지 않음)**: 이 항목은 Phase 1 코드가 아니라 상위 계획(Development_M2_Quality_Baseline_Development_Plan.md) §3.5의 nDCG 호출 시퀀스에 대한 지적이므로, 상위 계획 문서를 직접 수정해 `normalize_relevance_grades()` 헬퍼와 `evaluate_retrieval()`의 정규화 순서를 명시했다(별도 커밋 대상 문서, 이 설계 문서의 검증 대상은 아님).
- **P3 (`main()`이 "모든 경로에서" int를 반환한다는 설명이 부정확함)**: `main([])`은 여전히 예외 없이 `2`를 반환하지만, `main(["bogus"])`는 argparse의 choice 검증에 의해 `SystemExit(2)`를 던짐을 재확인 — docstring을 두 경로를 구분해서 설명하도록 수정했다(§3.3).

design_review.md 2차의 P1 2건, P2 2건(코드 수정 1건 + 상위 계획 문서 수정 1건), P3 1건이 모두 반영됐다.

### 9.3 design_review.md 3차 검토 결과

3차 검토(종합 평가 9.2/10, P1 1건·P2 2건·P3 1건)에서 지적한 P1("`model_validator` import 누락으로 NameError")은 **이 문서의 현재 상태에는 해당하지 않았다** — §2.2의 import 문(`from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator`)은 이미 2차 반영 시점(§9.2)부터 `model_validator`를 포함하고 있었다. 실제로 문서에서 §2.2/§3.3 python 코드 블록을 정규식으로 그대로 추출해 스크래치패드에 저장한 뒤 `import evaluation.schema`/`import evaluation.dataset`를 실행해 `NameError` 없이 성공함을 재확인했다(3차 리뷰가 참조한 스냅숏이 실제 파일보다 앞선 시점이었을 가능성이 있다). 이 항목은 코드 수정 없이 "이미 반영됨"으로 결론짓되, 재발 방지를 위해 리뷰가 권고한 두 가지 방어 장치는 그대로 채택했다 — §6.1에 `test_module_imports_without_error` smoke test를 추가하고, §7/§8 검증 절차에 `python -c "import evaluation.schema, evaluation.dataset"`과 `python -m evaluation.dataset --help`를 추가했다(둘 다 정상 동작을 실행 검증함, 후자는 `SystemExit(0)`으로 종료됨을 확인).

- **P2 (Answer eligibility 규칙이 Phase 1 `dataset.py`와 Phase 6 `answers.py`에 중복 정의됨)**: 실제로 상위 계획 §Phase 6에 `_is_answer_eval_eligible()`이 별도로 정의돼 있음을 확인(중복이 사실임을 검증). `is_answer_eval_eligible()`을 `evaluation/schema.py`의 공개 함수로 옮기고 `dataset.py`가 여기서 import하도록 수정한 뒤, 스크래치패드에서 (1) `evaluation.schema`에서 직접 import되는지, (2) assertion-only/abstention-only/둘 다/둘 다 없음 4가지 조합이 리뷰가 요구한 대로 각각 True/True/True/False로 판정되는지, (3) 60개 사례 회귀 fixture가 여전히 `is_valid=True`로 통과하는지 모두 확인했다. 상위 계획의 `answers.py`(Phase 6)도 이 공개 함수를 import하도록 Development_Plan.md를 함께 수정했다(별도 문서, 이 설계 문서의 검증 대상 밖).
- **P2 (positive grade model validator와 metrics.py의 정규화 mapping 생성이 책임 분리 없이 나뉨)**: 구현 blocker가 아니라는 리뷰의 평가에 동의하고, 권고된 두 선택지 중 "schema는 무결성 검증만, metrics는 계산용 mapping만" 쪽을 택해 `positive_grades_are_in_relevant_sources`의 docstring에 이 책임 분리를 명시적으로 문서화했다(§2.2). 코드 동작 변경은 없다.
- **P3 (`21 passed, 1 skipped` 고정 숫자가 Phase 1 테스트 추가 후 오래됨)**: 완료 조건과 검증 절차의 표현을 "고정 숫자와 일치"에서 "기존 실패가 늘지 않았는지 확인, Phase 0 착수 전 기준선은 21 passed/1 skipped였다는 사실만 별도로 유지"로 바꿨다(§7, §8).

design_review.md 3차의 P1은 이미 해당 없음으로 확인, P2 2건과 P3 1건은 모두 반영됐다.

### 9.4 M2_Phase1_code_review_result.md 반영 결과

[M2_Phase1_code_review_result.md](../../reviews/m2-quality-baseline/Phase_1_Code_Review.md)는 설계 문서가 아니라 **실제로 작성된 `evaluation/schema.py`/`evaluation/dataset.py`/테스트 코드**를 리뷰한 결과다(종합 평가 8.8/10, P1 1건·P2 1건·P3 2건). 지적된 두 버그(P1, P2)는 실제 구현 코드에서 먼저 재현해 존재를 확인한 뒤 수정했고, 이 문서의 §2.2/§3.3 코드 블록도 수정된 구현과 동일하게 갱신했다 — `diff`로 두 코드 블록이 `evaluation/schema.py`/`evaluation/dataset.py`와 byte-identical함을 재확인했다.

- **P1 (assertion과 abstention이 동시에 설정된 모순 사례 허용)**: 실행으로 `GoldenCase(answer_assertions=[...], expect_abstention=True)`가 예외 없이 생성되고 `is_answer_eval_eligible()`이 `True`를 반환함을 먼저 재현했다. `assertions_and_abstention_are_mutually_exclusive` model_validator를 추가해 수정한 뒤 같은 입력이 `ValidationError`를 던짐을 재확인했고, assertion-only/abstention-only 등 정상 조합은 그대로 통과함도 함께 확인했다. 기존 `test_both_assertions_and_abstention_is_eligible`(둘 다 있어도 eligible이라고 주장하던 테스트)은 삭제하고 `TestGoldenCaseInvalid.test_assertions_and_abstention_together_raises`로 대체했다.
- **P2 (all-zero relevance_grades 허용)**: `GoldenCase(relevance_grades={"irrelevant.pdf": 0})`(relevant_sources 없음)가 예외 없이 생성됨을 먼저 재현했다. `relevance_grades_have_a_positive_grade` field_validator를 추가해 수정한 뒤 같은 입력과 `{"a.pdf": 0, "b.pdf": 0}`(다중 zero) 모두 거부됨을 확인했고, grade 0과 양수 grade가 섞인 기존 유효 사례(`test_grade_zero_source_outside_relevant_sources_is_valid`)는 여전히 통과함을 재확인했다.
- **P3 (CLI "다음 조치" 안내가 오류 종류와 안 맞음)**: `DatasetError`에 `kind`(기본값 `"content"`)를 추가하고 파일 없음/`OSError`/`UnicodeDecodeError` 3곳에서 `kind="io"`를 지정했다. `_run_validate()`가 `kind`에 따라 다른 안내를 출력하도록 수정한 뒤, 파일 없음/디렉터리 경로에서는 "경로가 올바른지" 안내가 나오고 "golden.jsonl에서 수정한 뒤" 안내는 나오지 않음을, 스키마 위반에서는 반대로 "golden.jsonl에서 수정한 뒤" 안내가 나옴을 각각 확인했다.
- **P3 (CLI 테스트가 정확한 종료 코드/출력 채널을 단언하지 않음)**: no-argv 테스트를 `main([]) != 0`에서 `main([]) == 2` + stderr의 `"usage"` 포함 여부로, invalid-subcommand 테스트를 `pytest.raises(SystemExit)`에서 `exc_info.value.code == 2`까지 확인하도록 강화했다. missing-file/스키마-위반 테스트에도 `captured.out == ""`(report가 계산되지 않았는지) 확인을 추가했다.
- **회귀 확인**: `pytest -q test_evaluation_schema.py test_evaluation_dataset.py` 81 passed(기존 78 + 신규/수정 3), 전체 `pytest -q` 102 passed, 1 skipped(기존 99 + 신규 3), 새 실패 없음을 확인했다.

요구사항 문서 M2-REQ-003에도 assertion/abstention 상호 배타성과 all-zero relevance_grades 금지 규칙을 반영해 Design.md와 동기화했다. P3의 나머지 권고(`ndcg_at_k()`의 IDCG=0 정책 명시)는 Phase 3/4 `metrics.py` 설계 시점에 다루면 되는 항목이라 이 문서에서는 다루지 않는다.
