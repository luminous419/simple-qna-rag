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
