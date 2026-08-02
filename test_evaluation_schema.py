"""evaluation/schema.py 단위 테스트 (M2 Phase 1, Development_M2_Quality_Baseline_Design.md §6.1)."""

from __future__ import annotations

import unicodedata

import pytest
from pydantic import ValidationError

from evaluation.schema import (
    AnswerAssertion,
    GoldenCase,
    is_answer_eval_eligible,
    normalize_source_id,
)


def _base(**overrides) -> dict:
    base = {
        "id": "case-1",
        "question": "테스트 질문입니다",
        "category": "document_qa",
        "expected_route": "document_qa",
        "tags": [],
    }
    base.update(overrides)
    return base


def test_module_imports_without_error():
    """design_review.md 3차 P1 반영: import 자체가 module-level 코드(예:
    빠뜨린 pydantic import로 인한 class 정의 시점 NameError)로 깨지지
    않는지 가장 먼저 확인하는 smoke test."""
    import importlib

    importlib.import_module("evaluation.schema")


class TestNormalizeSourceId:
    def test_nfc_nfd_equivalent(self):
        nfc = unicodedata.normalize("NFC", "가나다.pdf")
        nfd = unicodedata.normalize("NFD", "가나다.pdf")
        assert nfc != nfd
        assert normalize_source_id(nfc) == normalize_source_id(nfd)

    def test_backslash_and_forward_slash_equivalent(self):
        assert normalize_source_id("data\\sub\\a.pdf") == normalize_source_id("data/sub/a.pdf")

    def test_extracts_basename_from_nested_path(self):
        assert normalize_source_id("data/sub/dir/a.pdf") == "a.pdf"

    def test_case_insensitive(self):
        assert normalize_source_id("A.PDF") == normalize_source_id("a.pdf")


class TestAnswerAssertion:
    def test_empty_any_of_raises(self):
        with pytest.raises(ValidationError):
            AnswerAssertion(any_of=[])

    def test_whitespace_only_entries_raise(self):
        with pytest.raises(ValidationError):
            AnswerAssertion(any_of=["   "])

    def test_valid_any_of_passes(self):
        assertion = AnswerAssertion(any_of=["foo", "bar"])
        assert assertion.any_of == ["foo", "bar"]


class TestGoldenCaseValid:
    def test_minimal_required_fields_passes(self):
        case = GoldenCase(**_base())
        assert case.tags == []
        assert case.relevant_sources == []
        assert case.relevance_grades == {}
        assert case.answer_assertions == []
        assert case.expect_abstention is False

    def test_full_fields_passes(self):
        case = GoldenCase(
            **_base(
                expected_intent="explanation",
                relevant_sources=["a.pdf"],
                relevance_grades={"a.pdf": 3},
                answer_assertions=[{"any_of": ["foo"]}],
                expect_abstention=False,
                tags=["korean"],
                notes="사람 검토용 설명",
            )
        )
        assert case.expected_intent.value == "explanation"
        assert case.notes == "사람 검토용 설명"

    def test_default_factories_do_not_share_state(self):
        """pydantic mutable-default 버그 회귀 방지: 두 인스턴스의
        relevant_sources가 같은 리스트 객체를 참조하지 않아야 함."""
        case1 = GoldenCase(**_base())
        case2 = GoldenCase(**_base(id="case-2"))
        case1.relevant_sources.append("x.pdf")
        assert case2.relevant_sources == []


class TestGoldenCaseInvalid:
    @pytest.mark.parametrize(
        "missing_field", ["id", "question", "category", "expected_route", "tags"]
    )
    def test_missing_required_field_raises(self, missing_field):
        kwargs = _base()
        del kwargs[missing_field]
        with pytest.raises(ValidationError):
            GoldenCase(**kwargs)

    def test_invalid_category_enum_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(category="not_a_category"))

    def test_invalid_route_enum_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(expected_route="not_a_route"))

    def test_blank_id_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(id="   "))

    def test_blank_question_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(question=""))

    def test_relevance_grade_above_range_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(relevant_sources=["a.pdf"], relevance_grades={"a.pdf": 4}))

    def test_relevance_grade_below_range_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(relevant_sources=["a.pdf"], relevance_grades={"a.pdf": -1}))

    def test_relevance_grade_bool_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(relevant_sources=["a.pdf"], relevance_grades={"a.pdf": True}))

    def test_duplicate_normalized_relevant_sources_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(relevant_sources=["X.pdf", "x.pdf"]))

    def test_exact_duplicate_relevant_sources_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(relevant_sources=["x.pdf", "x.pdf"]))

    def test_blank_relevant_source_element_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(relevant_sources=[""]))

    def test_whitespace_only_relevant_source_element_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(relevant_sources=["   "]))

    def test_relevant_source_normalizing_to_empty_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(relevant_sources=["a/b/"]))

    def test_blank_relevance_grade_key_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(relevant_sources=["a.pdf"], relevance_grades={"": 1}))

    def test_relevance_grade_key_normalized_collision_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(
                **_base(
                    relevant_sources=["a.pdf"],
                    relevance_grades={"X.pdf": 1, "x.pdf": 2},
                )
            )

    def test_unknown_top_level_field_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(not_a_field="oops"))

    def test_answer_assertion_unknown_field_raises(self):
        with pytest.raises(ValidationError):
            AnswerAssertion.model_validate({"any_of": ["a"], "bogus": 1})

    def test_expect_abstention_quoted_string_true_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(expect_abstention="true"))

    def test_expect_abstention_int_one_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(expect_abstention=1))

    def test_grades_only_case_with_empty_relevant_sources_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(**_base(relevance_grades={"a.pdf": 3}))

    def test_positive_grade_source_missing_from_relevant_sources_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(
                **_base(relevant_sources=["b.pdf"], relevance_grades={"a.pdf": 3})
            )

    def test_relevance_grades_non_str_key_raises_validation_error_not_attributeerror(self):
        with pytest.raises(ValidationError):
            GoldenCase.model_validate({**_base(), "relevance_grades": {1: 2}})

    def test_relevant_sources_non_str_element_raises_validation_error_not_attributeerror(self):
        with pytest.raises(ValidationError):
            GoldenCase.model_validate({**_base(), "relevant_sources": [1]})

    def test_assertions_and_abstention_together_raises(self):
        """M2_Phase1_code_review_result.md P1: "모델이 답하면 이 사실을
        포함해야 한다"(assertion)와 "모델은 거절해야 한다"(abstention)는
        하나의 답변이 동시에 만족할 수 없는 모순 조건이므로 schema 단에서
        거부돼야 한다."""
        with pytest.raises(ValidationError):
            GoldenCase(
                **_base(
                    category="unanswerable",
                    answer_assertions=[{"any_of": ["정답"]}],
                    expect_abstention=True,
                )
            )

    def test_all_zero_relevance_grades_raises(self):
        """M2_Phase1_code_review_result.md P2: 등급이 전부 0이면 IDCG가 항상
        0이 돼 nDCG가 어떤 검색 결과에도 무의미하므로 거부돼야 한다."""
        with pytest.raises(ValidationError):
            GoldenCase(**_base(relevance_grades={"irrelevant.pdf": 0}))

    def test_multiple_zero_relevance_grades_raises(self):
        with pytest.raises(ValidationError):
            GoldenCase(
                **_base(relevance_grades={"a.pdf": 0, "b.pdf": 0})
            )


class TestGoldenCaseValidCombinations:
    def test_document_qa_without_assertions_or_abstention_is_valid(self):
        """§3.2 설계 정정의 핵심 회귀 테스트: assertion도 abstention도 없는
        document_qa(Retrieval 전용 사례)가 스키마 단에서 거부되면 안 됨."""
        case = GoldenCase(**_base(relevant_sources=["a.pdf"]))
        assert case.answer_assertions == []
        assert case.expect_abstention is False

    def test_unanswerable_with_expect_abstention_and_no_assertions_is_valid(self):
        case = GoldenCase(
            **_base(category="unanswerable", expected_route="document_qa", expect_abstention=True)
        )
        assert case.expect_abstention is True
        assert case.answer_assertions == []

    def test_distinct_relevant_sources_and_grade_keys_are_valid(self):
        case = GoldenCase(
            **_base(
                relevant_sources=["a.pdf", "b.pdf"],
                relevance_grades={"a.pdf": 1, "b.pdf": 2},
            )
        )
        assert case.relevance_grades == {"a.pdf": 1, "b.pdf": 2}

    def test_grade_zero_source_outside_relevant_sources_is_valid(self):
        case = GoldenCase(
            **_base(
                relevant_sources=["a.pdf"],
                relevance_grades={"a.pdf": 3, "irrelevant.pdf": 0},
            )
        )
        assert case.relevance_grades["irrelevant.pdf"] == 0

    def test_positive_grade_matching_via_normalization_is_valid(self):
        case = GoldenCase(
            **_base(relevant_sources=["A.PDF"], relevance_grades={"a.pdf": 2})
        )
        assert case.relevance_grades == {"a.pdf": 2}

    def test_relevant_sources_without_any_relevance_grades_is_valid(self):
        case = GoldenCase(**_base(relevant_sources=["a.pdf"]))
        assert case.relevance_grades == {}


class TestIsAnswerEvalEligible:
    """design_review.md 3차 P2: dataset.py와 상위 계획의 answers.py(Phase 6)가
    이 함수 하나를 공유하므로, 조합을 여기서 한 번만 검증한다.
    answer_assertions와 expect_abstention을 동시에 쓰는 조합은
    M2_Phase1_code_review_result.md P1 반영으로 GoldenCase 자체가 거부하므로
    (TestGoldenCaseInvalid.test_assertions_and_abstention_together_raises)
    is_answer_eval_eligible()에 도달할 수 없다 — 여기서는 남은 3가지 조합만
    검증한다."""

    def test_assertions_only_is_eligible(self):
        case = GoldenCase(**_base(answer_assertions=[{"any_of": ["x"]}]))
        assert is_answer_eval_eligible(case) is True

    def test_expect_abstention_only_is_eligible(self):
        case = GoldenCase(**_base(expect_abstention=True))
        assert is_answer_eval_eligible(case) is True

    def test_neither_is_not_eligible(self):
        case = GoldenCase(**_base(relevant_sources=["a.pdf"]))
        assert is_answer_eval_eligible(case) is False
