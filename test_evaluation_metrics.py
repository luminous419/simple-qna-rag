"""evaluation/metrics.py 단위 테스트 (M2 Phase 3, M2-REQ-011).

모델/vectorstore/Ollama/네트워크를 전혀 사용하지 않는다.
"""

from __future__ import annotations

import unicodedata

import pytest

from evaluation.metrics import (
    assertion_coverage,
    dedupe_preserve_order,
    mean_median,
    mrr_at_k,
    ndcg_at_k,
    normalize_relevance_grades,
    percentile,
    precision_recall_f1,
    recall_at_k,
)
from evaluation.schema import AnswerAssertion


class TestDedupePreserveOrder:
    def test_preserves_first_occurrence_order(self):
        assert dedupe_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_no_duplicates_is_noop(self):
        assert dedupe_preserve_order(["a", "b", "c"]) == ["a", "b", "c"]

    def test_empty_list(self):
        assert dedupe_preserve_order([]) == []


class TestRecallAtK:
    def test_empty_search_results_returns_zero(self):
        assert recall_at_k([], {"a", "b"}, 5) == 0.0

    def test_multiple_relevant_sources_partial_hit(self):
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5

    def test_all_relevant_found(self):
        assert recall_at_k(["a", "b", "x"], {"a", "b"}, 3) == 1.0

    def test_empty_relevant_ids_raises(self):
        with pytest.raises(ValueError):
            recall_at_k(["a", "b"], set(), 5)

    def test_duplicate_source_at_top_k_boundary(self):
        """M2-REQ-011: 중복 source가 top-k 경계에 걸치는 경우, ranked_ids는 이미
        dedupe_preserve_order()를 거쳤다고 전제하므로 A,A,A,B 원본이 아니라
        [A, B]가 입력으로 들어와야 한다. k=1이면 B는 포함되지 않아야 한다."""
        deduped = dedupe_preserve_order(["A", "A", "A", "B"])
        assert deduped == ["A", "B"]
        assert recall_at_k(deduped, {"A", "B"}, 1) == 0.5
        assert recall_at_k(deduped, {"A", "B"}, 2) == 1.0

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_raises(self, k):
        with pytest.raises(ValueError):
            recall_at_k(["a", "b"], {"a"}, k)


class TestMrrAtK:
    def test_empty_search_results_returns_zero(self):
        assert mrr_at_k([], {"a"}, 5) == 0.0

    def test_first_relevant_at_rank_one(self):
        assert mrr_at_k(["a", "b"], {"a"}, 5) == 1.0

    def test_first_relevant_at_rank_three(self):
        assert mrr_at_k(["x", "y", "a"], {"a"}, 5) == pytest.approx(1 / 3)

    def test_relevant_outside_k_returns_zero(self):
        assert mrr_at_k(["x", "y", "a"], {"a"}, 2) == 0.0

    def test_not_found_returns_zero(self):
        assert mrr_at_k(["x", "y"], {"a"}, 5) == 0.0

    def test_empty_relevant_ids_raises(self):
        with pytest.raises(ValueError):
            mrr_at_k(["a", "b"], set(), 5)

    def test_duplicate_source_at_top_k_boundary(self):
        deduped = dedupe_preserve_order(["A", "A", "A", "B"])
        assert mrr_at_k(deduped, {"B"}, 1) == 0.0
        assert mrr_at_k(deduped, {"B"}, 2) == 0.5

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_raises(self, k):
        with pytest.raises(ValueError):
            mrr_at_k(["a", "b"], {"a"}, k)


class TestNdcgAtK:
    def test_hand_computed_example(self):
        # rank1 grade3 gain=7 /log2(2)=1 -> 7.0 ; rank2 grade1 gain=1 /log2(3) -> 0.6309...
        # ideal order is the same as ranked order here, so ndcg == 1.0
        grades = {"a": 3, "b": 1}
        assert ndcg_at_k(["a", "b"], grades, 2) == pytest.approx(1.0)

    def test_non_ideal_order_is_below_one(self):
        grades = {"a": 3, "b": 1}
        # worst order first: b then a
        value = ndcg_at_k(["b", "a"], grades, 2)
        assert 0.0 < value < 1.0

    def test_relevance_grade_missing_treated_as_zero(self):
        grades = {"a": 3}
        # "x" has no grade entry -> gain 0, should not error and should not inflate score
        value = ndcg_at_k(["x", "a"], grades, 2)
        assert 0.0 < value <= 1.0

    def test_empty_search_results_returns_zero(self):
        assert ndcg_at_k([], {"a": 3}, 5) == 0.0

    def test_empty_relevance_grades_returns_zero_without_raising(self):
        assert ndcg_at_k(["a", "b"], {}, 5) == 0.0

    def test_duplicate_source_at_top_k_boundary(self):
        deduped = dedupe_preserve_order(["A", "A", "A", "B"])
        grades = {"A": 1, "B": 3}
        # B is the more relevant doc but appears second after dedupe; k=1 should miss it
        assert ndcg_at_k(deduped, grades, 1) < ndcg_at_k(deduped, grades, 2)

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_raises(self, k):
        with pytest.raises(ValueError):
            ndcg_at_k(["a", "b"], {"a": 1}, k)


class TestPrecisionRecallF1:
    def test_perfect_predictions(self):
        result = precision_recall_f1(
            ["document_qa", "web_search"], ["document_qa", "web_search"], ["document_qa", "web_search"]
        )
        assert result["document_qa"]["precision"] == 1.0
        assert result["document_qa"]["recall"] == 1.0
        assert result["document_qa"]["f1"] == 1.0
        assert result["web_search"]["f1"] == 1.0

    def test_all_predictions_one_route(self):
        """M2-REQ-011: 모든 예측이 한 route인 경우."""
        y_true = ["document_qa", "web_search", "document_qa"]
        y_pred = ["document_qa", "document_qa", "document_qa"]
        result = precision_recall_f1(y_true, y_pred, ["document_qa", "web_search"])
        assert result["document_qa"]["precision"] == pytest.approx(2 / 3)
        assert result["document_qa"]["recall"] == 1.0
        assert result["web_search"]["precision"] == 0.0
        assert result["web_search"]["recall"] == 0.0
        assert result["web_search"]["f1"] == 0.0

    def test_confusion_matrix_values(self):
        y_true = ["document_qa", "web_search"]
        y_pred = ["web_search", "web_search"]
        result = precision_recall_f1(y_true, y_pred, ["document_qa", "web_search"])
        assert result["confusion_matrix"]["document_qa"]["web_search"] == 1
        assert result["confusion_matrix"]["web_search"]["web_search"] == 1
        assert result["confusion_matrix"]["document_qa"]["document_qa"] == 0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            precision_recall_f1(["document_qa"], ["document_qa", "web_search"], ["document_qa", "web_search"])


class TestPercentile:
    def test_p95_hand_computed(self):
        values = list(range(1, 101))  # 1..100
        assert percentile(values, 95) == 95

    def test_p50_hand_computed(self):
        values = [10, 20, 30, 40]
        assert percentile(values, 50) == 20

    def test_empty_latency_list_returns_none(self):
        assert percentile([], 95) is None

    def test_single_value(self):
        assert percentile([42.0], 50) == 42.0

    def test_p_zero_and_p_hundred_are_valid_boundaries(self):
        values = [10, 20, 30]
        assert percentile(values, 0) == 10
        assert percentile(values, 100) == 30

    @pytest.mark.parametrize("p", [-1, 100.0001, 101])
    def test_out_of_range_p_raises_with_nonempty_values(self, p):
        """M2_Phase3_code_review_result.md P2: 범위 밖 p는 조용히 clamp되면 안 된다."""
        with pytest.raises(ValueError):
            percentile([10, 20, 30], p)

    @pytest.mark.parametrize("p", [-1, 101])
    def test_out_of_range_p_raises_even_with_empty_values(self, p):
        """values가 비어 있어 결국 None을 반환하더라도, 잘못된 p 호출 자체는
        숨기지 않고 즉시 실패해야 한다."""
        with pytest.raises(ValueError):
            percentile([], p)


class TestMeanMedian:
    def test_basic(self):
        mean, median = mean_median([1, 2, 3, 4])
        assert mean == 2.5
        assert median == 2.5

    def test_empty_latency_list_returns_none_none(self):
        assert mean_median([]) == (None, None)


class TestAssertionCoverage:
    def test_full_match(self):
        assertions = [AnswerAssertion(any_of=["검색과 생성을 통합"])]
        passed, total = assertion_coverage("RAG는 검색과 생성을 통합합니다.", assertions)
        assert (passed, total) == (1, 1)

    def test_partial_match(self):
        assertions = [
            AnswerAssertion(any_of=["단방향"]),
            AnswerAssertion(any_of=["순환", "Cycle"]),
        ]
        passed, total = assertion_coverage("RAG는 단방향 구조입니다.", assertions)
        assert (passed, total) == (1, 2)

    def test_no_assertions_returns_zero_zero(self):
        assert assertion_coverage("아무 답변", []) == (0, 0)

    def test_nfc_nfd_and_casefold_equivalence(self):
        nfd_phrase = unicodedata.normalize("NFD", "가나다")
        assertions = [AnswerAssertion(any_of=[nfd_phrase])]
        nfc_answer = unicodedata.normalize("NFC", "이것은 가나다 입니다")
        passed, total = assertion_coverage(nfc_answer, assertions)
        assert (passed, total) == (1, 1)

    def test_any_of_synonym_alternative_matches(self):
        assertions = [AnswerAssertion(any_of=["교차 인코더", "cross-encoder"])]
        passed, total = assertion_coverage("Reranker는 CROSS-ENCODER 구조입니다.", assertions)
        assert (passed, total) == (1, 1)


class TestNormalizeRelevanceGrades:
    def test_basic_normalization(self):
        result = normalize_relevance_grades({"A.PDF": 3, "b.pdf": 1})
        assert result == {"a.pdf": 3, "b.pdf": 1}

    def test_collision_raises(self):
        with pytest.raises(ValueError):
            normalize_relevance_grades({"A.pdf": 1, "a.pdf": 2})

    def test_nfc_nfd_collision_raises(self):
        nfc = unicodedata.normalize("NFC", "가나다.pdf")
        nfd = unicodedata.normalize("NFD", "가나다.pdf")
        assert nfc != nfd
        with pytest.raises(ValueError):
            normalize_relevance_grades({nfc: 1, nfd: 2})

    def test_windows_posix_path_collision_raises(self):
        with pytest.raises(ValueError):
            normalize_relevance_grades({"data\\a.pdf": 1, "data/a.pdf": 2})

    def test_no_collision_distinct_files(self):
        result = normalize_relevance_grades({"a.pdf": 1, "b.pdf": 2})
        assert result == {"a.pdf": 1, "b.pdf": 2}
