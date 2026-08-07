"""evaluation/answers.py 단위 테스트 (M2 Phase 6).

이 모듈은 실제 Ollama/vectorstore/네트워크를 전혀 사용하지 않는다.
`evaluation.answers._get_engine`과 `evaluation.answers.build_reproducibility_metadata`를
monkeypatch해 fake engine과 fake reproducibility metadata를 주입한다.

테스트는 고정 fixture(이 파일 안에서 직접 만든 GoldenCase 리스트)를 사용하고,
실제 golden.jsonl의 사례 수를 하드코딩해 assert하지 않는다 — Phase 5가 병렬로
golden.jsonl에 사례를 추가하고 있어서 병합 후 전체 건수가 달라지기 때문이다.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

import evaluation.answers as answers_module
from evaluation.answers import (
    ABSTENTION_PHRASES,
    _abstention_confusion,
    _detect_abstention,
    _extract_returned_source_ids,
    _fence_for,
    _source_match,
    evaluate_answers,
    main,
    write_review_worksheet,
)
from evaluation.schema import AnswerAssertion, GoldenCase, is_answer_eval_eligible


def _case(**overrides) -> GoldenCase:
    base = {
        "id": "case-1",
        "question": "테스트 질문입니다",
        "category": "document_qa",
        "expected_route": "document_qa",
        "tags": ["unit"],
    }
    base.update(overrides)
    return GoldenCase(**base)


ASSERTION_CASE = _case(
    id="assertion-1",
    question="RAG는 무엇인가요?",
    expected_intent="explanation",
    relevant_sources=["doc1.pdf"],
    answer_assertions=[AnswerAssertion(any_of=["핵심 사실"])],
)

ABSTENTION_CASE = _case(
    id="abstention-1",
    question="이 문서에 없는 질문입니다",
    category="unanswerable",
    expect_abstention=True,
)

RETRIEVAL_ONLY_CASE = _case(
    id="retrieval-only-1",
    question="검색만 평가하는 사례입니다",
    relevant_sources=["doc2.pdf"],
    relevance_grades={"doc2.pdf": 3},
)


class FakeEngine:
    """question -> response(dict) 또는 Exception 매핑으로 동작하는 fake engine."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[str] = []

    def query(self, question: str) -> dict:
        self.calls.append(question)
        response = self.responses[question]
        if isinstance(response, Exception):
            raise response
        return response


def _success_response(answer: str, sources=None, intent="explanation") -> dict:
    return {
        "answer": answer,
        "sources": sources if sources is not None else [],
        "success": True,
        "intent": intent,
    }


def _patch_repro(monkeypatch, data_dir_ok: bool = True):
    if data_dir_ok:
        monkeypatch.setattr(
            answers_module,
            "build_reproducibility_metadata",
            lambda data_dir, vs_path: {
                "corpus_manifest": [{"source_id": "a.pdf", "size_bytes": 1, "sha256": "x"}],
                "corpus_manifest_sha256": "deadbeef",
                "vectorstore_fingerprint": {"index_faiss_sha256": "y", "index_pkl_sha256": "z"},
                "reproducibility_note": None,
            },
        )
    else:
        def _raise(data_dir, vs_path):
            raise FileNotFoundError("data_dir가 존재하지 않습니다: ./data")

        monkeypatch.setattr(answers_module, "build_reproducibility_metadata", _raise)


class TestEligibilityUsesCommonFunction:
    def test_answers_module_reuses_schema_eligibility_function(self):
        """evaluate_answers()가 evaluation.schema.is_answer_eval_eligible을 그대로
        import해 쓰고 있으며, 별도로 재정의하지 않았는지 확인한다."""
        assert answers_module.is_answer_eval_eligible is is_answer_eval_eligible

    def test_assertion_case_and_abstention_case_are_eligible(self):
        assert is_answer_eval_eligible(ASSERTION_CASE) is True
        assert is_answer_eval_eligible(ABSTENTION_CASE) is True

    def test_retrieval_only_case_is_not_eligible(self):
        assert is_answer_eval_eligible(RETRIEVAL_ONLY_CASE) is False


class TestDetectAbstention:
    def test_first_official_phrase_detected(self):
        assert _detect_abstention("제공된 문서에서 관련 정보를 찾을 수 없습니다.") is True

    def test_second_official_phrase_detected(self):
        assert _detect_abstention("제공된 문서만으로는 확실한 답변이 어렵습니다.") is True

    def test_non_matching_answer_not_detected(self):
        assert _detect_abstention("RAG는 검색과 생성을 결합한 방식입니다.") is False

    def test_nfd_normalized_phrase_is_still_detected(self):
        nfd_phrase = unicodedata.normalize("NFD", ABSTENTION_PHRASES[0])
        answer = f"안내: {nfd_phrase} 죄송합니다."
        assert _detect_abstention(answer) is True

    def test_similar_but_not_exact_phrase_is_not_detected(self):
        # 문구 일부를 바꾼 문장은 부분일치라도 오탐되지 않아야 한다(정확 포함 매칭 확인).
        altered = "제공된 자료에서 관련 정보를 찾을 수 없습니다."
        assert _detect_abstention(altered) is False


class TestExtractReturnedSourceIds:
    def test_valid_dict_entries_are_kept(self):
        sources = [
            {"index": 1, "source": "a.pdf", "page": 2, "content": "..."},
            {"index": 2, "source": "b.pdf", "page": None, "content": "..."},
        ]
        ids, skipped = _extract_returned_source_ids(sources)
        assert ids == ["a.pdf", "b.pdf"]
        assert skipped == 0

    def test_missing_source_key_is_skipped(self):
        sources = [{"index": 1, "page": 1, "content": "..."}]
        ids, skipped = _extract_returned_source_ids(sources)
        assert ids == []
        assert skipped == 1

    def test_non_string_source_is_skipped(self):
        sources = [{"index": 1, "source": 123, "page": 1, "content": "..."}]
        ids, skipped = _extract_returned_source_ids(sources)
        assert ids == []
        assert skipped == 1

    def test_empty_string_source_is_skipped(self):
        sources = [{"index": 1, "source": "", "page": 1, "content": "..."}]
        ids, skipped = _extract_returned_source_ids(sources)
        assert ids == []
        assert skipped == 1

    def test_mixed_valid_and_invalid_entries(self):
        sources = [
            {"index": 1, "source": "a.pdf", "page": 1, "content": "..."},
            {"index": 2, "page": 1, "content": "..."},
            {"index": 3, "source": None, "page": 1, "content": "..."},
            {"index": 4, "source": "", "page": 1, "content": "..."},
        ]
        ids, skipped = _extract_returned_source_ids(sources)
        assert ids == ["a.pdf"]
        assert skipped == 3


class TestSourceMatch:
    def test_no_relevant_sources_returns_none(self):
        assert _source_match(["a.pdf"], []) is None

    def test_full_match(self):
        result = _source_match(["a.pdf", "b.pdf"], ["a.pdf", "b.pdf"])
        assert result["source_any_hit"] is True
        assert result["source_recall"] == 1.0

    def test_partial_match(self):
        result = _source_match(["a.pdf"], ["a.pdf", "b.pdf"])
        assert result["source_any_hit"] is True
        assert result["source_recall"] == 0.5

    def test_no_match(self):
        result = _source_match(["c.pdf"], ["a.pdf", "b.pdf"])
        assert result["source_any_hit"] is False
        assert result["source_recall"] == 0.0

    def test_normalization_and_dedupe_applied(self):
        result = _source_match(["A.PDF", "a.pdf", "sub/a.pdf"], ["a.pdf"])
        assert result["source_any_hit"] is True
        assert result["source_recall"] == 1.0


class TestAbstentionConfusion:
    def test_all_four_outcomes_counted_correctly(self):
        flags = [
            (True, True),  # TP
            (False, False),  # TN
            (False, True),  # FP
            (True, False),  # FN
        ]
        result = _abstention_confusion(flags)
        assert result["true_positive"] == 1
        assert result["true_negative"] == 1
        assert result["false_positive"] == 1
        assert result["false_negative"] == 1
        assert result["accuracy"] == 0.5
        assert result["abstention_accuracy_excluded_reason"] is None

    def test_empty_flags_returns_none_accuracy_with_reason(self):
        result = _abstention_confusion([])
        assert result["accuracy"] is None
        assert result["abstention_accuracy_excluded_reason"]


class TestFenceFor:
    def test_no_backticks_uses_minimum_three(self):
        assert _fence_for("plain answer") == "```"

    def test_three_consecutive_backticks_uses_four(self):
        assert _fence_for("here is ```code``` inline") == "````"

    def test_four_or_more_consecutive_backticks_uses_five(self):
        assert _fence_for("````long fence````") == "`````"

    def test_longest_run_wins_over_multiple_shorter_runs(self):
        text = "``a`` and ```b``` and `c`"
        assert _fence_for(text) == "````"


class TestEvaluateAnswers:
    def _write_dataset(self, tmp_path, cases):
        path = tmp_path / "golden.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(json.loads(case.model_dump_json()), ensure_ascii=False) + "\n")
        return path

    @pytest.mark.parametrize("bad_limit", [0, -1])
    def test_non_positive_limit_raises_even_when_called_directly(self, tmp_path, bad_limit):
        """M2_Phase_4_5_6_code_review_result.md 재리뷰 P3: CLI의 _positive_int는
        --limit 0/-1을 막지만, evaluate_answers()를 Python에서 직접 호출하면
        이 계층을 거치지 않는다 — 공개 함수 자체도 방어해야 한다."""
        dataset_path = self._write_dataset(tmp_path, [ASSERTION_CASE])
        with pytest.raises(ValueError):
            evaluate_answers(dataset_path, tmp_path / "reports", limit=bad_limit)

    def test_retrieval_only_case_excluded_and_others_scored(self, tmp_path, monkeypatch):
        dataset_path = self._write_dataset(
            tmp_path, [ASSERTION_CASE, ABSTENTION_CASE, RETRIEVAL_ONLY_CASE]
        )
        _patch_repro(monkeypatch)
        fake_engine = FakeEngine(
            {
                ASSERTION_CASE.question: _success_response(
                    "이것은 핵심 사실을 포함한 답변입니다.",
                    sources=[{"index": 1, "source": "doc1.pdf", "page": 1, "content": "청크 원문 비밀"}],
                    intent="explanation",
                ),
                ABSTENTION_CASE.question: _success_response(
                    "제공된 문서에서 관련 정보를 찾을 수 없습니다.", intent="other"
                ),
            }
        )
        monkeypatch.setattr(answers_module, "_get_engine", lambda: fake_engine)

        output_dir = tmp_path / "reports"
        result = evaluate_answers(dataset_path, output_dir)

        assert result["case_counts"]["total_considered"] == 3
        assert result["case_counts"]["eligible"] == 2
        assert result["case_counts"]["excluded_non_eligible"] == 1
        assert result["case_counts"]["success"] == 2
        assert result["case_counts"]["failure"] == 0
        assert set(fake_engine.calls) == {ASSERTION_CASE.question, ABSTENTION_CASE.question}
        assert result["corpus_manifest_sha256"] == "deadbeef"
        assert result["vectorstore_fingerprint"] is not None

    def test_assertion_cases_scored_excludes_no_assertion_and_failure_cases(
        self, tmp_path, monkeypatch
    ):
        """M2_Phase_4_5_6_code_review_result.md P2: assertion.cases_scored는
        success_count 전체가 아니라 "성공했고 answer_assertions가 1개 이상 있는"
        사례 수만 세야 한다. ASSERTION_CASE(assertion 1개)는 채점되고,
        ABSTENTION_CASE(assertion 없음, expect_abstention=True로만 대상)는
        cases_excluded_no_assertion으로, 질의 실패 사례는 cases_excluded_failure로
        각각 별도 집계돼야 하며 세 값의 합은 eligible 수와 같아야 한다."""
        failing_case = _case(
            id="fail-assertion",
            question="질의가 실패하는 질문",
            answer_assertions=[AnswerAssertion(any_of=["핵심"])],
        )
        dataset_path = self._write_dataset(
            tmp_path, [ASSERTION_CASE, ABSTENTION_CASE, failing_case]
        )
        _patch_repro(monkeypatch)
        fake_engine = FakeEngine(
            {
                ASSERTION_CASE.question: _success_response("핵심 사실이 포함된 답변"),
                ABSTENTION_CASE.question: _success_response(
                    "제공된 문서에서 관련 정보를 찾을 수 없습니다."
                ),
                failing_case.question: RuntimeError("모델 호출 실패"),
            }
        )
        monkeypatch.setattr(answers_module, "_get_engine", lambda: fake_engine)

        result = evaluate_answers(dataset_path, tmp_path / "reports")

        assertion = result["assertion"]
        assert assertion["cases_scored"] == 1  # ASSERTION_CASE만
        assert assertion["cases_excluded_no_assertion"] == 1  # ABSTENTION_CASE
        assert assertion["cases_excluded_failure"] == 1  # failing_case
        assert (
            assertion["cases_scored"]
            + assertion["cases_excluded_no_assertion"]
            + assertion["cases_excluded_failure"]
            == result["case_counts"]["eligible"]
        )

    def test_query_exception_and_success_false_do_not_pollute_scoring_and_continue(
        self, tmp_path, monkeypatch
    ):
        exception_case = _case(
            id="exc-1",
            question="예외가 발생하는 질문",
            answer_assertions=[AnswerAssertion(any_of=["핵심"])],
        )
        success_false_case = _case(
            id="fail-1",
            question="success=False가 반환되는 질문",
            answer_assertions=[AnswerAssertion(any_of=["핵심"])],
        )
        after_case = _case(
            id="after-1",
            question="이후에도 계속 평가되는 질문",
            answer_assertions=[AnswerAssertion(any_of=["핵심 사실"])],
        )
        dataset_path = self._write_dataset(
            tmp_path, [exception_case, success_false_case, after_case]
        )
        _patch_repro(monkeypatch)
        fake_engine = FakeEngine(
            {
                exception_case.question: RuntimeError("모델 호출 실패"),
                success_false_case.question: {
                    "answer": "오류가 발생했습니다: boom",
                    "sources": [],
                    "success": False,
                    "intent": "error",
                },
                after_case.question: _success_response("이것은 핵심 사실을 포함합니다."),
            }
        )
        monkeypatch.setattr(answers_module, "_get_engine", lambda: fake_engine)

        result = evaluate_answers(dataset_path, tmp_path / "reports")

        assert result["case_counts"]["eligible"] == 3
        assert result["case_counts"]["success"] == 1
        assert result["case_counts"]["failure"] == 2
        assert len(result["failures"]) == 2
        failure_ids = {f["id"] for f in result["failures"]}
        assert failure_ids == {"exc-1", "fail-1"}
        # 실패 사례는 assertion/abstention 채점에서 제외되고 성공 사례(after-1)만 반영된다.
        assert result["assertion"]["cases_scored"] == 1
        assert result["assertion"]["assertions_total"] == 1
        assert result["assertion"]["assertions_passed"] == 1
        assert result["abstention"]["evaluated_count"] == 1
        # after_case가 계속 처리됐는지(사례 하나 실패해도 전체가 멈추지 않는지) 확인.
        assert set(fake_engine.calls) == {
            exception_case.question,
            success_false_case.question,
            after_case.question,
        }

    def test_intent_match_and_mismatch_and_exclusion(self, tmp_path, monkeypatch):
        """ANSWER_TEMPLATE_MODE="intent"일 때만 성립하는 기존 채점 계약(Design.md
        §8.6)을 검증한다 — classifier가 실제로 호출되는 모드이므로 expected_intent가
        있는 사례만 채점하고 없는 사례는 제외한다."""
        monkeypatch.setattr(answers_module.config, "ANSWER_TEMPLATE_MODE", "intent")
        match_case = _case(
            id="intent-match",
            question="일치하는 intent 질문",
            expected_intent="explanation",
            answer_assertions=[AnswerAssertion(any_of=["사실"])],
        )
        mismatch_case = _case(
            id="intent-mismatch",
            question="불일치하는 intent 질문",
            expected_intent="comparison",
            answer_assertions=[AnswerAssertion(any_of=["사실"])],
        )
        no_expected_case = _case(
            id="intent-none",
            question="expected_intent가 없는 질문",
            answer_assertions=[AnswerAssertion(any_of=["사실"])],
        )
        dataset_path = self._write_dataset(
            tmp_path, [match_case, mismatch_case, no_expected_case]
        )
        _patch_repro(monkeypatch)
        fake_engine = FakeEngine(
            {
                match_case.question: _success_response("사실이 포함된 답변", intent="explanation"),
                mismatch_case.question: _success_response("사실이 포함된 답변", intent="procedure"),
                no_expected_case.question: _success_response("사실이 포함된 답변", intent="other"),
            }
        )
        monkeypatch.setattr(answers_module, "_get_engine", lambda: fake_engine)

        result = evaluate_answers(dataset_path, tmp_path / "reports")

        assert result["answer_template_mode"] == "intent"
        assert result["intent"]["evaluated_count"] == 2
        assert result["intent"]["correct_count"] == 1
        assert result["intent"]["excluded_count"] == 1
        assert result["intent"]["accuracy"] == 0.5
        assert result["intent"]["intent_excluded_reason"] is None

    def test_default_template_mode_excludes_intent_entirely(self, tmp_path, monkeypatch):
        """M2 (Code_Review_Iteration_2.md): ANSWER_TEMPLATE_MODE="default"에서는
        RAGEngine.query()가 classify_intent()를 호출하지 않고 응답 계약 보존을
        위해 intent="other"만 반환한다(Design.md §8.6). expected_intent와 비교해
        채점하면 비활성 classifier가 실패한 classifier로 오독되는 잘못된 0%
        accuracy가 나온다 — default 모드에서는 intent 채점을 전부 제외하고
        accuracy=null, evaluated_count=0, excluded_count=eligible 전체,
        intent_excluded_reason을 명시해야 한다."""
        monkeypatch.setattr(answers_module.config, "ANSWER_TEMPLATE_MODE", "default")
        match_case = _case(
            id="intent-match",
            question="일치하는 intent 질문",
            expected_intent="explanation",
            answer_assertions=[AnswerAssertion(any_of=["사실"])],
        )
        mismatch_case = _case(
            id="intent-mismatch",
            question="불일치하는 intent 질문",
            expected_intent="comparison",
            answer_assertions=[AnswerAssertion(any_of=["사실"])],
        )
        no_expected_case = _case(
            id="intent-none",
            question="expected_intent가 없는 질문",
            answer_assertions=[AnswerAssertion(any_of=["사실"])],
        )
        dataset_path = self._write_dataset(
            tmp_path, [match_case, mismatch_case, no_expected_case]
        )
        _patch_repro(monkeypatch)
        fake_engine = FakeEngine(
            {
                match_case.question: _success_response("사실이 포함된 답변", intent="other"),
                mismatch_case.question: _success_response("사실이 포함된 답변", intent="other"),
                no_expected_case.question: _success_response("사실이 포함된 답변", intent="other"),
            }
        )
        monkeypatch.setattr(answers_module, "_get_engine", lambda: fake_engine)

        result = evaluate_answers(dataset_path, tmp_path / "reports")

        assert result["answer_template_mode"] == "default"
        assert result["intent"]["evaluated_count"] == 0
        assert result["intent"]["correct_count"] == 0
        assert result["intent"]["excluded_count"] == result["case_counts"]["eligible"] == 3
        assert result["intent"]["accuracy"] is None
        assert result["intent"]["intent_excluded_reason"] == (
            "ANSWER_TEMPLATE_MODE=default (classifier 비활성)"
        )
        # 사례별 intent_match도 default 모드에서는 채점 대상이 아니므로 항상 None이어야 한다.
        assert all(c["intent_match"] is None for c in result["case_results"])

    def test_source_evaluation_excludes_cases_without_relevant_sources(self, tmp_path, monkeypatch):
        dataset_path = self._write_dataset(tmp_path, [ABSTENTION_CASE])
        _patch_repro(monkeypatch)
        fake_engine = FakeEngine(
            {
                ABSTENTION_CASE.question: _success_response(
                    "제공된 문서에서 관련 정보를 찾을 수 없습니다."
                )
            }
        )
        monkeypatch.setattr(answers_module, "_get_engine", lambda: fake_engine)

        result = evaluate_answers(dataset_path, tmp_path / "reports")
        assert result["source"]["evaluated_count"] == 0
        assert result["source"]["excluded_count"] == 1

    def test_reproducibility_fields_are_non_null(self, tmp_path, monkeypatch):
        dataset_path = self._write_dataset(tmp_path, [ASSERTION_CASE])
        _patch_repro(monkeypatch)
        fake_engine = FakeEngine(
            {ASSERTION_CASE.question: _success_response("핵심 사실이 포함된 답변")}
        )
        monkeypatch.setattr(answers_module, "_get_engine", lambda: fake_engine)

        result = evaluate_answers(dataset_path, tmp_path / "reports")
        assert result["corpus_manifest"] is not None
        assert result["corpus_manifest_sha256"] is not None
        assert result["vectorstore_fingerprint"] is not None

    def test_worksheet_never_contains_source_chunk_content(self, tmp_path, monkeypatch):
        secret_chunk = "이것은-검색된-청크-원문-절대노출금지"
        dataset_path = self._write_dataset(tmp_path, [ASSERTION_CASE])
        _patch_repro(monkeypatch)
        fake_engine = FakeEngine(
            {
                ASSERTION_CASE.question: _success_response(
                    "핵심 사실이 포함된 답변",
                    sources=[{"index": 1, "source": "doc1.pdf", "page": 1, "content": secret_chunk}],
                )
            }
        )
        monkeypatch.setattr(answers_module, "_get_engine", lambda: fake_engine)

        output_dir = tmp_path / "reports"
        result = evaluate_answers(dataset_path, output_dir)

        worksheet_text = open(result["worksheet_path"], encoding="utf-8").read()
        assert secret_chunk not in worksheet_text
        assert "doc1.pdf" in worksheet_text

    def test_markdown_report_shows_aggregate_metrics_and_failures_not_just_json_pointer(
        self, tmp_path, monkeypatch
    ):
        """M2_Phase_4_5_6_code_review_result.md P1: Answer aggregate Markdown에는
        assertion pass rate/abstention 정확도/source hit·recall/intent 정확도/
        latency/실패 수가 표시돼야 한다."""
        failing_case = _case(
            id="fail-1",
            question="질의가 실패하는 질문",
            answer_assertions=[AnswerAssertion(any_of=["핵심"])],
        )
        dataset_path = self._write_dataset(tmp_path, [ASSERTION_CASE, failing_case])
        _patch_repro(monkeypatch)
        fake_engine = FakeEngine(
            {
                ASSERTION_CASE.question: _success_response(
                    "핵심 사실이 포함된 답변",
                    sources=[{"index": 1, "source": "doc1.pdf", "page": 1, "content": "x"}],
                ),
                failing_case.question: RuntimeError("모델 호출 실패"),
            }
        )
        monkeypatch.setattr(answers_module, "_get_engine", lambda: fake_engine)

        output_dir = tmp_path / "reports"
        result = evaluate_answers(dataset_path, output_dir)
        md_path = Path(result["report_markdown_path"])
        text = md_path.read_text(encoding="utf-8")

        assert "pass rate" in text
        assert "100.0%" in text  # assertion pass rate: 1/1
        assert "fail-1" in text
        assert "모델 호출 실패" in text

    def test_markdown_failure_row_with_pipe_and_newline_keeps_column_count(self, tmp_path, monkeypatch):
        """M2_Phase_4_5_6_code_review_result.md 재리뷰 P2: 질문이나 오류 메시지에
        pipe/줄바꿈이 들어가도 실패 사례 표의 열 개수(3개)가 유지돼야 한다."""
        failing_case = _case(
            id="fail-1",
            question="질문 | 파이프\n줄바꿈",
            answer_assertions=[AnswerAssertion(any_of=["핵심"])],
        )
        dataset_path = self._write_dataset(tmp_path, [failing_case])
        _patch_repro(monkeypatch)
        fake_engine = FakeEngine({failing_case.question: RuntimeError("오류 | 발생")})
        monkeypatch.setattr(answers_module, "_get_engine", lambda: fake_engine)

        output_dir = tmp_path / "reports"
        result = evaluate_answers(dataset_path, output_dir)
        md_path = Path(result["report_markdown_path"])
        text = md_path.read_text(encoding="utf-8")

        failure_row = next(line for line in text.splitlines() if line.startswith("| fail-1"))
        # delimiter 4개(3열 경계) + question/error에 각 1개씩 이스케이프된 pipe = 6개.
        assert failure_row.count("|") == 6
        assert "\n" not in failure_row.strip()

    def test_tag_filter_and_limit_applied_deterministically(self, tmp_path, monkeypatch):
        tagged_case = _case(
            id="tagged-1",
            question="태그가 있는 질문",
            tags=["unit", "special"],
            answer_assertions=[AnswerAssertion(any_of=["사실"])],
        )
        untagged_case = _case(
            id="untagged-1",
            question="태그가 없는 질문",
            tags=["unit"],
            answer_assertions=[AnswerAssertion(any_of=["사실"])],
        )
        dataset_path = self._write_dataset(tmp_path, [tagged_case, untagged_case])
        _patch_repro(monkeypatch)
        fake_engine = FakeEngine(
            {tagged_case.question: _success_response("사실이 포함된 답변")}
        )
        monkeypatch.setattr(answers_module, "_get_engine", lambda: fake_engine)

        result = evaluate_answers(dataset_path, tmp_path / "reports", tag="special")
        assert result["case_counts"]["total_considered"] == 1
        assert result["case_counts"]["eligible"] == 1
        assert fake_engine.calls == [tagged_case.question]


class TestWriteReviewWorksheet:
    def test_answer_with_backticks_pipes_and_multiple_lines_preserves_structure(self, tmp_path):
        tricky_answer = "```코드블록``` 안에 | 파이프와\n여러 줄이\n섞여 있습니다 ````."
        results = [
            {
                "id": "tricky-1",
                "question": "복잡한 질문 | 파이프 포함\n줄바꿈도 포함",
                "status": "success",
                "error": None,
                "assertion_passed": 1,
                "assertion_total": 1,
                "abstention_match": True,
                "source_any_hit": True,
                "source_recall": 1.0,
                "intent_match": True,
                "returned_sources": [{"source": "doc1.pdf", "page": 2}],
                "answer_assertions_summary": ["핵심 사실 | 대안 표현"],
                "answer": tricky_answer,
            }
        ]
        output_path = tmp_path / "worksheet.md"
        write_review_worksheet(results, output_path)
        text = output_path.read_text(encoding="utf-8")

        expected_fence = _fence_for(tricky_answer)
        assert text.count(expected_fence) >= 2
        assert tricky_answer in text
        assert "## tricky-1" in text
        assert "doc1.pdf" in text
        assert "Faithfulness (1-5)" in text
        assert "Reviewer note" in text

    def test_empty_results_produces_valid_placeholder(self, tmp_path):
        output_path = tmp_path / "worksheet.md"
        write_review_worksheet([], output_path)
        text = output_path.read_text(encoding="utf-8")
        assert "Answer 평가 대상 사례가 없습니다" in text

    def test_failure_case_section_does_not_crash_and_shows_error(self, tmp_path):
        results = [
            {
                "id": "fail-1",
                "question": "실패한 질문",
                "status": "failure",
                "error": "모델 호출 실패",
                "returned_sources": [],
                "answer_assertions_summary": [],
                "answer": None,
            }
        ]
        output_path = tmp_path / "worksheet.md"
        write_review_worksheet(results, output_path)
        text = output_path.read_text(encoding="utf-8")
        assert "모델 호출 실패" in text
        assert "## fail-1" in text


class TestCLI:
    def test_help_exits_zero_without_loading_model(self, monkeypatch, capsys):
        def _fail_if_called():
            raise AssertionError("--help 처리 중 RAG 엔진을 로드하면 안 됩니다")

        monkeypatch.setattr(answers_module, "_get_engine", lambda: _fail_if_called())

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    @pytest.mark.parametrize("bad_limit", ["0", "-1"])
    def test_non_positive_limit_is_parser_error_not_negative_slice(self, tmp_path, bad_limit):
        """M2_Phase_4_5_6_code_review_result.md P3: --limit -1은 cases[:-1](마지막
        하나 제외)로 조용히 재해석되면 안 되고 argparse 오류(exit 2)여야 한다."""
        dataset_path = tmp_path / "golden.jsonl"
        dataset_path.write_text(
            json.dumps(json.loads(ASSERTION_CASE.model_dump_json()), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--dataset",
                    str(dataset_path),
                    "--output",
                    str(tmp_path / "out"),
                    "--limit",
                    bad_limit,
                ]
            )
        assert exc_info.value.code == 2

    def test_missing_dataset_file_returns_nonzero_with_guidance(self, tmp_path, capsys):
        missing_path = tmp_path / "does_not_exist.jsonl"
        exit_code = main(["--dataset", str(missing_path), "--output", str(tmp_path / "out")])
        assert exit_code != 0
        captured = capsys.readouterr()
        assert "찾을 수 없습니다" in captured.err
        assert "다음 조치" in captured.err

    def test_missing_vectorstore_artifact_returns_nonzero_with_guidance(
        self, tmp_path, monkeypatch
    ):
        dataset_path = tmp_path / "golden.jsonl"
        dataset_path.write_text(
            json.dumps(json.loads(ASSERTION_CASE.model_dump_json()), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _patch_repro(monkeypatch, data_dir_ok=False)

        exit_code = main(["--dataset", str(dataset_path), "--output", str(tmp_path / "out")])
        assert exit_code != 0

    def test_successful_run_exits_zero_and_prints_summary(self, tmp_path, monkeypatch, capsys):
        dataset_path = tmp_path / "golden.jsonl"
        dataset_path.write_text(
            json.dumps(json.loads(ASSERTION_CASE.model_dump_json()), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _patch_repro(monkeypatch)
        fake_engine = FakeEngine(
            {ASSERTION_CASE.question: _success_response("핵심 사실이 포함된 답변")}
        )
        monkeypatch.setattr(answers_module, "_get_engine", lambda: fake_engine)

        exit_code = main(
            ["--dataset", str(dataset_path), "--output", str(tmp_path / "out")]
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert "report_json_path" in payload
        assert "worksheet_path" in payload
