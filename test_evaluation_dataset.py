"""evaluation/dataset.py 단위 테스트 (M2 Phase 1, Development_M2_Quality_Baseline_Design.md §6.2)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evaluation.dataset import (
    DatasetError,
    load_jsonl,
    main,
    validate_composition,
)
from evaluation.schema import GoldenCase


def _case(id_: str, *, question: str = "테스트 질문입니다", category: str = "document_qa",
          route: str = "document_qa", **overrides) -> dict:
    d = {
        "id": id_,
        "question": question,
        "category": category,
        "expected_route": route,
        "tags": [],
    }
    d.update(overrides)
    return d


def _write_jsonl(path: Path, cases: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
        encoding="utf-8",
    )


def _to_cases(cases_data: list[dict]) -> list[GoldenCase]:
    return [GoldenCase.model_validate(d) for d in cases_data]


def _minimal_valid_cases() -> list[dict]:
    """9개 구성 규칙을 정확히 최소치로 충족하는 60개 사례
    (document_qa 40/web_search 10/boundary 5/unanswerable 5).
    document_qa 40개 중 20개는 answer_assertions를 갖고(그 20개 안에서 4개
    필수 intent가 정확히 5개씩 분포), 30개는 relevant_sources를 갖는다."""
    cases: list[dict] = []
    intents = ["explanation", "comparison", "procedure", "yesno"]
    for i in range(40):
        overrides: dict = {"expected_intent": intents[i % 4]}
        if i < 30:
            overrides["relevant_sources"] = [f"doc{i}.pdf"]
        if i < 20:
            overrides["answer_assertions"] = [{"any_of": ["핵심답변"]}]
        cases.append(_case(f"dq{i}", question=f"문서 관련 질문입니다 {i}", **overrides))
    for i in range(10):
        cases.append(
            _case(
                f"ws{i}",
                question=f"웹 검색 질문입니다 {i}",
                category="web_search",
                route="web_search",
                expected_intent="other",
            )
        )
    for i in range(5):
        overrides = {"expected_intent": "uncertain"}
        if i < 3:
            overrides["expect_abstention"] = True
        cases.append(
            _case(f"bd{i}", question=f"경계 사례 질문입니다 {i}", category="boundary", **overrides)
        )
    for i in range(5):
        overrides = {"expected_intent": "uncertain"}
        if i < 2:
            overrides["expect_abstention"] = True
        cases.append(
            _case(
                f"ua{i}", question=f"답변불가 질문입니다 {i}", category="unanswerable", **overrides
            )
        )
    return cases


class TestLoadJsonl:
    def test_valid_file_parses_all_lines(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        _write_jsonl(path, [_case("a"), _case("b")])
        cases = load_jsonl(path)
        assert [c.id for c in cases] == ["a", "b"]

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        path.write_text(
            json.dumps(_case("a"), ensure_ascii=False)
            + "\n\n\n"
            + json.dumps(_case("b"), ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        cases = load_jsonl(path)
        assert len(cases) == 2

    def test_malformed_json_raises_with_line_number(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(DatasetError) as exc_info:
            load_jsonl(path)
        assert "line 1" in str(exc_info.value)

    def test_schema_violation_raises_with_case_id_and_line_number(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        _write_jsonl(path, [_case("bad", question="")])
        with pytest.raises(DatasetError) as exc_info:
            load_jsonl(path)
        message = str(exc_info.value)
        assert "line 1" in message
        assert "id=bad" in message

    def test_duplicate_id_raises_referencing_first_occurrence_line(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        _write_jsonl(path, [_case("dup", question="질문1"), _case("dup", question="질문2")])
        with pytest.raises(DatasetError) as exc_info:
            load_jsonl(path)
        message = str(exc_info.value)
        assert "line 2" in message
        assert "line 1" in message

    def test_missing_file_raises_dataset_error(self, tmp_path):
        with pytest.raises(DatasetError):
            load_jsonl(tmp_path / "nope.jsonl")

    def test_non_utf8_file_raises_dataset_error(self, tmp_path):
        path = tmp_path / "bad_encoding.jsonl"
        path.write_bytes(b"\xff\xfe not utf8 \x00\x01")
        with pytest.raises(DatasetError) as exc_info:
            load_jsonl(path)
        assert "UTF-8" in str(exc_info.value)

    def test_directory_path_raises_dataset_error(self, tmp_path):
        directory = tmp_path / "adir"
        directory.mkdir()
        with pytest.raises(DatasetError):
            load_jsonl(directory)


class TestValidateComposition:
    def test_minimum_valid_dataset_has_no_errors(self):
        report = validate_composition(_to_cases(_minimal_valid_cases()))
        assert report.is_valid
        assert report.errors == []
        assert report.answer_eval_case_count == 25
        assert report.answer_eval_cases_without_intent == 0

    def test_total_below_minimum_flagged(self):
        cases_data = [c for c in _minimal_valid_cases() if c["id"] != "dq35"]
        report = validate_composition(_to_cases(cases_data))
        assert not report.is_valid
        assert any("전체 사례 수가" in e for e in report.errors)

    def test_document_qa_below_minimum_flagged(self):
        cases_data = [c for c in _minimal_valid_cases() if c["id"] != "dq35"]
        report = validate_composition(_to_cases(cases_data))
        assert not report.is_valid
        assert any("document_qa 사례가" in e for e in report.errors)

    def test_web_search_below_minimum_flagged(self):
        cases_data = [c for c in _minimal_valid_cases() if c["id"] != "ws9"]
        report = validate_composition(_to_cases(cases_data))
        assert not report.is_valid
        assert any("web_search 사례가" in e for e in report.errors)

    def test_boundary_plus_unanswerable_below_minimum_flagged(self):
        cases_data = [c for c in _minimal_valid_cases() if c["id"] != "bd4"]
        report = validate_composition(_to_cases(cases_data))
        assert not report.is_valid
        assert any("boundary+unanswerable 사례가" in e for e in report.errors)

    def test_document_qa_with_sources_below_minimum_flagged(self):
        cases_data = copy.deepcopy(_minimal_valid_cases())
        for c in cases_data:
            if c["id"] == "dq25":
                c.pop("relevant_sources", None)
        report = validate_composition(_to_cases(cases_data))
        assert not report.is_valid
        assert any("relevant_sources가 있는 document_qa 사례가" in e for e in report.errors)

    def test_document_qa_with_assertions_below_minimum_flagged(self):
        cases_data = copy.deepcopy(_minimal_valid_cases())
        for c in cases_data:
            if c["id"] == "dq15":
                c.pop("answer_assertions", None)
        report = validate_composition(_to_cases(cases_data))
        assert not report.is_valid
        assert any("answer_assertions가 있는 document_qa 사례가" in e for e in report.errors)

    def test_expect_abstention_below_minimum_flagged(self):
        cases_data = copy.deepcopy(_minimal_valid_cases())
        for c in cases_data:
            if c["id"] == "bd0":
                c["expect_abstention"] = False
        report = validate_composition(_to_cases(cases_data))
        assert not report.is_valid
        assert any("expect_abstention=true 사례가" in e for e in report.errors)

    def test_korean_ratio_below_minimum_flagged(self):
        cases_data = copy.deepcopy(_minimal_valid_cases())
        english_target_ids = {f"ws{i}" for i in range(10)} | {"bd0", "bd1", "bd2"}
        for c in cases_data:
            if c["id"] in english_target_ids:
                c["question"] = "This is a test question written entirely in English."
        report = validate_composition(_to_cases(cases_data))
        assert not report.is_valid
        assert any("한국어 질문 비율이" in e for e in report.errors)

    @pytest.mark.parametrize("intent", ["explanation", "comparison", "procedure", "yesno"])
    def test_each_required_intent_below_minimum_flagged(self, intent):
        intents_order = ["explanation", "comparison", "procedure", "yesno"]
        idx = intents_order.index(intent)
        target_id = f"dq{idx}"
        cases_data = copy.deepcopy(_minimal_valid_cases())
        for c in cases_data:
            if c["id"] == target_id:
                c.pop("answer_assertions", None)
        report = validate_composition(_to_cases(cases_data))
        assert not report.is_valid
        assert any(f"intent={intent} 사례가" in e for e in report.errors)

    def test_multiple_violations_all_reported_together(self):
        cases_data = _minimal_valid_cases()[:5]
        report = validate_composition(_to_cases(cases_data))
        assert not report.is_valid
        assert len(report.errors) > 1

    def test_intent_only_on_non_answer_eval_cases_is_flagged(self):
        """design_review.md 2차 P1의 핵심 회귀 테스트: web_search/Retrieval
        전용 사례에만 각 intent를 5개씩 배치하면(answer_assertions도
        expect_abstention도 없음) 전체 개수는 충족해 보여도 구성 검증이
        반드시 실패해야 한다 — intent_counts가 Answer 평가 대상만 센다."""
        cases: list[dict] = []
        intents = ["explanation", "comparison", "procedure", "yesno"]
        for i in range(40):
            overrides: dict = {"relevant_sources": [f"doc{i}.pdf"]}
            if i < 20:
                overrides["expected_intent"] = intents[i % 4]
            cases.append(_case(f"dq{i}", question=f"문서 관련 질문입니다 {i}", **overrides))
        for i in range(10):
            cases.append(
                _case(
                    f"ws{i}",
                    question=f"웹 검색 질문입니다 {i}",
                    category="web_search",
                    route="web_search",
                )
            )
        for i in range(5):
            cases.append(
                _case(
                    f"bd{i}",
                    question=f"경계 사례 질문입니다 {i}",
                    category="boundary",
                    expect_abstention=(i < 5),
                )
            )
        for i in range(5):
            cases.append(
                _case(
                    f"ua{i}",
                    question=f"답변불가 질문입니다 {i}",
                    category="unanswerable",
                    expect_abstention=(i < 5),
                )
            )
        report = validate_composition(_to_cases(cases))
        assert not report.is_valid
        for intent in ["explanation", "comparison", "procedure", "yesno"]:
            assert any(f"intent={intent} 사례가 0개" in e for e in report.errors)

    def test_answer_eval_case_count_and_without_intent_reported(self):
        """expected_intent=None인 Answer 평가 대상 사례가 있으면 구성
        검증에서 거부하지 않되 answer_eval_cases_without_intent로
        별도 집계돼야 한다."""
        cases_data = _minimal_valid_cases()
        cases_data.append(
            _case(
                "dq-extra",
                question="추가 답변 평가 사례입니다",
                answer_assertions=[{"any_of": ["추가"]}],
            )
        )
        report = validate_composition(_to_cases(cases_data))
        assert report.is_valid
        assert report.answer_eval_case_count == 26
        assert report.answer_eval_cases_without_intent == 1


class TestCli:
    def test_validate_valid_dataset_returns_zero(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        _write_jsonl(path, _minimal_valid_cases())
        assert main(["validate", str(path)]) == 0

    def test_validate_invalid_schema_returns_one_with_stderr_detail(self, tmp_path, capsys):
        path = tmp_path / "golden.jsonl"
        _write_jsonl(path, [_case("bad", question="")])
        assert main(["validate", str(path)]) == 1
        captured = capsys.readouterr()
        assert "오류" in captured.err
        assert "id=bad" in captured.err
        assert "golden.jsonl에서 수정한 뒤" in captured.err
        assert captured.out == ""

    def test_validate_missing_file_shows_io_advice_not_content_advice(self, tmp_path, capsys):
        """M2_Phase1_code_review_result.md P3: 파일 없음은 kind="io" 오류이므로
        "줄/사례 수정" 안내가 아니라 경로/권한/인코딩 확인 안내가 나와야 하고,
        report가 계산되지 않으므로 stdout에는 아무것도 찍히면 안 된다."""
        assert main(["validate", str(tmp_path / "nope.jsonl")]) == 1
        captured = capsys.readouterr()
        assert "찾을 수 없습니다" in captured.err
        assert "경로가 올바른지" in captured.err
        assert "golden.jsonl에서 수정한 뒤" not in captured.err
        assert captured.out == ""

    def test_validate_directory_path_shows_io_advice(self, tmp_path, capsys):
        directory = tmp_path / "adir"
        directory.mkdir()
        assert main(["validate", str(directory)]) == 1
        captured = capsys.readouterr()
        assert "경로가 올바른지" in captured.err
        assert "golden.jsonl에서 수정한 뒤" not in captured.err

    def test_validate_prints_valid_json_report_to_stdout(self, tmp_path, capsys):
        """stdout 전체를 json.loads()로 되읽어 유효한 JSON인지 확인."""
        path = tmp_path / "golden.jsonl"
        _write_jsonl(path, _minimal_valid_cases())
        main(["validate", str(path)])
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert report["valid"] is True

    def test_no_argv_prints_usage_and_returns_nonzero(self, capsys):
        """design_review.md 1차 P3: main([])이 SystemExit을 던지지 않고 정수
        2를 반환하는지 직접 assert로 확인한다. M2_Phase1_code_review_result.md
        P3: 반환값이 정확히 계약값(2)인지, stderr에 usage가 찍히는지도 확인."""
        assert main([]) == 2
        captured = capsys.readouterr()
        assert "usage" in captured.err

    def test_invalid_subcommand_raises_systemexit(self):
        """argparse 자체의 choice 검증은 여전히 SystemExit(2)를 던진다.
        M2_Phase1_code_review_result.md P3: 코드 값 2까지 정확히 확인한다."""
        with pytest.raises(SystemExit) as exc_info:
            main(["bogus"])
        assert exc_info.value.code == 2
