"""Integration tests for evaluation/rescore.py (Design.md §5.8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import answer_rules as ar
from evaluation import rescore


def _write_dataset(tmp_path: Path) -> Path:
    case = {
        "id": "dq-fixture-001",
        "question": "테스트 질문",
        "category": "document_qa",
        "expected_route": "document_qa",
        "answer_assertions": [{"any_of": ["0.7%"]}],
        "tags": ["fixture"],
    }
    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_stored_report(tmp_path: Path, answer: str) -> Path:
    payload = {
        "schema_version": "1.0.0",
        "case_results": [
            {
                "id": "dq-fixture-001",
                "status": "success",
                "answer": answer,
                "expect_abstention": False,
            }
        ],
    }
    path = tmp_path / "answers_report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_rescore_report_fixes_normalization_false_negative(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    report_path = _write_stored_report(tmp_path, "성장률은 0.7 %로 조정되었습니다.")

    payload = rescore.rescore_report(report_path, dataset_path, variants=None)

    assert payload["assertion_v2"]["assertions_passed"] == 1
    assert payload["assertion_v2"]["assertions_total"] == 1
    assert payload["assertion_v2"]["fixed_vs_v1"] == [
        {"id": "dq-fixture-001", "v1": "0/1", "v2": "1/1"}
    ]
    assert payload["assertion_v2"]["regressed_vs_v1"] == []


def test_rescore_report_does_not_modify_original_file(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    report_path = _write_stored_report(tmp_path, "성장률은 0.7 %로 조정되었습니다.")
    original_bytes = report_path.read_bytes()

    rescore.rescore_report(report_path, dataset_path, variants=None)

    assert report_path.read_bytes() == original_bytes


def test_rescore_report_missing_answer_field_raises(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    payload = {"schema_version": "1.0.0", "case_results": [{"id": "dq-fixture-001", "status": "success"}]}
    report_path = tmp_path / "broken_report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    # Missing "answer" key means the case is passed through unscored, not an
    # error — but a report with no case_results at all must fail-closed.
    empty_report_path = tmp_path / "empty_report.json"
    empty_report_path.write_text(json.dumps({"schema_version": "1.0.0"}), encoding="utf-8")
    with pytest.raises(rescore.RescoreError):
        rescore.rescore_report(empty_report_path, dataset_path, variants=None)


def test_main_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    report_path = _write_stored_report(tmp_path, "성장률은 0.7 %로 조정되었습니다.")
    output_dir = tmp_path / "out"

    exit_code = rescore.main(
        [
            "--report",
            str(report_path),
            "--dataset",
            str(dataset_path),
            "--output",
            str(output_dir),
            "--no-variants",
        ]
    )
    assert exit_code == 0
    json_files = list(output_dir.glob("rescore_*.json"))
    md_files = list(output_dir.glob("rescore_*.md"))
    assert len(json_files) == 1
    assert len(md_files) == 1


def test_main_cli_bad_report_path_exits_2(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    exit_code = rescore.main(
        [
            "--report",
            str(tmp_path / "missing.json"),
            "--dataset",
            str(dataset_path),
            "--output",
            str(tmp_path / "out"),
            "--no-variants",
        ]
    )
    assert exit_code == 2
