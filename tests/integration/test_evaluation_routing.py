"""evaluation/routing.py 단위 테스트 (M2 Phase 5, M2-REQ-007).

이 파일의 테스트는 fake decide_tool() callable만 주입해 evaluate_routing()/
main()의 오프라인 경로를 검증한다. data/, vectorstore/, Ollama, 네트워크를 전혀
사용하지 않는다 — live 전용 opt-in 검증(TestLiveOptIn)만 예외적으로 subprocess로
`python -m evaluation.routing --mode live`를 실행하지만, RUN_LIVE_LLM_TESTS=1이
없으면 즉시 종료되므로 여전히 네트워크나 모델을 호출하지 않는다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.dataset import load_jsonl
from evaluation.routing import (
    _offline_mock_decide_tool,
    render_routing_markdown,
    evaluate_routing,
    main,
)
from evaluation.schema import GoldenCase

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DATASET_PATH = REPO_ROOT / "evaluation" / "datasets" / "golden.jsonl"
TEST_AGENT_ROUTING_PATH = REPO_ROOT / "tests" / "integration" / "test_agent_routing.py"


def _make_case(id_: str, question: str, expected_route: str, tags=None) -> GoldenCase:
    return GoldenCase(
        id=id_,
        question=question,
        category=expected_route,
        expected_route=expected_route,
        tags=list(tags) if tags is not None else ["test"],
    )


def _write_dataset(tmp_path: Path, cases: list[dict]) -> Path:
    path = tmp_path / "golden.jsonl"
    lines = [json.dumps(case, ensure_ascii=False) for case in cases]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _case_dict(id_: str, question: str, route: str, tags=None) -> dict:
    return {
        "id": id_,
        "question": question,
        "category": route,
        "expected_route": route,
        "tags": list(tags) if tags is not None else ["test"],
    }


class TestEvaluateRoutingCore:
    def test_perfect_prediction_yields_accuracy_one(self):
        cases = [
            _make_case("a", "질문 1", "document_qa"),
            _make_case("b", "질문 2", "web_search"),
            _make_case("c", "질문 3", "document_qa"),
        ]

        def decide(question: str):
            expected = {"질문 1": "document_qa", "질문 2": "web_search", "질문 3": "document_qa"}
            return expected[question], question

        result = evaluate_routing(cases, decide, measure_latency=False)

        assert result["total_cases"] == 3
        assert result["correct_count"] == 3
        assert result["accuracy"] == 1.0
        assert result["failures"] == []
        assert result["exception_count"] == 0
        assert result["no_tool_count"] == 0
        assert result["unknown_route_count"] == 0
        assert result["excluded_count"] == 0

        pr = result["precision_recall_f1"]
        assert pr["document_qa"]["precision"] == 1.0
        assert pr["document_qa"]["recall"] == 1.0
        assert pr["document_qa"]["f1"] == 1.0
        assert pr["web_search"]["precision"] == 1.0
        assert pr["web_search"]["recall"] == 1.0
        assert pr["web_search"]["f1"] == 1.0
        assert pr["confusion_matrix"]["document_qa"]["document_qa"] == 2
        assert pr["confusion_matrix"]["web_search"]["web_search"] == 1

    def test_partial_misclassification_computes_accuracy_pr_f1_confusion(self):
        # y_true = [dq, dq, ws, ws]; y_pred = [dq, ws, ws, ws]
        cases = [
            _make_case("dq1", "dq1", "document_qa"),
            _make_case("dq2", "dq2", "document_qa"),
            _make_case("ws1", "ws1", "web_search"),
            _make_case("ws2", "ws2", "web_search"),
        ]
        predictions = {
            "dq1": "document_qa",
            "dq2": "web_search",  # 오분류
            "ws1": "web_search",
            "ws2": "web_search",
        }

        def decide(question: str):
            return predictions[question], question

        result = evaluate_routing(cases, decide, measure_latency=False)

        assert result["total_cases"] == 4
        assert result["correct_count"] == 3
        assert result["accuracy"] == pytest.approx(0.75)
        assert len(result["failures"]) == 1
        assert result["failures"][0]["id"] == "dq2"
        assert result["failures"][0]["failure_type"] == "mismatch"
        assert result["failures"][0]["expected_route"] == "document_qa"
        assert result["failures"][0]["actual_route"] == "web_search"

        pr = result["precision_recall_f1"]
        assert pr["document_qa"]["precision"] == pytest.approx(1.0)  # tp=1, fp=0
        assert pr["document_qa"]["recall"] == pytest.approx(0.5)  # tp=1, fn=1
        assert pr["document_qa"]["f1"] == pytest.approx(2 / 3)
        assert pr["web_search"]["precision"] == pytest.approx(2 / 3)  # tp=2, fp=1
        assert pr["web_search"]["recall"] == pytest.approx(1.0)  # tp=2, fn=0
        assert pr["confusion_matrix"]["document_qa"]["web_search"] == 1

    def test_all_predictions_single_route(self):
        """모든 예측이 한 route로 쏠린 퇴화된 분포도 0-division 없이 계산된다."""
        cases = [
            _make_case("a", "a", "document_qa"),
            _make_case("b", "b", "web_search"),
            _make_case("c", "c", "document_qa"),
        ]

        def decide(question: str):
            return "document_qa", question  # 항상 document_qa만 예측

        result = evaluate_routing(cases, decide, measure_latency=False)

        pr = result["precision_recall_f1"]
        assert pr["web_search"]["precision"] == 0.0
        assert pr["web_search"]["recall"] == 0.0
        assert pr["web_search"]["f1"] == 0.0
        assert pr["document_qa"]["recall"] == 1.0  # 두 dq 사례 모두 dq로 예측됨(tp=2,fn=0)
        assert pr["document_qa"]["precision"] == pytest.approx(2 / 3)  # tp=2, fp=1(ws 오분류)

    def test_no_tool_and_exception_are_distinguished_and_evaluation_continues(self):
        cases = [
            _make_case("ok", "ok", "document_qa"),
            _make_case("boom", "boom", "web_search"),
            _make_case("wrong", "wrong", "web_search"),
            _make_case("none", "none", "document_qa"),
        ]

        def decide(question: str):
            if question == "ok":
                return "document_qa", question
            if question == "boom":
                raise RuntimeError("모델 호출 실패")
            if question == "wrong":
                return "document_qa", question  # 기대는 web_search
            if question == "none":
                return None, None
            raise AssertionError("unreachable")

        result = evaluate_routing(cases, decide, measure_latency=False)

        assert result["total_cases"] == 4
        assert result["exception_count"] == 1
        assert result["no_tool_count"] == 1
        assert result["failure_count"] == 1  # failure_count == exception_count
        assert result["success_count"] == 3  # total - exception_count
        assert result["excluded_count"] == 1  # no_tool_count + unknown_route_count
        assert result["correct_count"] == 1  # "ok"만 정답

        failure_types = {f["id"]: f["failure_type"] for f in result["failures"]}
        assert failure_types["boom"] == "exception"
        assert failure_types["none"] == "no_tool"
        assert failure_types["wrong"] == "mismatch"
        boom_failure = next(f for f in result["failures"] if f["id"] == "boom")
        assert boom_failure["error"] is not None
        assert "RuntimeError" in boom_failure["error"]

    def test_unknown_route_is_recorded_as_explicit_confusion_column(self):
        """M2_Phase_4_5_6_code_review_result.md P1: unknown_route는 confusion
        matrix에서 완전히 빠지는 게 아니라 명시적 예측 열로 남고, 기대 클래스의
        false negative(recall 감소)로 반영돼야 한다. decide_tool()이 실제로
        반환한 임의의 문자열("not_a_real_tool")이 아니라 고정 카테고리
        "unknown_route"가 predicted 값으로 쓰인다."""
        cases = [_make_case("a", "a", "document_qa")]

        def decide(question: str):
            return "not_a_real_tool", question

        result = evaluate_routing(cases, decide, measure_latency=False)

        assert result["unknown_route_count"] == 1
        assert result["excluded_count"] == 1
        assert result["correct_count"] == 0
        assert result["failures"][0]["failure_type"] == "unknown_route"
        assert result["failures"][0]["actual_route"] == "not_a_real_tool"
        # KeyError 없이 통과해야 하며(labels 집합은 pseudo-label로 고정돼 있음),
        # unknown_route는 실제 반환값이 아니라 정규화된 카테고리 이름으로 기록된다.
        pr = result["precision_recall_f1"]
        assert pr["document_qa"]["precision"] == 0.0
        assert pr["document_qa"]["recall"] == 0.0  # tp=0, fn=1(unknown_route로 실패)
        assert pr["confusion_matrix"]["document_qa"]["unknown_route"] == 1
        assert "not_a_real_tool" not in pr["confusion_matrix"]["document_qa"]

    def test_failure_types_each_reduce_recall_as_false_negative(self):
        """M2_Phase_4_5_6_code_review_result.md P1 권고: 정상 1건 + no-tool 1건,
        정상 1건 + exception 1건, unknown-route 사례를 각각 포함해 recall 감소를
        검증한다. 세 유형 모두 기대 클래스 recall의 분모(false negative)에
        반영돼야 한다."""

        def make_decide(no_tool_q, exception_q, unknown_q):
            def decide(question: str):
                if question == no_tool_q:
                    return None, None
                if question == exception_q:
                    raise RuntimeError("boom")
                if question == unknown_q:
                    return "not_a_real_tool", question
                return "document_qa", question

            return decide

        cases = [
            _make_case("ok1", "ok1", "document_qa"),
            _make_case("no_tool", "no_tool", "document_qa"),
            _make_case("exc", "exc", "document_qa"),
            _make_case("unk", "unk", "document_qa"),
        ]
        result = evaluate_routing(
            cases, make_decide("no_tool", "exc", "unk"), measure_latency=False
        )

        pr = result["precision_recall_f1"]
        # tp=1("ok1"), fn=3(no_tool/exception/unknown_route 각각 1건) -> recall = 1/4
        assert pr["document_qa"]["recall"] == pytest.approx(0.25)
        assert pr["confusion_matrix"]["document_qa"]["no_tool"] == 1
        assert pr["confusion_matrix"]["document_qa"]["exception"] == 1
        assert pr["confusion_matrix"]["document_qa"]["unknown_route"] == 1
        assert pr["confusion_matrix"]["document_qa"]["document_qa"] == 1

    def test_confusion_matrix_row_sums_match_expected_route_counts(self):
        """불변식: accuracy == correct/total이고, confusion matrix의 각 실제
        클래스 행의 합은 그 클래스가 expected_route인 전체 사례 수와 같아야
        한다(실패 사례를 포함해서) — M2_Phase_4_5_6_code_review_result.md P1."""
        cases = [
            _make_case("dq_ok", "dq_ok", "document_qa"),
            _make_case("dq_no_tool", "dq_no_tool", "document_qa"),
            _make_case("ws_ok", "ws_ok", "web_search"),
            _make_case("ws_exc", "ws_exc", "web_search"),
        ]

        def decide(question: str):
            if question == "dq_no_tool":
                return None, None
            if question == "ws_exc":
                raise RuntimeError("boom")
            if question == "dq_ok":
                return "document_qa", question
            if question == "ws_ok":
                return "web_search", question
            raise AssertionError("unreachable")

        result = evaluate_routing(cases, decide, measure_latency=False)

        assert result["accuracy"] == result["correct_count"] / result["total_cases"]
        confusion = result["precision_recall_f1"]["confusion_matrix"]
        dq_row_sum = sum(confusion["document_qa"].values())
        ws_row_sum = sum(confusion["web_search"].values())
        assert dq_row_sum == 2  # dq_ok + dq_no_tool
        assert ws_row_sum == 2  # ws_ok + ws_exc

    def test_empty_cases_raises_explicit_error_not_treated_as_success(self):
        with pytest.raises(ValueError):
            evaluate_routing([], _offline_mock_decide_tool)

    def test_latency_disabled_is_distinct_from_zero_ms(self):
        cases = [_make_case("a", "a", "document_qa")]

        def decide(question: str):
            return "document_qa", question

        result = evaluate_routing(cases, decide, measure_latency=False)
        latency = result["latency_ms"]
        assert latency["measured"] is False
        assert latency["mean"] is None
        assert latency["median"] is None
        assert latency["p95"] is None

    def test_latency_enabled_produces_non_none_values(self):
        cases = [_make_case("a", "a", "document_qa"), _make_case("b", "b", "web_search")]

        def decide(question: str):
            return ("document_qa" if question == "a" else "web_search"), question

        result = evaluate_routing(cases, decide, measure_latency=True)
        latency = result["latency_ms"]
        assert latency["measured"] is True
        assert latency["mean"] is not None
        assert latency["median"] is not None
        assert latency["p95"] is not None
        assert latency["mean"] >= 0.0


class TestCliOfflineMode:
    def test_offline_run_produces_report_without_data_or_vectorstore(self, tmp_path, monkeypatch):
        """offline 모드는 data/, vectorstore/, Ollama, 네트워크 없이 동작해야 한다."""
        # 실수로 실제 data/vectorstore 경로에 의존하면 이 테스트 디렉터리 안에서는
        # 실패해야 하므로, 작업 디렉터리를 격리된 tmp_path로 옮겨 확인한다.
        monkeypatch.chdir(tmp_path)
        dataset_path = _write_dataset(
            tmp_path,
            [
                _case_dict("a", "질문 1", "document_qa", tags=["t"]),
                _case_dict("b", "질문 2", "web_search", tags=["t"]),
            ],
        )
        output_dir = tmp_path / "reports"

        exit_code = main(
            [
                "--dataset",
                str(dataset_path),
                "--output",
                str(output_dir),
                "--mode",
                "offline",
            ]
        )

        assert exit_code == 0
        json_files = sorted(output_dir.glob("routing_*.json"))
        assert len(json_files) == 1

    def test_offline_report_has_null_reproducibility_fields_with_reason(self, tmp_path):
        dataset_path = _write_dataset(tmp_path, [_case_dict("a", "질문", "document_qa")])
        output_dir = tmp_path / "reports"

        exit_code = main(
            ["--dataset", str(dataset_path), "--output", str(output_dir), "--mode", "offline"]
        )
        assert exit_code == 0

        json_path = next(output_dir.glob("routing_*.json"))
        payload = json.loads(json_path.read_text(encoding="utf-8"))

        assert payload["corpus_manifest"] is None
        assert payload["corpus_manifest_sha256"] is None
        assert payload["vectorstore_fingerprint"] is None
        assert payload["reproducibility_note"]  # non-empty 문자열

    def test_markdown_report_shows_metrics_confusion_and_failures_not_just_json_pointer(
        self, tmp_path
    ):
        """M2_Phase_4_5_6_code_review_result.md P1: Markdown 리포트는 accuracy,
        클래스별 PR/F1, confusion matrix, latency, 실패 사례를 사람이 직접 읽을
        수 있어야 한다 — "같은 이름의 .json을 참고하라"는 안내만 있으면 안 된다.
        offline mock은 항상 document_qa를 반환하므로 web_search 사례("b")는
        반드시 mismatch 실패로 기록된다."""
        dataset_path = _write_dataset(
            tmp_path,
            [
                _case_dict("a", "질문 1", "document_qa", tags=["t"]),
                _case_dict("b", "질문 2", "web_search", tags=["t"]),
            ],
        )
        output_dir = tmp_path / "reports"

        exit_code = main(
            ["--dataset", str(dataset_path), "--output", str(output_dir), "--mode", "offline"]
        )
        assert exit_code == 0

        md_path = next(output_dir.glob("routing_*.md"))
        text = md_path.read_text(encoding="utf-8")

        assert "50.0%" in text  # accuracy = 1/2
        assert "document_qa" in text and "web_search" in text
        assert "confusion" in text.lower()
        assert "b" in text  # 실패 사례 id
        assert "질문 2" in text  # 실패 사례 질문
        assert "mismatch" in text

    def test_markdown_failure_row_with_pipe_backslash_and_newline_keeps_column_count(self):
        """M2_Phase_4_5_6_code_review_result.md 재리뷰 P2: 질문이나 오류 메시지에
        pipe/backslash/줄바꿈이 들어가도 실패 사례 표의 열 개수(6개)가 유지돼야
        한다 — 이스케이프하지 않으면 pipe가 열을 늘려 표 구조가 깨진다."""
        payload = {
            "generated_at_utc": "2026-01-01T00:00:00Z",
            "mode": "offline",
            "tag_filter": None,
            "limit": None,
            "total_cases": 1,
            "correct_count": 0,
            "accuracy": 0.0,
            "exception_count": 0,
            "no_tool_count": 0,
            "unknown_route_count": 1,
            "precision_recall_f1": {
                "document_qa": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "web_search": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "confusion_matrix": {
                    "document_qa": {"document_qa": 0, "web_search": 0, "no_tool": 0, "unknown_route": 1, "exception": 0},
                    "web_search": {"document_qa": 0, "web_search": 0, "no_tool": 0, "unknown_route": 0, "exception": 0},
                },
            },
            "latency_ms": {"measured": False, "mean": None, "median": None, "p95": None},
            "failures": [
                {
                    "id": "a|b",
                    "question": "질문 | 파이프\n줄바꿈\\백슬래시",
                    "expected_route": "document_qa",
                    "actual_route": "weird|tool\\name",
                    "failure_type": "unknown_route",
                    "error": None,
                }
            ],
        }
        text = render_routing_markdown(payload)
        failure_row = next(line for line in text.splitlines() if line.startswith("| a"))
        # 표 구조상 delimiter 7개(6열 경계) + id/question/actual_route에 각각 1개씩
        # 들어있던 원본 pipe가 이스케이프돼 총 10개여야 한다 — 이스케이프하지 않으면
        # 열이 늘어나 표가 깨진다.
        assert failure_row.count("|") == 10
        assert "\n" not in failure_row.strip()
        assert "weird\\|tool\\\\name" in failure_row

    def test_tag_filter_matching_zero_cases_is_explicit_error_not_success(self, tmp_path, capsys):
        dataset_path = _write_dataset(
            tmp_path, [_case_dict("a", "질문", "document_qa", tags=["only-this-tag"])]
        )
        output_dir = tmp_path / "reports"

        exit_code = main(
            [
                "--dataset",
                str(dataset_path),
                "--output",
                str(output_dir),
                "--mode",
                "offline",
                "--tag",
                "no-such-tag",
            ]
        )

        assert exit_code != 0
        captured = capsys.readouterr()
        assert "없습니다" in captured.err
        assert not output_dir.exists()

    def test_empty_dataset_file_is_explicit_error_not_success(self, tmp_path, capsys):
        dataset_path = tmp_path / "golden.jsonl"
        dataset_path.write_text("", encoding="utf-8")
        output_dir = tmp_path / "reports"

        exit_code = main(
            ["--dataset", str(dataset_path), "--output", str(output_dir), "--mode", "offline"]
        )

        assert exit_code != 0
        captured = capsys.readouterr()
        assert captured.err.strip() != ""

    def test_limit_and_tag_are_applied_deterministically(self, tmp_path):
        dataset_path = _write_dataset(
            tmp_path,
            [
                _case_dict("a", "a", "document_qa", tags=["keep"]),
                _case_dict("b", "b", "web_search", tags=["drop"]),
                _case_dict("c", "c", "document_qa", tags=["keep"]),
                _case_dict("d", "d", "web_search", tags=["keep"]),
            ],
        )
        output_dir = tmp_path / "reports"

        exit_code = main(
            [
                "--dataset",
                str(dataset_path),
                "--output",
                str(output_dir),
                "--mode",
                "offline",
                "--tag",
                "keep",
                "--limit",
                "2",
            ]
        )
        assert exit_code == 0
        json_path = next(output_dir.glob("routing_*.json"))
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        # tag="keep" 필터 후 [a, c, d] 중 원본 순서대로 limit=2 -> [a, c]
        assert payload["total_cases"] == 2

    @pytest.mark.parametrize("bad_limit", ["0", "-1"])
    def test_non_positive_limit_is_parser_error_not_negative_slice(self, tmp_path, capsys, bad_limit):
        """M2_Phase_4_5_6_code_review_result.md P3: --limit -1은 cases[:-1](마지막
        하나 제외)로 조용히 재해석되면 안 되고 argparse 오류(exit 2)여야 한다."""
        dataset_path = _write_dataset(tmp_path, [_case_dict("a", "질문", "document_qa")])
        output_dir = tmp_path / "reports"

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--dataset",
                    str(dataset_path),
                    "--output",
                    str(output_dir),
                    "--mode",
                    "offline",
                    "--limit",
                    bad_limit,
                ]
            )
        assert exc_info.value.code == 2
        assert not output_dir.exists()


class TestHelpDoesNotLoadModel:
    def test_help_exits_zero_without_reaching_mode_logic(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_missing_required_args_is_argparse_error_not_model_load(self):
        with pytest.raises(SystemExit):
            main([])


class TestLiveOptIn:
    def test_live_without_opt_in_exits_nonzero_before_model_call(self, tmp_path):
        dataset_path = _write_dataset(tmp_path, [_case_dict("a", "질문", "document_qa")])
        output_dir = tmp_path / "reports"

        env = os.environ.copy()
        env.pop("RUN_LIVE_LLM_TESTS", None)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.routing",
                "--dataset",
                str(dataset_path),
                "--output",
                str(output_dir),
                "--mode",
                "live",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 2
        assert "RUN_LIVE_LLM_TESTS" in result.stderr
        assert not output_dir.exists()

    def test_live_mode_check_precedes_dataset_load(self, tmp_path):
        """dataset 경로가 아예 존재하지 않아도 opt-in 부재가 먼저 걸려야 한다
        (모델 import/호출 전에 종료해야 한다는 계약이 dataset 유무보다 우선한다)."""
        env = os.environ.copy()
        env.pop("RUN_LIVE_LLM_TESTS", None)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.routing",
                "--dataset",
                str(tmp_path / "does-not-exist.jsonl"),
                "--output",
                str(tmp_path / "reports"),
                "--mode",
                "live",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 2
        assert "찾을 수 없습니다" not in result.stderr


class TestRoutingRegressionMigration:
    """기존 test_agent_routing.py의 ROUTING_CASES 16쌍이 golden.jsonl로 이관됐는지
    구조적으로만 검증한다(§6.4, §6.5).

    이전에는 이 클래스에 이관 전 16쌍(질문, 기대 route)을 다시 하드코딩한
    ORIGINAL_ROUTING_CASES 상수를 두고 golden.jsonl과 완전 일치를 대조했다.
    그 목록 자체가 골든셋과 별개인 두 번째 정답 원천이 되어, 질문 문구를 바꿀 때
    두 곳을 함께 고쳐야 하는 문제가 있었다(M2_Phase_4_5_6_code_review_result.md
    P2). 이관 시점의 원본 16쌍 대조 증적은 이관 커밋(c6a65b4, "routing evaluator
    및 기존 사례 통합")과 M2_Phase_4_5_6_code_review_result.md에 남아 있으므로,
    이후 회귀 테스트는 골든셋 자체의 구조적 불변식(개수/ID 유일성/route 유효성)만
    지속적으로 검증한다."""

    def test_exactly_sixteen_cases_tagged(self):
        cases = load_jsonl(GOLDEN_DATASET_PATH)
        tagged = [c for c in cases if "routing_regression" in c.tags]
        assert len(tagged) == 16

    def test_tagged_case_ids_are_unique(self):
        cases = load_jsonl(GOLDEN_DATASET_PATH)
        tagged = [c for c in cases if "routing_regression" in c.tags]
        ids = [c.id for c in tagged]
        assert len(ids) == len(set(ids))

    def test_tagged_cases_have_non_empty_question_and_valid_route(self):
        cases = load_jsonl(GOLDEN_DATASET_PATH)
        tagged = [c for c in cases if "routing_regression" in c.tags]
        for case in tagged:
            assert case.question.strip()
            assert case.expected_route.value in {"document_qa", "web_search"}

    def test_no_duplicate_answer_list_left_in_test_agent_routing(self):
        text = TEST_AGENT_ROUTING_PATH.read_text(encoding="utf-8")
        assert "ROUTING_CASES" not in text

    def test_test_agent_routing_reads_from_golden_dataset(self):
        text = TEST_AGENT_ROUTING_PATH.read_text(encoding="utf-8")
        assert "routing_regression" in text
        assert "golden.jsonl" in text
