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
from evaluation.routing import _offline_mock_decide_tool, evaluate_routing, main
from evaluation.schema import GoldenCase

REPO_ROOT = Path(__file__).resolve().parent
GOLDEN_DATASET_PATH = REPO_ROOT / "evaluation" / "datasets" / "golden.jsonl"
TEST_AGENT_ROUTING_PATH = REPO_ROOT / "test_agent_routing.py"


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

    def test_unknown_route_is_recorded_and_excluded_from_confusion_matrix(self):
        cases = [_make_case("a", "a", "document_qa")]

        def decide(question: str):
            return "not_a_real_tool", question

        result = evaluate_routing(cases, decide, measure_latency=False)

        assert result["unknown_route_count"] == 1
        assert result["excluded_count"] == 1
        assert result["correct_count"] == 0
        assert result["failures"][0]["failure_type"] == "unknown_route"
        assert result["failures"][0]["actual_route"] == "not_a_real_tool"
        # labels 밖 값이 precision_recall_f1()에 전달되지 않았어야 하며, KeyError 없이 통과한다.
        pr = result["precision_recall_f1"]
        assert pr["document_qa"]["precision"] == 0.0

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


ORIGINAL_ROUTING_CASES = [
    ("오늘 서울 날씨 좀 웹에서 검색해줘", "web_search"),
    ("최신 파이썬 버전이 몇이야? 인터넷에서 찾아줘", "web_search"),
    ("비트코인 시세를 온라인에서 검색해줘", "web_search"),
    ("삼성전자 주가를 웹검색으로 알아봐줘", "web_search"),
    ("지금 서울 기온이 몇 도야?", "web_search"),
    ("오늘 환율이 어떻게 돼?", "web_search"),
    ("RAG에서 MMR이 뭐야?", "document_qa"),
    ("임베딩이 뭔지 설명해줘", "document_qa"),
    ("리랭커의 역할이 뭐야?", "document_qa"),
    ("FAISS와 Elasticsearch를 비교해줘", "document_qa"),
    ("BM25와 Dense Retrieval의 차이점은?", "document_qa"),
    ("Python에서 FAISS 설치하는 방법을 알려줘", "document_qa"),
    ("벡터스토어를 만드는 절차를 단계별로 설명해줘", "document_qa"),
    ("LangChain은 무료로 사용할 수 있나요?", "document_qa"),
    ("Ollama는 로컬에서 실행되나요?", "document_qa"),
    ("이 문서에서 관련 내용을 찾아줘", "document_qa"),
]


class TestRoutingRegressionMigration:
    """기존 test_agent_routing.py의 ROUTING_CASES 16쌍이 golden.jsonl로 손실 없이
    이관됐는지 검증한다(§6.4, §6.5)."""

    def test_exactly_sixteen_cases_tagged(self):
        cases = load_jsonl(GOLDEN_DATASET_PATH)
        tagged = [c for c in cases if "routing_regression" in c.tags]
        assert len(tagged) == 16

    def test_tagged_cases_match_original_question_route_pairs_one_to_one(self):
        cases = load_jsonl(GOLDEN_DATASET_PATH)
        tagged = [c for c in cases if "routing_regression" in c.tags]
        actual_pairs = sorted((c.question, c.expected_route.value) for c in tagged)
        expected_pairs = sorted(ORIGINAL_ROUTING_CASES)
        assert actual_pairs == expected_pairs

    def test_no_duplicate_answer_list_left_in_test_agent_routing(self):
        text = TEST_AGENT_ROUTING_PATH.read_text(encoding="utf-8")
        assert "ROUTING_CASES" not in text

    def test_test_agent_routing_reads_from_golden_dataset(self):
        text = TEST_AGENT_ROUTING_PATH.read_text(encoding="utf-8")
        assert "routing_regression" in text
        assert "golden.jsonl" in text
