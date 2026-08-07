"""evaluation/baseline.py 단위 테스트 (M2 Phase 7, M2-REQ-009/010).

이 파일의 테스트는 `evaluation.baseline.evaluate_retrieval`,
`evaluation.baseline.evaluate_routing`, `evaluation.baseline.evaluate_answers`,
`evaluation.baseline._resolve_decide_tool`을 fake 함수로 monkeypatch해
orchestration(순서, 상태 보존, fingerprint invariant, skip/limit/tag 전달,
리포트 생성)만 검증한다. 각 evaluator 자체의 내부 알고리즘(Recall/MRR/nDCG
계산, routing confusion matrix, assertion coverage 등)은 이미
test_evaluation_retrieval.py/test_evaluation_routing.py/test_evaluation_answers.py가
검증했으므로 여기서 재검증하지 않는다.

실제 모델, `data/`, `vectorstore/`, Ollama, 네트워크를 전혀 사용하지 않는다 —
`--mode live` 전용 opt-in 검증(TestLiveOptIn)만 예외적으로 subprocess로
`python -m evaluation.baseline`을 실행하지만, RUN_LIVE_LLM_TESTS=1이 없으면
즉시 종료되므로 여전히 모델/네트워크를 호출하지 않는다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import evaluation.baseline as baseline_module
from evaluation.baseline import _positive_int, _render_baseline_markdown, main, run_baseline
from evaluation.compare import evaluate_gates
from evaluation.dataset import DatasetError

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------


def _case(
    id_: str,
    *,
    question: str = "테스트 질문입니다",
    category: str = "document_qa",
    route: str = "document_qa",
    tags: list[str] | None = None,
    **overrides,
) -> dict:
    d = {
        "id": id_,
        "question": question,
        "category": category,
        "expected_route": route,
        "tags": tags if tags is not None else [],
    }
    d.update(overrides)
    return d


def _minimal_valid_dataset_cases() -> list[dict]:
    """9개 구성 규칙(M2-REQ-002)을 정확히 최소치로 충족하는 60개 사례.
    document_qa 40개(30개는 relevant_sources, 20개는 answer_assertions, intent는
    4종 x 5개씩), web_search 10개, boundary 5개, unanswerable 5개(둘 중 5개는
    expect_abstention=True)로 구성된다(test_evaluation_dataset.py의
    `_minimal_valid_cases()`와 동일한 구성 규칙을 재사용해 만들었다).

    document_qa 사례 중 처음 3개(dq0~dq2)에만 "keep" 태그를 붙이고 나머지는
    "drop" 태그를 붙여 --tag/--limit 필터링 전달을 결정론적으로 검증할 수
    있게 한다."""
    cases: list[dict] = []
    intents = ["explanation", "comparison", "procedure", "yesno"]
    for i in range(40):
        overrides: dict = {"expected_intent": intents[i % 4]}
        if i < 30:
            overrides["relevant_sources"] = [f"doc{i}.pdf"]
        if i < 20:
            overrides["answer_assertions"] = [{"any_of": ["핵심답변"]}]
        tag = "keep" if i < 3 else "drop"
        cases.append(_case(f"dq{i}", question=f"문서 관련 질문입니다 {i}", tags=[tag], **overrides))
    for i in range(10):
        cases.append(
            _case(
                f"ws{i}",
                question=f"웹 검색 질문입니다 {i}",
                category="web_search",
                route="web_search",
                expected_intent="other",
                tags=["drop"],
            )
        )
    for i in range(5):
        overrides = {"expected_intent": "uncertain"}
        if i < 3:
            overrides["expect_abstention"] = True
        cases.append(
            _case(f"bd{i}", question=f"경계 사례 질문입니다 {i}", category="boundary", tags=["drop"], **overrides)
        )
    for i in range(5):
        overrides = {"expected_intent": "uncertain"}
        if i < 2:
            overrides["expect_abstention"] = True
        cases.append(
            _case(
                f"ua{i}", question=f"답변불가 질문입니다 {i}", category="unanswerable", tags=["drop"], **overrides
            )
        )
    return cases


def _write_dataset(tmp_path: Path, cases: list[dict], filename: str = "golden.jsonl") -> Path:
    path = tmp_path / filename
    lines = [json.dumps(c, ensure_ascii=False) for c in cases]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _valid_dataset(tmp_path: Path) -> Path:
    return _write_dataset(tmp_path, _minimal_valid_dataset_cases())


def _invalid_dataset(tmp_path: Path) -> Path:
    """구성 검증(validate_composition)이 확실히 실패하는 최소 사례(2건)."""
    return _write_dataset(tmp_path, [_case("only-1"), _case("only-2")])


# ---------------------------------------------------------------------------
# Fake evaluator factories
# ---------------------------------------------------------------------------

DEFAULT_CORPUS_SHA = "corpus-sha-AAA"
DEFAULT_VS_FP = {"index_faiss_sha256": "faiss-AAA", "index_pkl_sha256": "pkl-AAA"}


def _make_fake_evaluate_retrieval(
    calls: list,
    *,
    corpus_sha: str = DEFAULT_CORPUS_SHA,
    vs_fp: dict | None = None,
    raise_exc: Exception | None = None,
    write_files: bool = True,
    extra_failures: list[dict] | None = None,
):
    vs_fp = vs_fp if vs_fp is not None else dict(DEFAULT_VS_FP)

    def fake(dataset_path, output_dir, k_values=(1, 3, 5, 10), limit=None, tag=None, **kwargs):
        calls.append(
            {
                "stage": "retrieval",
                "dataset_path": dataset_path,
                "output_dir": output_dir,
                "limit": limit,
                "tag": tag,
                **kwargs,
            }
        )
        if raise_exc is not None:
            raise raise_exc
        if write_files:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "retrieval_20260101T000000000000Z.json").write_text("{}", encoding="utf-8")
            (output_dir / "retrieval_20260101T000000000000Z.md").write_text("# retrieval", encoding="utf-8")
        case_results = [
            {"id": "r1", "question": "검색 질문 1", "success": True},
            {"id": "r2", "question": "검색 질문 2", "success": True},
        ]
        if extra_failures:
            case_results += extra_failures
        return {
            "case_counts": {"total": len(case_results), "success": 2, "failure": len(extra_failures or []), "excluded": 0},
            "metrics": {
                "recall@1": 0.5,
                "recall@3": 1.0,
                "recall@5": 1.0,
                "recall@10": 1.0,
                "mrr@10": 0.75,
                "ndcg@10": 0.9,
            },
            "latency_ms": {"mean": 10.0, "median": 9.0, "p95": 15.0},
            "stage_summary": {"mmr": {"latency_ms_mean": 5.0}},
            "warmup": {
                "requested_cases": 0,
                "executed_cases": 0,
                "succeeded_cases": 0,
                "failed_cases": 0,
                "case_ids": [],
                "same_process": True,
                "engine_object_id_matches": True,
                "discarded_from_metrics": True,
                "performed": False,
            },
            "mmr_instrumentation": {
                "query_embedding_calls_total": 2,
                "candidate_embedding_calls_total": 0,
                "vector_lookup_hits_total": 100,
                "vector_lookup_misses_total": 0,
                "fallback_case_count": 0,
                "fallback_reasons": {},
            },
            "case_results": case_results,
            "corpus_manifest": [{"source_id": "a.pdf", "size_bytes": 1, "sha256": "x"}],
            "corpus_manifest_sha256": corpus_sha,
            "vectorstore_fingerprint": vs_fp,
            "reproducibility_note": None,
            "report_json_path": str(output_dir / "retrieval_20260101T000000000000Z.json") if write_files else None,
            "report_markdown_path": str(output_dir / "retrieval_20260101T000000000000Z.md") if write_files else None,
            "candidate": {
                "candidate_id": None,
                "label": None,
                "phase": None,
                "baseline_ref": "evaluation/baselines/m2_initial.json",
                "baseline_dataset_sha256": "61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a",
                "notes": None,
            },
        }

    return fake


def _make_fake_evaluate_routing(calls: list, *, raise_exc: Exception | None = None, failures: list[dict] | None = None):
    def fake(cases, decide_tool, *, runs=1, measure_latency=True):
        calls.append(
            {
                "stage": "routing",
                "cases": list(cases),
                "runs": runs,
                "measure_latency": measure_latency,
                "decide_tool": decide_tool,
            }
        )
        if raise_exc is not None:
            raise raise_exc
        document_qa_total = sum(1 for c in cases if getattr(c.expected_route, "value", c.expected_route) == "document_qa")
        web_search_total = len(cases) - document_qa_total
        return {
            "total_cases": len(cases),
            "success_count": len(cases),
            "failure_count": 0,
            "excluded_count": 0,
            "correct_count": len(cases),
            "accuracy": 1.0 if cases else 0.0,
            "no_tool_count": 0,
            "unknown_route_count": 0,
            "exception_count": 0,
            "precision_recall_f1": {
                "document_qa": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "web_search": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "confusion_matrix": {
                    "document_qa": {"document_qa": len(cases), "web_search": 0},
                    "web_search": {"document_qa": 0, "web_search": 0},
                },
            },
            "recall_denominators": {"document_qa": document_qa_total, "web_search": web_search_total},
            "document_route_correct": document_qa_total,
            "document_route_recall": 1.0 if document_qa_total else None,
            "web_search_correct": web_search_total,
            "web_search_recall": 1.0 if web_search_total else None,
            "latency_ms": {"measured": True, "mean": 5.0, "median": 5.0, "p95": 5.0},
            "failures": failures or [],
            "case_routes": [
                {"id": c.id, "expected_route": getattr(c.expected_route, "value", c.expected_route), "route": getattr(c.expected_route, "value", c.expected_route)}
                for c in cases
            ],
            "run_count": runs,
            "median_run_index": 0,
            "per_run": None if runs == 1 else [],
            "aggregate": None if runs == 1 else {},
            "case_variation": None if runs == 1 else [],
        }

    return fake


def _make_fake_evaluate_answers(
    calls: list,
    *,
    corpus_sha: str = DEFAULT_CORPUS_SHA,
    vs_fp: dict | None = None,
    raise_exc: Exception | None = None,
    failures: list[dict] | None = None,
):
    vs_fp = vs_fp if vs_fp is not None else dict(DEFAULT_VS_FP)

    def fake(dataset_path, output_dir, limit=None, tag=None, **kwargs):
        calls.append(
            {
                "stage": "answers",
                "dataset_path": dataset_path,
                "output_dir": output_dir,
                "limit": limit,
                "tag": tag,
                **kwargs,
            }
        )
        if raise_exc is not None:
            raise raise_exc
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "answers_20260101T000000000000Z.json"
        md_path = output_dir / "answers_20260101T000000000000Z.md"
        worksheet_path = output_dir / "answers_20260101T000000000000Z_worksheet.md"
        json_path.write_text("{}", encoding="utf-8")
        md_path.write_text("# answers", encoding="utf-8")
        worksheet_path.write_text("worksheet", encoding="utf-8")
        return {
            "case_counts": {"total_considered": 2, "eligible": 2, "excluded_non_eligible": 0, "success": 2, "failure": 0},
            "assertion": {
                "cases_scored": 2,
                "cases_excluded_no_assertion": 0,
                "cases_excluded_failure": 0,
                "assertions_total": 2,
                "assertions_passed": 2,
                "pass_rate": 1.0,
                "limitation_note": "assertion coverage는 faithfulness를 보증하지 않습니다.",
            },
            "abstention": {
                "true_positive": 0,
                "true_negative": 2,
                "false_positive": 0,
                "false_negative": 0,
                "accuracy": 1.0,
                "abstention_accuracy_excluded_reason": None,
                "evaluated_count": 2,
            },
            "source": {"evaluated_count": 2, "excluded_count": 0, "skipped_entries_total": 0, "any_hit_rate": 1.0, "mean_recall": 1.0},
            "intent": {"evaluated_count": 2, "excluded_count": 0, "correct_count": 2, "accuracy": 1.0},
            "evaluator_versions": {
                "assertion": "v1+v2",
                "abstention": "v1+v2",
                "rules_fingerprint": "fake",
                "evaluator_profile": "v2",
                "reviewed_variants_loaded": True,
                "reviewed_variants_sha256": "fake",
                "official": True,
            },
            "assertion_v2": {
                "cases_scored": 2,
                "assertions_total": 2,
                "assertions_passed": 2,
                "pass_rate": 1.0,
                "fixed_vs_v1": [],
                "regressed_vs_v1": [],
            },
            "abstention_v2": {
                "true_positive": 0,
                "true_negative": 2,
                "false_positive": 0,
                "false_negative": 0,
                "accuracy": 1.0,
                "evaluated_count": 2,
                "abstention_accuracy_excluded_reason": None,
                "fixed_vs_v1": [],
                "regressed_vs_v1": [],
            },
            "latency_ms": {"mean_ms": 20.0, "median_ms": 20.0, "p95_ms": 20.0, "count": 2},
            "measured_case_count": 2,
            "warmup": {
                "requested_cases": 0,
                "executed_cases": 0,
                "succeeded_cases": 0,
                "failed_cases": 0,
                "case_ids": [],
                "same_process": True,
                "engine_object_id_matches": True,
                "discarded_from_metrics": True,
                "performed": False,
            },
            "failures": failures or [],
            "corpus_manifest": [{"source_id": "a.pdf", "size_bytes": 1, "sha256": "x"}],
            "corpus_manifest_sha256": corpus_sha,
            "vectorstore_fingerprint": vs_fp,
            "reproducibility_note": None,
            "report_json_path": str(json_path),
            "report_markdown_path": str(md_path),
            "worksheet_path": str(worksheet_path),
            "candidate": {
                "candidate_id": None,
                "label": None,
                "phase": None,
                "baseline_ref": "evaluation/baselines/m2_initial.json",
                "baseline_dataset_sha256": "61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a",
                "notes": None,
            },
        }

    return fake


def _make_fake_evaluate_retrieval_matching_disk(calls: list, **kwargs):
    """`_make_fake_evaluate_retrieval()`처럼 orchestration을 검증하는 대부분의
    테스트는 산출 JSON 파일 내용("{}")과 반환 payload가 달라도 상관없다(
    run_baseline()은 반환 payload만 gate 계산에 쓰기 때문). 하지만 저장된 child
    JSON을 다시 읽어 evaluate_gates()가 같은 결과를 내는지 검증하려면(Iteration 2
    MINOR m1) 디스크에 쓰는 내용이 실제 evaluate_retrieval()처럼 반환 payload와
    동일해야 한다 — 그래서 base fake가 만든 payload를 그대로 같은 파일에 다시
    직렬화한다."""
    base_fake = _make_fake_evaluate_retrieval(calls, **kwargs)

    def fake(dataset_path, output_dir, k_values=(1, 3, 5, 10), limit=None, tag=None, **fake_kwargs):
        payload = base_fake(dataset_path, output_dir, k_values=k_values, limit=limit, tag=tag, **fake_kwargs)
        Path(payload["report_json_path"]).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        return payload

    return fake


def _make_fake_evaluate_answers_matching_disk(calls: list, **kwargs):
    """`_make_fake_evaluate_retrieval_matching_disk()`와 동일한 이유로 answers
    child JSON도 반환 payload와 동일한 내용을 디스크에 남긴다."""
    base_fake = _make_fake_evaluate_answers(calls, **kwargs)

    def fake(dataset_path, output_dir, limit=None, tag=None, **fake_kwargs):
        payload = base_fake(dataset_path, output_dir, limit=limit, tag=tag, **fake_kwargs)
        Path(payload["report_json_path"]).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        return payload

    return fake


_SENTINEL_DECIDE_TOOL = object()


def _patch_all_success(monkeypatch, calls: list, **overrides):
    monkeypatch.setattr(baseline_module, "evaluate_retrieval", overrides.get("retrieval") or _make_fake_evaluate_retrieval(calls))
    monkeypatch.setattr(baseline_module, "evaluate_routing_multi", overrides.get("routing") or _make_fake_evaluate_routing(calls))
    monkeypatch.setattr(baseline_module, "evaluate_answers", overrides.get("answers") or _make_fake_evaluate_answers(calls))
    monkeypatch.setattr(baseline_module, "_resolve_decide_tool", lambda: _SENTINEL_DECIDE_TOOL)
    # run_baseline() 자신도 opt-in을 검사하므로(M2_Phase_7_8_code_review_result.md
    # P2) fake evaluator만 주입하는 대부분의 orchestration 테스트는 opt-in도
    # 함께 켜 둔다. opt-in 자체를 검증하는 테스트(TestLiveOptIn)는 이 헬퍼를
    # 쓴 뒤 명시적으로 monkeypatch.delenv()로 다시 지운다.
    monkeypatch.setenv("RUN_LIVE_LLM_TESTS", "1")


# ---------------------------------------------------------------------------
# 1~2: 실행 순서, 4단계 상태/집계 결과 보존
# ---------------------------------------------------------------------------


class TestOrchestrationOrderAndPreservation:
    def test_stage_call_order_is_retrieval_then_routing_then_answers(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        assert [c["stage"] for c in calls] == ["retrieval", "routing", "answers"]
        assert result["overall_success"] is True

    def test_successful_run_preserves_all_four_stage_statuses_and_aggregates(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        stages = result["stages"]
        assert stages["validate"]["status"] == "success"
        assert stages["retrieval"]["status"] == "success"
        assert stages["routing"]["status"] == "success"
        assert stages["answers"]["status"] == "success"

        assert stages["retrieval"]["metrics"]["recall@1"] == 0.5
        assert stages["routing"]["accuracy"] == 1.0
        assert stages["answers"]["assertion"]["pass_rate"] == 1.0
        assert result["overall_success"] is True


# ---------------------------------------------------------------------------
# 3: dataset validation 실패
# ---------------------------------------------------------------------------


class TestDatasetValidationFailureShortCircuits:
    def test_composition_invalid_calls_no_evaluator_and_fails_overall(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _invalid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        assert calls == []
        assert result["overall_success"] is False
        assert result["stages"]["validate"]["status"] == "failed"
        assert result["stages"]["retrieval"]["status"] == "not_run"
        assert result["stages"]["routing"]["status"] == "not_run"
        assert result["stages"]["answers"]["status"] == "not_run"

    def test_composition_invalid_still_writes_baseline_report(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _invalid_dataset(tmp_path)
        output_dir = tmp_path / "reports"

        run_baseline(dataset_path, output_dir)

        assert list(output_dir.glob("baseline_*.json"))

    def test_missing_dataset_file_calls_no_evaluator_and_fails_overall(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = tmp_path / "does-not-exist.jsonl"

        result = run_baseline(dataset_path, tmp_path / "reports")

        assert calls == []
        assert result["overall_success"] is False
        assert result["stages"]["validate"]["status"] == "failed"
        assert result["stages"]["validate"]["error_type"] == DatasetError.__name__
        assert result["stages"]["retrieval"]["status"] == "not_run"
        assert result["stages"]["routing"]["status"] == "not_run"
        assert result["stages"]["answers"]["status"] == "not_run"

    def test_missing_dataset_file_does_not_write_report(self, tmp_path, monkeypatch):
        """dataset 자체를 읽을 수 없으면 build_metadata()가 요구하는
        dataset_path.read_bytes()가 불가능하므로 baseline 리포트 파일을 만들지
        않는다(기존 evaluator main()들의 동일 상황 처리와 일치)."""
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = tmp_path / "does-not-exist.jsonl"
        output_dir = tmp_path / "reports"

        run_baseline(dataset_path, output_dir)

        assert not output_dir.exists()


# ---------------------------------------------------------------------------
# 4: Retrieval 실패 후 처리 정책과 결과 보존
# ---------------------------------------------------------------------------


class TestRetrievalFailurePolicy:
    def test_retrieval_failure_still_attempts_routing_and_answers(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(
            monkeypatch,
            calls,
            retrieval=_make_fake_evaluate_retrieval(calls, raise_exc=RuntimeError("엔진 초기화 실패")),
        )
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        stage_names = [c["stage"] for c in calls]
        assert "routing" in stage_names
        assert "answers" in stage_names
        assert result["stages"]["retrieval"]["status"] == "failed"
        assert result["stages"]["retrieval"]["error_type"] == "RuntimeError"
        assert result["overall_success"] is False

    def test_retrieval_failure_leaves_top_level_fingerprint_null_with_note(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(
            monkeypatch,
            calls,
            retrieval=_make_fake_evaluate_retrieval(calls, raise_exc=FileNotFoundError("vectorstore 없음")),
        )
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        assert result["corpus_manifest"] is None
        assert result["corpus_manifest_sha256"] is None
        assert result["vectorstore_fingerprint"] is None
        assert result["reproducibility_note"]
        assert "Retrieval" in result["reproducibility_note"]

    def test_retrieval_failure_preserves_routing_and_answers_success_results(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(
            monkeypatch,
            calls,
            retrieval=_make_fake_evaluate_retrieval(calls, raise_exc=RuntimeError("boom")),
        )
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        assert result["stages"]["routing"]["status"] == "success"
        assert result["stages"]["routing"]["accuracy"] == 1.0
        assert result["stages"]["answers"]["status"] == "success"
        assert result["stages"]["answers"]["assertion"]["pass_rate"] == 1.0


# ---------------------------------------------------------------------------
# 5: Routing 실패 후 Answer는 계속 실행
# ---------------------------------------------------------------------------


class TestRoutingFailurePolicy:
    def test_routing_exception_does_not_block_answers_and_overall_fails(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(
            monkeypatch,
            calls,
            routing=_make_fake_evaluate_routing(calls, raise_exc=RuntimeError("Ollama 연결 실패")),
        )
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        stage_names = [c["stage"] for c in calls]
        assert "answers" in stage_names
        assert result["stages"]["routing"]["status"] == "failed"
        assert result["stages"]["routing"]["error_type"] == "RuntimeError"
        assert result["stages"]["retrieval"]["status"] == "success"
        assert result["stages"]["answers"]["status"] == "success"
        assert result["overall_success"] is False

    def test_routing_empty_cases_value_error_is_recorded_as_failed_not_crash(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        # "no-such-tag"에 매칭되는 사례가 없으므로 evaluate_routing_multi()의
        # 실제 계약(ValueError)과 동일하게 fake도 재현한다.
        def raising_routing(cases, decide_tool, *, runs=1, measure_latency=True):
            calls.append({"stage": "routing", "cases": list(cases)})
            if not cases:
                raise ValueError("평가할 사례가 없습니다")
            raise AssertionError("unreachable")

        monkeypatch.setattr(baseline_module, "evaluate_routing_multi", raising_routing)

        result = run_baseline(dataset_path, tmp_path / "reports", tag="no-such-tag")

        assert result["stages"]["routing"]["status"] == "failed"
        assert result["stages"]["routing"]["error_type"] == "ValueError"
        assert result["overall_success"] is False

    def test_routing_report_write_failure_does_not_block_answers_and_preserves_retrieval(
        self, tmp_path, monkeypatch
    ):
        """M2_Phase_7_8_code_review_result.md P1: evaluate_routing() 자체는
        성공했지만 그 결과를 report로 남기는 write_report() 호출(디스크 공간
        부족, 권한 문제 등)이 실패하는 경우를 재현한다. 이전에는 이 예외가
        run_baseline() 밖으로 그대로 전파되어 Answer 단계가 전혀 실행되지
        않고, 이미 성공한 Retrieval 결과도 최종 report에 남지 않았다.
        수정 후에는 report 작성 실패도 routing 단계의 failed 상태로 기록되고,
        Retrieval 결과는 보존되며, Answer 단계는 계속 실행되어야 한다."""
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        real_write_report = baseline_module.write_report

        def flaky_write_report(payload, output_dir, name, render_markdown=None):
            if name == "routing":
                raise OSError("디스크 여유 공간 부족")
            return real_write_report(payload, output_dir, name, render_markdown=render_markdown)

        monkeypatch.setattr(baseline_module, "write_report", flaky_write_report)

        result = run_baseline(dataset_path, tmp_path / "reports")

        stage_names = [c["stage"] for c in calls]
        assert "answers" in stage_names
        assert result["stages"]["retrieval"]["status"] == "success"
        assert result["stages"]["routing"]["status"] == "failed"
        assert result["stages"]["routing"]["error_type"] == "OSError"
        assert result["stages"]["routing"]["evaluation_status"] == "success"
        assert result["stages"]["routing"]["report_status"] == "failed"
        assert result["stages"]["answers"]["status"] == "success"
        assert result["overall_success"] is False


# ---------------------------------------------------------------------------
# 6: Answer 실패 후 앞 단계 결과 유지
# ---------------------------------------------------------------------------


class TestAnswersFailurePolicy:
    def test_answers_failure_preserves_retrieval_and_routing_success(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(
            monkeypatch,
            calls,
            answers=_make_fake_evaluate_answers(calls, raise_exc=RuntimeError("RAG 엔진 실패")),
        )
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        assert result["stages"]["retrieval"]["status"] == "success"
        assert result["stages"]["routing"]["status"] == "success"
        assert result["stages"]["answers"]["status"] == "failed"
        assert result["stages"]["answers"]["error_type"] == "RuntimeError"
        assert result["overall_success"] is False


# ---------------------------------------------------------------------------
# 7: --skip-routing / --skip-answers
# ---------------------------------------------------------------------------


class TestSkipOptions:
    def test_skip_routing_marks_skipped_with_reason_and_does_not_call_evaluator(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports", skip_routing=True)

        assert [c["stage"] for c in calls] == ["retrieval", "answers"]
        assert result["stages"]["routing"] == {"stage": "routing", "status": "skipped", "reason": "사용자가 --skip-routing으로 명시적으로 제외함"}
        assert result["overall_success"] is True

    def test_skip_answers_marks_skipped_with_reason_and_does_not_call_evaluator(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports", skip_answers=True)

        assert [c["stage"] for c in calls] == ["retrieval", "routing"]
        assert result["stages"]["answers"]["status"] == "skipped"
        assert result["stages"]["answers"]["reason"]
        assert result["overall_success"] is True

    def test_both_skipped_only_retrieval_runs(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports", skip_routing=True, skip_answers=True)

        assert [c["stage"] for c in calls] == ["retrieval"]
        assert result["stages"]["routing"]["status"] == "skipped"
        assert result["stages"]["answers"]["status"] == "skipped"
        assert result["overall_success"] is True


# ---------------------------------------------------------------------------
# 8: limit 양수 검증
# ---------------------------------------------------------------------------


class TestLimitValidation:
    @pytest.mark.parametrize("bad_limit", [0, -1])
    def test_api_rejects_non_positive_limit(self, tmp_path, bad_limit):
        with pytest.raises(ValueError):
            run_baseline(tmp_path / "golden.jsonl", tmp_path / "reports", limit=bad_limit)

    @pytest.mark.parametrize("bad_limit", ["0", "-1"])
    def test_cli_rejects_non_positive_limit_with_exit_2(self, tmp_path, bad_limit, monkeypatch):
        monkeypatch.setenv("RUN_LIVE_LLM_TESTS", "1")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--dataset", str(tmp_path / "golden.jsonl"),
                    "--output", str(tmp_path / "reports"),
                    "--limit", bad_limit,
                ]
            )
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# 9: tag/limit 전달
# ---------------------------------------------------------------------------


class TestTagAndLimitPropagation:
    def test_tag_and_limit_forwarded_verbatim_to_retrieval_and_answers(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        run_baseline(dataset_path, tmp_path / "reports", tag="keep", limit=2)

        retrieval_call = next(c for c in calls if c["stage"] == "retrieval")
        answers_call = next(c for c in calls if c["stage"] == "answers")
        assert retrieval_call["tag"] == "keep"
        assert retrieval_call["limit"] == 2
        assert answers_call["tag"] == "keep"
        assert answers_call["limit"] == 2

    def test_tag_and_limit_applied_to_routing_cases_preserving_order(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        run_baseline(dataset_path, tmp_path / "reports", tag="keep", limit=2)

        routing_call = next(c for c in calls if c["stage"] == "routing")
        ids = [c.id for c in routing_call["cases"]]
        # "keep" 태그는 dq0/dq1/dq2 세 사례에만 붙어 있다(원본 순서 유지) ->
        # limit=2 적용 후 [dq0, dq1]이어야 한다.
        assert ids == ["dq0", "dq1"]

    def test_decide_tool_resolved_via_hook_is_forwarded_to_evaluate_routing(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        run_baseline(dataset_path, tmp_path / "reports")

        routing_call = next(c for c in calls if c["stage"] == "routing")
        assert routing_call["decide_tool"] is _SENTINEL_DECIDE_TOOL


# ---------------------------------------------------------------------------
# 10~15: fingerprint invariant
# ---------------------------------------------------------------------------


class TestFingerprintInvariant:
    def test_top_level_fingerprint_matches_retrieval_value(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        assert result["corpus_manifest_sha256"] == DEFAULT_CORPUS_SHA
        assert result["vectorstore_fingerprint"] == DEFAULT_VS_FP
        assert result["corpus_manifest"] is not None

    def test_top_level_fingerprint_preserved_with_skip_answers(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports", skip_answers=True)

        assert result["corpus_manifest_sha256"] == DEFAULT_CORPUS_SHA
        assert result["vectorstore_fingerprint"] == DEFAULT_VS_FP

    def test_matching_fingerprints_between_retrieval_and_answers_yield_success(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        assert result["fingerprint_invariant"]["checked"] is True
        assert result["fingerprint_invariant"]["ok"] is True
        assert result["overall_success"] is True

    def test_corpus_manifest_sha256_mismatch_fails_overall(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(
            monkeypatch,
            calls,
            answers=_make_fake_evaluate_answers(calls, corpus_sha="corpus-sha-DIFFERENT"),
        )
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        assert result["fingerprint_invariant"]["checked"] is True
        assert result["fingerprint_invariant"]["corpus_manifest_sha256_match"] is False
        assert result["fingerprint_invariant"]["ok"] is False
        assert result["overall_success"] is False

    def test_vectorstore_fingerprint_mismatch_fails_overall(self, tmp_path, monkeypatch):
        calls: list = []
        mismatched_vs_fp = {"index_faiss_sha256": "faiss-DIFFERENT", "index_pkl_sha256": "pkl-AAA"}
        _patch_all_success(
            monkeypatch,
            calls,
            answers=_make_fake_evaluate_answers(calls, vs_fp=mismatched_vs_fp),
        )
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        assert result["fingerprint_invariant"]["vectorstore_fingerprint_match"] is False
        assert result["fingerprint_invariant"]["corpus_manifest_sha256_match"] is True
        assert result["fingerprint_invariant"]["ok"] is False
        assert result["overall_success"] is False

    def test_fingerprint_mismatch_preserves_both_values_and_stage_results(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(
            monkeypatch,
            calls,
            answers=_make_fake_evaluate_answers(calls, corpus_sha="corpus-sha-DIFFERENT"),
        )
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        fp = result["fingerprint_invariant"]
        assert fp["retrieval_corpus_manifest_sha256"] == DEFAULT_CORPUS_SHA
        assert fp["answers_corpus_manifest_sha256"] == "corpus-sha-DIFFERENT"
        # 불일치가 발생해도 각 단계의 성공 결과 자체는 지워지지 않는다.
        assert result["stages"]["retrieval"]["status"] == "success"
        assert result["stages"]["answers"]["status"] == "success"
        assert result["corpus_manifest_sha256"] == DEFAULT_CORPUS_SHA  # top-level은 Retrieval 값 유지

    def test_routing_null_reproducibility_does_not_pollute_top_level(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        # top-level은 여전히 Retrieval 값이어야 하고 null이 아니어야 한다.
        assert result["corpus_manifest_sha256"] == DEFAULT_CORPUS_SHA
        assert result["vectorstore_fingerprint"] == DEFAULT_VS_FP

        # Routing 자체 리포트는 not_applicable(null) 값을 유지해야 한다.
        routing_json_path = Path(result["stages"]["routing"]["report_json_path"])
        routing_payload = json.loads(routing_json_path.read_text(encoding="utf-8"))
        assert routing_payload["corpus_manifest"] is None
        assert routing_payload["corpus_manifest_sha256"] is None
        assert routing_payload["vectorstore_fingerprint"] is None
        assert routing_payload["reproducibility_note"]
        assert routing_payload["schema_version"] == "1.1.0"
        assert routing_payload["router_prompt_sha256"]
        assert routing_payload["routing_policy"]["signal_override"] is True
        assert routing_payload["routing_policy"]["signal_counts"] == {
            "web": 0,
            "document": 0,
            "none": 60,
        }


# ---------------------------------------------------------------------------
# 16~17: 리포트 내용/Markdown escape
# ---------------------------------------------------------------------------


class TestReportContent:
    def test_json_report_contains_real_metric_values(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)
        output_dir = tmp_path / "reports"

        run_baseline(dataset_path, output_dir)

        json_path = next(output_dir.glob("baseline_*.json"))
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["stages"]["retrieval"]["metrics"]["recall@1"] == 0.5
        assert payload["stages"]["routing"]["accuracy"] == 1.0
        assert payload["stages"]["answers"]["assertion"]["pass_rate"] == 1.0
        assert payload["overall_success"] is True

    def test_markdown_report_shows_real_metrics_and_failures_not_just_json_pointer(self, tmp_path, monkeypatch):
        calls: list = []
        failing_case = {"id": "ans-fail-1", "question": "실패한 질문", "error": "RAG 호출 실패"}
        _patch_all_success(
            monkeypatch,
            calls,
            answers=_make_fake_evaluate_answers(calls, failures=[failing_case]),
        )
        dataset_path = _valid_dataset(tmp_path)
        output_dir = tmp_path / "reports"

        run_baseline(dataset_path, output_dir)

        md_path = next(output_dir.glob("baseline_*.md"))
        text = md_path.read_text(encoding="utf-8")
        assert "성공" in text
        assert "50.0%" not in text or "recall@1" in text  # recall@1=0.5가 표시되는지
        assert "recall@1" in text
        assert "100.0%" in text  # routing accuracy 1.0 표시
        assert "ans-fail-1" in text
        assert "실패한 질문" in text
        assert "RAG 호출 실패" in text

    def test_markdown_survives_pipe_backslash_newline_in_failure_fields(self):
        """escape_markdown_table_cell()을 재사용해 표 구조가 깨지지 않는지
        `_render_baseline_markdown()`을 직접 호출해 확인한다."""
        payload = {
            "generated_at_utc": "2026-01-01T00:00:00Z",
            "command": ["python", "-m", "evaluation.baseline"],
            "dataset_path": "golden.jsonl",
            "dataset_sha256": "abc",
            "git_commit": "deadbeef",
            "git_dirty": False,
            "overall_success": False,
            "stages": {
                "validate": {"stage": "validate", "status": "success"},
                "retrieval": {"stage": "retrieval", "status": "success", "case_counts": {}, "metrics": {}, "latency_ms": {}, "failures": []},
                "routing": {
                    "stage": "routing",
                    "status": "success",
                    "total_cases": 1,
                    "correct_count": 0,
                    "accuracy": 0.0,
                    "precision_recall_f1": {"document_qa": {}, "web_search": {}},
                    "failures": [
                        {
                            "id": "a|b",
                            "question": "질문 | 파이프\n줄바꿈\\백슬래시",
                            "error": "오류|메시지\\포함",
                        }
                    ],
                },
                "answers": {"stage": "answers", "status": "skipped", "reason": "테스트"},
            },
            "stage_duration_seconds": {"validate": 0.0, "retrieval": 0.0, "routing": 0.0, "answers": 0.0},
            "total_duration_seconds": 0.0,
            "corpus_manifest_sha256": None,
            "vectorstore_fingerprint": {},
            "fingerprint_invariant": {"checked": False},
            "reproducibility_limitations": [],
            "reproducibility_note": None,
        }
        text = _render_baseline_markdown(payload)
        failure_row = next(line for line in text.splitlines() if "a\\|b" in line)
        assert "\n" not in failure_row.strip()
        assert "파이프" in failure_row
        # pipe가 이스케이프되어 표 열 구분자와 섞이지 않아야 한다.
        assert "\\|" in failure_row

    def test_report_reruns_do_not_overwrite_previous_files(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)
        output_dir = tmp_path / "reports"

        run_baseline(dataset_path, output_dir)
        run_baseline(dataset_path, output_dir)

        json_files = list(output_dir.glob("baseline_*.json"))
        assert len(json_files) == 2


# ---------------------------------------------------------------------------
# 19: --help / import이 live dependency를 초기화하지 않음
# ---------------------------------------------------------------------------


class TestNoLiveDependencyOnImportOrHelp:
    def test_help_exits_zero_without_reaching_live_check(self, monkeypatch):
        monkeypatch.delenv("RUN_LIVE_LLM_TESTS", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_missing_required_args_is_argparse_error(self):
        with pytest.raises(SystemExit):
            main([])

    def test_module_import_does_not_import_agent_or_rag_engine(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import evaluation.baseline, sys; "
                "assert 'simple_qna_rag.agent' not in sys.modules; "
                "assert 'simple_qna_rag.rag_engine' not in sys.modules; "
                "print('OK')",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_cli_help_subprocess_does_not_touch_live_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-m", "evaluation.baseline", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            env={k: v for k, v in os.environ.items() if k != "RUN_LIVE_LLM_TESTS"},
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# live opt-in
# ---------------------------------------------------------------------------


class TestLiveOptIn:
    def test_live_without_opt_in_exits_nonzero_before_dataset_load(self, tmp_path):
        env = os.environ.copy()
        env.pop("RUN_LIVE_LLM_TESTS", None)

        result = subprocess.run(
            [
                sys.executable, "-m", "evaluation.baseline",
                "--dataset", str(tmp_path / "does-not-exist.jsonl"),
                "--output", str(tmp_path / "reports"),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 2
        assert "RUN_LIVE_LLM_TESTS" in result.stderr
        assert "찾을 수 없습니다" not in result.stderr
        assert not (tmp_path / "reports").exists()

    def test_run_baseline_api_itself_requires_opt_in(self, tmp_path, monkeypatch):
        """M2_Phase_7_8_code_review_result.md P2: run_baseline()은 라이브러리
        API로 직접 호출될 수 있으므로, CLI main()과 별개로 자기 자신도
        RUN_LIVE_LLM_TESTS=1 opt-in을 검사해야 한다 — 그렇지 않으면
        monkeypatch 없이 직접 호출하는 코드가 opt-in 없이 실제
        Ollama/vectorstore를 건드릴 수 있다. evaluator가 모두 mock된
        상태에서도(=실제로는 안전하더라도) opt-in이 없으면 RuntimeError를
        내야 한다(§4.5, run_baseline()은 sys.exit()을 호출하지 않으므로
        SystemExit이 아니라 RuntimeError)."""
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        monkeypatch.delenv("RUN_LIVE_LLM_TESTS", raising=False)
        dataset_path = _valid_dataset(tmp_path)

        with pytest.raises(RuntimeError, match="RUN_LIVE_LLM_TESTS"):
            run_baseline(dataset_path, tmp_path / "reports")


# ---------------------------------------------------------------------------
# 20: CLI exit code
# ---------------------------------------------------------------------------


class TestCliExitCodes:
    def test_full_success_exits_zero(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        monkeypatch.setenv("RUN_LIVE_LLM_TESTS", "1")
        dataset_path = _valid_dataset(tmp_path)

        exit_code = main(["--dataset", str(dataset_path), "--output", str(tmp_path / "reports")])

        assert exit_code == 0

    def test_stage_failure_exits_nonzero(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(
            monkeypatch,
            calls,
            answers=_make_fake_evaluate_answers(calls, raise_exc=RuntimeError("boom")),
        )
        monkeypatch.setenv("RUN_LIVE_LLM_TESTS", "1")
        dataset_path = _valid_dataset(tmp_path)

        exit_code = main(["--dataset", str(dataset_path), "--output", str(tmp_path / "reports")])

        assert exit_code != 0

    def test_dataset_validation_failure_exits_nonzero(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        monkeypatch.setenv("RUN_LIVE_LLM_TESTS", "1")
        dataset_path = _invalid_dataset(tmp_path)

        exit_code = main(["--dataset", str(dataset_path), "--output", str(tmp_path / "reports")])

        assert exit_code != 0
        assert calls == []


# ---------------------------------------------------------------------------
# _positive_int 단위 테스트
# ---------------------------------------------------------------------------


class TestPositiveInt:
    def test_accepts_positive_values(self):
        assert _positive_int("1") == 1
        assert _positive_int("10") == 10

    @pytest.mark.parametrize("bad", ["0", "-1", "-100"])
    def test_rejects_non_positive_values(self, bad):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int(bad)


# ---------------------------------------------------------------------------
# M3 Phase 6: gate_evaluation / candidate / routing_runs / warmup_cases
# ---------------------------------------------------------------------------


class TestM3GateWiring:
    def test_gate_evaluation_present_and_uses_evaluate_gates(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        assert "gate_evaluation" in result
        assert result["gate_evaluation"]["spec_version"] == "m3-4.1"
        assert isinstance(result["gate_evaluation"]["items"], list)
        assert len(result["gate_evaluation"]["items"]) == 14  # M3_GATES 개수

    def test_candidate_block_forwarded_from_candidate_id(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports", candidate_id="m3-final")

        assert result["candidate"]["candidate_id"] == "m3-final"
        assert result["candidate"]["phase"] == 6

    def test_routing_runs_forwarded_to_evaluate_routing_multi(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        run_baseline(dataset_path, tmp_path / "reports", routing_runs=3)

        routing_calls = [c for c in calls if c["stage"] == "routing"]
        assert routing_calls[0]["runs"] == 3

    def test_warmup_cases_forwarded_to_retrieval_and_answers(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        run_baseline(dataset_path, tmp_path / "reports", warmup_cases=3)

        retrieval_calls = [c for c in calls if c["stage"] == "retrieval"]
        answers_calls = [c for c in calls if c["stage"] == "answers"]
        assert retrieval_calls[0]["warmup_cases"] == 3
        assert answers_calls[0]["warmup_cases"] == 3

    def test_schema_version_bumped_to_1_1_0(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")
        assert result["schema_version"] == "1.1.0"

    def test_gate_evaluation_absent_stage_reports_pass_none(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_all_success(monkeypatch, calls)
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports", skip_answers=True, skip_routing=True)

        gate_items = result["gate_evaluation"]["items"]
        answer_latency_item = next(i for i in gate_items if i["id"] == "answer_latency_mean_s")
        assert answer_latency_item["pass"] is None
        assert result["gate_evaluation"]["overall_pass"] is False


# ---------------------------------------------------------------------------
# Code_Review_Iteration_2.md MINOR m1: 저장된 child artifact를 다시 읽어
# evaluate_gates()가 baseline이 기록한 gate_evaluation과 동일한 결과를 내는지
# 확인하는 회귀 테스트. Iteration 1에서 발생했던 in-memory payload와 저장된
# artifact 간 불일치(M1)가 재발하면 이 테스트가 잡아야 한다.
# ---------------------------------------------------------------------------


class TestChildArtifactReloadGateParity:
    def test_reloaded_retrieval_routing_answers_json_reproduce_stored_gate_evaluation(
        self, tmp_path, monkeypatch
    ):
        calls: list = []
        _patch_all_success(
            monkeypatch,
            calls,
            retrieval=_make_fake_evaluate_retrieval_matching_disk(calls),
            answers=_make_fake_evaluate_answers_matching_disk(calls),
        )
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        retrieval_json_path = Path(result["stages"]["retrieval"]["report_json_path"])
        routing_json_path = Path(result["stages"]["routing"]["report_json_path"])
        answers_json_path = Path(result["stages"]["answers"]["report_json_path"])

        reloaded_retrieval = json.loads(retrieval_json_path.read_text(encoding="utf-8"))
        reloaded_routing = json.loads(routing_json_path.read_text(encoding="utf-8"))
        reloaded_answers = json.loads(answers_json_path.read_text(encoding="utf-8"))

        recomputed_gate_evaluation = evaluate_gates(
            {
                "retrieval": reloaded_retrieval,
                "routing": reloaded_routing,
                "answers": reloaded_answers,
            }
        )

        assert recomputed_gate_evaluation == result["gate_evaluation"]
        # gate 전부가 판정 불가(None)면 이 parity 검증 자체가 무의미해지므로,
        # 최소한 일부 gate는 실제로 True/False 판정이 났는지 확인한다.
        assert any(item["pass"] is not None for item in recomputed_gate_evaluation["items"])

    def test_reloaded_child_json_matches_stage_summary_forwarded_into_baseline(
        self, tmp_path, monkeypatch
    ):
        """저장된 child JSON이 baseline stages 요약에 그대로 반영된 핵심 지표
        (recall@10, source any_hit_rate)와도 일치하는지 확인해, 저장/재로드
        경로에서 값이 조용히 바뀌지 않았음을 이중 확인한다."""
        calls: list = []
        _patch_all_success(
            monkeypatch,
            calls,
            retrieval=_make_fake_evaluate_retrieval_matching_disk(calls),
            answers=_make_fake_evaluate_answers_matching_disk(calls),
        )
        dataset_path = _valid_dataset(tmp_path)

        result = run_baseline(dataset_path, tmp_path / "reports")

        reloaded_retrieval = json.loads(
            Path(result["stages"]["retrieval"]["report_json_path"]).read_text(encoding="utf-8")
        )
        reloaded_answers = json.loads(
            Path(result["stages"]["answers"]["report_json_path"]).read_text(encoding="utf-8")
        )

        assert reloaded_retrieval["metrics"]["recall@10"] == result["stages"]["retrieval"]["metrics"]["recall@10"]
        assert reloaded_answers["source"]["any_hit_rate"] == result["stages"]["answers"]["source"]["any_hit_rate"]
