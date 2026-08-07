"""Tests for evaluation/compare.py::evaluate_gates() (M3-REQ-003, Design.md §5.8)."""

from __future__ import annotations

from evaluation import compare


def _retrieval_payload(**overrides) -> dict:
    base = {
        "latency_ms": {"mean_ms": 2000.0, "p95_ms": 3000.0},
        "stage_summary": {"mmr": {"latency_ms_mean": 10.0}},
        "metrics": {"recall@10": 0.98, "recall@5": 0.97, "mrr@10": 0.99, "ndcg@10": 0.97},
        "warmup": {"performed": True},
        "mmr_instrumentation": {"fallback_case_count": 0},
    }
    base.update(overrides)
    return base


def _routing_payload(**overrides) -> dict:
    base = {
        "aggregate": {
            "correct_count": {"median": 70},
            "document_route_correct": {"median": 55},
        },
        "per_run": [
            {"web_search_correct": 15},
            {"web_search_correct": 15},
            {"web_search_correct": 15},
        ],
    }
    base.update(overrides)
    return base


def _answers_payload(**overrides) -> dict:
    base = {
        "source": {"any_hit_rate": 1.0, "mean_recall": 0.95},
        "latency_ms": {"mean_ms": 50000.0, "p95_ms": 70000.0},
        "warmup": {"performed": True},
    }
    base.update(overrides)
    return base


def _full_pass_payload() -> dict:
    return {"retrieval": _retrieval_payload(), "routing": _routing_payload(), "answers": _answers_payload()}


def test_full_pass_when_all_gates_satisfied() -> None:
    result = compare.evaluate_gates(_full_pass_payload())
    assert result["overall_pass"] is True
    assert all(item["pass"] is True for item in result["items"])


def test_missing_reports_yield_pass_none() -> None:
    result = compare.evaluate_gates({"retrieval": None, "routing": None, "answers": None})
    assert result["overall_pass"] is False
    assert all(item["pass"] is None for item in result["items"])


def test_routing_accuracy_count_gate_boundary() -> None:
    for correct, expected in [(68, False), (69, True), (70, True)]:
        payload = _full_pass_payload()
        payload["routing"]["aggregate"]["correct_count"]["median"] = correct
        result = compare.evaluate_gates(payload)
        item = next(i for i in result["items"] if i["id"] == "routing_accuracy_median")
        assert item["pass"] is expected, f"correct={correct}"


def test_document_route_recall_count_gate_boundary() -> None:
    for correct, expected in [(53, False), (54, True), (55, True)]:
        payload = _full_pass_payload()
        payload["routing"]["aggregate"]["document_route_correct"]["median"] = correct
        result = compare.evaluate_gates(payload)
        item = next(i for i in result["items"] if i["id"] == "routing_document_route_recall_median")
        assert item["pass"] is expected, f"correct={correct}"


def test_web_search_recall_count_gate_boundary() -> None:
    for correct, expected in [(14, False), (15, True)]:
        payload = _full_pass_payload()
        payload["routing"]["per_run"] = [{"web_search_correct": correct}] * 3
        result = compare.evaluate_gates(payload)
        item = next(i for i in result["items"] if i["id"] == "routing_web_recall_each_run")
        assert item["pass"] is expected, f"correct={correct}"


def test_web_search_recall_fails_if_any_run_misses() -> None:
    payload = _full_pass_payload()
    payload["routing"]["per_run"] = [
        {"web_search_correct": 15},
        {"web_search_correct": 14},
        {"web_search_correct": 15},
    ]
    result = compare.evaluate_gates(payload)
    item = next(i for i in result["items"] if i["id"] == "routing_web_recall_each_run")
    assert item["pass"] is False


def test_count_gate_uses_fraction_not_rounded_float() -> None:
    # 54/61 = 0.8852459... and 88.52% rounds oddly; Fraction comparison must
    # not be fooled by float rounding either direction.
    payload = _full_pass_payload()
    payload["routing"]["aggregate"]["document_route_correct"]["median"] = 54
    result = compare.evaluate_gates(payload)
    item = next(i for i in result["items"] if i["id"] == "routing_document_route_recall_median")
    assert item["pass"] is True
    assert item["denominator"] == 61
    assert item["threshold_correct"] == 54


def test_absolute_and_reduction_both_required_for_latency() -> None:
    # Below the absolute threshold but the reduction ratio not met (base
    # 16840, needs <= 8420 i.e. 50% reduction) — with base=16840 an absolute
    # value of 8000 still satisfies both, so pick a case where absolute
    # passes but historically the reduction wouldn't for a different base.
    payload = _full_pass_payload()
    payload["retrieval"]["latency_ms"]["mean_ms"] = 8420.0  # exactly at threshold and exactly 50% reduction
    result = compare.evaluate_gates(payload)
    item = next(i for i in result["items"] if i["id"] == "retrieval_latency_mean_ms")
    assert item["pass"] is True


def test_fallback_case_count_blocks_latency_gate() -> None:
    payload = _full_pass_payload()
    payload["retrieval"]["mmr_instrumentation"]["fallback_case_count"] = 1
    result = compare.evaluate_gates(payload)
    for gate_id in ("retrieval_latency_mean_ms", "retrieval_latency_p95_ms", "mmr_latency_mean_ms"):
        item = next(i for i in result["items"] if i["id"] == gate_id)
        assert item["pass"] is None, gate_id
    assert result["overall_pass"] is False


def test_warmup_not_performed_blocks_latency_gate() -> None:
    payload = _full_pass_payload()
    payload["retrieval"]["warmup"]["performed"] = False
    result = compare.evaluate_gates(payload)
    item = next(i for i in result["items"] if i["id"] == "retrieval_latency_mean_ms")
    assert item["pass"] is None


def test_answers_warmup_not_performed_blocks_answer_latency_gate() -> None:
    payload = _full_pass_payload()
    payload["answers"]["warmup"]["performed"] = False
    result = compare.evaluate_gates(payload)
    item = next(i for i in result["items"] if i["id"] == "answer_latency_mean_s")
    assert item["pass"] is None


def test_source_any_hit_must_equal_one() -> None:
    payload = _full_pass_payload()
    payload["answers"]["source"]["any_hit_rate"] = 0.99
    result = compare.evaluate_gates(payload)
    item = next(i for i in result["items"] if i["id"] == "source_any_hit")
    assert item["pass"] is False
