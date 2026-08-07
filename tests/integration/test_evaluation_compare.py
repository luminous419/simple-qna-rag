"""Integration tests for evaluation/compare.py (Design.md §5.8)."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation import compare


def _baseline_payload(**overrides) -> dict:
    base = {
        "dataset_sha256": "abc123",
        "corpus_manifest_sha256": "def456",
        "vectorstore_fingerprint": {"index_faiss_sha256": "faiss1", "index_pkl_sha256": "pkl1"},
        "metrics": {"recall@10": 0.9762, "recall@5": 0.9524, "mrr@10": 0.9821, "ndcg@10": 0.9543},
    }
    base.update(overrides)
    return base


def _candidate_payload(**overrides) -> dict:
    base = {
        "dataset_sha256": "abc123",
        "corpus_manifest_sha256": "def456",
        "vectorstore_fingerprint": {"index_faiss_sha256": "faiss1", "index_pkl_sha256": "pkl1"},
        "metrics": {"recall@10": 0.9762, "recall@5": 0.9524, "mrr@10": 0.9821, "ndcg@10": 0.9543},
    }
    base.update(overrides)
    return base


def test_compare_reports_comparable_when_fingerprints_match() -> None:
    result = compare.compare_reports(_baseline_payload(), _candidate_payload(), kind="retrieval")
    assert result["comparable"] is True
    assert result["fingerprint_diff"] == []


def test_compare_reports_not_comparable_on_dataset_mismatch() -> None:
    result = compare.compare_reports(
        _baseline_payload(), _candidate_payload(dataset_sha256="different"), kind="retrieval"
    )
    assert result["comparable"] is False
    assert any(d["field"] == "dataset_sha256" for d in result["fingerprint_diff"])


def test_compare_reports_not_comparable_on_vectorstore_mismatch() -> None:
    result = compare.compare_reports(
        _baseline_payload(),
        _candidate_payload(vectorstore_fingerprint={"index_faiss_sha256": "other", "index_pkl_sha256": "pkl1"}),
        kind="retrieval",
    )
    assert result["comparable"] is False


def test_compare_reports_metric_deltas_for_retrieval() -> None:
    result = compare.compare_reports(
        _baseline_payload(),
        _candidate_payload(metrics={"recall@10": 0.98, "recall@5": 0.95, "mrr@10": 0.99, "ndcg@10": 0.96}),
        kind="retrieval",
    )
    deltas = result["metric_deltas"]
    assert deltas["recall@10"]["delta"] == pytest_approx(0.98 - 0.9762)


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=1e-9)


def test_main_cli_exit_3_on_mismatch(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(_baseline_payload()), encoding="utf-8")
    candidate_path.write_text(json.dumps(_candidate_payload(dataset_sha256="different")), encoding="utf-8")

    exit_code = compare.main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--kind",
            "retrieval",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 3


def test_main_cli_exit_0_on_match(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(_baseline_payload()), encoding="utf-8")
    candidate_path.write_text(json.dumps(_candidate_payload()), encoding="utf-8")

    exit_code = compare.main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--kind",
            "retrieval",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "out" / "compare.json").exists()


def test_schema_1_0_0_report_readable_without_crash() -> None:
    # compare.py must tolerate older (schema 1.0.0) reports missing newer
    # keys instead of raising (Design.md §13.2 migration contract).
    old_baseline = {"dataset_sha256": "abc", "corpus_manifest_sha256": "def"}
    old_candidate = {"dataset_sha256": "abc", "corpus_manifest_sha256": "def"}
    result = compare.compare_reports(old_baseline, old_candidate, kind="retrieval")
    assert result["comparable"] is True


def test_retrieval_gates_accept_current_latency_schema() -> None:
    result = compare.evaluate_gates(
        {
            "retrieval": {
                "warmup": {"performed": True},
                "mmr_instrumentation": {"fallback_case_count": 0},
                "latency_ms": {"mean": 2191.0, "p95": 2369.0},
                "stage_summary": {"mmr": {"latency_ms_mean": 11.0}},
                "metrics": {
                    "recall@10": 0.9762,
                    "recall@5": 0.9524,
                    "mrr@10": 0.9821,
                    "ndcg@10": 0.9543,
                },
            }
        }
    )
    items = {item["id"]: item for item in result["items"]}
    assert items["retrieval_latency_mean_ms"]["metric"] == 2191.0
    assert items["retrieval_latency_p95_ms"]["metric"] == 2369.0
