"""Tests for evaluation/fingerprint.py (Phase 0, M3-REQ-001)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from evaluation import fingerprint


def _fake_current() -> dict:
    return {
        "dataset_sha256": "abc",
        "corpus_manifest_sha256": "def",
        "corpus_file_count": 3,
        "index_faiss_sha256": "ghi",
        "index_pkl_sha256": "jkl",
        "git_commit": "deadbeef",
        "git_dirty": False,
        "python_version": "3.11.13",
    }


def _write_baseline(tmp_path: Path, **overrides) -> Path:
    payload = {
        "baseline_id": "m2_initial",
        "execution": {"dataset_sha256": overrides.get("dataset_sha256", "abc")},
        "reproducibility": {
            "corpus_manifest_sha256": overrides.get("corpus_manifest_sha256", "def"),
            "vectorstore_fingerprint": {
                "index_faiss_sha256": overrides.get("index_faiss_sha256", "ghi"),
                "index_pkl_sha256": overrides.get("index_pkl_sha256", "jkl"),
            },
        },
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compare_with_baseline_match(tmp_path: Path) -> None:
    baseline_path = _write_baseline(tmp_path)
    result = fingerprint.compare_with_baseline(_fake_current(), baseline_path)
    assert result["match"] is True
    assert result["diff"] == []
    assert result["baseline_id"] == "m2_initial"


def test_compare_with_baseline_mismatch(tmp_path: Path) -> None:
    baseline_path = _write_baseline(tmp_path, dataset_sha256="different")
    result = fingerprint.compare_with_baseline(_fake_current(), baseline_path)
    assert result["match"] is False
    assert len(result["diff"]) == 1
    assert result["diff"][0]["field"] == "dataset_sha256"
    assert result["diff"][0]["expected"] == "different"
    assert result["diff"][0]["actual"] == "abc"


def test_compare_with_baseline_multiple_mismatches(tmp_path: Path) -> None:
    baseline_path = _write_baseline(
        tmp_path, dataset_sha256="x", index_faiss_sha256="y"
    )
    result = fingerprint.compare_with_baseline(_fake_current(), baseline_path)
    assert result["match"] is False
    fields = {d["field"] for d in result["diff"]}
    assert fields == {"dataset_sha256", "index_faiss_sha256"}


def test_main_exit_codes(tmp_path: Path) -> None:
    dataset_path = tmp_path / "golden.jsonl"
    dataset_path.write_text('{"id": "x"}\n', encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc.txt").write_text("hello", encoding="utf-8")
    vectorstore_dir = tmp_path / "vectorstore"
    vectorstore_dir.mkdir()
    (vectorstore_dir / "index.faiss").write_bytes(b"faiss-bytes")
    (vectorstore_dir / "index.pkl").write_bytes(b"pkl-bytes")

    with patch.object(fingerprint.config, "DATA_DIR", str(data_dir)), \
         patch.object(fingerprint.config, "VECTORSTORE_PATH", str(vectorstore_dir)):
        # No baseline: just collects and prints, exit 0.
        exit_code = fingerprint.main(["--dataset", str(dataset_path)])
        assert exit_code == 0

        # Baseline mismatch -> exit 3.
        baseline_path = _write_baseline(tmp_path, dataset_sha256="mismatch")
        exit_code = fingerprint.main(
            ["--dataset", str(dataset_path), "--baseline", str(baseline_path)]
        )
        assert exit_code == 3

    # Missing dataset file -> exit 2.
    exit_code = fingerprint.main(["--dataset", str(tmp_path / "missing.jsonl")])
    assert exit_code == 2
