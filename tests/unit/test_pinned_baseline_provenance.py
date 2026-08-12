"""M4.3-REQ-002 — pinned legacy-import constants vs. tracked M3 baseline
(Design.md §4.7, DR-I2-MIN-06/DR-I3-MIN-06/DR-I4-MIN-03)."""

from __future__ import annotations

import json
from pathlib import Path

from simple_qna_rag.index.lifecycle import (
    _PINNED_M3_APPROVED_INDEX_FAISS_SHA256,
    _PINNED_M3_APPROVED_INDEX_PKL_SHA256,
    _parse_m3_baseline_fingerprint,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACKED_BASELINE = _REPO_ROOT / "evaluation" / "baselines" / "m3_initial.json"


def test_pinned_constants_match_tracked_m3_baseline_bytes():
    raw = _TRACKED_BASELINE.read_bytes()
    fingerprint = _parse_m3_baseline_fingerprint(raw)
    assert fingerprint["index_faiss_sha256"] == _PINNED_M3_APPROVED_INDEX_FAISS_SHA256
    assert fingerprint["index_pkl_sha256"] == _PINNED_M3_APPROVED_INDEX_PKL_SHA256


def test_tampered_baseline_copy_diverges_from_pinned_constants(tmp_path):
    raw = bytearray(_TRACKED_BASELINE.read_bytes())
    doc = json.loads(bytes(raw))
    original = doc["reproducibility"]["vectorstore_fingerprint"]["index_faiss_sha256"]
    tampered_hex = ("0" if original[0] != "0" else "1") + original[1:]
    doc["reproducibility"]["vectorstore_fingerprint"]["index_faiss_sha256"] = tampered_hex
    tampered_bytes = json.dumps(doc).encode("utf-8")

    copy_path = tmp_path / "m3_initial_tampered.json"
    copy_path.write_bytes(tampered_bytes)

    fingerprint = _parse_m3_baseline_fingerprint(copy_path.read_bytes())
    assert fingerprint["index_faiss_sha256"] != _PINNED_M3_APPROVED_INDEX_FAISS_SHA256

    # tracked original must remain byte-identical (read-only test — only the
    # tmp_path copy was ever mutated)
    assert _TRACKED_BASELINE.read_bytes() == bytes(raw)
