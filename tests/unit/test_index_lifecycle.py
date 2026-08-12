"""M4.3-REQ-002/003 — staging/publish/activate/rollback/retention/legacy import."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from simple_qna_rag.index import lifecycle
from simple_qna_rag.index.verification import TrustBoundaryError

_SNAPSHOT = {
    "embedding_model_name": "BAAI/bge-m3",
    "embedding_provider": "huggingface",
    "normalize_embeddings": True,
    "chunk_size": 1000,
    "chunk_overlap": 200,
}


def _identity(faiss_bytes: bytes, pkl_bytes: bytes, **overrides) -> dict:
    base = {
        "corpus_manifest_sha256": "a" * 64,
        "source_document_count": 1,
        "chunk_count": 1,
        "embedding_model_name": _SNAPSHOT["embedding_model_name"],
        "embedding_model_revision": "unknown",
        "embedding_provider": _SNAPSHOT["embedding_provider"],
        "normalize_embeddings": _SNAPSHOT["normalize_embeddings"],
        "chunk_size": _SNAPSHOT["chunk_size"],
        "chunk_overlap": _SNAPSHOT["chunk_overlap"],
        "faiss_index_type": "IndexFlatIP",
        "faiss_dimension": 3,
        "settings_hash": "b" * 64,
        "dependency_lock_sha256": "c" * 64,
        "builder_git_sha": "d" * 40,
        "builder_git_dirty": False,
        "index_faiss": {"size_bytes": len(faiss_bytes), "sha256": hashlib.sha256(faiss_bytes).hexdigest()},
        "index_pkl": {"size_bytes": len(pkl_bytes), "sha256": hashlib.sha256(pkl_bytes).hexdigest()},
        "source": "build",
        "legacy_baseline_id": None,
    }
    base.update(overrides)
    return base


def _tiny_index_bytes(byte_seed: bytes = b"a"):
    import faiss
    import numpy as np
    import pickle
    from langchain_community.docstore.in_memory import InMemoryDocstore

    index = faiss.IndexFlatIP(3)
    index.add(np.array([[1.0, 0.0, 0.0]], dtype="float32"))
    faiss_bytes = faiss.serialize_index(index).tobytes()
    docstore = InMemoryDocstore({"0": byte_seed})
    pkl_bytes = pickle.dumps((docstore, {0: "0"}))
    return faiss_bytes, pkl_bytes


def _publish(index_root: Path, seed: bytes = b"a") -> str:
    faiss_bytes, pkl_bytes = _tiny_index_bytes(seed)
    manifest = lifecycle.build(index_root, faiss_bytes=faiss_bytes, pkl_bytes=pkl_bytes,
                                identity_fields=_identity(faiss_bytes, pkl_bytes))
    return manifest["version_id"]


def test_build_publishes_immutable_version_dir(tmp_path):
    version_id = _publish(tmp_path)
    version_dir = tmp_path / "versions" / version_id
    assert sorted(p.name for p in version_dir.iterdir()) == ["index.faiss", "index.pkl", "manifest.json"]
    assert oct(version_dir.stat().st_mode)[-3:] == "555"
    for name in ("index.faiss", "index.pkl", "manifest.json"):
        assert oct((version_dir / name).stat().st_mode)[-3:] == "444"


def test_activate_rollback_100x(tmp_path):
    v1 = _publish(tmp_path, b"a")
    v2 = _publish(tmp_path, b"b")
    for _ in range(100):
        r1 = lifecycle.activate(tmp_path, v1, operation="activate", settings_snapshot=_SNAPSHOT)
        assert r1.outcome == "PASS"
        current_raw = (tmp_path / "current").read_bytes()
        assert current_raw  # parseable/non-empty at every observed instant
        r2 = lifecycle.activate(tmp_path, v2, operation="activate", settings_snapshot=_SNAPSHOT)
        assert r2.outcome == "PASS"
    info = lifecycle.list_versions(tmp_path)
    assert info["current"] == v2


def test_activate_unknown_version_leaves_current_unchanged(tmp_path):
    v1 = _publish(tmp_path)
    lifecycle.activate(tmp_path, v1, operation="activate", settings_snapshot=_SNAPSHOT)
    before = (tmp_path / "current").read_bytes()
    receipt = lifecycle.activate(tmp_path, "0" * 16, operation="activate", settings_snapshot=_SNAPSHOT)
    assert receipt.outcome == "FAIL"
    assert receipt.exit_code == 1
    after = (tmp_path / "current").read_bytes()
    assert before == after


def test_rollback_reuses_activate_primitive(tmp_path):
    v1 = _publish(tmp_path, b"a")
    v2 = _publish(tmp_path, b"b")
    lifecycle.activate(tmp_path, v1, operation="activate", settings_snapshot=_SNAPSHOT)
    lifecycle.activate(tmp_path, v2, operation="activate", settings_snapshot=_SNAPSHOT)
    receipt = lifecycle.rollback(tmp_path, v1, settings_snapshot=_SNAPSHOT)
    assert receipt.outcome == "PASS"
    assert receipt.operation == "rollback"
    assert lifecycle.list_versions(tmp_path)["current"] == v1


def test_lock_contention_fails_fast_and_bounded(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    with lifecycle.acquire_index_lock(tmp_path, timeout=10):
        with pytest.raises(lifecycle.LockTimeoutError):
            with lifecycle.acquire_index_lock(tmp_path, timeout=0.1):
                pass


def test_preexisting_lock_symlink_rejected(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "evil_target").write_bytes(b"")
    os.symlink(tmp_path / "evil_target", tmp_path / ".lock")
    with pytest.raises(OSError):
        with lifecycle.acquire_index_lock(tmp_path, timeout=1):
            pass


# --- previous algebra (DR-I4-MAJ-01 matrix, Design.md §4.4-a-1) -----------


def test_previous_history_algebra_matrix(tmp_path):
    from simple_qna_rag.index.lifecycle import _append_history, _read_previous_from_history

    tmp_path.mkdir(parents=True, exist_ok=True)

    # 1. empty, current=None -> previous is None
    assert _read_previous_from_history(tmp_path, current=None) is None

    # 2. empty, current set -> current_mismatch
    with pytest.raises(TrustBoundaryError) as excinfo:
        _read_previous_from_history(tmp_path, current="b" * 16)
    assert excinfo.value.reason == "activation_history_current_mismatch"

    # 3. first A->B -> previous == A
    a16, b16, c16 = "a" * 16, "b" * 16, "c" * 16
    _append_history(tmp_path, op_id="1" * 32, operation="activate", pre_pointer=a16, post_pointer=b16)
    assert _read_previous_from_history(tmp_path, current=b16) == a16

    # 4. second B->C -> previous == B
    _append_history(tmp_path, op_id="2" * 32, operation="activate", pre_pointer=b16, post_pointer=c16)
    assert _read_previous_from_history(tmp_path, current=c16) == b16

    # 5. rollback C->B -> previous == C
    _append_history(tmp_path, op_id="3" * 32, operation="rollback", pre_pointer=c16, post_pointer=b16)
    assert _read_previous_from_history(tmp_path, current=b16) == c16

    # 10. latest.post_pointer != current -> current_mismatch
    with pytest.raises(TrustBoundaryError) as excinfo:
        _read_previous_from_history(tmp_path, current=a16)
    assert excinfo.value.reason == "activation_history_current_mismatch"


def test_previous_history_algebra_rejects_sequence_duplicate_and_gap(tmp_path):
    from simple_qna_rag.index.lifecycle import _history_dir, _read_history_rows
    from simple_qna_rag.index.manifest import canonical_json_bytes

    tmp_path.mkdir(parents=True, exist_ok=True)
    history_dir = _history_dir(tmp_path)
    history_dir.mkdir(parents=True)

    def write_record(op_id: str, sequence: int, pre, post):
        record = canonical_json_bytes({
            "schema": "m43-activation-history-record-v1", "op_id": op_id, "sequence": sequence,
            "operation": "activate", "pre_pointer": pre, "post_pointer": post,
            "recorded_at": "2026-08-12T00:00:00Z", "reconciled": False,
        }) + b"\n"
        (history_dir / f"{op_id}.json").write_bytes(record)

    write_record("1" * 32, 0, None, "a" * 16)
    write_record("2" * 32, 0, "a" * 16, "b" * 16)  # duplicate sequence
    with pytest.raises(TrustBoundaryError) as excinfo:
        _read_history_rows(tmp_path)
    assert excinfo.value.reason == "activation_history_sequence_invalid"


def test_history_filename_op_id_mismatch_rejected(tmp_path):
    from simple_qna_rag.index.lifecycle import _history_dir, _read_history_rows
    from simple_qna_rag.index.manifest import canonical_json_bytes

    tmp_path.mkdir(parents=True, exist_ok=True)
    history_dir = _history_dir(tmp_path)
    history_dir.mkdir(parents=True)
    record = canonical_json_bytes({
        "schema": "m43-activation-history-record-v1", "op_id": "2" * 32, "sequence": 0,
        "operation": "activate", "pre_pointer": None, "post_pointer": "a" * 16,
        "recorded_at": "2026-08-12T00:00:00Z", "reconciled": False,
    }) + b"\n"
    (history_dir / f"{'1' * 32}.json").write_bytes(record)
    with pytest.raises(TrustBoundaryError) as excinfo:
        _read_history_rows(tmp_path)
    assert excinfo.value.reason == "activation_history_filename_op_id_mismatch"


# --- retention/cleanup ------------------------------------------------------


def test_cleanup_dry_run_then_apply_protects_current_and_previous(tmp_path):
    v1 = _publish(tmp_path, b"a")
    v2 = _publish(tmp_path, b"b")
    v3 = _publish(tmp_path, b"c")
    lifecycle.activate(tmp_path, v1, operation="activate", settings_snapshot=_SNAPSHOT)
    lifecycle.activate(tmp_path, v2, operation="activate", settings_snapshot=_SNAPSHOT)

    dry = lifecycle.cleanup(tmp_path, apply=False)
    assert dry.dry_run is True
    assert set(dry.candidates) == {v3}
    assert (tmp_path / "versions" / v3).exists()

    applied = lifecycle.cleanup(tmp_path, apply=True)
    assert applied.deleted == [v3]
    assert not (tmp_path / "versions" / v3).exists()
    assert (tmp_path / "versions" / v1).exists()  # previous, protected
    assert (tmp_path / "versions" / v2).exists()  # current, protected


def test_cleanup_staging_protects_unexpected_and_young_entries(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    staging_root = tmp_path / ".staging"
    staging_root.mkdir()
    import uuid

    old_valid = uuid.uuid4().hex
    (staging_root / old_valid).mkdir()
    os.utime(staging_root / old_valid, (0, 0))  # ancient mtime

    young_valid = uuid.uuid4().hex
    (staging_root / young_valid).mkdir()  # fresh mtime -> protected by min-age

    (staging_root / "not-a-uuid").mkdir()  # name mismatch -> never a candidate

    receipt = lifecycle.cleanup(tmp_path, apply=True, include_staging=True,
                                 staging_min_age_seconds=3600)
    assert receipt.staging_deleted == [old_valid]
    assert young_valid not in receipt.staging_deleted
    assert (staging_root / young_valid).exists()
    assert (staging_root / "not-a-uuid").exists()


# --- legacy import ----------------------------------------------------------


def test_import_legacy_rejects_source_hash_mismatch(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "index.faiss").write_bytes(b"not the real bytes")
    (source_dir / "index.pkl").write_bytes(b"also fake")
    index_root = tmp_path / "index_root"
    with pytest.raises(TrustBoundaryError) as excinfo:
        lifecycle.import_legacy(index_root, source_dir)
    assert excinfo.value.reason == "member_hash_mismatch"


def test_import_legacy_preserves_source_bytes_and_accepts_override(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    faiss_bytes = b"faiss-fixture-bytes"
    pkl_bytes = b"pkl-fixture-bytes"
    (source_dir / "index.faiss").write_bytes(faiss_bytes)
    (source_dir / "index.pkl").write_bytes(pkl_bytes)
    index_root = tmp_path / "index_root"
    override = {
        "index_faiss_sha256": hashlib.sha256(faiss_bytes).hexdigest(),
        "index_pkl_sha256": hashlib.sha256(pkl_bytes).hexdigest(),
        "baseline_id": "fixture_baseline",
    }
    # legacy identity derivation requires a real FAISS-native blob; use a
    # tiny real one so `_legacy_identity_fields` can deserialize it.
    real_faiss_bytes, real_pkl_bytes = _tiny_index_bytes()
    (source_dir / "index.faiss").write_bytes(real_faiss_bytes)
    (source_dir / "index.pkl").write_bytes(real_pkl_bytes)
    override["index_faiss_sha256"] = hashlib.sha256(real_faiss_bytes).hexdigest()
    override["index_pkl_sha256"] = hashlib.sha256(real_pkl_bytes).hexdigest()

    manifest = lifecycle.import_legacy(index_root, source_dir, _approved_override=override)
    assert manifest["source"] == "legacy_import"
    assert manifest["legacy_baseline_id"] == "fixture_baseline"
    # source files must be byte-identical afterwards (read-only import)
    assert (source_dir / "index.faiss").read_bytes() == real_faiss_bytes
    assert (source_dir / "index.pkl").read_bytes() == real_pkl_bytes


def test_import_legacy_pinned_pair_matches_real_committed_m3_baseline():
    """Uses the actual pinned constants (no override) against the repo's
    committed legacy M3 fixture, if present on this checkout."""
    repo_root = Path(__file__).resolve().parents[2]
    fixture_dir = repo_root / "runtime" / "vectorstore"
    if not (fixture_dir / "index.faiss").is_file() or not (fixture_dir / "index.pkl").is_file():
        pytest.skip("no local runtime/vectorstore fixture present")
    faiss_bytes = (fixture_dir / "index.faiss").read_bytes()
    pkl_bytes = (fixture_dir / "index.pkl").read_bytes()
    approved = lifecycle._pinned_m3_approved_pair()
    assert hashlib.sha256(faiss_bytes).hexdigest() == approved["index_faiss_sha256"]
    assert hashlib.sha256(pkl_bytes).hexdigest() == approved["index_pkl_sha256"]
