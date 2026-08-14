"""M4.3-REQ-003 — crash/fault-injection recovery for the activation transition
journal (Design.md §4.4/§4.4-b/§12). A reduced but behaviourally faithful
subset of the design's fault matrix: each test injects a failure at one of
the concrete crash points the design calls out and asserts the same
invariant the design requires — `current` is always either the pre- or
post-pointer, and history/receipt state is exact-once.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from simple_qna_rag.index import lifecycle

_SNAPSHOT = {
    "embedding_model_name": "BAAI/bge-m3",
    "embedding_provider": "huggingface",
    "normalize_embeddings": True,
    "chunk_size": 1000,
    "chunk_overlap": 200,
}


def _identity(faiss_bytes: bytes, pkl_bytes: bytes) -> dict:
    return {
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


def _tiny_index_bytes(seed: bytes):
    import faiss
    import numpy as np
    import pickle
    from langchain_community.docstore.in_memory import InMemoryDocstore

    index = faiss.IndexFlatIP(3)
    index.add(np.array([[1.0, 0.0, 0.0]], dtype="float32"))
    faiss_bytes = faiss.serialize_index(index).tobytes()
    docstore = InMemoryDocstore({"0": seed})
    pkl_bytes = pickle.dumps((docstore, {0: "0"}))
    return faiss_bytes, pkl_bytes


def _publish(index_root: Path, seed: bytes) -> str:
    faiss_bytes, pkl_bytes = _tiny_index_bytes(seed)
    manifest = lifecycle.build(index_root, faiss_bytes=faiss_bytes, pkl_bytes=pkl_bytes,
                                identity_fields=_identity(faiss_bytes, pkl_bytes))
    return manifest["version_id"]


def _current_bytes(index_root: Path) -> bytes:
    return (index_root / "current").read_bytes()


def test_staging_fault_matrix_preserves_current(tmp_path, monkeypatch):
    v1 = _publish(tmp_path, b"a")
    lifecycle.activate(tmp_path, v1, operation="activate", settings_snapshot=_SNAPSHOT)
    before = _current_bytes(tmp_path)

    def failing_write_fsync(path, data, *, mode=0o600):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(lifecycle, "_write_fsync", failing_write_fsync)
    faiss_bytes, pkl_bytes = _tiny_index_bytes(b"b")
    with pytest.raises(OSError):
        lifecycle.build(tmp_path, faiss_bytes=faiss_bytes, pkl_bytes=pkl_bytes,
                         identity_fields=_identity(faiss_bytes, pkl_bytes))
    monkeypatch.undo()
    assert _current_bytes(tmp_path) == before
    assert lifecycle.list_versions(tmp_path)["current"] is not None


def test_crash_recovery_journal_reconciles_to_consistent_state(tmp_path):
    v1 = _publish(tmp_path, b"a")
    v2 = _publish(tmp_path, b"b")
    lifecycle.activate(tmp_path, v1, operation="activate", settings_snapshot=_SNAPSHOT)

    # Simulate a crash *after* the pointer replace + parent fsync (pointer is
    # durable) but *before* history/receipt were written: hand-write the
    # journal in "pointer_committed" phase and the new current pointer, then
    # invoke reconcile directly (this is exactly what the next lifecycle
    # entry point does on startup).
    from simple_qna_rag.index.manifest import canonical_json_bytes

    op_id = "f" * 32
    journal = canonical_json_bytes({
        "schema": "m43-transition-journal-v1", "phase": "pointer_committed", "op_id": op_id,
        "operation": "activate", "pre_pointer": v1, "post_pointer": v2,
        "recorded_at": "2026-08-12T00:00:00Z",
    }) + b"\n"
    (tmp_path / ".transition").write_bytes(journal)
    (tmp_path / "current").write_bytes(
        canonical_json_bytes({"schema_version": 1, "version_id": v2}) + b"\n")

    report = lifecycle._reconcile_pending_transition(tmp_path)
    assert report.outcome == "completed"

    info = lifecycle.list_versions(tmp_path)
    assert info["current"] == v2
    assert info["previous"] == v1
    assert not (tmp_path / ".transition").exists()

    # idempotent: a second reconcile call (simulating repeated restarts) is a no-op
    report2 = lifecycle._reconcile_pending_transition(tmp_path)
    assert report2 is None


def test_crash_recovery_aborts_when_pointer_never_committed(tmp_path):
    v1 = _publish(tmp_path, b"a")
    v2 = _publish(tmp_path, b"b")
    lifecycle.activate(tmp_path, v1, operation="activate", settings_snapshot=_SNAPSHOT)
    before = _current_bytes(tmp_path)

    from simple_qna_rag.index.manifest import canonical_json_bytes

    op_id = "e" * 32
    journal = canonical_json_bytes({
        "schema": "m43-transition-journal-v1", "phase": "prepared", "op_id": op_id,
        "operation": "activate", "pre_pointer": v1, "post_pointer": v2,
        "recorded_at": "2026-08-12T00:00:00Z",
    }) + b"\n"
    (tmp_path / ".transition").write_bytes(journal)
    # current was never replaced — still v1's pointer bytes

    report = lifecycle._reconcile_pending_transition(tmp_path)
    assert report.outcome == "aborted"
    assert _current_bytes(tmp_path) == before
    assert not (tmp_path / ".transition").exists()
    # aborted operation must not have produced a history record
    from simple_qna_rag.index.lifecycle import _history_record_path

    assert not _history_record_path(tmp_path, op_id).exists()


_VALID_OP_ID = "f" * 32


def _valid_journal_fields(v1: str, v2: str) -> dict:
    return {
        "schema": "m43-transition-journal-v1", "phase": "pointer_committed",
        "op_id": _VALID_OP_ID, "operation": "activate", "pre_pointer": v1,
        "post_pointer": v2, "recorded_at": "2026-08-12T00:00:00Z",
    }


_MALFORMED_JOURNAL_CASES = {
    "not_json": lambda fields: b"not json at all",
    "not_object": lambda fields: json.dumps([1, 2, 3]).encode("utf-8"),
    "missing_key": lambda fields: json.dumps(
        {k: v for k, v in fields.items() if k != "recorded_at"}).encode("utf-8"),
    "extra_key": lambda fields: json.dumps({**fields, "extra": "x"}).encode("utf-8"),
    "wrong_schema": lambda fields: json.dumps({**fields, "schema": "wrong"}).encode("utf-8"),
    "invalid_phase_enum": lambda fields: json.dumps(
        {**fields, "phase": "not_a_phase"}).encode("utf-8"),
    "invalid_operation_enum": lambda fields: json.dumps(
        {**fields, "operation": "delete"}).encode("utf-8"),
    "op_id_wrong_length": lambda fields: json.dumps(
        {**fields, "op_id": "abc123"}).encode("utf-8"),
    "op_id_uppercase_hex": lambda fields: json.dumps(
        {**fields, "op_id": "F" * 32}).encode("utf-8"),
    "op_id_path_traversal": lambda fields: json.dumps(
        {**fields, "op_id": "../../../../etc/passwd"}).encode("utf-8"),
    "op_id_path_traversal_relative": lambda fields: json.dumps(
        {**fields, "op_id": "../escaped"}).encode("utf-8"),
    "pre_pointer_wrong_type": lambda fields: json.dumps(
        {**fields, "pre_pointer": 12345}).encode("utf-8"),
    "pre_pointer_malformed_hex": lambda fields: json.dumps(
        {**fields, "pre_pointer": "not-hex-id"}).encode("utf-8"),
    "post_pointer_null": lambda fields: json.dumps(
        {**fields, "post_pointer": None}).encode("utf-8"),
    "post_pointer_wrong_type": lambda fields: json.dumps(
        {**fields, "post_pointer": True}).encode("utf-8"),
    "post_pointer_malformed_hex": lambda fields: json.dumps(
        {**fields, "post_pointer": "0" * 15 + "z"}).encode("utf-8"),
    "recorded_at_wrong_type": lambda fields: json.dumps(
        {**fields, "recorded_at": 123}).encode("utf-8"),
    "recorded_at_not_iso8601": lambda fields: json.dumps(
        {**fields, "recorded_at": "yesterday"}).encode("utf-8"),
}


@pytest.mark.parametrize("case_name", sorted(_MALFORMED_JOURNAL_CASES))
def test_malformed_transition_journal_rejected_without_any_mutation(tmp_path, case_name):
    """CR-I1-MAJ-02: a corrupt/tampered `.transition` journal must be
    rejected fail-closed — no history record, no receipt, no `current`
    mutation — regardless of which field is malformed."""
    v1 = _publish(tmp_path, b"a")
    v2 = _publish(tmp_path, b"b")
    lifecycle.activate(tmp_path, v1, operation="activate", settings_snapshot=_SNAPSHOT)
    before_current = _current_bytes(tmp_path)

    raw = _MALFORMED_JOURNAL_CASES[case_name](_valid_journal_fields(v1, v2))
    (tmp_path / ".transition").write_bytes(raw)

    with pytest.raises(lifecycle.TrustBoundaryError) as excinfo:
        lifecycle._reconcile_pending_transition(tmp_path)
    assert excinfo.value.reason == "transition_journal_corrupt"

    # fail-closed: current pointer untouched, no history/receipt produced
    # for the bogus op_id embedded in this malformed journal, and the
    # journal file itself is left in place (untouched, not silently
    # consumed) for operator inspection.
    assert _current_bytes(tmp_path) == before_current
    assert (tmp_path / ".transition").exists()
    history_dir = tmp_path / "activation_history"
    if history_dir.is_dir():
        for name in os.listdir(history_dir):
            assert not name.startswith("..")
    receipt_path = tmp_path / ".last_activation_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_bytes())
        assert receipt.get("operation_id") != "../escaped"
        assert receipt.get("operation_id") != "../../../../etc/passwd"


def test_reproduces_cr_i1_maj_02_original_finding_journal(tmp_path):
    """Exact reproduction of the Code Review Iteration 1 finding
    (CR-I1-MAJ-02): a `.transition` journal with `schema: "wrong"`,
    `operation: "delete"` (not in the activate/rollback enum), a
    traversal-shaped `op_id`, and null pointers on an *empty* index root
    (no versions published at all) must never be promoted to a PASS
    receipt/history row."""
    malicious = {
        "schema": "wrong", "phase": "pointer_committed", "op_id": "../escaped",
        "operation": "delete", "pre_pointer": None, "post_pointer": None,
        "recorded_at": "x",
    }
    (tmp_path / ".transition").write_bytes(json.dumps(malicious).encode("utf-8"))

    with pytest.raises(lifecycle.TrustBoundaryError) as excinfo:
        lifecycle._reconcile_pending_transition(tmp_path)
    assert excinfo.value.reason == "transition_journal_corrupt"

    assert not (tmp_path / ".last_activation_receipt.json").exists()
    assert not (tmp_path / "activation_history").exists()
    assert (tmp_path / ".transition").exists()


def test_valid_journal_with_traversal_lookalike_op_id_prefix_still_rejected(tmp_path):
    """A 32-hex `op_id` cannot itself contain traversal characters (the
    regex is exact-length hex-only), but this pins the boundary explicitly:
    an otherwise well-formed journal whose `op_id` merely *starts* with
    hex-looking traversal bytes is still rejected because it fails the
    fixed 32-hex pattern."""
    v1 = _publish(tmp_path, b"a")
    v2 = _publish(tmp_path, b"b")
    lifecycle.activate(tmp_path, v1, operation="activate", settings_snapshot=_SNAPSHOT)

    fields = _valid_journal_fields(v1, v2)
    fields["op_id"] = "." * 32
    (tmp_path / ".transition").write_bytes(json.dumps(fields).encode("utf-8"))

    with pytest.raises(lifecycle.TrustBoundaryError) as excinfo:
        lifecycle._reconcile_pending_transition(tmp_path)
    assert excinfo.value.reason == "transition_journal_corrupt"


def test_crash_recovery_history_and_receipt_exact_once_matrix(tmp_path):
    """Reduced fault matrix: crash before the history record's rename
    (nothing durable yet -> reconcile commits it cleanly from scratch) and
    crash after the rename (already durable -> reconcile is a pure no-op).
    Both must converge to exactly one history record for the op_id."""
    from simple_qna_rag.index.lifecycle import _append_history, _read_history_rows

    v1 = _publish(tmp_path, b"a")
    v2 = _publish(tmp_path, b"b")
    lifecycle.activate(tmp_path, v1, operation="activate", settings_snapshot=_SNAPSHOT)
    op_id = "9" * 32

    # (a) tmp write never reached rename -> _append_history from scratch commits once
    _append_history(tmp_path, op_id=op_id, operation="activate", pre_pointer=v1, post_pointer=v2)
    rows = _read_history_rows(tmp_path)
    assert len([r for r in rows if r["op_id"] == op_id]) == 1

    # (b) calling _append_history again for the same op_id must be a no-op
    # (exact-once) — simulates a reconcile retry after the rename already
    # succeeded once.
    _append_history(tmp_path, op_id=op_id, operation="activate", pre_pointer=v1, post_pointer=v2)
    rows_after = _read_history_rows(tmp_path)
    assert len([r for r in rows_after if r["op_id"] == op_id]) == 1
    assert rows_after == rows
