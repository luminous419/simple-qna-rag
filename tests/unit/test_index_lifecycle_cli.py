"""M4.3-REQ-004 — lifecycle CLI exit code/receipt contract (Design.md §6)."""

from __future__ import annotations

import hashlib
import io
import json
import sys

import pytest

from simple_qna_rag.cli import index_lifecycle as cli


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


def _run_cli(argv, capsys):
    code = cli.main(argv)
    captured = capsys.readouterr()
    receipt = json.loads(captured.out.strip().splitlines()[-1])
    return code, receipt


def test_verify_unknown_version_exits_1_with_domain_receipt(tmp_path, capsys):
    index_root = tmp_path / "root"
    index_root.mkdir()
    code, receipt = _run_cli(
        ["verify", "--version", "0" * 16, "--index-root", str(index_root)], capsys)
    assert code == 1
    assert receipt["outcome"] == "FAIL"
    assert receipt["exit_code"] == 1
    assert receipt["error_code"] in {"version_dir_missing", "current_pointer_unknown_version"}
    assert "Traceback" not in json.dumps(receipt)


def test_import_legacy_rejects_arbitrary_source_hash(tmp_path, capsys):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "index.faiss").write_bytes(b"not-real")
    (source_dir / "index.pkl").write_bytes(b"not-real-either")
    index_root = tmp_path / "root"
    code, receipt = _run_cli(
        ["import-legacy", "--source-dir", str(source_dir), "--index-root", str(index_root)], capsys)
    assert code == 1
    assert receipt["error_code"] == "member_hash_mismatch"


def test_activate_rollback_receipt_schema_and_exit_codes(tmp_path, capsys, monkeypatch):
    from simple_qna_rag.index import lifecycle

    _SNAPSHOT = {
        "embedding_model_name": "BAAI/bge-m3", "embedding_provider": "huggingface",
        "normalize_embeddings": True, "chunk_size": 1000, "chunk_overlap": 200,
    }

    def identity(faiss_bytes, pkl_bytes):
        return {
            "corpus_manifest_sha256": "a" * 64, "source_document_count": 1, "chunk_count": 1,
            "embedding_model_name": _SNAPSHOT["embedding_model_name"], "embedding_model_revision": "unknown",
            "embedding_provider": _SNAPSHOT["embedding_provider"],
            "normalize_embeddings": _SNAPSHOT["normalize_embeddings"],
            "chunk_size": _SNAPSHOT["chunk_size"], "chunk_overlap": _SNAPSHOT["chunk_overlap"],
            "faiss_index_type": "IndexFlatIP", "faiss_dimension": 3,
            "settings_hash": "b" * 64, "dependency_lock_sha256": "c" * 64,
            "builder_git_sha": "d" * 40, "builder_git_dirty": False,
            "index_faiss": {"size_bytes": len(faiss_bytes), "sha256": hashlib.sha256(faiss_bytes).hexdigest()},
            "index_pkl": {"size_bytes": len(pkl_bytes), "sha256": hashlib.sha256(pkl_bytes).hexdigest()},
            "source": "build", "legacy_baseline_id": None,
        }

    index_root = tmp_path / "root"
    faiss_bytes, pkl_bytes = _tiny_index_bytes(b"a")
    manifest = lifecycle.build(index_root, faiss_bytes=faiss_bytes, pkl_bytes=pkl_bytes,
                                identity_fields=identity(faiss_bytes, pkl_bytes))
    v1 = manifest["version_id"]

    code, receipt = _run_cli(
        ["activate", "--to-version", v1, "--index-root", str(index_root)], capsys)
    assert code == 0
    assert receipt["outcome"] == "PASS"
    assert receipt["schema"] == "m43-lifecycle-receipt-v1"
    assert receipt["post_pointer"] == v1

    code2, receipt2 = _run_cli(
        ["rollback", "--to-previous", "--index-root", str(index_root)], capsys)
    assert code2 == 1
    assert receipt2["error_code"] == "no_previous_version"


def test_list_exit_0_with_versions_field(tmp_path, capsys):
    index_root = tmp_path / "root"
    index_root.mkdir()
    code, receipt = _run_cli(["list", "--index-root", str(index_root)], capsys)
    assert code == 0
    assert receipt["versions"] == []


def test_cleanup_dry_run_is_default_non_destructive(tmp_path, capsys):
    index_root = tmp_path / "root"
    index_root.mkdir()
    code, receipt = _run_cli(["cleanup", "--index-root", str(index_root)], capsys)
    assert code == 0
    assert receipt["dry_run"] is True


def test_argparse_usage_error_exits_2():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["not-a-real-command"])
    assert excinfo.value.code == 2
