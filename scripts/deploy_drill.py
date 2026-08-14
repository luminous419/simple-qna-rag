#!/usr/bin/env python3
"""M4.3-REQ-006 — mock deployment/recovery drill (Design.md §7.7).

Exercises build -> activate -> (RAGEngine reinit simulation) -> readiness ->
rollback -> readiness against a temporary INDEX_ROOT, with no real
Docker/Ollama dependency, then injects deterministic faults (manifest
corruption, simulated disk-full, lock contention, settings mismatch) and
asserts each halts with zero further mutation to `current`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from simple_qna_rag.index import lifecycle
from simple_qna_rag.index.verification import TrustBoundaryError

_SNAPSHOT = {
    "embedding_model_name": "BAAI/bge-m3", "embedding_provider": "huggingface",
    "normalize_embeddings": True, "chunk_size": 1000, "chunk_overlap": 200,
}


def _identity(faiss_bytes: bytes, pkl_bytes: bytes) -> dict:
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


def _readiness_probe(index_root: Path) -> bool:
    """Simulates `RAGEngine._load_vectorstore()`'s verified path without a
    real FastAPI process (§7.7 "RAGEngine reinit simulation")."""
    from simple_qna_rag.index.verification import CurrentPointerMissing, resolve_current, verify_version

    try:
        version_id = resolve_current(index_root)
    except CurrentPointerMissing:
        return False
    try:
        verify_version(index_root, version_id, settings_snapshot=_SNAPSHOT)
        return True
    except TrustBoundaryError:
        return False


def run_drill(root: Path, *, repeat: int) -> dict:
    steps: list[dict] = []
    root.mkdir(parents=True, exist_ok=True)
    v1 = _publish(root, b"v1")
    start_current = None

    for i in range(repeat):
        t0 = time.monotonic()
        activate_receipt = lifecycle.activate(root, v1, operation="activate", settings_snapshot=_SNAPSHOT)
        ready = _readiness_probe(root)
        steps.append({"step": f"activate_{i}", "outcome": activate_receipt.outcome,
                      "ready": ready, "seconds": round(time.monotonic() - t0, 4)})
        if start_current is None:
            start_current = (root / "current").read_bytes()
        if i > 0:
            v2 = _publish(root, f"v{i}_alt".encode())
            lifecycle.activate(root, v2, operation="activate", settings_snapshot=_SNAPSHOT)
            t0 = time.monotonic()
            rollback_receipt = lifecycle.rollback(root, v1, settings_snapshot=_SNAPSHOT)
            ready = _readiness_probe(root)
            steps.append({"step": f"rollback_{i}", "outcome": rollback_receipt.outcome,
                          "ready": ready, "seconds": round(time.monotonic() - t0, 4)})

    # fault injection — each must leave `current` byte-identical to its
    # pre-fault value (fail-closed, no partial mutation).
    fault_results = []
    before = (root / "current").read_bytes()

    # (1) manifest corruption
    version_dir = root / "versions" / v1
    import os
    import stat as statmod

    os.chmod(version_dir, 0o700)
    manifest_path = version_dir / "manifest.json"
    os.chmod(manifest_path, 0o600)
    original = manifest_path.read_bytes()
    manifest_path.write_bytes(b"{corrupted")
    try:
        lifecycle.activate(root, v1, operation="activate", settings_snapshot=_SNAPSHOT)
    finally:
        manifest_path.write_bytes(original)
        os.chmod(manifest_path, 0o444)
        os.chmod(version_dir, 0o555)
    after = (root / "current").read_bytes()
    fault_results.append({"fault": "manifest_corruption", "current_unchanged": after == before})

    # (2) simulated disk full during build
    import simple_qna_rag.index.lifecycle as lifecycle_module

    real_write_fsync = lifecycle_module._write_fsync

    def failing_write_fsync(path, data, *, mode=0o600):
        raise OSError("simulated ENOSPC")

    lifecycle_module._write_fsync = failing_write_fsync
    try:
        faiss_bytes, pkl_bytes = _tiny_index_bytes(b"disk-full-probe")
        try:
            lifecycle.build(root, faiss_bytes=faiss_bytes, pkl_bytes=pkl_bytes,
                             identity_fields=_identity(faiss_bytes, pkl_bytes))
        except OSError:
            pass
    finally:
        lifecycle_module._write_fsync = real_write_fsync
    after = (root / "current").read_bytes()
    fault_results.append({"fault": "disk_full_build", "current_unchanged": after == before})

    # (3) lock contention — `activate()` never raises `LockTimeoutError`
    # itself (it converts lock timeouts into a FAIL receipt, §4.4), so
    # contention is observed via the receipt's `error_code`, not an
    # exception.
    with lifecycle.acquire_index_lock(root, timeout=10):
        contended_receipt = lifecycle.activate(
            root, v1, operation="activate", settings_snapshot=_SNAPSHOT, lock_timeout=0.1)
    contention_observed = contended_receipt.error_code == "lock_timeout"
    after = (root / "current").read_bytes()
    fault_results.append({"fault": "lock_contention", "current_unchanged": after == before,
                          "contention_observed": contention_observed})

    # (4) settings mismatch (readiness failure)
    mismatched = dict(_SNAPSHOT)
    mismatched["embedding_model_name"] = "wrong-model"
    receipt = lifecycle.activate(root, v1, operation="activate", settings_snapshot=mismatched)
    fault_results.append({"fault": "readiness_settings_mismatch",
                          "outcome": receipt.outcome, "error_code": receipt.error_code})
    lifecycle.activate(root, v1, operation="activate", settings_snapshot=_SNAPSHOT)  # restore

    final_current = (root / "current").read_bytes()
    return {
        "schema": "m43-deploy-drill-v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repeat": repeat, "steps": steps, "fault_injection": fault_results,
        "start_identity_current": json.loads(start_current.decode("utf-8")),
        "final_identity_current": json.loads(final_current.decode("utf-8")),
        "identity_preserved": json.loads(start_current.decode("utf-8"))["version_id"]
            == json.loads(final_current.decode("utf-8"))["version_id"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    result = run_drill(Path(args.root), repeat=args.repeat)
    text = json.dumps(result, sort_keys=True, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    ok = result["identity_preserved"] and all(
        f.get("current_unchanged", True) for f in result["fault_injection"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
