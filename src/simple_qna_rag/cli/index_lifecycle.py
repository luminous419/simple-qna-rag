#!/usr/bin/env python3
"""M4.3-REQ-004 — `simple-qna-rag-index-lifecycle` CLI (Design.md §6).

Subcommands: build, import-legacy, verify, activate, rollback, list, cleanup.
Exit codes: 0=PASS, 1=domain failure (trust boundary/hash/schema), 2=usage
error (argparse standard), 3=lock contention/timeout. Every receipt is
canonical JSON with a fixed-vocabulary `error_code` — no exception text,
absolute path, or document content is ever written to stdout/stderr.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

from simple_qna_rag.index import lifecycle
from simple_qna_rag.index.manifest import ManifestError, canonical_json_bytes
from simple_qna_rag.index.verification import TrustBoundaryError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simple-qna-rag-index-lifecycle",
        description="Canonical versioned FAISS index lifecycle (M4.3).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build")
    build_p.add_argument("--documents-dir")
    build_p.add_argument("--index-root")
    build_p.add_argument("--lock-timeout-seconds", type=float, default=10.0)

    import_p = sub.add_parser("import-legacy")
    import_p.add_argument("--source-dir", required=True)
    import_p.add_argument("--index-root")
    import_p.add_argument("--lock-timeout-seconds", type=float, default=10.0)

    verify_p = sub.add_parser("verify")
    verify_p.add_argument("--version", required=True)
    verify_p.add_argument("--index-root")
    verify_p.add_argument("--reconcile-check", action="store_true")

    activate_p = sub.add_parser("activate")
    activate_p.add_argument("--to-version", required=True)
    activate_p.add_argument("--index-root")
    activate_p.add_argument("--lock-timeout-seconds", type=float, default=10.0)

    rollback_p = sub.add_parser("rollback")
    rollback_group = rollback_p.add_mutually_exclusive_group(required=True)
    rollback_group.add_argument("--to-version")
    rollback_group.add_argument("--to-previous", action="store_true")
    rollback_p.add_argument("--index-root")
    rollback_p.add_argument("--lock-timeout-seconds", type=float, default=10.0)

    list_p = sub.add_parser("list")
    list_p.add_argument("--index-root")

    cleanup_p = sub.add_parser("cleanup")
    cleanup_group = cleanup_p.add_mutually_exclusive_group()
    cleanup_group.add_argument("--dry-run", action="store_true")
    cleanup_group.add_argument("--apply", action="store_true")
    cleanup_p.add_argument("--protect", action="append", default=[])
    cleanup_p.add_argument("--include-staging", action="store_true")
    cleanup_p.add_argument("--staging-min-age-seconds", type=int, default=3600)
    cleanup_p.add_argument("--index-root")
    cleanup_p.add_argument("--lock-timeout-seconds", type=float, default=10.0)

    return parser


def _resolve_index_root(raw: str | None) -> Path:
    if raw is not None:
        return Path(raw)
    from simple_qna_rag.config import INDEX_ROOT

    return Path(INDEX_ROOT)


def _op_receipt(*, operation: str, operation_id: str, outcome: str, exit_code: int,
                 error_code: str | None, source_version_id: str | None,
                 target_version_id: str | None, pre_pointer: str | None,
                 post_pointer: str | None, started_at: str, finished_at: str,
                 verifications: list[str]) -> dict:
    from simple_qna_rag.index.manifest import canonical_json_bytes as _cjb

    payload = {
        "schema": "m43-lifecycle-receipt-v1",
        "operation": operation,
        "operation_id": operation_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "outcome": outcome,
        "exit_code": exit_code,
        "error_code": error_code,
        "source_version_id": source_version_id,
        "target_version_id": target_version_id,
        "pre_pointer": pre_pointer,
        "post_pointer": post_pointer,
        "verifications": verifications,
    }
    body_for_hash = {k: v for k, v in payload.items()}
    payload["artifact_sha256"] = hashlib.sha256(_cjb(body_for_hash)).hexdigest()
    return payload


def _emit(receipt: dict) -> None:
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False))


def _receipt_from_activation(receipt: "lifecycle.ActivationReceipt") -> dict:
    return _op_receipt(
        operation=receipt.operation, operation_id=receipt.operation_id,
        outcome=receipt.outcome, exit_code=receipt.exit_code,
        error_code=receipt.error_code, source_version_id=None,
        target_version_id=receipt.post_pointer, pre_pointer=receipt.pre_pointer,
        post_pointer=receipt.post_pointer, started_at=receipt.started_at,
        finished_at=receipt.finished_at, verifications=list(receipt.verifications))


def _cmd_build(args) -> dict:
    from datetime import datetime, timezone

    from evaluation.reporting import _git_commit, _git_dirty, build_corpus_manifest
    from simple_qna_rag.index.manifest import canonical_json_bytes as _cjb

    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    index_root = _resolve_index_root(args.index_root)
    from simple_qna_rag import config as _config

    documents_dir = Path(args.documents_dir) if args.documents_dir else Path(_config.DATA_DIR)

    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from simple_qna_rag.rag_engine import _build_embeddings, _settings_binding_snapshot

    documents = []
    documents.extend(DirectoryLoader(str(documents_dir), glob="**/*.pdf",
                                      loader_cls=PyPDFLoader, show_progress=False).load())
    documents.extend(DirectoryLoader(str(documents_dir), glob="**/*.txt",
                                      loader_cls=TextLoader, show_progress=False).load())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_config.CHUNK_SIZE, chunk_overlap=_config.CHUNK_OVERLAP,
        length_function=len, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_documents(documents)

    embeddings = _build_embeddings()
    import numpy as np
    import faiss as faiss_native
    import pickle
    from langchain_community.docstore.in_memory import InMemoryDocstore

    texts = [c.page_content for c in chunks]
    vectors = np.array(embeddings.embed_documents(texts) if texts else [], dtype="float32")
    if vectors.size == 0:
        dimension = getattr(embeddings, "DIMENSION", 1)
        vectors = vectors.reshape(0, dimension)
    dimension = vectors.shape[1] if vectors.size else getattr(embeddings, "DIMENSION", 1)
    native_index = faiss_native.IndexFlatIP(dimension)
    if vectors.size:
        native_index.add(vectors)
    index_to_id = {i: str(i) for i in range(len(chunks))}
    docstore = InMemoryDocstore({str(i): chunks[i] for i in range(len(chunks))})

    faiss_bytes = faiss_native.serialize_index(native_index).tobytes()
    pkl_bytes = pickle.dumps((docstore, index_to_id))

    corpus = build_corpus_manifest(documents_dir)
    binding = _settings_binding_snapshot()
    identity_fields = {
        "corpus_manifest_sha256": corpus["manifest_sha256"],
        "source_document_count": len(documents),
        "chunk_count": len(chunks),
        "embedding_model_name": binding["embedding_model_name"],
        "embedding_model_revision": "unknown",
        "embedding_provider": binding["embedding_provider"],
        "normalize_embeddings": binding["normalize_embeddings"],
        "chunk_size": binding["chunk_size"],
        "chunk_overlap": binding["chunk_overlap"],
        "faiss_index_type": type(native_index).__name__,
        "faiss_dimension": int(native_index.d),
        "settings_hash": hashlib.sha256(_cjb(binding)).hexdigest(),
        "dependency_lock_sha256": _dependency_lock_sha256(),
        "builder_git_sha": _git_commit() or "0" * 40,
        "builder_git_dirty": bool(_git_dirty()),
        "index_faiss": {"size_bytes": len(faiss_bytes), "sha256": hashlib.sha256(faiss_bytes).hexdigest()},
        "index_pkl": {"size_bytes": len(pkl_bytes), "sha256": hashlib.sha256(pkl_bytes).hexdigest()},
        "source": "build",
        "legacy_baseline_id": None,
    }
    manifest = lifecycle.build(index_root, faiss_bytes=faiss_bytes, pkl_bytes=pkl_bytes,
                                identity_fields=identity_fields,
                                lock_timeout=args.lock_timeout_seconds)
    finished = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return _op_receipt(operation="build", operation_id=manifest["version_id"], outcome="PASS",
                        exit_code=0, error_code=None, source_version_id=None,
                        target_version_id=manifest["version_id"], pre_pointer=None,
                        post_pointer=None, started_at=started, finished_at=finished,
                        verifications=["schema", "hash", "size"])


def _dependency_lock_sha256() -> str:
    import sys as _sys

    sys_path = list(_sys.path)
    scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    if scripts_dir not in sys_path:
        _sys.path.insert(0, scripts_dir)
    from dependency_snapshot import LOCK_FILE, _lock_sha256_canonical

    return _lock_sha256_canonical(LOCK_FILE.read_text(encoding="utf-8"))


def _cmd_import_legacy(args) -> dict:
    from datetime import datetime, timezone

    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    index_root = _resolve_index_root(args.index_root)
    manifest = lifecycle.import_legacy(index_root, Path(args.source_dir),
                                        lock_timeout=args.lock_timeout_seconds)
    finished = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return _op_receipt(operation="import_legacy", operation_id=manifest["version_id"], outcome="PASS",
                        exit_code=0, error_code=None, source_version_id=None,
                        target_version_id=manifest["version_id"], pre_pointer=None,
                        post_pointer=None, started_at=started, finished_at=finished,
                        verifications=["schema", "hash", "size"])


def _cmd_verify(args) -> dict:
    from datetime import datetime, timezone

    from simple_qna_rag.index.verification import verify_version
    from simple_qna_rag.rag_engine import _settings_binding_snapshot

    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    index_root = _resolve_index_root(args.index_root)
    verify_version(index_root, args.version, settings_snapshot=_settings_binding_snapshot())
    if args.reconcile_check:
        lifecycle._diagnose_pending_transition(index_root)
    finished = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return _op_receipt(operation="verify", operation_id=args.version, outcome="PASS",
                        exit_code=0, error_code=None, source_version_id=args.version,
                        target_version_id=args.version, pre_pointer=None, post_pointer=None,
                        started_at=started, finished_at=finished,
                        verifications=["schema", "hash", "size", "settings_binding"])


def _cmd_activate(args) -> dict:
    from simple_qna_rag.rag_engine import _settings_binding_snapshot

    index_root = _resolve_index_root(args.index_root)
    receipt = lifecycle.activate(index_root, args.to_version, operation="activate",
                                  settings_snapshot=_settings_binding_snapshot(),
                                  lock_timeout=args.lock_timeout_seconds)
    if receipt.outcome != "PASS":
        raise _DomainFailure(receipt)
    return _receipt_from_activation(receipt)


def _cmd_rollback(args) -> dict:
    from simple_qna_rag.rag_engine import _settings_binding_snapshot

    index_root = _resolve_index_root(args.index_root)
    to_version = args.to_version
    if args.to_previous:
        to_version = lifecycle.resolve_previous_for_rollback(index_root)
    receipt = lifecycle.rollback(index_root, to_version,
                                  settings_snapshot=_settings_binding_snapshot(),
                                  lock_timeout=args.lock_timeout_seconds)
    if receipt.outcome != "PASS":
        raise _DomainFailure(receipt)
    return _receipt_from_activation(receipt)


def _cmd_list(args) -> dict:
    from datetime import datetime, timezone

    index_root = _resolve_index_root(args.index_root)
    info = lifecycle.list_versions(index_root)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    receipt = _op_receipt(operation="list", operation_id="list", outcome="PASS", exit_code=0,
                           error_code=None, source_version_id=None,
                           target_version_id=info["current"], pre_pointer=info["previous"],
                           post_pointer=info["current"], started_at=now, finished_at=now,
                           verifications=[])
    receipt["versions"] = info["versions"]
    return receipt


def _cmd_cleanup(args) -> dict:
    from datetime import datetime, timezone

    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    index_root = _resolve_index_root(args.index_root)
    apply = bool(args.apply)
    result = lifecycle.cleanup(index_root, apply=apply, protect=list(args.protect),
                                lock_timeout=args.lock_timeout_seconds,
                                include_staging=args.include_staging,
                                staging_min_age_seconds=args.staging_min_age_seconds)
    finished = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    receipt = _op_receipt(operation="cleanup", operation_id="cleanup", outcome="PASS", exit_code=0,
                           error_code=None, source_version_id=None, target_version_id=None,
                           pre_pointer=None, post_pointer=None, started_at=started,
                           finished_at=finished, verifications=[])
    receipt["candidates"] = result.candidates
    receipt["deleted"] = result.deleted
    receipt["dry_run"] = result.dry_run
    receipt["staging_candidates"] = result.staging_candidates
    receipt["staging_deleted"] = result.staging_deleted
    return receipt


class _DomainFailure(Exception):
    def __init__(self, receipt: "lifecycle.ActivationReceipt") -> None:
        self.receipt = receipt
        super().__init__(receipt.error_code or "domain_failure")


_DISPATCH = {
    "build": _cmd_build,
    "import-legacy": _cmd_import_legacy,
    "verify": _cmd_verify,
    "activate": _cmd_activate,
    "rollback": _cmd_rollback,
    "list": _cmd_list,
    "cleanup": _cmd_cleanup,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = _DISPATCH[args.command](args)
    except _DomainFailure as exc:
        receipt = _receipt_from_activation(exc.receipt)
        _emit(receipt)
        return receipt["exit_code"]
    except lifecycle.LockTimeoutError:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        receipt = _op_receipt(operation=args.command, operation_id="lock_timeout", outcome="FAIL",
                               exit_code=3, error_code="lock_timeout", source_version_id=None,
                               target_version_id=None, pre_pointer=None, post_pointer=None,
                               started_at=now, finished_at=now, verifications=[])
        _emit(receipt)
        return 3
    except (TrustBoundaryError, ManifestError, lifecycle.LifecycleError) as exc:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        reason = getattr(exc, "reason", str(type(exc).__name__))
        receipt = _op_receipt(operation=args.command, operation_id="failed", outcome="FAIL",
                               exit_code=1, error_code=reason, source_version_id=None,
                               target_version_id=None, pre_pointer=None, post_pointer=None,
                               started_at=now, finished_at=now, verifications=[])
        _emit(receipt)
        return 1
    _emit(receipt)
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
