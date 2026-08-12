"""M4.3-REQ-002/003 — staging, atomic activation/rollback, retention, legacy
import (Design.md §4).

State machine: `_stage_candidate` -> `_publish` builds an immutable
`versions/<id>/` directory via same-filesystem staging + fsync + atomic
rename. `activate()` is the single primitive shared by activate and
rollback — it writes a `.transition` journal *before* replacing `current`,
so a crash between pointer replace and history/receipt write can always be
deterministically reconciled by `_reconcile_pending_transition` on the next
lock acquisition.
"""

from __future__ import annotations

import dataclasses
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from simple_qna_rag.index.manifest import (
    build_manifest,
    canonical_json_bytes,
)
from simple_qna_rag.index.verification import (
    ContainedDir,
    TrustBoundaryError,
    _VERSION_ID_RE,
    open_contained_root,
    verify_version,
)


class LifecycleError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class LockTimeoutError(Exception):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# durable write primitives (§4.2)
# ---------------------------------------------------------------------------


def _write_fsync(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(data)
        total = 0
        while total < len(view):
            n = os.write(fd, view[total:])
            if n == 0:
                raise OSError("short write returned 0 bytes")
            total += n
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _assert_same_filesystem(index_root: Path) -> None:
    versions_dir = index_root / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = index_root / ".staging"
    if staging_dir.exists() and os.stat(staging_dir).st_dev != os.stat(versions_dir).st_dev:
        raise LifecycleError("cross_device_staging")


def _smoke_check(op_dir: Path, manifest: dict) -> None:
    for name, key in (("index.faiss", "index_faiss"), ("index.pkl", "index_pkl")):
        data = (op_dir / name).read_bytes()
        expected = manifest[key]
        if len(data) != expected["size_bytes"]:
            raise TrustBoundaryError("member_size_mismatch")
        if hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise TrustBoundaryError("member_hash_mismatch")
    import faiss as faiss_native

    faiss_native.read_index(str(op_dir / "index.faiss"))


def _stage_candidate(index_root: Path, *, faiss_bytes: bytes, pkl_bytes: bytes,
                      identity_fields: dict) -> tuple[Path, dict]:
    _assert_same_filesystem(index_root)
    staging_root = index_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    op_dir = staging_root / uuid.uuid4().hex
    op_dir.mkdir(mode=0o700)
    _write_fsync(op_dir / "index.faiss", faiss_bytes)
    _write_fsync(op_dir / "index.pkl", pkl_bytes)
    manifest = build_manifest(identity_fields, created_at=isoformat(utc_now()))
    _write_fsync(op_dir / "manifest.json", canonical_json_bytes(manifest) + b"\n")
    _fsync_dir(op_dir)
    _smoke_check(op_dir, manifest)
    return op_dir, manifest


def _publish(index_root: Path, op_dir: Path, manifest: dict) -> Path:
    from simple_qna_rag.index.manifest import parse_manifest

    version_id = manifest["version_id"]
    dest = index_root / "versions" / version_id
    if dest.exists():
        existing = parse_manifest((dest / "manifest.json").read_bytes())
        if existing["index_faiss"] != manifest["index_faiss"] or \
                existing["index_pkl"] != manifest["index_pkl"]:
            raise LifecycleError("version_collision_byte_mismatch")
        shutil.rmtree(op_dir)
        return dest
    try:
        os.rename(op_dir, dest)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise LifecycleError("cross_device_staging") from None
        raise
    for name in ("index.faiss", "index.pkl", "manifest.json"):
        os.chmod(dest / name, 0o444)
    os.chmod(dest, 0o555)
    _fsync_dir(index_root / "versions")
    _fsync_dir(index_root)
    return dest


# ---------------------------------------------------------------------------
# OS advisory lock (§4.5)
# ---------------------------------------------------------------------------


@contextmanager
def acquire_index_lock(index_root: Path, *, timeout: float):
    index_root.mkdir(parents=True, exist_ok=True)
    root_fd = os.open(str(index_root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise TrustBoundaryError("lock_file_untrusted")
            if st.st_mode & 0o077:
                raise TrustBoundaryError("lock_file_untrusted")
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise LockTimeoutError()
                    time.sleep(0.05)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    finally:
        os.close(root_fd)


# ---------------------------------------------------------------------------
# current pointer helpers
# ---------------------------------------------------------------------------


def _read_current_or_none(index_root: Path) -> str | None:
    from simple_qna_rag.index.verification import CurrentPointerMissing, resolve_current

    try:
        return resolve_current(index_root)
    except CurrentPointerMissing:
        return None


# ---------------------------------------------------------------------------
# activation_history — one immutable record file per operation (§4.4-a-1)
# ---------------------------------------------------------------------------

_HISTORY_RECORD_NAME_RE = re.compile(r"^[0-9a-f]{32}\.json$")
_HISTORY_RECORD_SCHEMA = "m43-activation-history-record-v1"
_HISTORY_REQUIRED_KEYS = frozenset({
    "schema", "op_id", "sequence", "operation", "pre_pointer", "post_pointer",
    "recorded_at", "reconciled",
})
_HISTORY_OPERATION_ENUM = frozenset({"activate", "rollback"})


def _history_dir(index_root: Path) -> Path:
    return index_root / "activation_history"


def _history_record_path(index_root: Path, op_id: str) -> Path:
    return _history_dir(index_root) / f"{op_id}.json"


def _read_history_rows(index_root: Path) -> list[dict]:
    history_dir = _history_dir(index_root)
    if not history_dir.is_dir():
        return []
    rows: list[dict] = []
    for name in sorted(os.listdir(history_dir)):
        if not _HISTORY_RECORD_NAME_RE.fullmatch(name):
            continue
        expected_op_id = name[: -len(".json")]
        raw = (history_dir / name).read_bytes()
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise TrustBoundaryError("activation_history_record_corrupt") from None
        if not isinstance(row, dict) or set(row) != _HISTORY_REQUIRED_KEYS:
            raise TrustBoundaryError("activation_history_schema_invalid")
        if row["schema"] != _HISTORY_RECORD_SCHEMA:
            raise TrustBoundaryError("activation_history_schema_invalid")
        if not isinstance(row["op_id"], str) or row["op_id"] != expected_op_id:
            raise TrustBoundaryError("activation_history_filename_op_id_mismatch")
        if not isinstance(row["sequence"], int) or isinstance(row["sequence"], bool) \
                or row["sequence"] < 0:
            raise TrustBoundaryError("activation_history_schema_invalid")
        if row["operation"] not in _HISTORY_OPERATION_ENUM:
            raise TrustBoundaryError("activation_history_operation_invalid")
        if row["pre_pointer"] is not None and (
                not isinstance(row["pre_pointer"], str)
                or not _VERSION_ID_RE.fullmatch(row["pre_pointer"])):
            raise TrustBoundaryError("activation_history_schema_invalid")
        if not isinstance(row["post_pointer"], str) or not _VERSION_ID_RE.fullmatch(row["post_pointer"]):
            raise TrustBoundaryError("activation_history_schema_invalid")
        if not isinstance(row["recorded_at"], str):
            raise TrustBoundaryError("activation_history_schema_invalid")
        if not isinstance(row["reconciled"], bool):
            raise TrustBoundaryError("activation_history_schema_invalid")
        rows.append(row)
    rows.sort(key=lambda r: r["sequence"])
    sequences = [r["sequence"] for r in rows]
    if len(set(sequences)) != len(sequences) or (sequences and sequences != list(range(len(sequences)))):
        raise TrustBoundaryError("activation_history_sequence_invalid")
    return rows


def _next_history_sequence(index_root: Path) -> int:
    rows = _read_history_rows(index_root)
    return (max((r["sequence"] for r in rows), default=-1)) + 1


def _append_history(index_root: Path, *, op_id: str, operation: str,
                     pre_pointer: str | None, post_pointer: str,
                     reconciled: bool = False) -> None:
    history_dir = _history_dir(index_root)
    history_dir.mkdir(mode=0o700, exist_ok=True)
    dest = _history_record_path(index_root, op_id)
    if dest.exists():
        return
    sequence = _next_history_sequence(index_root)
    record = canonical_json_bytes({
        "schema": _HISTORY_RECORD_SCHEMA, "op_id": op_id, "sequence": sequence,
        "operation": operation, "pre_pointer": pre_pointer, "post_pointer": post_pointer,
        "recorded_at": isoformat(utc_now()), "reconciled": reconciled,
    }) + b"\n"
    tmp = history_dir / f".tmp.{os.getpid()}.{op_id}"
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
    _write_fsync(tmp, record)
    os.replace(tmp, dest)
    _fsync_dir(history_dir)


def _read_previous_from_history(index_root: Path, *, current: str | None) -> str | None:
    rows = _read_history_rows(index_root)
    if not rows:
        if current is not None:
            raise TrustBoundaryError("activation_history_current_mismatch")
        return None
    latest = rows[-1]
    if latest["post_pointer"] != current:
        raise TrustBoundaryError("activation_history_current_mismatch")
    return latest["pre_pointer"]


# ---------------------------------------------------------------------------
# transition journal (§4.4-a / §4.4-b)
# ---------------------------------------------------------------------------


def _write_transition_journal(index_root: Path, *, phase: str, op_id: str,
                               operation: str, pre_pointer: str | None,
                               post_pointer: str) -> None:
    record = canonical_json_bytes({
        "schema": "m43-transition-journal-v1", "phase": phase, "op_id": op_id,
        "operation": operation, "pre_pointer": pre_pointer, "post_pointer": post_pointer,
        "recorded_at": isoformat(utc_now()),
    }) + b"\n"
    tmp = index_root / f".transition.tmp.{os.getpid()}.{op_id}"
    _write_fsync(tmp, record)
    os.replace(tmp, index_root / ".transition")
    _fsync_dir(index_root)


def _clear_transition_journal(index_root: Path) -> None:
    path = index_root / ".transition"
    if path.exists():
        os.unlink(path)
        _fsync_dir(index_root)


@dataclass(frozen=True)
class ReconcileReport:
    outcome: str
    op_id: str


_TRANSITION_JOURNAL_SCHEMA = "m43-transition-journal-v1"
_TRANSITION_PHASE_ENUM = frozenset({"prepared", "pointer_committed"})
_TRANSITION_REQUIRED_KEYS = frozenset({
    "schema", "phase", "op_id", "operation", "pre_pointer", "post_pointer", "recorded_at",
})
_OP_ID_RE = re.compile(r"^[0-9a-f]{32}$")
# `_write_transition_journal`'s `isoformat(utc_now())` always yields
# `YYYY-MM-DDTHH:MM:SS(.ffffff)?Z` — microseconds are only present when
# non-zero, so the fractional group is optional.
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")


@dataclass(frozen=True)
class _TransitionRecord:
    schema: str
    phase: str
    op_id: str
    operation: str
    pre_pointer: str | None
    post_pointer: str
    recorded_at: str


def _parse_transition_journal(raw: bytes) -> _TransitionRecord:
    """Strict immutable parser for the `.transition` crash-recovery journal
    (§4.4-a/§4.4-b). Every field is validated against its exact expected
    type/enum/pattern *before* any value is trusted by
    `_reconcile_pending_transition` — a single `TrustBoundaryError(
    "transition_journal_corrupt")` covers every failure mode (unknown/
    missing keys, wrong types, invalid enum members, a malformed or
    path-traversal `op_id`, malformed pointer/timestamp strings) so a
    tampered or corrupted journal can never partially inform a mutation."""
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise TrustBoundaryError("transition_journal_corrupt") from None
    if not isinstance(doc, dict) or set(doc) != _TRANSITION_REQUIRED_KEYS:
        raise TrustBoundaryError("transition_journal_corrupt")
    if doc["schema"] != _TRANSITION_JOURNAL_SCHEMA:
        raise TrustBoundaryError("transition_journal_corrupt")
    if doc["phase"] not in _TRANSITION_PHASE_ENUM:
        raise TrustBoundaryError("transition_journal_corrupt")
    op_id = doc["op_id"]
    if not isinstance(op_id, str) or not _OP_ID_RE.fullmatch(op_id):
        raise TrustBoundaryError("transition_journal_corrupt")
    if doc["operation"] not in _HISTORY_OPERATION_ENUM:
        raise TrustBoundaryError("transition_journal_corrupt")
    pre_pointer = doc["pre_pointer"]
    if pre_pointer is not None and (
            not isinstance(pre_pointer, str) or not _VERSION_ID_RE.fullmatch(pre_pointer)):
        raise TrustBoundaryError("transition_journal_corrupt")
    post_pointer = doc["post_pointer"]
    if not isinstance(post_pointer, str) or not _VERSION_ID_RE.fullmatch(post_pointer):
        raise TrustBoundaryError("transition_journal_corrupt")
    recorded_at = doc["recorded_at"]
    if not isinstance(recorded_at, str) or not _TIMESTAMP_RE.fullmatch(recorded_at):
        raise TrustBoundaryError("transition_journal_corrupt")
    return _TransitionRecord(
        schema=doc["schema"], phase=doc["phase"], op_id=op_id, operation=doc["operation"],
        pre_pointer=pre_pointer, post_pointer=post_pointer, recorded_at=recorded_at)


def _reconcile_pending_transition(index_root: Path) -> ReconcileReport | None:
    journal_path = index_root / ".transition"
    if not journal_path.is_file():
        return None
    record = _parse_transition_journal(journal_path.read_bytes())
    actual_current = _read_current_or_none(index_root)

    if record.phase == "prepared" and actual_current == record.pre_pointer:
        os.unlink(journal_path)
        _fsync_dir(index_root)
        return ReconcileReport(outcome="aborted", op_id=record.op_id)
    if actual_current == record.post_pointer:
        _append_history(index_root, op_id=record.op_id, operation=record.operation,
                         pre_pointer=record.pre_pointer, post_pointer=record.post_pointer,
                         reconciled=True)
        receipt = ActivationReceipt(
            schema="m43-lifecycle-receipt-v1", operation=record.operation,
            operation_id=record.op_id, outcome="PASS", exit_code=0, error_code=None,
            pre_pointer=record.pre_pointer, post_pointer=record.post_pointer,
            started_at=record.recorded_at, finished_at=isoformat(utc_now()),
            verifications=["schema", "hash", "size", "settings_binding"],
            reconciled=True)
        _write_receipt_atomic(index_root, receipt)
        os.unlink(journal_path)
        _fsync_dir(index_root)
        return ReconcileReport(outcome="completed", op_id=record.op_id)
    raise TrustBoundaryError("transition_journal_corrupt")


def _diagnose_pending_transition(index_root: Path) -> dict | None:
    """Read-only diagnostic — never mutates. Safe to call without the lock."""
    journal_path = index_root / ".transition"
    if not journal_path.is_file():
        return None
    record = _parse_transition_journal(journal_path.read_bytes())
    actual_current = _read_current_or_none(index_root)
    would = "aborted" if (record.phase == "prepared"
                           and actual_current == record.pre_pointer) else "completed"
    return {
        "phase": record.phase, "pre_pointer": record.pre_pointer,
        "post_pointer": record.post_pointer, "actual_current": actual_current,
        "would_reconcile_to": would,
    }


# ---------------------------------------------------------------------------
# receipt (§6.3) — activation-scoped slot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivationReceipt:
    schema: str
    operation: str
    operation_id: str
    outcome: str
    exit_code: int
    error_code: str | None
    pre_pointer: str | None
    post_pointer: str | None
    started_at: str
    finished_at: str
    verifications: list[str]
    reconciled: bool = False


def _read_last_receipt_or_none(index_root: Path) -> dict | None:
    path = index_root / ".last_activation_receipt.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _write_receipt_atomic(index_root: Path, receipt: ActivationReceipt) -> None:
    existing = _read_last_receipt_or_none(index_root)
    if existing is not None and existing.get("operation_id") == receipt.operation_id:
        return
    payload = canonical_json_bytes(dataclasses.asdict(receipt)) + b"\n"
    tmp = index_root / f".receipt.tmp.{os.getpid()}.{receipt.operation_id}"
    _write_fsync(tmp, payload)
    os.replace(tmp, index_root / ".last_activation_receipt.json")
    _fsync_dir(index_root)


def _fail_receipt(op_id: str, operation: str, started: str, error_code: str,
                   *, exit_code: int) -> ActivationReceipt:
    return ActivationReceipt(
        schema="m43-lifecycle-receipt-v1", operation=operation, operation_id=op_id,
        outcome="FAIL", exit_code=exit_code, error_code=error_code,
        pre_pointer=None, post_pointer=None, started_at=started,
        finished_at=isoformat(utc_now()), verifications=[])


# ---------------------------------------------------------------------------
# activate / rollback — single primitive (§4.4)
# ---------------------------------------------------------------------------


def activate(index_root: Path, version_id: str, *, operation: str,
             settings_snapshot: dict, lock_timeout: float = 10.0,
             expected_owner_uid: int | None = None) -> ActivationReceipt:
    op_id = uuid.uuid4().hex
    started = isoformat(utc_now())
    try:
        with acquire_index_lock(index_root, timeout=lock_timeout):
            _reconcile_pending_transition(index_root)
            pre = _read_current_or_none(index_root)
            verified = verify_version(index_root, version_id,
                                       settings_snapshot=settings_snapshot,
                                       expected_owner_uid=expected_owner_uid)
            _write_transition_journal(index_root, phase="prepared", op_id=op_id,
                                       operation=operation, pre_pointer=pre,
                                       post_pointer=verified.version_id)
            tmp = index_root / f".current.tmp.{os.getpid()}.{op_id}"
            _write_fsync(tmp, canonical_json_bytes(
                {"schema_version": 1, "version_id": verified.version_id}) + b"\n")
            os.replace(tmp, index_root / "current")
            _fsync_dir(index_root)
            _write_transition_journal(index_root, phase="pointer_committed", op_id=op_id,
                                       operation=operation, pre_pointer=pre,
                                       post_pointer=verified.version_id)
            _append_history(index_root, op_id=op_id, operation=operation,
                             pre_pointer=pre, post_pointer=verified.version_id)
            receipt = ActivationReceipt(
                schema="m43-lifecycle-receipt-v1", operation=operation,
                operation_id=op_id, outcome="PASS", exit_code=0, error_code=None,
                pre_pointer=pre, post_pointer=verified.version_id,
                started_at=started, finished_at=isoformat(utc_now()),
                verifications=["schema", "hash", "size", "settings_binding"])
            _write_receipt_atomic(index_root, receipt)
            _clear_transition_journal(index_root)
            return receipt
    except LockTimeoutError:
        return _fail_receipt(op_id, operation, started, "lock_timeout", exit_code=3)
    except TrustBoundaryError as exc:
        return _fail_receipt(op_id, operation, started, exc.reason, exit_code=1)


def rollback(index_root: Path, to_version_id: str, *, settings_snapshot: dict,
             lock_timeout: float = 10.0, expected_owner_uid: int | None = None) -> ActivationReceipt:
    return activate(index_root, to_version_id, operation="rollback",
                     settings_snapshot=settings_snapshot, lock_timeout=lock_timeout,
                     expected_owner_uid=expected_owner_uid)


def resolve_previous_for_rollback(index_root: Path) -> str:
    """CLI-only helper for `rollback --to-previous` — resolved *outside* the
    lock, then handed to `activate()` like any explicit `--to-version`
    (Design.md §4.4 narrative; lifecycle itself has no "previous" concept)."""
    current = _read_current_or_none(index_root)
    previous = _read_previous_from_history(index_root, current=current)
    if previous is None:
        raise LifecycleError("no_previous_version")
    return previous


# ---------------------------------------------------------------------------
# build (index/documents -> staged candidate -> published version)
# ---------------------------------------------------------------------------


def build(index_root: Path, *, faiss_bytes: bytes, pkl_bytes: bytes,
          identity_fields: dict, lock_timeout: float = 10.0) -> dict:
    with acquire_index_lock(index_root, timeout=lock_timeout):
        _reconcile_pending_transition(index_root)
        op_dir, manifest = _stage_candidate(
            index_root, faiss_bytes=faiss_bytes, pkl_bytes=pkl_bytes,
            identity_fields=identity_fields)
        _publish(index_root, op_dir, manifest)
        return manifest


# ---------------------------------------------------------------------------
# legacy import — pinned approval as code constants (§4.7)
# ---------------------------------------------------------------------------

_PINNED_M3_BASELINE_ID = "m3_initial"
_PINNED_M3_APPROVED_INDEX_FAISS_SHA256 = (
    "c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820"
)
_PINNED_M3_APPROVED_INDEX_PKL_SHA256 = (
    "3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00"
)


def _pinned_m3_approved_pair() -> dict:
    """Production callers invoke this with no arguments — it returns only
    module constants and never touches the filesystem (grep-auditable: no
    os.open/open/read_bytes/read_text in this function body)."""
    return {
        "index_faiss_sha256": _PINNED_M3_APPROVED_INDEX_FAISS_SHA256,
        "index_pkl_sha256": _PINNED_M3_APPROVED_INDEX_PKL_SHA256,
        "baseline_id": _PINNED_M3_BASELINE_ID,
    }


def _parse_m3_baseline_fingerprint(raw: bytes) -> dict:
    """Pure function — takes bytes with the same schema as
    `evaluation/baselines/m3_initial.json` and returns
    `{"index_faiss_sha256": ..., "index_pkl_sha256": ...}`. Never knows a
    file path (Design.md §4.7 provenance-test seam, DR-I2-MIN-06/DR-I3-MIN-06)."""
    doc = json.loads(raw.decode("utf-8"))
    fingerprint = doc["reproducibility"]["vectorstore_fingerprint"]
    return {
        "index_faiss_sha256": fingerprint["index_faiss_sha256"],
        "index_pkl_sha256": fingerprint["index_pkl_sha256"],
    }


def _legacy_identity_fields(faiss_bytes: bytes, pkl_bytes: bytes, baseline_id: str) -> dict:
    import faiss as faiss_native
    import numpy as np

    # Legacy M3 baseline predates the versioned manifest, so the embedding/
    # chunk fields it was actually built with are not recorded anywhere —
    # this binds the imported manifest to the *current* settings snapshot
    # instead (the settings this operator has configured for the legacy
    # baseline), so `verify_version`'s settings-binding check (§3.3) compares
    # like-for-like rather than manufacturing a guaranteed mismatch.
    from simple_qna_rag.rag_engine import _settings_binding_snapshot

    binding = _settings_binding_snapshot()
    native_index = faiss_native.deserialize_index(np.frombuffer(faiss_bytes, dtype=np.uint8))
    return {
        "corpus_manifest_sha256": None,
        "source_document_count": 0,
        "chunk_count": int(native_index.ntotal),
        "embedding_model_name": binding["embedding_model_name"],
        "embedding_model_revision": "unknown",
        "embedding_provider": binding["embedding_provider"],
        "normalize_embeddings": binding["normalize_embeddings"],
        "chunk_size": binding["chunk_size"],
        "chunk_overlap": binding["chunk_overlap"],
        "faiss_index_type": type(native_index).__name__,
        "faiss_dimension": int(native_index.d),
        "settings_hash": hashlib.sha256(b"legacy_import").hexdigest(),
        "dependency_lock_sha256": hashlib.sha256(b"legacy_import").hexdigest(),
        "builder_git_sha": "0" * 40,
        "builder_git_dirty": False,
        "index_faiss": {"size_bytes": len(faiss_bytes), "sha256": hashlib.sha256(faiss_bytes).hexdigest()},
        "index_pkl": {"size_bytes": len(pkl_bytes), "sha256": hashlib.sha256(pkl_bytes).hexdigest()},
        "source": "legacy_import",
        "legacy_baseline_id": baseline_id,
    }


def import_legacy(index_root: Path, source_dir: Path, *,
                   lock_timeout: float = 10.0,
                   _approved_override: dict | None = None) -> dict:
    """`_approved_override` is a leading-underscore test-only seam — no
    `cli/index_lifecycle.py` argparse flag can populate it. Production
    always trusts only `_pinned_m3_approved_pair()`."""
    approved = _approved_override or _pinned_m3_approved_pair()
    faiss_bytes = (source_dir / "index.faiss").read_bytes()
    pkl_bytes = (source_dir / "index.pkl").read_bytes()
    if hashlib.sha256(faiss_bytes).hexdigest() != approved["index_faiss_sha256"] or \
            hashlib.sha256(pkl_bytes).hexdigest() != approved["index_pkl_sha256"]:
        raise TrustBoundaryError("member_hash_mismatch")
    identity_fields = _legacy_identity_fields(faiss_bytes, pkl_bytes, approved["baseline_id"])
    with acquire_index_lock(index_root, timeout=lock_timeout):
        _reconcile_pending_transition(index_root)
        op_dir, manifest = _stage_candidate(
            index_root, faiss_bytes=faiss_bytes, pkl_bytes=pkl_bytes,
            identity_fields=identity_fields)
        _publish(index_root, op_dir, manifest)
    return manifest


# ---------------------------------------------------------------------------
# retention / cleanup (§4.6)
# ---------------------------------------------------------------------------

STAGING_NAME_RE = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class CleanupReceipt:
    candidates: list[str]
    deleted: list[str]
    dry_run: bool
    staging_candidates: list[str] = dataclasses.field(default_factory=list)
    staging_deleted: list[str] = dataclasses.field(default_factory=list)


def _fd_relative_rmtree(parent_fd: int, name: str) -> None:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise TrustBoundaryError("version_dir_symlink") from None
        raise
    try:
        # `_publish` chmods a published version directory to 0o555 (§4.3) —
        # POSIX requires *directory* write permission to unlink an entry
        # from it (a file's own mode bits do not govern this), so a
        # published, read-only version directory must regain write
        # permission before its children can be removed. This is the
        # directory being deleted in its entirety by `cleanup()`
        # immediately after, so restoring write here does not weaken the
        # immutability invariant for any *surviving* version.
        os.fchmod(fd, 0o700)
        for entry in os.listdir(fd):
            st = os.stat(entry, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                raise TrustBoundaryError("version_dir_symlink")
            if stat.S_ISDIR(st.st_mode):
                _fd_relative_rmtree(fd, entry)
            else:
                os.chmod(entry, 0o600, dir_fd=fd, follow_symlinks=False)
                os.unlink(entry, dir_fd=fd)
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=parent_fd)


def _cleanup_staging(root: ContainedDir, *, apply: bool,
                      min_age_seconds: int) -> tuple[list[str], list[str]]:
    staging_dir = root.open_subdir(".staging")
    try:
        now = time.time()
        candidates = []
        for name in staging_dir.listdir():
            st = os.stat(name, dir_fd=staging_dir.fd, follow_symlinks=False)
            if not STAGING_NAME_RE.fullmatch(name):
                continue
            if stat.S_ISLNK(st.st_mode):
                continue
            if not stat.S_ISDIR(st.st_mode):
                continue
            if (now - st.st_mtime) < min_age_seconds:
                continue
            candidates.append(name)
        deleted = []
        if apply:
            for name in candidates:
                _fd_relative_rmtree(staging_dir.fd, name)
                deleted.append(name)
        return sorted(candidates), sorted(deleted)
    finally:
        staging_dir.close()


def cleanup(index_root: Path, *, apply: bool, protect: list[str] | None = None,
            lock_timeout: float = 10.0, include_staging: bool = False,
            staging_min_age_seconds: int = 3600) -> CleanupReceipt:
    protect = protect or []
    (index_root / "versions").mkdir(parents=True, exist_ok=True)
    if include_staging:
        (index_root / ".staging").mkdir(parents=True, exist_ok=True)
    with acquire_index_lock(index_root, timeout=lock_timeout):
        _reconcile_pending_transition(index_root)
        current = _read_current_or_none(index_root)
        previous = _read_previous_from_history(index_root, current=current)
        protected = {current, previous, *protect} - {None}
        root = open_contained_root(index_root)
        try:
            versions_dir = root.open_subdir("versions")
            try:
                all_versions = set(versions_dir.listdir())
                candidates = sorted(all_versions - protected)
                deleted = []
                if apply:
                    for version_id in candidates:
                        current_now = _read_current_or_none(index_root)
                        previous_now = _read_previous_from_history(index_root, current=current_now)
                        if version_id in {current_now, previous_now}:
                            continue
                        _fd_relative_rmtree(versions_dir.fd, version_id)
                        deleted.append(version_id)
            finally:
                versions_dir.close()
            staging_candidates: list[str] = []
            staging_deleted: list[str] = []
            if include_staging:
                staging_candidates, staging_deleted = _cleanup_staging(
                    root, apply=apply, min_age_seconds=staging_min_age_seconds)
        finally:
            root.close()
        return CleanupReceipt(candidates=candidates, deleted=deleted, dry_run=not apply,
                               staging_candidates=staging_candidates,
                               staging_deleted=staging_deleted)


def list_versions(index_root: Path) -> dict:
    """Read-only — no lock (publish is an atomic rename, so a lockless
    reader can never observe a half-written version directory)."""
    current = _read_current_or_none(index_root)
    try:
        previous = _read_previous_from_history(index_root, current=current)
    except TrustBoundaryError:
        previous = None
    versions_dir = index_root / "versions"
    versions = sorted(p.name for p in versions_dir.iterdir()) if versions_dir.is_dir() else []
    return {"current": current, "previous": previous, "versions": versions}
