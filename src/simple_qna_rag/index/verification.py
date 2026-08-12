"""M4.3-REQ-001 — contained-open and trust-before-pickle (Design.md §3).

`ContainedDir` opens a root-to-leaf dirfd chain with `O_NOFOLLOW` so every
member read is relative to an fd whose identity was fixed at open time — a
racer that renames/replaces a directory after it was opened cannot affect an
already-open fd (TOCTOU is closed structurally, not by a path re-check).

`load_verified_faiss` never calls `FAISS.load_local` — it deserializes the
exact bytes `verify_version()` already hashed, so there is no reopen between
verification and pickle load. The only `FAISS.load_local(...,
allow_dangerous_deserialization=True)` call left in the codebase is
`rag_engine.py::RAGEngine._load_vectorstore_legacy` (byte-identical to
648e3ab) — `grep -rn "load_local" src/simple_qna_rag/` must return exactly
one hit.
"""

from __future__ import annotations

import errno
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from simple_qna_rag.index.manifest import (
    MAX_MANIFEST_BYTES,
    ManifestError,
    canonical_json_bytes,
    parse_manifest,
)

_VERSION_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_MEMBER_READ_CHUNK = 1024 * 1024
_MAX_CURRENT_BYTES = 4096


class TrustBoundaryError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CurrentPointerMissing(Exception):
    """`INDEX_ROOT/current` itself does not exist (ENOENT) — genuine absence,
    the only condition allowed to fall back to the legacy loader. A symlink
    (dangling or not) is `TrustBoundaryError("current_pointer_symlink")`,
    never this exception."""


REASONS = frozenset({
    "manifest_missing", "manifest_schema_invalid", "manifest_self_hash_mismatch",
    "version_dir_missing", "version_dir_not_directory", "version_dir_symlink",
    "extra_or_missing_member", "member_not_regular_file", "member_is_symlink",
    "member_mode_forbidden", "member_owner_forbidden",
    "member_size_mismatch", "member_hash_mismatch",
    "root_escape", "cross_device_staging",
    "settings_mismatch", "current_pointer_malformed",
    "current_pointer_unknown_version",
    "current_pointer_symlink",
    "staging_entry_untrusted",
    "lock_file_untrusted",
    "transition_journal_corrupt",
    "manifest_oversize",
    "manifest_non_canonical",
    "activation_history_record_corrupt",
    "activation_history_schema_invalid",
    "activation_history_filename_op_id_mismatch",
    "activation_history_operation_invalid",
    "activation_history_sequence_invalid",
    "activation_history_current_mismatch",
    "test_embedding_seam_unavailable",
})


@dataclass
class ContainedDir:
    fd: int

    def open_subdir(self, name: str) -> "ContainedDir":
        if "/" in name or name in (".", ".."):
            raise TrustBoundaryError("root_escape")
        try:
            fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise TrustBoundaryError("version_dir_symlink") from None
            if exc.errno == errno.ENOENT:
                raise TrustBoundaryError("version_dir_missing") from None
            if exc.errno == errno.ENOTDIR:
                raise TrustBoundaryError("version_dir_not_directory") from None
            raise
        return ContainedDir(fd)

    def open_member(self, name: str, *, allowed_mode_mask: int = 0o022,
                     expected_owner_uid: int | None = None,
                     missing_reason: str = "manifest_missing") -> int:
        if "/" in name or name in (".", ".."):
            raise TrustBoundaryError("root_escape")
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise TrustBoundaryError("member_is_symlink") from None
            if exc.errno == errno.ENOENT:
                raise TrustBoundaryError(missing_reason) from None
            raise
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            raise TrustBoundaryError("member_not_regular_file")
        if st.st_mode & allowed_mode_mask:
            os.close(fd)
            raise TrustBoundaryError("member_mode_forbidden")
        if expected_owner_uid is not None and st.st_uid != expected_owner_uid:
            os.close(fd)
            raise TrustBoundaryError("member_owner_forbidden")
        return fd

    def listdir(self) -> list[str]:
        return os.listdir(self.fd)

    def close(self) -> None:
        os.close(self.fd)


def open_contained_root(root: Path) -> ContainedDir:
    try:
        fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise TrustBoundaryError("root_escape") from None
        raise
    return ContainedDir(fd)


@dataclass(frozen=True)
class VerifiedVersion:
    version_id: str
    manifest: dict
    faiss_bytes: bytes
    pkl_bytes: bytes


def _read_bounded(fd: int, *, max_bytes: int) -> bytes:
    """Reads at most `max_bytes` from `fd`, regardless of how much data the
    underlying file/stream actually holds beyond that — a racer that grows
    the file after this fd was opened, or a manifest that declares a size
    far smaller than the actual on-disk file, can never force more than
    `max_bytes` into memory. Callers pass `expected_size + 1` so that an
    oversize file is distinguishable from an exact-size file by length alone
    (`len(result) > expected_size`) without ever reading past the +1 byte."""
    chunks: list[bytes] = []
    total = 0
    while total < max_bytes:
        chunk = os.read(fd, min(_MEMBER_READ_CHUNK, max_bytes - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _read_and_verify_member(version_dir: ContainedDir, name: str, expected: dict,
                             expected_owner_uid: int | None) -> bytes:
    import hashlib

    fd = version_dir.open_member(name, allowed_mode_mask=0o022,
                                  expected_owner_uid=expected_owner_uid,
                                  missing_reason="extra_or_missing_member")
    try:
        data = _read_bounded(fd, max_bytes=expected["size_bytes"] + 1)
    finally:
        os.close(fd)
    if len(data) != expected["size_bytes"]:
        raise TrustBoundaryError("member_size_mismatch")
    if hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise TrustBoundaryError("member_hash_mismatch")
    return data


def _verify_settings_binding(manifest: dict, settings_snapshot: dict) -> None:
    keys = ("embedding_model_name", "embedding_provider", "normalize_embeddings",
            "chunk_size", "chunk_overlap")
    for key in keys:
        if manifest.get(key) != settings_snapshot.get(key):
            raise TrustBoundaryError("settings_mismatch")


def verify_version(index_root: Path, version_id: str, *,
                    settings_snapshot: dict,
                    expected_owner_uid: int | None = None) -> VerifiedVersion:
    if not _VERSION_ID_RE.fullmatch(version_id):
        raise TrustBoundaryError("current_pointer_malformed")
    root = open_contained_root(index_root)
    try:
        versions_dir = root.open_subdir("versions")
        try:
            version_dir = versions_dir.open_subdir(version_id)
        finally:
            versions_dir.close()
        try:
            entries = sorted(version_dir.listdir())
            if entries != ["index.faiss", "index.pkl", "manifest.json"]:
                raise TrustBoundaryError("extra_or_missing_member")

            manifest_fd = version_dir.open_member(
                "manifest.json", allowed_mode_mask=0o022,
                expected_owner_uid=expected_owner_uid, missing_reason="manifest_missing")
            try:
                raw = _read_bounded(manifest_fd, max_bytes=MAX_MANIFEST_BYTES + 1)
            finally:
                os.close(manifest_fd)
            if len(raw) > MAX_MANIFEST_BYTES:
                raise TrustBoundaryError("manifest_oversize")
            try:
                manifest = parse_manifest(raw)
            except ManifestError as exc:
                raise TrustBoundaryError("manifest_schema_invalid") from exc
            if manifest["version_id"] != version_id:
                raise TrustBoundaryError("manifest_self_hash_mismatch")
            canonical = canonical_json_bytes(manifest) + b"\n"
            if raw != canonical and raw != canonical[:-1]:
                raise TrustBoundaryError("manifest_non_canonical")

            faiss_bytes = _read_and_verify_member(
                version_dir, "index.faiss", manifest["index_faiss"], expected_owner_uid)
            pkl_bytes = _read_and_verify_member(
                version_dir, "index.pkl", manifest["index_pkl"], expected_owner_uid)

            _verify_settings_binding(manifest, settings_snapshot)
            return VerifiedVersion(version_id, manifest, faiss_bytes, pkl_bytes)
        finally:
            version_dir.close()
    finally:
        root.close()


def _construct_faiss_from_verified_bytes(verified: VerifiedVersion, embeddings):
    """Deserializes only from `verified.faiss_bytes`/`verified.pkl_bytes` —
    this function never calls os.open/open() (grep-auditable)."""
    import pickle

    import numpy as np
    import faiss as faiss_native
    from langchain_community.vectorstores import FAISS

    native_index = faiss_native.deserialize_index(
        np.frombuffer(verified.faiss_bytes, dtype=np.uint8))
    docstore, index_to_docstore_id = pickle.loads(verified.pkl_bytes)
    return FAISS(embeddings.embed_query, native_index, docstore, index_to_docstore_id)


def load_verified_faiss(index_root: Path, version_id: str, *, embeddings,
                         settings_snapshot: dict,
                         expected_owner_uid: int | None = None):
    verified = verify_version(index_root, version_id,
                               settings_snapshot=settings_snapshot,
                               expected_owner_uid=expected_owner_uid)
    return _construct_faiss_from_verified_bytes(verified, embeddings)


def resolve_current(index_root: Path) -> str:
    import json

    root = open_contained_root(index_root)
    try:
        try:
            fd = os.open("current", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root.fd)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                raise CurrentPointerMissing() from None
            if exc.errno == errno.ELOOP:
                raise TrustBoundaryError("current_pointer_symlink") from None
            raise
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise TrustBoundaryError("current_pointer_malformed")
            if st.st_mode & 0o022:
                raise TrustBoundaryError("current_pointer_malformed")
            raw = _read_bounded(fd, max_bytes=_MAX_CURRENT_BYTES + 1)
        finally:
            os.close(fd)
    finally:
        root.close()
    if len(raw) > _MAX_CURRENT_BYTES:
        raise TrustBoundaryError("current_pointer_malformed")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise TrustBoundaryError("current_pointer_malformed") from None
    if not isinstance(doc, dict) or set(doc) != {"schema_version", "version_id"} \
            or doc.get("schema_version") != 1:
        raise TrustBoundaryError("current_pointer_malformed")
    canonical = canonical_json_bytes(doc) + b"\n"
    if raw != canonical and raw != canonical[:-1]:
        raise TrustBoundaryError("current_pointer_malformed")
    version_id = doc["version_id"]
    if not isinstance(version_id, str) or not _VERSION_ID_RE.fullmatch(version_id):
        raise TrustBoundaryError("current_pointer_malformed")
    versions_root = open_contained_root(index_root)
    try:
        versions_dir = versions_root.open_subdir("versions")
        try:
            try:
                version_dir = versions_dir.open_subdir(version_id)
            except TrustBoundaryError as exc:
                if exc.reason in ("version_dir_missing", "version_dir_not_directory",
                                   "version_dir_symlink"):
                    raise TrustBoundaryError("current_pointer_unknown_version") from None
                raise
            version_dir.close()
        finally:
            versions_dir.close()
    finally:
        versions_root.close()
    return version_id
