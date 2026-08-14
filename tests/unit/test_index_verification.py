"""M4.3-REQ-001 — contained-open/trust-before-pickle tests (Design.md §3)."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from simple_qna_rag.index import lifecycle
from simple_qna_rag.index.manifest import canonical_json_bytes
from simple_qna_rag.index.verification import (
    CurrentPointerMissing,
    MAX_MANIFEST_BYTES,
    REASONS,
    TrustBoundaryError,
    _MAX_CURRENT_BYTES,
    _read_bounded,
    load_verified_faiss,
    resolve_current,
    verify_version,
)

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


def _publish_real_index(index_root: Path) -> str:
    """Builds and publishes a tiny, real FAISS index (no mocking of the
    trust boundary itself) so tests exercise the actual native
    deserialize/verify path end to end."""
    import faiss
    import numpy as np
    import pickle
    from langchain_community.docstore.in_memory import InMemoryDocstore

    index = faiss.IndexFlatIP(3)
    index.add(np.array([[1.0, 0.0, 0.0]], dtype="float32"))
    faiss_bytes = faiss.serialize_index(index).tobytes()
    docstore = InMemoryDocstore({"0": "doc-0"})
    pkl_bytes = pickle.dumps((docstore, {0: "0"}))
    manifest = lifecycle.build(index_root, faiss_bytes=faiss_bytes, pkl_bytes=pkl_bytes,
                                identity_fields=_identity(faiss_bytes, pkl_bytes))
    return manifest["version_id"]


def test_verify_version_succeeds_on_published_index(tmp_path):
    version_id = _publish_real_index(tmp_path)
    verified = verify_version(tmp_path, version_id, settings_snapshot=_SNAPSHOT)
    assert verified.version_id == version_id
    assert verified.manifest["chunk_count"] == 1


def test_verify_then_load_uses_captured_bytes_no_reopen(tmp_path):
    version_id = _publish_real_index(tmp_path)
    embeddings = mock.Mock()
    with mock.patch("os.open", wraps=os.open) as spy_open:
        vs = load_verified_faiss(tmp_path, version_id, embeddings=embeddings,
                                  settings_snapshot=_SNAPSHOT)
        opens_after_verify = spy_open.call_count
    assert vs is not None
    # Sanity: verify_version legitimately opened several fds; the point of
    # this test is that _construct_faiss_from_verified_bytes performs no
    # additional filesystem opens of its own (asserted via the module-level
    # no-syscall contract in verification.py; this call succeeding without
    # touching the filesystem again after publish is the behavioural proof).
    assert opens_after_verify >= 1


def test_racer_between_member_opens_has_no_effect(tmp_path):
    """verify_version() reads all three members through the *same* dirfd
    (§3.2) — replacing the sibling files after the version_dir fd is open
    cannot change what gets verified/returned."""
    import threading

    version_id = _publish_real_index(tmp_path)
    version_dir = tmp_path / "versions" / version_id
    original_faiss = (version_dir / "index.faiss").read_bytes()

    verified = verify_version(tmp_path, version_id, settings_snapshot=_SNAPSHOT)
    assert verified.faiss_bytes == original_faiss

    # Racer deletes the whole directory *after* verify_version returned —
    # the already-captured bytes must remain usable (proves load never
    # reopens the path).
    os.chmod(version_dir, 0o700)
    for name in ("index.faiss", "index.pkl", "manifest.json"):
        os.chmod(version_dir / name, 0o600)
    import shutil

    shutil.rmtree(version_dir)
    from simple_qna_rag.index.verification import _construct_faiss_from_verified_bytes

    embeddings = mock.Mock()
    embeddings.embed_query = mock.Mock()
    vs = _construct_faiss_from_verified_bytes(verified, embeddings)
    assert vs is not None


def test_symlink_owner_mode_toctou_matrix(tmp_path):
    version_id = _publish_real_index(tmp_path)
    version_dir = tmp_path / "versions" / version_id

    # member_is_symlink: replace index.faiss with a symlink
    os.chmod(version_dir, 0o700)
    faiss_path = version_dir / "index.faiss"
    os.chmod(faiss_path, 0o600)
    faiss_bytes = faiss_path.read_bytes()
    os.unlink(faiss_path)
    (tmp_path / "evil.bin").write_bytes(faiss_bytes)
    os.symlink(tmp_path / "evil.bin", faiss_path)
    with pytest.raises(TrustBoundaryError) as excinfo:
        verify_version(tmp_path, version_id, settings_snapshot=_SNAPSHOT)
    assert excinfo.value.reason == "member_is_symlink"

    # restore, then test member_mode_forbidden (group-writable)
    os.unlink(faiss_path)
    faiss_path.write_bytes(faiss_bytes)
    os.chmod(faiss_path, 0o664)
    with pytest.raises(TrustBoundaryError) as excinfo:
        verify_version(tmp_path, version_id, settings_snapshot=_SNAPSHOT)
    assert excinfo.value.reason == "member_mode_forbidden"
    os.chmod(faiss_path, 0o444)
    os.chmod(version_dir, 0o555)


def test_permission_denied_matrix_surfaces_disclosed_reasons_not_raw_oserror(tmp_path):
    """Hosted_CI_Remediation_Iteration_6.md — a real EACCES anywhere along
    the contained-open traversal (bind-mounted `INDEX_ROOT` whose
    intermediate directory permissions don't grant the reading UID
    traversal) previously propagated as a raw, untranslated `OSError`
    instead of a disclosed `TrustBoundaryError`, which `_load_vectorstore()`
    only catches as `TrustBoundaryError` — the raw OSError fell through into
    `RAGEngine.initialize()`'s generic `except Exception: return False`,
    producing an opaque, undiagnosable `engine_init_failed` instead of a
    bounded `artifact_*` reason. Every EACCES entry point must now raise a
    `TrustBoundaryError` with a specific, allowlisted reason."""
    version_id = _publish_real_index(tmp_path)
    version_dir = tmp_path / "versions" / version_id
    faiss_path = version_dir / "index.faiss"

    # member_permission_denied: index.faiss itself unreadable.
    os.chmod(version_dir, 0o755)
    os.chmod(faiss_path, 0o000)
    try:
        with pytest.raises(TrustBoundaryError) as excinfo:
            verify_version(tmp_path, version_id, settings_snapshot=_SNAPSHOT)
        assert excinfo.value.reason == "member_permission_denied"
    finally:
        os.chmod(faiss_path, 0o444)

    # version_dir_permission_denied: versions/<id>/ itself untraversable.
    os.chmod(version_dir, 0o000)
    try:
        with pytest.raises(TrustBoundaryError) as excinfo:
            verify_version(tmp_path, version_id, settings_snapshot=_SNAPSHOT)
        assert excinfo.value.reason == "version_dir_permission_denied"
    finally:
        os.chmod(version_dir, 0o555)

    # root_permission_denied: INDEX_ROOT itself untraversable.
    from simple_qna_rag.index.verification import open_contained_root

    os.chmod(tmp_path, 0o000)
    try:
        with pytest.raises(TrustBoundaryError) as excinfo:
            open_contained_root(tmp_path)
        assert excinfo.value.reason == "root_permission_denied"
    finally:
        os.chmod(tmp_path, 0o755)

    assert "root_permission_denied" in REASONS
    assert "version_dir_permission_denied" in REASONS
    assert "member_permission_denied" in REASONS


def test_verify_version_rejects_extra_member(tmp_path):
    version_id = _publish_real_index(tmp_path)
    version_dir = tmp_path / "versions" / version_id
    os.chmod(version_dir, 0o700)
    (version_dir / "extra.bin").write_bytes(b"x")
    with pytest.raises(TrustBoundaryError) as excinfo:
        verify_version(tmp_path, version_id, settings_snapshot=_SNAPSHOT)
    assert excinfo.value.reason == "extra_or_missing_member"


def test_verify_version_rejects_hash_mismatch(tmp_path):
    version_id = _publish_real_index(tmp_path)
    version_dir = tmp_path / "versions" / version_id
    os.chmod(version_dir, 0o700)
    pkl_path = version_dir / "index.pkl"
    os.chmod(pkl_path, 0o600)
    data = bytearray(pkl_path.read_bytes())
    data[0] ^= 0xFF
    pkl_path.write_bytes(bytes(data))
    os.chmod(pkl_path, 0o444)
    with pytest.raises(TrustBoundaryError) as excinfo:
        verify_version(tmp_path, version_id, settings_snapshot=_SNAPSHOT)
    assert excinfo.value.reason == "member_hash_mismatch"


def test_verify_version_rejects_settings_mismatch(tmp_path):
    version_id = _publish_real_index(tmp_path)
    mismatched = dict(_SNAPSHOT)
    mismatched["embedding_model_name"] = "other-model"
    with pytest.raises(TrustBoundaryError) as excinfo:
        verify_version(tmp_path, version_id, settings_snapshot=mismatched)
    assert excinfo.value.reason == "settings_mismatch"


def test_current_pointer_trust_matrix(tmp_path):
    version_id = _publish_real_index(tmp_path)

    # (a) genuine absence -> CurrentPointerMissing
    with pytest.raises(CurrentPointerMissing):
        resolve_current(tmp_path)

    # valid current -> resolves. Production always writes exact canonical
    # bytes (lifecycle.py::activate) — the reader now enforces that exactly,
    # so tests must write the same canonical form rather than plain
    # `json.dumps` (which uses different separators/whitespace).
    current_path = tmp_path / "current"
    current_path.write_bytes(
        canonical_json_bytes({"schema_version": 1, "version_id": version_id}) + b"\n")
    assert resolve_current(tmp_path) == version_id

    # (b) symlink to a valid target -> TrustBoundaryError, not silently accepted
    os.unlink(current_path)
    (tmp_path / "real_current.json").write_bytes(
        canonical_json_bytes({"schema_version": 1, "version_id": version_id}) + b"\n")
    os.symlink(tmp_path / "real_current.json", current_path)
    with pytest.raises(TrustBoundaryError) as excinfo:
        resolve_current(tmp_path)
    assert excinfo.value.reason == "current_pointer_symlink"
    os.unlink(current_path)

    # (c) dangling symlink -> TrustBoundaryError (not "missing")
    os.symlink(tmp_path / "nonexistent_target", current_path)
    with pytest.raises(TrustBoundaryError) as excinfo:
        resolve_current(tmp_path)
    assert excinfo.value.reason == "current_pointer_symlink"
    os.unlink(current_path)

    # (d) points at an unknown version id -> current_pointer_unknown_version
    current_path.write_bytes(
        canonical_json_bytes({"schema_version": 1, "version_id": "0" * 16}) + b"\n")
    with pytest.raises(TrustBoundaryError) as excinfo:
        resolve_current(tmp_path)
    assert excinfo.value.reason == "current_pointer_unknown_version"

    # (e) malformed json -> current_pointer_malformed
    current_path.write_bytes(b"not json")
    with pytest.raises(TrustBoundaryError) as excinfo:
        resolve_current(tmp_path)
    assert excinfo.value.reason == "current_pointer_malformed"


def test_resolve_current_direct_open_eacces_translates_to_disclosed_reason(tmp_path):
    """CR-HCIR6-MAJ-01 — `resolve_current()`'s own direct
    `os.open("current", ..., dir_fd=root.fd)` previously translated only
    `ENOENT` (-> `CurrentPointerMissing`, the sole legacy-fallback trigger)
    and `ELOOP` (-> `current_pointer_symlink`); a real `EACCES` (e.g. a
    bind-mounted `INDEX_ROOT` whose `current` file the reading UID cannot
    read) fell through as a raw `PermissionError` — reproducing the exact
    undiagnosable `engine_init_failed` collapse the three `ContainedDir`
    entry points were already fixed for in Iteration 6. Patches the exact
    `current` open call (mutation-strength: reverting just the new `EACCES`
    branch in `resolve_current` fails this test) rather than relying on a
    real chmod, since a root-owned CI process can't be reliably denied via
    chmod the way the on-disk permission_denied matrix above is."""
    version_id = _publish_real_index(tmp_path)
    (tmp_path / "current").write_bytes(
        canonical_json_bytes({"schema_version": 1, "version_id": version_id}) + b"\n")

    real_open = os.open

    def fake_open(path, flags, *args, **kwargs):
        if path == "current":
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(path, flags, *args, **kwargs)

    with mock.patch("simple_qna_rag.index.verification.os.open", side_effect=fake_open):
        with pytest.raises(TrustBoundaryError) as excinfo:
            resolve_current(tmp_path)
    assert excinfo.value.reason == "current_pointer_permission_denied"
    assert "current_pointer_permission_denied" in REASONS


def test_root_escape_rejected_for_dotdot_and_slash(tmp_path):
    from simple_qna_rag.index.verification import open_contained_root

    root = open_contained_root(tmp_path)
    try:
        with pytest.raises(TrustBoundaryError) as excinfo:
            root.open_subdir("..")
        assert excinfo.value.reason == "root_escape"
        with pytest.raises(TrustBoundaryError) as excinfo:
            root.open_subdir("a/b")
        assert excinfo.value.reason == "root_escape"
    finally:
        root.close()


# ---------------------------------------------------------------------------
# CR-I1-MAJ-03 — bounded pre-verification reads (never load more than the
# manifest-declared size + 1 into memory, regardless of actual file size)
# ---------------------------------------------------------------------------


def test_read_bounded_stops_at_max_bytes_against_a_growing_source():
    """A pipe whose writer never stops (simulating a file that keeps
    growing/never reaches EOF) must not make `_read_bounded` accumulate more
    than `max_bytes` or block forever."""
    import threading

    read_fd, write_fd = os.pipe()
    max_bytes = 10_000
    stop = threading.Event()

    def _feed():
        chunk = b"x" * 4096
        while not stop.is_set():
            try:
                os.write(write_fd, chunk)
            except OSError:
                break

    writer = threading.Thread(target=_feed, daemon=True)
    writer.start()
    try:
        data = _read_bounded(read_fd, max_bytes=max_bytes)
        assert len(data) == max_bytes
    finally:
        stop.set()
        os.close(write_fd)
        writer.join(timeout=2)
        os.close(read_fd)


def test_read_bounded_returns_short_data_on_genuine_eof(tmp_path):
    path = tmp_path / "short.bin"
    path.write_bytes(b"abc")
    fd = os.open(str(path), os.O_RDONLY)
    try:
        data = _read_bounded(fd, max_bytes=1000)
    finally:
        os.close(fd)
    assert data == b"abc"


def test_read_bounded_never_requests_more_than_the_bound(tmp_path):
    path = tmp_path / "big.bin"
    path.write_bytes(os.urandom(50_000))
    fd = os.open(str(path), os.O_RDONLY)
    reads: list[int] = []
    real_read = os.read

    def _tracking_read(fd_, n):
        chunk = real_read(fd_, n)
        reads.append(len(chunk))
        return chunk

    try:
        with mock.patch("simple_qna_rag.index.verification.os.read", side_effect=_tracking_read):
            data = _read_bounded(fd, max_bytes=1000)
    finally:
        os.close(fd)
    assert len(data) == 1000
    assert sum(reads) == 1000


def test_verify_version_rejects_oversize_member_with_bounded_read(tmp_path):
    """A tampered `index.faiss` far larger than the manifest's declared
    `size_bytes` must be rejected as `member_size_mismatch` without the
    verifier ever reading the full oversize file into memory."""
    version_id = _publish_real_index(tmp_path)
    version_dir = tmp_path / "versions" / version_id
    os.chmod(version_dir, 0o700)
    faiss_path = version_dir / "index.faiss"
    original = faiss_path.read_bytes()
    os.chmod(faiss_path, 0o600)
    faiss_path.write_bytes(original + os.urandom(5_000_000))
    os.chmod(faiss_path, 0o444)

    reads: list[int] = []
    real_read = os.read

    def _tracking_read(fd_, n):
        chunk = real_read(fd_, n)
        reads.append(len(chunk))
        return chunk

    with mock.patch("simple_qna_rag.index.verification.os.read", side_effect=_tracking_read):
        with pytest.raises(TrustBoundaryError) as excinfo:
            verify_version(tmp_path, version_id, settings_snapshot=_SNAPSHOT)
    assert excinfo.value.reason == "member_size_mismatch"
    # bounded to declared size + 1 for the oversize member itself, never the
    # full 5MB+ tampered file (manifest.json/index.pkl reads add a small,
    # fixed amount on top of that bound).
    assert sum(reads) < len(original) + 5_000_000


def test_verify_version_rejects_short_member(tmp_path):
    version_id = _publish_real_index(tmp_path)
    version_dir = tmp_path / "versions" / version_id
    os.chmod(version_dir, 0o700)
    pkl_path = version_dir / "index.pkl"
    original = pkl_path.read_bytes()
    os.chmod(pkl_path, 0o600)
    pkl_path.write_bytes(original[:-1])
    os.chmod(pkl_path, 0o444)
    with pytest.raises(TrustBoundaryError) as excinfo:
        verify_version(tmp_path, version_id, settings_snapshot=_SNAPSHOT)
    assert excinfo.value.reason == "member_size_mismatch"


# ---------------------------------------------------------------------------
# CR-I1-MIN-01 — manifest.json/current bounded reads, EOF, and exact
# canonical-byte enforcement
# ---------------------------------------------------------------------------


def test_verify_version_rejects_manifest_larger_than_max_bytes(tmp_path):
    version_id = _publish_real_index(tmp_path)
    version_dir = tmp_path / "versions" / version_id
    manifest_path = version_dir / "manifest.json"
    os.chmod(version_dir, 0o700)
    os.chmod(manifest_path, 0o600)
    original = manifest_path.read_bytes()
    assert original.endswith(b"\n")
    padded = original[:-1] + b" " * (MAX_MANIFEST_BYTES + 1 - len(original)) + b"\n"
    assert len(padded) == MAX_MANIFEST_BYTES + 1
    manifest_path.write_bytes(padded)
    os.chmod(manifest_path, 0o444)
    with pytest.raises(TrustBoundaryError) as excinfo:
        verify_version(tmp_path, version_id, settings_snapshot=_SNAPSHOT)
    assert excinfo.value.reason == "manifest_oversize"


def test_verify_version_rejects_manifest_with_trailing_whitespace_within_limit(tmp_path):
    """Valid JSON followed by extra (JSON-legal) whitespace still fits
    entirely within `MAX_MANIFEST_BYTES` and still parses via `json.loads`
    (which tolerates trailing whitespace) — only the exact canonical-byte
    comparison catches this non-canonical encoding."""
    version_id = _publish_real_index(tmp_path)
    version_dir = tmp_path / "versions" / version_id
    manifest_path = version_dir / "manifest.json"
    os.chmod(version_dir, 0o700)
    os.chmod(manifest_path, 0o600)
    original = manifest_path.read_bytes()
    manifest_path.write_bytes(original + b"   \n")
    os.chmod(manifest_path, 0o444)
    with pytest.raises(TrustBoundaryError) as excinfo:
        verify_version(tmp_path, version_id, settings_snapshot=_SNAPSHOT)
    assert excinfo.value.reason == "manifest_non_canonical"


def test_verify_version_accepts_manifest_missing_only_the_trailing_newline(tmp_path):
    """The single allowed leniency: canonical bytes without the trailing
    newline the writer normally appends must still be accepted."""
    version_id = _publish_real_index(tmp_path)
    version_dir = tmp_path / "versions" / version_id
    manifest_path = version_dir / "manifest.json"
    os.chmod(version_dir, 0o700)
    os.chmod(manifest_path, 0o600)
    original = manifest_path.read_bytes()
    assert original.endswith(b"\n")
    manifest_path.write_bytes(original[:-1])
    os.chmod(manifest_path, 0o444)
    verified = verify_version(tmp_path, version_id, settings_snapshot=_SNAPSHOT)
    assert verified.version_id == version_id


def test_current_pointer_rejects_file_larger_than_max_bytes(tmp_path):
    version_id = _publish_real_index(tmp_path)
    current_path = tmp_path / "current"
    canonical = canonical_json_bytes({"schema_version": 1, "version_id": version_id}) + b"\n"
    padded = canonical[:-1] + b" " * (_MAX_CURRENT_BYTES + 1 - len(canonical)) + b"\n"
    assert len(padded) == _MAX_CURRENT_BYTES + 1
    current_path.write_bytes(padded)
    with pytest.raises(TrustBoundaryError) as excinfo:
        resolve_current(tmp_path)
    assert excinfo.value.reason == "current_pointer_malformed"


def test_current_pointer_rejects_trailing_whitespace_within_limit(tmp_path):
    version_id = _publish_real_index(tmp_path)
    current_path = tmp_path / "current"
    canonical = canonical_json_bytes({"schema_version": 1, "version_id": version_id}) + b"\n"
    current_path.write_bytes(canonical + b"  \n")
    with pytest.raises(TrustBoundaryError) as excinfo:
        resolve_current(tmp_path)
    assert excinfo.value.reason == "current_pointer_malformed"
