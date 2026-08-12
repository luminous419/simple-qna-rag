"""M4.3-REQ-001 — canonical versioned index manifest (Design.md §2).

`build_manifest`/`derive_version_id` produce a deterministic, self-hashing
manifest from already-staged identity fields. `parse_manifest` is a strict
reader: unknown/missing keys, non-finite numbers, wrong types, and a
self-hash mismatch are all rejected before any value is trusted by callers
(`index/verification.py`). No function in this module ever opens a file or
calls FAISS/pickle — it is pure JSON/hash logic (NFR-001 reproducibility).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

MANIFEST_SCHEMA_VERSION = 1

# Self-referential / execution-metadata fields excluded from the identity hash
# (NFR-001 — generation timestamp and the derived id itself must not feed
# back into their own derivation).
EXCLUDED_FROM_IDENTITY = ("version_id", "created_at")

REQUIRED_KEYS = frozenset({
    "schema_version", "version_id", "created_at",
    "corpus_manifest_sha256",
    "source_document_count", "chunk_count",
    "embedding_model_name", "embedding_model_revision", "embedding_provider",
    "normalize_embeddings", "chunk_size", "chunk_overlap",
    "faiss_index_type", "faiss_dimension",
    "settings_hash", "dependency_lock_sha256",
    "builder_git_sha", "builder_git_dirty",
    "index_faiss", "index_pkl",
    "source",
    "legacy_baseline_id",
})

# `build_manifest()` callers supply identity fields only — `schema_version`
# is a fixed constant that `derive_version_id`/`build_manifest` add
# themselves (it still feeds the hash via `derive_version_id`'s explicit
# `{"schema_version": MANIFEST_SCHEMA_VERSION, **identity_fields}` payload,
# it is just not part of the caller-supplied input contract).
_IDENTITY_KEYS = REQUIRED_KEYS - set(EXCLUDED_FROM_IDENTITY) - {"schema_version"}

_EMBEDDING_PROVIDER_ENUM = frozenset({"huggingface", "deterministic_test"})
_SOURCE_ENUM = frozenset({"build", "legacy_import"})
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_VERSION_ID_RE = re.compile(r"^[0-9a-f]{16}$")

MAX_MANIFEST_BYTES = 65536

# Reasonable per-artifact upper bound applied to `index_faiss`/`index_pkl`
# `size_bytes` at manifest-parse time (Design.md §3.3 bounds pre-verification
# reads to the declared size, but a manifest can still *declare* an
# unreasonable size — this caps the declaration itself before any read is
# attempted). 8 GiB comfortably exceeds any realistic FAISS/pickle artifact
# this project produces.
MAX_MEMBER_SIZE_BYTES = 8 * 1024 * 1024 * 1024


class ManifestError(Exception):
    """Base for every manifest failure. `.reason` is a fixed-vocabulary token
    — never the raw exception text — so callers can build stable receipts."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ManifestSchemaError(ManifestError):
    pass


class ManifestValueError(ManifestError):
    pass


def canonical_json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def derive_version_id(identity_fields: dict) -> str:
    payload = {"schema_version": MANIFEST_SCHEMA_VERSION, **identity_fields}
    raw = canonical_json_bytes(payload)
    return hashlib.sha256(raw).hexdigest()[:16]


def build_manifest(identity_fields: dict, *, created_at: str) -> dict:
    if set(identity_fields) != _IDENTITY_KEYS:
        raise ManifestSchemaError("unknown_or_missing_key")
    version_id = derive_version_id(identity_fields)
    return {
        **identity_fields,
        "version_id": version_id,
        "created_at": created_at,
        "schema_version": MANIFEST_SCHEMA_VERSION,
    }


def _reject_nonfinite(value: str) -> None:
    raise ManifestValueError("non_finite")


def _require_int(doc: dict, key: str, *, minimum: int | None = None) -> None:
    value = doc[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestSchemaError(f"invalid_type:{key}")
    if minimum is not None and value < minimum:
        raise ManifestValueError(f"invalid_range:{key}")


def _require_bool(doc: dict, key: str) -> None:
    if not isinstance(doc[key], bool):
        raise ManifestSchemaError(f"invalid_type:{key}")


def _require_hex(doc: dict, key: str, pattern: re.Pattern) -> None:
    value = doc[key]
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ManifestValueError(f"invalid_hex:{key}")


def _require_file_ref(doc: dict, key: str) -> None:
    value = doc[key]
    if not isinstance(value, dict) or set(value) != {"size_bytes", "sha256"}:
        raise ManifestSchemaError(f"invalid_type:{key}")
    size_bytes = value["size_bytes"]
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0 \
            or size_bytes > MAX_MEMBER_SIZE_BYTES:
        raise ManifestValueError(f"invalid_range:{key}.size_bytes")
    if not isinstance(value["sha256"], str) or not _HEX64_RE.fullmatch(value["sha256"]):
        raise ManifestValueError(f"invalid_hex:{key}.sha256")


def parse_manifest(raw: bytes) -> dict:
    """Strict reader (Design.md §2.1). Every failure raises `ManifestError`
    with a fixed `.reason` — no code path returns a partially-trusted dict."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ManifestValueError("not_utf8") from None
    try:
        doc = json.loads(text, parse_constant=_reject_nonfinite)
    except ManifestValueError:
        raise
    except json.JSONDecodeError:
        raise ManifestSchemaError("not_json") from None

    if not isinstance(doc, dict):
        raise ManifestSchemaError("not_object")
    if set(doc) != REQUIRED_KEYS:
        raise ManifestSchemaError("unknown_or_missing_key")
    if doc.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestSchemaError("wrong_schema_version")

    if not isinstance(doc["created_at"], str):
        raise ManifestSchemaError("invalid_type:created_at")

    _require_int(doc, "source_document_count", minimum=0)
    _require_int(doc, "chunk_count", minimum=0)
    _require_int(doc, "chunk_size", minimum=1)
    _require_int(doc, "chunk_overlap", minimum=0)
    _require_int(doc, "faiss_dimension", minimum=1)
    _require_bool(doc, "normalize_embeddings")
    _require_bool(doc, "builder_git_dirty")

    for key in ("embedding_model_name", "embedding_model_revision", "faiss_index_type"):
        if not isinstance(doc[key], str) or not doc[key]:
            raise ManifestSchemaError(f"invalid_type:{key}")

    if not isinstance(doc["embedding_provider"], str) or \
            doc["embedding_provider"] not in _EMBEDDING_PROVIDER_ENUM:
        raise ManifestValueError("embedding_provider_invalid")

    _require_hex(doc, "settings_hash", _HEX64_RE)
    _require_hex(doc, "dependency_lock_sha256", _HEX64_RE)
    _require_hex(doc, "builder_git_sha", _HEX40_RE)

    _require_file_ref(doc, "index_faiss")
    _require_file_ref(doc, "index_pkl")

    if not isinstance(doc["source"], str) or doc["source"] not in _SOURCE_ENUM:
        raise ManifestValueError("source_invalid")
    if doc["legacy_baseline_id"] is not None and not isinstance(doc["legacy_baseline_id"], str):
        raise ManifestSchemaError("invalid_type:legacy_baseline_id")
    if doc["source"] == "legacy_import":
        if doc["legacy_baseline_id"] is None:
            raise ManifestValueError("legacy_baseline_id_required")
        if doc["corpus_manifest_sha256"] is not None:
            raise ManifestValueError("corpus_manifest_sha256_must_be_null")
    else:
        if doc["legacy_baseline_id"] is not None:
            raise ManifestValueError("legacy_baseline_id_must_be_null")
        if doc["corpus_manifest_sha256"] is not None and (
                not isinstance(doc["corpus_manifest_sha256"], str)
                or not _HEX64_RE.fullmatch(doc["corpus_manifest_sha256"])):
            raise ManifestValueError("invalid_hex:corpus_manifest_sha256")

    if not isinstance(doc["version_id"], str) or not _VERSION_ID_RE.fullmatch(doc["version_id"]):
        raise ManifestValueError("version_id_malformed")
    identity_fields = {k: v for k, v in doc.items() if k not in EXCLUDED_FROM_IDENTITY}
    if derive_version_id(identity_fields) != doc["version_id"]:
        raise ManifestValueError("self_hash_mismatch")

    return doc


def round_trip_stable(manifest: dict, iterations: int = 100) -> bool:
    """Test-only helper: re-serializing `manifest` `iterations` times always
    yields identical bytes (NFR-001, backed by the 1000x primitive check in
    Design.md §0.3-2)."""
    first = canonical_json_bytes(manifest)
    for _ in range(iterations):
        if canonical_json_bytes(manifest) != first:
            return False
    return True
