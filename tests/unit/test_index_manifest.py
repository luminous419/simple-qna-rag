"""M4.3-REQ-001 — canonical manifest schema/version-id/parser tests."""

from __future__ import annotations

import copy
import json

import pytest

from simple_qna_rag.index.manifest import (
    ManifestSchemaError,
    ManifestValueError,
    build_manifest,
    canonical_json_bytes,
    derive_version_id,
    parse_manifest,
    round_trip_stable,
)

_VALID_IDENTITY = {
    "corpus_manifest_sha256": "a" * 64,
    "source_document_count": 3,
    "chunk_count": 10,
    "embedding_model_name": "BAAI/bge-m3",
    "embedding_model_revision": "unknown",
    "embedding_provider": "huggingface",
    "normalize_embeddings": True,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "faiss_index_type": "IndexFlatIP",
    "faiss_dimension": 1024,
    "settings_hash": "b" * 64,
    "dependency_lock_sha256": "c" * 64,
    "builder_git_sha": "d" * 40,
    "builder_git_dirty": False,
    "index_faiss": {"size_bytes": 100, "sha256": "e" * 64},
    "index_pkl": {"size_bytes": 200, "sha256": "f" * 64},
    "source": "build",
    "legacy_baseline_id": None,
}


def _valid_manifest() -> dict:
    return build_manifest(dict(_VALID_IDENTITY), created_at="2026-08-12T00:00:00Z")


def test_canonical_round_trip_100x():
    manifest = _valid_manifest()
    assert round_trip_stable(manifest, iterations=100)


def test_version_id_is_deterministic_and_excludes_created_at():
    m1 = build_manifest(dict(_VALID_IDENTITY), created_at="2026-08-12T00:00:00Z")
    m2 = build_manifest(dict(_VALID_IDENTITY), created_at="2030-01-01T00:00:00Z")
    assert m1["version_id"] == m2["version_id"]


def test_build_manifest_rejects_unknown_or_missing_identity_key():
    bad = dict(_VALID_IDENTITY)
    bad["extra_field"] = 1
    with pytest.raises(ManifestSchemaError):
        build_manifest(bad, created_at="2026-08-12T00:00:00Z")
    missing = dict(_VALID_IDENTITY)
    del missing["chunk_count"]
    with pytest.raises(ManifestSchemaError):
        build_manifest(missing, created_at="2026-08-12T00:00:00Z")


def test_parse_manifest_round_trips_valid_bytes():
    manifest = _valid_manifest()
    raw = canonical_json_bytes(manifest) + b"\n"
    parsed = parse_manifest(raw)
    assert parsed == manifest


@pytest.mark.parametrize(
    "mutate, error_cls",
    [
        (lambda d: d.pop("chunk_count"), ManifestSchemaError),
        (lambda d: d.update(unknown_key=1), ManifestSchemaError),
        (lambda d: d.update(schema_version=99), ManifestSchemaError),
        (lambda d: d.update(chunk_size=0), ManifestValueError),
        (lambda d: d.update(chunk_size="1000"), ManifestSchemaError),
        (lambda d: d.update(normalize_embeddings=1), ManifestSchemaError),
        (lambda d: d.update(embedding_provider="bogus"), ManifestValueError),
        (lambda d: d.update(settings_hash="not-hex"), ManifestValueError),
        (lambda d: d.update(index_faiss={"size_bytes": True, "sha256": "e" * 64}), ManifestValueError),
        (lambda d: d.update(index_faiss={"size_bytes": 1}), ManifestSchemaError),
        (lambda d: d.update(source="bogus"), ManifestValueError),
        (lambda d: d.update(version_id="not16hex"), ManifestValueError),
    ],
)
def test_parse_manifest_rejects_malformed_fields(mutate, error_cls):
    manifest = _valid_manifest()
    mutated = copy.deepcopy(manifest)
    mutate(mutated)
    raw = json.dumps(mutated, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with pytest.raises(error_cls):
        parse_manifest(raw)


def test_parse_manifest_rejects_self_hash_mismatch_after_tamper():
    manifest = _valid_manifest()
    tampered = dict(manifest)
    tampered["chunk_count"] = manifest["chunk_count"] + 1
    raw = canonical_json_bytes(tampered)
    with pytest.raises(ManifestValueError):
        parse_manifest(raw)


def test_parse_manifest_rejects_non_finite_json():
    manifest = _valid_manifest()
    raw = canonical_json_bytes(manifest).decode("utf-8")
    raw = raw.replace('"chunk_size":1000', '"chunk_size":NaN')
    with pytest.raises(ManifestValueError):
        parse_manifest(raw.encode("utf-8"))


def test_legacy_import_requires_baseline_id_and_null_corpus_hash():
    identity = dict(_VALID_IDENTITY)
    identity["source"] = "legacy_import"
    identity["legacy_baseline_id"] = "m3_initial"
    identity["corpus_manifest_sha256"] = None
    manifest = build_manifest(identity, created_at="2026-08-12T00:00:00Z")
    raw = canonical_json_bytes(manifest)
    parsed = parse_manifest(raw)
    assert parsed["source"] == "legacy_import"

    bad = dict(manifest)
    bad["legacy_baseline_id"] = None
    with pytest.raises(ManifestValueError):
        parse_manifest(canonical_json_bytes(bad))
