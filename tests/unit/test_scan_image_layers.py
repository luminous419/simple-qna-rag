"""M4.3-REQ-005.4 — layer scanner positive/negative/traversal/whiteout fixtures."""

from __future__ import annotations

import io
import tarfile

import pytest

from scripts import scan_image_layers as scanner


def _make_layer_tar(names: list[str]) -> tarfile.TarFile:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name in names:
            data = b"x"
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return tarfile.open(fileobj=buf)


def test_forbidden_layer_detects_multiple_categories():
    layer = _make_layer_tar([".git/HEAD", "runtime/vectorstore/index.faiss", "id_rsa"])
    violations = []
    for member in layer.getmembers():
        hit = scanner.classify_member(member.name)
        if hit:
            violations.append(hit[0])
    assert set(violations) == {"vcs_directory", "index_artifact", "credential"}


def test_clean_layer_has_no_violations():
    layer = _make_layer_tar(["src/simple_qna_rag/__init__.py", "README.md"])
    for member in layer.getmembers():
        assert scanner.classify_member(member.name) is None


def test_clean_web_asset_layer_has_no_violations():
    layer = _make_layer_tar(["web/static/style.css", "web/templates/index.html"])
    for member in layer.getmembers():
        assert scanner.classify_member(member.name) is None


def test_traversal_member_detected():
    layer = _make_layer_tar(["../../etc/passwd"])
    member = layer.getmembers()[0]
    hit = scanner.classify_member(member.name)
    assert hit is not None
    assert hit[0] == "path_traversal"


def test_whiteout_only_layer_has_zero_violations():
    layer = _make_layer_tar([".wh..wh..opq", "runtime/.wh.vectorstore"])
    violations = []
    for member in layer.getmembers():
        if scanner.is_whiteout(member.name):
            continue
        hit = scanner.classify_member(member.name)
        if hit:
            violations.append(hit)
    assert violations == []


def test_test_seam_leak_layer_detected():
    layer = _make_layer_tar(["tests/support/simple_qna_rag_test_seam/deterministic_embeddings.py"])
    member = layer.getmembers()[0]
    hit = scanner.classify_member(member.name)
    assert hit is not None
    assert hit[0] == "test_embedding_seam"


def test_positive_negative_traversal_whiteout_fixtures():
    """Single node id referenced by run_m43_acceptance.py's PROFILE_NODE_IDS
    — exercises the five fixture categories above in one collected test."""
    test_forbidden_layer_detects_multiple_categories()
    test_clean_layer_has_no_violations()
    test_traversal_member_detected()
    test_whiteout_only_layer_has_zero_violations()
    test_test_seam_leak_layer_detected()
