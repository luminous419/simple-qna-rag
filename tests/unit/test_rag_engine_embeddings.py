"""M4.3 §5.2-a — `_build_embeddings` provider branch and Layer 2 physical seam."""

from __future__ import annotations

import sys

import pytest


def test_build_embeddings_default_uses_huggingface_provider():
    from langchain_huggingface import HuggingFaceEmbeddings

    from simple_qna_rag.rag_engine import _build_embeddings

    embeddings = _build_embeddings()
    assert isinstance(embeddings, HuggingFaceEmbeddings)


def test_build_embeddings_raises_seam_unavailable_when_module_absent(monkeypatch):
    from simple_qna_rag import rag_engine

    monkeypatch.setattr(rag_engine, "EMBEDDING_PROVIDER", "deterministic_test")
    monkeypatch.setitem(sys.modules, "simple_qna_rag_test_seam.deterministic_embeddings", None)
    monkeypatch.setitem(sys.modules, "simple_qna_rag_test_seam", None)

    with pytest.raises(rag_engine.TestEmbeddingSeamUnavailable) as excinfo:
        rag_engine._build_embeddings()
    assert excinfo.value.reason == "test_embedding_seam_unavailable"
