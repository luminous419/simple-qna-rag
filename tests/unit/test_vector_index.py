"""Tests for src/simple_qna_rag/vector_index.py (M3-REQ-002, Design.md §6.3).

Uses a fake vectorstore (duck-typed .index/.docstore/.index_to_docstore_id/
.embeddings) — no FAISS/model/network required (M3-NFR-004).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from langchain_core.documents import Document

from simple_qna_rag.vector_index import (
    StoredVectorIndex,
    VectorIndexValidationError,
    VectorLookupError,
)


class _FakeIndex:
    def __init__(self, vectors: dict[int, list[float]], dimension: int):
        self._vectors = vectors
        self.ntotal = len(vectors)
        self.d = dimension

    def reconstruct(self, row: int):
        return list(self._vectors[row])


class _FakeDocstore:
    def __init__(self, docs: dict[str, Document]):
        self._dict = docs


class _FakeEmbeddings:
    """embed_query()가 문서 page_content를 그대로 저장된 벡터와 동일하게
    반환하도록 구성 가능한 더미. dimension_canary 처리도 지원한다."""

    def __init__(self, content_to_vector: dict[str, list[float]], dimension: int):
        self._map = content_to_vector
        self._dimension = dimension

    def embed_query(self, text: str):
        if text in self._map:
            return list(self._map[text])
        # canary or unknown text: return a deterministic unit vector of the
        # right dimension.
        vec = [0.0] * self._dimension
        vec[0] = 1.0
        return vec


class _FakeVectorstore:
    def __init__(self, index, docstore, index_to_docstore_id, embeddings):
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.embeddings = embeddings


def _make_valid_vectorstore(n: int = 4, dimension: int = 3):
    """정상적으로 검증을 통과하는 fake vectorstore를 만든다. 문서 i의 저장
    벡터와 embed_query(page_content) 결과가 완전히 동일해 cosine==1.0이다."""
    docs = {}
    vectors = {}
    content_to_vector = {}
    index_to_docstore_id = {}
    for i in range(n):
        docstore_id = f"id{i}"
        content = f"content-{i}"
        vec = [0.0] * dimension
        vec[i % dimension] = 1.0
        doc = Document(page_content=content, metadata={"source": f"s{i}.pdf"})
        docs[docstore_id] = doc
        vectors[i] = vec
        content_to_vector[content] = vec
        index_to_docstore_id[i] = docstore_id

    index = _FakeIndex(vectors, dimension)
    docstore = _FakeDocstore(docs)
    embeddings = _FakeEmbeddings(content_to_vector, dimension)
    vectorstore = _FakeVectorstore(index, docstore, index_to_docstore_id, embeddings)
    return vectorstore, docs


def test_build_succeeds_on_valid_vectorstore():
    vectorstore, docs = _make_valid_vectorstore()
    idx = StoredVectorIndex.build(vectorstore, sample_size=2, cosine_floor=0.99)
    assert idx.stats.document_count == 4
    assert idx.stats.dimension == 3
    assert idx.stats.validated_samples == 2
    assert idx.stats.min_sample_cosine >= 0.99


def test_row_for_and_vectors_for_preserve_order():
    vectorstore, docs = _make_valid_vectorstore()
    idx = StoredVectorIndex.build(vectorstore, sample_size=1, cosine_floor=0.99)
    doc_list = [docs["id2"], docs["id0"], docs["id3"]]
    vectors = idx.vectors_for(doc_list)
    assert vectors.dtype == np.float64
    assert vectors.shape == (3, 3)
    np.testing.assert_array_equal(vectors[0], [0.0, 0.0, 1.0])  # id2 -> index 2
    np.testing.assert_array_equal(vectors[1], [1.0, 0.0, 0.0])  # id0 -> index 0


def test_row_for_unregistered_document_raises_lookup_miss():
    vectorstore, docs = _make_valid_vectorstore()
    idx = StoredVectorIndex.build(vectorstore, sample_size=1, cosine_floor=0.99)
    stray = Document(page_content="not registered", metadata={})
    with pytest.raises(VectorLookupError) as exc_info:
        idx.row_for(stray)
    assert exc_info.value.reason == "lookup_miss"


def test_vectors_for_raises_lookup_miss_for_unregistered_document():
    vectorstore, docs = _make_valid_vectorstore()
    idx = StoredVectorIndex.build(vectorstore, sample_size=1, cosine_floor=0.99)
    stray = Document(page_content="not registered", metadata={})
    with pytest.raises(VectorLookupError) as exc_info:
        idx.vectors_for([docs["id0"], stray])
    assert exc_info.value.reason == "lookup_miss"


# ---------------------------------------------------------------------------
# V1-V6 validation failure modes
# ---------------------------------------------------------------------------


def test_v1_count_mismatch_raises():
    vectorstore, docs = _make_valid_vectorstore(n=4)
    # Sabotage: pretend ntotal is different from docstore size.
    vectorstore.index.ntotal = 99
    with pytest.raises(VectorIndexValidationError) as exc_info:
        StoredVectorIndex.build(vectorstore, sample_size=1, cosine_floor=0.99)
    assert exc_info.value.reason == "count_mismatch"


def test_v2_key_mismatch_raises():
    vectorstore, docs = _make_valid_vectorstore(n=4)
    # Sabotage: index_to_docstore_id points to a docstore_id not in docstore.
    vectorstore.index_to_docstore_id[0] = "does-not-exist"
    with pytest.raises(VectorIndexValidationError) as exc_info:
        StoredVectorIndex.build(vectorstore, sample_size=1, cosine_floor=0.99)
    assert exc_info.value.reason == "key_mismatch"


def test_v3_row_out_of_range_raises():
    vectorstore, docs = _make_valid_vectorstore(n=4)
    # Sabotage: a row index outside [0, ntotal).
    last_key = max(vectorstore.index_to_docstore_id)
    docstore_id = vectorstore.index_to_docstore_id.pop(last_key)
    vectorstore.index_to_docstore_id[999] = docstore_id
    with pytest.raises(VectorIndexValidationError) as exc_info:
        StoredVectorIndex.build(vectorstore, sample_size=1, cosine_floor=0.99)
    assert exc_info.value.reason in ("row_out_of_range", "key_mismatch")


def test_v4_dimension_mismatch_raises():
    vectorstore, docs = _make_valid_vectorstore(n=4, dimension=3)
    vectorstore.index.d = 5  # mismatched vs embed_query dimension (3)
    with pytest.raises(VectorIndexValidationError) as exc_info:
        StoredVectorIndex.build(vectorstore, sample_size=1, cosine_floor=0.99)
    assert exc_info.value.reason == "dimension_mismatch"


def test_v5_semantic_mismatch_raises_when_cosine_below_floor():
    vectorstore, docs = _make_valid_vectorstore(n=4, dimension=3)
    # Sabotage: make the fresh embedding for one document orthogonal to its
    # stored vector.
    vectorstore.embeddings._map["content-0"] = [0.0, 1.0, 0.0]  # stored is [1,0,0]
    with pytest.raises(VectorIndexValidationError) as exc_info:
        StoredVectorIndex.build(vectorstore, sample_size=4, cosine_floor=0.99)
    assert exc_info.value.reason == "semantic_mismatch"


def test_v6_non_finite_stored_vector_raises():
    vectorstore, docs = _make_valid_vectorstore(n=4, dimension=3)
    vectorstore.index._vectors[0] = [math.nan, 0.0, 0.0]
    with pytest.raises(VectorIndexValidationError) as exc_info:
        StoredVectorIndex.build(vectorstore, sample_size=4, cosine_floor=0.99)
    assert exc_info.value.reason == "non_finite"


def test_sample_size_zero_rejected():
    vectorstore, docs = _make_valid_vectorstore()
    with pytest.raises(VectorIndexValidationError):
        StoredVectorIndex.build(vectorstore, sample_size=0, cosine_floor=0.99)


def test_deterministic_sampling_is_evenly_spaced():
    # Sampling must be deterministic (sorted docstore keys, evenly spaced) so
    # repeated builds validate the same documents.
    vectorstore, docs = _make_valid_vectorstore(n=10, dimension=3)
    idx1 = StoredVectorIndex.build(vectorstore, sample_size=3, cosine_floor=0.99)
    idx2 = StoredVectorIndex.build(vectorstore, sample_size=3, cosine_floor=0.99)
    assert idx1.stats.min_sample_cosine == idx2.stats.min_sample_cosine
