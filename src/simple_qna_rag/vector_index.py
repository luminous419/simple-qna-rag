"""FAISS row <-> docstore Document mapping for MMR stored-vector reuse
(M3-REQ-002, Design.md §6.3).

`StoredVectorIndex.build()` runs the V1-V6 validation checks once at engine
initialization and is read-only afterwards (§3.6 — no lock needed). The
vectorstore's `.index` object is only used through the duck-typed
`ntotal`/`d`/`reconstruct` interface, never `import faiss` directly, so this
module can be unit-tested with a fake vectorstore (M3-NFR-004).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class VectorLookupError(RuntimeError):
    """질의 시 특정 후보 문서의 벡터를 안전하게 조회할 수 없을 때 던진다.
    `.reason`은 `"lookup_miss"` | `"dimension_mismatch"` | `"non_finite"` 중
    하나다. 호출부(`rag_engine._candidate_vectors()`)는 이 예외를 잡아 해당
    질의 전체를 legacy embed 경로로 폴백한다(부분 혼용 금지)."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or reason)


class VectorIndexValidationError(RuntimeError):
    """`StoredVectorIndex.build()`의 V1~V6 검증 실패. `.reason`은
    `"count_mismatch"` | `"key_mismatch"` | `"row_out_of_range"` |
    `"dimension_mismatch"` | `"semantic_mismatch"` | `"non_finite"` 중
    하나다. 이 예외가 나면 엔진은 `mmr_vector_source="embed"`로 강등한다."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or reason)


@dataclass(frozen=True)
class VectorIndexStats:
    document_count: int
    dimension: int
    validated_samples: int
    min_sample_cosine: float


_CANARY = "vector_index_dimension_canary"


class StoredVectorIndex:
    """FAISS row <-> docstore Document 매핑을 1회 구축·검증하고, 이후
    읽기 전용으로 후보 `Document`의 저장 벡터를 돌려준다."""

    def __init__(
        self,
        *,
        row_by_object_id: dict[int, int],
        docstore_dict: dict,
        reconstruct: Any,
        dimension: int,
        stats: VectorIndexStats,
    ) -> None:
        self._row_by_object_id = row_by_object_id
        # docstore_dict에 대한 강한 참조를 보유해, 이 인스턴스가 살아있는 동안
        # 매핑에 쓰인 Document 객체들의 id()가 재사용되지 않도록 보장한다.
        self._docstore_dict = docstore_dict
        self._reconstruct = reconstruct
        self._dimension = dimension
        self._stats = stats

    @property
    def stats(self) -> VectorIndexStats:
        return self._stats

    @classmethod
    def build(cls, vectorstore, *, sample_size: int, cosine_floor: float) -> "StoredVectorIndex":
        if sample_size < 1:
            raise VectorIndexValidationError("invalid_sample_size", "sample_size는 1 이상이어야 합니다")

        index = vectorstore.index
        docstore_dict: dict = vectorstore.docstore._dict
        index_to_docstore_id: dict[int, str] = vectorstore.index_to_docstore_id

        ntotal = index.ntotal
        # V1: count 일치
        if not (ntotal == len(docstore_dict) == len(index_to_docstore_id)):
            raise VectorIndexValidationError(
                "count_mismatch",
                f"ntotal={ntotal} len(docstore)={len(docstore_dict)} "
                f"len(index_to_docstore_id)={len(index_to_docstore_id)}",
            )

        # V2: key 집합 일치, row 값 중복 없음
        mapped_keys = set(index_to_docstore_id.values())
        if mapped_keys != set(docstore_dict.keys()):
            raise VectorIndexValidationError("key_mismatch")
        rows = list(index_to_docstore_id.keys())
        if len(set(rows)) != len(rows):
            raise VectorIndexValidationError("key_mismatch", "row 값이 중복됩니다")

        # V3: row 범위
        for row in rows:
            if not (isinstance(row, int) and 0 <= row < ntotal):
                raise VectorIndexValidationError("row_out_of_range", f"row={row!r}")

        # V4: 차원 일치
        embeddings = vectorstore.embeddings
        canary_vector = embeddings.embed_query(_CANARY)
        query_dim = len(canary_vector)
        index_dim = index.d
        if index_dim != query_dim:
            raise VectorIndexValidationError(
                "dimension_mismatch", f"index.d={index_dim} embed_query dim={query_dim}"
            )

        row_by_object_id: dict[int, int] = {}
        for row, docstore_id in index_to_docstore_id.items():
            document = docstore_dict[docstore_id]
            row_by_object_id[id(document)] = row

        # V5: 결정론적 표본 검증(cosine 유사도 >= cosine_floor)
        sorted_ids = sorted(docstore_dict.keys())
        n = len(sorted_ids)
        actual_sample_size = min(sample_size, n)
        if actual_sample_size == 0:
            raise VectorIndexValidationError("semantic_mismatch", "검증할 문서가 없습니다")
        if actual_sample_size == 1:
            sample_ids = [sorted_ids[0]]
        else:
            step = n / actual_sample_size
            sample_ids = [sorted_ids[min(int(i * step), n - 1)] for i in range(actual_sample_size)]

        min_cosine = math.inf
        for docstore_id in sample_ids:
            document = docstore_dict[docstore_id]
            row = index_to_docstore_id_reverse_lookup(index_to_docstore_id, docstore_id)
            stored_vector = index.reconstruct(row)
            # V6: 유한값
            if not all(math.isfinite(float(x)) for x in stored_vector):
                raise VectorIndexValidationError("non_finite", f"docstore_id={docstore_id}")
            fresh_vector = embeddings.embed_query(document.page_content)
            if not all(math.isfinite(float(x)) for x in fresh_vector):
                raise VectorIndexValidationError("non_finite", f"docstore_id={docstore_id} (fresh)")
            cosine = _cosine(stored_vector, fresh_vector)
            min_cosine = min(min_cosine, cosine)
            if cosine < cosine_floor:
                raise VectorIndexValidationError(
                    "semantic_mismatch", f"docstore_id={docstore_id} cosine={cosine:.4f} < {cosine_floor}"
                )

        stats = VectorIndexStats(
            document_count=ntotal,
            dimension=index_dim,
            validated_samples=actual_sample_size,
            min_sample_cosine=min_cosine,
        )
        return cls(
            row_by_object_id=row_by_object_id,
            docstore_dict=docstore_dict,
            reconstruct=index.reconstruct,
            dimension=index_dim,
            stats=stats,
        )

    def row_for(self, document) -> int:
        row = self._row_by_object_id.get(id(document))
        if row is None:
            raise VectorLookupError("lookup_miss", f"document id={id(document)!r}가 매핑에 없습니다")
        return row

    def vectors_for(self, documents) -> "Any":
        import numpy as np

        rows = [self.row_for(doc) for doc in documents]
        vectors = np.empty((len(rows), self._dimension), dtype=np.float64)
        for i, row in enumerate(rows):
            raw = self._reconstruct(row)
            vec = np.asarray(raw, dtype=np.float64)
            if vec.shape != (self._dimension,):
                raise VectorLookupError("dimension_mismatch", f"row={row} shape={vec.shape}")
            if not np.all(np.isfinite(vec)):
                raise VectorLookupError("non_finite", f"row={row}")
            vectors[i] = vec
        return vectors


def index_to_docstore_id_reverse_lookup(index_to_docstore_id: dict[int, str], docstore_id: str) -> int:
    """`index_to_docstore_id`는 row -> docstore_id다. row 수가 작으므로(≤ 수백)
    검증 시점의 표본 조회 몇 회는 선형 탐색으로 충분하고, `build()` 외에는
    호출되지 않는다(런타임 질의 경로는 `row_by_object_id`만 쓴다)."""
    for row, did in index_to_docstore_id.items():
        if did == docstore_id:
            return row
    raise VectorIndexValidationError("key_mismatch", f"docstore_id={docstore_id}에 대응하는 row가 없습니다")


def _cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
