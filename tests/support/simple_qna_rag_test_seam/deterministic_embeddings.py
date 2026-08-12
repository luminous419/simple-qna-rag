"""Deterministic, network/model-free embedding test seam (Design.md §5.2-a).

Used only by `container_smoke.py`'s hosted Linux plumbing check
(build -> activate -> serve -> query 200) so it never needs to download or
initialize a real embedding model inside the container. Implements only the
LangChain `Embeddings` protocol (`embed_documents`/`embed_query`).

This file is intentionally outside `src/` — the production image never
COPYs `tests/`, so this module cannot be imported from a production
container without an explicit read-only bind mount + `PYTHONPATH`
(Design.md §7.5).
"""

from __future__ import annotations

import hashlib
import math


class DeterministicTestEmbeddings:
    DIMENSION = 32  # arbitrary fixed value, unrelated to any production model dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = (digest * ((self.DIMENSION // len(digest)) + 1))[: self.DIMENSION]
        vec = [b / 255.0 for b in raw]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
