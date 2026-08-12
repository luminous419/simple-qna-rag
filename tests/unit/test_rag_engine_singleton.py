"""CR-I5-MAJ-02 — RAGEngine singleton failure-state and artifact-reason
allowlist regression tests (Code_Review_Iteration_5.md).

Code_Review_Iteration_5.md found that a failed `RAGEngine` construction
survived in the class-level `_instance` singleton even though
`get_rag_engine()` already avoided caching it in the module-level
`_rag_engine` global, and that `_artifact_error_reason` was never reset
at the top of a later attempt on the same object — so an artifact
failure followed by an unrelated ordinary failure could be misreported
as the earlier artifact failure. It also found `EngineArtifactError`
accepted any string and that `server.py` classified any exception
carrying a `.reason` attribute as an artifact failure. This module
proves: a failed construction never survives in either singleton layer,
`_artifact_error_reason` never leaks across attempts, a retry after
failure always gets a fresh `RAGEngine` identity, and only the public
`index_verification.REASONS` allowlist can ever become a disclosed
`EngineArtifactError.reason`.
"""

from __future__ import annotations

import pytest

from simple_qna_rag import rag_engine
from simple_qna_rag.index import verification as index_verification
from simple_qna_rag.rag_engine import EngineArtifactError, RAGEngine


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Isolate every test in this module from the process-wide RAGEngine
    class singleton and module-level cache, both before and after — a
    failing assertion could otherwise leak broken state into unrelated
    later tests in the same process."""
    RAGEngine._instance = None
    RAGEngine._initialized = False
    rag_engine._rag_engine = None
    yield
    RAGEngine._instance = None
    RAGEngine._initialized = False
    rag_engine._rag_engine = None


def _fail_with_reason(monkeypatch, reason, seen_ids=None):
    def _initialize(self):
        if seen_ids is not None:
            seen_ids.append(id(self))
        self._artifact_error_reason = reason
        return False

    monkeypatch.setattr(RAGEngine, "initialize", _initialize)


def _fail_ordinary(monkeypatch, seen_ids=None):
    def _initialize(self):
        if seen_ids is not None:
            seen_ids.append(id(self))
        return False

    monkeypatch.setattr(RAGEngine, "initialize", _initialize)


def _succeed(monkeypatch, seen_ids=None):
    def _initialize(self):
        if seen_ids is not None:
            seen_ids.append(id(self))
        return True

    monkeypatch.setattr(RAGEngine, "initialize", _initialize)


def test_artifact_failure_discards_both_singleton_layers(monkeypatch):
    _fail_with_reason(monkeypatch, "test_embedding_seam_unavailable")
    with pytest.raises(EngineArtifactError) as excinfo:
        rag_engine.get_rag_engine()
    assert excinfo.value.reason == "test_embedding_seam_unavailable"
    assert RAGEngine._instance is None
    assert RAGEngine._initialized is False
    assert rag_engine._rag_engine is None


def test_artifact_then_retry_ordinary_failure_yields_ordinary_engine_init_failed(monkeypatch):
    """The core CR-I5-MAJ-02 regression: an artifact failure followed by
    an unrelated ordinary failure on retry must report plain
    `engine_init_failed` (via a bare RuntimeError, not EngineArtifactError),
    never the stale artifact reason from the first attempt."""
    _fail_with_reason(monkeypatch, "test_embedding_seam_unavailable")
    with pytest.raises(EngineArtifactError):
        rag_engine.get_rag_engine()

    _fail_ordinary(monkeypatch)
    with pytest.raises(RuntimeError) as excinfo:
        rag_engine.get_rag_engine()
    assert not isinstance(excinfo.value, EngineArtifactError)
    assert str(excinfo.value) == "RAG 엔진 초기화 실패"


def test_artifact_then_retry_success_uses_fresh_identity(monkeypatch):
    """A successful retry after an artifact failure must construct a
    genuinely new RAGEngine object, not resurrect the failed one."""
    seen_ids: list[int] = []

    _fail_with_reason(monkeypatch, "test_embedding_seam_unavailable", seen_ids)
    with pytest.raises(EngineArtifactError):
        rag_engine.get_rag_engine()

    _succeed(monkeypatch, seen_ids)
    engine = rag_engine.get_rag_engine()
    seen_ids.append(id(engine))

    assert len(seen_ids) == 3
    assert seen_ids[0] != seen_ids[1], "retry must construct a fresh RAGEngine identity"
    assert seen_ids[1] == seen_ids[2]
    assert engine._artifact_error_reason is None
    assert engine is rag_engine._rag_engine


def test_ordinary_failure_then_retry_success_is_unaffected(monkeypatch):
    """CR-I5-MAJ-02 must not change the pre-existing ordinary
    failure/retry contract — only the artifact-reason bug is fixed."""
    seen_ids: list[int] = []

    _fail_ordinary(monkeypatch, seen_ids)
    with pytest.raises(RuntimeError) as excinfo:
        rag_engine.get_rag_engine()
    assert not isinstance(excinfo.value, EngineArtifactError)
    assert str(excinfo.value) == "RAG 엔진 초기화 실패"

    _succeed(monkeypatch, seen_ids)
    engine = rag_engine.get_rag_engine()
    seen_ids.append(id(engine))

    assert seen_ids[0] != seen_ids[1], "retry must construct a fresh RAGEngine identity"
    assert engine._artifact_error_reason is None


def test_path_like_reason_never_becomes_artifact_error(monkeypatch):
    """A reason outside the public allowlist (e.g. path-like text) must
    degrade to the ordinary un-disclosed failure path, never leak as a
    disclosed artifact reason."""
    _fail_with_reason(monkeypatch, "../../etc/passwd")
    with pytest.raises(RuntimeError) as excinfo:
        rag_engine.get_rag_engine()
    assert not isinstance(excinfo.value, EngineArtifactError)
    assert str(excinfo.value) == "RAG 엔진 초기화 실패"


def test_arbitrary_unallowlisted_reason_never_becomes_artifact_error(monkeypatch):
    _fail_with_reason(monkeypatch, "totally_made_up_reason_not_in_allowlist")
    with pytest.raises(RuntimeError) as excinfo:
        rag_engine.get_rag_engine()
    assert not isinstance(excinfo.value, EngineArtifactError)


def test_engine_artifact_error_rejects_reason_outside_allowlist():
    for bad_reason in (
        "../../etc/passwd",
        "API_TOKEN=supersecret",
        "",
        "member_hash_mismatch_but_misspelled",
    ):
        with pytest.raises(ValueError):
            EngineArtifactError(bad_reason)


def test_engine_artifact_error_accepts_every_allowlisted_reason():
    for reason in index_verification.REASONS:
        err = EngineArtifactError(reason)
        assert err.reason == reason
        assert str(err) == reason


def test_initialize_resets_stale_artifact_reason_before_each_attempt(monkeypatch):
    """Defense-in-depth layer independent of get_rag_engine()'s singleton
    discard: initialize() itself must clear any reason left over from a
    previous attempt on the same object before an unrelated ordinary
    failure this attempt."""
    engine = object.__new__(RAGEngine)
    engine.vectorstore = None
    engine.bm25_retriever = None
    engine.dense_retriever = None
    engine.llm = None
    engine.intent_classifier_loaded = False
    engine.stored_vector_index = None
    engine.mmr_vector_status = {"source": "embed", "reason": None}
    engine._pending_fallback_events = []
    engine._artifact_error_reason = "test_embedding_seam_unavailable"  # stale

    def _raise_ordinary():
        raise RuntimeError("unrelated ordinary failure")

    monkeypatch.setattr(engine, "_load_vectorstore", _raise_ordinary)

    assert engine.initialize() is False
    assert engine._artifact_error_reason is None
