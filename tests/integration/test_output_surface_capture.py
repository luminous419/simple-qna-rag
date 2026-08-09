"""M4.1 §6.1 — dynamic capsys/caplog capture across the 5 request paths, plus
the real agent -> fallback -> web/retrieval code paths (CR-I1-MAJ-02 closure).

Complements the static AST audit (`test_logging_callsite_disposition.py`):
this test actually executes each request-handling scenario and inspects the
real stdout/stderr text for forbidden payload — catching f-string
combinations a static grep could miss. The `TestRealFallbackPathsNoLeak` and
`TestRealRagEngineNoLeak` classes below are the dynamic capture the Iteration
1 review found missing: they run the *real* `agent.route_query()` ->
`query_router.route_query()` -> `web_search.search_web()` fallback chain and
the real `RAGEngine.query()` retrieval/generation stages (mocking only the
true I/O boundary — DDGS, the LLM, the vectorstore) instead of mocking
`agent.route_query` itself away.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from simple_qna_rag.settings import Settings, SettingsError
from simple_qna_rag.web.server import create_app

_SECRET_QUESTION = "what is my extremely private secret question about launch codes?"
_SECRET_ANSWER = "the sensitive rag answer reveals launch codes here"
_SECRET_SOURCE_CONTENT = "raw doc excerpt with confidential figures"


def _ready_app():
    return create_app(
        settings_loader=lambda: Settings.from_sources(),
        engine_factory=lambda settings: object(),
    )


def _assert_clean(captured_text: str) -> None:
    assert _SECRET_QUESTION not in captured_text
    assert _SECRET_ANSWER not in captured_text
    assert _SECRET_SOURCE_CONTENT not in captured_text


def test_normal_rag_200_no_leak(capsys, monkeypatch):
    monkeypatch.setattr(
        "simple_qna_rag.agent.route_query",
        lambda question, **kwargs: {
            "answer": _SECRET_ANSWER,
            "sources": [{"source": "doc.pdf", "content": _SECRET_SOURCE_CONTENT}],
            "success": True,
            "search_type": "document_qa",
        },
    )
    with TestClient(_ready_app()) as client:
        r = client.post("/rag", json={"question": _SECRET_QUESTION})
    assert r.status_code == 200
    captured = capsys.readouterr()
    _assert_clean(captured.out + captured.err)


def test_nonexistent_route_404_no_leak(capsys):
    with TestClient(_ready_app()) as client:
        r = client.get(f"/does-not-exist?q={_SECRET_QUESTION}")
    assert r.status_code == 404
    captured = capsys.readouterr()
    _assert_clean(captured.out + captured.err)


def test_validation_failure_422_no_leak(capsys):
    with TestClient(_ready_app()) as client:
        r = client.post("/rag", json={"not_question": _SECRET_QUESTION})
    assert r.status_code == 422
    captured = capsys.readouterr()
    _assert_clean(captured.out + captured.err)


def test_engine_not_ready_503_no_leak(capsys):
    def _raise_settings():
        raise SettingsError("boom", exit_code=2)

    app = create_app(settings_loader=_raise_settings)
    with TestClient(app) as client:
        r = client.post("/rag", json={"question": _SECRET_QUESTION})
    assert r.status_code == 503
    captured = capsys.readouterr()
    _assert_clean(captured.out + captured.err)


def test_handler_internal_exception_500_no_leak(capsys, monkeypatch):
    def _boom(question, **kwargs):
        raise RuntimeError(_SECRET_ANSWER)

    monkeypatch.setattr("simple_qna_rag.agent.route_query", _boom)
    with TestClient(_ready_app(), raise_server_exceptions=False) as client:
        r = client.post("/rag", json={"question": _SECRET_QUESTION})
    assert r.status_code == 500
    captured = capsys.readouterr()
    _assert_clean(captured.out + captured.err)


# ---------------------------------------------------------------------------
# CR-I1-MAJ-02 closure — real agent.route_query() -> fallback -> web/retrieval
# dynamic capture. These tests do NOT mock `agent.route_query` itself; they
# mock only the true I/O boundary (DDGS, the LLM, the vectorstore/embeddings)
# so the actual product callsites the review flagged
# (`web_search.search_web`, `query_router.route_query`,
# `RAGEngine.initialize`/helpers, `RAGEngine.query`) really execute.
# ---------------------------------------------------------------------------

_SECRET_SEARCH_QUERY = "웹검색으로 극비-launch-code-9f3e 좀 찾아줘"
_SECRET_DDGS_TITLE = "극비 결과 제목 launch-code-leak-marker"
_SECRET_DDGS_SUMMARY = "극비 요약 confidential-summary-marker"
_SECRET_DDGS_URL = "https://example.com/secret-marker-path"
_SECRET_EXTRACTED_QUERY_FRAGMENT = "launch-code-9f3e"


def _mock_ddgs(results=None, side_effect=None):
    mock_ddgs_instance = MagicMock()
    if side_effect is not None:
        mock_ddgs_instance.__enter__.return_value.text.side_effect = side_effect
    else:
        mock_ddgs_instance.__enter__.return_value.text.return_value = results or []
    return mock_ddgs_instance


class TestRealFallbackPathsNoLeak:
    """Exercises `agent.route_query()`'s real keyword-router fallback ->
    `web_search.search_web()` chain (Design.md §6.1 REPLACE files:
    `agent.py`, `query_router.py`, `web_search.py`)."""

    def test_llm_router_failure_falls_back_to_real_keyword_router_web_search(
        self, capsys, monkeypatch
    ):
        """`_decide_tool` 예외 -> `keyword_fallback_route`(query_router.py 실제
        구현) -> `search_and_format`/`search_web`(web_search.py 실제 구현),
        DDGS만 mock. 민감 검색어/제목/요약/URL이 stdout에 전혀 없어야 한다."""
        from simple_qna_rag import agent

        monkeypatch.setattr(agent, "USE_WEB_SEARCH", True)
        monkeypatch.setattr("simple_qna_rag.query_router.USE_WEB_SEARCH", True)
        monkeypatch.setattr(
            agent, "_decide_tool", MagicMock(side_effect=RuntimeError("ollama down"))
        )
        monkeypatch.setattr(
            "simple_qna_rag.web_search.DDGS",
            lambda *a, **kw: _mock_ddgs(
                [{"href": _SECRET_DDGS_URL, "title": _SECRET_DDGS_TITLE, "body": _SECRET_DDGS_SUMMARY}]
            ),
        )

        result = agent.route_query(_SECRET_SEARCH_QUERY)

        assert result["success"] is True
        assert result["search_type"] == "web_search"
        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert _SECRET_EXTRACTED_QUERY_FRAGMENT not in text
        assert _SECRET_DDGS_TITLE not in text
        assert _SECRET_DDGS_SUMMARY not in text
        assert _SECRET_DDGS_URL not in text

    def test_web_search_upstream_exception_no_leak_and_retries_document_qa(
        self, capsys, monkeypatch
    ):
        """DDGS가 민감 정보를 담은 예외를 던져도 예외 원문이 stdout에 남지
        않아야 한다(web_search.search_web의 except 경로, Design.md §6.1)."""
        from simple_qna_rag import agent, rag_engine

        secret_exception_text = "connection failed for secret-internal-host-marker"

        monkeypatch.setattr(agent, "USE_WEB_SEARCH", True)
        monkeypatch.setattr(
            agent, "_decide_tool", lambda question: ("web_search", "정제된 검색어")
        )
        monkeypatch.setattr(
            "simple_qna_rag.web_search.DDGS",
            MagicMock(side_effect=RuntimeError(secret_exception_text)),
        )
        mock_engine = MagicMock()
        mock_engine.query.return_value = {
            "answer": "문서 기반 답변",
            "sources": [],
            "success": True,
            "intent": "other",
        }
        # The document_qa fallback goes through `rag_tool.func` ->
        # `tools.rag_function` -> `tools.get_rag_engine()`, not
        # `agent.get_rag_engine` — patching the module-global singleton
        # cache is the one seam every call site actually reads (see
        # test_metrics_live_traffic.py for the same fix).
        monkeypatch.setattr(rag_engine, "_rag_engine", mock_engine)

        result = agent.route_query(_SECRET_QUESTION)

        assert result["search_type"] == "document_qa"
        captured = capsys.readouterr()
        assert secret_exception_text not in (captured.out + captured.err)


class TestRealRagEngineNoLeak:
    """Exercises the real `RAGEngine.query()` retrieval/generation stages
    (Design.md §6.1 REPLACE file `rag_engine.py`) with sensitive
    question/context/document/exception payload — mocking only the LLM and
    the retrieval result, never `query()` itself."""

    def _make_engine(self):
        from simple_qna_rag.rag_engine import RAGEngine

        engine = object.__new__(RAGEngine)
        engine.intent_classifier_loaded = True
        engine._pending_fallback_events = []
        return engine

    def test_successful_query_with_sensitive_docs_no_leak(self, capsys, monkeypatch):
        from langchain_core.documents import Document
        from langchain_core.runnables import RunnableLambda
        from simple_qna_rag import rag_engine

        engine = self._make_engine()
        engine.llm = RunnableLambda(lambda prompt_value: _SECRET_ANSWER)
        docs = [Document(page_content=_SECRET_SOURCE_CONTENT, metadata={"source": "a.pdf"})]
        engine._retrieve_documents = lambda question: docs

        monkeypatch.setattr(rag_engine, "ANSWER_TEMPLATE_MODE", "default")
        result = engine.query(_SECRET_QUESTION)

        assert result["success"] is True
        captured = capsys.readouterr()
        _assert_clean(captured.out + captured.err)

    def test_retrieval_exception_with_sensitive_text_no_leak(self, capsys, monkeypatch):
        from simple_qna_rag import rag_engine

        secret_retrieval_exception = "vectorstore path leak /Users/marker/secret-vectorstore"
        engine = self._make_engine()
        engine.llm = object()

        def _boom(question):
            raise RuntimeError(secret_retrieval_exception)

        engine._retrieve_documents = _boom

        monkeypatch.setattr(rag_engine, "ANSWER_TEMPLATE_MODE", "default")
        result = engine.query(_SECRET_QUESTION)

        assert result["success"] is False
        captured = capsys.readouterr()
        # 응답 body(client 계약, REQ-006.1)에는 str(e)가 남아도 되지만 콘솔에는
        # 전혀 남으면 안 된다.
        assert secret_retrieval_exception not in (captured.out + captured.err)

    def test_generation_exception_with_sensitive_text_no_leak(self, capsys, monkeypatch):
        from simple_qna_rag import rag_engine

        secret_generation_exception = "ollama upstream secret-token-marker failure"
        engine = self._make_engine()

        class _RaisingLLM:
            def invoke(self, *_a, **_kw):
                raise RuntimeError(secret_generation_exception)

        engine.llm = _RaisingLLM()
        engine._retrieve_documents = lambda question: []

        monkeypatch.setattr(rag_engine, "ANSWER_TEMPLATE_MODE", "default")
        result = engine.query(_SECRET_QUESTION)

        assert result["success"] is False
        captured = capsys.readouterr()
        assert secret_generation_exception not in (captured.out + captured.err)
