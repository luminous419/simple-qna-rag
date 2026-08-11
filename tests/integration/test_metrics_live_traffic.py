"""M4.1 REQ-004.1 closure — CR-I1-MAJ-03.

Iteration 1 review: the existing 1000-sample cardinality test
(`test_observability_metrics.py::test_1000_unique_requests_sample_count_is_bounded_and_equals_102`)
pokes the registry directly, which proves the *schema* is bounded but not
that any real product code path actually increments it. This test instead
drives 1000 unique payloads through the real FastAPI `/rag` endpoint and the
real `agent.route_query()` -> `RAGEngine.query()` / `web_search` fallback
wiring (mocking only the true I/O boundary: the LLM tool-selection call and
DDGS), then scrapes `/metrics` and asserts both the bounded sample count and
that routing/retrieval/generation/web_search stage duration and the
web_search fallback counter actually moved.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from prometheus_client.parser import text_string_to_metric_families

from simple_qna_rag import agent, rag_engine
from simple_qna_rag.settings import Settings, get_settings
from simple_qna_rag.web.server import create_app

REQUEST_COUNT = 1000


def _fake_engine() -> rag_engine.RAGEngine:
    """A real `RAGEngine.query()` bound to a fast fake LLM/retrieval — not a
    mock of `query()` itself, so retrieval/generation stage instrumentation
    inside `query()` really executes."""
    engine = object.__new__(rag_engine.RAGEngine)
    engine.intent_classifier_loaded = True
    engine._pending_fallback_events = []
    engine.llm = RunnableLambda(lambda _: "fixed-answer")
    engine._retrieve_documents = lambda question: [
        Document(page_content="doc content", metadata={"source": "a.pdf"})
    ]
    return engine


def _ready_app():
    return create_app(
        settings_loader=get_settings,
        engine_factory=lambda settings: object(),
    )


def _sample(families, name: str, labels: dict) -> float | None:
    for family in families:
        for s in family.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                return s.value
    return None


def test_1000_unique_real_requests_bounded_scrape_and_counters_increment(monkeypatch):
    monkeypatch.setattr(rag_engine, "ANSWER_TEMPLATE_MODE", "default")
    # `get_rag_engine()`'s module-global singleton cache (`rag_engine._rag_engine`)
    # is the one true seam every caller (agent.py, tools.py, query_router.py —
    # each holds its own `from ... import get_rag_engine` binding) reads
    # through. Pre-seeding it here — rather than monkeypatching each import
    # site's `get_rag_engine` name separately — guarantees none of those
    # call sites falls through to constructing/initializing the real
    # singleton (real FAISS load + real Ollama call) if a patch is missed.
    monkeypatch.setattr(rag_engine, "_rag_engine", _fake_engine())
    monkeypatch.setattr(agent, "USE_WEB_SEARCH", True)

    calls = {"n": 0}

    def _fake_decide_tool(question: str):
        calls["n"] += 1
        # 1-in-5 requests route to web_search, exercising that stage.
        if calls["n"] % 5 == 0:
            return "web_search", question
        return "document_qa", question

    monkeypatch.setattr(agent, "_decide_tool", _fake_decide_tool)

    def _fake_ddgs(*_a, **_kw):
        instance = MagicMock()
        # Every 3rd web_search call returns no results, forcing the
        # web_search -> document_qa fallback (rag_fallback_total kind="web_search").
        if calls["n"] % 15 == 0:
            instance.__enter__.return_value.text.return_value = []
        else:
            instance.__enter__.return_value.text.return_value = [
                {"href": "https://example.com", "title": "t", "body": "b"}
            ]
        return instance

    monkeypatch.setattr("simple_qna_rag.web_search.DDGS", _fake_ddgs)

    with TestClient(_ready_app()) as client:
        for i in range(REQUEST_COUNT):
            r = client.post("/rag", json={"question": f"unique payload #{i}"})
            assert r.status_code == 200, r.text
        metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 200
    families = list(text_string_to_metric_families(metrics_response.text))
    sample_count = sum(len(f.samples) for f in families)

    assert sample_count <= 150
    assert _sample(families, "rag_requests_total", {"route": "rag", "status": "2xx"}) == REQUEST_COUNT
    assert (_sample(families, "rag_stage_duration_seconds_count", {"stage": "routing"}) or 0) > 0
    assert (_sample(families, "rag_stage_duration_seconds_count", {"stage": "retrieval"}) or 0) > 0
    assert (_sample(families, "rag_stage_duration_seconds_count", {"stage": "generation"}) or 0) > 0
    assert (_sample(families, "rag_stage_duration_seconds_count", {"stage": "web_search"}) or 0) > 0
    assert (
        _sample(families, "rag_fallback_total", {"kind": "web_search", "reason": "other"}) or 0
    ) > 0
