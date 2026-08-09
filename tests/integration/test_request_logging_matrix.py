"""M4.1 §9.1 — request start/end event ownership across 5 response paths."""

import json
import logging as stdlib_logging

import pytest
from fastapi.testclient import TestClient

from simple_qna_rag.observability import logging as L
from simple_qna_rag.settings import Settings, SettingsError
from simple_qna_rag.web.server import create_app

_FORBIDDEN_KEYS = {"question", "answer", "context", "sources_content", "prompt", "exception_text"}


class _ListHandler(stdlib_logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        payload = record.__dict__.get("event_payload")
        if payload is not None:
            self.records.append(payload)


@pytest.fixture
def sink():
    handler = _ListHandler()
    original = L._logger.handlers
    L._logger.handlers = [handler]
    yield handler
    L._logger.handlers = original


def _events_for(sink, event_name):
    return [r for r in sink.records if r["event"] == event_name]


def _ready_app(engine_obj=object()):
    return create_app(
        settings_loader=lambda: Settings.from_sources(),
        engine_factory=lambda settings: engine_obj,
    )


def test_row1_normal_rag_200(sink, monkeypatch):
    monkeypatch.setattr(
        "simple_qna_rag.agent.route_query",
        lambda question, **kwargs: {"answer": "ok", "sources": [], "success": True, "search_type": "document_qa"},
    )
    with TestClient(_ready_app()) as client:
        r = client.post("/rag", json={"question": "hello"})
    assert r.status_code == 200
    starts = _events_for(sink, "request_start")
    ends = _events_for(sink, "request_end")
    assert len(starts) == 1
    assert len(ends) == 1
    assert ends[0]["status_code"] == 200


def test_row2_nonexistent_route_404(sink):
    with TestClient(_ready_app()) as client:
        r = client.get("/does-not-exist")
    assert r.status_code == 404
    assert len(_events_for(sink, "request_start")) == 1
    ends = _events_for(sink, "request_end")
    assert len(ends) == 1
    assert ends[0]["status_code"] == 404


def test_row3_validation_failure_422(sink):
    with TestClient(_ready_app()) as client:
        r = client.post("/rag", json={"not_question": "x"})
    assert r.status_code == 422
    ends = _events_for(sink, "request_end")
    assert len(ends) == 1
    assert ends[0]["status_code"] == 422


def test_row4_engine_not_ready_503(sink):
    def _raise_settings():
        raise SettingsError("boom", exit_code=2)

    app = create_app(settings_loader=_raise_settings)
    with TestClient(app) as client:
        r = client.post("/rag", json={"question": "hello"})
    assert r.status_code == 503
    ends = _events_for(sink, "request_end")
    assert len(ends) == 1
    assert ends[0]["status_code"] == 503


def test_row5_handler_internal_exception_500(sink, monkeypatch):
    def _boom(question, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr("simple_qna_rag.agent.route_query", _boom)
    with TestClient(_ready_app(), raise_server_exceptions=False) as client:
        r = client.post("/rag", json={"question": "hello"})
    assert r.status_code == 500
    ends = _events_for(sink, "request_end")
    assert len(ends) == 1
    assert ends[0]["status_code"] == 500
    assert ends[0]["error_code"] == "internal"


def test_no_forbidden_payload_across_any_captured_event(sink, monkeypatch):
    monkeypatch.setattr(
        "simple_qna_rag.agent.route_query",
        lambda question, **kwargs: {"answer": "sensitive answer text", "sources": [], "success": True},
    )
    with TestClient(_ready_app()) as client:
        client.post("/rag", json={"question": "what is the secret plan?"})
        client.get("/does-not-exist")
        client.post("/rag", json={"bad": True})

    for record in sink.records:
        for forbidden in _FORBIDDEN_KEYS:
            assert forbidden not in record
        serialized = json.dumps(record, ensure_ascii=False)
        assert "secret plan" not in serialized
        assert "sensitive answer" not in serialized
