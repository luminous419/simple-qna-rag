import asyncio

import pytest
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from simple_qna_rag.settings import get_settings
from simple_qna_rag.web.server import create_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _scope(app, headers):
    return {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/rag", "raw_path": b"/rag",
        "query_string": b"", "root_path": "", "headers": headers,
        "client": ("127.0.0.1", 1), "server": ("test", 80), "state": {}, "app": app,
    }


@pytest.mark.anyio
@pytest.mark.parametrize("headers,status,reason", [
    ([(b"content-type", b"application/json"), (b"content-encoding", b"gzip")], 400, "invalid_request"),
    ([(b"content-type", b"application/json"), (b"content-length", b"wat")], 400, "invalid_request"),
    ([(b"content-type", b"application/json"), (b"content-length", b"-1")], 413, "payload_too_large"),
    ([(b"content-type", b"application/json"), (b"content-length", b"999999999")], 413, "payload_too_large"),
])
async def test_pre_body_rejections_are_observed_once_and_body_is_unread(monkeypatch, headers, status, reason):
    logs = []
    monkeypatch.setattr("simple_qna_rag.observability.logging.log_event",
                        lambda event, **fields: logs.append((event, fields)))
    app = create_app(settings_loader=get_settings, engine_factory=lambda _: object())
    receive_calls = 0
    frames = []

    async def send(message):
        frames.append(message)

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        raise AssertionError("forbidden body was read")

    async with app.router.lifespan_context(app):
        await app(_scope(app, headers), receive, send)
    assert receive_calls == 0
    assert frames[0]["status"] == status
    assert [event for event, _ in logs].count("request_start") == 1
    assert [event for event, _ in logs].count("request_end") == 1
    metrics = generate_latest(app.state.metrics_registry).decode()
    assert 'rag_requests_total{route="rag",status="4xx"} 1.0' in metrics
    assert 'rag_request_duration_seconds_count{route="rag"} 1.0' in metrics
    assert f'rag_input_rejected_total{{reason="{reason}"}} 1.0' in metrics


@pytest.mark.anyio
async def test_identity_chunk_overflow_and_disconnect_each_observed_once(monkeypatch):
    logs = []
    monkeypatch.setattr("simple_qna_rag.observability.logging.log_event",
                        lambda event, **fields: logs.append((event, fields)))
    app = create_app(settings_loader=get_settings, engine_factory=lambda _: object())
    limit = get_settings().MAX_REQUEST_BODY_BYTES
    messages = iter((
        {"type": "http.request", "body": b"x" * limit, "more_body": True},
        {"type": "http.request", "body": b"yz", "more_body": True},
        {"type": "http.disconnect"},
    ))
    calls = 0
    frames = []

    async def send(message):
        frames.append(message)

    async def receive():
        nonlocal calls
        calls += 1
        return next(messages)

    async with app.router.lifespan_context(app):
        scope = _scope(app, [(b"content-type", b"application/json"),
                             (b"content-encoding", b"identity")])
        await app(scope, receive, send)
    assert calls == 2 and frames[0]["status"] == 413
    assert scope["state"]["body_limit_observation"]["application_consumed_bytes"] == limit + 1
    assert [event for event, _ in logs].count("request_start") == 1
    assert [event for event, _ in logs].count("request_end") == 1
    metrics = generate_latest(app.state.metrics_registry).decode()
    assert 'rag_input_rejected_total{reason="payload_too_large"} 1.0' in metrics


def test_body_profiles_stop_at_limit_plus_one(m42_receipt):
    settings = get_settings()
    app = create_app(settings_loader=get_settings, engine_factory=lambda _: object())
    with TestClient(app) as client:
        before = app.state.query_executor.snapshot()
        response = client.post(
            "/rag", content=b"x" * (settings.MAX_REQUEST_BODY_BYTES + 1),
            headers={"content-type": "application/json"},
        )
        after = app.state.query_executor.snapshot()
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
