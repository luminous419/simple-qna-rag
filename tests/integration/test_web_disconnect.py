import asyncio
import json
import threading

import pytest
from prometheus_client import generate_latest

from simple_qna_rag import agent
from simple_qna_rag.settings import get_settings
from simple_qna_rag.web.server import create_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _scope(app):
    body = json.dumps({"question": "bounded question"}).encode()
    return ({
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/rag", "raw_path": b"/rag",
        "query_string": b"", "root_path": "", "headers": [
            (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        "client": ("127.0.0.1", 1), "server": ("test", 80), "state": {}, "app": app,
    }, body)


class _OrderedClaims:
    def __init__(self):
        self.arrived = {kind: asyncio.Event() for kind in ("result", "disconnect")}
        self.permit = {kind: asyncio.Event() for kind in ("result", "disconnect")}

    async def __call__(self, kind):
        self.arrived[kind].set()
        await self.permit[kind].wait()


async def _exchange(app):
    scope, body = _scope(app)
    messages = iter((
        {"type": "http.request", "body": body, "more_body": False},
        {"type": "http.disconnect"},
    ))
    frames = []
    in_flight = max_in_flight = receive_calls = 0

    async def receive():
        nonlocal in_flight, max_in_flight, receive_calls
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        receive_calls += 1
        try:
            return next(messages)
        finally:
            in_flight -= 1

    async def send(message):
        frames.append(message)

    await app(scope, receive, send)
    return frames, max_in_flight, receive_calls, scope["state"]


def _terminal_total(snapshot):
    return sum((snapshot.queue_timeout_total, snapshot.execution_timeout_total,
                snapshot.cancelled_total, snapshot.completed_total,
                snapshot.terminal_rejected_total))


async def _prove_orders(app, executor, blocker_factory=None):
    orders = ("disconnect_first", "result_first", "tie_disconnect_inserted_first",
              "tie_result_inserted_first")
    winners = {}
    for index in range(100):
        order = orders[index % len(orders)]
        controller = _OrderedClaims()
        app.state.rag_race_controller = controller
        blocker = blocker_factory() if blocker_factory else None
        before = executor.snapshot()
        task = asyncio.create_task(_exchange(app))
        await asyncio.wait_for(controller.arrived["disconnect"].wait(), 2)
        if blocker is not None and order != "disconnect_first":
            blocker[0].set()
        if order != "disconnect_first":
            await asyncio.wait_for(controller.arrived["result"].wait(), 2)
        first = "disconnect" if order in ("disconnect_first", "tie_disconnect_inserted_first") else "result"
        controller.permit[first].set()
        await asyncio.sleep(0)
        controller.permit["result" if first == "disconnect" else "disconnect"].set()
        frames, receive_max, calls, state = await asyncio.wait_for(task, 2)
        if blocker is not None:
            blocker[0].set()
            await blocker[1].result()
        assert await executor.wait_drained(2)
        after = executor.snapshot()
        winner = state["rag_arbiter"]["winner"]
        winners[order] = winner
        assert winner == first and state["rag_arbiter"]["sequence"] == 1
        assert receive_max == 1 and calls == 2
        if winner == "disconnect":
            assert frames == [] and state["rag_terminal"] == "client_disconnected"
        else:
            assert sum(frame["type"] == "http.response.start" for frame in frames) == 1
            assert frames[-1]["type"] == "http.response.body" and not frames[-1].get("more_body", False)
        assert after.accepted_total - before.accepted_total == 1
        # The request owns exactly one terminal; the queued profile also completes its
        # deliberately pre-existing capacity blocker during the measurement window.
        assert _terminal_total(after) - _terminal_total(before) == (2 if blocker else 1)
        assert (after.queued, after.running, after.orphaned) == (0, 0, 0)
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()
                    and t.get_name() in {"rag-result-owner", "rag-disconnect-owner"} and not t.done()]
    del app.state.rag_race_controller
    assert winners == {
        "disconnect_first": "disconnect", "result_first": "result",
        "tie_disconnect_inserted_first": "disconnect", "tie_result_inserted_first": "result",
    }


@pytest.mark.anyio
async def test_asgi_disconnect_queued_100_races(monkeypatch, m42_receipt):
    monkeypatch.setattr(agent, "route_query", lambda *_args, **_kwargs: {"answer": "ok", "sources": [], "success": True})
    logs = []
    monkeypatch.setattr("simple_qna_rag.observability.logging.log_event",
                        lambda event, **fields: logs.append((event, fields)))
    app = create_app(settings_loader=get_settings, engine_factory=lambda _: object())
    async with app.router.lifespan_context(app):
        executor = app.state.query_executor
        before = executor.snapshot()

        def blocker_factory():
            release = threading.Event()
            return release, executor.submit(lambda: release.wait(2))

        await _prove_orders(app, executor, blocker_factory)
        after = executor.snapshot()
    assert [event for event, _ in logs].count("request_start") == 100
    assert [event for event, _ in logs].count("request_end") == 100
    metrics = generate_latest(app.state.metrics_registry).decode()
    assert 'rag_requests_total{route="rag",status="2xx"} 50.0' in metrics
    assert 'rag_requests_total{route="rag",status="4xx"} 50.0' in metrics


@pytest.mark.anyio
async def test_asgi_disconnect_running_100_races(monkeypatch, m42_receipt):
    monkeypatch.setattr(agent, "route_query", lambda *_args, **_kwargs: {"answer": "ok", "sources": [], "success": True})
    logs = []
    monkeypatch.setattr("simple_qna_rag.observability.logging.log_event",
                        lambda event, **fields: logs.append((event, fields)))
    app = create_app(settings_loader=get_settings, engine_factory=lambda _: object())
    async with app.router.lifespan_context(app):
        before = app.state.query_executor.snapshot()
        await _prove_orders(app, app.state.query_executor)
        after = app.state.query_executor.snapshot()
    assert [event for event, _ in logs].count("request_start") == 100
    assert [event for event, _ in logs].count("request_end") == 100
    metrics = generate_latest(app.state.metrics_registry).decode()
    assert 'rag_query_outcomes_total{result="completed"} 50.0' in metrics
    assert 'rag_query_outcomes_total{result="cancelled"} 50.0' in metrics


@pytest.mark.anyio
async def test_downstream_no_response_without_disconnect_is_internal():
    async def no_response(_scope, _receive, _send):
        return None

    from simple_qna_rag.observability.request_context import RequestContextMiddleware

    scope = {"type": "http", "method": "GET", "path": "/x", "headers": [], "state": {}}
    with pytest.raises(RuntimeError, match="downstream_no_response"):
        await RequestContextMiddleware(no_response)(scope, lambda: None, lambda _message: None)


@pytest.mark.anyio
async def test_failing_log_and_metric_observers_do_not_corrupt_request_state(monkeypatch):
    monkeypatch.setattr(agent, "route_query", lambda *_args, **_kwargs: {"answer": "ok", "sources": [], "success": True})
    monkeypatch.setattr("simple_qna_rag.observability.logging.log_event",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("log_spy")))
    monkeypatch.setattr("simple_qna_rag.web.server.record_ticket_outcome",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("metric_spy")))
    app = create_app(settings_loader=get_settings, engine_factory=lambda _: object())
    async with app.router.lifespan_context(app):
        controller = _OrderedClaims()
        app.state.rag_race_controller = controller
        task = asyncio.create_task(_exchange(app))
        await controller.arrived["disconnect"].wait()
        await controller.arrived["result"].wait()
        controller.permit["result"].set()
        controller.permit["disconnect"].set()
        frames, _, _, state = await task
        assert frames[0]["status"] == 200 and state["rag_arbiter"]["winner"] == "result"
        snap = app.state.query_executor.snapshot()
        assert (snap.queued, snap.running, snap.orphaned, snap.completed_total) == (0, 0, 0, 1)
