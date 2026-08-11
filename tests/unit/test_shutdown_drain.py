import asyncio
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest

from simple_qna_rag.settings import get_settings
from simple_qna_rag.web.concurrency import AdmissionRejected, QueryExecutor
from simple_qna_rag.web.server import (
    _ACTIVE_LIFESPAN_GUARD, _dispatch_lifecycle_cleanup, _teardown_lifecycle_owner,
    create_app,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_running_queued_and_grace_expiry_profiles(m42_receipt):
    release = threading.Event()
    executor = QueryExecutor(concurrency_limit=1, queue_limit=2, queue_timeout=2,
                             execution_timeout=2)
    before = executor.snapshot()
    running = executor.submit(lambda: release.wait())
    queued = [executor.submit(lambda: None), executor.submit(lambda: None)]
    executor.begin_drain()
    for handle in queued:
        with pytest.raises(AdmissionRejected):
            await handle.result()
    assert not await executor.wait_drained(0)
    with pytest.raises(AdmissionRejected):
        executor.submit(lambda: None)
    release.set()
    await running.result()
    assert await executor.wait_drained(1)
    executor.shutdown()
    assert executor.snapshot().lifecycle == "STOPPED"
    executor.snapshot()


class _TraceState:
    def __init__(self, trace):
        object.__setattr__(self, "trace", trace)

    def __setattr__(self, name, value):
        if name != "trace":
            self.trace.append((name, value))
        object.__setattr__(self, name, value)


class _LifecycleExecutor:
    def __init__(self, trace, failures=(), drained=True):
        self.trace, self.failures, self.drained = trace, set(failures), drained
        self.calls = {name: 0 for name in ("begin", "wait", "shutdown", "snapshot")}

    def _call(self, name):
        self.calls[name] += 1
        self.trace.append((name, self.calls[name]))
        if name in self.failures:
            exc = asyncio.CancelledError(name) if name == "wait_cancel" else RuntimeError(name)
            raise exc

    def begin_drain(self):
        self._call("begin")

    async def wait_drained(self, grace):
        self.trace.append(("grace", grace))
        if "wait_cancel" in self.failures:
            self.calls["wait"] += 1
            raise asyncio.CancelledError("wait_cancel")
        self._call("wait")
        return self.drained

    def shutdown(self, **kwargs):
        assert kwargs == {"wait": False, "cancel_futures": True}
        self._call("shutdown")

    def snapshot(self):
        self._call("snapshot")
        return SimpleNamespace(lifecycle="STOPPED")


@pytest.mark.anyio
@pytest.mark.parametrize("failures,expected", [
    ((), (1, 1, 1, 1)),
    (("begin",), (1, 0, 1, 1)),
    (("wait",), (1, 1, 1, 1)),
    (("shutdown",), (1, 1, 1, 1)),
    (("snapshot",), (1, 1, 1, 1)),
    (("wait", "shutdown", "snapshot"), (1, 1, 1, 1)),
    (("wait_cancel", "shutdown"), (1, 1, 1, 1)),
])
async def test_lifecycle_teardown_complete_failure_table(failures, expected):
    trace = []
    state = _TraceState(trace)
    app = SimpleNamespace(state=state)
    owner = _ACTIVE_LIFESPAN_GUARD.acquire()
    executor = _LifecycleExecutor(trace, failures, drained=False)
    errors = await _teardown_lifecycle_owner(executor, app, owner, 0.125)
    assert tuple(executor.calls.values()) == expected
    assert [str(error) for error in errors] == [name for name in ("begin", "wait", "wait_cancel", "shutdown", "snapshot") if name in failures]
    assert trace[-1:] == [("lifecycle", "STOPPED")]
    # release is deliberately invisible to fallible observers; successful immediate
    # reacquire proves it was the exact canonical final owner action.
    reacquired = _ACTIVE_LIFESPAN_GUARD.acquire()
    assert _ACTIVE_LIFESPAN_GUARD.release_exact_owner(reacquired) == "released"


@pytest.mark.anyio
async def test_lifecycle_teardown_executor_absent_and_observer_failure_tail():
    trace = []
    app = SimpleNamespace(state=_TraceState(trace))
    owner = _ACTIVE_LIFESPAN_GUARD.acquire()
    assert await _teardown_lifecycle_owner(None, app, owner, 0.0) == []
    assert trace == [("lifecycle", "STOPPED")]
    reacquired = _ACTIVE_LIFESPAN_GUARD.acquire()
    assert _ACTIVE_LIFESPAN_GUARD.release_exact_owner(reacquired) == "released"


@pytest.mark.anyio
async def test_cleanup_dispatcher_task_creation_failure_uses_inline_tail(monkeypatch):
    trace = []
    app = SimpleNamespace(state=_TraceState(trace))
    owner = _ACTIVE_LIFESPAN_GUARD.acquire()
    executor = _LifecycleExecutor(trace)
    monkeypatch.setattr(asyncio, "create_task",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("create")))
    errors, cancellation = await _dispatch_lifecycle_cleanup(executor, app, owner, 0.0)
    assert [str(error) for error in errors] == ["create"] and cancellation is None
    assert tuple(executor.calls.values()) == (1, 1, 1, 1)
    assert trace[-1] == ("lifecycle", "STOPPED")
    reacquired = _ACTIVE_LIFESPAN_GUARD.acquire()
    assert _ACTIVE_LIFESPAN_GUARD.release_exact_owner(reacquired) == "released"


@pytest.mark.anyio
async def test_actual_lifespan_task_creation_failure_propagates_after_mandatory_tail(monkeypatch):
    settings = get_settings()
    app = create_app(settings_loader=lambda: settings, engine_factory=lambda _: object())
    with pytest.raises(RuntimeError, match="create-task-primary"):
        async with app.router.lifespan_context(app):
            executor = app.state.query_executor
            monkeypatch.setattr(
                asyncio, "create_task",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("create-task-primary")),
            )
    assert executor.snapshot().lifecycle == "STOPPED"
    reacquired = _ACTIVE_LIFESPAN_GUARD.acquire()
    assert _ACTIVE_LIFESPAN_GUARD.release_exact_owner(reacquired) == "released"


@pytest.mark.anyio
async def test_actual_lifespan_body_primary_identity_wins_and_creation_is_secondary(monkeypatch):
    settings = get_settings()
    app = create_app(settings_loader=lambda: settings, engine_factory=lambda _: object())
    primary = LookupError("body-primary")
    logs = []
    monkeypatch.setattr("simple_qna_rag.web.server.log_event",
                        lambda event, **fields: logs.append((event, fields)))
    with pytest.raises(LookupError) as raised:
        async with app.router.lifespan_context(app):
            monkeypatch.setattr(
                asyncio, "create_task",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("create-task-secondary")),
            )
            raise primary
    assert raised.value is primary
    assert [(event, fields["error_type"]) for event, fields in logs
            if event == "cleanup_secondary"] == [("cleanup_secondary", "RuntimeError")]
    assert app.state.lifecycle == "STOPPED"
    reacquired = _ACTIVE_LIFESPAN_GUARD.acquire()
    assert _ACTIVE_LIFESPAN_GUARD.release_exact_owner(reacquired) == "released"


def test_fresh_process_actual_lifespan_creation_failure_primary_and_body_primary():
    program = textwrap.dedent("""
        import asyncio
        from simple_qna_rag.settings import get_settings
        from simple_qna_rag.web.server import create_app

        async def probe(body_primary):
            settings = get_settings()
            app = create_app(settings_loader=lambda: settings, engine_factory=lambda _: object())
            original = asyncio.create_task
            primary = LookupError("body-primary")
            try:
                try:
                    async with app.router.lifespan_context(app):
                        asyncio.create_task = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("create"))
                        if body_primary:
                            raise primary
                except BaseException as exc:
                    assert exc is primary if body_primary else isinstance(exc, RuntimeError)
                    assert app.state.lifecycle == "STOPPED"
            finally:
                asyncio.create_task = original

        asyncio.run(probe(False))
        asyncio.run(probe(True))
        print("fresh-lifespan-pass")
    """)
    result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith("fresh-lifespan-pass")


@pytest.mark.anyio
async def test_cleanup_dispatcher_drains_double_cancellation_and_releases_owner():
    trace = []
    app = SimpleNamespace(state=_TraceState(trace))
    owner = _ACTIVE_LIFESPAN_GUARD.acquire()
    executor = _LifecycleExecutor(trace)
    original_wait = executor.wait_drained
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_wait(grace):
        entered.set()
        await release.wait()
        return await original_wait(grace)

    executor.wait_drained = blocked_wait
    task = asyncio.create_task(_dispatch_lifecycle_cleanup(executor, app, owner, 0.1))
    await entered.wait()
    task.cancel("first")
    await asyncio.sleep(0)
    task.cancel("second")
    await asyncio.sleep(0)
    release.set()
    errors, cancellation = await task
    assert errors == []
    assert isinstance(cancellation, asyncio.CancelledError) and cancellation.args == ("first",)
    assert tuple(executor.calls.values()) == (1, 1, 1, 1)
    reacquired = _ACTIVE_LIFESPAN_GUARD.acquire()
    assert _ACTIVE_LIFESPAN_GUARD.release_exact_owner(reacquired) == "released"
