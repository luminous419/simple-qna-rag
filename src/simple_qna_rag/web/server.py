#!/usr/bin/env python3
"""FastAPI 기반 웹 서버 — M4.1 bootstrap/lifespan/health redesign (Design.md §3).

`create_app()`은 `Bootstrap`(§3.2, config.py/settings.py 미import)만으로
health route를 최우선 등록한다 — Settings 로딩이나 RAG 엔진 초기화가
실패해도 이 모듈의 import와 `create_app()` 호출 자체는 항상 성공한다.
Settings/엔진 로딩은 lifespan(§3.3) 안에서만 시도되고, 실패는 예외로
전파되지 않으며 `/health/ready` 503으로 표현된다.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import json
import threading
import unicodedata
from typing import Any, Callable, Optional, Literal

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
import uvicorn
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from simple_qna_rag.observability.health import SaturationDebounce, evaluate_readiness
from simple_qna_rag.observability.logging import SERVICE_VERSION, log_event
from simple_qna_rag.observability.metrics import (
    build_metrics_registry, clamp_readiness_reason, record_admission_rejected,
    record_input_rejected, record_ticket_outcome, sync_executor_gauges,
)
from simple_qna_rag.observability.request_context import RequestContextMiddleware
from simple_qna_rag.settings import Settings, SettingsError, get_settings
from simple_qna_rag.web.bootstrap import Bootstrap, load_bootstrap
from simple_qna_rag.web.body_limit import BOOTSTRAP_BODY_CEILING, BodyLimitMiddleware
from simple_qna_rag.web.concurrency import (
    AdmissionRejected, ExecutionTimeoutError, QueryExecutor, QueueTimeoutError, SubmitFailed,
)
from simple_qna_rag.web.errors import error_response

_HEALTH_DEPRECATION_SUNSET = "Fri, 06 Nov 2026 00:00:00 GMT"


class _SingleActiveLifespanGuard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: object | None = None

    def acquire(self) -> object:
        with self._lock:
            if self._owner is not None:
                raise RuntimeError("lifespan_already_active")
            self._owner = object()
            return self._owner

    def release_exact_owner(self, owner: object) -> str:
        with self._lock:
            if self._owner is owner:
                self._owner = None
                return "released"
            return "owner_mismatch"


_ACTIVE_LIFESPAN_GUARD = _SingleActiveLifespanGuard()
_PROCESS_SETTINGS_LOCK = threading.Lock()
_PROCESS_SETTINGS_IDENTITY: Any = None


def _observe(function: Callable[..., Any], *args: Any) -> None:
    """Observers are bounded side effects and never own request/lifecycle state."""
    try:
        function(*args)
    except Exception:
        pass


def _commit_process_settings(candidate: Any) -> None:
    global _PROCESS_SETTINGS_IDENTITY
    with _PROCESS_SETTINGS_LOCK:
        if _PROCESS_SETTINGS_IDENTITY is None:
            _PROCESS_SETTINGS_IDENTITY = candidate
        elif _PROCESS_SETTINGS_IDENTITY is not candidate:
            raise RuntimeError("process_settings_identity_mismatch")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list
    success: bool
    search_type: str = "unknown"
    intent: Optional[str] = None


class _RaceArbiter:
    """Single-CAS result/disconnect arbiter; task scheduling order breaks exact ties."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sequence = 0
        self.winner: Literal["result", "disconnect"] | None = None
        self.winner_sequence: int | None = None
        self.future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def claim(self, kind: Literal["result", "disconnect"]) -> None:
        async with self._lock:
            self._sequence += 1
            if self.winner is None:
                self.winner = kind
                self.winner_sequence = self._sequence
                self.future.set_result(kind)


async def _cancel_and_reap(task: asyncio.Task[Any] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except BaseException:
        pass


class RagServingMiddleware:
    """Actual-ASGI owner for `/rag` body-to-disconnect receive handoff."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") != "/rag" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        registry = getattr(scope["app"].state, "metrics_registry", None)

        async def respond(response: Response) -> None:
            await response(scope, _receive_never, send)

        def terminal(name: str) -> None:
            scope.setdefault("state", {})["rag_terminal"] = name

        content_type = Headers(scope=scope).get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            terminal("invalid_request")
            _observe(record_input_rejected, registry, "invalid_request")
            await respond(error_response("invalid_request"))
            return
        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                scope.setdefault("state", {})["rag_terminal"] = "client_disconnected"
                return
            if message["type"] != "http.request":
                raise RuntimeError("unexpected_request_message")
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        raw = b"".join(chunks)
        overflow = scope.get("state", {}).get("body_limit_overflow")
        if callable(overflow) and overflow():
            terminal("payload_too_large")
            _observe(record_input_rejected, registry, "payload_too_large")
            await respond(error_response("payload_too_large"))
            return
        try:
            body = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            terminal("invalid_request")
            _observe(record_input_rejected, registry, "invalid_request")
            await respond(error_response("invalid_request"))
            return
        settings = getattr(scope["app"].state, "settings", None)
        question = _validate_question(body, getattr(settings, "MAX_QUESTION_CHARS", 4000))
        if question is None:
            terminal("invalid_request")
            _observe(record_input_rejected, registry, "invalid_request")
            await respond(error_response("invalid_request"))
            return
        executor = getattr(scope["app"].state, "query_executor", None)
        if getattr(scope["app"].state, "engine", None) is None or executor is None:
            terminal("not_ready")
            await respond(error_response("not_ready"))
            return

        from simple_qna_rag.agent import route_query

        try:
            handle = executor.submit(lambda: route_query(question, metrics_registry=registry))
        except AdmissionRejected as exc:
            terminal("not_ready" if exc.reason == "not_ready" else "overloaded")
            _observe(record_admission_rejected, registry, exc.reason)
            await respond(error_response("not_ready" if exc.reason == "not_ready" else "overloaded"))
            return
        except SubmitFailed:
            terminal("internal")
            _observe(record_admission_rejected, registry, "submit_failed")
            await respond(error_response("internal"))
            return

        arbiter = _RaceArbiter()

        async def controlled_claim(kind: Literal["result", "disconnect"]) -> None:
            """Allow deterministic acceptance to order callbacks without replacing the app stack."""
            controller = getattr(scope["app"].state, "rag_race_controller", None)
            if controller is not None:
                await controller(kind)
            await arbiter.claim(kind)

        async def result_owner() -> Any:
            try:
                return await handle.result()
            finally:
                await controlled_claim("result")

        async def disconnect_owner() -> None:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    await controlled_claim("disconnect")
                    return
                if message["type"] == "http.request":
                    raise RuntimeError("http_request_after_body")

        result_task = asyncio.create_task(result_owner(), name="rag-result-owner")
        disconnect_task = asyncio.create_task(disconnect_owner(), name="rag-disconnect-owner")
        try:
            winner = await arbiter.future
            scope.setdefault("state", {})["rag_arbiter"] = {
                "winner": winner, "sequence": arbiter.winner_sequence
            }
            if winner == "disconnect":
                scope["state"]["rag_terminal"] = "client_disconnected"
                handle.cancel()
                _observe(record_ticket_outcome, registry, "cancelled")
                await _cancel_and_reap(result_task)
                await _cancel_and_reap(disconnect_task)
                return
            await _cancel_and_reap(disconnect_task)
            try:
                result = await result_task
            except QueueTimeoutError:
                terminal("queue_timeout")
                _observe(record_ticket_outcome, registry, "queue_timeout")
                await respond(error_response("queue_timeout"))
            except ExecutionTimeoutError:
                terminal("execution_timeout")
                _observe(record_ticket_outcome, registry, "execution_timeout")
                await respond(error_response("execution_timeout"))
            except asyncio.CancelledError:
                raise
            except Exception:
                terminal("internal")
                _observe(record_ticket_outcome, registry, "rejected")
                await respond(error_response("internal"))
            else:
                terminal("success")
                _observe(record_ticket_outcome, registry, "completed")
                model = QueryResponse(**result)
                await respond(JSONResponse(model.model_dump()))
        except asyncio.CancelledError:
            handle.cancel()
            await _cancel_and_reap(result_task)
            await _cancel_and_reap(disconnect_task)
            raise
        finally:
            await _cancel_and_reap(result_task)
            await _cancel_and_reap(disconnect_task)


async def _receive_never() -> Message:
    raise RuntimeError("response attempted to receive")


def _validate_question(body: Any, max_chars: int) -> str | None:
    if not isinstance(body, dict) or set(body) != {"question"}:
        return None
    question = body["question"]
    if not isinstance(question, str):
        return None
    question = question.strip()
    if not question or len(question) > max_chars or "\x00" in question:
        return None
    if any(unicodedata.category(char) in {"Cc", "Cs"} for char in question):
        return None
    return question


def _mount_static_and_templates(app: FastAPI, bootstrap: Bootstrap) -> str | None:
    try:
        if not bootstrap.static_dir.is_dir():
            raise NotADirectoryError
        app.mount("/static", StaticFiles(directory=bootstrap.static_dir), name="static")
        if not bootstrap.templates_dir.is_dir():
            raise NotADirectoryError
        app.state.templates = Jinja2Templates(directory=bootstrap.templates_dir)
    except (FileNotFoundError, NotADirectoryError):
        return "static_mount_failed"
    return None


def _default_engine_factory(settings: Any) -> Any:
    # Lazy import — importing rag_engine.py transitively imports config.py,
    # which materializes Settings eagerly. Deferring this import until after
    # `settings_loader()` has already succeeded (see `_make_lifespan` below)
    # guarantees config.py's own `get_settings()` call hits the already-valid
    # process cache instead of re-validating and potentially raising.
    from simple_qna_rag.rag_engine import RAGEngine

    return RAGEngine.from_settings(settings)


def _make_lifespan(
    settings_loader: Callable[[], Any], engine_factory: Callable[[Any], Any]
):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owner = _ACTIVE_LIFESPAN_GUARD.acquire()
        executor = None
        grace = 0.0
        attempt_class = "prepublication"
        primary: BaseException | None = None
        try:
            try:
                candidate = settings_loader()
            except SettingsError as exc:
                attempt_class = "invalid_loader"
                # One atomic fail-soft publication; generic STOPPED publication is forbidden.
                app.state.settings = None
                app.state.settings_error = str(exc)
                app.state.engine, app.state.engine_error = None, None
                yield
            else:
                # Candidate remains local until immutable process identity is verified.
                _commit_process_settings(candidate)
                attempt_class = "lifecycle_owner"
                app.state.settings = candidate
                app.state.settings_error = None
                grace = candidate.SHUTDOWN_GRACE_SECONDS
                app.state.engine_artifact_reason = None
                # Lazy import mirrors _default_engine_factory above — safe
                # here because settings_loader() has already succeeded.
                from simple_qna_rag.rag_engine import EngineArtifactError

                try:
                    app.state.engine = engine_factory(candidate)
                    app.state.engine_error = None
                    executor = QueryExecutor(
                        concurrency_limit=candidate.QUERY_CONCURRENCY_LIMIT,
                        queue_limit=candidate.QUERY_QUEUE_LIMIT,
                        queue_timeout=candidate.QUERY_QUEUE_TIMEOUT_SECONDS,
                        execution_timeout=candidate.QUERY_EXECUTION_TIMEOUT_SECONDS,
                    )
                    app.state.query_executor = executor
                    app.state.lifecycle = "READY"
                except EngineArtifactError as exc:  # fail-soft, disclosed reason
                    # engine_factory never returns the failed instance on
                    # failure (get_rag_engine() discards it), so the only
                    # way to recover a disclosed artifact reason is off
                    # this dedicated exception type — see
                    # rag_engine.EngineArtifactError. CR-I5-MAJ-02: this
                    # branch must classify *only* this specific type, never
                    # an arbitrary exception that merely happens to define
                    # a `.reason` attribute (that used to be reclassified
                    # as an artifact failure via a bare `getattr`).
                    app.state.engine = None
                    app.state.engine_error = str(exc)
                    app.state.engine_artifact_reason = exc.reason
                except Exception as exc:  # fail-soft engine diagnostic
                    app.state.engine = None
                    app.state.engine_error = str(exc)
                _, reason = evaluate_readiness(
                    getattr(app.state, "bootstrap_error", None),
                    app.state.settings_error,
                    app.state.engine_error,
                    artifact_error_reason=app.state.engine_artifact_reason,
                )
                registry = app.state.metrics_registry
                registry.rag_readiness.labels(reason=clamp_readiness_reason(reason)).set(1)
                log_event("startup", reason=reason, metrics_registry=registry)
                yield
        except BaseException as exc:
            primary = exc
        finally:
            cleanup_errors: list[BaseException]
            if attempt_class == "lifecycle_owner":
                cleanup_errors, cleanup_cancel = await _dispatch_lifecycle_cleanup(
                    executor, app, owner, grace
                )
                if primary is None and cleanup_cancel is not None:
                    primary = cleanup_cancel
            else:
                cleanup_errors = []
                _ACTIVE_LIFESPAN_GUARD.release_exact_owner(owner)
            if primary is not None:
                for index, error in enumerate(cleanup_errors):
                    try:
                        log_event(
                            "cleanup_secondary", index=index,
                            error_type=type(error).__name__,
                            metrics_registry=app.state.metrics_registry,
                        )
                    except Exception:
                        pass
                raise primary
            if len(cleanup_errors) == 1:
                raise cleanup_errors[0]
            if cleanup_errors:
                raise ExceptionGroup("lifespan_cleanup_failed", cleanup_errors)

    return lifespan


async def _dispatch_lifecycle_cleanup(
    executor: QueryExecutor | None, app: FastAPI, owner: object, grace: float
) -> tuple[list[BaseException], asyncio.CancelledError | None]:
    """Run lifecycle cleanup to completion despite construction or cancellation failures."""
    cleanup_task: asyncio.Task[list[BaseException]] | None = None
    cleanup_coro = None
    first_cancel: asyncio.CancelledError | None = None
    try:
        cleanup_coro = _teardown_lifecycle_owner(executor, app, owner, grace)
        cleanup_task = asyncio.create_task(cleanup_coro, name="rag-lifespan-teardown")
    except BaseException as creation_error:
        if cleanup_coro is not None:
            cleanup_coro.close()
        # Task construction is an optimization, never the owner of the tail.
        try:
            inline_errors = await _teardown_lifecycle_owner(executor, app, owner, grace)
            return [creation_error, *inline_errors], None
        except BaseException as inline_error:
            return [creation_error, inline_error], None

    while True:
        try:
            return await asyncio.shield(cleanup_task), first_cancel
        except asyncio.CancelledError as exc:
            if first_cancel is None:
                first_cancel = exc
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            # Repeated cancellation is drained; the same cleanup task remains
            # authoritative and is never restarted.
            continue
        except BaseException as cleanup_error:
            return [cleanup_error], first_cancel


async def _teardown_lifecycle_owner(
    executor: QueryExecutor | None, app: FastAPI, owner: object, grace: float
) -> list[BaseException]:
    """Finish every fallible operation before the nonthrowing STOPPED/release tail."""
    errors: list[BaseException] = []
    if executor is not None:
        began = False
        try:
            app.state.lifecycle = "DRAINING"
            executor.begin_drain()
            began = True
        except BaseException as exc:
            errors.append(exc)
        if began:
            try:
                await executor.wait_drained(grace)
            except BaseException as exc:
                errors.append(exc)
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except BaseException as exc:
            errors.append(exc)
        try:
            executor.snapshot()
        except BaseException as exc:
            errors.append(exc)
    # Canonical durable tail: release cannot be skipped even if a hostile state
    # observer makes publication raise.
    try:
        app.state.lifecycle = "STOPPED"
    except BaseException as exc:
        errors.append(exc)
    finally:
        _ACTIVE_LIFESPAN_GUARD.release_exact_owner(owner)
    return errors


def _register_health_routes(app: FastAPI) -> None:
    @app.get("/health/live")
    async def health_live() -> JSONResponse:
        return JSONResponse(status_code=200, content={"status": "ok"})

    @app.get("/health/ready")
    async def health_ready(request: Request) -> JSONResponse:
        executor = getattr(request.app.state, "query_executor", None)
        snap = executor.snapshot() if executor else None
        saturated = request.app.state.saturation_debounce.evaluate(
            full=snap.capacity_full, edge_at=snap.capacity_edge_at, version=snap.capacity_version
        ) if snap else False
        status_code, reason = evaluate_readiness(
            getattr(request.app.state, "bootstrap_error", None),
            getattr(request.app.state, "settings_error", None),
            getattr(request.app.state, "engine_error", None),
            lifecycle=snap.lifecycle if snap else getattr(request.app.state, "lifecycle", None),
            saturated=saturated,
            orphaned=snap.orphaned if snap else 0,
            concurrency_limit=snap.concurrency_limit if snap else 0,
            artifact_error_reason=getattr(request.app.state, "engine_artifact_reason", None),
        )
        return JSONResponse(
            status_code=status_code,
            content={"status": "ok" if status_code == 200 else "not_ready", "reason": reason},
        )

    @app.get("/health")
    async def health_deprecated(request: Request) -> JSONResponse:
        """1-release deprecated alias(REQ-005.3) — body shape/semantics는
        M4.1 이전 버전과 동일하게 보존한다: 엔진이 없어도 status는 항상
        "healthy"였다."""
        engine = getattr(request.app.state, "engine", None)
        response = JSONResponse(
            content={"status": "healthy", "rag_engine_initialized": engine is not None}
        )
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = _HEALTH_DEPRECATION_SUNSET
        return response

    @app.get("/metrics")
    async def metrics_endpoint(request: Request) -> Response:
        registry = request.app.state.metrics_registry
        executor = getattr(request.app.state, "query_executor", None)
        sync_executor_gauges(registry, executor.snapshot() if executor else None)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


def _register_api_routes(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        templates = getattr(request.app.state, "templates", None)
        if templates is None:
            return HTMLResponse("static assets unavailable", status_code=503)
        return templates.TemplateResponse(request, "index.html", {"request": request})

    @app.post("/rag", response_model=QueryResponse)
    async def rag_query(request: Request):
        registry = getattr(request.app.state, "metrics_registry", None)
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            record_input_rejected(registry, "invalid_request")
            return error_response("invalid_request")
        try:
            raw = await request.body()
        except Exception:
            record_input_rejected(registry, "payload_too_large")
            return error_response("payload_too_large")
        settings = getattr(request.app.state, "settings", None)
        if settings is not None and len(raw) > settings.MAX_REQUEST_BODY_BYTES:
            record_input_rejected(registry, "payload_too_large")
            return error_response("payload_too_large")
        overflow = request.scope.get("state", {}).get("body_limit_overflow")
        if callable(overflow) and overflow():
            record_input_rejected(registry, "payload_too_large")
            return error_response("payload_too_large")
        try:
            body = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            record_input_rejected(registry, "invalid_request")
            return error_response("invalid_request")
        question = _validate_question(body, getattr(settings, "MAX_QUESTION_CHARS", 4000))
        if question is None:
            record_input_rejected(registry, "invalid_request")
            return error_response("invalid_request")
        if getattr(request.app.state, "engine", None) is None:
            return error_response("not_ready")

        from simple_qna_rag.agent import route_query  # lazy — engine already proved importable

        executor = getattr(request.app.state, "query_executor", None)
        if executor is None:
            return error_response("not_ready")
        try:
            handle = executor.submit(lambda: route_query(question, metrics_registry=registry))
            result = await handle.result()
            record_ticket_outcome(registry, "completed")
            return QueryResponse(**result)
        except AdmissionRejected as exc:
            record_admission_rejected(registry, exc.reason)
            return error_response("not_ready" if exc.reason == "not_ready" else "overloaded")
        except QueueTimeoutError:
            record_ticket_outcome(registry, "queue_timeout")
            return error_response("queue_timeout")
        except ExecutionTimeoutError:
            record_ticket_outcome(registry, "execution_timeout")
            return error_response("execution_timeout")
        except SubmitFailed:
            record_admission_rejected(registry, "submit_failed")
            return error_response("internal")
        except asyncio.CancelledError:
            record_ticket_outcome(registry, "cancelled")
            request.scope.setdefault("state", {})["rag_terminal"] = "client_disconnected"
            raise
        except Exception:
            record_ticket_outcome(registry, "rejected")
            return error_response("internal")


def create_app(
    bootstrap: Bootstrap | None = None,
    settings_loader: Callable[[], Any] | None = None,
    engine_factory: Callable[[Any], Any] | None = None,
) -> FastAPI:
    bootstrap = bootstrap if bootstrap is not None else load_bootstrap()
    settings_loader = settings_loader if settings_loader is not None else get_settings
    engine_factory = engine_factory if engine_factory is not None else _default_engine_factory

    app = FastAPI(
        title="Simple Q&A RAG System",
        description="문서 질의응답 시스템 웹 인터페이스",
        version=SERVICE_VERSION,
        lifespan=_make_lifespan(settings_loader, engine_factory),
    )
    app.add_middleware(RagServingMiddleware)
    app.add_middleware(BodyLimitMiddleware, bootstrap_max_bytes=BOOTSTRAP_BODY_CEILING)
    # Starlette wraps in reverse registration order: request ownership must be
    # outermost, while the limiter remains outside the only body consumer.
    app.add_middleware(RequestContextMiddleware)
    app.state.metrics_registry = build_metrics_registry()
    app.state.saturation_debounce = SaturationDebounce()
    app.state.query_executor = None
    app.state.lifecycle = "STARTING"
    _register_health_routes(app)
    app.state.bootstrap_error = _mount_static_and_templates(app, bootstrap)
    _register_api_routes(app)
    return app


# Module-level `app` — importable regardless of Settings validity (M1-01),
# and required by the CI smoke test `python -c "from ...web.server import app"`.
app = create_app()


def start_server(
    host: str = "0.0.0.0", port: int = 8000, cli_overrides: dict[str, str] | None = None
) -> None:
    cli_overrides = cli_overrides or {}
    server_app = create_app(
        settings_loader=lambda: Settings.from_sources(cli_overrides=cli_overrides)
    )
    uvicorn.run(server_app, host=host, port=port)
