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
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
import uvicorn

from simple_qna_rag.observability.health import evaluate_readiness
from simple_qna_rag.observability.logging import SERVICE_VERSION, log_event
from simple_qna_rag.observability.metrics import build_metrics_registry, clamp_readiness_reason
from simple_qna_rag.observability.request_context import RequestContextMiddleware
from simple_qna_rag.settings import Settings, SettingsError, get_settings
from simple_qna_rag.web.bootstrap import Bootstrap, load_bootstrap

_HEALTH_DEPRECATION_SUNSET = "Fri, 06 Nov 2026 00:00:00 GMT"


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list
    success: bool
    search_type: str = "unknown"
    intent: Optional[str] = None


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
        try:
            app.state.settings = settings_loader()
            app.state.settings_error = None
        except SettingsError as exc:
            app.state.settings = None
            app.state.settings_error = str(exc)

        if app.state.settings is not None:
            try:
                app.state.engine = engine_factory(app.state.settings)
                app.state.engine_error = None
            except Exception as exc:  # noqa: BLE001 - expressed via health, never re-raised
                app.state.engine = None
                app.state.engine_error = str(exc)
        else:
            app.state.engine, app.state.engine_error = None, None

        _, reason = evaluate_readiness(
            getattr(app.state, "bootstrap_error", None),
            app.state.settings_error,
            app.state.engine_error,
        )
        registry = app.state.metrics_registry
        registry.rag_readiness.labels(reason=clamp_readiness_reason(reason)).set(1)
        log_event("startup", reason=reason, metrics_registry=registry)
        yield

    return lifespan


def _register_health_routes(app: FastAPI) -> None:
    @app.get("/health/live")
    async def health_live() -> JSONResponse:
        return JSONResponse(status_code=200, content={"status": "ok"})

    @app.get("/health/ready")
    async def health_ready(request: Request) -> JSONResponse:
        status_code, reason = evaluate_readiness(
            getattr(request.app.state, "bootstrap_error", None),
            getattr(request.app.state, "settings_error", None),
            getattr(request.app.state, "engine_error", None),
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
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


def _register_api_routes(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        templates = getattr(request.app.state, "templates", None)
        if templates is None:
            return HTMLResponse("static assets unavailable", status_code=503)
        return templates.TemplateResponse(request, "index.html", {"request": request})

    @app.post("/rag", response_model=QueryResponse)
    async def rag_query(payload: QueryRequest, request: Request):
        if getattr(request.app.state, "engine", None) is None:
            return JSONResponse(
                status_code=503,
                content={
                    "answer": "RAG 엔진이 초기화되지 않았습니다.",
                    "sources": [],
                    "success": False,
                },
            )

        from simple_qna_rag.agent import route_query  # lazy — engine already proved importable

        registry = getattr(request.app.state, "metrics_registry", None)
        result = route_query(payload.question, metrics_registry=registry)
        return QueryResponse(**result)


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
    app.add_middleware(RequestContextMiddleware)
    app.state.metrics_registry = build_metrics_registry()
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
