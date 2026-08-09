"""M4.1 §6.3 — request-scoped request ID plumbing."""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

REQUEST_ID: ContextVar[str | None] = ContextVar("REQUEST_ID", default=None)

_VALID_CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _resolve_request_id(header_value: str | None) -> str:
    if header_value and _VALID_CLIENT_ID.match(header_value):
        return header_value
    return str(uuid.uuid4())


class RequestContextMiddleware(BaseHTTPMiddleware):
    """§9.1 — 단일 request start/end 이벤트 소유자. 실제 로깅은
    `observability.logging.log_event`가 담당하고, 이 미들웨어는 request_id
    설정/리셋과 start/end 이벤트 발행만 책임진다."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        from simple_qna_rag.observability.logging import log_event

        request_id = _resolve_request_id(request.headers.get("X-Request-Id"))
        token = REQUEST_ID.set(request_id)
        route = "rag" if request.url.path == "/rag" else "health"
        registry = getattr(request.app.state, "metrics_registry", None)

        import time

        start = time.perf_counter()
        log_event(
            "request_start",
            route=route,
            method=request.method,
            metrics_registry=registry,
        )
        status_code = 500
        error_code: str | None = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            error_code = "internal"
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            log_event(
                "request_end",
                route=route,
                status_code=status_code,
                duration_ms=duration_ms,
                error_code=error_code,
                metrics_registry=registry,
            )
            if registry is not None:
                from simple_qna_rag.observability.metrics import clamp_route, clamp_status

                status_class = f"{status_code // 100}xx"
                registry.rag_requests_total.labels(
                    route=clamp_route(route), status=clamp_status(status_class)
                ).inc()
                registry.rag_request_duration_seconds.labels(
                    route=clamp_route(route)
                ).observe(duration_ms / 1000)
            REQUEST_ID.reset(token)
