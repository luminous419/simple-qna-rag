"""M4.1 §6.3 — request-scoped request ID plumbing."""

from __future__ import annotations

import re
import uuid
import asyncio
from contextvars import ContextVar

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID: ContextVar[str | None] = ContextVar("REQUEST_ID", default=None)

_VALID_CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _resolve_request_id(header_value: str | None) -> str:
    if header_value and _VALID_CLIENT_ID.match(header_value):
        return header_value
    return str(uuid.uuid4())


class RequestContextMiddleware:
    """§9.1 — 단일 request start/end 이벤트 소유자. 실제 로깅은
    `observability.logging.log_event`가 담당하고, 이 미들웨어는 request_id
    설정/리셋과 start/end 이벤트 발행만 책임진다."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from simple_qna_rag.observability.logging import log_event
        from simple_qna_rag.observability.terminal_ledger import terminal_ledger

        ledger_producer = terminal_ledger.bind()
        request_id = _resolve_request_id(Headers(scope=scope).get("X-Request-Id"))
        token = REQUEST_ID.set(request_id)
        route = "rag" if scope.get("path") == "/rag" else "health"
        app = scope.get("app")
        registry = getattr(getattr(app, "state", None), "metrics_registry", None)

        import time

        start = time.perf_counter()
        try:
            log_event(
                "request_start", route=route, method=scope.get("method", ""),
                metrics_registry=registry,
            )
        except Exception:
            pass
        status_code = 500
        error_code: str | None = None
        frames = 0
        disconnect_observed = False

        async def observed_receive() -> Message:
            nonlocal disconnect_observed
            message = await receive()
            disconnect_observed |= message["type"] == "http.disconnect"
            return message

        async def observed_send(message: Message) -> None:
            nonlocal status_code, frames
            frames += 1
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)
        try:
            await self.app(scope, observed_receive, observed_send)
            marker = scope.get("state", {}).get("rag_terminal")
            if frames == 0 and (marker == "client_disconnected" or disconnect_observed):
                status_code, error_code = 499, "client_disconnected"
            elif frames == 0:
                error_code = "internal"
                raise RuntimeError("downstream_no_response")
        except asyncio.CancelledError:
            if disconnect_observed:
                status_code, error_code = 499, "client_disconnected"
            else:
                error_code = "cancelled"
            raise
        except Exception:
            error_code = "internal"
            raise
        finally:
            from simple_qna_rag.observability.terminal_ledger import (
                record_rag_terminal, record_request_terminal,
            )

            marker = scope.get("state", {}).get("rag_terminal")
            terminal = marker if marker else (
                "success" if 200 <= status_code < 400 else
                "invalid_request" if status_code == 400 else
                "payload_too_large" if status_code == 413 else
                "not_ready" if status_code == 503 else
                "internal"
            )
            record_request_terminal(ledger_producer, terminal)
            if route == "rag":
                record_rag_terminal(ledger_producer, terminal)
            duration_ms = (time.perf_counter() - start) * 1000
            try:
                log_event(
                    "request_end", route=route, status_code=status_code,
                    duration_ms=duration_ms, error_code=error_code,
                    metrics_registry=registry,
                )
            except Exception:
                pass
            if registry is not None:
                from simple_qna_rag.observability.metrics import clamp_route, clamp_status

                status_class = f"{status_code // 100}xx"
                try:
                    registry.rag_requests_total.labels(
                        route=clamp_route(route), status=clamp_status(status_class)
                    ).inc()
                    registry.rag_request_duration_seconds.labels(
                        route=clamp_route(route)
                    ).observe(duration_ms / 1000)
                except Exception:
                    pass
            REQUEST_ID.reset(token)
