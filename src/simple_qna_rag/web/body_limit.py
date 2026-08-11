"""ASGI request-body and content-encoding boundary."""

from __future__ import annotations

from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from simple_qna_rag.web.errors import error_response
from simple_qna_rag.observability.metrics import record_input_rejected

BOOTSTRAP_BODY_CEILING = 1_048_576


def _header(scope: Scope, name: bytes) -> bytes | None:
    for key, value in scope.get("headers", ()):
        if key.lower() == name:
            return value
    return None


def _content_encoding(scope: Scope) -> str:
    value = _header(scope, b"content-encoding")
    return value.decode("latin-1").strip().lower() if value else "identity"


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, bootstrap_max_bytes: int = BOOTSTRAP_BODY_CEILING) -> None:
        self.app = app
        self.bootstrap_max_bytes = bootstrap_max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") != "/rag":
            await self.app(scope, receive, send)
            return
        state = getattr(scope.get("app"), "state", None)
        settings = getattr(state, "settings", None)
        max_bytes = getattr(settings, "MAX_REQUEST_BODY_BYTES", self.bootstrap_max_bytes)
        observation: dict[str, Any] = {
            "wire_delivered_bytes": 0, "application_consumed_bytes": 0,
            "receive_calls": 0, "overflow": False,
        }
        scope.setdefault("state", {})["body_limit"] = max_bytes
        scope["state"]["body_limit_observation"] = observation

        def reject(reason: str):
            registry = getattr(state, "metrics_registry", None)
            try:
                record_input_rejected(registry, reason)
            except Exception:
                pass
            return error_response(reason)

        if _content_encoding(scope) not in {"", "identity"}:
            await reject("invalid_request")(scope, receive, send)
            return
        raw_length = _header(scope, b"content-length")
        try:
            declared = int(raw_length) if raw_length is not None else None
        except ValueError:
            await reject("invalid_request")(scope, receive, send)
            return
        if declared is not None and (declared < 0 or declared > max_bytes):
            await reject("payload_too_large")(scope, receive, send)
            return

        consumed = 0
        overflow = False

        async def limited_receive() -> Message:
            nonlocal consumed, overflow
            if overflow:
                return {"type": "http.disconnect"}
            message = await receive()
            observation["receive_calls"] += 1
            if message["type"] != "http.request":
                return message
            body = message.get("body", b"")
            observation["wire_delivered_bytes"] += len(body)
            allowed = max(0, max_bytes + 1 - consumed)
            prefix = body[:allowed]
            consumed += len(prefix)
            if consumed > max_bytes or len(prefix) < len(body):
                overflow = True
            observation["application_consumed_bytes"] = consumed
            observation["overflow"] = overflow
            return {**message, "body": prefix, "more_body": False if overflow else message.get("more_body", False)}

        scope["state"]["body_limit_overflow"] = lambda: overflow
        await self.app(scope, limited_receive, send)
