"""Safe, fixed-shape public API errors."""

from dataclasses import dataclass

from fastapi.responses import JSONResponse


@dataclass(frozen=True)
class ApiError:
    code: str
    http_status: int
    retryable: bool


ERRORS = {
    "invalid_request": ApiError("invalid_request", 400, False),
    "payload_too_large": ApiError("payload_too_large", 413, False),
    "not_ready": ApiError("not_ready", 503, True),
    "overloaded": ApiError("overloaded", 503, True),
    "queue_timeout": ApiError("queue_timeout", 503, True),
    "execution_timeout": ApiError("execution_timeout", 504, True),
    "internal": ApiError("internal", 500, False),
}


def error_response(code: str) -> JSONResponse:
    error = ERRORS.get(code, ERRORS["internal"])
    return JSONResponse(
        status_code=error.http_status,
        content={
            "success": False,
            "answer": "요청을 처리할 수 없습니다. 잠시 후 다시 시도해주세요.",
            "sources": [],
            "search_type": "unknown",
            "error": {"code": error.code, "retryable": error.retryable},
        },
    )
