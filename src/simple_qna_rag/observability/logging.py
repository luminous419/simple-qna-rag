"""M4.1 §6 — payload-safe structured JSON logging.

`log_event()` is the single entry point for request/routing/retrieval/
generation/startup/readiness events. It never raises for a payload-safety
violation (disallowed key or an absolute-path-looking value): the offending
field is dropped and, if a metrics registry was injected, the drop is
counted via `logging_dropped_fields_total` (REQ-003.4). Only
`log_event_strict()` (dev/CI use) raises.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any, Callable

from simple_qna_rag.observability.metrics import ERROR_CODES, ROUTES, STAGES

SERVICE_NAME = "simple-qna-rag"

try:
    SERVICE_VERSION = _pkg_version("simple-qna-rag")
except PackageNotFoundError:
    SERVICE_VERSION = "unknown"

ALLOWED_EVENTS = frozenset(
    {
        "request_start",
        "request_end",
        "routing",
        "web_search",
        "retrieval",
        "generation",
        "startup",
        "readiness",
    }
)

_PROCESS_SCOPED_EVENTS = frozenset({"startup", "readiness"})

_COMMON_KEYS = frozenset({"timestamp", "level", "event", "service", "version", "request_id"})

# (required, optional) additional keys per event, beyond the common set.
_EVENT_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "request_start": (frozenset({"route", "method"}), frozenset()),
    "request_end": (
        frozenset({"route", "status_code", "duration_ms"}),
        frozenset({"method", "error_code"}),
    ),
    "routing": (frozenset({"decision", "confidence"}), frozenset()),
    "web_search": (frozenset({"stage", "duration_ms"}), frozenset({"error_code"})),
    "retrieval": (frozenset({"stage", "duration_ms"}), frozenset({"error_code"})),
    "generation": (frozenset({"stage", "duration_ms"}), frozenset({"error_code"})),
    "startup": (frozenset(), frozenset({"reason"})),
    "readiness": (frozenset(), frozenset({"reason"})),
}

# Field names that are always rejected regardless of the per-event schema —
# defense in depth against a caller accidentally forwarding raw payload.
_FORBIDDEN_KEY_NAMES = frozenset(
    {"question", "answer", "context", "sources_content", "prompt", "exception_text"}
)
_ABS_PATH_RE = re.compile(r"^(/Users/|/home/|C:\\)")


def _is_forbidden_value(value: Any) -> bool:
    return isinstance(value, str) and bool(_ABS_PATH_RE.match(value))


# ---------------------------------------------------------------------------
# CR-I1-MIN-01 closure — required key/type/enum safety policy.
#
# `_EVENT_KEYS`'s required/optional split was previously used only to build
# the allowed-key union; a wrong-typed/out-of-enum `route`/`stage`/
# `error_code`/`level`/`status_code`, or a silently missing required key,
# was recorded as-is. `_FIELD_VALIDATORS` (reusing `observability.metrics`'s
# label enums as the single source of truth) now checks each of those five
# fields; `log_event()` (non-strict, product path) clamps an invalid value
# to `_FIELD_SAFE_DEFAULTS`/fills a missing required key from
# `_REQUIRED_FIELD_DEFAULTS`, while `log_event_strict()` (dev/CI fixture-
# schema gate) raises ValueError for either violation instead.
# `confidence` keeps its pre-existing unconditional round-and-clamp
# behavior in both paths (not a review-named enum/type field).
# ---------------------------------------------------------------------------

_LEVEL_VALUES = frozenset({"info", "warning", "error"})


def _is_valid_status_code(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599


def _is_valid_duration_ms(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


_FIELD_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "route": lambda v: v in ROUTES,
    "stage": lambda v: v in STAGES,
    "error_code": lambda v: v in ERROR_CODES,
    "level": lambda v: v in _LEVEL_VALUES,
    "status_code": _is_valid_status_code,
    "duration_ms": _is_valid_duration_ms,
}

_FIELD_SAFE_DEFAULTS: dict[str, Any] = {
    "route": "health",
    "stage": "generation",
    "error_code": "internal",
    "level": "info",
    "status_code": 500,
    "duration_ms": 0.0,
}

_REQUIRED_FIELD_DEFAULTS: dict[str, Any] = {
    "route": "health",
    "method": "UNKNOWN",
    "status_code": 500,
    "duration_ms": 0.0,
    "decision": "unknown",
    "confidence": 0.0,
    "stage": "generation",
}


def _normalize_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return round(min(1.0, max(0.0, float(value))), 3)


class SafeJsonHandler(logging.Handler):
    """Writes one canonical JSON object per line to `stream` (default
    stdout). Any emit-time failure is swallowed here — `log_event()` also
    wraps its own call in try/except so a broken handler never fails a
    product request (REQ-003.4)."""

    def __init__(self, stream=None) -> None:
        super().__init__()
        self.stream = stream if stream is not None else sys.stdout

    def emit(self, record: logging.LogRecord) -> None:
        payload = record.__dict__.get("event_payload")
        if payload is None:
            payload = {"level": record.levelname, "message": record.getMessage()}
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        self.stream.write(text + "\n")
        if hasattr(self.stream, "flush"):
            self.stream.flush()


_logger = logging.getLogger("simple_qna_rag.observability")
_logger.setLevel(logging.INFO)
_logger.propagate = False
if not any(isinstance(h, SafeJsonHandler) for h in _logger.handlers):
    _logger.addHandler(SafeJsonHandler())


def _build_record(
    event: str, fields: dict[str, Any], *, strict: bool = False
) -> tuple[dict[str, Any], list[str]]:
    if event not in ALLOWED_EVENTS:
        raise ValueError(f"unknown event: {event!r}")

    required, optional = _EVENT_KEYS.get(event, (frozenset(), frozenset()))
    allowed_keys = _COMMON_KEYS | required | optional
    request_id = None if event in _PROCESS_SCOPED_EVENTS else _current_request_id()

    level = fields.pop("level", "info")
    if not _FIELD_VALIDATORS["level"](level):
        if strict:
            raise ValueError(f"invalid level for event {event!r}: {level!r}")
        level = _FIELD_SAFE_DEFAULTS["level"]

    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "request_id": fields.pop("request_id", request_id),
    }

    dropped: list[str] = []
    provided_required: set[str] = set()
    for key, value in fields.items():
        if key not in allowed_keys or key in _FORBIDDEN_KEY_NAMES or _is_forbidden_value(value):
            dropped.append(key)
            continue
        if key == "confidence":
            value = _normalize_confidence(value)
        else:
            validator = _FIELD_VALIDATORS.get(key)
            if validator is not None and not validator(value):
                if strict:
                    raise ValueError(f"invalid {key} for event {event!r}: {value!r}")
                value = _FIELD_SAFE_DEFAULTS[key]
        record[key] = value
        if key in required:
            provided_required.add(key)

    missing_required = required - provided_required
    if missing_required:
        if strict:
            raise ValueError(f"missing required fields for event {event!r}: {sorted(missing_required)}")
        for key in missing_required:
            record[key] = _REQUIRED_FIELD_DEFAULTS.get(key)

    return record, dropped


def _current_request_id() -> str | None:
    from simple_qna_rag.observability.request_context import REQUEST_ID

    return REQUEST_ID.get()


def log_event(event: str, *, metrics_registry: Any = None, **fields: Any) -> None:
    """Non-strict entry point — drops disallowed fields, clamps out-of-enum/
    wrong-typed values to a safe default, and fills a missing required field
    with a safe default (CR-I1-MIN-01) instead of raising."""
    record, dropped = _build_record(event, dict(fields), strict=False)
    if dropped and metrics_registry is not None:
        metrics_registry.logging_dropped_fields_total.inc(len(dropped))
    try:
        _logger.info(event, extra={"event_payload": record})
    except Exception:  # noqa: BLE001 - a broken logging backend must not fail the request
        pass


def log_event_strict(event: str, **fields: Any) -> None:
    """Dev/CI-only entry point — raises ValueError for a disallowed field, an
    out-of-enum/wrong-typed value, or a missing required field, instead of
    dropping/clamping/defaulting (CR-I1-MIN-01 fixture-schema gate)."""
    record, dropped = _build_record(event, dict(fields), strict=True)
    if dropped:
        raise ValueError(f"disallowed fields for event {event!r}: {sorted(dropped)}")
    _logger.info(event, extra={"event_payload": record})
