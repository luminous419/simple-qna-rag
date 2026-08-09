"""M4.1 §7 — bounded process-local Prometheus metrics.

Seven families, enum/allowlist-only label values, and a registry-per-call
factory that keeps tests isolated from a shared global REGISTRY. Every
consumer of these metrics goes through `app.state.metrics_registry`
(§7.4) — this module never touches `prometheus_client.REGISTRY` directly.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, disable_created_metrics

ROUTES = frozenset({"rag", "health"})
STATUSES = frozenset({"2xx", "4xx", "5xx"})
STAGES = frozenset({"routing", "web_search", "retrieval", "generation"})
ERROR_CODES = frozenset({"timeout", "upstream", "validation", "internal"})
READINESS_REASONS = frozenset(
    {"ok", "settings_invalid", "engine_init_failed", "static_mount_failed", "other"}
)
FALLBACK_KINDS = frozenset({"web_search", "mmr_vector_source"})
FALLBACK_REASONS = frozenset({"low_confidence", "empty_retrieval", "validation_failed", "other"})

DURATION_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)


def _clamp_label(value: str, allowed: frozenset[str], default: str = "other") -> str:
    return value if value in allowed else default


def clamp_route(value: str) -> str:
    return _clamp_label(value, ROUTES, default="health")


def clamp_status(value: str) -> str:
    return _clamp_label(value, STATUSES, default="5xx")


def clamp_stage(value: str) -> str:
    return _clamp_label(value, STAGES, default="generation")


def clamp_error_code(value: str) -> str:
    return _clamp_label(value, ERROR_CODES, default="internal")


def clamp_readiness_reason(value: str) -> str:
    return _clamp_label(value, READINESS_REASONS, default="other")


def clamp_fallback_kind(value: str) -> str:
    return _clamp_label(value, FALLBACK_KINDS, default="web_search")


def clamp_fallback_reason(value: str) -> str:
    return _clamp_label(value, FALLBACK_REASONS, default="other")


# ---------------------------------------------------------------------------
# Product-path observation helpers (CR-I1-MAJ-03) — the single place agent.py
# and rag_engine.py reach into a registry from, so every call site clamps
# labels the same way and is a no-op when no registry was injected (CLI
# paths, or tests that don't care about metrics).
# ---------------------------------------------------------------------------


def record_stage_duration(registry: Any, stage: str, duration_seconds: float) -> None:
    if registry is None:
        return
    registry.rag_stage_duration_seconds.labels(stage=clamp_stage(stage)).observe(duration_seconds)


def record_stage_error(registry: Any, stage: str, error_code: str) -> None:
    if registry is None:
        return
    registry.rag_stage_errors_total.labels(
        stage=clamp_stage(stage), error_code=clamp_error_code(error_code)
    ).inc()


def record_fallback(registry: Any, kind: str, reason: str) -> None:
    if registry is None:
        return
    registry.rag_fallback_total.labels(
        kind=clamp_fallback_kind(kind), reason=clamp_fallback_reason(reason)
    ).inc()


_process_metrics_configured = False


def configure_process_metrics() -> None:
    """process-global, 1회만 실제 동작. Counter/Histogram/Gauge 생성 전에
    `disable_created_metrics()`를 호출한다. registry와 무관."""
    global _process_metrics_configured
    if _process_metrics_configured:
        return
    disable_created_metrics()
    _process_metrics_configured = True


def build_metrics_registry(registry: CollectorRegistry | None = None) -> CollectorRegistry:
    """registry별 factory — 호출마다 §7.2 7-family를 등록한 CollectorRegistry를
    반환한다(fresh registry로 테스트 격리)."""
    configure_process_metrics()
    reg = registry if registry is not None else CollectorRegistry()

    reg.rag_requests_total = Counter(
        "rag_requests_total",
        "Total /rag and /health requests by route and status class.",
        ["route", "status"],
        registry=reg,
    )
    reg.rag_request_duration_seconds = Histogram(
        "rag_request_duration_seconds",
        "Request duration by route.",
        ["route"],
        buckets=DURATION_BUCKETS,
        registry=reg,
    )
    reg.rag_stage_duration_seconds = Histogram(
        "rag_stage_duration_seconds",
        "Pipeline stage duration.",
        ["stage"],
        buckets=DURATION_BUCKETS,
        registry=reg,
    )
    reg.rag_stage_errors_total = Counter(
        "rag_stage_errors_total",
        "Pipeline stage errors by stage and error_code.",
        ["stage", "error_code"],
        registry=reg,
    )
    reg.rag_readiness = Gauge(
        "rag_readiness",
        "Current readiness state (1 for the active reason, 0 otherwise).",
        ["reason"],
        registry=reg,
    )
    reg.rag_fallback_total = Counter(
        "rag_fallback_total",
        "Fallback activations by kind and reason.",
        ["kind", "reason"],
        registry=reg,
    )
    reg.logging_dropped_fields_total = Counter(
        "logging_dropped_fields_total",
        "log_event() calls that dropped a disallowed field.",
        registry=reg,
    )

    return reg
