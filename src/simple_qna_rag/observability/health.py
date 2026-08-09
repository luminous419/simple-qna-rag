"""M4.1 §3.4 — readiness state table.

Pure function, no I/O, no `app.state` access — `evaluate_readiness()` only
combines three already-computed error states into a `(status_code, reason)`
pair. `bootstrap_error` takes precedence: a static/template mount failure is
a deployment defect that must surface regardless of settings/engine state.
"""

from __future__ import annotations


def evaluate_readiness(
    bootstrap_error: str | None,
    settings_error: str | None,
    engine_error: str | None,
) -> tuple[int, str]:
    if bootstrap_error is not None:
        return 503, "static_mount_failed"
    if settings_error is not None:
        return 503, "settings_invalid"
    if engine_error is not None:
        return 503, "engine_init_failed"
    return 200, "ok"
