"""M4.1 §3.4 — readiness state table.

Pure function, no I/O, no `app.state` access — `evaluate_readiness()` only
combines three already-computed error states into a `(status_code, reason)`
pair. `bootstrap_error` takes precedence: a static/template mount failure is
a deployment defect that must surface regardless of settings/engine state.
"""

from __future__ import annotations

import time
from typing import Callable


def evaluate_readiness(
    bootstrap_error: str | None,
    settings_error: str | None,
    engine_error: str | None,
    *, lifecycle: str | None = None, saturated: bool = False,
    orphaned: int = 0, concurrency_limit: int = 0,
) -> tuple[int, str]:
    if bootstrap_error is not None:
        return 503, "static_mount_failed"
    if settings_error is not None:
        return 503, "settings_invalid"
    if engine_error is not None:
        return 503, "engine_init_failed"
    if lifecycle in {"DRAINING", "STOPPED"}:
        return 503, "draining"
    if concurrency_limit > 0 and orphaned == concurrency_limit:
        return 503, "orphan_workers"
    if saturated:
        return 503, "queue_saturated"
    return 200, "ok"


class SaturationDebounce:
    def __init__(self, clock: Callable[[], float] = time.monotonic, hold_seconds: float = 1.0) -> None:
        self._clock = clock
        self._hold = hold_seconds
        self._active = False

    def evaluate(self, *, full: bool, edge_at: float, version: int) -> bool:
        del version
        if self._clock() - edge_at >= self._hold:
            self._active = full
        return self._active
