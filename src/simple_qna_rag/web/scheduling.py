"""Absolute-deadline schedulers used by serving-boundary state machines."""

from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass
from typing import Callable, Protocol


class Cancellable(Protocol):
    def cancel(self) -> None: ...


class DeadlineScheduler(Protocol):
    def schedule_at(self, when: float, callback: Callable[[], None]) -> Cancellable: ...


class AsyncioDeadlineScheduler:
    def __init__(self, loop: asyncio.AbstractEventLoop, clock: Callable[[], float]) -> None:
        self._loop = loop
        self._clock = clock

    def schedule_at(self, when: float, callback: Callable[[], None]) -> asyncio.TimerHandle:
        return self._loop.call_later(max(0.0, when - self._clock()), callback)


@dataclass
class _ManualHandle:
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


class ManualDeadlineScheduler:
    """Deterministic scheduler ordered by ``(deadline, insertion_sequence)``."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._sequence = 0
        self._items: list[tuple[float, int, _ManualHandle, Callable[[], None]]] = []

    def schedule_at(self, when: float, callback: Callable[[], None]) -> _ManualHandle:
        self._sequence += 1
        handle = _ManualHandle()
        heapq.heappush(self._items, (when, self._sequence, handle, callback))
        return handle

    def advance_to(self, when: float) -> None:
        while self._items and self._items[0][0] <= when:
            _, _, handle, callback = heapq.heappop(self._items)
            if not handle.cancelled:
                callback()

    @property
    def pending(self) -> int:
        return sum(not item[2].cancelled for item in self._items)
