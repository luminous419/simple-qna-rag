"""Context-local execution deadlines and request-owned Ollama clients."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Callable, Iterator

import httpx


class UpstreamDeadlineExceeded(TimeoutError):
    pass


@dataclass(frozen=True)
class Deadline:
    monotonic_deadline: float
    clock: Callable[[], float] = time.monotonic

    def remaining(self) -> float:
        return max(0.0, self.monotonic_deadline - self.clock())

    def expired(self) -> bool:
        return self.remaining() <= 0.0


_DEADLINE: ContextVar[Deadline | None] = ContextVar("rag_execution_deadline", default=None)


def bind_deadline(deadline: Deadline) -> Token[Deadline | None]:
    return _DEADLINE.set(deadline)


def reset_deadline(token: Token[Deadline | None]) -> None:
    _DEADLINE.reset(token)


def current_deadline() -> Deadline | None:
    return _DEADLINE.get()


@contextmanager
def ollama_call_client(*, host: str, connect_timeout: float) -> Iterator[object]:
    import ollama

    deadline = current_deadline()
    remaining = deadline.remaining() if deadline is not None else 0.0
    if remaining <= 0:
        raise UpstreamDeadlineExceeded()
    bounded_connect = min(connect_timeout, remaining)
    timeout = httpx.Timeout(
        connect=bounded_connect, read=remaining, write=remaining, pool=bounded_connect
    )
    client = ollama.Client(host=host, timeout=timeout)
    try:
        yield client
    finally:
        transport = getattr(client, "_client", None)
        close = getattr(transport, "close", None)
        if close is not None:
            close()
