"""Bounded, epoch-isolated production evidence for M4.2 terminal conservation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from types import MappingProxyType
from typing import Any
from uuid import uuid4

TERMINALS = (
    "success", "invalid_request", "payload_too_large", "not_ready", "overloaded",
    "queue_timeout", "execution_timeout", "internal", "client_disconnected",
)
SOURCES = ("request", "rag", "executor")


@dataclass(frozen=True)
class LedgerProducer:
    """Immutable authority captured before a producer begins work."""

    epoch: int
    node_token: str
    node_id: str | None


@dataclass(frozen=True)
class TerminalLedgerSnapshot:
    node_id: str | None
    epoch: int
    node_token: str
    request_terminals: MappingProxyType
    rag_terminals: MappingProxyType
    executor_terminals: MappingProxyType
    stale_terminals: MappingProxyType
    executor_id: str | None
    snapshot_epoch: int | None
    executor_identity_conflicts: int
    before: MappingProxyType | None
    after: MappingProxyType | None


class TerminalLedger:
    """A fixed-cardinality ledger that never attributes stale work to a new node."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._epoch = 0
        self.reset(None)

    def reset(self, node_id: str | None) -> LedgerProducer:
        with self._lock:
            self._epoch += 1
            self._node_id = node_id
            self._node_token = uuid4().hex
            self._terminals = {
                source: {name: 0 for name in TERMINALS} for source in SOURCES
            }
            self._stale = {
                source: {name: 0 for name in TERMINALS} for source in SOURCES
            }
            self._executor_id: str | None = None
            self._snapshot_epoch: int | None = None
            self._executor_identity_conflicts = 0
            self._before: dict[str, Any] | None = None
            self._after: dict[str, Any] | None = None
            return self._binding_locked()

    def _binding_locked(self) -> LedgerProducer:
        return LedgerProducer(self._epoch, self._node_token, self._node_id)

    def bind(self) -> LedgerProducer:
        with self._lock:
            return self._binding_locked()

    def _active_locked(self, producer: LedgerProducer) -> bool:
        return (producer.epoch == self._epoch and producer.node_token == self._node_token
                and producer.node_id == self._node_id)

    def record(self, producer: LedgerProducer, source: str, terminal: str) -> bool:
        if terminal not in TERMINALS or source not in SOURCES:
            return False
        with self._lock:
            target = self._terminals if self._active_locked(producer) else self._stale
            target[source][terminal] += 1
            return target is self._terminals

    def observe_executor_snapshot(
        self, producer: LedgerProducer, executor_id: str, snapshot: Any
    ) -> bool:
        value = asdict(snapshot)
        with self._lock:
            if not self._active_locked(producer):
                return False
            if self._executor_id is None:
                self._executor_id = executor_id
                self._snapshot_epoch = producer.epoch
            elif self._executor_id != executor_id or self._snapshot_epoch != producer.epoch:
                self._executor_identity_conflicts += 1
                return False
            if self._before is None:
                self._before = value
            self._after = value
            return True

    def snapshot(self) -> TerminalLedgerSnapshot:
        with self._lock:
            maps = {source: MappingProxyType(dict(values))
                    for source, values in self._terminals.items()}
            stale = MappingProxyType({source: MappingProxyType(dict(values))
                                     for source, values in self._stale.items()})
            before = MappingProxyType(dict(self._before)) if self._before is not None else None
            after = MappingProxyType(dict(self._after)) if self._after is not None else None
            return TerminalLedgerSnapshot(
                self._node_id, self._epoch, self._node_token, maps["request"], maps["rag"],
                maps["executor"], stale, self._executor_id, self._snapshot_epoch,
                self._executor_identity_conflicts, before, after,
            )


terminal_ledger = TerminalLedger()


def record_request_terminal(producer: LedgerProducer, terminal: str) -> bool:
    return terminal_ledger.record(producer, "request", terminal)


def record_rag_terminal(producer: LedgerProducer, terminal: str) -> bool:
    return terminal_ledger.record(producer, "rag", terminal)


def record_executor_terminal(producer: LedgerProducer, terminal: str) -> bool:
    return terminal_ledger.record(producer, "executor", terminal)
