"""Bounded FIFO admission for blocking RAG calls."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Callable, Literal
from uuid import uuid4

from simple_qna_rag.observability.deadline import Deadline, bind_deadline, reset_deadline
from simple_qna_rag.observability.terminal_ledger import (
    record_executor_terminal, terminal_ledger,
)
from simple_qna_rag.web.scheduling import AsyncioDeadlineScheduler, DeadlineScheduler


class TicketState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    REJECTED = "REJECTED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"
    ABANDONED = "ABANDONED"


class AdmissionRejected(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SubmitFailed(RuntimeError):
    pass


class QueueTimeoutError(TimeoutError):
    pass


class ExecutionTimeoutError(TimeoutError):
    pass


@dataclass(frozen=True)
class ExecutorSnapshot:
    lifecycle: str
    running: int
    queued: int
    orphaned: int
    concurrency_limit: int
    queue_limit: int
    accepted_total: int
    admission_rejected_total: int
    queue_timeout_total: int
    execution_timeout_total: int
    cancelled_total: int
    completed_total: int
    terminal_rejected_total: int
    capacity_full: bool
    capacity_edge_at: float
    capacity_version: int
    stopped_with_running: int | None
    stopped_with_orphaned: int | None


@dataclass
class _Ticket:
    ticket_id: int
    callable: Callable[[], Any]
    loop_future: asyncio.Future[Any]
    state: TicketState = TicketState.QUEUED
    queue_timer: asyncio.TimerHandle | None = None
    exec_timer: asyncio.TimerHandle | None = None
    pool_future: Future[Any] | None = None
    caller_finalized: bool = False
    resource_finalized: bool = False


@dataclass
class _DrainWaiter:
    absolute_deadline: float
    future: asyncio.Future[str]
    winner: Literal["resource_zero", "deadline"] | None = None
    winner_sequence: int | None = None


@dataclass(frozen=True)
class TicketHandle:
    ticket_id: int
    _loop_future: asyncio.Future[Any]
    _executor: "QueryExecutor"

    async def result(self) -> Any:
        try:
            return await asyncio.shield(self._loop_future)
        except asyncio.CancelledError:
            self._executor.cancel(self.ticket_id)
            raise

    def cancel(self) -> None:
        self._executor.cancel(self.ticket_id)


def _run_ticket(callable_: Callable[[], Any], deadline_value: float, clock: Callable[[], float]) -> Any:
    token = bind_deadline(Deadline(deadline_value, clock))
    try:
        return callable_()
    finally:
        reset_deadline(token)


class QueryExecutor:
    def __init__(
        self, *, concurrency_limit: int, queue_limit: int, queue_timeout: float,
        execution_timeout: float, loop: asyncio.AbstractEventLoop | None = None,
        clock: Callable[[], float] = time.monotonic,
        deadline_scheduler: DeadlineScheduler | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_running_loop()
        self._ledger_producer = terminal_ledger.bind()
        self._executor_id = uuid4().hex
        self._clock = clock
        self._deadline_scheduler = deadline_scheduler or AsyncioDeadlineScheduler(self._loop, clock)
        self._concurrency_limit = concurrency_limit
        self._queue_limit = queue_limit
        self._queue_timeout = queue_timeout
        self._execution_timeout = execution_timeout
        self._pool = ThreadPoolExecutor(max_workers=concurrency_limit, thread_name_prefix="rag-query")
        self._lock = Lock()
        self._queue: deque[int] = deque()
        self._tickets: dict[int, _Ticket] = {}
        self._next_id = 0
        self._running = self._orphaned = 0
        self._lifecycle = "READY"
        self._accepted = 0
        self._admission = {k: 0 for k in ("not_ready", "overloaded", "submit_failed")}
        self._terminal = {k: 0 for k in ("queue_timeout", "execution_timeout", "cancelled", "completed", "rejected")}
        self._capacity_full = False
        self._capacity_edge_at = clock()
        self._capacity_version = 0
        self._shutdown_called = False
        self._stopped_running: int | None = None
        self._stopped_orphaned: int | None = None
        self._drained = asyncio.Event()
        self._drained.set()
        self._drain_waiters: list[_DrainWaiter] = []
        self._linearization_sequence = 0

    def _edge_locked(self) -> None:
        full = self._running == self._concurrency_limit and len(self._queue) == self._queue_limit
        if full != self._capacity_full:
            self._capacity_full = full
            self._capacity_edge_at = self._clock()
            self._capacity_version += 1

    def submit(self, callable_: Callable[[], Any]) -> TicketHandle:
        commit: tuple[int, Future[Any]] | None = None
        with self._lock:
            if self._lifecycle != "READY":
                self._admission["not_ready"] += 1
                record_executor_terminal(self._ledger_producer, "not_ready")
                raise AdmissionRejected("not_ready")
            if self._running >= self._concurrency_limit and len(self._queue) >= self._queue_limit:
                self._admission["overloaded"] += 1
                record_executor_terminal(self._ledger_producer, "overloaded")
                raise AdmissionRejected("overloaded")
            self._next_id += 1
            ticket = _Ticket(self._next_id, callable_, self._loop.create_future())
            self._tickets[ticket.ticket_id] = ticket
            self._accepted += 1
            if self._running < self._concurrency_limit:
                try:
                    commit = self._start_locked(ticket)
                except RuntimeError as exc:
                    self._tickets.pop(ticket.ticket_id, None)
                    self._accepted -= 1
                    self._admission["submit_failed"] += 1
                    record_executor_terminal(self._ledger_producer, "internal")
                    raise SubmitFailed() from exc
            else:
                self._queue.append(ticket.ticket_id)
                ticket.queue_timer = self._loop.call_later(
                    self._queue_timeout, self._queue_timeout_fired, ticket.ticket_id
                )
                self._edge_locked()
        if commit:
            commit[1].add_done_callback(lambda future, tid=commit[0]: self._pool_done(tid, future))
        return TicketHandle(ticket.ticket_id, ticket.loop_future, self)

    def _start_locked(self, ticket: _Ticket) -> tuple[int, Future[Any]]:
        deadline = self._clock() + self._execution_timeout
        pool_future = self._pool.submit(_run_ticket, ticket.callable, deadline, self._clock)
        ticket.pool_future = pool_future
        ticket.state = TicketState.RUNNING
        if ticket.queue_timer:
            ticket.queue_timer.cancel()
        ticket.exec_timer = self._loop.call_later(
            self._execution_timeout, self._execution_timeout_fired, ticket.ticket_id
        )
        self._running += 1
        self._drained.clear()
        self._edge_locked()
        return ticket.ticket_id, pool_future

    def _promote_locked(self) -> tuple[tuple[int, Future[Any]] | None, list[_Ticket]]:
        """Consume failed FIFO heads until one starts or no work remains.

        The lock owns state transitions only.  Future delivery and callback
        registration happen after it is released, so a user future cannot
        re-enter the executor while its invariants are mid-update.
        """
        failed: list[_Ticket] = []
        while (self._lifecycle == "READY" and self._running < self._concurrency_limit
               and self._queue):
            ticket = self._tickets[self._queue.popleft()]
            try:
                return self._start_locked(ticket), failed
            except RuntimeError:
                if ticket.queue_timer:
                    ticket.queue_timer.cancel()
                    ticket.queue_timer = None
                ticket.state = TicketState.REJECTED
                ticket.caller_finalized = ticket.resource_finalized = True
                self._terminal["rejected"] += 1
                record_executor_terminal(self._ledger_producer, "internal")
                self._tickets.pop(ticket.ticket_id, None)
                failed.append(ticket)
                self._edge_locked()
        return None, failed

    def _register(self, commit: tuple[int, Future[Any]] | None) -> None:
        if commit:
            commit[1].add_done_callback(lambda future, tid=commit[0]: self._pool_done(tid, future))

    def _finish_promotion(
        self, result: tuple[tuple[int, Future[Any]] | None, list[_Ticket]]
    ) -> None:
        commit, failed = result
        for ticket in failed:
            self._deliver_error(ticket, SubmitFailed())
        self._register(commit)

    def _queue_timeout_fired(self, ticket_id: int) -> None:
        promotion = (None, [])
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if not ticket or ticket.state != TicketState.QUEUED or ticket.caller_finalized:
                return
            self._queue.remove(ticket_id)
            ticket.state = TicketState.TIMED_OUT
            ticket.caller_finalized = ticket.resource_finalized = True
            self._terminal["queue_timeout"] += 1
            record_executor_terminal(self._ledger_producer, "queue_timeout")
            if ticket.queue_timer:
                ticket.queue_timer.cancel()
                ticket.queue_timer = None
            self._tickets.pop(ticket_id, None)
            self._edge_locked()
            promotion = self._promote_locked()
        self._deliver_error(ticket, QueueTimeoutError())
        self._finish_promotion(promotion)

    def _execution_timeout_fired(self, ticket_id: int) -> None:
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if not ticket or ticket.state != TicketState.RUNNING or ticket.caller_finalized:
                return
            ticket.state = TicketState.ABANDONED
            ticket.caller_finalized = True
            self._orphaned += 1
            self._terminal["execution_timeout"] += 1
            record_executor_terminal(self._ledger_producer, "execution_timeout")
        self._deliver_error(ticket, ExecutionTimeoutError())

    def cancel(self, ticket_id: int) -> None:
        promotion = (None, [])
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if not ticket or ticket.caller_finalized:
                return
            ticket.caller_finalized = True
            self._terminal["cancelled"] += 1
            record_executor_terminal(self._ledger_producer, "client_disconnected")
            if ticket.state == TicketState.QUEUED:
                ticket.state = TicketState.CANCELLED
                ticket.resource_finalized = True
                self._queue.remove(ticket_id)
                if ticket.queue_timer:
                    ticket.queue_timer.cancel()
                    ticket.queue_timer = None
                self._tickets.pop(ticket_id, None)
                self._edge_locked()
                promotion = self._promote_locked()
            elif ticket.state == TicketState.RUNNING:
                ticket.state = TicketState.ABANDONED
                self._orphaned += 1
                if ticket.exec_timer:
                    ticket.exec_timer.cancel()
        self._finish_promotion(promotion)

    def _pool_done(self, ticket_id: int, future: Future[Any]) -> None:
        try:
            result, error = future.result(), None
        except BaseException as exc:  # result is normalized by the HTTP boundary
            result, error = None, exc
        promotion = (None, [])
        deliver = False
        ticket: _Ticket | None = None
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if not ticket or ticket.resource_finalized:
                return
            ticket.resource_finalized = True
            self._running -= 1
            if ticket.state == TicketState.ABANDONED:
                self._orphaned -= 1
            elif not ticket.caller_finalized:
                ticket.caller_finalized = True
                ticket.state = TicketState.DONE
                self._terminal["completed"] += 1
                record_executor_terminal(self._ledger_producer, "internal" if error else "success")
                deliver = True
            self._tickets.pop(ticket_id, None)
            self._edge_locked()
            if self._running == 0:
                for waiter in tuple(self._drain_waiters):
                    self._claim_drain_waiter_locked(waiter, "resource_zero")
                try:
                    self._loop.call_soon_threadsafe(self._drained.set)
                except RuntimeError:
                    pass
            promotion = self._promote_locked()
        if deliver and ticket is not None:
            try:
                self._loop.call_soon_threadsafe(self._deliver, ticket, result, error)
            except RuntimeError:
                pass
        self._finish_promotion(promotion)

    def _deliver(self, ticket: _Ticket, result: Any, error: BaseException | None) -> None:
        if ticket.exec_timer:
            ticket.exec_timer.cancel()
        if not ticket.loop_future.done():
            ticket.loop_future.set_exception(error) if error else ticket.loop_future.set_result(result)

    def _deliver_error(self, ticket: _Ticket, error: BaseException) -> None:
        if not ticket.loop_future.done():
            ticket.loop_future.set_exception(error)

    def begin_drain(self) -> None:
        rejected: list[_Ticket] = []
        with self._lock:
            if self._lifecycle != "READY":
                return
            self._lifecycle = "DRAINING"
            while self._queue:
                ticket = self._tickets[self._queue.popleft()]
                ticket.state = TicketState.REJECTED
                ticket.caller_finalized = ticket.resource_finalized = True
                if ticket.queue_timer:
                    ticket.queue_timer.cancel()
                    ticket.queue_timer = None
                self._terminal["rejected"] += 1
                record_executor_terminal(self._ledger_producer, "not_ready")
                self._tickets.pop(ticket.ticket_id, None)
                rejected.append(ticket)
            self._edge_locked()
        for ticket in rejected:
            self._deliver_error(ticket, AdmissionRejected("not_ready"))

    async def wait_drained(self, timeout: float) -> bool:
        absolute_deadline = self._clock() + max(0.0, timeout)
        waiter = _DrainWaiter(absolute_deadline, self._loop.create_future())
        with self._lock:
            self._drain_waiters.append(waiter)
            if timeout > 0.0 and self._running == 0:
                self._claim_drain_waiter_locked(waiter, "resource_zero")
        deadline_handle = self._deadline_scheduler.schedule_at(
            absolute_deadline, lambda: self._claim_drain_waiter(waiter, "deadline")
        )
        try:
            return await waiter.future == "resource_zero"
        finally:
            deadline_handle.cancel()
            with self._lock:
                if waiter in self._drain_waiters:
                    self._drain_waiters.remove(waiter)

    def _claim_drain_waiter(self, waiter: _DrainWaiter, winner: Literal["resource_zero", "deadline"]) -> None:
        with self._lock:
            self._claim_drain_waiter_locked(waiter, winner)

    def _claim_drain_waiter_locked(
        self, waiter: _DrainWaiter, winner: Literal["resource_zero", "deadline"]
    ) -> bool:
        self._linearization_sequence += 1
        if waiter.winner is not None:
            return False
        waiter.winner = winner
        waiter.winner_sequence = self._linearization_sequence
        try:
            self._loop.call_soon_threadsafe(self._publish_drain_winner, waiter)
        except RuntimeError:
            pass
        return True

    @staticmethod
    def _publish_drain_winner(waiter: _DrainWaiter) -> None:
        if not waiter.future.done() and waiter.winner is not None:
            waiter.future.set_result(waiter.winner)

    def shutdown(self, wait: bool = False, cancel_futures: bool = True) -> None:
        with self._lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True
            self._lifecycle = "STOPPED"
            self._stopped_running = self._running
            self._stopped_orphaned = self._orphaned
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)

    def snapshot(self) -> ExecutorSnapshot:
        with self._lock:
            snapshot = ExecutorSnapshot(
                self._lifecycle, self._running, len(self._queue), self._orphaned,
                self._concurrency_limit, self._queue_limit, self._accepted,
                sum(self._admission.values()), self._terminal["queue_timeout"],
                self._terminal["execution_timeout"], self._terminal["cancelled"],
                self._terminal["completed"], self._terminal["rejected"],
                self._capacity_full, self._capacity_edge_at, self._capacity_version,
                self._stopped_running, self._stopped_orphaned,
            )
        terminal_ledger.observe_executor_snapshot(
            self._ledger_producer, self._executor_id, snapshot
        )
        return snapshot
