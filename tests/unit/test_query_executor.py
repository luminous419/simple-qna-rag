import asyncio
import threading

import pytest

from simple_qna_rag.web.concurrency import AdmissionRejected, QueryExecutor, SubmitFailed


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_profile_1_2_5_exact_receipt(m42_receipt):
    loop = asyncio.get_running_loop()
    release = threading.Event()
    started = threading.Event()
    executor = QueryExecutor(concurrency_limit=1, queue_limit=2, queue_timeout=5,
                             execution_timeout=5, loop=loop)
    before = executor.snapshot()

    def stall():
        started.set()
        release.wait()
        return "ok"

    handles = [executor.submit(stall)]
    await asyncio.to_thread(started.wait)
    handles.extend([executor.submit(lambda: "a"), executor.submit(lambda: "b")])
    rejected = 0
    for _ in range(2):
        with pytest.raises(AdmissionRejected) as exc:
            executor.submit(lambda: None)
        assert exc.value.reason == "overloaded"
        rejected += 1
    snap = executor.snapshot()
    assert (snap.running, snap.queued, rejected) == (1, 2, 2)
    release.set()
    assert await handles[0].result() == "ok"
    assert await handles[1].result() == "a"
    assert await handles[2].result() == "b"
    executor.begin_drain()
    assert await executor.wait_drained(1)
    executor.snapshot()
    executor.shutdown()


@pytest.mark.anyio
async def test_fifo_cancel_head_promotes_once(m42_receipt):
    release = threading.Event()
    started = threading.Event()
    order = []
    executor = QueryExecutor(concurrency_limit=1, queue_limit=2, queue_timeout=5,
                             execution_timeout=5)
    before = executor.snapshot()

    def running():
        order.append("running")
        started.set()
        release.wait()

    first = executor.submit(running)
    await asyncio.to_thread(started.wait)
    a = executor.submit(lambda: order.append("A"))
    b = executor.submit(lambda: order.append("B"))
    a.cancel()
    release.set()
    await first.result()
    await b.result()
    assert order == ["running", "B"]
    assert executor.snapshot().orphaned == 0
    executor.snapshot()
    executor.shutdown()


@pytest.mark.anyio
async def test_queue_head_timeout_single_promotion(m42_receipt):
    release = threading.Event()
    executor = QueryExecutor(concurrency_limit=1, queue_limit=2, queue_timeout=.01,
                             execution_timeout=2)
    before = executor.snapshot()
    first = executor.submit(lambda: release.wait())
    queued = executor.submit(lambda: "next")
    with pytest.raises(TimeoutError):
        await queued.result()
    release.set()
    await first.result()
    assert executor.snapshot().queue_timeout_total == 1
    executor.snapshot()
    executor.shutdown()


@pytest.mark.anyio
async def test_abandoned_holds_slot_until_future_done(m42_receipt):
    release = threading.Event()
    executor = QueryExecutor(concurrency_limit=1, queue_limit=0, queue_timeout=1,
                             execution_timeout=.01)
    before = executor.snapshot()
    handle = executor.submit(lambda: release.wait())
    with pytest.raises(TimeoutError):
        await handle.result()
    assert (executor.snapshot().running, executor.snapshot().orphaned) == (1, 1)
    release.set()
    assert await executor.wait_drained(2)
    assert (executor.snapshot().running, executor.snapshot().orphaned) == (0, 0)
    executor.snapshot()
    executor.shutdown()


def test_terminal_conservation_all_origins():
    assert QueryExecutor.snapshot.__annotations__["return"] == "ExecutorSnapshot"


@pytest.mark.anyio
@pytest.mark.parametrize("failed_promotions", [1, 2])
async def test_promotion_submit_failures_consume_fifo_heads_and_keep_capacity_busy(failed_promotions):
    release = threading.Event()
    started = threading.Event()
    trace = []
    executor = QueryExecutor(concurrency_limit=1, queue_limit=3, queue_timeout=5,
                             execution_timeout=5)

    def running():
        started.set()
        release.wait()
        trace.append("running")

    first = executor.submit(running)
    await asyncio.to_thread(started.wait)
    queued = [executor.submit(lambda value=value: trace.append(value)) for value in ("A", "B", "C")]
    original_submit = executor._pool.submit
    remaining = failed_promotions

    def adversarial_submit(*args, **kwargs):
        nonlocal remaining
        if remaining:
            remaining -= 1
            raise RuntimeError("injected promotion failure")
        return original_submit(*args, **kwargs)

    executor._pool.submit = adversarial_submit
    release.set()
    await first.result()
    for handle in queued[:failed_promotions]:
        with pytest.raises(SubmitFailed):
            await handle.result()
    for handle in queued[failed_promotions:]:
        await handle.result()
    snap = executor.snapshot()
    assert trace == ["running", *("A", "B", "C")[failed_promotions:]]
    assert (snap.running, snap.queued, snap.orphaned) == (0, 0, 0)
    assert snap.accepted_total == snap.completed_total + snap.terminal_rejected_total
    assert not executor._tickets
    assert all(ticket.queue_timer is None or ticket.queue_timer.cancelled()
               for ticket in executor._tickets.values())
    executor.shutdown()


@pytest.mark.anyio
async def test_terminal_conservation_scenario_matrix():
    scenarios = []

    completed = QueryExecutor(concurrency_limit=1, queue_limit=0, queue_timeout=1,
                              execution_timeout=1)
    before = completed.snapshot()
    assert await completed.submit(lambda: "ok").result() == "ok"
    scenarios.append(("completed", before, completed.snapshot(), 1, 0))
    completed.shutdown()

    cancelled = QueryExecutor(concurrency_limit=1, queue_limit=1, queue_timeout=1,
                              execution_timeout=1)
    release = threading.Event()
    running = cancelled.submit(lambda: release.wait())
    queued = cancelled.submit(lambda: None)
    queued.cancel()
    release.set()
    await running.result()
    baseline = QueryExecutor(concurrency_limit=1, queue_limit=0,
                             queue_timeout=1, execution_timeout=1)
    scenarios.append(("queued_cancel", baseline.snapshot(), cancelled.snapshot(), 2, 0))
    baseline.shutdown()
    cancelled.shutdown()

    running_cancel = QueryExecutor(concurrency_limit=1, queue_limit=0, queue_timeout=1,
                                   execution_timeout=1)
    release = threading.Event()
    handle = running_cancel.submit(lambda: release.wait())
    handle.cancel(); release.set()
    while running_cancel.snapshot().running:
        await asyncio.sleep(0)
    baseline = QueryExecutor(concurrency_limit=1, queue_limit=0, queue_timeout=1, execution_timeout=1)
    scenarios.append(("running_cancel", baseline.snapshot(), running_cancel.snapshot(), 1, 0))
    baseline.shutdown(); running_cancel.shutdown()

    queue_timeout = QueryExecutor(concurrency_limit=1, queue_limit=1, queue_timeout=.01,
                                  execution_timeout=1)
    release = threading.Event()
    first = queue_timeout.submit(lambda: release.wait())
    timed = queue_timeout.submit(lambda: None)
    with pytest.raises(TimeoutError):
        await timed.result()
    release.set()
    await first.result()
    baseline = QueryExecutor(concurrency_limit=1, queue_limit=0, queue_timeout=1, execution_timeout=1)
    scenarios.append(("queue_timeout", baseline.snapshot(), queue_timeout.snapshot(), 2, 0))
    baseline.shutdown(); queue_timeout.shutdown()

    execution_timeout = QueryExecutor(concurrency_limit=1, queue_limit=0, queue_timeout=1,
                                      execution_timeout=.01)
    release = threading.Event()
    timed = execution_timeout.submit(lambda: release.wait())
    with pytest.raises(TimeoutError):
        await timed.result()
    release.set()
    while execution_timeout.snapshot().running:
        await asyncio.sleep(0)
    baseline = QueryExecutor(concurrency_limit=1, queue_limit=0, queue_timeout=1, execution_timeout=1)
    scenarios.append(("execution_timeout", baseline.snapshot(), execution_timeout.snapshot(), 1, 0))
    baseline.shutdown(); execution_timeout.shutdown()

    drained = QueryExecutor(concurrency_limit=1, queue_limit=1, queue_timeout=1, execution_timeout=1)
    release = threading.Event()
    running = drained.submit(lambda: release.wait())
    rejected = drained.submit(lambda: None)
    drained.begin_drain()
    with pytest.raises(AdmissionRejected):
        await rejected.result()
    with pytest.raises(AdmissionRejected):
        drained.submit(lambda: None)
    release.set(); await running.result()
    baseline = QueryExecutor(concurrency_limit=1, queue_limit=0, queue_timeout=1, execution_timeout=1)
    scenarios.append(("drain_rejection", baseline.snapshot(), drained.snapshot(), 2, 1))
    baseline.shutdown(); drained.shutdown()

    submit_failed = QueryExecutor(concurrency_limit=1, queue_limit=0, queue_timeout=1,
                                  execution_timeout=1)
    before = submit_failed.snapshot()
    submit_failed._pool.shutdown()
    with pytest.raises(SubmitFailed):
        submit_failed.submit(lambda: None)
    scenarios.append(("submit_failure", before, submit_failed.snapshot(), 0, 1))
    submit_failed.shutdown()

    overloaded = QueryExecutor(concurrency_limit=1, queue_limit=0, queue_timeout=1,
                               execution_timeout=1)
    release = threading.Event()
    running = overloaded.submit(lambda: release.wait())
    with pytest.raises(AdmissionRejected):
        overloaded.submit(lambda: None)
    release.set(); await running.result()
    baseline = QueryExecutor(concurrency_limit=1, queue_limit=0, queue_timeout=1, execution_timeout=1)
    scenarios.append(("admission_rejection", baseline.snapshot(), overloaded.snapshot(), 1, 1))
    baseline.shutdown(); overloaded.shutdown()

    assert {name for name, *_ in scenarios} == {
        "completed", "queued_cancel", "running_cancel", "queue_timeout",
        "execution_timeout", "drain_rejection", "submit_failure", "admission_rejection",
    }
    for name, before, after, expected_accepted, expected_admission in scenarios:
        accepted = after.accepted_total - before.accepted_total
        terminals = sum((after.completed_total - before.completed_total,
                         after.queue_timeout_total - before.queue_timeout_total,
                         after.execution_timeout_total - before.execution_timeout_total,
                         after.cancelled_total - before.cancelled_total,
                         after.terminal_rejected_total - before.terminal_rejected_total))
        assert accepted == after.queued + after.running - after.orphaned + terminals
        admission = after.admission_rejected_total - before.admission_rejected_total
        assert (accepted, admission) == (expected_accepted, expected_admission), name
        assert accepted + admission == expected_accepted + expected_admission, name
