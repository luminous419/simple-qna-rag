import asyncio
import threading

import pytest

from simple_qna_rag.web.concurrency import QueryExecutor


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_event_loop_remains_responsive(m42_receipt):
    release = threading.Event()
    started = threading.Event()
    executor = QueryExecutor(concurrency_limit=1, queue_limit=4, queue_timeout=2,
                             execution_timeout=2)
    before = executor.snapshot()

    def work():
        started.set()
        release.wait()

    handle = executor.submit(work)
    await asyncio.to_thread(started.wait)
    for _ in range(20):
        await asyncio.sleep(0)
    release.set()
    await handle.result()
    executor.snapshot()
    executor.shutdown()


@pytest.mark.anyio
async def test_normal_mock_load_40_single_worker():
    executor = QueryExecutor(concurrency_limit=1, queue_limit=64, queue_timeout=5,
                             execution_timeout=5)
    handles = [executor.submit(lambda value=i: value) for i in range(40)]
    assert await asyncio.gather(*(handle.result() for handle in handles)) == list(range(40))
    assert executor.snapshot().completed_total == 40
    executor.shutdown()
