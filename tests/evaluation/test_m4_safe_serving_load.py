import asyncio
import threading

import pytest

from simple_qna_rag.web.concurrency import QueryExecutor


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_single_thread_40(m42_receipt):
    executor = QueryExecutor(concurrency_limit=1, queue_limit=40, queue_timeout=5,
                             execution_timeout=5)
    before = executor.snapshot()
    trace: list[str] = []
    lock = threading.Lock()

    def service(request_id: str):
        with lock:
            trace.append(request_id)
        return request_id

    handles = [executor.submit(lambda rid=f"{i:02d}": service(rid)) for i in range(40)]
    assert await asyncio.gather(*(handle.result() for handle in handles)) == [f"{i:02d}" for i in range(40)]
    assert trace == [f"{i:02d}" for i in range(40)]
    snap = executor.snapshot()
    assert (snap.running, snap.queued, snap.completed_total) == (0, 0, 40)
    executor.shutdown()
