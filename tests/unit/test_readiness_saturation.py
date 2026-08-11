import asyncio

import pytest

from simple_qna_rag.observability.health import SaturationDebounce
from simple_qna_rag.web.concurrency import QueryExecutor


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_edge_timestamp_debounce_with_intermediate_clear(m42_receipt):
    now = [0.0]
    debounce = SaturationDebounce(lambda: now[0])
    assert not debounce.evaluate(full=True, edge_at=0.0, version=1)
    now[0] = 0.4
    assert not debounce.evaluate(full=False, edge_at=0.4, version=2)
    now[0] = 0.6
    assert not debounce.evaluate(full=True, edge_at=0.6, version=3)
    now[0] = 1.0
    assert not debounce.evaluate(full=True, edge_at=0.6, version=3)
    now[0] = 1.6
    assert debounce.evaluate(full=True, edge_at=0.6, version=3)
    now[0] = 1.7
    assert debounce.evaluate(full=False, edge_at=1.7, version=4)
    now[0] = 2.7
    assert not debounce.evaluate(full=False, edge_at=1.7, version=4)
    executor = QueryExecutor(concurrency_limit=1, queue_limit=0, queue_timeout=1,
                             execution_timeout=1)
    before = executor.snapshot()
    assert await executor.submit(lambda: "probe").result() == "probe"
    executor.snapshot()
    executor.shutdown()
