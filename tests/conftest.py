"""Shared deterministic acceptance receipt fixture."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def m42_receipt(request):
    from simple_qna_rag.observability.terminal_ledger import terminal_ledger

    binding = terminal_ledger.reset(request.node.nodeid)
    yield
    target = os.environ.get("M42_RECEIPT_PATH")
    if target:
        receipt = terminal_ledger.snapshot()
        Path(target).write_text(json.dumps({
            "schema": "m42-node-runtime-receipt-v1",
            "node_id": request.node.nodeid,
            "ledger_epoch": binding.epoch,
            "node_token": binding.node_token,
            "rows": [{
                "node_id": receipt.node_id,
                "ledger_epoch": receipt.epoch,
                "node_token": receipt.node_token,
                "executor_id": receipt.executor_id,
                "snapshot_epoch": receipt.snapshot_epoch,
                "executor_identity_conflicts": receipt.executor_identity_conflicts,
                "before": dict(receipt.before) if receipt.before is not None else None,
                "after": dict(receipt.after) if receipt.after is not None else None,
                "request_terminals": dict(receipt.request_terminals),
                "rag_terminals": dict(receipt.rag_terminals),
                "executor_terminals": dict(receipt.executor_terminals),
                "stale_terminals": {
                    source: dict(values) for source, values in receipt.stale_terminals.items()
                },
            }],
        }, sort_keys=True), encoding="utf-8")
