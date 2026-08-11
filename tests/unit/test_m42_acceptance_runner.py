import json
from dataclasses import asdict

import pytest

from scripts.run_m42_acceptance import (
    PROFILE_NODE_IDS, _receipt_is_complete, parse_node_receipt, main,
)
from simple_qna_rag.web.concurrency import ExecutorSnapshot
from simple_qna_rag.observability.terminal_ledger import TerminalLedger


def _document():
    before = ExecutorSnapshot("READY", 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                              False, 1.0, 0, None, None)
    after = ExecutorSnapshot("READY", 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                             False, 2.0, 0, None, None)
    terminals = {name: 0 for name in (
        "success", "invalid_request", "payload_too_large", "not_ready", "overloaded",
        "queue_timeout", "execution_timeout", "internal", "client_disconnected",
    )}
    terminals["success"] = 1
    return {"schema": "m42-node-runtime-receipt-v1", "node_id": "node",
            "ledger_epoch": 1, "node_token": "token", "rows": [{
        "node_id": "node", "ledger_epoch": 1, "node_token": "token",
        "executor_id": "executor", "snapshot_epoch": 1,
        "executor_identity_conflicts": 0,
        "before": asdict(before), "after": asdict(after),
        "request_terminals": terminals, "rag_terminals": dict(terminals),
        "executor_terminals": dict(terminals),
        "stale_terminals": {source: {name: 0 for name in terminals}
                            for source in ("request", "rag", "executor")},
    }]}


def test_profile_node_inventory_exact():
    assert tuple(PROFILE_NODE_IDS) == (
        "event_loop", "bounded_admission", "fifo_cancel", "queue_timeout",
        "execution_timeout", "caller_cancellation", "drain", "saturation_readiness",
        "payload", "normal_mock_load",
    )
    flattened = [node for nodes in PROFILE_NODE_IDS.values() for node in nodes]
    assert len(flattened) == len(set(flattened)) == 11


def test_negative_conservation_mismatch_exits_nonzero(tmp_path):
    output = tmp_path / "negative.json"
    assert main(["--repeat", "10", "--output", str(output),
                 "--inject-conservation-mismatch"]) == 1
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "FAIL"
    assert receipt["negative_control"]["status"] == "FAIL"
    assert receipt["negative_control"]["diagnostic"] == "conservation_mismatch"
    assert receipt["negative_control"]["rejected_receipt"]["rows"][0]["node_id"]


def test_acceptance_fails_closed_on_missing_or_zero_receipts():
    assert not _receipt_is_complete({})
    assert not _receipt_is_complete({
        "results": [{"exit_code": 0}] * 100,
        "node_results": [{"exit_code": 0}] * 110,
        "profile_conservation": [{"request_count": 0, "accepted_lhs": 0,
                                  "submit_attempt_lhs": 0, "unknown": 0}] * 100,
        "conservation": [{}] * 10,
    })


@pytest.mark.parametrize("mutation", [
    lambda document: document.update(rows=[]),
    lambda document: document.update(rows=document["rows"] * 2),
    lambda document: document["rows"][0]["after"].pop("completed_total"),
    lambda document: document["rows"][0]["after"].update(completed_total="1"),
    lambda document: document["rows"][0]["request_terminals"].update(success=0),
    lambda document: document["rows"][0]["after"].update(running=0, orphaned=1),
    lambda document: document["rows"][0].update(snapshot_epoch=2),
    lambda document: document["rows"][0].update(ledger_epoch=True),
    lambda document: document["rows"][0].update(snapshot_epoch=True),
    lambda document: document["rows"][0]["after"].update(stopped_with_running=-1),
    lambda document: document["rows"][0]["after"].update(stopped_with_orphaned=-1),
    lambda document: document["rows"][0].update(executor_identity_conflicts=1),
    lambda document: document["rows"][0]["after"].update(capacity_edge_at=float("nan")),
    lambda document: document["rows"][0]["after"].update(capacity_edge_at=float("inf")),
    lambda document: document["rows"][0]["after"].update(capacity_edge_at=float("-inf")),
    lambda document: document["rows"][0]["after"].update(accepted_total=True),
    lambda document: document["rows"][0]["executor_terminals"].update(success=0, queue_timeout=1),
])
def test_typed_receipt_parser_fails_closed(mutation):
    document = _document()
    mutation(document)
    with pytest.raises(ValueError):
        parse_node_receipt(document, "node")


def test_receipt_parser_retains_reason_mismatch_diagnostic():
    document = _document()
    document["rows"][0]["executor_terminals"].update(success=0, queue_timeout=1)
    with pytest.raises(ValueError, match="request_executor_terminal_mismatch"):
        parse_node_receipt(document, "node")


def test_receipt_parser_rejects_request_success_vs_executor_internal():
    document = _document()
    document["rows"][0]["executor_terminals"].update(success=0, internal=1)
    with pytest.raises(ValueError, match="request_executor_terminal_mismatch"):
        parse_node_receipt(document, "node")


def test_terminal_ledger_old_callback_after_reset_is_stale():
    ledger = TerminalLedger()
    old = ledger.reset("old")
    new = ledger.reset("new")
    assert not ledger.record(old, "executor", "success")
    assert ledger.record(new, "executor", "internal")
    receipt = ledger.snapshot()
    assert receipt.executor_terminals["success"] == 0
    assert receipt.executor_terminals["internal"] == 1
    assert receipt.stale_terminals["executor"]["success"] == 1


def test_terminal_ledger_overlapping_executor_identity_is_rejected():
    ledger = TerminalLedger()
    producer = ledger.reset("node")
    snapshot = ExecutorSnapshot("READY", 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                False, 1.0, 0, None, None)
    assert ledger.observe_executor_snapshot(producer, "first", snapshot)
    assert not ledger.observe_executor_snapshot(producer, "second", snapshot)
    receipt = ledger.snapshot()
    assert receipt.executor_id == "first"
    assert receipt.snapshot_epoch == producer.epoch
    assert receipt.executor_identity_conflicts == 1


def test_zero_work_second_executor_overlap_is_retained_and_rejected():
    ledger = TerminalLedger()
    producer = ledger.reset("node")
    snapshot = ExecutorSnapshot("READY", 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                False, 1.0, 0, None, None)
    assert ledger.observe_executor_snapshot(producer, "first", snapshot)
    assert not ledger.observe_executor_snapshot(producer, "second", snapshot)
    receipt = ledger.snapshot()
    document = _document()
    document["rows"][0]["executor_identity_conflicts"] = receipt.executor_identity_conflicts
    with pytest.raises(ValueError, match="snapshot_identity_mismatch"):
        parse_node_receipt(document, "node")
