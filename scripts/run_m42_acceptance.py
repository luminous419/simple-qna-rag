#!/usr/bin/env python3
"""Deterministic M4.2 acceptance inventory and repeat runner."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Sequence

from simple_qna_rag.web.concurrency import ExecutorSnapshot

PROFILE_NODE_IDS = MappingProxyType({
    "event_loop": ("tests/integration/test_web_concurrency.py::test_event_loop_remains_responsive",),
    "bounded_admission": ("tests/unit/test_query_executor.py::test_profile_1_2_5_exact_receipt",),
    "fifo_cancel": ("tests/unit/test_query_executor.py::test_fifo_cancel_head_promotes_once",),
    "queue_timeout": ("tests/unit/test_query_executor.py::test_queue_head_timeout_single_promotion",),
    "execution_timeout": ("tests/unit/test_query_executor.py::test_abandoned_holds_slot_until_future_done",),
    "caller_cancellation": (
        "tests/integration/test_web_disconnect.py::test_asgi_disconnect_queued_100_races",
        "tests/integration/test_web_disconnect.py::test_asgi_disconnect_running_100_races",
    ),
    "drain": ("tests/unit/test_shutdown_drain.py::test_running_queued_and_grace_expiry_profiles",),
    "saturation_readiness": ("tests/unit/test_readiness_saturation.py::test_edge_timestamp_debounce_with_intermediate_clear",),
    "payload": ("tests/integration/test_web_input_boundary.py::test_body_profiles_stop_at_limit_plus_one",),
    "normal_mock_load": ("tests/evaluation/test_m4_safe_serving_load.py::test_single_thread_40",),
})
REQUEST_TERMINALS = (
    "success", "invalid_request", "payload_too_large", "not_ready", "overloaded",
    "queue_timeout", "execution_timeout", "internal", "client_disconnected",
)


def collect_profile_nodes(profiles: Sequence[str]) -> tuple[str, ...]:
    nodes = tuple(node for profile in profiles for node in PROFILE_NODE_IDS[profile])
    if len(nodes) != len(set(nodes)):
        raise ValueError("duplicate profile node")
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *nodes],
        text=True, capture_output=True,
    )
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)
    collected = tuple(line.strip() for line in run.stdout.splitlines() if "::test_" in line)
    if collected != nodes:
        raise ValueError(f"profile inventory mismatch: {collected!r}")
    return nodes


def validate_conservation(values: dict[str, int]) -> dict[str, int]:
    checks = {
        "request_count": values["request_count"],
        "request_terminal_sum": sum(values.get(name, 0) for name in REQUEST_TERMINALS),
        "accepted_lhs": values["accepted"],
        "accepted_rhs": sum(values[name] for name in (
            "queued", "running_non_orphaned", "completed", "queue_timeout",
            "execution_timeout", "cancelled", "terminal_rejected",
        )),
        "submit_attempt_lhs": values["submit_attempts"],
        "submit_attempt_rhs": values["accepted"] + values["admission_not_ready"] + values["admission_overloaded"] + values["admission_submit_failed"],
        "unknown": values.get("unknown", 0),
    }
    if (checks["request_count"] != checks["request_terminal_sum"] or
            checks["accepted_lhs"] != checks["accepted_rhs"] or
            checks["submit_attempt_lhs"] != checks["submit_attempt_rhs"] or checks["unknown"]):
        raise ValueError("conservation_mismatch")
    return checks


def _delta(before: dict, after: dict, name: str) -> int:
    value = after[name] - before[name]
    if type(value) is not int or value < 0:
        raise ValueError(f"malformed_snapshot_delta:{name}")
    return value


def _parse_node_receipt(document: object, expected_node: str) -> dict[str, int]:
    outer_keys = {"schema", "node_id", "ledger_epoch", "node_token", "rows"}
    if not isinstance(document, dict) or set(document) != outer_keys:
        raise ValueError("malformed_runtime_receipt_document")
    if (document["schema"] != "m42-node-runtime-receipt-v1"
            or document["node_id"] != expected_node
            or type(document["ledger_epoch"]) is not int or document["ledger_epoch"] < 0
            or not isinstance(document["node_token"], str) or not document["node_token"]):
        raise ValueError("malformed_runtime_receipt_identity")
    rows = document.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("absent_or_duplicate_runtime_receipt")
    row = rows[0]
    row_keys = {
        "node_id", "ledger_epoch", "node_token", "executor_id", "snapshot_epoch",
        "executor_identity_conflicts",
        "before", "after", "request_terminals", "rag_terminals", "executor_terminals",
        "stale_terminals",
    }
    if (not isinstance(row, dict) or set(row) != row_keys or row.get("node_id") != expected_node
            or type(row.get("ledger_epoch")) is not int
            or row["ledger_epoch"] != document["ledger_epoch"]
            or row["node_token"] != document["node_token"]
            or not isinstance(row.get("before"), dict) or not isinstance(row.get("after"), dict)):
        raise ValueError("malformed_runtime_receipt")
    source_maps = {}
    for source in ("request", "rag", "executor"):
        terminals = row[f"{source}_terminals"]
        if (not isinstance(terminals, dict) or set(terminals) != set(REQUEST_TERMINALS)
                or any(type(value) is not int or value < 0 for value in terminals.values())):
            raise ValueError(f"malformed_{source}_terminals")
        source_maps[source] = terminals
    stale = row["stale_terminals"]
    if (not isinstance(stale, dict) or set(stale) != {"request", "rag", "executor"}
            or any(not isinstance(values, dict) or set(values) != set(REQUEST_TERMINALS)
                   or any(type(value) is not int or value < 0 for value in values.values())
                   for values in stale.values())):
        raise ValueError("malformed_stale_terminals")
    if any(value for values in stale.values() for value in values.values()):
        raise ValueError("stale_publication")
    before, after = row["before"], row["after"]
    snapshot_fields = set(ExecutorSnapshot.__dataclass_fields__)
    if set(before) != snapshot_fields or set(after) != snapshot_fields:
        raise ValueError("malformed_snapshot_fields")
    int_fields = snapshot_fields - {"lifecycle", "capacity_full"}
    for snapshot in (before, after):
        if snapshot["lifecycle"] not in {"READY", "DRAINING", "STOPPED"} or type(snapshot["capacity_full"]) is not bool:
            raise ValueError("malformed_snapshot_types")
        if any((snapshot[name] is not None and
                (type(snapshot[name]) is not int or snapshot[name] < 0)) for name in {
                    "stopped_with_running", "stopped_with_orphaned"}):
            raise ValueError("malformed_snapshot_types")
        if type(snapshot["capacity_edge_at"]) not in {int, float} or not math.isfinite(snapshot["capacity_edge_at"]):
            raise ValueError("malformed_snapshot_types")
        required_ints = int_fields - {"capacity_edge_at", "stopped_with_running", "stopped_with_orphaned"}
        if any(type(snapshot[name]) is not int or snapshot[name] < 0 for name in required_ints):
            raise ValueError("malformed_snapshot_types")
        if snapshot["running"] < snapshot["orphaned"]:
            raise ValueError("snapshot_invariant_mismatch")
    executor_count = sum(source_maps["executor"].values())
    if (type(row["executor_id"]) is not str or not row["executor_id"]
            or type(row["snapshot_epoch"]) is not int
            or row["snapshot_epoch"] != row["ledger_epoch"]
            or type(row["executor_identity_conflicts"]) is not int
            or row["executor_identity_conflicts"] != 0):
        raise ValueError("snapshot_identity_mismatch")
    request, rag, executor = (source_maps[name] for name in ("request", "rag", "executor"))
    if request != rag:
        raise ValueError("request_rag_terminal_mismatch")
    terminals = request if sum(request.values()) else executor
    values = {name: terminals[name] for name in REQUEST_TERMINALS}
    values.update({
        "request_count": sum(terminals.values()),
        "accepted": _delta(before, after, "accepted_total"),
        "queued": after["queued"],
        "running_non_orphaned": after["running"] - after["orphaned"],
        "completed": _delta(before, after, "completed_total"),
        "queue_timeout": _delta(before, after, "queue_timeout_total"),
        "execution_timeout": _delta(before, after, "execution_timeout_total"),
        "cancelled": _delta(before, after, "cancelled_total"),
        "terminal_rejected": _delta(before, after, "terminal_rejected_total"),
        "admission_not_ready": 0, "admission_overloaded": 0,
        "admission_submit_failed": 0, "unknown": 0,
    })
    admission = _delta(before, after, "admission_rejected_total")
    values["admission_not_ready"] = admission
    values["submit_attempts"] = values["accepted"] + admission
    # Application terminals are a positional subset of executor terminals;
    # their exact per-reason remainder is the separately accounted direct
    # control-ticket stream. Pre-executor input rejections are the only
    # application reasons that intentionally have no executor counterpart.
    executor_backed = set(REQUEST_TERMINALS) - {"invalid_request", "payload_too_large"}
    consumed = {name: request[name] for name in executor_backed}
    # A disconnect can win the application response race after the worker has
    # already completed successfully. Executor-side cancellation is therefore
    # consumed first and the remaining disconnects must be proven by success.
    executor_disconnects_for_application = min(
        request["client_disconnected"], executor["client_disconnected"]
    )
    disconnected_after_success = (
        request["client_disconnected"] - executor_disconnects_for_application
    )
    consumed["client_disconnected"] = executor_disconnects_for_application
    consumed["success"] += disconnected_after_success
    direct = {name: executor[name] - consumed[name] for name in executor_backed}
    executor_expected_total = values["accepted"] + admission
    executor_reason_algebra = (
        executor["invalid_request"] == executor["payload_too_large"] == 0
        and all(value >= 0 for value in direct.values())
        and executor["queue_timeout"] == values["queue_timeout"]
        and executor["execution_timeout"] == values["execution_timeout"]
        and executor["client_disconnected"] == values["cancelled"]
        and executor["success"] + executor["internal"]
            == values["completed"] + values["terminal_rejected"]
               + admission - executor["not_ready"] - executor["overloaded"]
        and executor["not_ready"] + executor["overloaded"]
            <= admission + values["terminal_rejected"]
        and sum(executor.values()) == executor_expected_total
    )
    if not executor_reason_algebra:
        raise ValueError("request_executor_terminal_mismatch")
    if values["request_count"] <= 0:
        raise ValueError("zero_work_runtime_receipt")
    if values["accepted"] <= 0 and not any(values[name] for name in (
            "invalid_request", "payload_too_large", "not_ready", "overloaded", "internal")):
        raise ValueError("zero_executor_work_runtime_receipt")
    validate_conservation(values)
    return values


def parse_node_receipt(document: object, expected_node: str) -> dict[str, int]:
    """Normalize every schema/parser/validator failure to one fail-closed type."""
    try:
        return _parse_node_receipt(document, expected_node)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"malformed_runtime_receipt:{type(exc).__name__}") from exc


def harvest_node_receipt(path: Path, expected_node: str) -> dict[str, int]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite:{value}")),
        )
        return parse_node_receipt(document, expected_node)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("absent_or_malformed_runtime_receipt") from exc


def _sum_values(rows: Sequence[dict[str, int]]) -> dict[str, int]:
    keys = set().union(*(row.keys() for row in rows))
    return {key: sum(row.get(key, 0) for row in rows) for key in keys}


def _receipt_is_complete(receipt: dict) -> bool:
    expected_profiles = 10 * len(PROFILE_NODE_IDS)
    expected_nodes = 10 * sum(len(nodes) for nodes in PROFILE_NODE_IDS.values())
    rows = receipt.get("profile_conservation", [])
    return (
        len(receipt.get("results", [])) == expected_profiles
        and len(receipt.get("node_results", [])) == expected_nodes
        and len(rows) == expected_profiles
        and len(receipt.get("conservation", [])) == 10
        and all(row.get("request_count", 0) > 0 and row.get("unknown") == 0
                for row in rows)
        and all(row.get("exit_code") == 0 for row in receipt.get("results", []))
        and all(row.get("exit_code") == 0 for row in receipt.get("node_results", []))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("deterministic", "live"), default="deterministic")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4202)
    parser.add_argument("--manifest")
    parser.add_argument("--output", default="evaluation/reports/m42_acceptance.json")
    parser.add_argument("--inject-conservation-mismatch", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.profile == "live":
        receipt = {"status": "NOT_RUN", "live": {"reason": "opt-in environment required"},
                   "m3_regression": {"status": "NOT_RUN"}, "M4.1_BLOCKED": True}
        output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return 0
    if args.repeat != 10:
        return 1
    try:
        collected_nodes = collect_profile_nodes(tuple(PROFILE_NODE_IDS))
    except (RuntimeError, ValueError) as exc:
        output.write_text(json.dumps({"status": "FAIL", "diagnostic": str(exc)}, indent=2), encoding="utf-8")
        return 1
    results = []
    node_results = []
    conservation = []
    profile_conservation = []
    status = "PASS"
    first_genuine_document: dict | None = None
    repeat_values: dict[int, list[dict[str, int]]] = {index: [] for index in range(args.repeat)}
    for repeat_index in range(args.repeat):
        for profile, nodes in PROFILE_NODE_IDS.items():
            observed = []
            node_exit_codes = []
            for node in nodes:
                command = [sys.executable, "-m", "pytest", "-q", node]
                with tempfile.TemporaryDirectory(prefix="m42-receipt-") as temp_dir:
                    receipt_path = Path(temp_dir) / "node.json"
                    env = dict(os.environ, M42_RECEIPT_PATH=str(receipt_path))
                    run = subprocess.run(command, text=True, capture_output=True, env=env)
                    document = None
                    parse_diagnostic = None
                    try:
                        if run.returncode == 0:
                            document = json.loads(receipt_path.read_text(encoding="utf-8"))
                            values = parse_node_receipt(document, node)
                            if first_genuine_document is None:
                                first_genuine_document = copy.deepcopy(document)
                        else:
                            values = None
                    except Exception as exc:
                        values = None
                        parse_diagnostic = str(exc)
                        run = subprocess.CompletedProcess(command, 1, run.stdout, run.stderr + str(exc))
                    node_exit_codes.append(run.returncode)
                    node_results.append({"node_id": node, "profile": profile,
                                         "repeat_index": repeat_index, "exit_code": run.returncode,
                                         "runtime_receipt": document,
                                         "diagnostic": parse_diagnostic})
                    if values is not None:
                        observed.append(values)
            profile_exit = max(node_exit_codes, default=1)
            results.append({"profile": profile, "repeat_index": repeat_index,
                            "nodes": list(nodes), "exit_code": profile_exit})
            if profile_exit == 0 and len(observed) == len(nodes):
                values = _sum_values(observed)
                try:
                    checked = validate_conservation(values)
                except ValueError as exc:
                    checked = {"unknown": 1, "mismatch": str(exc)}
                    status = "FAIL"
                profile_conservation.append({
                    "profile": profile, "repeat_index": repeat_index,
                    **checked,
                })
                repeat_values[repeat_index].append(values)
            if (profile_exit or len(observed) != len(nodes)
                    or (profile_conservation and "mismatch" in profile_conservation[-1])):
                status = "FAIL"
                break
        else:
            continue
        break
    else:
        status = "PASS"
    for repeat_index in range(args.repeat):
        rows = repeat_values[repeat_index]
        if rows:
            values = _sum_values(rows)
        else:
            values = {name: 0 for name in REQUEST_TERMINALS}
            values.update({name: 0 for name in (
                "request_count", "accepted", "queued", "running_non_orphaned",
                "completed", "queue_timeout", "execution_timeout", "cancelled",
                "terminal_rejected", "submit_attempts", "admission_not_ready",
                "admission_overloaded", "admission_submit_failed",
            )})
            values["unknown"] = 1
        try:
            row = validate_conservation(values)
        except ValueError as exc:
            status = "FAIL"
            row = {"unknown": 1, "mismatch": str(exc)}
        conservation.append({"repeat_index": repeat_index, **row})
    receipt = {"status": status, "seed": args.seed, "python": platform.python_version(),
               "platform": platform.platform(), "profiles": list(PROFILE_NODE_IDS),
               "collected_nodes": list(collected_nodes), "results": results,
               "node_results": node_results, "profile_conservation": profile_conservation,
               "conservation": conservation,
               "negative_control": {"status": "NOT_RUN", "kind": "corrupted_genuine_runtime_receipt"},
               "M4.1_BLOCKED": True}
    if args.inject_conservation_mismatch:
        rejected = copy.deepcopy(first_genuine_document)
        diagnostic = "missing_genuine_runtime_receipt"
        if rejected is not None:
            rejected["rows"][0]["after"]["queued"] += 1
            try:
                parse_node_receipt(rejected, rejected["node_id"])
                diagnostic = "corrupted_genuine_row_accepted"
            except Exception as exc:
                diagnostic = str(exc)
        receipt["negative_control"] = {
            "status": "FAIL", "kind": "corrupted_genuine_runtime_receipt",
            "rejected_receipt": rejected, "diagnostic": diagnostic,
        }
        status = receipt["status"] = "FAIL"
    if status == "PASS" and not _receipt_is_complete(receipt):
        status = receipt["status"] = "FAIL"
        receipt["diagnostic"] = "missing_zero_or_unproven_receipt"
    encoded = json.dumps(receipt, sort_keys=True).encode()
    receipt["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
