"""M4.3-REQ-007 — assembler exact-schema/payload-verification negative matrix
(Design.md §8.2, DR-I5-MAJ-01, and DR-I6-MIN-01's non-string semantic_status
guard)."""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import re
import subprocess
import tokenize
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import assemble_m4_evidence as assembler
from scripts import run_m43_acceptance

EXPECTED = SimpleNamespace(
    expected_sha="a" * 40, expected_run_id="1001", expected_run_attempt="1",
    expected_workflow_path=".github/workflows/ci.yml", expected_event="pull_request",
)


def _m43_payload(*, negative: bool) -> dict:
    nodes = {name: {"repeat": 10, "success_count": 10, "status": "PASS"}
             for name in assembler.EXPECTED_M43_NODE_IDS}
    if negative:
        negative_control = {"executed": True, "expected_to_fail": True,
                             "actual_exit_code": 1, "result": "REJECTED_AS_EXPECTED"}
        status = "REJECTED_AS_EXPECTED"
    else:
        negative_control = {"executed": False, "expected_to_fail": None,
                             "actual_exit_code": None, "result": None}
        status = "PASS"
    return {
        "schema": assembler.M43_SCHEMA, "profile": "deterministic",
        "seed": assembler.M43_SEED, "repeat": assembler.M43_REPEAT,
        "command": assembler.M43_EXPECTED_COMMAND,
        "started_at": "2026-08-12T00:00:00Z", "finished_at": "2026-08-12T00:00:01Z",
        "nodes": nodes, "negative_control": negative_control, "status": status,
    }


def _write_payload_file(payload_dir: Path, filename: str, content_bytes: bytes) -> dict:
    payload_dir.mkdir(parents=True, exist_ok=True)
    (payload_dir / filename).write_bytes(content_bytes)
    return {"filename": filename, "sha256": hashlib.sha256(content_bytes).hexdigest(),
            "size_bytes": len(content_bytes)}


def _write_receipt(fresh_dir: Path, job: str, *, payload_entries: list[dict],
                    semantic_status="PASS", overrides: dict | None = None) -> Path:
    job_dir = fresh_dir / job
    job_dir.mkdir(parents=True, exist_ok=True)
    payload_hashes = {e["filename"]: e["sha256"] for e in payload_entries}
    receipt = {
        "schema": assembler.RECEIPT_SCHEMA, "job": job,
        "sha": EXPECTED.expected_sha, "run_id": EXPECTED.expected_run_id,
        "run_attempt": EXPECTED.expected_run_attempt,
        "workflow_path": EXPECTED.expected_workflow_path, "event_name": EXPECTED.expected_event,
        "semantic_status": semantic_status,
        "payload_manifest_sha256": assembler._payload_manifest_sha256(payload_hashes),
        "payloads": payload_entries,
    }
    if overrides:
        receipt.update(overrides)
    path = job_dir / "ci_producer_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def _build_positive_fresh_dir(tmp_path: Path) -> Path:
    fresh_dir = tmp_path / "assemble"
    _write_receipt(fresh_dir, "python-tests", payload_entries=[])
    _write_receipt(fresh_dir, "frontend-tests", payload_entries=[])

    container_dir = fresh_dir / "container"
    layer_scan = _write_payload_file(container_dir, "layer_scan.json",
                                      json.dumps({"forbidden_count": 0}).encode())
    smoke = _write_payload_file(container_dir, "container_smoke.json", json.dumps({
        "host_gateway_reachable": True, "mock_query_ok": True, "root_page_ok": True,
        "static_asset_ok": True, "production_test_seam_sealed": True,
    }).encode())
    _write_receipt(fresh_dir, "container", payload_entries=[layer_scan, smoke])

    m43_dir = fresh_dir / "m43-deterministic"
    m43_json = _write_payload_file(m43_dir, "m43.json", json.dumps(_m43_payload(negative=False)).encode())
    m43_negative = _write_payload_file(m43_dir, "m43-negative.json",
                                        json.dumps(_m43_payload(negative=True)).encode())
    _write_receipt(fresh_dir, "m43-deterministic", payload_entries=[m43_json, m43_negative])
    return fresh_dir


def _needs_success() -> dict:
    return {job: "success" for job in assembler.REQUIRED_PRODUCERS}


def _evidence_paths(fresh_dir: Path) -> dict:
    return {job: [str(fresh_dir / job / "ci_producer_receipt.json")]
            for job in assembler.REQUIRED_PRODUCERS}


def test_positive_all_producers_ok(tmp_path):
    fresh_dir = _build_positive_fresh_dir(tmp_path)
    needs = _needs_success()
    evidence = _evidence_paths(fresh_dir)
    for job in assembler.REQUIRED_PRODUCERS:
        result = assembler._evaluate_producer(job, needs[job], evidence[job], fresh_dir, EXPECTED)
        assert result["status"] == "OK", (job, result)


def test_expected_node_ids_matches_producer_profile_node_ids():
    assert assembler.EXPECTED_M43_NODE_IDS == set(run_m43_acceptance.PROFILE_NODE_IDS)


# --- DR-I6-MIN-01: semantic_status must be type-guarded before enum check --


@pytest.mark.parametrize("bad_value", [["PASS"], {"nested": "PASS"}, None, 1, True, 1.5])
def test_check_identity_rejects_non_string_semantic_status_without_crashing(bad_value):
    doc = {
        "schema": assembler.RECEIPT_SCHEMA, "job": "python-tests",
        "sha": EXPECTED.expected_sha, "run_id": EXPECTED.expected_run_id,
        "run_attempt": EXPECTED.expected_run_attempt,
        "workflow_path": EXPECTED.expected_workflow_path, "event_name": EXPECTED.expected_event,
        "semantic_status": bad_value,
        "payload_manifest_sha256": assembler._payload_manifest_sha256({}),
        "payloads": [],
    }
    ok, reason = assembler._check_identity(doc, "python-tests", EXPECTED)
    assert ok is False
    assert reason == "semantic_status_invalid"


def test_check_identity_rejects_invalid_string_semantic_status_enum():
    doc = {
        "schema": assembler.RECEIPT_SCHEMA, "job": "python-tests",
        "sha": EXPECTED.expected_sha, "run_id": EXPECTED.expected_run_id,
        "run_attempt": EXPECTED.expected_run_attempt,
        "workflow_path": EXPECTED.expected_workflow_path, "event_name": EXPECTED.expected_event,
        "semantic_status": "MAYBE",
        "payload_manifest_sha256": assembler._payload_manifest_sha256({}),
        "payloads": [],
    }
    ok, reason = assembler._check_identity(doc, "python-tests", EXPECTED)
    assert ok is False
    assert reason == "semantic_status_invalid"


# --- negative control matrix (representative subset of Design.md §8.3) ----


def test_negative_control_matrix(tmp_path):
    # missing evidence file
    missing_dir = tmp_path / "assemble_missing"
    missing_dir.mkdir()
    result = assembler._evaluate_producer("python-tests", "success", [], missing_dir, EXPECTED)
    assert result == {"status": "MISSING"}

    fresh_dir = tmp_path / "assemble"
    (fresh_dir / "container").mkdir(parents=True)
    (fresh_dir / "container" / "ci_producer_receipt.json").write_text("not json", encoding="utf-8")
    result = assembler._evaluate_producer("container", "success",
                                           [str(fresh_dir / "container" / "ci_producer_receipt.json")],
                                           fresh_dir, EXPECTED)
    assert result["status"] == "MALFORMED"

    # cross-sha mismatch
    tampered_dir = tmp_path / "assemble2"
    receipt_path = _write_receipt(tampered_dir, "python-tests", payload_entries=[])
    doc = json.loads(receipt_path.read_text())
    doc["sha"] = "b" * 40
    receipt_path.write_text(json.dumps(doc), encoding="utf-8")
    result = assembler._evaluate_producer("python-tests", "success", [str(receipt_path)],
                                           tampered_dir, EXPECTED)
    assert result == {"status": "IDENTITY_MISMATCH", "reason": "cross_sha_mismatch"}

    # duplicate producer
    fresh_dir3 = tmp_path / "assemble3"
    receipt_path3 = _write_receipt(fresh_dir3, "python-tests", payload_entries=[])
    result = assembler._evaluate_producer("python-tests", "success",
                                           [str(receipt_path3), str(receipt_path3)],
                                           fresh_dir3, EXPECTED)
    assert result == {"status": "DUPLICATE_PRODUCER", "count": 2}

    # skipped producer
    result = assembler._evaluate_producer("python-tests", "skipped", [], fresh_dir3, EXPECTED)
    assert result == {"status": "FAILED_OR_SKIPPED", "needs_result": "skipped"}

    # path traversal
    result = assembler._evaluate_producer(
        "python-tests", "success", [str(fresh_dir3.parent / "outside.json")], fresh_dir3, EXPECTED)
    assert result == {"status": "PATH_TRAVERSAL"}

    # unknown top-level key
    fresh_dir4 = tmp_path / "assemble4"
    receipt_path4 = _write_receipt(fresh_dir4, "python-tests", payload_entries=[])
    doc4 = json.loads(receipt_path4.read_text())
    doc4["note"] = "unexpected"
    receipt_path4.write_text(json.dumps(doc4), encoding="utf-8")
    result = assembler._evaluate_producer("python-tests", "success", [str(receipt_path4)],
                                           fresh_dir4, EXPECTED)
    assert result == {"status": "IDENTITY_MISMATCH", "reason": "unknown_or_missing_top_level_key"}

    # receipt job swap
    fresh_dir5 = tmp_path / "assemble5"
    receipt_path5 = _write_receipt(fresh_dir5, "container",
                                    payload_entries=[], overrides={"job": "m43-deterministic"})
    result = assembler._evaluate_producer("container", "success", [str(receipt_path5)],
                                           fresh_dir5, EXPECTED)
    assert result == {"status": "IDENTITY_MISMATCH", "reason": "receipt_job_mismatch"}

    # duplicate filename in payloads (same hash)
    fresh_dir6 = tmp_path / "assemble6"
    container_dir6 = fresh_dir6 / "container"
    layer_scan6 = _write_payload_file(container_dir6, "layer_scan.json",
                                       json.dumps({"forbidden_count": 0}).encode())
    smoke6 = _write_payload_file(container_dir6, "container_smoke.json", json.dumps({
        "host_gateway_reachable": True, "mock_query_ok": True, "root_page_ok": True,
        "static_asset_ok": True, "production_test_seam_sealed": True,
    }).encode())
    dup_entries = [layer_scan6, dict(layer_scan6), smoke6]
    receipt_path6 = fresh_dir6 / "container" / "ci_producer_receipt.json"
    payload_hashes6 = {"layer_scan.json": layer_scan6["sha256"], "container_smoke.json": smoke6["sha256"]}
    receipt6 = {
        "schema": assembler.RECEIPT_SCHEMA, "job": "container",
        "sha": EXPECTED.expected_sha, "run_id": EXPECTED.expected_run_id,
        "run_attempt": EXPECTED.expected_run_attempt,
        "workflow_path": EXPECTED.expected_workflow_path, "event_name": EXPECTED.expected_event,
        "semantic_status": "PASS",
        "payload_manifest_sha256": assembler._payload_manifest_sha256(payload_hashes6),
        "payloads": dup_entries,
    }
    receipt_path6.write_text(json.dumps(receipt6), encoding="utf-8")
    result = assembler._evaluate_producer("container", "success", [str(receipt_path6)],
                                           fresh_dir6, EXPECTED)
    assert result == {"status": "PAYLOAD_INVALID", "reason": "payload_duplicate_filename"}

    # payload hash mismatch (file replaced after receipt written)
    fresh_dir7 = _build_positive_fresh_dir(tmp_path / "assemble7_root")
    (fresh_dir7 / "container" / "layer_scan.json").write_text(
        json.dumps({"forbidden_count": 1}), encoding="utf-8")
    result = assembler._evaluate_producer("container", "success",
                                           [str(fresh_dir7 / "container" / "ci_producer_receipt.json")],
                                           fresh_dir7, EXPECTED)
    assert result["status"] == "PAYLOAD_INVALID"
    assert result["reason"] == "payload_hash_mismatch:layer_scan.json"

    # semantic mismatch: hash matches but semantic field is false
    fresh_dir8 = tmp_path / "assemble8"
    container_dir8 = fresh_dir8 / "container"
    layer_scan8 = _write_payload_file(container_dir8, "layer_scan.json",
                                       json.dumps({"forbidden_count": 0}).encode())
    smoke8 = _write_payload_file(container_dir8, "container_smoke.json", json.dumps({
        "host_gateway_reachable": False, "mock_query_ok": True, "root_page_ok": True,
        "static_asset_ok": True, "production_test_seam_sealed": True,
    }).encode())
    _write_receipt(fresh_dir8, "container", payload_entries=[layer_scan8, smoke8])
    result = assembler._evaluate_producer("container", "success",
                                           [str(fresh_dir8 / "container" / "ci_producer_receipt.json")],
                                           fresh_dir8, EXPECTED)
    assert result == {"status": "PAYLOAD_INVALID", "reason": "payload_semantic_mismatch:container_smoke.json"}

    # m43 node set mismatch
    fresh_dir9 = tmp_path / "assemble9"
    m43_dir9 = fresh_dir9 / "m43-deterministic"
    bad_m43 = _m43_payload(negative=False)
    del bad_m43["nodes"]["manifest_canonical"]
    m43_json9 = _write_payload_file(m43_dir9, "m43.json", json.dumps(bad_m43).encode())
    m43_negative9 = _write_payload_file(m43_dir9, "m43-negative.json",
                                         json.dumps(_m43_payload(negative=True)).encode())
    _write_receipt(fresh_dir9, "m43-deterministic", payload_entries=[m43_json9, m43_negative9])
    result = assembler._evaluate_producer("m43-deterministic", "success",
                                           [str(fresh_dir9 / "m43-deterministic" / "ci_producer_receipt.json")],
                                           fresh_dir9, EXPECTED)
    assert result == {"status": "PAYLOAD_INVALID", "reason": "m43_node_set_mismatch:m43.json"}

    # m43 negative control not rejected
    fresh_dir10 = tmp_path / "assemble10"
    m43_dir10 = fresh_dir10 / "m43-deterministic"
    m43_json10 = _write_payload_file(m43_dir10, "m43.json", json.dumps(_m43_payload(negative=False)).encode())
    tampered_negative = _m43_payload(negative=True)
    tampered_negative["negative_control"]["result"] = "TAMPERING_ACCEPTED_BUG"
    m43_negative10 = _write_payload_file(m43_dir10, "m43-negative.json", json.dumps(tampered_negative).encode())
    _write_receipt(fresh_dir10, "m43-deterministic", payload_entries=[m43_json10, m43_negative10])
    result = assembler._evaluate_producer("m43-deterministic", "success",
                                           [str(fresh_dir10 / "m43-deterministic" / "ci_producer_receipt.json")],
                                           fresh_dir10, EXPECTED)
    assert result == {"status": "PAYLOAD_INVALID",
                       "reason": "m43_negative_control_not_rejected:m43-negative.json"}


# ============================================================================
# M4 Operational Acceptance Recovery — v2 schema (Design.md §7.1)
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_REVISION = "adda1759754b56b514b3ab6252c2dc1032e03d28"


def _base_source_bytes() -> bytes:
    return subprocess.run(
        ["git", "show", f"{_BASE_REVISION}:scripts/assemble_m4_evidence.py"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout


def _current_source_bytes() -> bytes:
    return (REPO_ROOT / "scripts" / "assemble_m4_evidence.py").read_bytes()


def _assemble_args(fresh_dir: Path, needs_overrides: dict | None = None) -> SimpleNamespace:
    needs = _needs_success()
    if needs_overrides:
        needs.update(needs_overrides)
    evidence = _evidence_paths(fresh_dir)
    return SimpleNamespace(
        fresh_dir=str(fresh_dir),
        expected_sha=EXPECTED.expected_sha, expected_run_id=EXPECTED.expected_run_id,
        expected_run_attempt=EXPECTED.expected_run_attempt,
        expected_workflow_path=EXPECTED.expected_workflow_path, expected_event=EXPECTED.expected_event,
        needs_result=[f"{job}={result}" for job, result in needs.items()],
        evidence=[f"{job}={evidence[job][0]}" for job in assembler.REQUIRED_PRODUCERS],
        output=None,
    )


# --- §3.1a whole-file allowed-delta oracle (test-only audit tool; kept out
# of scripts/assemble_m4_evidence.py itself so the oracle never has to audit
# its own presence in the file it audits — Plan.md §2's exact-scope table
# lists only constants/assemble/main for the production script). ----------

_DECORATABLE_NODE_TYPES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _statement_source_slice(source: str, node: ast.stmt) -> str:
    """Full original-text slice for one statement. Identical to
    `ast.get_source_segment(source, node)` unless node is a decorated
    ClassDef/FunctionDef/AsyncFunctionDef (DR-RC1-I1-MAJ-01), in which case
    the slice start is extended back to the earliest decorator's `@`."""
    decorator_list = getattr(node, "decorator_list", None) or []
    if not isinstance(node, _DECORATABLE_NODE_TYPES) or not decorator_list:
        return ast.get_source_segment(source, node)

    lines = source.splitlines(keepends=True)
    first_decorator = decorator_list[0]
    decorator_line = lines[first_decorator.lineno - 1]
    at_index = decorator_line.rfind("@", 0, first_decorator.col_offset)
    start_lineno = first_decorator.lineno
    start_col = at_index if at_index != -1 else 0
    end_lineno = node.end_lineno
    end_col = node.end_col_offset

    if start_lineno == end_lineno:
        return lines[start_lineno - 1][start_col:end_col]
    first_line = lines[start_lineno - 1][start_col:]
    middle_lines = "".join(lines[start_lineno:end_lineno - 1])
    last_line = lines[end_lineno - 1][:end_col]
    return first_line + middle_lines + last_line


def _top_level_statement_slices(source: str) -> list[str]:
    tree = ast.parse(source)
    return [_statement_source_slice(source, node) for node in tree.body]


def _first_divergence_violation(expected: list[str], actual: list[str]) -> str:
    for index, (expected_slice, actual_slice) in enumerate(zip(expected, actual)):
        if expected_slice != actual_slice:
            return f"top_level_statement_changed:index={index}"
    if len(actual) > len(expected):
        return f"unapproved_new_top_level_statement:index={len(expected)}"
    return f"missing_top_level_statement:index={len(actual)}"


PINNED_BUILD_BASELINE_OLD_SLICE = 'def _build_baseline(producers: dict, deterministic_status: str, args) -> dict:\n    gates = {}\n    for job, gate_key in {\n        "python-tests": "python_tests", "frontend-tests": "frontend_tests",\n        "container": "container", "m43-deterministic": "m43_deterministic",\n    }.items():\n        gates[gate_key] = "PASS" if producers[job]["status"] == "OK" else "FAIL"\n    gates["m3_live_regression"] = "NOT_RUN"\n    gates["m41_operational"] = "BLOCKED"\n\n    m43_receipt_sha = None\n    if producers["m43-deterministic"]["status"] == "OK":\n        m43_receipt_sha = producers["m43-deterministic"]["payload_hashes"].get("m43.json")\n    image_digest = None\n    if producers["container"]["status"] == "OK":\n        image_digest = producers["container"]["payload_hashes"].get("container_smoke.json")\n\n    return {\n        "schema": "m4-baseline-v1", "schema_version": "1.0.0",\n        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),\n        "git_sha": args.expected_sha,\n        "workflow_run": {\n            "run_id": args.expected_run_id, "run_attempt": args.expected_run_attempt,\n            "workflow_path": args.expected_workflow_path, "event_name": args.expected_event,\n        },\n        "m3_fingerprint_reference": "evaluation/baselines/m3_initial.json",\n        "dependency_snapshot_sha256": _dependency_snapshot_sha256(),\n        "settings_hash": _settings_hash(),\n        "image_digest": image_digest,\n        "m43_deterministic_receipt_sha256": m43_receipt_sha,\n        "producers": producers,\n        "gates": gates,\n        "deterministic_status": deterministic_status,\n        "operational_status": "BLOCKED",\n        "M4.1_BLOCKED": True,\n        "overall_release_ready": False,\n    }'
PINNED_BUILD_BASELINE_NEW_SLICE = 'def _build_baseline(producers: dict, deterministic_status: str, args) -> dict:\n    gates = {}\n    for job, gate_key in {\n        "python-tests": "python_tests", "frontend-tests": "frontend_tests",\n        "container": "container", "m43-deterministic": "m43_deterministic",\n    }.items():\n        gates[gate_key] = "PASS" if producers[job]["status"] == "OK" else "FAIL"\n    gates["m3_live_regression"] = "NOT_ADOPTED"\n    gates["m41_operational"] = "NOT_ADOPTED"\n\n    m43_receipt_sha = None\n    if producers["m43-deterministic"]["status"] == "OK":\n        m43_receipt_sha = producers["m43-deterministic"]["payload_hashes"].get("m43.json")\n    image_digest = None\n    if producers["container"]["status"] == "OK":\n        image_digest = producers["container"]["payload_hashes"].get("container_smoke.json")\n\n    hosted_release_ready = deterministic_status == "PASS"\n    full_production_release_ready = False\n\n    return {\n        "schema": BASELINE_SCHEMA_V2, "schema_version": BASELINE_SCHEMA_VERSION_V2,\n        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),\n        "git_sha": args.expected_sha,\n        "workflow_run": {\n            "run_id": args.expected_run_id, "run_attempt": args.expected_run_attempt,\n            "workflow_path": args.expected_workflow_path, "event_name": args.expected_event,\n        },\n        "m3_fingerprint_reference": "evaluation/baselines/m3_initial.json",\n        "dependency_snapshot_sha256": _dependency_snapshot_sha256(),\n        "settings_hash": _settings_hash(),\n        "image_digest": image_digest,\n        "m43_deterministic_receipt_sha256": m43_receipt_sha,\n        "producers": producers,\n        "gates": gates,\n        "deterministic_status": deterministic_status,\n        "support_policy": dict(SUPPORT_POLICY_FIXED),\n        "operational_status": "NOT_ADOPTED",\n        "M4.1_BLOCKED": False,\n        "hosted_release_ready": hosted_release_ready,\n        "native_linux_release_ready": False,\n        "full_production_release_ready": full_production_release_ready,\n        "overall_release_ready": full_production_release_ready,\n    }'
PINNED_MAIN_OLD_SLICE = 'def main(argv: list[str] | None = None) -> int:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--fresh-dir", required=True)\n    parser.add_argument("--expected-sha", required=True)\n    parser.add_argument("--expected-run-id", default="local")\n    parser.add_argument("--expected-run-attempt", default="1")\n    parser.add_argument("--expected-workflow-path", default=".github/workflows/ci.yml")\n    parser.add_argument("--expected-event", default="local")\n    parser.add_argument("--needs-result", action="append", default=[])\n    parser.add_argument("--evidence", action="append", default=[])\n    parser.add_argument("--output", default=None)\n    args = parser.parse_args(argv)\n\n    baseline = assemble(args)\n    text = json.dumps(baseline, sort_keys=True, ensure_ascii=False, indent=2)\n    output = Path(args.output) if args.output else Path(args.fresh_dir) / "m4-baseline.json"\n    output.parent.mkdir(parents=True, exist_ok=True)\n    output.write_text(text + "\\n", encoding="utf-8")\n    print(text)\n    return 0 if baseline["deterministic_status"] == "PASS" else 1'
PINNED_MAIN_NEW_SLICE = 'def main(argv: list[str] | None = None) -> int:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--fresh-dir", required=True)\n    parser.add_argument("--expected-sha", required=True)\n    parser.add_argument("--expected-run-id", default="local")\n    parser.add_argument("--expected-run-attempt", default="1")\n    parser.add_argument("--expected-workflow-path", default=".github/workflows/ci.yml")\n    parser.add_argument("--expected-event", default="local")\n    parser.add_argument("--needs-result", action="append", default=[])\n    parser.add_argument("--evidence", action="append", default=[])\n    parser.add_argument("--output", default=None)\n    args = parser.parse_args(argv)\n\n    baseline = assemble(args)\n    text = json.dumps(baseline, sort_keys=True, ensure_ascii=False, indent=2)\n    output = Path(args.output) if args.output else Path(args.fresh_dir) / "m4-baseline.json"\n    output.parent.mkdir(parents=True, exist_ok=True)\n    output.write_text(text + "\\n", encoding="utf-8")\n    print(text)\n    return 0 if baseline["hosted_release_ready"] else 1'
PINNED_NEW_CONSTANTS_ANCHOR_SLICE = (
    'M43_NEGATIVE_KEYS = frozenset({"executed", "expected_to_fail", '
    '"actual_exit_code", "result"})'
)
PINNED_NEW_CONSTANT_SLICES = (
    'BASELINE_SCHEMA_V2 = "m4-baseline-v2"',
    'BASELINE_SCHEMA_VERSION_V2 = "2.0.0"',
    'SUPPORT_POLICY_SCHEMA = "m4-support-policy-v1"',
    'SUPPORT_POLICY_DECISION_DATE = "2026-08-15"',
    'SUPPORT_POLICY_FIXED = {\n'
    '    "schema": SUPPORT_POLICY_SCHEMA,\n'
    '    "adopted_scope": "HOSTED_OCI",\n'
    '    "native_linux_ollama": "NOT_ADOPTED",\n'
    '    "decision_date": SUPPORT_POLICY_DECISION_DATE,\n'
    '}',
)


def audit_exact_allowed_delta(base_source: str, current_source: str) -> list[str]:
    base_slices = _top_level_statement_slices(base_source)
    if base_slices.count(PINNED_BUILD_BASELINE_OLD_SLICE) != 1:
        return ["pinned_build_baseline_old_slice_not_uniquely_present_in_base"]
    if base_slices.count(PINNED_MAIN_OLD_SLICE) != 1:
        return ["pinned_main_old_slice_not_uniquely_present_in_base"]
    if base_slices.count(PINNED_NEW_CONSTANTS_ANCHOR_SLICE) != 1:
        return ["pinned_new_constants_anchor_not_uniquely_present_in_base"]

    expected_slices: list[str] = []
    for slice_ in base_slices:
        if slice_ == PINNED_BUILD_BASELINE_OLD_SLICE:
            expected_slices.append(PINNED_BUILD_BASELINE_NEW_SLICE)
        elif slice_ == PINNED_MAIN_OLD_SLICE:
            expected_slices.append(PINNED_MAIN_NEW_SLICE)
        else:
            expected_slices.append(slice_)
        if slice_ == PINNED_NEW_CONSTANTS_ANCHOR_SLICE:
            expected_slices.extend(PINNED_NEW_CONSTANT_SLICES)

    try:
        current_slices = _top_level_statement_slices(current_source)
    except SyntaxError:
        return ["current_source_not_parsable"]

    if current_slices == expected_slices:
        return []
    return [_first_divergence_violation(expected_slices, current_slices)]


# --- §3.1b preamble byte/token-aware boundary (DR-RC1-I2-MAJ-01), with the
# DR-RC1-I3-MIN-01 fix: classify a cookie using the same two `consumed`
# lines and line-position rule as `tokenize.detect_encoding` itself,
# independently of whether the first comment is a shebang. -----------------

_SHEBANG_PREFIX = b"#!"
_UTF8_BOM = b"\xef\xbb\xbf"
_ENCODING_COOKIE_RE = re.compile(rb"^[ \t\f]*#.*coding[:=][ \t]*([-_.a-zA-Z0-9]+)")

PINNED_BASE_PREAMBLE_BYTES = b"#!/usr/bin/env python3\n"


def _source_preamble(raw: bytes) -> bytes | None:
    """The byte-exact prefix of `raw` that is meaningful as an
    execution/decoding boundary: an optional UTF-8 BOM, followed by an
    optional shebang line, followed by an optional PEP 263 encoding-cookie
    line at whatever position CPython's own `tokenize.detect_encoding`
    would accept one — line 1 if line 1 itself is the cookie, otherwise
    line 2 if `tokenize.detect_encoding` needed to consume a second line
    (which happens whenever line 1 is blank-or-comment-only, including but
    not limited to a shebang; DR-RC1-I3-MIN-01). Returns `None` if `raw`
    is unparsable for encoding-detection purposes (fail-closed)."""
    try:
        _, consumed = tokenize.detect_encoding(io.BytesIO(raw).readline)
    except SyntaxError:
        return None
    bom = _UTF8_BOM if raw.startswith(_UTF8_BOM) else b""
    has_shebang = bool(consumed) and consumed[0].startswith(_SHEBANG_PREFIX)

    cookie_index = None
    if consumed and _ENCODING_COOKIE_RE.match(consumed[0]) is not None:
        cookie_index = 0
    elif len(consumed) > 1 and _ENCODING_COOKIE_RE.match(consumed[1]) is not None:
        cookie_index = 1

    if cookie_index is not None:
        kept_line_count = cookie_index + 1
    elif has_shebang:
        kept_line_count = 1
    else:
        kept_line_count = 0
    return bom + b"".join(consumed[:kept_line_count])


def audit_exact_allowed_delta_bytes(base_bytes: bytes, current_bytes: bytes) -> list[str]:
    base_preamble = _source_preamble(base_bytes)
    if base_preamble is None:
        return ["base_source_encoding_conflict"]
    current_preamble = _source_preamble(current_bytes)
    if current_preamble is None:
        return ["current_source_encoding_conflict"]
    if current_preamble != base_preamble:
        return ["preamble_shebang_or_encoding_declaration_changed"]

    base_encoding, _ = tokenize.detect_encoding(io.BytesIO(base_bytes).readline)
    current_encoding, _ = tokenize.detect_encoding(io.BytesIO(current_bytes).readline)
    try:
        base_text = base_bytes.decode(base_encoding)
    except UnicodeDecodeError:
        return ["base_source_undecodable"]
    try:
        current_text = current_bytes.decode(current_encoding)
    except UnicodeDecodeError:
        return ["current_source_undecodable"]

    return audit_exact_allowed_delta(base_text, current_text)


# --- mutant generator helpers (test-file-only) -----------------------------

def _append_top_level_statement(base_source: str, statement: str) -> str:
    return base_source.rstrip("\n") + "\n\n" + statement.rstrip("\n") + "\n"


def _in_place_whitespace_mutation(base_source: str, target_slice: str) -> str:
    assert base_source.count(target_slice) == 1, "fixture bug: slice not unique in base"
    head = target_slice.split("=")[0]
    if "(" in head:
        idx = target_slice.index("(")
    elif "=" in target_slice:
        idx = target_slice.index("=")
    else:
        idx = target_slice.index(" ")
    mutated_slice = target_slice[: idx + 1] + " " + target_slice[idx + 1 :]
    assert mutated_slice != target_slice
    return base_source.replace(target_slice, mutated_slice, 1)


def _with_decorators(source: str, target_slice: str, decorator_lines: tuple[str, ...]) -> str:
    assert source.count(target_slice) == 1, "fixture bug: slice not unique in source"
    prefix = "".join(line.rstrip("\n") + "\n" for line in decorator_lines)
    return source.replace(target_slice, prefix + target_slice, 1)


# --- v2 assembler tests -----------------------------------------------------


def test_assemble_v2_schema_and_version_constants(tmp_path):
    fresh_dir = _build_positive_fresh_dir(tmp_path)
    baseline = assembler.assemble(_assemble_args(fresh_dir))
    assert baseline["schema"] == "m4-baseline-v2"
    assert baseline["schema_version"] == "2.0.0"


def test_assemble_v2_support_policy_exact_fixed_object(tmp_path):
    fresh_dir = _build_positive_fresh_dir(tmp_path)
    baseline = assembler.assemble(_assemble_args(fresh_dir))
    assert baseline["support_policy"] == {
        "schema": "m4-support-policy-v1", "adopted_scope": "HOSTED_OCI",
        "native_linux_ollama": "NOT_ADOPTED", "decision_date": "2026-08-15",
    }


def test_assemble_v2_gates_m3_live_regression_and_m41_operational_are_not_adopted(tmp_path):
    fresh_dir = _build_positive_fresh_dir(tmp_path)
    baseline = assembler.assemble(_assemble_args(fresh_dir))
    assert baseline["gates"]["m3_live_regression"] == "NOT_ADOPTED"
    assert baseline["gates"]["m41_operational"] == "NOT_ADOPTED"


def test_assemble_v2_m41_blocked_is_false(tmp_path):
    fresh_dir = _build_positive_fresh_dir(tmp_path)
    baseline = assembler.assemble(_assemble_args(fresh_dir))
    assert baseline["M4.1_BLOCKED"] is False


def test_assemble_v2_operational_status_is_not_adopted(tmp_path):
    fresh_dir = _build_positive_fresh_dir(tmp_path)
    baseline = assembler.assemble(_assemble_args(fresh_dir))
    assert baseline["operational_status"] == "NOT_ADOPTED"


def test_assemble_v2_hosted_release_ready_true_when_all_four_producers_ok(tmp_path):
    fresh_dir = _build_positive_fresh_dir(tmp_path)
    baseline = assembler.assemble(_assemble_args(fresh_dir))
    assert baseline["hosted_release_ready"] is True


@pytest.mark.parametrize("failing_job", ["python-tests", "frontend-tests", "container", "m43-deterministic"])
def test_assemble_v2_hosted_release_ready_false_when_any_producer_not_ok(tmp_path, failing_job):
    fresh_dir = _build_positive_fresh_dir(tmp_path)
    baseline = assembler.assemble(_assemble_args(fresh_dir, needs_overrides={failing_job: "failure"}))
    assert baseline["hosted_release_ready"] is False


@pytest.mark.parametrize("needs_overrides", [None, {"container": "failure"}])
def test_assemble_v2_native_full_overall_always_false_regardless_of_producer_outcome(tmp_path, needs_overrides):
    fresh_dir = _build_positive_fresh_dir(tmp_path)
    baseline = assembler.assemble(_assemble_args(fresh_dir, needs_overrides=needs_overrides))
    assert baseline["native_linux_release_ready"] is False
    assert baseline["full_production_release_ready"] is False
    assert baseline["overall_release_ready"] is False


def test_assemble_v2_overall_release_ready_equals_full_production_release_ready_alias(tmp_path):
    fresh_dir = _build_positive_fresh_dir(tmp_path)
    baseline = assembler.assemble(_assemble_args(fresh_dir))
    assert baseline["overall_release_ready"] == baseline["full_production_release_ready"]


def test_assemble_v2_producers_and_m43_receipt_sha_shape_unchanged(tmp_path):
    fresh_dir = _build_positive_fresh_dir(tmp_path)
    baseline = assembler.assemble(_assemble_args(fresh_dir))
    assert set(baseline["producers"]) == set(assembler.REQUIRED_PRODUCERS)
    assert baseline["m43_deterministic_receipt_sha256"] == \
        baseline["producers"]["m43-deterministic"]["payload_hashes"]["m43.json"]
    assert baseline["image_digest"] == \
        baseline["producers"]["container"]["payload_hashes"]["container_smoke.json"]


@pytest.mark.parametrize("needs_overrides,expected_exit", [(None, 0), ({"container": "failure"}, 1)])
def test_assemble_v2_main_exit_code_reflects_hosted_release_ready(tmp_path, needs_overrides, expected_exit):
    fresh_dir = _build_positive_fresh_dir(tmp_path)
    argv = [
        "--fresh-dir", str(fresh_dir),
        "--expected-sha", EXPECTED.expected_sha,
        "--expected-run-id", EXPECTED.expected_run_id,
        "--expected-run-attempt", EXPECTED.expected_run_attempt,
        "--expected-workflow-path", EXPECTED.expected_workflow_path,
        "--expected-event", EXPECTED.expected_event,
        "--output", str(tmp_path / "out.json"),
    ]
    needs = _needs_success()
    if needs_overrides:
        needs.update(needs_overrides)
    for job, result in needs.items():
        argv += ["--needs-result", f"{job}={result}"]
    evidence = _evidence_paths(fresh_dir)
    for job in assembler.REQUIRED_PRODUCERS:
        argv += ["--evidence", f"{job}={evidence[job][0]}"]
    exit_code = assembler.main(argv)
    assert exit_code == expected_exit


# --- whole-file allowed-delta oracle: positive cases ------------------------


def test_audit_exact_allowed_delta_positive_actual_v2_file():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    assert audit_exact_allowed_delta(base_source, current_source) == []


def test_audit_exact_allowed_delta_positive_synthetic_fixture():
    base_source = _base_source_bytes().decode("utf-8")
    synthetic = base_source.replace(PINNED_BUILD_BASELINE_OLD_SLICE, PINNED_BUILD_BASELINE_NEW_SLICE, 1)
    synthetic = synthetic.replace(PINNED_MAIN_OLD_SLICE, PINNED_MAIN_NEW_SLICE, 1)
    synthetic = synthetic.replace(
        PINNED_NEW_CONSTANTS_ANCHOR_SLICE,
        PINNED_NEW_CONSTANTS_ANCHOR_SLICE + "\n" + "\n".join(PINNED_NEW_CONSTANT_SLICES),
        1,
    )
    assert audit_exact_allowed_delta(base_source, synthetic) == []


# --- whole-file allowed-delta oracle: negative (bypass) cases ---------------


def test_audit_exact_allowed_delta_rejects_new_import_statement():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    mutated = _append_top_level_statement(current_source, "import os")
    assert audit_exact_allowed_delta(base_source, mutated) == \
        [f"unapproved_new_top_level_statement:index={len(_top_level_statement_slices(current_source))}"]


def test_audit_exact_allowed_delta_rejects_import_rebinding_of_protected_name():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    mutated = _append_top_level_statement(current_source, "from attacker import REQUIRED_PRODUCERS")
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_class_shadow_of_protected_name():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    mutated = _append_top_level_statement(current_source, "class _evaluate_producer:\n    pass")
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_for_loop_target_rebinding():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    mutated = _append_top_level_statement(current_source, "for REQUIRED_PRODUCERS in ():\n    pass")
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_with_alias_rebinding():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    mutated = _append_top_level_statement(current_source, "with open('x') as _settings_hash:\n    pass")
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_async_function_shadow():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    mutated = _append_top_level_statement(current_source, "async def _check_identity(*a, **k):\n    return None")
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_exception_alias_rebinding():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    mutated = _append_top_level_statement(
        current_source, "try:\n    pass\nexcept Exception as _settings_hash:\n    pass")
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_top_level_named_expression_statement():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    mutated = _append_top_level_statement(current_source, "(REQUIRED_PRODUCERS := ('x',))")
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_duplicate_assignment_rebinding():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    mutated = _append_top_level_statement(current_source, 'REQUIRED_PRODUCERS = ("attacker-job",)')
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_duplicate_function_rebinding():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    mutated = _append_top_level_statement(current_source, "def _evaluate_producer(*a, **k):\n    return None")
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_new_executable_statement():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    mutated = _append_top_level_statement(current_source, 'print("x")')
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_new_unrelated_function():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    mutated = _append_top_level_statement(current_source, "def _new_helper():\n    return 1")
    assert audit_exact_allowed_delta(base_source, mutated) != []


@pytest.mark.parametrize("target_slice_name",
                          ["import sys", "REQUIRED_PRODUCERS", "def _check_identity(", "if str(_SRC)"])
def test_audit_exact_allowed_delta_rejects_in_place_whitespace_mutation(target_slice_name):
    base_source = _base_source_bytes().decode("utf-8")
    slices = _top_level_statement_slices(base_source)
    target_slice = next(s for s in slices if s.startswith(target_slice_name))
    mutated = _in_place_whitespace_mutation(base_source, target_slice)
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_assemble_modified():
    base_source = _base_source_bytes().decode("utf-8")
    slices = _top_level_statement_slices(base_source)
    assemble_slice = next(s for s in slices if s.startswith("def assemble("))
    mutated = _in_place_whitespace_mutation(base_source, assemble_slice)
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_main_non_exit_line_modified():
    base_source = _base_source_bytes().decode("utf-8")
    mutated_main = PINNED_MAIN_OLD_SLICE.replace('"--fresh-dir"', '"--fresh-dir2"')
    assert mutated_main != PINNED_MAIN_OLD_SLICE
    mutated = base_source.replace(PINNED_MAIN_OLD_SLICE, mutated_main, 1)
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_main_left_as_base_v1():
    base_source = _base_source_bytes().decode("utf-8")
    mutated = base_source.replace(PINNED_BUILD_BASELINE_OLD_SLICE, PINNED_BUILD_BASELINE_NEW_SLICE, 1)
    mutated = mutated.replace(
        PINNED_NEW_CONSTANTS_ANCHOR_SLICE,
        PINNED_NEW_CONSTANTS_ANCHOR_SLICE + "\n" + "\n".join(PINNED_NEW_CONSTANT_SLICES),
        1,
    )
    # main() left as base v1 — not replaced.
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_build_baseline_arbitrary_rewrite():
    base_source = _base_source_bytes().decode("utf-8")
    arbitrary = PINNED_BUILD_BASELINE_OLD_SLICE.replace(
        '"overall_release_ready": False,', '"overall_release_ready": False,\n        "extra_field": 1,')
    assert arbitrary != PINNED_BUILD_BASELINE_OLD_SLICE
    mutated = base_source.replace(PINNED_BUILD_BASELINE_OLD_SLICE, arbitrary, 1)
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_build_baseline_left_as_base_v1():
    base_source = _base_source_bytes().decode("utf-8")
    mutated = base_source.replace(PINNED_MAIN_OLD_SLICE, PINNED_MAIN_NEW_SLICE, 1)
    mutated = mutated.replace(
        PINNED_NEW_CONSTANTS_ANCHOR_SLICE,
        PINNED_NEW_CONSTANTS_ANCHOR_SLICE + "\n" + "\n".join(PINNED_NEW_CONSTANT_SLICES),
        1,
    )
    # _build_baseline left as base v1 — not replaced.
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_new_constants_inserted_at_wrong_location():
    base_source = _base_source_bytes().decode("utf-8")
    mutated = base_source.replace(PINNED_BUILD_BASELINE_OLD_SLICE, PINNED_BUILD_BASELINE_NEW_SLICE, 1)
    mutated = mutated.replace(PINNED_MAIN_OLD_SLICE, PINNED_MAIN_NEW_SLICE, 1)
    mutated = _append_top_level_statement(mutated, "\n\n".join(PINNED_NEW_CONSTANT_SLICES))
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_partial_pinned_constants_block():
    base_source = _base_source_bytes().decode("utf-8")
    mutated = base_source.replace(PINNED_BUILD_BASELINE_OLD_SLICE, PINNED_BUILD_BASELINE_NEW_SLICE, 1)
    mutated = mutated.replace(PINNED_MAIN_OLD_SLICE, PINNED_MAIN_NEW_SLICE, 1)
    mutated = mutated.replace(
        PINNED_NEW_CONSTANTS_ANCHOR_SLICE,
        PINNED_NEW_CONSTANTS_ANCHOR_SLICE + "\n" + "\n".join(PINNED_NEW_CONSTANT_SLICES[:-1]),
        1,
    )
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_missing_statement_removed_from_base():
    base_source = _base_source_bytes().decode("utf-8")
    mutated = base_source.replace("import re\n", "", 1)
    assert mutated != base_source
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_current_source_with_syntax_error():
    base_source = _base_source_bytes().decode("utf-8")
    assert audit_exact_allowed_delta(base_source, "def broken(:\n") == ["current_source_not_parsable"]


# --- decorator-span mutants (DR-RC1-I1-MAJ-01) ------------------------------


@pytest.mark.parametrize("before,after", [
    ((), ("@staticmethod",)),
    (("@staticmethod",), ()),
    (("@decorator_a",), ("@decorator_b",)),
    (("@decorator_a", "@decorator_b"), ("@decorator_b", "@decorator_a")),
])
def test_audit_exact_allowed_delta_rejects_decorator_mutations_on_assemble(before, after):
    base_source = _base_source_bytes().decode("utf-8")
    slices = _top_level_statement_slices(base_source)
    assemble_slice = next(s for s in slices if s.startswith("def assemble("))
    base_with_before = _with_decorators(base_source, assemble_slice, before)
    current_with_after = _with_decorators(base_source, assemble_slice, after)
    assert audit_exact_allowed_delta(base_with_before, current_with_after) != []


@pytest.mark.parametrize("function_prefix", ["def _evaluate_producer(", "def _check_identity("])
def test_audit_exact_allowed_delta_rejects_decorator_added_to_other_protected_function(function_prefix):
    base_source = _base_source_bytes().decode("utf-8")
    slices = _top_level_statement_slices(base_source)
    target_slice = next(s for s in slices if s.startswith(function_prefix))
    mutated = _with_decorators(base_source, target_slice, ("@staticmethod",))
    assert audit_exact_allowed_delta(base_source, mutated) != []


def test_audit_exact_allowed_delta_rejects_decorator_added_to_synthetic_class():
    base_source = _base_source_bytes().decode("utf-8")
    new_base = _append_top_level_statement(base_source, "class _Shadow:\n    pass")
    mutated = _with_decorators(new_base, "class _Shadow:\n    pass", ("@some_decorator",))
    assert audit_exact_allowed_delta(new_base, mutated) != []


def test_audit_exact_allowed_delta_rejects_decorator_added_to_synthetic_async_function():
    base_source = _base_source_bytes().decode("utf-8")
    stmt = "async def _shadow():\n    return None"
    new_base = _append_top_level_statement(base_source, stmt)
    mutated = _with_decorators(new_base, stmt, ("@some_decorator",))
    assert audit_exact_allowed_delta(new_base, mutated) != []


@pytest.mark.parametrize("decorator_lines", [("@staticmethod",), ("@decorator_a", "@decorator_b")])
def test_statement_source_slice_decorated_function_starts_at_at_symbol_and_includes_all_decorators(decorator_lines):
    base_source = _base_source_bytes().decode("utf-8")
    slices = _top_level_statement_slices(base_source)
    assemble_slice = next(s for s in slices if s.startswith("def assemble("))
    decorated_source = _with_decorators(base_source, assemble_slice, decorator_lines)
    decorated_slices = _top_level_statement_slices(decorated_source)
    decorated_assemble_slice = next(s for s in decorated_slices if "def assemble(" in s)
    assert decorated_assemble_slice.startswith("@")
    for line in decorator_lines:
        assert line in decorated_assemble_slice
    assert decorated_assemble_slice.endswith(assemble_slice[assemble_slice.index("def assemble("):][-20:])


def test_audit_exact_allowed_delta_comment_and_blank_line_insertions_between_statements_are_invisible():
    base_source = _base_source_bytes().decode("utf-8")
    current_source = _current_source_bytes().decode("utf-8")
    slices = _top_level_statement_slices(current_source)
    assemble_slice = next(s for s in slices if s.startswith("def assemble("))
    injected = '# REQUIRED_PRODUCERS = ("attacker-job",)\n\n' + assemble_slice
    current = current_source.replace(assemble_slice, injected, 1)
    assert audit_exact_allowed_delta(base_source, current) == []


# --- preamble byte/token-aware mutants (DR-RC1-I2-MAJ-01, DR-RC1-I3-MIN-01) -


def test_source_preamble_matches_pinned_base_preamble_bytes():
    base_bytes = _base_source_bytes()
    assert _source_preamble(base_bytes) == PINNED_BASE_PREAMBLE_BYTES == b"#!/usr/bin/env python3\n"


def test_audit_exact_allowed_delta_bytes_positive_actual_v2_file():
    base_bytes = _base_source_bytes()
    current_bytes = _current_source_bytes()
    assert audit_exact_allowed_delta_bytes(base_bytes, current_bytes) == []


def test_audit_exact_allowed_delta_bytes_rejects_shebang_modified():
    base_bytes = _base_source_bytes()
    current_bytes = base_bytes.replace(b"#!/usr/bin/env python3\n", b"#!/usr/bin/env -S python3 -O\n", 1)
    assert audit_exact_allowed_delta_bytes(base_bytes, current_bytes) == \
        ["preamble_shebang_or_encoding_declaration_changed"]


def test_audit_exact_allowed_delta_bytes_rejects_shebang_removed():
    base_bytes = _base_source_bytes()
    current_bytes = base_bytes.replace(b"#!/usr/bin/env python3\n", b"", 1)
    assert audit_exact_allowed_delta_bytes(base_bytes, current_bytes) == \
        ["preamble_shebang_or_encoding_declaration_changed"]


def test_audit_exact_allowed_delta_bytes_rejects_shebang_inserted_into_no_shebang_base():
    base_bytes = b'"""doc"""\nimport os\n'
    current_bytes = b"#!/usr/bin/env python3\n" + base_bytes
    assert audit_exact_allowed_delta_bytes(base_bytes, current_bytes) == \
        ["preamble_shebang_or_encoding_declaration_changed"]


def test_audit_exact_allowed_delta_bytes_rejects_encoding_cookie_inserted_with_non_ascii_semantic_reproduction():
    base_bytes = _base_source_bytes()
    lines = base_bytes.split(b"\n", 1)
    current_bytes = lines[0] + b"\n# coding: latin-1\n" + lines[1]
    assert audit_exact_allowed_delta_bytes(base_bytes, current_bytes) == \
        ["preamble_shebang_or_encoding_declaration_changed"]

    base_docstring_line = base_bytes.splitlines()[1]
    base_text = base_docstring_line.decode("utf-8")
    current_encoding, _ = tokenize.detect_encoding(io.BytesIO(current_bytes).readline)
    assert current_encoding == "iso-8859-1"
    current_text = base_docstring_line.decode(current_encoding)
    assert "—" in base_text
    assert "—" not in current_text
    assert base_text != current_text


def test_audit_exact_allowed_delta_bytes_rejects_encoding_cookie_modified():
    base_bytes = b"# coding: utf-8\nimport os\n"
    current_bytes = b"# coding: latin-1\nimport os\n"
    assert audit_exact_allowed_delta_bytes(base_bytes, current_bytes) == \
        ["preamble_shebang_or_encoding_declaration_changed"]


def test_audit_exact_allowed_delta_bytes_rejects_encoding_cookie_removed():
    base_bytes = b"# coding: utf-8\nimport os\n"
    current_bytes = b"import os\n"
    assert audit_exact_allowed_delta_bytes(base_bytes, current_bytes) == \
        ["preamble_shebang_or_encoding_declaration_changed"]


def test_audit_exact_allowed_delta_bytes_rejects_bom_inserted():
    base_bytes = _base_source_bytes()
    current_bytes = b"\xef\xbb\xbf" + base_bytes
    assert audit_exact_allowed_delta_bytes(base_bytes, current_bytes) == \
        ["preamble_shebang_or_encoding_declaration_changed"]


def test_audit_exact_allowed_delta_bytes_rejects_bom_plus_conflicting_cookie_fails_closed():
    base_bytes = _base_source_bytes()
    lines = base_bytes.split(b"\n", 1)
    current_bytes = b"\xef\xbb\xbf" + lines[0] + b"\n# coding: latin-1\n" + lines[1]
    assert audit_exact_allowed_delta_bytes(base_bytes, current_bytes) == \
        ["current_source_encoding_conflict"]


def test_audit_exact_allowed_delta_bytes_accepts_identical_bom_present_in_base_and_current():
    base_bytes = b"\xef\xbb\xbf" + _base_source_bytes()
    current_bytes = b"\xef\xbb\xbf" + _current_source_bytes()
    assert audit_exact_allowed_delta_bytes(base_bytes, current_bytes) == []


def test_audit_exact_allowed_delta_bytes_accepts_leading_non_cookie_comment_as_inert():
    base_bytes = _base_source_bytes()
    current_bytes = _current_source_bytes()
    lines = current_bytes.split(b"\n", 1)
    injected = lines[0] + b"\n# TODO: unrelated\n" + lines[1]
    assert audit_exact_allowed_delta_bytes(base_bytes, injected) == []


# --- DR-RC1-I3-MIN-01: comment-first, second-line cookie matrix ------------


def test_source_preamble_detects_comment_first_second_line_cookie():
    raw = b"# ordinary comment\n# coding: latin-1\nimport os\n"
    assert _source_preamble(raw) == b"# ordinary comment\n# coding: latin-1\n"
    detected_encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    assert detected_encoding == "iso-8859-1"


def test_audit_exact_allowed_delta_bytes_rejects_comment_first_second_line_cookie_modified():
    base_bytes = b"# ordinary comment\n# coding: latin-1\nimport os\n"
    current_bytes = b"# ordinary comment\n# coding: cp1252\nimport os\n"
    assert audit_exact_allowed_delta_bytes(base_bytes, current_bytes) == \
        ["preamble_shebang_or_encoding_declaration_changed"]


def test_audit_exact_allowed_delta_bytes_rejects_comment_first_second_line_cookie_removed():
    base_bytes = b"# ordinary comment\n# coding: latin-1\nimport os\n"
    current_bytes = b"# ordinary comment\nimport os\n"
    assert audit_exact_allowed_delta_bytes(base_bytes, current_bytes) == \
        ["preamble_shebang_or_encoding_declaration_changed"]


def test_audit_exact_allowed_delta_bytes_accepts_comment_first_second_line_cookie_unchanged():
    # This checks the §3.1b preamble-equality boundary directly (rather than
    # routing through the full §3.1a whole-file AST oracle, which is pinned
    # to the real assembler's base revision and cannot accept an unrelated
    # synthetic base) — an unchanged comment-first second-line cookie must
    # not be treated as a preamble difference.
    base_bytes = b"# ordinary comment\n# coding: latin-1\nimport os\nx = 1\n"
    current_bytes = b"# ordinary comment\n# coding: latin-1\nimport os\nx = 1\n"
    assert _source_preamble(base_bytes) == _source_preamble(current_bytes) == \
        b"# ordinary comment\n# coding: latin-1\n"
