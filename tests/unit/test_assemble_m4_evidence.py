"""M4.3-REQ-007 — assembler exact-schema/payload-verification negative matrix
(Design.md §8.2, DR-I5-MAJ-01, and DR-I6-MIN-01's non-string semantic_status
guard)."""

from __future__ import annotations

import copy
import hashlib
import json
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
