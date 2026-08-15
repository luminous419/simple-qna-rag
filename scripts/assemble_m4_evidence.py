#!/usr/bin/env python3
"""M4.3-REQ-007 — fresh-directory evidence assembler (Design.md §8.2).

Reads only downloaded producer receipts inside `--fresh-dir`, verifies each
against an exact tagged schema *before* any dict/set reduction (DR-I5-MAJ-01
— this is what closes duplicate-filename-swallowing and job/schema
substitution), re-derives each payload's hash/size/semantic state from the
actual payload bytes (never trusts a receipt's self-reported
`semantic_status`), and only then builds the M4 baseline candidate.
`semantic_status` itself is explicitly guarded against non-string JSON types
(DR-I6-MIN-01) before the enum membership check, so a malformed
list/object/bool/number value produces a typed `IDENTITY_MISMATCH` instead of
crashing the assembler process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from simple_qna_rag.index.manifest import canonical_json_bytes  # noqa: E402

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_PRODUCERS = ("python-tests", "frontend-tests", "container", "m43-deterministic")

RECEIPT_SCHEMA = "m43-producer-receipt-v1"
RECEIPT_TOP_KEYS = frozenset({
    "schema", "job", "sha", "run_id", "run_attempt", "workflow_path",
    "event_name", "semantic_status", "payload_manifest_sha256", "payloads",
})
SEMANTIC_STATUS_ENUM = frozenset({"PASS", "FAIL"})
PAYLOAD_ENTRY_KEYS = frozenset({"filename", "sha256", "size_bytes"})

REQUIRED_PAYLOADS = {
    "python-tests": {}, "frontend-tests": {},
    "container": {
        "layer_scan.json": ("forbidden_count", 0),
        "container_smoke.json": (
            ("host_gateway_reachable", "mock_query_ok", "root_page_ok", "static_asset_ok",
             "production_test_seam_sealed"),
            (True, True, True, True, True)),
    },
    "m43-deterministic": {
        "m43.json": ("_typed_m43", False),
        "m43-negative.json": ("_typed_m43", True),
    },
}
KNOWN_PAYLOAD_FILENAMES = frozenset().union(*(set(v) for v in REQUIRED_PAYLOADS.values()))

# Assembler's own review-pinned node set — deliberately does NOT import
# run_m43_acceptance.PROFILE_NODE_IDS (DR-I3-MAJ-03 independence). Kept in
# sync with scripts/run_m43_acceptance.py::PROFILE_NODE_IDS by
# tests/unit/test_assemble_m4_evidence.py::test_expected_node_ids_matches_producer_profile_node_ids.
EXPECTED_M43_NODE_IDS = frozenset({
    "manifest_canonical", "manifest_negative", "verification_trust",
    "verification_reopen_race", "legacy_baseline_pin", "staging_fault",
    "activation_rollback", "crash_recovery_journal", "lock_untrusted_symlink",
    "legacy_import", "retention", "lock_contention", "layer_scanner",
    "container_static_and_connectivity", "embedding_provider_seam_guard",
    "assemble_payload_verification", "baseline_strict_schema",
})

M43_SCHEMA = "m43-acceptance-receipt-v1"
M43_SEED = 4303
M43_REPEAT = 10
M43_EXPECTED_COMMAND = (f"run_m43_acceptance.py --profile deterministic "
                        f"--repeat {M43_REPEAT} --seed {M43_SEED}")
M43_TOP_KEYS = frozenset({"schema", "profile", "seed", "repeat", "command",
                          "started_at", "finished_at", "nodes",
                          "negative_control", "status"})
M43_NODE_KEYS = frozenset({"repeat", "success_count", "status"})
M43_NEGATIVE_KEYS = frozenset({"executed", "expected_to_fail", "actual_exit_code", "result"})
BASELINE_SCHEMA_V2 = "m4-baseline-v2"
BASELINE_SCHEMA_VERSION_V2 = "2.0.0"
SUPPORT_POLICY_SCHEMA = "m4-support-policy-v1"
SUPPORT_POLICY_DECISION_DATE = "2026-08-15"
SUPPORT_POLICY_FIXED = {
    "schema": SUPPORT_POLICY_SCHEMA,
    "adopted_scope": "HOSTED_OCI",
    "native_linux_ollama": "NOT_ADOPTED",
    "decision_date": SUPPORT_POLICY_DECISION_DATE,
}


def _payload_manifest_sha256(payload_hashes: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload_hashes)).hexdigest()


def _resolve_contained(path: str, fresh_dir: Path) -> Path | None:
    resolved = Path(path).resolve()
    if not str(resolved).startswith(str(fresh_dir) + "/") and resolved != fresh_dir:
        return None
    return resolved


def _check_identity(doc, job, args) -> tuple[bool, str | None]:
    if not isinstance(doc, dict):
        return False, "receipt_not_object"
    if set(doc) != RECEIPT_TOP_KEYS:
        return False, "unknown_or_missing_top_level_key"
    if doc["schema"] != RECEIPT_SCHEMA:
        return False, "wrong_schema"
    if doc["job"] != job:
        return False, "receipt_job_mismatch"
    if not isinstance(doc["sha"], str) or doc["sha"] != args.expected_sha:
        return False, "cross_sha_mismatch"
    if not isinstance(doc["run_id"], (str, int)) or isinstance(doc["run_id"], bool) \
            or str(doc["run_id"]) != str(args.expected_run_id):
        return False, "cross_run_mismatch"
    if not isinstance(doc["run_attempt"], (str, int)) or isinstance(doc["run_attempt"], bool) \
            or str(doc["run_attempt"]) != str(args.expected_run_attempt):
        return False, "cross_run_attempt_mismatch"
    if not isinstance(doc["workflow_path"], str) or doc["workflow_path"] != args.expected_workflow_path:
        return False, "workflow_path_mismatch"
    if not isinstance(doc["event_name"], str) or doc["event_name"] != args.expected_event:
        return False, "event_mismatch"
    # DR-I6-MIN-01: type-guard *before* the frozenset membership check —
    # unhashable JSON types (list/dict) would otherwise raise TypeError and
    # crash the assembler process instead of producing a typed FAIL.
    if not isinstance(doc["semantic_status"], str) or doc["semantic_status"] not in SEMANTIC_STATUS_ENUM:
        return False, "semantic_status_invalid"
    if not isinstance(doc["payload_manifest_sha256"], str):
        return False, "payload_manifest_sha256_not_string"
    if not isinstance(doc["payloads"], list):
        return False, "payloads_not_list"
    return True, None


def _parse_and_verify_m43_payload(raw: bytes, *, expect_negative: bool) -> tuple[bool, str | None]:
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, "m43_payload_malformed_json"
    if not isinstance(doc, dict) or set(doc) != M43_TOP_KEYS:
        return False, "m43_payload_key_mismatch"
    if doc.get("schema") != M43_SCHEMA or doc.get("profile") != "deterministic" or \
            doc.get("seed") != M43_SEED or doc.get("repeat") != M43_REPEAT:
        return False, "m43_identity_mismatch"
    if doc.get("command") != M43_EXPECTED_COMMAND:
        return False, "m43_command_mismatch"
    nodes = doc.get("nodes")
    if not isinstance(nodes, dict) or set(nodes) != EXPECTED_M43_NODE_IDS:
        return False, "m43_node_set_mismatch"
    for name, node in nodes.items():
        if not isinstance(node, dict) or set(node) != M43_NODE_KEYS:
            return False, f"m43_node_schema_mismatch:{name}"
        if node.get("repeat") != M43_REPEAT or node.get("success_count") != M43_REPEAT \
                or node.get("status") != "PASS":
            return False, f"m43_node_not_fully_passed:{name}"
    neg = doc.get("negative_control")
    if not isinstance(neg, dict) or set(neg) != M43_NEGATIVE_KEYS:
        return False, "m43_negative_control_schema_mismatch"
    if expect_negative:
        if neg.get("executed") is not True or neg.get("expected_to_fail") is not True or \
                neg.get("actual_exit_code") != 1 or neg.get("result") != "REJECTED_AS_EXPECTED" or \
                doc.get("status") != "REJECTED_AS_EXPECTED":
            return False, "m43_negative_control_not_rejected"
    else:
        if neg.get("executed") is not False or neg.get("expected_to_fail") is not None or \
                neg.get("actual_exit_code") is not None or neg.get("result") is not None or \
                doc.get("status") != "PASS":
            return False, "m43_positive_status_not_pass"
    return True, None


def _verify_payloads(job, doc, payload_dir, fresh_dir) -> tuple[bool, str | None, dict[str, str] | None]:
    required_files = REQUIRED_PAYLOADS.get(job, {})
    raw_payloads = doc["payloads"]

    for entry in raw_payloads:
        if not isinstance(entry, dict) or set(entry) != PAYLOAD_ENTRY_KEYS:
            return False, "payload_entry_schema_invalid", None
        filename, sha256_value, size_bytes = entry["filename"], entry["sha256"], entry["size_bytes"]
        if not isinstance(filename, str) or filename not in KNOWN_PAYLOAD_FILENAMES:
            return False, "payload_entry_filename_not_allowlisted", None
        if not isinstance(sha256_value, str) or not _HEX64_RE.fullmatch(sha256_value):
            return False, "payload_entry_sha256_invalid", None
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            return False, "payload_entry_size_bytes_invalid", None

    raw_filenames = [entry["filename"] for entry in raw_payloads]
    if len(raw_filenames) != len(set(raw_filenames)):
        return False, "payload_duplicate_filename", None

    declared = {entry["filename"]: entry for entry in raw_payloads}
    if set(required_files) != set(declared):
        return False, "payload_set_mismatch", None
    for filename, spec in required_files.items():
        entry = declared[filename]
        target = _resolve_contained(str(payload_dir / filename), fresh_dir)
        if target is None or not target.is_file():
            return False, f"payload_missing:{filename}", None
        actual_bytes = target.read_bytes()
        if len(actual_bytes) != entry["size_bytes"]:
            return False, f"payload_size_mismatch:{filename}", None
        if hashlib.sha256(actual_bytes).hexdigest() != entry["sha256"]:
            return False, f"payload_hash_mismatch:{filename}", None
        if spec[0] == "_typed_m43":
            ok, reason = _parse_and_verify_m43_payload(actual_bytes, expect_negative=spec[1])
            if not ok:
                return False, f"{reason}:{filename}", None
            continue
        try:
            payload_doc = json.loads(actual_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False, f"payload_malformed:{filename}", None
        fields, expected = spec
        if isinstance(fields, tuple):
            actual = tuple(payload_doc.get(f) for f in fields)
        else:
            actual = payload_doc.get(fields)
        if actual != expected:
            return False, f"payload_semantic_mismatch:{filename}", None
    payload_hashes = {filename: declared[filename]["sha256"] for filename in required_files}
    return True, None, payload_hashes


def _evaluate_producer(job, needs_result, paths, fresh_dir, args) -> dict:
    if needs_result != "success":
        return {"status": "FAILED_OR_SKIPPED", "needs_result": needs_result}
    if len(paths) == 0:
        return {"status": "MISSING"}
    if len(paths) > 1:
        return {"status": "DUPLICATE_PRODUCER", "count": len(paths)}
    receipt_path = _resolve_contained(paths[0], fresh_dir)
    if receipt_path is None:
        return {"status": "PATH_TRAVERSAL"}
    try:
        doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "MALFORMED"}
    ok, reason = _check_identity(doc, job, args)
    if not ok:
        return {"status": "IDENTITY_MISMATCH", "reason": reason}
    payload_dir = receipt_path.parent
    ok, reason, payload_hashes = _verify_payloads(job, doc, payload_dir, fresh_dir)
    if not ok:
        return {"status": "PAYLOAD_INVALID", "reason": reason}
    computed_manifest_sha256 = _payload_manifest_sha256(payload_hashes)
    declared_manifest_sha256 = doc["payload_manifest_sha256"]
    if not _HEX64_RE.fullmatch(declared_manifest_sha256):
        return {"status": "PAYLOAD_INVALID", "reason": "payload_manifest_sha256_malformed"}
    if declared_manifest_sha256 != computed_manifest_sha256:
        return {"status": "PAYLOAD_INVALID", "reason": "payload_manifest_sha256_mismatch"}
    return {"status": "OK", "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "payload_hashes": payload_hashes,
            "payload_manifest_sha256": computed_manifest_sha256}


def _group_by_job(evidence_args: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in evidence_args:
        job, _, path = item.partition("=")
        grouped.setdefault(job, []).append(path)
    return grouped


def _parse_needs_result(items: list[str]) -> dict[str, str]:
    result = {}
    for item in items:
        job, _, value = item.partition("=")
        result[job] = value
    return result


def _assert_no_unexpected_entries(fresh_dir: Path, expected_subdirs: tuple[str, ...]) -> None:
    if not fresh_dir.is_dir():
        return
    allowed = set(expected_subdirs)
    for entry in fresh_dir.iterdir():
        if entry.name not in allowed:
            raise ValueError(f"unexpected_entry_in_fresh_dir:{entry.name}")


def _build_baseline(producers: dict, deterministic_status: str, args) -> dict:
    gates = {}
    for job, gate_key in {
        "python-tests": "python_tests", "frontend-tests": "frontend_tests",
        "container": "container", "m43-deterministic": "m43_deterministic",
    }.items():
        gates[gate_key] = "PASS" if producers[job]["status"] == "OK" else "FAIL"
    gates["m3_live_regression"] = "NOT_ADOPTED"
    gates["m41_operational"] = "NOT_ADOPTED"

    m43_receipt_sha = None
    if producers["m43-deterministic"]["status"] == "OK":
        m43_receipt_sha = producers["m43-deterministic"]["payload_hashes"].get("m43.json")
    image_digest = None
    if producers["container"]["status"] == "OK":
        image_digest = producers["container"]["payload_hashes"].get("container_smoke.json")

    hosted_release_ready = deterministic_status == "PASS"
    full_production_release_ready = False

    return {
        "schema": BASELINE_SCHEMA_V2, "schema_version": BASELINE_SCHEMA_VERSION_V2,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": args.expected_sha,
        "workflow_run": {
            "run_id": args.expected_run_id, "run_attempt": args.expected_run_attempt,
            "workflow_path": args.expected_workflow_path, "event_name": args.expected_event,
        },
        "m3_fingerprint_reference": "evaluation/baselines/m3_initial.json",
        "dependency_snapshot_sha256": _dependency_snapshot_sha256(),
        "settings_hash": _settings_hash(),
        "image_digest": image_digest,
        "m43_deterministic_receipt_sha256": m43_receipt_sha,
        "producers": producers,
        "gates": gates,
        "deterministic_status": deterministic_status,
        "support_policy": dict(SUPPORT_POLICY_FIXED),
        "operational_status": "NOT_ADOPTED",
        "M4.1_BLOCKED": False,
        "hosted_release_ready": hosted_release_ready,
        "native_linux_release_ready": False,
        "full_production_release_ready": full_production_release_ready,
        "overall_release_ready": full_production_release_ready,
    }


def _dependency_snapshot_sha256() -> str:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from dependency_snapshot import LOCK_FILE, _lock_sha256_canonical

    return _lock_sha256_canonical(LOCK_FILE.read_text(encoding="utf-8"))


def _settings_hash() -> str:
    from simple_qna_rag.rag_engine import _settings_binding_snapshot

    return hashlib.sha256(canonical_json_bytes(_settings_binding_snapshot())).hexdigest()


def assemble(args) -> dict:
    fresh_dir = Path(args.fresh_dir).resolve()
    _assert_no_unexpected_entries(fresh_dir, expected_subdirs=REQUIRED_PRODUCERS)
    needs = _parse_needs_result(args.needs_result)
    evidence_paths = _group_by_job(args.evidence)
    producers = {}
    for job in REQUIRED_PRODUCERS:
        producers[job] = _evaluate_producer(job, needs.get(job), evidence_paths.get(job, []),
                                             fresh_dir, args)
    deterministic_status = "PASS" if all(p["status"] == "OK" for p in producers.values()) else "FAIL"
    return _build_baseline(producers, deterministic_status, args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-dir", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--expected-run-id", default="local")
    parser.add_argument("--expected-run-attempt", default="1")
    parser.add_argument("--expected-workflow-path", default=".github/workflows/ci.yml")
    parser.add_argument("--expected-event", default="local")
    parser.add_argument("--needs-result", action="append", default=[])
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    baseline = assemble(args)
    text = json.dumps(baseline, sort_keys=True, ensure_ascii=False, indent=2)
    output = Path(args.output) if args.output else Path(args.fresh_dir) / "m4-baseline.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if baseline["hosted_release_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
