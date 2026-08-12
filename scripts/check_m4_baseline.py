#!/usr/bin/env python3
"""M4.3-REQ-008 — M4 baseline state-algebra checker (Design.md §9.2).

Never trusts `assemble_m4_evidence.py`'s self-reported gates/status fields —
recomputes `gates` from `producers[job].status`, and recomputes
`deterministic_status`/`operational_status`/`overall_release_ready` from
that recomputed-gate local variable (never from the candidate's own
self-report), so a candidate that lies about its own conclusion is caught
even if `producers` looks internally plausible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from simple_qna_rag.index.manifest import canonical_json_bytes  # noqa: E402

GATE_ENUM = frozenset({"NOT_RUN", "SKIPPED", "UNKNOWN", "BLOCKED", "PASS", "FAIL"})
REQUIRED_TOP_KEYS = frozenset({
    "schema", "schema_version", "generated_at", "git_sha", "workflow_run",
    "m3_fingerprint_reference", "dependency_snapshot_sha256", "settings_hash",
    "image_digest", "m43_deterministic_receipt_sha256", "producers", "gates",
    "deterministic_status", "operational_status", "M4.1_BLOCKED",
    "overall_release_ready",
})
REQUIRED_GATE_KEYS = frozenset({
    "python_tests", "frontend_tests", "container", "m43_deterministic",
    "m3_live_regression", "m41_operational",
})
DETERMINISTIC_GATE_KEYS = frozenset({
    "python_tests", "frontend_tests", "container", "m43_deterministic",
})
PRODUCER_TO_GATE_KEY = {
    "python-tests": "python_tests", "frontend-tests": "frontend_tests",
    "container": "container", "m43-deterministic": "m43_deterministic",
}
REQUIRED_PRODUCER_KEYS = frozenset(PRODUCER_TO_GATE_KEY)
PRODUCER_STATUS_ENUM = frozenset({
    "OK", "MISSING", "FAILED_OR_SKIPPED", "DUPLICATE_PRODUCER",
    "IDENTITY_MISMATCH", "PATH_TRAVERSAL", "MALFORMED", "PAYLOAD_INVALID",
})
PRODUCER_STATUS_SCHEMA = {
    "OK": frozenset({"status", "receipt_sha256", "payload_hashes", "payload_manifest_sha256"}),
    "MISSING": frozenset({"status"}),
    "FAILED_OR_SKIPPED": frozenset({"status", "needs_result"}),
    "DUPLICATE_PRODUCER": frozenset({"status", "count"}),
    "IDENTITY_MISMATCH": frozenset({"status", "reason"}),
    "PATH_TRAVERSAL": frozenset({"status"}),
    "MALFORMED": frozenset({"status"}),
    "PAYLOAD_INVALID": frozenset({"status", "reason"}),
}
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
PRODUCER_EXPECTED_PAYLOAD_FILENAMES = {
    "python-tests": frozenset(), "frontend-tests": frozenset(),
    "container": frozenset({"layer_scan.json", "container_smoke.json"}),
    "m43-deterministic": frozenset({"m43.json", "m43-negative.json"}),
}


def _payload_manifest_sha256(payload_hashes: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload_hashes)).hexdigest()


def check(candidate: dict, *, expect_operational_blocked: bool) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not isinstance(candidate, dict):
        return False, ["candidate_not_object"]
    top_keys = set(candidate)
    if top_keys != REQUIRED_TOP_KEYS:
        issues.append(f"top_level_key_mismatch:missing={sorted(REQUIRED_TOP_KEYS - top_keys)}"
                      f",extra={sorted(top_keys - REQUIRED_TOP_KEYS)}")
        return False, issues

    gates = candidate["gates"]
    if not isinstance(gates, dict) or set(gates) != REQUIRED_GATE_KEYS:
        return False, ["gate_key_set_mismatch"]
    for name, value in gates.items():
        if not isinstance(value, str) or value not in GATE_ENUM:
            issues.append(f"unknown_gate_enum:{name}={value!r}")
    if issues:
        return False, issues

    producers = candidate["producers"]
    if not isinstance(producers, dict) or set(producers) != REQUIRED_PRODUCER_KEYS:
        return False, ["producer_key_set_mismatch"]
    expected_gates_from_producers: dict[str, str] = {}
    for job, gate_key in PRODUCER_TO_GATE_KEY.items():
        entry = producers[job]
        if not isinstance(entry, dict) or "status" not in entry:
            issues.append(f"producer_schema_invalid:{job}")
            continue
        status = entry["status"]
        if status not in PRODUCER_STATUS_ENUM:
            issues.append(f"producer_status_unknown:{job}={status!r}")
            continue
        if set(entry) != PRODUCER_STATUS_SCHEMA[status]:
            issues.append(f"producer_variant_schema_mismatch:{job}:status={status}:keys={sorted(entry)}")
            continue
        if status == "OK":
            receipt_sha = entry["receipt_sha256"]
            if not isinstance(receipt_sha, str) or not _HEX64_RE.fullmatch(receipt_sha):
                issues.append(f"producer_receipt_sha256_malformed:{job}")
                continue
            payload_hashes = entry["payload_hashes"]
            expected_filenames = PRODUCER_EXPECTED_PAYLOAD_FILENAMES[job]
            if not isinstance(payload_hashes, dict) or set(payload_hashes) != expected_filenames:
                issues.append(f"producer_payload_filename_set_mismatch:{job}")
                continue
            if any(not isinstance(k, str) or not isinstance(v, str) or not _HEX64_RE.fullmatch(v)
                   for k, v in payload_hashes.items()):
                issues.append(f"producer_payload_hashes_malformed:{job}")
                continue
            manifest_sha = entry["payload_manifest_sha256"]
            if not isinstance(manifest_sha, str) or not _HEX64_RE.fullmatch(manifest_sha):
                issues.append(f"producer_payload_manifest_sha256_malformed:{job}")
                continue
            if manifest_sha != _payload_manifest_sha256(payload_hashes):
                issues.append(f"producer_payload_manifest_sha256_mismatch:{job}")
                continue
        expected_gates_from_producers[gate_key] = "PASS" if status == "OK" else "FAIL"
    if issues:
        return False, issues

    for gate_key, expected in expected_gates_from_producers.items():
        if gates.get(gate_key) != expected:
            issues.append(f"gate_producer_algebra_mismatch:{gate_key}:"
                          f"producers_imply={expected},gates_say={gates.get(gate_key)!r}")
    if issues:
        return False, issues

    expected_deterministic = "PASS" if all(
        expected_gates_from_producers[gate_key] == "PASS" for gate_key in DETERMINISTIC_GATE_KEYS
    ) else "FAIL"
    if candidate.get("deterministic_status") != expected_deterministic:
        issues.append("deterministic_status_algebra_mismatch")

    expected_operational = "PASS" if (gates["m41_operational"] == "PASS"
                                       and gates["m3_live_regression"] == "PASS") else "BLOCKED"
    if candidate.get("operational_status") != expected_operational:
        issues.append("operational_status_algebra_mismatch")

    expected_ready = (expected_deterministic == "PASS" and expected_operational == "PASS")
    if candidate.get("overall_release_ready") != expected_ready:
        issues.append("overall_release_ready_algebra_mismatch")

    for bool_key in ("M4.1_BLOCKED", "overall_release_ready"):
        if not isinstance(candidate.get(bool_key), bool):
            issues.append(f"non_boolean_field:{bool_key}={candidate.get(bool_key)!r}")

    if expect_operational_blocked:
        if candidate.get("operational_status") != "BLOCKED" or \
                candidate.get("M4.1_BLOCKED") is not True or \
                candidate.get("overall_release_ready") is not False:
            issues.append("expected_operational_blocked_not_satisfied")
    return (not issues, issues)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expect-operational-blocked", action="store_true")
    args = parser.parse_args(argv)

    try:
        candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "issues": [f"candidate_unreadable:{type(exc).__name__}"]}),
              file=sys.stderr)
        return 1

    ok, issues = check(candidate, expect_operational_blocked=args.expect_operational_blocked)
    if not ok:
        print(json.dumps({"ok": False, "issues": issues}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "issues": []}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
