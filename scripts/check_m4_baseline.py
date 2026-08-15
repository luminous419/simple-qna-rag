#!/usr/bin/env python3
"""M4.3-REQ-008 — M4 baseline state-algebra checker (Design.md §9.2).

Never trusts `assemble_m4_evidence.py`'s self-reported gates/status fields —
recomputes `gates` from `producers[job].status`, and recomputes
`deterministic_status`/`operational_status`/`overall_release_ready` from
that recomputed-gate local variable (never from the candidate's own
self-report), so a candidate that lies about its own conclusion is caught
even if `producers` looks internally plausible.

Schema `m4-baseline-v2` (M4 Operational Acceptance Recovery,
docs/milestones/m4-operational-acceptance-recovery/Design.md §4) is the
default-accepted schema. Schema `m4-baseline-v1` is accepted only with the
explicit `--allow-legacy-v1` compatibility flag and is checked under its
original, unmodified fail-closed algebra (live `NOT_RUN`, M4.1 `BLOCKED`,
`M4.1_BLOCKED=true`, `overall_release_ready=false`) — v1 acceptance never
migrates or infers any v2 concept such as `hosted_release_ready`.
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

# Shared (unchanged names/values, producer/payload sub-structure is
# schema-agnostic).
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

# v1-only.
BASELINE_SCHEMA_V1 = "m4-baseline-v1"
BASELINE_SCHEMA_VERSION_V1 = "1.0.0"
GATE_ENUM_V1 = frozenset({"NOT_RUN", "SKIPPED", "UNKNOWN", "BLOCKED", "PASS", "FAIL"})
REQUIRED_TOP_KEYS_V1 = frozenset({
    "schema", "schema_version", "generated_at", "git_sha", "workflow_run",
    "m3_fingerprint_reference", "dependency_snapshot_sha256", "settings_hash",
    "image_digest", "m43_deterministic_receipt_sha256", "producers", "gates",
    "deterministic_status", "operational_status", "M4.1_BLOCKED",
    "overall_release_ready",
})

# v2-only.
BASELINE_SCHEMA_V2 = "m4-baseline-v2"
BASELINE_SCHEMA_VERSION_V2 = "2.0.0"
GATE_ENUM_V2 = frozenset({"PASS", "FAIL", "NOT_ADOPTED"})
FIXED_NOT_ADOPTED_GATE_KEYS = frozenset({"m3_live_regression", "m41_operational"})
REQUIRED_TOP_KEYS_V2 = REQUIRED_TOP_KEYS_V1 | frozenset({
    "support_policy", "hosted_release_ready", "native_linux_release_ready",
    "full_production_release_ready",
})
WORKFLOW_RUN_KEYS = frozenset({"run_id", "run_attempt", "workflow_path", "event_name"})
SUPPORT_POLICY_SCHEMA = "m4-support-policy-v1"
SUPPORT_POLICY_FIXED = {
    "schema": SUPPORT_POLICY_SCHEMA, "adopted_scope": "HOSTED_OCI",
    "native_linux_ollama": "NOT_ADOPTED", "decision_date": "2026-08-15",
}
SUPPORT_POLICY_KEYS = frozenset(SUPPORT_POLICY_FIXED)


def _payload_manifest_sha256(payload_hashes: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload_hashes)).hexdigest()


def _validate_producers_and_recompute_gates(
    candidate: dict,
) -> tuple[dict[str, str] | None, list[str]]:
    """Validates the `producers` structure and returns the recomputed
    deterministic gate_key -> "PASS"/"FAIL" mapping. If `issues` is
    non-empty, the first return value is `None`."""
    issues: list[str] = []
    producers = candidate.get("producers")
    if not isinstance(producers, dict) or set(producers) != REQUIRED_PRODUCER_KEYS:
        return None, ["producer_key_set_mismatch"]
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
        return None, issues
    return expected_gates_from_producers, []


def _check_v2(candidate: dict, *, expect_hosted_release_ready: bool,
              expect_hosted_not_ready: bool,
              expect_sha: str | None = None, expect_run_id: str | None = None,
              expect_run_attempt: str | None = None,
              expect_workflow_path: str | None = None,
              expect_event: str | None = None) -> tuple[bool, list[str]]:
    issues: list[str] = []
    top_keys = set(candidate)
    if top_keys != REQUIRED_TOP_KEYS_V2:
        return False, [f"top_level_key_mismatch:missing={sorted(REQUIRED_TOP_KEYS_V2 - top_keys)}"
                       f",extra={sorted(top_keys - REQUIRED_TOP_KEYS_V2)}"]

    # DR-I1-MAJ-02 — identity binding. This never re-fetches or re-hashes
    # original payload bytes (that happens only inside the assembler at CI
    # time, before this artifact is uploaded, per §4.7); it only proves that
    # the candidate's own declared identity is internally well-typed and, if
    # the operator supplied expected values, matches them.
    git_sha = candidate.get("git_sha")
    if not isinstance(git_sha, str) or not git_sha:
        issues.append(f"git_sha_not_nonempty_string:{git_sha!r}")
    workflow_run = candidate.get("workflow_run")
    if not isinstance(workflow_run, dict) or set(workflow_run) != WORKFLOW_RUN_KEYS:
        issues.append("workflow_run_key_set_mismatch")
    else:
        for key in WORKFLOW_RUN_KEYS:
            if not isinstance(workflow_run[key], str) or not workflow_run[key]:
                issues.append(f"workflow_run_field_not_nonempty_string:{key}")
    if issues:
        return False, issues

    if expect_sha is not None and git_sha != expect_sha:
        issues.append(f"identity_sha_mismatch:expected={expect_sha!r},got={git_sha!r}")
    if expect_run_id is not None and workflow_run["run_id"] != expect_run_id:
        issues.append(f"identity_run_id_mismatch:expected={expect_run_id!r},got={workflow_run['run_id']!r}")
    if expect_run_attempt is not None and workflow_run["run_attempt"] != expect_run_attempt:
        issues.append("identity_run_attempt_mismatch:expected="
                      f"{expect_run_attempt!r},got={workflow_run['run_attempt']!r}")
    if expect_workflow_path is not None and workflow_run["workflow_path"] != expect_workflow_path:
        issues.append("identity_workflow_path_mismatch:expected="
                      f"{expect_workflow_path!r},got={workflow_run['workflow_path']!r}")
    if expect_event is not None and workflow_run["event_name"] != expect_event:
        issues.append(f"identity_event_mismatch:expected={expect_event!r},got={workflow_run['event_name']!r}")
    if issues:
        return False, issues

    gates = candidate["gates"]
    if not isinstance(gates, dict) or set(gates) != REQUIRED_GATE_KEYS:
        return False, ["gate_key_set_mismatch"]
    for name, value in gates.items():
        if not isinstance(value, str) or value not in GATE_ENUM_V2:
            issues.append(f"unknown_gate_enum_v2:{name}={value!r}")
    if issues:
        return False, issues

    # step 3 — NOT_ADOPTED fixed-value enforcement. GATE_ENUM_V2 membership
    # alone would let m3_live_regression="PASS" (a member of the enum)
    # through, so these two keys additionally require exact literal
    # equality (§0.3-1 double defense).
    for key in FIXED_NOT_ADOPTED_GATE_KEYS:
        if gates[key] != "NOT_ADOPTED":
            issues.append(f"gate_not_adopted_fixed_value_violation:{key}={gates[key]!r}")
    if issues:
        return False, issues

    expected_gates_from_producers, producer_issues = _validate_producers_and_recompute_gates(candidate)
    if producer_issues:
        return False, producer_issues

    for gate_key, expected in expected_gates_from_producers.items():
        if gates.get(gate_key) != expected:
            issues.append(f"gate_producer_algebra_mismatch:{gate_key}:"
                          f"producers_imply={expected},gates_say={gates.get(gate_key)!r}")
    if issues:
        return False, issues

    # DR-I1-MAJ-02 — recompute the two top-level provenance aliases from the
    # SAME producers dict already validated above, never trust the
    # candidate's own top-level `image_digest`/`m43_deterministic_receipt_sha256`.
    # `_validate_producers_and_recompute_gates` already proved each producer
    # entry's shape matches `PRODUCER_STATUS_SCHEMA[status]` exactly, so an
    # "OK" entry is guaranteed to have `payload_hashes` as a dict here.
    container_entry = candidate["producers"]["container"]
    expected_image_digest = (
        container_entry["payload_hashes"].get("container_smoke.json")
        if container_entry["status"] == "OK" else None
    )
    if candidate.get("image_digest") != expected_image_digest:
        issues.append("image_digest_alias_mismatch:expected="
                      f"{expected_image_digest!r},got={candidate.get('image_digest')!r}")

    m43_entry = candidate["producers"]["m43-deterministic"]
    expected_m43_receipt_sha = (
        m43_entry["payload_hashes"].get("m43.json")
        if m43_entry["status"] == "OK" else None
    )
    if candidate.get("m43_deterministic_receipt_sha256") != expected_m43_receipt_sha:
        issues.append("m43_deterministic_receipt_sha256_alias_mismatch:expected="
                      f"{expected_m43_receipt_sha!r},"
                      f"got={candidate.get('m43_deterministic_receipt_sha256')!r}")
    if issues:
        return False, issues

    expected_deterministic = "PASS" if all(
        expected_gates_from_producers[k] == "PASS" for k in DETERMINISTIC_GATE_KEYS
    ) else "FAIL"
    if candidate.get("deterministic_status") != expected_deterministic:
        issues.append("deterministic_status_algebra_mismatch")

    expected_hosted_ready = expected_deterministic == "PASS"
    hosted_ready = candidate.get("hosted_release_ready")
    if not isinstance(hosted_ready, bool) or hosted_ready != expected_hosted_ready:
        issues.append("hosted_release_ready_algebra_mismatch")

    native_ready = candidate.get("native_linux_release_ready")
    if native_ready is not False:
        issues.append(f"native_linux_release_ready_not_false:{native_ready!r}")

    full_ready = candidate.get("full_production_release_ready")
    if full_ready is not False:
        issues.append(f"full_production_release_ready_not_false:{full_ready!r}")

    overall_ready = candidate.get("overall_release_ready")
    if overall_ready != full_ready:
        issues.append("overall_release_ready_alias_mismatch")

    if candidate.get("operational_status") != "NOT_ADOPTED":
        issues.append(f"operational_status_not_not_adopted:{candidate.get('operational_status')!r}")

    blocked = candidate.get("M4.1_BLOCKED")
    if blocked is not False:
        issues.append(f"m41_blocked_not_false:{blocked!r}")

    support_policy = candidate.get("support_policy")
    if not isinstance(support_policy, dict) or set(support_policy) != SUPPORT_POLICY_KEYS:
        issues.append("support_policy_key_set_mismatch")
    elif support_policy != SUPPORT_POLICY_FIXED:
        for field, expected_value in SUPPORT_POLICY_FIXED.items():
            if support_policy.get(field) != expected_value:
                issues.append(f"support_policy_field_mismatch:{field}="
                              f"{support_policy.get(field)!r},expected={expected_value!r}")
    if issues:
        return False, issues

    if expect_hosted_release_ready and hosted_ready is not True:
        issues.append("expected_hosted_release_ready_not_satisfied")
    if expect_hosted_not_ready and hosted_ready is not False:
        issues.append("expected_hosted_not_ready_not_satisfied")
    return (not issues, issues)


def _check_v1_legacy(candidate: dict, *, expect_operational_blocked: bool = False) -> tuple[bool, list[str]]:
    issues: list[str] = []
    top_keys = set(candidate)
    if top_keys != REQUIRED_TOP_KEYS_V1:
        return False, [f"top_level_key_mismatch:missing={sorted(REQUIRED_TOP_KEYS_V1 - top_keys)}"
                       f",extra={sorted(top_keys - REQUIRED_TOP_KEYS_V1)}"]

    gates = candidate["gates"]
    if not isinstance(gates, dict) or set(gates) != REQUIRED_GATE_KEYS:
        return False, ["gate_key_set_mismatch"]
    for name, value in gates.items():
        if not isinstance(value, str) or value not in GATE_ENUM_V1:
            issues.append(f"unknown_gate_enum:{name}={value!r}")
    if issues:
        return False, issues

    # Unconditional frozen-blocked legacy contract (REQ-003.2). NOT gated
    # behind expect_operational_blocked — --allow-legacy-v1 alone must
    # enforce this, because the historical artifact meaning IS this exact
    # fixed state, not "whatever v1's internally self-consistent algebra
    # happens to compute." Runs before producer validation so a candidate
    # cannot use a fabricated producers dict to distract from this check.
    if gates["m3_live_regression"] != "NOT_RUN":
        issues.append(f"v1_legacy_live_regression_not_not_run:{gates['m3_live_regression']!r}")
    if gates["m41_operational"] != "BLOCKED":
        issues.append(f"v1_legacy_m41_operational_not_blocked:{gates['m41_operational']!r}")
    if candidate.get("operational_status") != "BLOCKED":
        issues.append(f"v1_legacy_operational_status_not_blocked:{candidate.get('operational_status')!r}")
    if candidate.get("M4.1_BLOCKED") is not True:
        issues.append(f"v1_legacy_m41_blocked_not_true:{candidate.get('M4.1_BLOCKED')!r}")
    if candidate.get("overall_release_ready") is not False:
        issues.append(f"v1_legacy_overall_release_ready_not_false:{candidate.get('overall_release_ready')!r}")
    if issues:
        return False, issues

    expected_gates_from_producers, producer_issues = _validate_producers_and_recompute_gates(candidate)
    if producer_issues:
        return False, producer_issues

    for gate_key, expected in expected_gates_from_producers.items():
        if gates.get(gate_key) != expected:
            issues.append(f"gate_producer_algebra_mismatch:{gate_key}:"
                          f"producers_imply={expected},gates_say={gates.get(gate_key)!r}")
    if issues:
        return False, issues

    expected_deterministic = "PASS" if all(
        expected_gates_from_producers[k] == "PASS" for k in DETERMINISTIC_GATE_KEYS
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

    # `expect_operational_blocked` is now a REDUNDANT compatibility CLI
    # assertion, not the switch that activates frozen-blocked semantics —
    # the unconditional block above already enforces the same three fields
    # (plus the two gate values) regardless of this flag. It is kept only so
    # existing `--expect-operational-blocked` call sites keep parsing and
    # keep asserting the same claim explicitly; removing the flag would not
    # weaken `--allow-legacy-v1`'s guarantees.
    if expect_operational_blocked:
        if candidate.get("operational_status") != "BLOCKED" or \
                candidate.get("M4.1_BLOCKED") is not True or \
                candidate.get("overall_release_ready") is not False:
            issues.append("expected_operational_blocked_not_satisfied")
    return (not issues, issues)


def check(candidate: dict, *, allow_legacy_v1: bool = False,
          expect_hosted_release_ready: bool = False,
          expect_hosted_not_ready: bool = False,
          expect_operational_blocked: bool = False,
          expect_sha: str | None = None, expect_run_id: str | None = None,
          expect_run_attempt: str | None = None,
          expect_workflow_path: str | None = None,
          expect_event: str | None = None) -> tuple[bool, list[str]]:
    if not isinstance(candidate, dict):
        return False, ["candidate_not_object"]
    schema = candidate.get("schema")
    version = candidate.get("schema_version")
    if schema == BASELINE_SCHEMA_V2 and version == BASELINE_SCHEMA_VERSION_V2:
        return _check_v2(candidate, expect_hosted_release_ready=expect_hosted_release_ready,
                          expect_hosted_not_ready=expect_hosted_not_ready,
                          expect_sha=expect_sha, expect_run_id=expect_run_id,
                          expect_run_attempt=expect_run_attempt,
                          expect_workflow_path=expect_workflow_path, expect_event=expect_event)
    if schema == BASELINE_SCHEMA_V1 and version == BASELINE_SCHEMA_VERSION_V1:
        if not allow_legacy_v1:
            return False, ["legacy_v1_schema_requires_allow_legacy_v1_flag"]
        return _check_v1_legacy(candidate, expect_operational_blocked=expect_operational_blocked)
    return False, ["unknown_or_unsupported_schema"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--allow-legacy-v1", action="store_true",
                         help="Accept schema=m4-baseline-v1 and check it under "
                              "its original (pre-M4-OAR) fail-closed algebra.")
    parser.add_argument("--expect-hosted-release-ready", action="store_true")
    parser.add_argument("--expect-hosted-not-ready", action="store_true")
    parser.add_argument("--expect-operational-blocked", action="store_true",
                         help="Legacy v1-only redundant compatibility assertion; requires "
                              "--allow-legacy-v1. Does not activate frozen-blocked semantics — "
                              "--allow-legacy-v1 alone already enforces them (DR-I1-MAJ-01).")
    parser.add_argument("--expect-sha", default=None,
                         help="v2-only identity binding: candidate.git_sha must equal this.")
    parser.add_argument("--expect-run-id", default=None,
                         help="v2-only identity binding: candidate.workflow_run.run_id must equal this.")
    parser.add_argument("--expect-run-attempt", default=None,
                         help="v2-only identity binding: candidate.workflow_run.run_attempt must equal this.")
    parser.add_argument("--expect-workflow-path", default=None,
                         help="v2-only identity binding: candidate.workflow_run.workflow_path must equal this.")
    parser.add_argument("--expect-event", default=None,
                         help="v2-only identity binding: candidate.workflow_run.event_name must equal this.")
    parser.add_argument("--require-identity-binding", action="store_true",
                         help="Post-merge mode (DR-I1-MAJ-02): makes all five --expect-sha/"
                              "--expect-run-id/--expect-run-attempt/--expect-workflow-path/"
                              "--expect-event flags mandatory alongside --expect-hosted-release-ready. "
                              "The pre-merge fixture check (Plan.md §5, no real run to bind to yet) "
                              "does NOT set this flag; the post-merge runbook procedure (§8.3 §6.1) "
                              "always does.")
    args = parser.parse_args(argv)

    _IDENTITY_FLAG_NAMES = ("expect_sha", "expect_run_id", "expect_run_attempt",
                            "expect_workflow_path", "expect_event")

    if args.expect_operational_blocked and not args.allow_legacy_v1:
        parser.error("--expect-operational-blocked requires --allow-legacy-v1")
    if args.allow_legacy_v1 and (args.expect_hosted_release_ready or args.expect_hosted_not_ready
                                  or args.require_identity_binding
                                  or any(getattr(args, f) is not None for f in _IDENTITY_FLAG_NAMES)):
        parser.error("--expect-hosted-release-ready/--expect-hosted-not-ready/--require-identity-binding/"
                      "--expect-sha/--expect-run-id/--expect-run-attempt/--expect-workflow-path/"
                      "--expect-event are incompatible with --allow-legacy-v1 (v1 has no "
                      "hosted_release_ready or checker-verified identity fields)")
    if args.expect_hosted_release_ready and args.expect_hosted_not_ready:
        parser.error("--expect-hosted-release-ready and --expect-hosted-not-ready are mutually exclusive")
    # DR-I1-MAJ-02 — the post-merge hosted-ready assertion MUST be
    # identity-bound; a bare --expect-hosted-release-ready with no identity
    # flags would let an operator point the checker at a baseline copied
    # from a different run/SHA and still get a clean PASS. This requirement
    # is opt-in via --require-identity-binding (rather than implied by
    # --expect-hosted-release-ready alone) so the pre-merge fixture command
    # in Plan.md §5 — which has no real workflow run to bind to yet — keeps
    # working unmodified.
    if args.require_identity_binding:
        if not args.expect_hosted_release_ready:
            parser.error("--require-identity-binding requires --expect-hosted-release-ready")
        missing = [f"--{name.replace('_', '-')}" for name in _IDENTITY_FLAG_NAMES
                   if getattr(args, name) is None]
        if missing:
            parser.error("--require-identity-binding requires all five identity flags; missing: "
                          + ", ".join(missing))

    try:
        candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "issues": [f"candidate_unreadable:{type(exc).__name__}"]}),
              file=sys.stderr)
        return 1

    ok, issues = check(candidate, allow_legacy_v1=args.allow_legacy_v1,
                        expect_hosted_release_ready=args.expect_hosted_release_ready,
                        expect_hosted_not_ready=args.expect_hosted_not_ready,
                        expect_operational_blocked=args.expect_operational_blocked,
                        expect_sha=args.expect_sha, expect_run_id=args.expect_run_id,
                        expect_run_attempt=args.expect_run_attempt,
                        expect_workflow_path=args.expect_workflow_path, expect_event=args.expect_event)
    if not ok:
        print(json.dumps({"ok": False, "issues": issues}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "issues": []}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
