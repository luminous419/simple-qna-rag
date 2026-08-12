#!/usr/bin/env python3
"""M4.3 deterministic acceptance repeat runner (Design.md §10).

Runs each named acceptance node's pytest ids `--repeat` times (default 10)
and reports per-node `success_count`. `--inject-evidence-mismatch` exercises
`assemble_m4_evidence.py::_check_identity` as a negative control: it tampers
a genuine producer receipt's `sha` field and asserts the same parser rejects
it — process exit 1 is the *expected success* of that mode (a rejected
tamper), matching `run_m42_acceptance.py`'s `--inject-conservation-mismatch`
convention.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

_SCRIPTS_DIR = Path(__file__).resolve().parent

# Node ids intentionally reference a reduced-but-representative subset of the
# fault/negative matrix implemented in this milestone — see
# Implementation_Report.md for the explicit list of design-cataloged cases
# not yet promoted into this profile.
PROFILE_NODE_IDS = MappingProxyType({
    "manifest_canonical": ("tests/unit/test_index_manifest.py::test_canonical_round_trip_100x",),
    "manifest_negative": (
        "tests/unit/test_index_manifest.py::test_parse_manifest_rejects_self_hash_mismatch_after_tamper",),
    "verification_trust": (
        "tests/unit/test_index_verification.py::test_symlink_owner_mode_toctou_matrix",
        "tests/unit/test_index_verification.py::test_current_pointer_trust_matrix",
    ),
    "verification_reopen_race": (
        "tests/unit/test_index_verification.py::test_verify_then_load_uses_captured_bytes_no_reopen",
        "tests/unit/test_index_verification.py::test_racer_between_member_opens_has_no_effect",
    ),
    "legacy_baseline_pin": (
        "tests/unit/test_pinned_baseline_provenance.py::test_pinned_constants_match_tracked_m3_baseline_bytes",
        "tests/unit/test_index_lifecycle.py::test_import_legacy_rejects_source_hash_mismatch",
    ),
    "staging_fault": (
        "tests/integration/test_index_lifecycle_fault_injection.py::test_staging_fault_matrix_preserves_current",),
    "activation_rollback": (
        "tests/unit/test_index_lifecycle.py::test_activate_rollback_100x",),
    "crash_recovery_journal": (
        "tests/integration/test_index_lifecycle_fault_injection.py::test_crash_recovery_journal_reconciles_to_consistent_state",
        "tests/integration/test_index_lifecycle_fault_injection.py::test_crash_recovery_history_and_receipt_exact_once_matrix",
    ),
    "lock_untrusted_symlink": (
        "tests/unit/test_index_lifecycle.py::test_preexisting_lock_symlink_rejected",),
    "legacy_import": (
        "tests/unit/test_index_lifecycle.py::test_import_legacy_preserves_source_bytes_and_accepts_override",),
    "retention": (
        "tests/unit/test_index_lifecycle.py::test_cleanup_dry_run_then_apply_protects_current_and_previous",
        "tests/unit/test_index_lifecycle.py::test_cleanup_staging_protects_unexpected_and_young_entries",
    ),
    "lock_contention": (
        "tests/unit/test_index_lifecycle.py::test_lock_contention_fails_fast_and_bounded",),
    "layer_scanner": (
        "tests/unit/test_scan_image_layers.py::test_positive_negative_traversal_whiteout_fixtures",),
    "container_static_and_connectivity": (
        "tests/unit/test_container_smoke_contract.py::test_docker_run_argv_includes_add_host_and_embedding_seam_env",
        "tests/unit/test_container_smoke_contract.py::test_reachability_probe_argv_targets_mock_ping_via_host_gateway",
        "tests/unit/test_container_smoke_contract.py::test_negative_activation_argv_omits_test_seam_mount_and_pythonpath",
    ),
    "embedding_provider_seam_guard": (
        "tests/unit/test_settings_inventory.py::test_deterministic_embedding_provider_without_allow_flag_rejected",
        "tests/unit/test_rag_engine_embeddings.py::test_build_embeddings_default_uses_huggingface_provider",
        "tests/unit/test_rag_engine_embeddings.py::test_build_embeddings_raises_seam_unavailable_when_module_absent",
    ),
    "assemble_payload_verification": (
        "tests/unit/test_assemble_m4_evidence.py::test_positive_all_producers_ok",
        "tests/unit/test_assemble_m4_evidence.py::test_negative_control_matrix",
    ),
    "baseline_strict_schema": (
        "tests/unit/test_check_m4_baseline.py::test_strict_schema_and_algebra_matrix",),
})

M43_SCHEMA = "m43-acceptance-receipt-v1"


def collect_profile_nodes(node_ids: tuple[str, ...]) -> tuple[str, ...]:
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("duplicate profile node")
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *node_ids],
        text=True, capture_output=True,
    )
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)
    collected = tuple(line.strip() for line in run.stdout.splitlines() if "::test_" in line)
    if collected != node_ids:
        raise ValueError(f"profile inventory mismatch: {collected!r}")
    return node_ids


def _run_node(node_id: str) -> bool:
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", node_id],
                          text=True, capture_output=True)
    return run.returncode == 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_negative_control() -> dict:
    """Tampers a genuine producer receipt and asserts
    `assemble_m4_evidence.py::_check_identity` rejects it — same parser,
    real rejection (not a string-matching stand-in)."""
    sys.path.insert(0, str(_SCRIPTS_DIR))
    import assemble_m4_evidence as assembler

    genuine = {
        "schema": assembler.RECEIPT_SCHEMA, "job": "python-tests", "sha": "a" * 40,
        "run_id": "1", "run_attempt": "1", "workflow_path": ".github/workflows/ci.yml",
        "event_name": "pull_request", "semantic_status": "PASS",
        "payload_manifest_sha256": assembler._payload_manifest_sha256({}),
        "payloads": [],
    }

    class _Args:
        expected_sha = "b" * 40  # deliberately different from genuine["sha"]
        expected_run_id = "1"
        expected_run_attempt = "1"
        expected_workflow_path = ".github/workflows/ci.yml"
        expected_event = "pull_request"

    ok, _reason = assembler._check_identity(genuine, "python-tests", _Args())
    if ok:
        return {"executed": True, "expected_to_fail": True, "actual_exit_code": 0,
                "result": "TAMPERING_ACCEPTED_BUG"}
    # negative_control's schema is exactly M43_NEGATIVE_KEYS (Design.md
    # §10.1/§8.2-c) — the rejection reason itself is not part of that
    # contract, only the fact that rejection happened.
    return {"executed": True, "expected_to_fail": True, "actual_exit_code": 1,
            "result": "REJECTED_AS_EXPECTED"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("deterministic", "live"), default="deterministic")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4303)
    parser.add_argument("--output", default="m43.json")
    parser.add_argument("--inject-evidence-mismatch", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.profile == "live":
        receipt = {"status": "NOT_RUN", "live": {"reason": "opt-in environment required"},
                   "M4.1_BLOCKED": True}
        output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return 0

    started = _now()
    node_ids = tuple(nid for nodes in PROFILE_NODE_IDS.values() for nid in nodes)
    try:
        collect_profile_nodes(node_ids)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"acceptance node collection drift: {exc}") from exc

    nodes_result: dict[str, dict] = {}
    for node_name, node_ids_for_node in PROFILE_NODE_IDS.items():
        success_count = 0
        for _ in range(args.repeat):
            if all(_run_node(nid) for nid in node_ids_for_node):
                success_count += 1
        nodes_result[node_name] = {
            "repeat": args.repeat, "success_count": success_count,
            "status": "PASS" if success_count == args.repeat else "FAIL",
        }

    all_pass = all(n["status"] == "PASS" for n in nodes_result.values())

    if args.inject_evidence_mismatch:
        negative = _run_negative_control()
        status = "REJECTED_AS_EXPECTED" if negative["result"] == "REJECTED_AS_EXPECTED" else "TAMPERING_ACCEPTED_BUG"
    else:
        negative = {"executed": False, "expected_to_fail": None, "actual_exit_code": None, "result": None}
        status = "PASS" if all_pass else "FAIL"

    receipt = {
        "schema": M43_SCHEMA, "profile": "deterministic", "seed": args.seed,
        "repeat": args.repeat,
        "command": f"run_m43_acceptance.py --profile deterministic --repeat {args.repeat} --seed {args.seed}",
        "started_at": started, "finished_at": _now(),
        "nodes": nodes_result, "negative_control": negative, "status": status,
    }
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")

    if args.inject_evidence_mismatch:
        return 1 if status == "REJECTED_AS_EXPECTED" else 0
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
