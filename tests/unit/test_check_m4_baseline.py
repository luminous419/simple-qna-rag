"""M4.3-REQ-008 — baseline strict schema + producer->gate algebra recompute
(Design.md §9.2)."""

from __future__ import annotations

import copy
import hashlib

import pytest

from scripts import check_m4_baseline as checker


def _ok_producer(payload_hashes: dict[str, str]) -> dict:
    return {
        "status": "OK",
        "receipt_sha256": "a" * 64,
        "payload_hashes": payload_hashes,
        "payload_manifest_sha256": checker._payload_manifest_sha256(payload_hashes),
    }


def _valid_candidate() -> dict:
    producers = {
        "python-tests": _ok_producer({}),
        "frontend-tests": _ok_producer({}),
        "container": _ok_producer({"layer_scan.json": "b" * 64, "container_smoke.json": "c" * 64}),
        "m43-deterministic": _ok_producer({"m43.json": "d" * 64, "m43-negative.json": "e" * 64}),
    }
    return {
        "schema": "m4-baseline-v1", "schema_version": "1.0.0", "generated_at": "2026-08-12T00:00:00Z",
        "git_sha": "f" * 40,
        "workflow_run": {"run_id": "1", "run_attempt": "1",
                          "workflow_path": ".github/workflows/ci.yml", "event_name": "pull_request"},
        "m3_fingerprint_reference": "evaluation/baselines/m3_initial.json",
        "dependency_snapshot_sha256": "0" * 64, "settings_hash": "0" * 64,
        "image_digest": None, "m43_deterministic_receipt_sha256": "d" * 64,
        "producers": producers,
        "gates": {
            "python_tests": "PASS", "frontend_tests": "PASS", "container": "PASS",
            "m43_deterministic": "PASS", "m3_live_regression": "NOT_RUN", "m41_operational": "BLOCKED",
        },
        "deterministic_status": "PASS", "operational_status": "BLOCKED",
        "M4.1_BLOCKED": True, "overall_release_ready": False,
    }


def test_strict_schema_and_algebra_matrix():
    valid = _valid_candidate()
    ok, issues = checker.check(valid, expect_operational_blocked=True)
    assert ok, issues

    # (a) missing top-level key
    missing_key = dict(valid)
    del missing_key["settings_hash"]
    ok, issues = checker.check(missing_key, expect_operational_blocked=True)
    assert not ok and issues[0].startswith("top_level_key_mismatch")

    # (b) extra top-level key
    extra_key = dict(valid)
    extra_key["extra"] = 1
    ok, issues = checker.check(extra_key, expect_operational_blocked=True)
    assert not ok and issues[0].startswith("top_level_key_mismatch")

    # (c) M4.1_BLOCKED as string "true"
    bad_bool = copy.deepcopy(valid)
    bad_bool["M4.1_BLOCKED"] = "true"
    ok, issues = checker.check(bad_bool, expect_operational_blocked=True)
    assert not ok and any("non_boolean_field" in i for i in issues)

    # (d) deterministic_status PASS but gates.container FAIL (self-contradiction)
    contradiction = copy.deepcopy(valid)
    contradiction["gates"]["container"] = "FAIL"
    ok, issues = checker.check(contradiction, expect_operational_blocked=True)
    assert not ok and any("gate_producer_algebra_mismatch" in i for i in issues)

    # (e) producers={} with gates all PASS
    empty_producers = copy.deepcopy(valid)
    empty_producers["producers"] = {}
    ok, issues = checker.check(empty_producers, expect_operational_blocked=True)
    assert not ok and issues == ["producer_key_set_mismatch"]

    # (f) all producers MISSING but gates say PASS
    all_missing = copy.deepcopy(valid)
    for job in all_missing["producers"]:
        all_missing["producers"][job] = {"status": "MISSING"}
    ok, issues = checker.check(all_missing, expect_operational_blocked=True)
    assert not ok
    assert sum("gate_producer_algebra_mismatch" in i for i in issues) == 4

    # (g) minimal {"status": "OK"} producer entries (DR-I3-MAJ-04 exact reproduction)
    minimal_ok = copy.deepcopy(valid)
    for job in minimal_ok["producers"]:
        minimal_ok["producers"][job] = {"status": "OK"}
    ok, issues = checker.check(minimal_ok, expect_operational_blocked=True)
    assert not ok
    assert all("producer_variant_schema_mismatch" in i for i in issues)

    # (h) receipt_sha256 wrong length
    bad_hash = copy.deepcopy(valid)
    bad_hash["producers"]["python-tests"]["receipt_sha256"] = "a" * 32
    ok, issues = checker.check(bad_hash, expect_operational_blocked=True)
    assert not ok and any("producer_receipt_sha256_malformed" in i for i in issues)

    # (i) container payload_hashes missing one required filename
    missing_filename = copy.deepcopy(valid)
    missing_filename["producers"]["container"]["payload_hashes"] = {"layer_scan.json": "b" * 64}
    missing_filename["producers"]["container"]["payload_manifest_sha256"] = checker._payload_manifest_sha256(
        {"layer_scan.json": "b" * 64})
    ok, issues = checker.check(missing_filename, expect_operational_blocked=True)
    assert not ok and any("producer_payload_filename_set_mismatch" in i for i in issues)

    # (j) same-count filename substitution (DR-I4-MAJ-02)
    substituted = copy.deepcopy(valid)
    substituted["producers"]["container"]["payload_hashes"] = {"a.json": "b" * 64, "b.json": "c" * 64}
    substituted["producers"]["container"]["payload_manifest_sha256"] = checker._payload_manifest_sha256(
        {"a.json": "b" * 64, "b.json": "c" * 64})
    ok, issues = checker.check(substituted, expect_operational_blocked=True)
    assert not ok and any("producer_payload_filename_set_mismatch" in i for i in issues)

    # (k) cross-job filename swap
    swapped = copy.deepcopy(valid)
    swapped["producers"]["m43-deterministic"]["payload_hashes"] = {
        "layer_scan.json": "b" * 64, "container_smoke.json": "c" * 64}
    swapped["producers"]["m43-deterministic"]["payload_manifest_sha256"] = checker._payload_manifest_sha256(
        {"layer_scan.json": "b" * 64, "container_smoke.json": "c" * 64})
    ok, issues = checker.check(swapped, expect_operational_blocked=True)
    assert not ok and any("producer_payload_filename_set_mismatch" in i for i in issues)

    # (l) payload_manifest_sha256 malformed length
    malformed_manifest = copy.deepcopy(valid)
    malformed_manifest["producers"]["container"]["payload_manifest_sha256"] = "b" * 32
    ok, issues = checker.check(malformed_manifest, expect_operational_blocked=True)
    assert not ok and any("producer_payload_manifest_sha256_malformed" in i for i in issues)

    # (m) payload_manifest_sha256 mismatch (hashes changed, manifest hash stale)
    stale_manifest = copy.deepcopy(valid)
    stale_manifest["producers"]["container"]["payload_hashes"] = {
        "layer_scan.json": "9" * 64, "container_smoke.json": "c" * 64}
    ok, issues = checker.check(stale_manifest, expect_operational_blocked=True)
    assert not ok and any("producer_payload_manifest_sha256_mismatch" in i for i in issues)

    # (n) overall_release_ready True while operational BLOCKED must never validate
    fake_ready = copy.deepcopy(valid)
    fake_ready["overall_release_ready"] = True
    ok, issues = checker.check(fake_ready, expect_operational_blocked=True)
    assert not ok
    assert any("overall_release_ready_algebra_mismatch" in i or
               "expected_operational_blocked_not_satisfied" in i for i in issues)
