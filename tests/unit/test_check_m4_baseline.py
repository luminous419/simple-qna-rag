"""M4.3-REQ-008 — baseline strict schema + producer->gate algebra recompute
(Design.md §9.2), plus M4 Operational Acceptance Recovery v1/v2 dispatch,
frozen-blocked legacy enforcement, and identity/alias binding
(docs/milestones/m4-operational-acceptance-recovery/Design.md §4, §7.2)."""

from __future__ import annotations

import copy

import pytest

from scripts import check_m4_baseline as checker


def _ok_producer(payload_hashes: dict[str, str]) -> dict:
    return {
        "status": "OK",
        "receipt_sha256": "a" * 64,
        "payload_hashes": payload_hashes,
        "payload_manifest_sha256": checker._payload_manifest_sha256(payload_hashes),
    }


def _valid_v1_legacy_candidate() -> dict:
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


def test_v1_legacy_strict_schema_and_algebra_matrix():
    valid = _valid_v1_legacy_candidate()
    ok, issues = checker.check(valid, allow_legacy_v1=True, expect_operational_blocked=True)
    assert ok, issues

    # (a) missing top-level key
    missing_key = dict(valid)
    del missing_key["settings_hash"]
    ok, issues = checker.check(missing_key, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok and issues[0].startswith("top_level_key_mismatch")

    # (b) extra top-level key
    extra_key = dict(valid)
    extra_key["extra"] = 1
    ok, issues = checker.check(extra_key, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok and issues[0].startswith("top_level_key_mismatch")

    # (c) M4.1_BLOCKED as string "true"
    bad_bool = copy.deepcopy(valid)
    bad_bool["M4.1_BLOCKED"] = "true"
    ok, issues = checker.check(bad_bool, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok and any("m41_blocked_not_true" in i or "non_boolean_field" in i for i in issues)

    # (d) deterministic_status PASS but gates.container FAIL (self-contradiction)
    contradiction = copy.deepcopy(valid)
    contradiction["gates"]["container"] = "FAIL"
    ok, issues = checker.check(contradiction, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok and any("gate_producer_algebra_mismatch" in i for i in issues)

    # (e) producers={} with gates all PASS
    empty_producers = copy.deepcopy(valid)
    empty_producers["producers"] = {}
    ok, issues = checker.check(empty_producers, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok and issues == ["producer_key_set_mismatch"]

    # (f) all producers MISSING but gates say PASS
    all_missing = copy.deepcopy(valid)
    for job in all_missing["producers"]:
        all_missing["producers"][job] = {"status": "MISSING"}
    ok, issues = checker.check(all_missing, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok
    assert sum("gate_producer_algebra_mismatch" in i for i in issues) == 4

    # (g) minimal {"status": "OK"} producer entries (DR-I3-MAJ-04 exact reproduction)
    minimal_ok = copy.deepcopy(valid)
    for job in minimal_ok["producers"]:
        minimal_ok["producers"][job] = {"status": "OK"}
    ok, issues = checker.check(minimal_ok, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok
    assert all("producer_variant_schema_mismatch" in i for i in issues)

    # (h) receipt_sha256 wrong length
    bad_hash = copy.deepcopy(valid)
    bad_hash["producers"]["python-tests"]["receipt_sha256"] = "a" * 32
    ok, issues = checker.check(bad_hash, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok and any("producer_receipt_sha256_malformed" in i for i in issues)

    # (i) container payload_hashes missing one required filename
    missing_filename = copy.deepcopy(valid)
    missing_filename["producers"]["container"]["payload_hashes"] = {"layer_scan.json": "b" * 64}
    missing_filename["producers"]["container"]["payload_manifest_sha256"] = checker._payload_manifest_sha256(
        {"layer_scan.json": "b" * 64})
    ok, issues = checker.check(missing_filename, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok and any("producer_payload_filename_set_mismatch" in i for i in issues)

    # (j) same-count filename substitution (DR-I4-MAJ-02)
    substituted = copy.deepcopy(valid)
    substituted["producers"]["container"]["payload_hashes"] = {"a.json": "b" * 64, "b.json": "c" * 64}
    substituted["producers"]["container"]["payload_manifest_sha256"] = checker._payload_manifest_sha256(
        {"a.json": "b" * 64, "b.json": "c" * 64})
    ok, issues = checker.check(substituted, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok and any("producer_payload_filename_set_mismatch" in i for i in issues)

    # (k) cross-job filename swap
    swapped = copy.deepcopy(valid)
    swapped["producers"]["m43-deterministic"]["payload_hashes"] = {
        "layer_scan.json": "b" * 64, "container_smoke.json": "c" * 64}
    swapped["producers"]["m43-deterministic"]["payload_manifest_sha256"] = checker._payload_manifest_sha256(
        {"layer_scan.json": "b" * 64, "container_smoke.json": "c" * 64})
    ok, issues = checker.check(swapped, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok and any("producer_payload_filename_set_mismatch" in i for i in issues)

    # (l) payload_manifest_sha256 malformed length
    malformed_manifest = copy.deepcopy(valid)
    malformed_manifest["producers"]["container"]["payload_manifest_sha256"] = "b" * 32
    ok, issues = checker.check(malformed_manifest, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok and any("producer_payload_manifest_sha256_malformed" in i for i in issues)

    # (m) payload_manifest_sha256 mismatch (hashes changed, manifest hash stale)
    stale_manifest = copy.deepcopy(valid)
    stale_manifest["producers"]["container"]["payload_hashes"] = {
        "layer_scan.json": "9" * 64, "container_smoke.json": "c" * 64}
    ok, issues = checker.check(stale_manifest, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok and any("producer_payload_manifest_sha256_mismatch" in i for i in issues)

    # (n) overall_release_ready True while operational BLOCKED must never validate
    fake_ready = copy.deepcopy(valid)
    fake_ready["overall_release_ready"] = True
    ok, issues = checker.check(fake_ready, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok
    assert any("overall_release_ready_algebra_mismatch" in i or
               "v1_legacy_overall_release_ready_not_false" in i or
               "expected_operational_blocked_not_satisfied" in i for i in issues)


# --- DR-I1-MAJ-01: frozen-blocked is enforced by --allow-legacy-v1 ALONE ---


@pytest.mark.parametrize("mutate,expected_issue", [
    (lambda c: c["gates"].__setitem__("m3_live_regression", "PASS"),
     "v1_legacy_live_regression_not_not_run"),
    (lambda c: c["gates"].__setitem__("m3_live_regression", "SKIPPED"),
     "v1_legacy_live_regression_not_not_run"),
    (lambda c: c["gates"].__setitem__("m41_operational", "PASS"),
     "v1_legacy_m41_operational_not_blocked"),
    (lambda c: c.__setitem__("M4.1_BLOCKED", False),
     "v1_legacy_m41_blocked_not_true"),
    (lambda c: c.__setitem__("overall_release_ready", True),
     "v1_legacy_overall_release_ready_not_false"),
    (lambda c: c.__setitem__("operational_status", "PASS"),
     "v1_legacy_operational_status_not_blocked"),
])
def test_v1_legacy_rejects_frozen_state_mutants_without_expect_flag(mutate, expected_issue):
    candidate = copy.deepcopy(_valid_v1_legacy_candidate())
    mutate(candidate)
    ok, issues = checker.check(candidate, allow_legacy_v1=True)
    assert not ok
    assert any(expected_issue in i for i in issues)


def test_v1_legacy_accepts_frozen_state_with_allow_legacy_v1_alone():
    ok, issues = checker.check(_valid_v1_legacy_candidate(), allow_legacy_v1=True)
    assert ok, issues


# --- v2 fixture and tests --------------------------------------------------


def _valid_v2_candidate() -> dict:
    container_hashes = {"layer_scan.json": "b" * 64, "container_smoke.json": "c" * 64}
    m43_hashes = {"m43.json": "d" * 64, "m43-negative.json": "e" * 64}
    producers = {
        "python-tests": _ok_producer({}),
        "frontend-tests": _ok_producer({}),
        "container": _ok_producer(container_hashes),
        "m43-deterministic": _ok_producer(m43_hashes),
    }
    return {
        "schema": "m4-baseline-v2", "schema_version": "2.0.0", "generated_at": "2026-08-15T00:00:00Z",
        "git_sha": "f" * 40,
        "workflow_run": {"run_id": "12345", "run_attempt": "1",
                          "workflow_path": ".github/workflows/ci.yml", "event_name": "push"},
        "m3_fingerprint_reference": "evaluation/baselines/m3_initial.json",
        "dependency_snapshot_sha256": "0" * 64, "settings_hash": "0" * 64,
        "image_digest": container_hashes["container_smoke.json"],
        "m43_deterministic_receipt_sha256": m43_hashes["m43.json"],
        "producers": producers,
        "gates": {
            "python_tests": "PASS", "frontend_tests": "PASS", "container": "PASS",
            "m43_deterministic": "PASS", "m3_live_regression": "NOT_ADOPTED", "m41_operational": "NOT_ADOPTED",
        },
        "deterministic_status": "PASS",
        "support_policy": {
            "schema": "m4-support-policy-v1", "adopted_scope": "HOSTED_OCI",
            "native_linux_ollama": "NOT_ADOPTED", "decision_date": "2026-08-15",
        },
        "operational_status": "NOT_ADOPTED", "M4.1_BLOCKED": False,
        "hosted_release_ready": True, "native_linux_release_ready": False,
        "full_production_release_ready": False, "overall_release_ready": False,
    }


def test_v2_valid_candidate_passes():
    ok, issues = checker.check(_valid_v2_candidate())
    assert ok, issues


@pytest.mark.parametrize("mutate", [
    lambda c: c.__delitem__("settings_hash"),
    lambda c: c.__setitem__("extra", 1),
])
def test_v2_rejects_missing_or_extra_top_level_key(mutate):
    candidate = copy.deepcopy(_valid_v2_candidate())
    mutate(candidate)
    ok, issues = checker.check(candidate)
    assert not ok and issues[0].startswith("top_level_key_mismatch")


def test_v2_rejects_live_regression_pass_substituted_for_not_adopted():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["gates"]["m3_live_regression"] = "PASS"
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("gate_not_adopted_fixed_value_violation" in i for i in issues)


def test_v2_rejects_m41_operational_pass_substituted_for_not_adopted():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["gates"]["m41_operational"] = "PASS"
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("gate_not_adopted_fixed_value_violation" in i for i in issues)


@pytest.mark.parametrize("key", ["python_tests", "m3_live_regression"])
def test_v2_rejects_waived_value_anywhere_in_gates(key):
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["gates"][key] = "WAIVED"
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("unknown_gate_enum_v2" in i for i in issues)


def test_v2_rejects_gate_producer_algebra_mismatch():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["gates"]["container"] = "FAIL"
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("gate_producer_algebra_mismatch" in i for i in issues)


def test_v2_rejects_deterministic_status_mismatch():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["deterministic_status"] = "FAIL"
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("deterministic_status_algebra_mismatch" in i for i in issues)


@pytest.mark.parametrize("failing_job", [None, "python-tests", "frontend-tests", "container", "m43-deterministic"])
def test_v2_hosted_release_ready_algebra_matrix(failing_job):
    candidate = copy.deepcopy(_valid_v2_candidate())
    if failing_job is None:
        ok, issues = checker.check(candidate)
        assert ok, issues
        return
    candidate["producers"][failing_job] = {"status": "MISSING"}
    gate_key = checker.PRODUCER_TO_GATE_KEY[failing_job]
    candidate["gates"][gate_key] = "FAIL"
    candidate["deterministic_status"] = "FAIL"
    candidate["hosted_release_ready"] = False
    if failing_job == "container":
        candidate["image_digest"] = None
    if failing_job == "m43-deterministic":
        candidate["m43_deterministic_receipt_sha256"] = None
    ok, issues = checker.check(candidate)
    assert ok, issues


def test_v2_rejects_hosted_release_ready_self_report_disagreeing_with_producers():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["producers"]["container"] = {"status": "MISSING"}
    candidate["gates"]["container"] = "FAIL"
    candidate["deterministic_status"] = "FAIL"
    candidate["image_digest"] = None
    # hosted_release_ready still self-reports True over the failed producer.
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("hosted_release_ready_algebra_mismatch" in i for i in issues)


def test_v2_rejects_true_native_linux_release_ready():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["native_linux_release_ready"] = True
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("native_linux_release_ready_not_false" in i for i in issues)


def test_v2_rejects_true_full_production_release_ready():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["full_production_release_ready"] = True
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("full_production_release_ready_not_false" in i for i in issues)


def test_v2_rejects_overall_release_ready_disagreeing_with_full_production():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["overall_release_ready"] = True
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("overall_release_ready_alias_mismatch" in i for i in issues)


@pytest.mark.parametrize("bad_value", ["PASS", "BLOCKED"])
def test_v2_rejects_operational_status_not_equal_not_adopted(bad_value):
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["operational_status"] = bad_value
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("operational_status_not_not_adopted" in i for i in issues)


def test_v2_rejects_m41_blocked_true():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["M4.1_BLOCKED"] = True
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("m41_blocked_not_false" in i for i in issues)


@pytest.mark.parametrize("field,bad_value", [
    ("schema", "m4-support-policy-v2"),
    ("adopted_scope", "HOSTED"),
    ("decision_date", "2026-01-01"),
    ("native_linux_ollama", "PASS"),
])
def test_v2_rejects_support_policy_wrong_schema_or_scope_or_date_or_native_ollama(field, bad_value):
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["support_policy"][field] = bad_value
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("support_policy_field_mismatch" in i for i in issues)


def test_v2_rejects_support_policy_extra_or_missing_key():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["support_policy"]["extra"] = "x"
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("support_policy_key_set_mismatch" in i for i in issues)


@pytest.mark.parametrize("bad_value", [None, "", 123])
def test_v2_rejects_git_sha_not_nonempty_string(bad_value):
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["git_sha"] = bad_value
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("git_sha_not_nonempty_string" in i for i in issues)


@pytest.mark.parametrize("mutate", [
    lambda wr: wr.__delitem__("run_id"),
    lambda wr: wr.__setitem__("extra", "x"),
])
def test_v2_rejects_workflow_run_key_set_mismatch(mutate):
    candidate = copy.deepcopy(_valid_v2_candidate())
    mutate(candidate["workflow_run"])
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("workflow_run_key_set_mismatch" in i for i in issues)


@pytest.mark.parametrize("field", ["run_id", "run_attempt", "workflow_path", "event_name"])
@pytest.mark.parametrize("bad_value", [None, 123, ""])
def test_v2_rejects_workflow_run_field_not_nonempty_string(field, bad_value):
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["workflow_run"][field] = bad_value
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("workflow_run_field_not_nonempty_string" in i for i in issues)


def test_v2_identity_flags_absent_by_default_no_regression():
    candidate = _valid_v2_candidate()
    ok, issues = checker.check(candidate)
    assert ok, issues


def test_v2_identity_flags_reject_cross_sha_mismatch():
    candidate = _valid_v2_candidate()
    ok, issues = checker.check(candidate, expect_sha="0" * 40)
    assert not ok
    assert any("identity_sha_mismatch" in i for i in issues)


def test_v2_identity_flags_reject_cross_run_id_mismatch():
    candidate = _valid_v2_candidate()
    ok, issues = checker.check(candidate, expect_run_id="99999")
    assert not ok
    assert any("identity_run_id_mismatch" in i for i in issues)


def test_v2_identity_flags_reject_cross_run_attempt_mismatch():
    candidate = _valid_v2_candidate()
    ok, issues = checker.check(candidate, expect_run_attempt="9")
    assert not ok
    assert any("identity_run_attempt_mismatch" in i for i in issues)


def test_v2_identity_flags_reject_cross_workflow_path_mismatch():
    candidate = _valid_v2_candidate()
    ok, issues = checker.check(candidate, expect_workflow_path=".github/workflows/other.yml")
    assert not ok
    assert any("identity_workflow_path_mismatch" in i for i in issues)


def test_v2_identity_flags_reject_cross_event_mismatch():
    candidate = _valid_v2_candidate()
    ok, issues = checker.check(candidate, expect_event="pull_request")
    assert not ok
    assert any("identity_event_mismatch" in i for i in issues)


def test_v2_identity_flags_accept_matching_identity():
    candidate = _valid_v2_candidate()
    ok, issues = checker.check(
        candidate, expect_sha=candidate["git_sha"], expect_run_id=candidate["workflow_run"]["run_id"],
        expect_run_attempt=candidate["workflow_run"]["run_attempt"],
        expect_workflow_path=candidate["workflow_run"]["workflow_path"],
        expect_event=candidate["workflow_run"]["event_name"])
    assert ok, issues


def test_v2_rejects_image_digest_alias_tampered_while_container_ok():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["image_digest"] = "9" * 64
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("image_digest_alias_mismatch" in i for i in issues)


def test_v2_rejects_image_digest_not_null_when_container_not_ok():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["producers"]["container"] = {"status": "MISSING"}
    candidate["gates"]["container"] = "FAIL"
    candidate["deterministic_status"] = "FAIL"
    candidate["hosted_release_ready"] = False
    # image_digest still self-reports a hex64 value despite the failed producer.
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("image_digest_alias_mismatch" in i for i in issues)


def test_v2_rejects_m43_receipt_sha_alias_tampered_while_m43_deterministic_ok():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["m43_deterministic_receipt_sha256"] = "9" * 64
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("m43_deterministic_receipt_sha256_alias_mismatch" in i for i in issues)


def test_v2_rejects_m43_receipt_sha_not_null_when_m43_deterministic_not_ok():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["producers"]["m43-deterministic"] = {"status": "MISSING"}
    candidate["gates"]["m43_deterministic"] = "FAIL"
    candidate["deterministic_status"] = "FAIL"
    candidate["hosted_release_ready"] = False
    ok, issues = checker.check(candidate)
    assert not ok
    assert any("m43_deterministic_receipt_sha256_alias_mismatch" in i for i in issues)


def test_v2_expect_hosted_release_ready_flag_satisfied_and_not_satisfied():
    candidate = _valid_v2_candidate()
    ok, issues = checker.check(candidate, expect_hosted_release_ready=True)
    assert ok, issues

    not_ready = copy.deepcopy(candidate)
    not_ready["producers"]["container"] = {"status": "MISSING"}
    not_ready["gates"]["container"] = "FAIL"
    not_ready["deterministic_status"] = "FAIL"
    not_ready["hosted_release_ready"] = False
    not_ready["image_digest"] = None
    ok, issues = checker.check(not_ready, expect_hosted_release_ready=True)
    assert not ok
    assert any("expected_hosted_release_ready_not_satisfied" in i for i in issues)


def test_v2_expect_hosted_not_ready_flag_satisfied_and_not_satisfied():
    ready = _valid_v2_candidate()
    ok, issues = checker.check(ready, expect_hosted_not_ready=True)
    assert not ok
    assert any("expected_hosted_not_ready_not_satisfied" in i for i in issues)

    not_ready = copy.deepcopy(ready)
    not_ready["producers"]["container"] = {"status": "MISSING"}
    not_ready["gates"]["container"] = "FAIL"
    not_ready["deterministic_status"] = "FAIL"
    not_ready["hosted_release_ready"] = False
    not_ready["image_digest"] = None
    ok, issues = checker.check(not_ready, expect_hosted_not_ready=True)
    assert ok, issues


def test_v1_candidate_rejected_without_allow_legacy_v1():
    ok, issues = checker.check(_valid_v1_legacy_candidate())
    assert not ok
    assert issues == ["legacy_v1_schema_requires_allow_legacy_v1_flag"]


def test_v1_candidate_with_v2_only_field_injected_still_rejected_under_allow_legacy_v1():
    candidate = copy.deepcopy(_valid_v1_legacy_candidate())
    candidate["hosted_release_ready"] = True
    ok, issues = checker.check(candidate, allow_legacy_v1=True, expect_operational_blocked=True)
    assert not ok
    assert issues[0].startswith("top_level_key_mismatch")


def test_unknown_schema_string_rejected():
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["schema"] = "m4-baseline-v3"
    ok, issues = checker.check(candidate)
    assert not ok
    assert issues == ["unknown_or_unsupported_schema"]


@pytest.mark.parametrize("schema,version", [("m4-baseline-v2", "1.0.0"), ("m4-baseline-v1", "2.0.0")])
def test_mismatched_schema_version_pair_rejected(schema, version):
    candidate = copy.deepcopy(_valid_v2_candidate())
    candidate["schema"] = schema
    candidate["schema_version"] = version
    ok, issues = checker.check(candidate)
    assert not ok
    assert issues == ["unknown_or_unsupported_schema"]


# --- CLI combination rules --------------------------------------------------


def _write_candidate(tmp_path, candidate) -> str:
    import json
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    return str(path)


def test_main_cli_expect_operational_blocked_without_allow_legacy_v1_exits_2(tmp_path):
    path = _write_candidate(tmp_path, _valid_v1_legacy_candidate())
    with pytest.raises(SystemExit) as exc_info:
        checker.main(["--candidate", path, "--expect-operational-blocked"])
    assert exc_info.value.code == 2


def test_main_cli_allow_legacy_v1_with_expect_hosted_release_ready_exits_2(tmp_path):
    path = _write_candidate(tmp_path, _valid_v1_legacy_candidate())
    with pytest.raises(SystemExit) as exc_info:
        checker.main(["--candidate", path, "--allow-legacy-v1", "--expect-hosted-release-ready"])
    assert exc_info.value.code == 2


def test_main_cli_both_hosted_expectation_flags_exits_2(tmp_path):
    path = _write_candidate(tmp_path, _valid_v2_candidate())
    with pytest.raises(SystemExit) as exc_info:
        checker.main(["--candidate", path, "--expect-hosted-release-ready", "--expect-hosted-not-ready"])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("extra_flag", [
    "--require-identity-binding", "--expect-sha=" + "a" * 40, "--expect-run-id=1",
    "--expect-run-attempt=1", "--expect-workflow-path=.github/workflows/ci.yml", "--expect-event=push",
])
def test_main_cli_allow_legacy_v1_with_require_identity_binding_or_expect_sha_exits_2(tmp_path, extra_flag):
    path = _write_candidate(tmp_path, _valid_v1_legacy_candidate())
    with pytest.raises(SystemExit) as exc_info:
        checker.main(["--candidate", path, "--allow-legacy-v1", extra_flag])
    assert exc_info.value.code == 2


def test_main_cli_require_identity_binding_without_expect_hosted_release_ready_exits_2(tmp_path):
    path = _write_candidate(tmp_path, _valid_v2_candidate())
    with pytest.raises(SystemExit) as exc_info:
        checker.main(["--candidate", path, "--require-identity-binding"])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("missing_flag", [
    "--expect-sha", "--expect-run-id", "--expect-run-attempt", "--expect-workflow-path", "--expect-event",
])
def test_main_cli_require_identity_binding_without_identity_flags_exits_2(tmp_path, missing_flag):
    candidate = _valid_v2_candidate()
    path = _write_candidate(tmp_path, candidate)
    all_flags = {
        "--expect-sha": candidate["git_sha"], "--expect-run-id": candidate["workflow_run"]["run_id"],
        "--expect-run-attempt": candidate["workflow_run"]["run_attempt"],
        "--expect-workflow-path": candidate["workflow_run"]["workflow_path"],
        "--expect-event": candidate["workflow_run"]["event_name"],
    }
    del all_flags[missing_flag]
    argv = ["--candidate", path, "--expect-hosted-release-ready", "--require-identity-binding"]
    for flag, value in all_flags.items():
        argv += [flag, value]
    with pytest.raises(SystemExit) as exc_info:
        checker.main(argv)
    assert exc_info.value.code == 2


def test_main_cli_v2_candidate_no_expectation_flags_exits_0(tmp_path):
    path = _write_candidate(tmp_path, _valid_v2_candidate())
    assert checker.main(["--candidate", path]) == 0


@pytest.mark.parametrize("ready,expected_exit", [(True, 0), (False, 1)])
def test_main_cli_expect_hosted_release_ready_alone_exits_0_without_identity_flags(tmp_path, ready, expected_exit):
    candidate = _valid_v2_candidate()
    if not ready:
        candidate["producers"]["container"] = {"status": "MISSING"}
        candidate["gates"]["container"] = "FAIL"
        candidate["deterministic_status"] = "FAIL"
        candidate["hosted_release_ready"] = False
        candidate["image_digest"] = None
    path = _write_candidate(tmp_path, candidate)
    exit_code = checker.main(["--candidate", path, "--expect-hosted-release-ready"])
    assert exit_code == expected_exit


def test_main_cli_require_identity_binding_exits_0_when_identity_and_hosted_ready_match(tmp_path):
    candidate = _valid_v2_candidate()
    path = _write_candidate(tmp_path, candidate)
    argv = ["--candidate", path, "--expect-hosted-release-ready", "--require-identity-binding",
            "--expect-sha", candidate["git_sha"], "--expect-run-id", candidate["workflow_run"]["run_id"],
            "--expect-run-attempt", candidate["workflow_run"]["run_attempt"],
            "--expect-workflow-path", candidate["workflow_run"]["workflow_path"],
            "--expect-event", candidate["workflow_run"]["event_name"]]
    assert checker.main(argv) == 0


def test_main_cli_require_identity_binding_exits_1_on_cross_sha(tmp_path):
    candidate = _valid_v2_candidate()
    path = _write_candidate(tmp_path, candidate)
    argv = ["--candidate", path, "--expect-hosted-release-ready", "--require-identity-binding",
            "--expect-sha", "0" * 40, "--expect-run-id", candidate["workflow_run"]["run_id"],
            "--expect-run-attempt", candidate["workflow_run"]["run_attempt"],
            "--expect-workflow-path", candidate["workflow_run"]["workflow_path"],
            "--expect-event", candidate["workflow_run"]["event_name"]]
    assert checker.main(argv) == 1
