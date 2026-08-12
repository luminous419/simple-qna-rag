"""M4.3-REQ-006 — mock deploy/recovery drill halts fault injection correctly."""

from __future__ import annotations

from scripts import deploy_drill


def test_drill_preserves_identity_and_halts_all_faults(tmp_path):
    result = deploy_drill.run_drill(tmp_path / "root", repeat=2)
    assert result["identity_preserved"] is True
    for fault in result["fault_injection"]:
        assert fault.get("current_unchanged", True) is True
    lock_fault = next(f for f in result["fault_injection"] if f["fault"] == "lock_contention")
    assert lock_fault["contention_observed"] is True
    settings_fault = next(
        f for f in result["fault_injection"] if f["fault"] == "readiness_settings_mismatch")
    assert settings_fault["error_code"] == "settings_mismatch"
