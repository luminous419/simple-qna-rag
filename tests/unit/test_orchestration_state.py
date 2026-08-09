import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import orchestration_state as state  # noqa: E402


def initialized(tmp_path):
    payload = state.init_state(tmp_path, "run_test123", "m4.1", "term_coord", "runtime_a", 180)
    return payload, payload["lease"]["owner"]


def test_init_creates_valid_state_and_journal(tmp_path):
    payload, _ = initialized(tmp_path)
    assert payload["run_id"] == "run_test123"
    assert state.load_state(tmp_path, "run_test123")["scope"] == "m4.1"
    entries = state.journal_path(tmp_path, "run_test123").read_text().splitlines()
    assert json.loads(entries[0])["operation"] == "coordinator_started"


def test_init_refuses_existing_state(tmp_path):
    initialized(tmp_path)
    with pytest.raises(FileExistsError):
        initialized(tmp_path)


def test_run_id_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        state.state_dir(tmp_path, "../run_bad")


def test_active_lease_rejects_different_owner(tmp_path):
    initialized(tmp_path)
    with pytest.raises(PermissionError, match="active lease"):
        state.acquire_lease(tmp_path, "run_test123", "runtime_b:term_b", 180)


def test_concurrent_takeover_allows_only_one_owner(tmp_path):
    payload, _ = initialized(tmp_path)
    payload["lease"]["expires_at"] = state.isoformat(datetime.now(UTC) - timedelta(seconds=1))
    state.save_state(tmp_path, "run_test123", payload)

    def acquire(owner):
        try:
            return state.acquire_lease(tmp_path, "run_test123", owner, 180, force_expired=True)["lease"]["owner"]
        except PermissionError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(acquire, ["runtime_b:term_b", "runtime_c:term_c"]))
    assert len([result for result in results if result]) == 1


def test_expired_lease_can_be_taken_over_explicitly(tmp_path):
    payload, _ = initialized(tmp_path)
    payload["lease"]["expires_at"] = state.isoformat(datetime.now(UTC) - timedelta(seconds=1))
    state.save_state(tmp_path, "run_test123", payload)
    result = state.acquire_lease(tmp_path, "run_test123", "runtime_b:term_b", 180, force_expired=True)
    assert result["lease"]["owner"] == "runtime_b:term_b"


def test_expired_lease_is_not_implicitly_renewed(tmp_path):
    payload, owner = initialized(tmp_path)
    payload["lease"]["expires_at"] = state.isoformat(datetime.now(UTC) - timedelta(seconds=1))
    state.save_state(tmp_path, "run_test123", payload)
    with pytest.raises(PermissionError, match="expired"):
        state.heartbeat(tmp_path, "run_test123", owner, 180)


def test_checkpoint_forbids_identity_mutation(tmp_path):
    _, owner = initialized(tmp_path)
    with pytest.raises(ValueError, match="cannot modify"):
        state.checkpoint(tmp_path, "run_test123", owner, {"run_id": "run_other"})


def test_checkpoint_persists_transition_state(tmp_path):
    _, owner = initialized(tmp_path)
    result = state.checkpoint(tmp_path, "run_test123", owner, {"phase": "review", "active": [{"task_id": "task_1"}]})
    assert result["phase"] == "review"
    assert state.load_state(tmp_path, "run_test123")["active"][0]["task_id"] == "task_1"


def test_journal_requires_operation(tmp_path):
    initialized(tmp_path)
    with pytest.raises(ValueError, match="operation"):
        state.append_journal(tmp_path, "run_test123", {"outcome": "failed"})


def test_audit_detects_unfinished_gap(tmp_path):
    payload, _ = initialized(tmp_path)
    payload["successor"] = None
    payload["active"] = []
    state.save_state(tmp_path, "run_test123", payload)
    assert "unfinished_without_active_or_successor" in state.audit(tmp_path, "run_test123")["issues"]


def test_release_records_terminal_outcome(tmp_path):
    _, owner = initialized(tmp_path)
    result = state.release_lease(tmp_path, "run_test123", owner, "stopped")
    assert result["terminal_outcome"] == "stopped"


def test_corrupt_state_is_rejected(tmp_path):
    path = state.state_path(tmp_path, "run_test123")
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="missing"):
        state.load_state(tmp_path, "run_test123")
