import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import orchestration_state as state  # noqa: E402
import orchestration_watchdog as watchdog  # noqa: E402


class FakeRunner:
    def __init__(self, *, tasks=None, messages=None, connected=True):
        self.tasks = tasks or []
        self.messages = messages or []
        self.connected = connected
        self.commands = []

    def __call__(self, command):
        self.commands.append(command)
        joined = " ".join(command)
        if "task-list" in joined:
            return {"result": {"tasks": self.tasks}}
        if "check" in joined:
            return {"result": {"messages": self.messages}}
        if "terminal show" in joined:
            return {"result": {"terminal": {"connected": self.connected}}}
        if "terminal send" in joined:
            return {"result": {"sent": True}}
        raise AssertionError(command)


def setup_state(tmp_path):
    payload = state.init_state(tmp_path, "run_watch123", "m4.1", "term_coord", "runtime_a", 180)
    state.checkpoint(tmp_path, "run_watch123", payload["lease"]["owner"], {"active": [{"task_id": "task_1"}], "successor": {"role": "review"}})


def test_healthy_run_does_not_wake(tmp_path):
    setup_state(tmp_path)
    runner = FakeRunner(tasks=[{"status": "dispatched"}])
    result = watchdog.check_once(tmp_path, "run_watch123", "orca", runner=runner)
    assert result["ok"] is True
    assert result["woke"] is False


def test_unread_completion_wakes_coordinator(tmp_path):
    setup_state(tmp_path)
    runner = FakeRunner(tasks=[{"status": "dispatched"}], messages=[{"id": "msg_1", "type": "worker_done"}])
    result = watchdog.check_once(tmp_path, "run_watch123", "orca", runner=runner)
    assert result["woke"] is True
    assert any("terminal send" in " ".join(cmd) for cmd in runner.commands)


def test_same_anomaly_is_deduplicated(tmp_path):
    setup_state(tmp_path)
    runner = FakeRunner(tasks=[], messages=[])
    assert watchdog.check_once(tmp_path, "run_watch123", "orca", runner=runner)["woke"] is True
    assert watchdog.check_once(tmp_path, "run_watch123", "orca", runner=runner)["woke"] is False


def test_cleared_anomaly_can_wake_again_later(tmp_path):
    setup_state(tmp_path)
    bad = FakeRunner(tasks=[])
    watchdog.check_once(tmp_path, "run_watch123", "orca", runner=bad)
    watchdog.check_once(tmp_path, "run_watch123", "orca", runner=FakeRunner(tasks=[{"status": "dispatched"}]))
    assert watchdog.check_once(tmp_path, "run_watch123", "orca", runner=bad)["woke"] is True


def test_dry_run_never_sends(tmp_path):
    setup_state(tmp_path)
    runner = FakeRunner(tasks=[])
    result = watchdog.check_once(tmp_path, "run_watch123", "orca", dry_run=True, runner=runner)
    assert result["woke"] is False
    assert not any("terminal send" in " ".join(cmd) for cmd in runner.commands)


def test_readiness_test_wake_reaches_coordinator(tmp_path):
    setup_state(tmp_path)
    runner = FakeRunner(tasks=[{"status": "dispatched"}])
    result = watchdog.check_once(tmp_path, "run_watch123", "orca", test_wake=True, runner=runner)
    assert result["woke"] is True
    assert "readiness_test_wake" in result["observation"]["reasons"]


def test_disconnected_terminal_is_anomaly(tmp_path):
    setup_state(tmp_path)
    result = watchdog.check_once(tmp_path, "run_watch123", "orca", runner=FakeRunner(tasks=[{"status": "ready"}], connected=False), dry_run=True)
    assert "coordinator_terminal_disconnected" in result["observation"]["reasons"]


def test_anomaly_key_is_order_independent():
    assert watchdog.anomaly_key("run_x", ["b", "a"], "m") == watchdog.anomaly_key("run_x", ["a", "b"], "m")
