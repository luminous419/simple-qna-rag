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


# --- M4.3-REQ-009.3 exact 8-test contract (Design.md §11.2) ---------------


def test_task_list_uses_exact_bound_argv(tmp_path):
    setup_state(tmp_path)
    runner = FakeRunner(tasks=[{"status": "dispatched"}])
    watchdog.check_once(tmp_path, "run_watch123", "orca", runner=runner)
    task_list_cmd = next(c for c in runner.commands if "task-list" in c)
    assert task_list_cmd == [
        "orca", "orchestration", "task-list",
        "--run", "run_watch123", "--from", "term_coord",
        "--brief", "--json",
    ]


def test_check_uses_exact_bound_argv(tmp_path):
    setup_state(tmp_path)
    runner = FakeRunner(tasks=[{"status": "dispatched"}])
    watchdog.check_once(tmp_path, "run_watch123", "orca", runner=runner)
    check_cmd = next(c for c in runner.commands if c[2] == "check")
    assert check_cmd == [
        "orca", "orchestration", "check",
        "--terminal", "term_coord", "--run", "run_watch123",
        "--peek", "--json",
    ]


def test_check_always_includes_peek_flag(tmp_path):
    """--peek이 빠지면 조회가 소비형(consuming)으로 바뀐다 — 이 플래그가
    항상 존재함을 회귀 고정한다."""
    setup_state(tmp_path)
    runner = FakeRunner(tasks=[])
    watchdog.check_once(tmp_path, "run_watch123", "orca", runner=runner)
    check_cmd = next(c for c in runner.commands if c[2] == "check")
    assert "--peek" in check_cmd


class ScopedFakeRunner:
    def __init__(self, *, owned_terminal, tasks=None, owned_messages=None,
                 foreign_messages=None, connected=True):
        self.owned_terminal = owned_terminal
        self.tasks = tasks or []
        self.owned_messages = owned_messages or []
        self.foreign_messages = foreign_messages or []  # owned by a different terminal
        self.connected = connected
        self.commands = []

    def __call__(self, command):
        self.commands.append(command)
        if "task-list" in command:
            requested = command[command.index("--from") + 1]
            assert requested == self.owned_terminal, "task-list leaked cross-terminal scope"
            return {"result": {"tasks": self.tasks}}
        if command[2] == "check":
            requested = command[command.index("--terminal") + 1]
            assert requested == self.owned_terminal, "check leaked cross-terminal scope"
            return {"result": {"messages": self.owned_messages}}  # foreign_messages never returned
        if command[:2] == ["orca", "terminal"] and command[2] == "show":
            return {"result": {"terminal": {"connected": self.connected}}}
        if command[:2] == ["orca", "terminal"] and command[2] == "send":
            return {"result": {"sent": True}}
        raise AssertionError(command)


def test_check_is_terminal_scoped_and_ignores_foreign_messages(tmp_path):
    setup_state(tmp_path)
    runner = ScopedFakeRunner(
        owned_terminal="term_coord", tasks=[{"status": "dispatched"}],
        owned_messages=[],
        foreign_messages=[{"id": "msg_other", "type": "worker_done"}],
    )
    result = watchdog.check_once(tmp_path, "run_watch123", "orca", runner=runner)
    assert result["woke"] is False   # foreign_messages are not this run's evidence


def test_check_propagates_consumer_fenced_fail_closed_with_no_send(tmp_path):
    import pytest

    setup_state(tmp_path)
    calls = []

    def fenced_runner(command):
        calls.append(command)
        if "task-list" in command:
            assert command == [
                "orca", "orchestration", "task-list",
                "--run", "run_watch123", "--from", "term_coord",
                "--brief", "--json",
            ]
            raise RuntimeError(
                "command failed (2): orca orchestration task-list --run "
                "run_watch123 --from term_coord --brief --json: "
                "consumer_fenced: term_coord is not the active consumer for run run_watch123")
        raise AssertionError(command)  # no command after task-list failure (fail-closed)

    with pytest.raises(RuntimeError, match="consumer_fenced"):
        watchdog.check_once(tmp_path, "run_watch123", "orca", runner=fenced_runner)
    assert not any(c[:2] == ["orca", "terminal"] and c[2] == "send" for c in calls)  # no-send


def test_check_subcommand_exits_2_with_bounded_stderr_and_no_stdout_success(tmp_path, monkeypatch, capsys):
    setup_state(tmp_path)

    def fenced_run_json(command):
        raise RuntimeError(
            "command failed (2): orca orchestration check --terminal term_coord "
            "--run run_watch123 --peek --json: consumer_fenced: stale coordinator lease")

    monkeypatch.setattr(watchdog, "run_json", fenced_run_json)
    monkeypatch.chdir(tmp_path)
    code = watchdog.main(["--root", str(tmp_path), "--run-id", "run_watch123", "check"])
    assert code == 2
    captured = capsys.readouterr()
    assert '"ok": true' not in captured.out and '"ok":true' not in captured.out  # no-success
    assert '"error": "consumer_fenced"' in captured.err or \
           '"error":"consumer_fenced"' in captured.err  # bounded reason, not raw stderr
    assert "stale coordinator lease" not in captured.err  # original CLI stderr not leaked
    assert "Traceback" not in captured.err  # no stack trace


def test_run_loop_terminates_nonzero_after_consumer_fenced_with_exact_once_journal(tmp_path, monkeypatch):
    setup_state(tmp_path)
    calls = {"n": 0}

    def fenced_run_json(command):
        calls["n"] += 1
        if "task-list" in command:
            raise RuntimeError(
                "command failed (2): orca orchestration task-list --run run_watch123 "
                "--from term_coord --brief --json: consumer_fenced: stale lease")
        raise AssertionError(command)  # no command should run after task-list failure

    monkeypatch.setattr(watchdog, "run_json", fenced_run_json)
    monkeypatch.setattr(watchdog.time, "sleep", lambda _: None)
    exit_code = watchdog.run_loop(tmp_path, "run_watch123", "orca", interval=1)
    assert exit_code != 0   # fail-closed — not left on a success (exit 0) path
    assert calls["n"] == 1  # check_once was not invoked a second time (no retry)
    journal_text = (state.journal_path(tmp_path, "run_watch123")).read_text(encoding="utf-8")
    fenced_lines = [line for line in journal_text.splitlines()
                     if "consumer_fenced" in line and "watchdog_check" in line]
    assert len(fenced_lines) == 1   # journal exact-one — not accumulated per interval
    assert "stale lease" not in journal_text  # journal bounded reason — no raw stderr leak
    assert not (state.state_dir(tmp_path, "run_watch123") / "watchdog.stop").exists()


def test_bound_run_dry_run_receipt_is_durable(tmp_path):
    import os as _os

    setup_state(tmp_path)
    runner = FakeRunner(tasks=[], messages=[{"id": "msg_1", "type": "worker_done"}])
    result = watchdog.check_once(tmp_path, "run_watch123", "orca", dry_run=True, runner=runner)
    assert result["dry_run"] is True
    assert result["woke"] is False   # dry_run never sends (existing contract)
    on_disk = watchdog.load_watchdog(
        state.state_dir(tmp_path, "run_watch123") / "watchdog_state.json")
    assert on_disk["last_check"] is not None
    assert on_disk["pid"] == _os.getpid()
