#!/usr/bin/env python3
"""Wake a durable Orca coordinator when its Run needs a resume audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import orchestration_state as state

CommandRunner = Callable[[list[str]], dict[str, Any]]


def run_json(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def anomaly_key(run_id: str, reasons: list[str], last_delivery: str | None) -> str:
    raw = json.dumps([run_id, sorted(reasons), last_delivery], separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def inspect_run(cli: str, payload: dict[str, Any], runner: CommandRunner = run_json) -> dict[str, Any]:
    run_id = payload["run_id"]
    reasons: list[str] = []
    if state.lease_expired(payload) and payload["terminal_outcome"] is None:
        reasons.append("coordinator_lease_expired")
    tasks = runner([cli, "orchestration", "task-list", "--run", run_id, "--brief", "--json"])
    rows = tasks.get("result", {}).get("tasks", [])
    active = [row for row in rows if row.get("status") in {"ready", "dispatched"}]
    unfinished = payload["terminal_outcome"] is None
    if unfinished and not active:
        reasons.append("unfinished_without_active_task")
    inbox = runner([cli, "orchestration", "check", "--run", run_id, "--peek", "--json"])
    messages = inbox.get("result", {}).get("messages", [])
    actionable = [m for m in messages if m.get("type") in {"worker_done", "question", "escalation"}]
    if actionable:
        reasons.append("unread_actionable_delivery")
    terminal = payload["coordinator"]["terminal"]
    terminal_status = runner([cli, "terminal", "show", "--terminal", terminal, "--json"])
    connected = terminal_status.get("result", {}).get("terminal", {}).get("connected", False)
    if not connected:
        reasons.append("coordinator_terminal_disconnected")
    delivery_ids = [m.get("id") for m in actionable]
    return {"reasons": sorted(set(reasons)), "active_count": len(active), "delivery_ids": delivery_ids, "terminal_connected": connected}


def load_watchdog(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"last_key": None, "last_check": None, "last_wake": None, "pid": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_watchdog(path: Path, payload: dict[str, Any]) -> None:
    state.atomic_write_json(path, payload)


def check_once(root: Path, run_id: str, cli: str, *, dry_run: bool = False, test_wake: bool = False, runner: CommandRunner = run_json) -> dict[str, Any]:
    coordinator = state.load_state(root, run_id)
    watch_path = state.state_dir(root, run_id) / "watchdog_state.json"
    watch = load_watchdog(watch_path)
    observation = inspect_run(cli, coordinator, runner)
    if test_wake:
        observation["reasons"] = sorted(set(observation["reasons"] + ["readiness_test_wake"]))
    key = anomaly_key(run_id, observation["reasons"], (observation["delivery_ids"] or [None])[0])
    should_wake = bool(observation["reasons"]) and key != watch.get("last_key")
    wake_result = None
    if should_wake and not dry_run:
        message = f"resume audit requested run={run_id} reasons={','.join(observation['reasons'])} transition={key}"
        wake_result = runner([cli, "terminal", "send", "--terminal", coordinator["coordinator"]["terminal"], "--text", message, "--enter", "--json"])
        watch["last_wake"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        watch["last_key"] = key
        state.append_journal(root, run_id, {"operation": "watchdog_wake", "outcome": "succeeded", "transition_id": key, "reasons": observation["reasons"]})
    elif not observation["reasons"]:
        watch["last_key"] = None
    watch["last_check"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    watch["pid"] = os.getpid()
    save_watchdog(watch_path, watch)
    return {"ok": not observation["reasons"], "observation": observation, "transition_id": key, "woke": bool(wake_result), "dry_run": dry_run, "test_wake": test_wake}


def run_loop(root: Path, run_id: str, cli: str, interval: int) -> int:
    stop_path = state.state_dir(root, run_id) / "watchdog.stop"
    while not stop_path.exists():
        try:
            check_once(root, run_id, cli)
        except Exception as exc:  # watchdog must record failures and keep checking
            state.append_journal(root, run_id, {"operation": "watchdog_check", "outcome": "failed", "error": str(exc)})
        time.sleep(interval)
    stop_path.unlink(missing_ok=True)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runtime/orchestration"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--orca-cli", default=os.environ.get("ORCA_CLI_COMMAND", "orca"))
    sub = p.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check"); check.add_argument("--dry-run", action="store_true"); check.add_argument("--test-wake", action="store_true")
    run = sub.add_parser("run"); run.add_argument("--interval", type=int, default=90)
    sub.add_parser("stop"); sub.add_parser("status")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "check":
            if args.dry_run and args.test_wake:
                raise ValueError("--dry-run and --test-wake are mutually exclusive")
            result = check_once(args.root, args.run_id, args.orca_cli, dry_run=args.dry_run, test_wake=args.test_wake)
        elif args.command == "run": return run_loop(args.root, args.run_id, args.orca_cli, max(10, args.interval))
        elif args.command == "stop":
            path = state.state_dir(args.root, args.run_id) / "watchdog.stop"; path.parent.mkdir(parents=True, exist_ok=True); path.touch(); result = {"stop_requested": True}
        else: result = load_watchdog(state.state_dir(args.root, args.run_id) / "watchdog_state.json")
    except (ValueError, PermissionError, FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
