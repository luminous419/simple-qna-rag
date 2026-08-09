#!/usr/bin/env python3
"""Durable coordinator state, lease, and transition journal primitives."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1
TERMINAL_OUTCOMES = {None, "completed", "stopped", "paused"}


def utc_now() -> datetime:
    return datetime.now(UTC)


def isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def state_dir(root: Path, run_id: str) -> Path:
    if not run_id.startswith("run_") or "/" in run_id or ".." in run_id:
        raise ValueError("run-id must be an Orca run_* identifier")
    return root / run_id


def state_path(root: Path, run_id: str) -> Path:
    return state_dir(root, run_id) / "coordinator_state.json"


def journal_path(root: Path, run_id: str) -> Path:
    return state_dir(root, run_id) / "transition_journal.jsonl"


@contextmanager
def directory_lock(path: Path, *, timeout: float = 5.0) -> Iterator[None]:
    lock = path.with_name(path.name + ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            lock.mkdir(parents=True)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"lock timeout: {lock}")
            time.sleep(0.02)
    try:
        yield
    finally:
        try:
            lock.rmdir()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def validate_state(payload: dict[str, Any]) -> None:
    required = {
        "schema_version", "run_id", "scope", "coordinator", "lease",
        "phase", "iteration", "gate", "active", "last_delivery_id",
        "successor", "terminal_outcome", "resume_action", "updated_at",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"state missing required keys: {', '.join(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if payload["terminal_outcome"] not in TERMINAL_OUTCOMES:
        raise ValueError("invalid terminal_outcome")
    if not isinstance(payload["active"], list):
        raise ValueError("active must be a list")
    lease = payload["lease"]
    if not isinstance(lease, dict) or not {"owner", "acquired_at", "heartbeat_at", "expires_at"} <= lease.keys():
        raise ValueError("invalid lease object")


def load_state(root: Path, run_id: str) -> dict[str, Any]:
    path = state_path(root, run_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_state(payload)
    return payload


def save_state(root: Path, run_id: str, payload: dict[str, Any]) -> None:
    payload["updated_at"] = isoformat(utc_now())
    validate_state(payload)
    atomic_write_json(state_path(root, run_id), payload)


def append_journal(root: Path, run_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    path = journal_path(root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": SCHEMA_VERSION, "timestamp": isoformat(utc_now()), **entry}
    if not record.get("operation"):
        raise ValueError("journal operation is required")
    with directory_lock(path):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return record


def init_state(root: Path, run_id: str, scope: str, terminal: str, runtime_id: str, ttl: int) -> dict[str, Any]:
    path = state_path(root, run_id)
    with directory_lock(path):
        if path.exists():
            raise FileExistsError(f"state already exists: {path}")
        now = utc_now()
        owner = f"{runtime_id}:{terminal}"
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "scope": scope,
            "coordinator": {"terminal": terminal, "runtime_id": runtime_id, "host": socket.gethostname(), "pid": os.getpid()},
            "lease": {"owner": owner, "acquired_at": isoformat(now), "heartbeat_at": isoformat(now), "expires_at": isoformat(now + timedelta(seconds=ttl))},
            "phase": "initializing", "iteration": 0,
            "gate": {"status": "not_run", "score": None},
            "active": [], "last_delivery_id": None, "successor": None,
            "terminal_outcome": None, "resume_action": "run readiness audit",
            "watchdog": None, "updated_at": isoformat(now),
        }
        save_state(root, run_id, payload)
    append_journal(root, run_id, {"operation": "coordinator_started", "outcome": "succeeded", "owner": owner})
    return payload


def lease_expired(payload: dict[str, Any], now: datetime | None = None) -> bool:
    return parse_time(payload["lease"]["expires_at"]) <= (now or utc_now())


def assert_owner(payload: dict[str, Any], owner: str) -> None:
    if payload["lease"]["owner"] != owner:
        raise PermissionError(f"lease owned by {payload['lease']['owner']}")


def acquire_lease(root: Path, run_id: str, owner: str, ttl: int, *, force_expired: bool = False) -> dict[str, Any]:
    path = state_path(root, run_id)
    with directory_lock(path):
        payload = load_state(root, run_id)
        current = payload["lease"]["owner"]
        if current != owner and not (force_expired and lease_expired(payload)):
            raise PermissionError(f"active lease owned by {current}")
        now = utc_now()
        payload["lease"] = {"owner": owner, "acquired_at": isoformat(now), "heartbeat_at": isoformat(now), "expires_at": isoformat(now + timedelta(seconds=ttl))}
        save_state(root, run_id, payload)
    append_journal(root, run_id, {"operation": "lease_acquired", "outcome": "succeeded", "owner": owner})
    return payload


def heartbeat(root: Path, run_id: str, owner: str, ttl: int) -> dict[str, Any]:
    path = state_path(root, run_id)
    with directory_lock(path):
        payload = load_state(root, run_id)
        assert_owner(payload, owner)
        if lease_expired(payload):
            raise PermissionError("lease expired; reacquire it explicitly")
        now = utc_now()
        payload["lease"]["heartbeat_at"] = isoformat(now)
        payload["lease"]["expires_at"] = isoformat(now + timedelta(seconds=ttl))
        save_state(root, run_id, payload)
    return payload


def checkpoint(root: Path, run_id: str, owner: str, patch: dict[str, Any]) -> dict[str, Any]:
    forbidden = {"schema_version", "run_id", "lease", "coordinator"}
    if forbidden & patch.keys():
        raise ValueError(f"checkpoint cannot modify: {', '.join(sorted(forbidden & patch.keys()))}")
    path = state_path(root, run_id)
    with directory_lock(path):
        payload = load_state(root, run_id)
        assert_owner(payload, owner)
        payload.update(patch)
        save_state(root, run_id, payload)
    append_journal(root, run_id, {"operation": "checkpoint", "outcome": "succeeded", "owner": owner, "fields": sorted(patch)})
    return payload


def release_lease(root: Path, run_id: str, owner: str, outcome: str) -> dict[str, Any]:
    if outcome not in TERMINAL_OUTCOMES - {None}:
        raise ValueError("outcome must be completed, stopped, or paused")
    path = state_path(root, run_id)
    with directory_lock(path):
        payload = load_state(root, run_id)
        assert_owner(payload, owner)
        payload["terminal_outcome"] = outcome
        payload["lease"]["expires_at"] = isoformat(utc_now())
        payload["resume_action"] = None if outcome in {"completed", "stopped"} else "reacquire lease and run resume audit"
        save_state(root, run_id, payload)
    append_journal(root, run_id, {"operation": "lease_released", "outcome": outcome, "owner": owner})
    return payload


def audit(root: Path, run_id: str) -> dict[str, Any]:
    payload = load_state(root, run_id)
    issues: list[str] = []
    if lease_expired(payload) and payload["terminal_outcome"] is None:
        issues.append("lease_expired_without_terminal_outcome")
    if payload["terminal_outcome"] is None and not payload["active"] and not payload["successor"]:
        issues.append("unfinished_without_active_or_successor")
    journal = journal_path(root, run_id)
    if not journal.exists() or not journal.read_text(encoding="utf-8").strip():
        issues.append("journal_missing_or_empty")
    return {"ok": not issues, "issues": issues, "state": payload}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runtime/orchestration"))
    p.add_argument("--run-id", required=True)
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init"); init.add_argument("--scope", required=True); init.add_argument("--terminal", required=True); init.add_argument("--runtime-id", required=True); init.add_argument("--ttl", type=int, default=180)
    acq = sub.add_parser("acquire-lease"); acq.add_argument("--owner", required=True); acq.add_argument("--ttl", type=int, default=180); acq.add_argument("--force-expired", action="store_true")
    beat = sub.add_parser("heartbeat"); beat.add_argument("--owner", required=True); beat.add_argument("--ttl", type=int, default=180)
    cp = sub.add_parser("checkpoint"); cp.add_argument("--owner", required=True); cp.add_argument("--patch-json", required=True)
    j = sub.add_parser("journal"); j.add_argument("--entry-json", required=True)
    rel = sub.add_parser("release-lease"); rel.add_argument("--owner", required=True); rel.add_argument("--outcome", required=True, choices=["completed", "stopped", "paused"])
    sub.add_parser("show"); sub.add_parser("audit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "init": result = init_state(args.root, args.run_id, args.scope, args.terminal, args.runtime_id, args.ttl)
        elif args.command == "acquire-lease": result = acquire_lease(args.root, args.run_id, args.owner, args.ttl, force_expired=args.force_expired)
        elif args.command == "heartbeat": result = heartbeat(args.root, args.run_id, args.owner, args.ttl)
        elif args.command == "checkpoint": result = checkpoint(args.root, args.run_id, args.owner, json.loads(args.patch_json))
        elif args.command == "journal": result = append_journal(args.root, args.run_id, json.loads(args.entry_json))
        elif args.command == "release-lease": result = release_lease(args.root, args.run_id, args.owner, args.outcome)
        elif args.command == "show": result = load_state(args.root, args.run_id)
        else: result = audit(args.root, args.run_id)
    except (ValueError, PermissionError, FileNotFoundError, FileExistsError, TimeoutError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
