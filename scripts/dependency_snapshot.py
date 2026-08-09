#!/usr/bin/env python3
"""M4.1 §8.3 — dependency snapshot artifact (REQ-001.4).

Writes a canonical JSON snapshot (stable key order, `generated_at` excluded
for hash stability) to stdout and to `dependency_snapshot.json`. All fields
are computed from the repository's own lock/manifest files — no field is a
hand-maintained magic number (REQ-002.6 principle applied here too).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = REPO_ROOT / "requirements.lock"
PACKAGE_LOCK_FILE = REPO_ROOT / "package-lock.json"

SCHEMA_VERSION = "1.0.0"
PROFILE = "linux-x86_64-py3.11-node22"


def _canonical_lock_body(lock_text: str) -> str:
    lines = lock_text.splitlines(keepends=True)
    skipping = True
    body_lines = []
    for line in lines:
        if skipping and line.startswith("#"):
            continue
        skipping = False
        body_lines.append(line)
    return "".join(body_lines)


def _lock_sha256_canonical(lock_text: str) -> str:
    return hashlib.sha256(_canonical_lock_body(lock_text).encode("utf-8")).hexdigest()


def _package_count(lock_text: str) -> int:
    body = _canonical_lock_body(lock_text)
    return len(re.findall(r"^[A-Za-z0-9]", body, flags=re.MULTILINE))


def _uv_version() -> str:
    try:
        out = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    match = re.search(r"(\d+\.\d+\.\d+)", out)
    return match.group(1) if match else out


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _node_version() -> str:
    try:
        out = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.lstrip("v")


def build_snapshot() -> dict:
    lock_text = LOCK_FILE.read_text(encoding="utf-8")
    package_lock_bytes = PACKAGE_LOCK_FILE.read_bytes()
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "uv_version": _uv_version(),
        "python_version": _python_version(),
        "node_version": _node_version(),
        "lock_sha256_canonical": _lock_sha256_canonical(lock_text),
        "package_lock_sha256": hashlib.sha256(package_lock_bytes).hexdigest(),
        "package_count": _package_count(lock_text),
    }


def main(argv: list[str] | None = None) -> int:
    snapshot = build_snapshot()
    text = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, indent=2)
    print(text)
    (REPO_ROOT / "dependency_snapshot.json").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
