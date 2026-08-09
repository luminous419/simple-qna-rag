"""M4.1 §6.1/§6.4 — output-surface audit self-consistency."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DISPOSITION_PATH = REPO_ROOT / "docs" / "generated" / "logging_callsite_disposition.json"

_KNOWN_DISPOSITIONS = {"REPLACE", "KEEP_CLI", "REMOVE"}
# CR-I1-MAJ-02 closure — Design.md §6.1: KEEP_CLI is limited to `cli/*.py`
# user stdout and the sanctioned structured-log sink itself. Every other
# product file/scope is REPLACE by default (see `_classify()`).
_KEEP_CLI_ALLOWED_PREFIXES = ("cli/",)
_KEEP_SINK_FILES = {"observability/logging.py"}


def _load() -> dict:
    return json.loads(DISPOSITION_PATH.read_text(encoding="utf-8"))


def test_audit_check_passes_no_drift():
    result = subprocess.run(
        [sys.executable, "scripts/logging_callsite_audit.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_unclassified_entries():
    report = _load()
    for entry in report["entries"]:
        assert entry["disposition"] in _KNOWN_DISPOSITIONS, entry


def test_no_replace_designated_callsites_remain():
    """Every callsite the audit still finds must already have disposition
    REPLACE == 0 entries system-wide — i.e. the migration to `log_event()`
    (or removal) is complete and nothing is left needing replacement."""
    report = _load()
    remaining = [e for e in report["entries"] if e["disposition"] == "REPLACE"]
    assert remaining == []


def test_keep_cli_entries_confined_to_cli_files_and_sanctioned_sink():
    """CR-I1-MAJ-02: KEEP_CLI is limited to `cli/*.py` user stdout and the
    sanctioned structured-log sink (Design.md §6.1) — no other file may carry
    a KEEP_CLI disposition."""
    report = _load()
    offenders = [
        e
        for e in report["entries"]
        if e["disposition"] == "KEEP_CLI"
        and e["file"] not in _KEEP_SINK_FILES
        and not e["file"].startswith(_KEEP_CLI_ALLOWED_PREFIXES)
    ]
    assert offenders == []


def test_totals_by_kind_matches_entry_count():
    report = _load()
    assert sum(report["totals"]["by_kind"].values()) == report["totals"]["count"]
    assert sum(report["totals"]["by_disposition"].values()) == report["totals"]["count"]
    assert len(report["entries"]) == report["totals"]["count"]
