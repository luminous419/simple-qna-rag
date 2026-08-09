"""M4.1 §8.3 — dependency_snapshot.py schema completeness (REQ-001.4)."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import dependency_snapshot as ds  # noqa: E402

EXPECTED_KEYS = {
    "schema_version",
    "profile",
    "uv_version",
    "python_version",
    "node_version",
    "lock_sha256_canonical",
    "package_lock_sha256",
    "package_count",
}


def test_snapshot_has_all_expected_keys():
    snapshot = ds.build_snapshot()
    assert set(snapshot) == EXPECTED_KEYS


def test_snapshot_excludes_generated_at_for_hash_stability():
    snapshot = ds.build_snapshot()
    assert "generated_at" not in snapshot


def test_snapshot_canonical_json_key_order_is_sorted():
    snapshot = ds.build_snapshot()
    text = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, indent=2)
    keys_in_text = [line.split('"')[1] for line in text.splitlines() if line.strip().startswith('"')]
    assert keys_in_text == sorted(keys_in_text)


def test_snapshot_package_count_matches_lock():
    snapshot = ds.build_snapshot()
    assert snapshot["package_count"] == 103


def test_snapshot_is_deterministic_for_same_lock_content():
    a = ds.build_snapshot()
    b = ds.build_snapshot()
    assert a["lock_sha256_canonical"] == b["lock_sha256_canonical"]
    assert a["package_lock_sha256"] == b["package_lock_sha256"]
