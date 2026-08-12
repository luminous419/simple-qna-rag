#!/usr/bin/env python3
"""M4.3-REQ-007 — hosted CI job producer receipt writer (Design.md §8.1/§8.2-a).

Run as the last step of a hosted job (only `if: success()`), so its mere
existence in the uploaded artifact already implies GitHub Actions computed
that job as a success — `m4-assemble` still independently re-derives every
semantic field from the payload bytes rather than trusting this receipt's
self-reported `semantic_status`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Standard-library only, deliberately no `simple_qna_rag`/`assemble_m4_evidence`
# import: `frontend-tests` writes this receipt without a Python package
# install step, so this script must run with a bare `python3` interpreter
# (the ubuntu-latest image ships one by default). `RECEIPT_SCHEMA` and the
# canonical payload-manifest hash are small enough to duplicate rather than
# import-couple across scripts/jobs — the same convention this codebase
# already applies to `_git_commit`/`_git_dirty`/`_lock_sha256_canonical`.
RECEIPT_SCHEMA = "m43-producer-receipt-v1"


def _canonical_json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _payload_manifest_sha256(payload_hashes: dict[str, str]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload_hashes)).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--payload", action="append", default=[])
    parser.add_argument("--output", default="ci_producer_receipt.json")
    parser.add_argument("--semantic-status", default="PASS", choices=("PASS", "FAIL"))
    args = parser.parse_args(argv)

    payloads = []
    payload_hashes: dict[str, str] = {}
    for path_str in args.payload:
        path = Path(path_str)
        data = path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        payloads.append({"filename": path.name, "sha256": sha256, "size_bytes": len(data)})
        payload_hashes[path.name] = sha256

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "job": args.job,
        "sha": os.environ.get("GITHUB_SHA", "local"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
        "workflow_path": os.environ.get("M43_WORKFLOW_PATH", ".github/workflows/ci.yml"),
        "event_name": os.environ.get("GITHUB_EVENT_NAME", "local"),
        "semantic_status": args.semantic_status,
        "payload_manifest_sha256": _payload_manifest_sha256(payload_hashes),
        "payloads": payloads,
    }
    Path(args.output).write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
