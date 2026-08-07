"""Phase 0/6 fingerprint confirmation CLI (M3-REQ-001, M3-REQ-010).

Compares the current worktree's dataset/corpus/vectorstore fingerprints
against an approved baseline JSON (e.g. evaluation/baselines/m2_initial.json)
without loading any model or deserializing FAISS — only
reporting.build_corpus_manifest()/build_vectorstore_fingerprint() and a
dataset byte hash are used (Design.md §4.3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from evaluation import reporting
from simple_qna_rag import config

_REPO_ROOT = Path(__file__).resolve().parent.parent


def collect_fingerprint(data_dir: Path, vectorstore_path: Path, dataset_path: Path) -> dict:
    """Collect the four M3-REQ-001 comparison fingerprints for the current
    worktree. Reuses reporting.build_corpus_manifest()/
    build_vectorstore_fingerprint() and never loads a model or deserializes
    the FAISS index itself."""
    dataset_bytes = dataset_path.read_bytes()
    manifest = reporting.build_corpus_manifest(data_dir)
    vectorstore_fp = reporting.build_vectorstore_fingerprint(vectorstore_path)
    return {
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "corpus_manifest_sha256": manifest["manifest_sha256"],
        "corpus_file_count": len(manifest["entries"]),
        "index_faiss_sha256": vectorstore_fp["index_faiss_sha256"],
        "index_pkl_sha256": vectorstore_fp["index_pkl_sha256"],
        "git_commit": reporting._git_commit(),
        "git_dirty": reporting._git_dirty(),
        "python_version": platform.python_version(),
    }


def _get(d: dict, path: str):
    node = d
    for part in path.split("/"):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def compare_with_baseline(current: dict, baseline_json: Path) -> dict:
    """Compare `current` (collect_fingerprint() output) against the four
    fingerprints recorded in an approved baseline JSON. Returns
    {"match": bool, "diff": [...], "baseline_id": str | None}."""
    baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
    expectations = [
        ("dataset_sha256", "execution/dataset_sha256"),
        ("corpus_manifest_sha256", "reproducibility/corpus_manifest_sha256"),
        ("index_faiss_sha256", "reproducibility/vectorstore_fingerprint/index_faiss_sha256"),
        ("index_pkl_sha256", "reproducibility/vectorstore_fingerprint/index_pkl_sha256"),
    ]
    diff = []
    for current_key, baseline_path in expectations:
        expected = _get(baseline, baseline_path)
        actual = current.get(current_key)
        if expected != actual:
            diff.append({"field": current_key, "expected": expected, "actual": actual})
    return {
        "match": not diff,
        "diff": diff,
        "baseline_id": baseline.get("baseline_id"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/datasets/golden.jsonl"))
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        current = collect_fingerprint(Path(config.DATA_DIR), Path(config.VECTORSTORE_PATH), args.dataset)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result: dict = {"current": current}
    exit_code = 0
    if args.baseline is not None:
        try:
            comparison = compare_with_baseline(current, args.baseline)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        result["comparison"] = comparison
        exit_code = 0 if comparison["match"] else 3

    if args.json:
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    else:
        print(f"dataset_sha256: {current['dataset_sha256']}")
        print(f"corpus_manifest_sha256: {current['corpus_manifest_sha256']} ({current['corpus_file_count']} files)")
        print(f"index_faiss_sha256: {current['index_faiss_sha256']}")
        print(f"index_pkl_sha256: {current['index_pkl_sha256']}")
        print(f"git_commit: {current['git_commit']} dirty={current['git_dirty']}")
        if "comparison" in result:
            comparison = result["comparison"]
            if comparison["match"]:
                print(f"MATCH baseline={comparison['baseline_id']}")
            else:
                print(f"MISMATCH baseline={comparison['baseline_id']}")
                for d in comparison["diff"]:
                    print(f"  {d['field']}: expected={d['expected']} actual={d['actual']}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
