#!/usr/bin/env python3
"""M4.3-REQ-005.4 — OCI archive/layer scanner (Design.md §7.4).

Exports the image with `docker save`, then walks every layer tar entry
(never `extractall()` — only `getmembers()`, so the scanner itself never
writes attacker-controlled paths to disk) looking for forbidden content and
path-traversal member names. Whiteout markers (`.wh.*`, OCI opaque
`.wh..wh..opq`) are not content and are skipped.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (".git/", "vcs_directory"),
    (".env", "env_file"),
    ("runtime/vectorstore/", "index_artifact"),
    ("runtime/documents/", "corpus_artifact"),
    ("runtime/index/versions/", "index_artifact"),
    ("models/intent_classifier/", "model_artifact"),
    (".ollama/", "ollama_data"),
    ("evaluation/reports/", "ci_report"),
    ("id_rsa", "credential"),
    (".pem", "credential"),
    (".pfx", "credential"),
    ("simple_qna_rag_test_seam", "test_embedding_seam"),
)


def export_image(image: str, out_tar: Path) -> None:
    subprocess.run(["docker", "save", image, "-o", str(out_tar)], check=True)


def normalize_member_path(name: str) -> str:
    return posixpath.normpath(name.lstrip("/"))


def classify_member(name: str) -> tuple[str, str] | None:
    norm = normalize_member_path(name)
    if norm.split("/", 1)[0] == "..":
        return ("path_traversal", name)
    for pattern, category in FORBIDDEN_PATTERNS:
        if pattern.rstrip("/") in norm:
            return (category, pattern)
    return None


def is_whiteout(name: str) -> bool:
    base = posixpath.basename(normalize_member_path(name))
    return base.startswith(".wh.")


def scan(image: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "image.tar"
        export_image(image, archive)
        with tarfile.open(archive) as outer:
            manifest = json.loads(outer.extractfile("manifest.json").read())
            layer_paths = [entry for cfg in manifest for entry in cfg["Layers"]]
            violations = []
            layer_reports = []
            for layer_path in layer_paths:
                members = []
                with tarfile.open(fileobj=outer.extractfile(layer_path)) as layer:
                    for member in layer.getmembers():
                        if is_whiteout(member.name):
                            continue
                        hit = classify_member(member.name)
                        if hit:
                            violations.append({
                                "layer": layer_path, "member": member.name,
                                "category": hit[0], "pattern": hit[1],
                            })
                        members.append(member.name)
                layer_reports.append({"layer": layer_path, "member_count": len(members)})
            return {
                "schema": "m43-layer-scan-v1", "image": image,
                "layers": layer_reports, "violations": violations,
                "forbidden_count": len(violations),
            }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    result = scan(args.image)
    text = json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["forbidden_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
