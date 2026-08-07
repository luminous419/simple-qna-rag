#!/usr/bin/env python3
"""Markdown local link checker (M3-NFR-005).

Repository-internal regression gate. Standard-library only (argparse, json,
pathlib, re, subprocess, sys, unicodedata, urllib.parse) — no new dependency.
Not a runtime module: lives under scripts/, not src/simple_qna_rag/.

Checks that every local relative link/image target and every heading anchor
reference inside tracked-or-untracked (non-ignored) Markdown files in this
repository resolves to something that actually exists. External URLs and
fenced/inline code spans are excluded. See Design.md §4.5 for the full
contract this implements.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit

_EXTERNAL_SCHEMES = ("http:", "https:", "mailto:", "tel:", "ftp:", "data:")

_SKIP_DIR_NAMES = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "build",
    "dist",
    "__pycache__",
}
_SKIP_PATH_PREFIX = ("evaluation/reports",)

_LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)]+)\)")
_REF_DEF_RE = re.compile(r"^[ \t]{0,3}\[([^\]]+)\]:[ \t]+(\S+)", re.MULTILINE)
_FENCE_RE = re.compile(r"^([ \t]{0,3})(`{3,}|~{3,})(.*)$")
_INLINE_CODE_RE = re.compile(r"(`+)(.+?)\1")
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_MD_EMPHASIS_RE = re.compile(r"[*_`]+")
_MD_LINK_TEXT_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_ANCHOR_STRIP_RE = re.compile(r"[^\w\- ]", re.UNICODE)


@dataclass
class LinkOccurrence:
    file: Path
    line: int
    target: str


@dataclass
class Failure:
    file: Path
    line: int
    target: str
    reason: str

    def format(self, root: Path) -> str:
        rel = _rel(self.file, root)
        return f"{rel}:{self.line}: broken link -> {self.target} ({self.reason})"


@dataclass
class EnumerationResult:
    files: list[Path] = field(default_factory=list)
    tracked: int = 0
    untracked: int = 0
    mode: str = "git"  # "git" | "walk"


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def enumerate_markdown_files(root: Path, *, use_git: bool = True) -> EnumerationResult:
    """Enumerate Markdown files under `root`.

    Default: union of tracked (`git ls-files --cached`) and untracked
    non-ignored (`git ls-files --others --exclude-standard`) files, combined
    in a single invocation, deduped, and stable-sorted by POSIX repo-relative
    path. Falls back to a hardcoded-skip recursive walk when `use_git` is
    False or git is unavailable.
    """
    if use_git:
        try:
            proc = subprocess.run(
                [
                    "git",
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "--",
                    "*.md",
                    "*.markdown",
                ],
                cwd=root,
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            print(f"error: git ls-files failed: {exc}", file=sys.stderr)
            sys.exit(2)

        raw_paths = [p for p in proc.stdout.decode("utf-8", "surrogateescape").split("\0") if p]

        try:
            cached_proc = subprocess.run(
                ["git", "ls-files", "-z", "--cached", "--", "*.md", "*.markdown"],
                cwd=root,
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            print(f"error: git ls-files failed: {exc}", file=sys.stderr)
            sys.exit(2)
        cached_set = {p for p in cached_proc.stdout.decode("utf-8", "surrogateescape").split("\0") if p}

        seen: set[str] = set()
        files: list[Path] = []
        tracked = 0
        untracked = 0
        for rel in sorted(set(raw_paths)):
            if rel in seen:
                continue
            seen.add(rel)
            abs_path = (root / rel)
            if not abs_path.exists():
                print(f"warning: {rel}: 파일이 없어 건너뜀", file=sys.stderr)
                continue
            if not abs_path.is_file():
                continue
            try:
                abs_path.read_text(encoding="utf-8")
            except OSError as exc:
                print(f"error: {rel}: 읽기 실패: {exc}", file=sys.stderr)
                sys.exit(2)
            files.append(abs_path)
            if rel in cached_set:
                tracked += 1
            else:
                untracked += 1
        return EnumerationResult(files=files, tracked=tracked, untracked=untracked, mode="git")

    files = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in (".md", ".markdown"):
            continue
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in _SKIP_DIR_NAMES for part in rel_parts):
            continue
        rel_posix = path.relative_to(root).as_posix()
        if any(rel_posix.startswith(prefix) for prefix in _SKIP_PATH_PREFIX):
            continue
        files.append(path)
    return EnumerationResult(files=files, tracked=0, untracked=0, mode="walk")


def _strip_code_regions(text: str) -> str:
    """Blank out fenced code blocks and inline code spans (keep line/col
    layout intact) so links inside them are never matched."""
    lines = text.split("\n")
    out_lines: list[str] = []
    fence_char = None
    fence_len = 0
    for line in lines:
        m = _FENCE_RE.match(line)
        if fence_char is None and m:
            fence_char = m.group(2)[0]
            fence_len = len(m.group(2))
            out_lines.append("")
            continue
        if fence_char is not None:
            m2 = _FENCE_RE.match(line)
            if m2 and m2.group(2)[0] == fence_char and len(m2.group(2)) >= fence_len:
                fence_char = None
                fence_len = 0
                out_lines.append("")
                continue
            out_lines.append("")
            continue
        # blank inline code spans, preserving length
        def _blank(match: re.Match) -> str:
            return " " * len(match.group(0))

        out_lines.append(_INLINE_CODE_RE.sub(_blank, line))
    return "\n".join(out_lines)


def _is_external(target: str) -> bool:
    if target.startswith("//"):
        return True
    scheme = urlsplit(target).scheme
    if scheme and f"{scheme}:" in _EXTERNAL_SCHEMES:
        return True
    return False


def _slugify(heading_text: str, used: dict[str, int]) -> str:
    text = _MD_LINK_TEXT_RE.sub(r"\1", heading_text)
    text = _MD_EMPHASIS_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = text.casefold()
    text = text.strip()
    text = re.sub(r"\s+", "-", text)
    text = _ANCHOR_STRIP_RE.sub("", text)
    text = text.replace(" ", "-")
    base = text
    count = used.get(base, 0)
    used[base] = count + 1
    if count == 0:
        return base
    return f"{base}-{count}"


def _heading_anchors(text: str) -> set[str]:
    used: dict[str, int] = {}
    anchors = set()
    for match in _HEADING_RE.finditer(text):
        heading_text = match.group(2)
        anchors.add(_slugify(heading_text, used))
    return anchors


def collect_links(path: Path, text: str) -> list[LinkOccurrence]:
    """Collect (file, line, target) for inline links/images and reference
    definitions, excluding fenced code / inline code spans."""
    stripped = _strip_code_regions(text)
    occurrences: list[LinkOccurrence] = []

    for match in _LINK_RE.finditer(stripped):
        target = match.group(3).strip()
        # split off an optional "title" after whitespace: [t](url "title")
        target = re.split(r"\s+", target, maxsplit=1)[0]
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        if not target:
            continue
        line = stripped.count("\n", 0, match.start()) + 1
        occurrences.append(LinkOccurrence(path, line, target))

    for match in _REF_DEF_RE.finditer(stripped):
        target = match.group(2).strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        line = stripped.count("\n", 0, match.start()) + 1
        occurrences.append(LinkOccurrence(path, line, target))

    return occurrences


def _check_target(occ: LinkOccurrence, root: Path, anchor_cache: dict[Path, set[str]]) -> Failure | None:
    target = occ.target
    if _is_external(target):
        return None

    path_part, _, anchor_part = target.partition("#")
    path_part = unquote(path_part)
    path_part = path_part.split("?", 1)[0]

    if not path_part:
        # "#anchor" only -> same file
        if not anchor_part:
            return None
        anchors = anchor_cache.setdefault(occ.file, _heading_anchors(occ.file.read_text(encoding="utf-8")))
        if anchor_part not in anchors:
            return Failure(occ.file, occ.line, target, "대상 파일에 anchor 없음")
        return None

    resolved = (occ.file.parent / path_part).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return Failure(occ.file, occ.line, target, "저장소 밖 경로")

    if not resolved.exists():
        return Failure(occ.file, occ.line, target, "파일 없음")

    if anchor_part and resolved.is_file() and resolved.suffix.lower() in (".md", ".markdown"):
        try:
            anchors = anchor_cache.setdefault(resolved, _heading_anchors(resolved.read_text(encoding="utf-8")))
        except OSError:
            return Failure(occ.file, occ.line, target, "대상 파일 읽기 실패")
        if anchor_part not in anchors:
            return Failure(occ.file, occ.line, target, "대상 파일에 anchor 없음")

    return None


def check_paths(files: list[Path], root: Path) -> tuple[list[Failure], int]:
    """Check all Markdown files. Returns (failures, total_links_checked)."""
    failures: list[Failure] = []
    anchor_cache: dict[Path, set[str]] = {}
    total_links = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        for occ in collect_links(path, text):
            total_links += 1
            failure = _check_target(occ, root, anchor_cache)
            if failure is not None:
                failures.append(failure)
    failures.sort(key=lambda f: (_rel(f.file, root), f.line))
    return failures, total_links


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="*", default=None, help="검사할 경로(디렉터리/파일) 목록")
    parser.add_argument("--no-git", action="store_true", help="git 대신 디렉터리 walk로 열거")
    parser.add_argument("--json", action="store_true", help="집계를 JSON으로 stdout에 출력")
    args = parser.parse_args(argv)

    root = Path.cwd()

    if args.paths:
        arg_paths = [Path(p) for p in args.paths]
        # A single directory arg is treated as its own root boundary so
        # isolated fixtures (tests, `--no-git --paths <tmp>`) can use ".."
        # boundary checks without being compared against the real cwd.
        if len(arg_paths) == 1 and arg_paths[0].is_dir():
            root = arg_paths[0].resolve()
        files: list[Path] = []
        for path in arg_paths:
            if path.is_dir():
                result = enumerate_markdown_files(path, use_git=not args.no_git)
                files.extend(result.files)
            elif path.is_file():
                files.append(path)
        files = sorted(set(files), key=lambda f: _rel(f, root))
        tracked = untracked = 0
        mode = "walk"
    else:
        result = enumerate_markdown_files(root, use_git=not args.no_git)
        files = result.files
        tracked = result.tracked
        untracked = result.untracked
        mode = result.mode

    failures, total_links = check_paths(files, root)

    for failure in failures:
        print(failure.format(root))

    if mode == "git":
        summary = (
            f"검사 파일 {len(files)}개(tracked {tracked} + untracked {untracked}), "
            f"링크 {total_links}개, 실패 {len(failures)}개"
        )
    else:
        summary = f"검사 파일 {len(files)}개(walk), 링크 {total_links}개, 실패 {len(failures)}개"
    print(summary)

    if args.json:
        payload = {
            "files": len(files),
            "tracked": tracked,
            "untracked": untracked,
            "links": total_links,
            "failures": [
                {
                    "file": _rel(f.file, root),
                    "line": f.line,
                    "target": f.target,
                    "reason": f.reason,
                }
                for f in failures
            ],
        }
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
