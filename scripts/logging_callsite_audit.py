#!/usr/bin/env python3
"""M4.1 §6.1 — AST-based output-surface audit.

Scans `src/simple_qna_rag/**/*.py` for four call kinds (`print`, `logging.*`
incl. `getLogger(...).<method>()`, `sys.std{out,err}.write`, uvicorn logging
config/reference) and classifies each callsite as REPLACE / KEEP_CLI /
REMOVE. Writes `docs/generated/logging_callsite_disposition.json`.

Identity is stable across line-number drift: `f"{module_path}::{scope}#{n}"`
where `scope` is the enclosing function/method qualname (or `<module>`) and
`n` is the 1-indexed occurrence of that call kind within that scope. `line`
is a diagnostic-only field.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "simple_qna_rag"
OUTPUT_PATH = REPO_ROOT / "docs" / "generated" / "logging_callsite_disposition.json"

# `observability/logging.py` is the sanctioned structured-log sink itself
# and is exempted (its `stream.write` IS the replacement mechanism, not a
# bypass of it).
_KEEP_SINK_FILES = frozenset({"observability/logging.py"})

# CR-I1-MAJ-02 closure: Design.md §6.1 limits KEEP_CLI to `cli/*.py` user
# stdout and the local uvicorn access log — every other product callsite is
# REPLACE by default. The previous version of this function special-cased a
# handful of files/scopes as REPLACE and let everything else fall through to
# a `KEEP_CLI` catch-all, which silently mis-classified 164 real product
# callsites (web_search.search_web, query_router.route_query,
# RAGEngine.initialize and its helpers, intent_classifier.load_intent_classifier,
# ...) as CLI-safe. The default is now REPLACE; only `cli/*.py` files and the
# uvicorn access-log kind are KEEP_CLI.
def _classify(rel_path: str, scope: str, kind: str) -> str:
    if kind == "uvicorn_logger":
        return "KEEP_CLI"
    if rel_path in _KEEP_SINK_FILES:
        return "KEEP_CLI"
    if rel_path.startswith("cli/"):
        return "KEEP_CLI"
    return "REPLACE"


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.entries: list[dict[str, object]] = []
        self._scope_stack: list[str] = []
        self._counters: dict[tuple[str, str], int] = {}
        self._logger_aliases: set[str] = set()

    def _scope(self) -> str:
        return ".".join(self._scope_stack) if self._scope_stack else "<module>"

    def _record(self, kind: str, node: ast.AST) -> None:
        scope = self._scope()
        key = (scope, kind)
        self._counters[key] = self._counters.get(key, 0) + 1
        ordinal = self._counters[key]
        identity = f"{self.rel_path}::{scope}#{ordinal}"
        self.entries.append(
            {
                "identity": identity,
                "file": self.rel_path,
                "scope": scope,
                "kind": kind,
                "line": getattr(node, "lineno", None),
                "disposition": _classify(self.rel_path, scope, kind),
            }
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "getLogger"
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "logging"
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._logger_aliases.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func

        if isinstance(func, ast.Name) and func.id == "print":
            self._record("print", node)

        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "logging" and func.attr != "getLogger":
                self._record("logging", node)
            elif func.value.id in self._logger_aliases:
                self._record("logging", node)

        if (
            isinstance(func, ast.Attribute)
            and func.attr == "write"
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "sys"
            and func.value.attr in ("stdout", "stderr")
        ):
            self._record("stdio_write", node)

        if (
            isinstance(func, ast.Attribute)
            and func.attr == "run"
            and isinstance(func.value, ast.Name)
            and func.value.id == "uvicorn"
        ):
            if any(kw.arg in ("access_log", "log_config") for kw in node.keywords):
                self._record("uvicorn_logger", node)

        if isinstance(func, ast.Attribute) and func.attr == "getLogger" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                if first.value.startswith("uvicorn"):
                    self._record("uvicorn_logger", node)

        self.generic_visit(node)


def scan() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel_path = str(path.relative_to(SRC_ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _Visitor(rel_path)
        visitor.visit(tree)
        entries.extend(visitor.entries)
    entries.sort(key=lambda e: (e["file"], e["identity"]))
    return entries


def build_report() -> dict[str, object]:
    entries = scan()
    by_kind: dict[str, int] = {}
    by_disposition: dict[str, int] = {}
    for e in entries:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
        by_disposition[e["disposition"]] = by_disposition.get(e["disposition"], 0) + 1
    return {
        "schema_version": "1.0.0",
        "totals": {"by_kind": by_kind, "by_disposition": by_disposition, "count": len(entries)},
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    check = "--check" in argv
    report = build_report()
    text = json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2)

    if check:
        if not OUTPUT_PATH.exists():
            print(f"{OUTPUT_PATH} does not exist", file=sys.stderr)
            return 1
        current = OUTPUT_PATH.read_text(encoding="utf-8")
        if current != text:
            print("logging_callsite_audit.py: drift detected", file=sys.stderr)
            return 1
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
