#!/usr/bin/env python3
"""M4.1 §4.1.1 — regenerate docs/generated/settings_field_spec.md from
`simple_qna_rag.settings.FIELD_SPECS`/`MODEL_VALIDATORS` (the single source).

Usage:
  python scripts/generate_field_spec.py            # overwrite the file
  python scripts/generate_field_spec.py --check     # diff 0 or exit 1
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from simple_qna_rag.settings import (  # noqa: E402
    FIELD_SPECS,
    MODEL_VALIDATORS,
    render_field_specs_table,
    render_model_validators_table,
)

OUTPUT_PATH = REPO_ROOT / "docs" / "generated" / "settings_field_spec.md"


def render() -> str:
    parts = [
        "# Settings Field Spec (generated)",
        "",
        "이 파일은 `scripts/generate_field_spec.py`가 `simple_qna_rag.settings."
        "FIELD_SPECS`/`MODEL_VALIDATORS`에서 재생성한다. 직접 편집하지 않는다.",
        "",
        "## FIELD_SPECS",
        "",
        render_field_specs_table(FIELD_SPECS),
        "",
        "## MODEL_VALIDATORS",
        "",
        render_model_validators_table(MODEL_VALIDATORS, FIELD_SPECS),
        "",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    check = "--check" in argv
    rendered = render()

    if check:
        if not OUTPUT_PATH.exists():
            print(f"{OUTPUT_PATH} does not exist", file=sys.stderr)
            return 1
        current = OUTPUT_PATH.read_text(encoding="utf-8")
        if current != rendered:
            import difflib

            diff = "".join(
                difflib.unified_diff(
                    current.splitlines(keepends=True),
                    rendered.splitlines(keepends=True),
                    fromfile=str(OUTPUT_PATH),
                    tofile="<generated>",
                )
            )
            sys.stdout.write(diff)
            return 1
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
