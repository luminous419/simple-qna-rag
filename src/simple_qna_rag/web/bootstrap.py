"""Package/repo-fixed web bootstrap (Design.md §3.2).

This module intentionally imports neither `config.py` nor `settings.py` —
`STATIC_DIR`/`TEMPLATES_DIR` are computed as `Path(__file__)`-relative
constants so `create_app()` remains importable/callable even when
`Settings` loading fails (REQ-002.4's web exception).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
STATIC_DIR: Final[Path] = _REPO_ROOT / "web" / "static"
TEMPLATES_DIR: Final[Path] = _REPO_ROOT / "web" / "templates"


@dataclass(frozen=True)
class Bootstrap:
    static_dir: Path = STATIC_DIR
    templates_dir: Path = TEMPLATES_DIR


def load_bootstrap() -> Bootstrap:
    return Bootstrap()
