"""Shared exit-2 Settings loading helper for the exit-2 CLI group
(query/index; Design.md §3.5, §5.1). `simple-qna-rag-web` does not use this
module for its default serve path — REQ-002.4's web exception expresses
settings failures via `/health/ready` 503 instead of exit codes.
"""

from __future__ import annotations

import sys

from simple_qna_rag.settings import Settings, SettingsError, set_settings_for_process


def load_settings_or_exit(cli_overrides: dict[str, str]) -> "Settings":
    """`Settings.from_sources(cli_overrides=...)` -> exit 2 on `SettingsError`.

    On success, populates the process-wide cache so `config.py`'s facade
    (materialized on first import, downstream of this call) sees the same
    override-applied instance rather than recomputing from `os.environ` alone.
    """
    try:
        settings = Settings.from_sources(cli_overrides=cli_overrides)
    except SettingsError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    set_settings_for_process(settings)
    return settings
