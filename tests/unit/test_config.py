"""M2.5 Phase 4 runtime path resolution tests."""

from pathlib import Path

import pytest

from simple_qna_rag.config import resolve_runtime_path


def test_environment_override_has_highest_priority(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    result = resolve_runtime_path(
        "TEST_PATH",
        tmp_path / "default",
        tmp_path / "legacy",
        environ={"TEST_PATH": str(configured)},
    )
    assert result == configured.resolve()


def test_new_default_used_when_present(tmp_path: Path) -> None:
    default = tmp_path / "default"
    legacy = tmp_path / "legacy"
    default.mkdir()

    assert resolve_runtime_path("TEST_PATH", default, legacy, environ={}) == default.resolve()


def test_new_and_legacy_conflict_stops_instead_of_merging(tmp_path: Path) -> None:
    default = tmp_path / "default"
    legacy = tmp_path / "legacy"
    default.mkdir()
    legacy.mkdir()

    with pytest.raises(RuntimeError, match="자동 병합하지 않으므로"):
        resolve_runtime_path("TEST_PATH", default, legacy, environ={})


def test_legacy_fallback_warns_when_only_legacy_exists(tmp_path: Path) -> None:
    default = tmp_path / "default"
    legacy = tmp_path / "legacy"
    legacy.mkdir()

    with pytest.warns(FutureWarning, match="기존 runtime 경로"):
        result = resolve_runtime_path("TEST_PATH", default, legacy, environ={})

    assert result == legacy.resolve()


def test_new_default_returned_when_neither_path_exists(tmp_path: Path) -> None:
    default = tmp_path / "default"
    legacy = tmp_path / "legacy"

    assert resolve_runtime_path("TEST_PATH", default, legacy, environ={}) == default.resolve()
