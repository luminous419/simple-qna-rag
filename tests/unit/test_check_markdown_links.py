"""Tests for scripts/check_markdown_links.py (M3-NFR-005, Design.md §4.5).

scripts/ is not a package, so the module under test is loaded via
importlib.util.spec_from_file_location(), matching the design's test note.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_markdown_links.py"
_spec = importlib.util.spec_from_file_location("check_markdown_links", _MODULE_PATH)
cml = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = cml
_spec.loader.exec_module(cml)


def run_main(args: list[str]) -> int:
    return cml.main(args)


# ---------------------------------------------------------------------------
# (a)-(i) fixture matrix — no git, no model, no network
# ---------------------------------------------------------------------------


def test_a_valid_relative_link(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / "target.md").write_text("# Target\n", encoding="utf-8")
    (tmp_path / "doc.md").write_text("[t](target.md)\n", encoding="utf-8")
    exit_code = run_main(["--no-git", "--paths", str(tmp_path)])
    assert exit_code == 0


def test_b_broken_relative_link(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / "doc.md").write_text("[t](missing.md)\n", encoding="utf-8")
    exit_code = run_main(["--no-git", "--paths", str(tmp_path)])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "missing.md" in out
    assert "(파일 없음)" in out


def test_c_valid_and_broken_anchor(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / "target.md").write_text("# Section One\n", encoding="utf-8")
    (tmp_path / "doc.md").write_text(
        "[ok](target.md#section-one)\n[bad](target.md#nope)\n", encoding="utf-8"
    )
    exit_code = run_main(["--no-git", "--paths", str(tmp_path)])
    out = capsys.readouterr().out
    assert exit_code == 1
    lines = out.splitlines()
    assert any("target.md#nope" in line and "anchor 없음" in line for line in lines)
    assert not any("target.md#section-one" in line for line in lines)


def test_d_code_blocks_and_inline_code_ignored(tmp_path: Path) -> None:
    content = (
        "```\n[fake](missing.md)\n```\n\n"
        "`[inline fake](missing2.md)`\n\n"
        "real text\n"
    )
    (tmp_path / "doc.md").write_text(content, encoding="utf-8")
    exit_code = run_main(["--no-git", "--paths", str(tmp_path)])
    assert exit_code == 0


def test_e_external_urls_ignored(tmp_path: Path) -> None:
    content = (
        "[a](https://example.com/x)\n"
        "[b](mailto:a@example.com)\n"
        "[c](//example.com/y)\n"
    )
    (tmp_path / "doc.md").write_text(content, encoding="utf-8")
    exit_code = run_main(["--no-git", "--paths", str(tmp_path)])
    assert exit_code == 0


def test_f_duplicate_heading_suffix(tmp_path: Path) -> None:
    (tmp_path / "target.md").write_text("# Dup\n\n# Dup\n", encoding="utf-8")
    (tmp_path / "doc.md").write_text(
        "[a](target.md#dup)\n[b](target.md#dup-1)\n", encoding="utf-8"
    )
    exit_code = run_main(["--no-git", "--paths", str(tmp_path)])
    assert exit_code == 0


def test_g_korean_heading_anchor(tmp_path: Path) -> None:
    (tmp_path / "target.md").write_text("# 한글 제목\n", encoding="utf-8")
    (tmp_path / "doc.md").write_text("[a](target.md#한글-제목)\n", encoding="utf-8")
    exit_code = run_main(["--no-git", "--paths", str(tmp_path)])
    assert exit_code == 0


def test_h_path_outside_repo_fails(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / "doc.md").write_text("[a](../../../etc/passwd)\n", encoding="utf-8")
    exit_code = run_main(["--no-git", "--paths", str(tmp_path)])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "저장소 밖" in out


def test_i_exit_codes(tmp_path: Path) -> None:
    (tmp_path / "ok.md").write_text("no links here\n", encoding="utf-8")
    assert run_main(["--no-git", "--paths", str(tmp_path)]) == 0

    (tmp_path / "bad.md").write_text("[a](missing.md)\n", encoding="utf-8")
    assert run_main(["--no-git", "--paths", str(tmp_path)]) == 1

    with pytest.raises(SystemExit) as exc_info:
        run_main(["--bogus-flag"])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Enumeration contract tests (E1-E6) — require a real git repo, skipped
# when git is unavailable. Never mock `git ls-files`.
# ---------------------------------------------------------------------------

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)


def _commit_all(root: Path, message: str = "commit") -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=root, check=True)


@requires_git
def test_e1_committed_broken_link(tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    _init_repo(tmp_path)
    (tmp_path / "doc.md").write_text("[a](missing.md)\n", encoding="utf-8")
    _commit_all(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert run_main([]) == 1


@requires_git
def test_e2_untracked_broken_link_still_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    (tmp_path / "seed.md").write_text("seed\n", encoding="utf-8")
    _commit_all(tmp_path)
    # new, unstaged Markdown with a broken link
    (tmp_path / "new.md").write_text("[a](missing.md)\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert run_main([]) == 1


@requires_git
def test_e3_gitignored_dir_excluded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    ignored_dir = tmp_path / "ignored"
    ignored_dir.mkdir()
    (ignored_dir / "bad.md").write_text("[a](missing.md)\n", encoding="utf-8")
    _commit_all(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert run_main([]) == 0


@requires_git
def test_e4_tracked_and_untracked_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    (tmp_path / "tracked.md").write_text("tracked\n", encoding="utf-8")
    _commit_all(tmp_path)
    (tmp_path / "untracked.md").write_text("untracked\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    exit_code = run_main(["--json"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert '"files": 2' in out
    assert '"tracked": 1' in out
    assert '"untracked": 1' in out


@requires_git
def test_e5_index_only_deleted_file_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    (tmp_path / "gone.md").write_text("gone\n", encoding="utf-8")
    _commit_all(tmp_path)
    (tmp_path / "gone.md").unlink()
    monkeypatch.chdir(tmp_path)
    exit_code = run_main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "파일이 없어 건너뜀" in captured.err


@requires_git
def test_e6_deterministic_repeat_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _init_repo(tmp_path)
    (tmp_path / "a.md").write_text("# A\n[b](b.md)\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n", encoding="utf-8")
    _commit_all(tmp_path)
    monkeypatch.chdir(tmp_path)

    files1 = cml.enumerate_markdown_files(tmp_path, use_git=True)
    files2 = cml.enumerate_markdown_files(tmp_path, use_git=True)
    assert [str(f) for f in files1.files] == [str(f) for f in files2.files]
    assert files1.tracked == files2.tracked
    assert files1.untracked == files2.untracked
