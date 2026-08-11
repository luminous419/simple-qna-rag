"""M4.1 §8.4 — requirements.lock shape (REQ-001.1/.2)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_FILE = REPO_ROOT / "requirements.lock"


def _lock_text() -> str:
    return LOCK_FILE.read_text(encoding="utf-8")


def test_lock_file_exists():
    assert LOCK_FILE.is_file()


def test_lock_has_no_nvidia_packages():
    """CPU profile (REQ-001.1) — no CUDA/nvidia-* wheels should be pinned."""
    text = _lock_text().lower()
    assert "\nnvidia-" not in text
    assert not text.startswith("nvidia-")


def test_lock_contains_prometheus_client_pinned():
    text = _lock_text()
    assert "prometheus-client==0.23.1" in text


def test_lock_every_package_has_hashes():
    """`--require-hashes`-compatible format: every top-level package line is
    followed by at least one `--hash=sha256:` entry."""
    lines = _lock_text().splitlines()
    package_lines = [
        i for i, line in enumerate(lines) if line and line[0].isalnum() and "==" in line
    ]
    assert len(package_lines) > 0
    for i in package_lines:
        # Hash lines follow on the same logical block until the next
        # non-indented line; at minimum the very next line must start the
        # hash block (package line ends with a trailing backslash).
        assert lines[i].rstrip().endswith("\\"), lines[i]
        assert "--hash=sha256:" in lines[i + 1]


def test_lock_package_count_is_102():
    lines = _lock_text().splitlines()
    package_lines = [line for line in lines if line and line[0].isalnum() and "==" in line]
    assert len(package_lines) == 102
