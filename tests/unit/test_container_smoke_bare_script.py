"""M4.3-REQ-005.5 — bare-script `python scripts/container_smoke.py` regression.

Hosted CI's `container` job invokes this script directly
(`python scripts/container_smoke.py ...`), never through `pytest`. A bare
script process only puts its own directory (`scripts/`) on `sys.path[0]` —
not the repository root — so `pyproject.toml`'s
`[tool.pytest.ini_options] pythonpath = ["."]` (a pytest-only setting) never
applies here, and `from tests.support.mock_ollama import ...` inside
`run_smoke()` used to fail with `ModuleNotFoundError: No module named
'tests'` (hosted run 31606183756; fixed by inserting `REPO_ROOT` into
`sys.path`, see Hosted_CI_Remediation_Iteration_2.md §3).

Code_Review_Iteration_4.md flagged that a pytest-import-based regression
test for this bug is not environment-independent: pytest's own import
machinery (or an unrelated system-installed `tests` package) can populate
`sys.modules["tests"]` before this module's `sys.path` insertion ever runs,
which would mask the exact failure this guard exists to catch. This test
instead launches the script as a real, separate `python` subprocess — the
same invocation shape as the hosted `container` job — so it reproduces the
bug class regardless of what the *test runner's own* process has already
imported.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_bare_script(image: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/container_smoke.py", "--image", image],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_bare_script_does_not_raise_module_not_found_for_tests_package():
    """Regression for the exact hosted-CI failure: importing
    `tests.support.mock_ollama` from a bare-script subprocess must not
    raise `ModuleNotFoundError`, regardless of what modules the pytest
    process running *this* test has already imported."""
    result = _run_bare_script("simple-qna-rag-nonexistent-regression-image:latest")

    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    assert "No module named 'tests'" not in result.stderr, result.stderr


def test_bare_script_reaches_docker_invocation_or_skips_cleanly():
    """Past the import boundary, the script must reach its real logic: on a
    host with docker, a nonexistent image produces the expected
    `docker_run_failed` result; without docker, it exits cleanly with the
    documented `SKIPPED`/`docker_unavailable` contract. Either outcome
    proves the import wiring did not fail first."""
    result = _run_bare_script("simple-qna-rag-nonexistent-regression-image:latest")

    payload = json.loads(result.stdout)
    assert payload["schema"] == "m43-container-smoke-v1"
    if payload.get("status") == "SKIPPED":
        assert payload.get("reason") == "docker_unavailable"
    else:
        assert payload.get("error") == "docker_run_failed"
