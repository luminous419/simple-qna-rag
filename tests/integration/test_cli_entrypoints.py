"""M2.5 Phase 2 package entry point smoke tests."""

from pathlib import Path
import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module,program_name",
    [
        ("simple_qna_rag.cli.query", "simple-qna-rag-query"),
        ("simple_qna_rag.cli.index_documents", "simple-qna-rag-index"),
        ("simple_qna_rag.cli.web", "simple-qna-rag-web"),
    ],
)
def test_module_help_works_outside_repository(
    tmp_path: Path, module: str, program_name: str
) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert program_name in result.stdout


def test_config_paths_are_repository_anchored_outside_repository(tmp_path: Path) -> None:
    code = (
        "import json; from simple_qna_rag import config; "
        "print(json.dumps({"
        "'project_root': str(config.PROJECT_ROOT), "
        "'data': config.DATA_DIR, 'vectorstore': config.VECTORSTORE_PATH, "
        "'templates': config.TEMPLATES_DIR, 'static': config.STATIC_DIR, "
        "'model': config.INTENT_MODEL_PATH}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    paths = json.loads(result.stdout)
    repository_root = Path(__file__).resolve().parents[2]
    assert Path(paths["project_root"]) == repository_root
    assert Path(paths["data"]) == repository_root / "runtime" / "documents"
    assert Path(paths["vectorstore"]) == repository_root / "runtime" / "vectorstore"
    assert Path(paths["templates"]) == repository_root / "web" / "templates"
    assert Path(paths["static"]) == repository_root / "web" / "static"
    assert Path(paths["model"]) == repository_root / "models" / "intent_classifier"


# ---------------------------------------------------------------------------
# M4.1 §5.2/§5.3 — exit-2 CLI group (query/index) settings matrix.
# `simple-qna-rag-web` (non --check-config) is intentionally excluded here —
# it always exits 0 and expresses invalid settings via /health/ready 503
# instead (REQ-002.4 web exception), covered by test_health_endpoints.py.
# ---------------------------------------------------------------------------


def _run_settings_probe(tmp_path: Path, module: str, env_overrides: dict) -> subprocess.CompletedProcess:
    import os

    code = (
        "import sys; "
        f"sys.argv = ['{module}'] + sys.argv[1:]; "
        f"from {module} import build_parser; "
        "from simple_qna_rag.cli._settings_bootstrap import load_settings_or_exit; "
        "args = build_parser().parse_args([]); "
        "cli_overrides = {}; "
        "load_settings_or_exit(cli_overrides); "
        "print('SETTINGS_OK')"
    )
    env = {**os.environ, **env_overrides}
    return subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=30, env=env
    )


@pytest.mark.parametrize("module", ["simple_qna_rag.cli.query", "simple_qna_rag.cli.index_documents"])
def test_valid_settings_exit_0(tmp_path: Path, module: str) -> None:
    result = _run_settings_probe(tmp_path, module, {})
    assert result.returncode == 0, result.stderr
    assert "SETTINGS_OK" in result.stdout


@pytest.mark.parametrize("module", ["simple_qna_rag.cli.query", "simple_qna_rag.cli.index_documents"])
def test_invalid_enum_exits_2(tmp_path: Path, module: str) -> None:
    result = _run_settings_probe(
        tmp_path, module, {"SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE": "bogus"}
    )
    assert result.returncode == 2
    assert "SETTINGS_OK" not in result.stdout


@pytest.mark.parametrize("module", ["simple_qna_rag.cli.query", "simple_qna_rag.cli.index_documents"])
def test_unknown_env_key_exits_2(tmp_path: Path, module: str) -> None:
    result = _run_settings_probe(tmp_path, module, {"SIMPLE_QNA_RAG_NOT_A_REAL_FIELD": "1"})
    assert result.returncode == 2


def test_query_cli_vectorstore_dir_override_reaches_facade(tmp_path: Path) -> None:
    override_dir = tmp_path / "custom_vectorstore"
    override_dir.mkdir()
    code = (
        "from simple_qna_rag.cli._settings_bootstrap import load_settings_or_exit; "
        f"load_settings_or_exit({{'SIMPLE_QNA_RAG_VECTORSTORE_DIR': {str(override_dir)!r}}}); "
        "from simple_qna_rag import config; "
        "assert config.VECTORSTORE_PATH == str(" + repr(str(override_dir)) + "), config.VECTORSTORE_PATH; "
        "print('OVERRIDE_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "OVERRIDE_OK" in result.stdout


def test_non_overridden_fields_and_m3_flags_preserved_alongside_override(tmp_path: Path) -> None:
    import os

    override_dir = tmp_path / "custom_vectorstore"
    override_dir.mkdir()
    env = {**os.environ, "SIMPLE_QNA_RAG_ANSWER_TEMPLATE_MODE": "intent"}
    code = (
        "from simple_qna_rag.cli._settings_bootstrap import load_settings_or_exit; "
        f"load_settings_or_exit({{'SIMPLE_QNA_RAG_VECTORSTORE_DIR': {str(override_dir)!r}}}); "
        "from simple_qna_rag import config; "
        "assert config.ANSWER_TEMPLATE_MODE == 'intent', config.ANSWER_TEMPLATE_MODE; "
        "print('PRESERVED_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=30, env=env
    )
    assert result.returncode == 0, result.stderr
    assert "PRESERVED_OK" in result.stdout
