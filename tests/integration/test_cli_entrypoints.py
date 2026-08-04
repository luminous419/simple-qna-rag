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
