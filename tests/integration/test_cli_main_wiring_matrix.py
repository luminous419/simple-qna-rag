"""M4.1 §5.1/§5.2 — CR-I1-MIN-02 closure.

Traceability REQ-002.4/REQ-006.1 claims a three-CLI valid/invalid/override
matrix that actually exercises `main()` wiring. The prior test only called
`load_settings_or_exit()` directly (a Settings-layer probe), never `main()`
itself, and never asserted the engine/index constructor was skipped on
invalid settings. This file drives the real `main()` of all three CLIs in a
subprocess (so each case gets a fresh interpreter/import, matching how
`config.py`'s facade actually snapshots `Settings` once per process) and:

- query.py / index_documents.py (exit-2 CLI group): invalid settings exits 2
  *before* the engine/document-load constructor runs; a valid CLI override
  reaches that constructor's observed config value.
- web.py (documented REQ-002.4 exception, Design.md §3.3): invalid settings
  does NOT block `main()` — `create_app()`/`uvicorn.run()` are still called,
  because failure is expressed via `/health/ready` 503, not an exit code;
  and a CLI override still reaches the `settings_loader` passed into
  `create_app()`.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(code: str, *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# query.py
# ---------------------------------------------------------------------------


def test_query_main_invalid_settings_exits_2_before_engine_constructed(tmp_path):
    """`rag_engine.py`(및 그 아래 `config.py`)는 settings 검증 통과 후에만
    지연 import된다(query.py의 exit-2 group 계약). settings가 무효면 이
    모듈이 `sys.modules`에 전혀 등록되지 않는다는 사실 자체가 엔진 생성자가
    호출되지 않았다는 가장 직접적인 증거다."""
    code = """
import json, os, sys
os.environ["SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE"] = "bogus"
sys.argv = ["simple-qna-rag-query"]
from simple_qna_rag.cli import query as query_mod
exit_code = None
try:
    query_mod.main()
except SystemExit as e:
    exit_code = e.code
engine_module_imported = "simple_qna_rag.rag_engine" in sys.modules
print(json.dumps({"exit_code": exit_code, "engine_module_imported": engine_module_imported}))
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["exit_code"] == 2
    assert payload["engine_module_imported"] is False


def test_query_main_valid_override_reaches_engine_constructor(tmp_path):
    """override가 `get_rag_engine()`이 이후 소비할 정확히 그 검증된 Settings
    인스턴스까지 도달함을 확인한다. `rag_engine.py`/`config.py`를 실제
    import하기 전에(무거운 FAISS/Ollama 초기화를 피하기 위해)
    `set_settings_for_process()` 호출 시점에서 가로챈다 — 이는
    `load_settings_or_exit()`가 성공 직후 반드시 거치는, 엔진 생성자
    (`get_rag_engine()` -> `config.py` facade)가 그대로 읽는 지점이다."""
    override_dir = tmp_path / "custom_vectorstore"
    code = """
import json, sys
override_dir = sys.argv[1]
sys.argv = ["simple-qna-rag-query", "--vectorstore-dir", override_dir]
captured = {}
from simple_qna_rag.cli import _settings_bootstrap as bootstrap_mod
def _capturing_set(settings):
    captured["vectorstore_path"] = str(settings.VECTORSTORE_PATH)
    raise SystemExit(0)
bootstrap_mod.set_settings_for_process = _capturing_set
from simple_qna_rag.cli import query as query_mod
try:
    query_mod.main()
except SystemExit:
    pass
print(json.dumps(captured))
"""
    result = _run(code, str(override_dir))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["vectorstore_path"] == str(override_dir)


# ---------------------------------------------------------------------------
# index_documents.py
# ---------------------------------------------------------------------------


def test_index_main_invalid_settings_exits_2_before_documents_loaded():
    code = """
import json, os, sys
os.environ["SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE"] = "bogus"
sys.argv = ["simple-qna-rag-index"]
called = {"load_documents_called": False}
from simple_qna_rag.cli import index_documents as idx_mod
def _fake_load_documents():
    called["load_documents_called"] = True
    return []
idx_mod.load_documents = _fake_load_documents
exit_code = None
try:
    idx_mod.main()
except SystemExit as e:
    exit_code = e.code
print(json.dumps({"exit_code": exit_code, **called}))
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["exit_code"] == 2
    assert payload["load_documents_called"] is False


def test_index_main_valid_override_reaches_document_load_boundary(tmp_path):
    override_dir = tmp_path / "custom_documents"
    override_dir.mkdir()
    code = """
import json, sys
override_dir = sys.argv[1]
sys.argv = ["simple-qna-rag-index", "--documents-dir", override_dir]
captured = {}
from simple_qna_rag.cli import index_documents as idx_mod
def _fake_load_documents():
    captured["data_dir"] = idx_mod.DATA_DIR
    return []
idx_mod.load_documents = _fake_load_documents
exit_code = None
try:
    idx_mod.main()
except SystemExit as e:
    exit_code = e.code
print(json.dumps({"exit_code": exit_code, **captured}))
"""
    result = _run(code, str(override_dir))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["data_dir"] == str(override_dir)
    # main() sys.exit(0)'s when no documents are found — the empty-document
    # early-return path, not a settings failure.
    assert payload["exit_code"] == 0


# ---------------------------------------------------------------------------
# web.py — documented REQ-002.4 exception: the default serve path never
# exits on invalid settings; failure is expressed via /health/ready 503.
# ---------------------------------------------------------------------------


def test_web_main_invalid_settings_still_starts_server_not_blocked():
    code = """
import json, os, sys
os.environ["SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE"] = "bogus"
sys.argv = ["simple-qna-rag-web"]
captured = {"create_app_called": False, "uvicorn_run_called": False}
from simple_qna_rag.web import server as server_mod
def _capturing_create_app(*a, **kw):
    captured["create_app_called"] = True
    return object()
server_mod.create_app = _capturing_create_app
def _fake_run(app, host=None, port=None):
    captured["uvicorn_run_called"] = True
server_mod.uvicorn.run = _fake_run
from simple_qna_rag.cli import web as web_mod
web_mod.main()
print(json.dumps(captured))
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["create_app_called"] is True
    assert payload["uvicorn_run_called"] is True


def test_web_main_valid_override_reaches_settings_loader_passed_to_create_app(tmp_path):
    override_dir = tmp_path / "custom_vectorstore"
    code = """
import json, sys
override_dir, host, port = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = ["simple-qna-rag-web", "--host", host, "--port", port, "--vectorstore-dir", override_dir]
captured = {}
from simple_qna_rag.web import server as server_mod
def _capturing_create_app(*a, **kw):
    settings = kw["settings_loader"]()
    captured["vectorstore_path"] = str(settings.VECTORSTORE_PATH)
    return object()
server_mod.create_app = _capturing_create_app
def _fake_run(app, host=None, port=None):
    captured["host"] = host
    captured["port"] = port
server_mod.uvicorn.run = _fake_run
from simple_qna_rag.cli import web as web_mod
web_mod.main()
print(json.dumps(captured))
"""
    result = _run(code, str(override_dir), "127.0.0.9", "9999")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["vectorstore_path"] == str(override_dir)
    assert payload["host"] == "127.0.0.9"
    assert payload["port"] == 9999
