"""M4.1 §3.6 — subprocess/TestClient bootstrap evidence matrix (M2-01)."""

import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from simple_qna_rag.settings import Settings, get_settings
from simple_qna_rag.web import bootstrap as bootstrap_module
from simple_qna_rag.web.bootstrap import Bootstrap, load_bootstrap
from simple_qna_rag.web.server import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_row1_subprocess_create_app_with_default_bootstrap_exits_0():
    code = (
        "from simple_qna_rag.web.server import create_app; "
        "from simple_qna_rag.web.bootstrap import load_bootstrap; "
        "app = create_app(load_bootstrap()); "
        "print('CREATE_APP_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "CREATE_APP_OK" in result.stdout


def _ready_app(bootstrap: Bootstrap):
    return create_app(
        bootstrap=bootstrap,
        settings_loader=get_settings,
        engine_factory=lambda settings: object(),
    )


def test_row2_missing_static_dir(tmp_path):
    bootstrap = Bootstrap(static_dir=tmp_path / "missing", templates_dir=bootstrap_module.TEMPLATES_DIR)
    with TestClient(_ready_app(bootstrap)) as client:
        assert client.get("/health/live").status_code == 200
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["reason"] == "static_mount_failed"


def test_row3_static_dir_is_a_regular_file(tmp_path):
    fake_file = tmp_path / "static_as_file"
    fake_file.write_text("not a directory")
    bootstrap = Bootstrap(static_dir=fake_file, templates_dir=bootstrap_module.TEMPLATES_DIR)
    with TestClient(_ready_app(bootstrap)) as client:
        assert client.get("/health/live").status_code == 200
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["reason"] == "static_mount_failed"


def test_row4_missing_templates_dir(tmp_path):
    bootstrap = Bootstrap(static_dir=bootstrap_module.STATIC_DIR, templates_dir=tmp_path / "missing")
    with TestClient(_ready_app(bootstrap)) as client:
        assert client.get("/health/live").status_code == 200
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["reason"] == "static_mount_failed"


def test_row5_templates_dir_is_a_regular_file(tmp_path):
    fake_file = tmp_path / "templates_as_file"
    fake_file.write_text("not a directory")
    bootstrap = Bootstrap(static_dir=bootstrap_module.STATIC_DIR, templates_dir=fake_file)
    with TestClient(_ready_app(bootstrap)) as client:
        assert client.get("/health/live").status_code == 200
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["reason"] == "static_mount_failed"


def test_bootstrap_static_dir_matches_settings_static_dir_no_drift():
    settings = Settings.from_sources()
    assert str(load_bootstrap().static_dir) == str(settings.STATIC_DIR)
    assert str(load_bootstrap().templates_dir) == str(settings.TEMPLATES_DIR)
