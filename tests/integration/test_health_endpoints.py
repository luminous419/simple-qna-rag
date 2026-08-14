"""M4.1 §3.4/§9.2/§11.2 — health status table + deprecated alias."""

import io
import json

from fastapi.testclient import TestClient

from simple_qna_rag.observability import logging as obs_logging
from simple_qna_rag.rag_engine import EngineArtifactError
from simple_qna_rag.settings import Settings, SettingsError, get_settings
from simple_qna_rag.web.bootstrap import Bootstrap
from simple_qna_rag.web.server import create_app


def _app_with(bootstrap=None, settings_loader=None, engine_factory=None):
    return create_app(
        bootstrap=bootstrap, settings_loader=settings_loader, engine_factory=engine_factory
    )


def test_health_live_is_always_200_even_when_everything_else_fails():
    def _raise_settings():
        raise SettingsError("boom", exit_code=2)

    app = _app_with(settings_loader=_raise_settings)
    with TestClient(app) as client:
        r = client.get("/health/live")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_health_ready_ok_when_all_states_clear():
    app = _app_with(
        settings_loader=get_settings,
        engine_factory=lambda settings: object(),
    )
    with TestClient(app) as client:
        r = client.get("/health/ready")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "reason": "ok"}


def test_health_ready_settings_invalid():
    def _raise_settings():
        raise SettingsError("bad env", exit_code=2)

    app = _app_with(settings_loader=_raise_settings)
    with TestClient(app) as client:
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["reason"] == "settings_invalid"


def test_health_ready_engine_init_failed():
    def _raise_engine(settings):
        raise RuntimeError("model load failed")

    app = _app_with(
        settings_loader=get_settings,
        engine_factory=_raise_engine,
    )
    with TestClient(app) as client:
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["reason"] == "engine_init_failed"


def test_health_ready_engine_init_failed_logs_safe_exception_type_not_message():
    """Hosted_CI_Remediation_Iteration_6.md — the generic `engine_init_failed`
    branch previously discarded all type information about the underlying
    exception. The public `/health/ready` body must still carry only the
    bounded `reason` string (never exception text), while the server's own
    structured `startup` log gains the bare exception *class name* (never
    `str(exc)`, which can carry paths/URLs/secrets — M4.1 REPLACE)."""

    def _raise_engine(settings):
        raise PermissionError("/some/secret/path denied")

    app = _app_with(settings_loader=get_settings, engine_factory=_raise_engine)

    buf = io.StringIO()
    handler = obs_logging.SafeJsonHandler(stream=buf)
    original_handlers = obs_logging._logger.handlers
    obs_logging._logger.handlers = [handler]
    try:
        with TestClient(app) as client:
            r = client.get("/health/ready")
            assert r.status_code == 503
            body = r.json()
            assert body == {"status": "not_ready", "reason": "engine_init_failed"}
            assert "PermissionError" not in json.dumps(body)
            assert "/some/secret/path" not in json.dumps(body)
    finally:
        obs_logging._logger.handlers = original_handlers

    records = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    startup_records = [rec for rec in records if rec.get("event") == "startup"]
    assert len(startup_records) == 1
    assert startup_records[0]["reason"] == "engine_init_failed"
    assert startup_records[0]["engine_error_type"] == "PermissionError"
    assert "/some/secret/path" not in json.dumps(startup_records[0])


def test_health_ready_engine_artifact_error_discloses_allowlisted_reason():
    """CR-I5-MAJ-02: the lifespan must classify a dedicated
    EngineArtifactError and disclose its (allowlisted) reason as
    `artifact_{reason}`."""

    def _raise_engine(settings):
        raise EngineArtifactError("test_embedding_seam_unavailable")

    app = _app_with(
        settings_loader=get_settings,
        engine_factory=_raise_engine,
    )
    with TestClient(app) as client:
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["reason"] == "artifact_test_embedding_seam_unavailable"


def test_health_ready_arbitrary_reason_bearing_exception_is_not_reclassified():
    """CR-I5-MAJ-02: the lifespan must catch EngineArtifactError
    specifically, not any exception that merely happens to define a
    `.reason` attribute — an unrelated exception carrying `.reason` must
    still report plain `engine_init_failed`, never a disclosed artifact
    reason."""

    class _UnrelatedError(RuntimeError):
        def __init__(self):
            super().__init__("unrelated failure")
            self.reason = "not_a_real_artifact_reason"

    def _raise_engine(settings):
        raise _UnrelatedError()

    app = _app_with(
        settings_loader=get_settings,
        engine_factory=_raise_engine,
    )
    with TestClient(app) as client:
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["reason"] == "engine_init_failed"


def test_health_ready_static_mount_failed_takes_precedence(tmp_path):
    missing = tmp_path / "does_not_exist"
    from simple_qna_rag.web.bootstrap import TEMPLATES_DIR

    bootstrap = Bootstrap(static_dir=missing, templates_dir=TEMPLATES_DIR)

    def _raise_engine(settings):
        raise RuntimeError("would also fail")

    app = _app_with(
        bootstrap=bootstrap,
        settings_loader=get_settings,
        engine_factory=_raise_engine,
    )
    with TestClient(app) as client:
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["reason"] == "static_mount_failed"


def test_deprecated_health_alias_headers_and_body_shape():
    app = _app_with(
        settings_loader=get_settings,
        engine_factory=lambda settings: object(),
    )
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.headers["Deprecation"] == "true"
        assert r.headers["Sunset"] == "Fri, 06 Nov 2026 00:00:00 GMT"
        body = r.json()
        assert body["status"] == "healthy"
        assert body["rag_engine_initialized"] is True


def test_deprecated_health_alias_still_healthy_status_when_engine_missing():
    """Pre-M4.1 semantics preserved: `status` was always "healthy" even when
    the engine failed to initialize — only `rag_engine_initialized` differed."""

    def _raise_settings():
        raise SettingsError("boom", exit_code=2)

    app = _app_with(settings_loader=_raise_settings)
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["rag_engine_initialized"] is False


def test_metrics_endpoint_exposes_readiness_gauge():
    app = _app_with(
        settings_loader=get_settings,
        engine_factory=lambda settings: object(),
    )
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "rag_readiness" in r.text
