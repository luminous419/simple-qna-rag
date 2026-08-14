"""M4.3-REQ-005 — container_smoke.py argv contracts (no docker required,
Design.md §7.5). These are pure functions of their arguments — subprocess is
never invoked here."""

from __future__ import annotations

import io
import json
import urllib.error

from scripts import container_smoke


def test_docker_run_argv_includes_add_host_and_embedding_seam_env():
    argv = container_smoke.build_docker_run_argv(
        "simple-qna-rag:test", index_root="/tmp/idx", host_port=8001, mock_port=9001,
        test_seam_dir="/repo/tests/support")
    assert "--add-host" in argv
    assert "host.docker.internal:host-gateway" in argv
    assert "SIMPLE_QNA_RAG_EMBEDDING_PROVIDER=deterministic_test" in argv
    assert "SIMPLE_QNA_RAG_ALLOW_TEST_EMBEDDING=1" in argv
    assert "/repo/tests/support:/opt/m43-test-seam:ro" in argv
    assert "PYTHONPATH=/opt/m43-test-seam" in argv
    assert "--user" in argv and "10001:10001" in argv
    assert "--read-only" in argv
    assert "--cap-drop" in argv and "ALL" in argv
    assert "--security-opt" in argv and "no-new-privileges" in argv


def test_reachability_probe_argv_targets_mock_ping_via_host_gateway():
    argv = container_smoke.build_reachability_probe_argv("container123", 9001)
    assert argv[:3] == ["docker", "exec", "container123"]
    joined = " ".join(argv)
    assert "host.docker.internal:9001/mock/ping" in joined


def test_negative_activation_argv_omits_test_seam_mount_and_pythonpath():
    argv = container_smoke.build_negative_activation_argv("simple-qna-rag:test", neg_host_port=8002)
    joined = " ".join(argv)
    assert "tests/support" not in joined
    assert "PYTHONPATH" not in joined
    assert "SIMPLE_QNA_RAG_EMBEDDING_PROVIDER=deterministic_test" in argv
    assert "SIMPLE_QNA_RAG_ALLOW_TEST_EMBEDDING=1" in argv
    assert "--user" in argv and "10001:10001" in argv
    assert "--read-only" in argv


def test_docker_unavailable_short_circuits_to_skipped(monkeypatch):
    monkeypatch.setattr(container_smoke.shutil, "which", lambda _: None)
    result = container_smoke.run_smoke("simple-qna-rag:test")
    assert result == {"status": "SKIPPED", "reason": "docker_unavailable"}


# ---------------------------------------------------------------------------
# CR-I1-MAJ-01 — real static asset HTTP verification (stubbed HTTP, no
# docker/network required). `check_static_asset` is the exact function
# `run_smoke` calls to populate `static_asset_ok`.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, *, status: int, body: bytes, content_type: str | None):
        self.status = status
        self._body = body
        self.headers = {"Content-Type": content_type} if content_type is not None else {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_check_static_asset_true_for_200_with_js_content_type_and_body(monkeypatch):
    monkeypatch.setattr(
        container_smoke.urllib.request, "urlopen",
        lambda url, timeout=None: _FakeResponse(
            status=200, body=b"console.log(1);", content_type="text/javascript; charset=utf-8"))
    assert container_smoke.check_static_asset(8000) is True


def test_check_static_asset_false_for_404_missing_copy_web_static(monkeypatch):
    def _raise(url, timeout=None):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))

    monkeypatch.setattr(container_smoke.urllib.request, "urlopen", _raise)
    assert container_smoke.check_static_asset(8000) is False


def test_check_static_asset_false_for_wrong_content_type(monkeypatch):
    monkeypatch.setattr(
        container_smoke.urllib.request, "urlopen",
        lambda url, timeout=None: _FakeResponse(
            status=200, body=b"<html>not js</html>", content_type="text/html"))
    assert container_smoke.check_static_asset(8000) is False


def test_check_static_asset_false_for_empty_body(monkeypatch):
    monkeypatch.setattr(
        container_smoke.urllib.request, "urlopen",
        lambda url, timeout=None: _FakeResponse(
            status=200, body=b"", content_type="text/javascript"))
    assert container_smoke.check_static_asset(8000) is False


def test_check_static_asset_false_on_connection_error(monkeypatch):
    def _raise(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(container_smoke.urllib.request, "urlopen", _raise)
    assert container_smoke.check_static_asset(8000) is False


def test_check_static_asset_requests_the_expected_path(monkeypatch):
    seen = {}

    def _capture(url, timeout=None):
        seen["url"] = url
        return _FakeResponse(status=200, body=b"x", content_type="application/javascript")

    monkeypatch.setattr(container_smoke.urllib.request, "urlopen", _capture)
    container_smoke.check_static_asset(9123)
    assert seen["url"] == f"http://127.0.0.1:9123{container_smoke.STATIC_ASSET_PATH}"


# ---------------------------------------------------------------------------
# `compute_all_ok` — proves static_asset_ok is part of the PASS/FAIL
# aggregation (this is the exact wiring gap CR-I1-MAJ-01 identified: the
# field was recorded in the receipt but never consulted by `all_ok`).
# ---------------------------------------------------------------------------


def _all_true_result() -> dict:
    return {
        "host_gateway_reachable": True, "mock_query_ok": True, "root_page_ok": True,
        "static_asset_ok": True, "production_test_seam_sealed": True,
    }


def test_compute_all_ok_true_when_every_field_true():
    assert container_smoke.compute_all_ok(_all_true_result()) is True


def test_compute_all_ok_false_when_static_asset_ok_false_even_if_others_true():
    """Regression oracle for the missing-`COPY web/static/` scenario: every
    other check can pass (the app boots, serves the root page, answers the
    mock query) while static assets 404 — `all_ok` must still be False."""
    result = _all_true_result()
    result["static_asset_ok"] = False
    assert container_smoke.compute_all_ok(result) is False


def test_compute_all_ok_false_when_static_asset_ok_missing_from_receipt():
    result = _all_true_result()
    del result["static_asset_ok"]
    assert container_smoke.compute_all_ok(result) is False


# ---------------------------------------------------------------------------
# main() negative control — a run_smoke() result representing the exact
# missing-`COPY web/static/` regression must exit 1, and a result where only
# static assets fail must not be masked by the other fields being true.
# ---------------------------------------------------------------------------


def test_main_exits_nonzero_when_static_asset_check_fails(monkeypatch, tmp_path):
    """Simulates what `run_smoke()` would report against a production image
    built without `COPY web/static/` (DR-I1-MAJ-06 regression): docker run
    succeeds, the app boots and answers the mock query, but the static
    asset GET 404s. `main()` must surface this as a nonzero exit so the
    hosted CI job fails."""
    regressed_result = {
        "schema": "m43-container-smoke-v1", "image": "simple-qna-rag:test",
        "host_gateway_reachable": True, "mock_query_ok": True, "root_page_ok": True,
        "static_asset_ok": False, "production_test_seam_sealed": True,
        "status": "FAIL",
    }
    regressed_result["status"] = "PASS" if container_smoke.compute_all_ok(regressed_result) else "FAIL"
    monkeypatch.setattr(container_smoke, "run_smoke", lambda image: regressed_result)
    exit_code = container_smoke.main(["--image", "simple-qna-rag:test"])
    assert exit_code == 1
    assert regressed_result["status"] == "FAIL"


def test_main_exits_zero_when_all_checks_including_static_asset_pass(monkeypatch):
    passing_result = {**_all_true_result(), "schema": "m43-container-smoke-v1",
                       "image": "simple-qna-rag:test"}
    passing_result["status"] = "PASS" if container_smoke.compute_all_ok(passing_result) else "FAIL"
    monkeypatch.setattr(container_smoke, "run_smoke", lambda image: passing_result)
    exit_code = container_smoke.main(["--image", "simple-qna-rag:test"])
    assert exit_code == 0
