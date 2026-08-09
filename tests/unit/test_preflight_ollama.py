"""CR-I2-MAJ-01 — Ollama model/version/endpoint health preflight."""

import json
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import preflight_ollama as po  # noqa: E402


def _fake_urlopen(responses: dict[str, dict]):
    def _urlopen(url, timeout=10.0):  # noqa: ARG001 - matches urllib signature
        for prefix, payload in responses.items():
            if url.startswith(prefix):
                body = json.dumps(payload).encode("utf-8")
                cm = mock.MagicMock()
                cm.__enter__.return_value.read.return_value = body
                return cm
        raise AssertionError(f"unexpected URL: {url}")

    return _urlopen


def test_check_ollama_health_returns_version_and_models_when_required_model_present():
    responses = {
        "http://x/api/version": {"version": "0.32.0"},
        "http://x/api/tags": {"models": [{"name": "gpt-oss:20b"}, {"name": "llama3.1:latest"}]},
    }
    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen(responses)):
        version, models = po.check_ollama_health("http://x", "gpt-oss:20b")

    assert version == "0.32.0"
    assert models == ["gpt-oss:20b", "llama3.1:latest"]


def test_check_ollama_health_raises_when_required_model_absent():
    responses = {
        "http://x/api/version": {"version": "0.32.0"},
        "http://x/api/tags": {"models": [{"name": "llama3.1:latest"}]},
    }
    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen(responses)):
        with pytest.raises(RuntimeError, match="gpt-oss:20b"):
            po.check_ollama_health("http://x", "gpt-oss:20b")


def test_check_ollama_health_raises_when_version_endpoint_unreachable():
    def _urlopen(url, timeout=10.0):  # noqa: ARG001
        raise urllib.error.URLError("connection refused")

    with mock.patch("urllib.request.urlopen", side_effect=_urlopen):
        with pytest.raises(RuntimeError, match="연결할 수 없습니다"):
            po.check_ollama_health("http://x", "gpt-oss:20b")


def test_check_ollama_health_raises_when_tags_endpoint_fails_after_version_ok():
    def _urlopen(url, timeout=10.0):  # noqa: ARG001
        if url.startswith("http://x/api/version"):
            cm = mock.MagicMock()
            cm.__enter__.return_value.read.return_value = json.dumps({"version": "0.32.0"}).encode()
            return cm
        raise urllib.error.URLError("timeout")

    with mock.patch("urllib.request.urlopen", side_effect=_urlopen):
        with pytest.raises(RuntimeError, match="모델 목록을 가져오지 못했습니다"):
            po.check_ollama_health("http://x", "gpt-oss:20b")


def test_main_exit_0_when_healthy(capsys):
    responses = {
        "http://x/api/version": {"version": "0.32.0"},
        "http://x/api/tags": {"models": [{"name": "gpt-oss:20b"}]},
    }
    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen(responses)):
        exit_code = po.main(["http://x", "gpt-oss:20b"])

    assert exit_code == 0
    assert "Ollama version=0.32.0" in capsys.readouterr().out


def test_main_exit_1_when_unhealthy(capsys):
    def _urlopen(url, timeout=10.0):  # noqa: ARG001
        raise urllib.error.URLError("down")

    with mock.patch("urllib.request.urlopen", side_effect=_urlopen):
        exit_code = po.main(["http://x", "gpt-oss:20b"])

    assert exit_code == 1
    assert "연결할 수 없습니다" in capsys.readouterr().err


def test_main_uses_env_defaults_when_no_argv(monkeypatch):
    responses = {
        "http://env-host/api/version": {"version": "1.0"},
        "http://env-host/api/tags": {"models": [{"name": "env-model"}]},
    }
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-host")
    monkeypatch.setenv("SIMPLE_QNA_RAG_OLLAMA_MODEL", "env-model")
    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen(responses)):
        assert po.main([]) == 0
