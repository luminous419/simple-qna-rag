"""M4.1 §5.2/§5.3 — `simple-qna-rag-web --check-config` (REQ-002.5)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_check_config(tmp_path: Path, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, **(extra_env or {})}
    return subprocess.run(
        [sys.executable, "-m", "simple_qna_rag.cli.web", "--check-config"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_check_config_valid_settings_exit_0_stdout_json(tmp_path: Path) -> None:
    result = _run_check_config(tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["CHUNK_SIZE"] == 1000
    assert payload["MMR_VECTOR_SOURCE"] == "stored"


def test_check_config_never_imports_model_or_engine(tmp_path: Path) -> None:
    result = _run_check_config(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "langchain" not in result.stdout.lower()


def test_check_config_invalid_settings_exit_2_stderr(tmp_path: Path) -> None:
    result = _run_check_config(tmp_path, {"SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE": "bogus"})
    assert result.returncode == 2
    assert result.stderr.strip() != ""
    assert result.stdout.strip() == ""


def test_check_config_redacts_absolute_paths(tmp_path: Path) -> None:
    result = _run_check_config(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "/Users/" not in result.stdout
    assert "/home/" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["VECTORSTORE_PATH"]["redacted"] is True
    assert payload["VECTORSTORE_PATH"]["kind"] == "path"
    assert payload["PROJECT_ROOT"]["redacted"] is True


def test_check_config_does_not_leak_secrets_field_names(tmp_path: Path) -> None:
    result = _run_check_config(tmp_path)
    lowered = result.stdout.lower()
    for forbidden in ("password", "api_key", "secret", "credential"):
        assert forbidden not in lowered


# ---------------------------------------------------------------------------
# CR-I1-MAJ-01 closure — schema-based disclosure policy adversarial matrix.
# Every plain `str` FIELD_SPEC (URLs, prompts, model names) must be redacted
# to non-reversible metadata regardless of what value an operator injects via
# env; only bool/int/float/Literal-enum fields may print their literal value.
# ---------------------------------------------------------------------------


def test_check_config_redacts_credential_bearing_ollama_base_url(tmp_path: Path) -> None:
    secret_password = "S3cr3t-Adversarial-Pw-9f13"
    result = _run_check_config(
        tmp_path,
        {"SIMPLE_QNA_RAG_OLLAMA_BASE_URL": f"http://user:{secret_password}@localhost:11434"},
    )
    assert result.returncode == 0, result.stderr
    assert secret_password not in result.stdout
    assert "user:" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["OLLAMA_BASE_URL"] == {"redacted": True, "kind": "string", "length": len(
        f"http://user:{secret_password}@localhost:11434"
    )}


def test_check_config_redacts_prompt_template_injected_secret(tmp_path: Path) -> None:
    secret_marker = "TOP_SECRET_VALUE_4b7e"
    prompt = f"credential={secret_marker} {{context}} {{question}}"
    result = _run_check_config(tmp_path, {"SIMPLE_QNA_RAG_PROMPT_TEMPLATE": prompt})
    assert result.returncode == 0, result.stderr
    assert secret_marker not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["PROMPT_TEMPLATE"] == {"redacted": True, "kind": "string", "length": len(prompt)}


def test_check_config_redacts_token_like_ollama_model_value(tmp_path: Path) -> None:
    token = "ghp_AdversarialTokenLike1234567890abcd"
    result = _run_check_config(tmp_path, {"SIMPLE_QNA_RAG_OLLAMA_MODEL": token})
    assert result.returncode == 0, result.stderr
    assert token not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["OLLAMA_MODEL"]["redacted"] is True


def test_check_config_combined_credential_url_and_prompt_adversarial(tmp_path: Path) -> None:
    """Reproduces the exact CR-I1-MAJ-01 repro command as a regression test."""
    result = _run_check_config(
        tmp_path,
        {
            "SIMPLE_QNA_RAG_PROMPT_TEMPLATE": "credential=TOP_SECRET_VALUE {context} {question}",
            "SIMPLE_QNA_RAG_OLLAMA_BASE_URL": "http://user:password@localhost:11434",
        },
    )
    assert result.returncode == 0, result.stderr
    for forbidden in ("TOP_SECRET_VALUE", "password", "user:password"):
        assert forbidden not in result.stdout


def test_check_config_bounded_domain_fields_still_print_values(tmp_path: Path) -> None:
    """Sanity check that the allowlist isn't collapsed to all-redacted: closed
    domain fields (bool/int/float/Literal enum) remain usable as literal
    values for operators verifying config without a secret risk."""
    result = _run_check_config(tmp_path, {"SIMPLE_QNA_RAG_CHUNK_SIZE": "1234"})
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["CHUNK_SIZE"] == 1234
    assert payload["USE_MMR"] is True
    assert payload["MMR_VECTOR_SOURCE"] == "stored"
