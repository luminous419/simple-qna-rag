#!/usr/bin/env python3
"""CR-I2-MAJ-01 preflight — verify the Ollama endpoint is reachable, report
its version, and confirm the required model is registered before the M3 live
regression gate depends on it mid-run. Uses only the stdlib (`urllib`) so it
carries no extra dependency risk for a CI-only preflight step.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def _get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed http(s) CI endpoint
        return json.loads(resp.read().decode("utf-8"))


def check_ollama_health(base_url: str, required_model: str) -> tuple[str, list[str]]:
    """`base_url`의 Ollama `/api/version`과 `/api/tags`를 조회해 endpoint가
    살아있고 `required_model`이 등록되어 있는지 확인한다. 실패하면
    `RuntimeError`를 던진다. 성공하면 (version, models) 튜플을 반환한다."""
    try:
        version_payload = _get_json(f"{base_url}/api/version")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Ollama endpoint에 연결할 수 없습니다({base_url}): {exc}") from exc
    version = version_payload.get("version", "unknown")

    try:
        tags_payload = _get_json(f"{base_url}/api/tags")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Ollama 모델 목록을 가져오지 못했습니다({base_url}): {exc}") from exc
    models = [m.get("name") for m in tags_payload.get("models", [])]
    if required_model not in models:
        raise RuntimeError(
            f"필요한 모델 {required_model!r}이 Ollama에 등록되어 있지 않습니다. "
            f"사용 가능한 모델: {models}"
        )
    return version, models


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    base_url = argv[0] if len(argv) > 0 else os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    required_model = (
        argv[1] if len(argv) > 1 else os.environ.get("SIMPLE_QNA_RAG_OLLAMA_MODEL", "gpt-oss:20b")
    )

    try:
        version, models = check_ollama_health(base_url, required_model)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Ollama version={version} required_model={required_model} models={models}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
