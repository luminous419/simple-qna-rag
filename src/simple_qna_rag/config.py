"""RAG 시스템 설정 파일 — M4.1 legacy facade (Design.md §4.4).

이 모듈은 `simple_qna_rag.settings.Settings`가 유일한 원본(single source)인
값을 그대로 노출하는 호환 facade다. REQ-002.3에 따라 기존 공개 상수 이름과
타입을 한 release 동안 보존한다. 직접 로직/검증을 추가하지 않는다 —
값/검증/파싱은 모두 `settings.py`의 `FIELD_SPECS`/`MODEL_VALIDATORS`가
유일한 원본이다.
"""

from pathlib import Path

from simple_qna_rag.settings import (
    FIELD_SPECS,
    ENV_PREFIX,
    SettingsError,
    _parse_bool,
    _parse_runtime_path,
    facade_value,
    get_settings,
    reset_settings_cache,
)

__all__ = [
    "resolve_runtime_path",
    "reset_settings_cache",
    *[spec.name for spec in FIELD_SPECS],
]


def resolve_runtime_path(
    env_name: str,
    default_path: Path,
    legacy_path: Path,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    """레거시 호환 wrapper — 원 시그니처와 `env > new default > legacy fallback`
    동작을 그대로 보존한다(Design §4.3/§4.4, m4-01). `FIELD_SPECS`의 path
    parser는 이 함수를 호출하지 않는다(별도 private `_parse_runtime_path`).
    """
    import os
    import warnings

    values = os.environ if environ is None else environ
    configured = values.get(env_name)
    if configured:
        return Path(configured).expanduser().resolve()

    default_path = default_path.resolve()
    legacy_path = legacy_path.resolve()
    if default_path.exists() and legacy_path.exists():
        raise RuntimeError(
            f"새 runtime 경로와 기존 경로가 모두 존재합니다: "
            f"{default_path}, {legacy_path}. 자동 병합하지 않으므로 하나를 명시적으로 선택하세요."
        )
    if default_path.exists() or not legacy_path.exists():
        return default_path

    warnings.warn(
        f"기존 runtime 경로 {legacy_path}를 사용합니다. {default_path}로 이전하세요.",
        FutureWarning,
        stacklevel=2,
    )
    return legacy_path


def _env_bool(name: str, default: bool) -> bool:
    """순수 파서 — env를 읽지 않는다(REQ-002.2). 기존 import 이름만 유지."""
    import os

    raw = os.environ.get(name)
    if raw is None:
        return default
    return _parse_bool(raw)


def _env_enum(name: str, default: str, allowed: frozenset[str]) -> str:
    import os

    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw not in allowed:
        raise ValueError(f"{name}={raw!r}은(는) {sorted(allowed)} 중 하나여야 합니다")
    return raw


_settings = get_settings()

for _spec in FIELD_SPECS:
    globals()[_spec.name] = facade_value(_settings, _spec)

del _spec
