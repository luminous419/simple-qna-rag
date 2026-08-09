"""M2.5 Phase 4 runtime path resolution tests + M3 §3.4 flag helper tests."""

from pathlib import Path

import pytest

from simple_qna_rag import config
from simple_qna_rag.config import _env_bool, _env_enum, resolve_runtime_path
from evaluation.reporting import build_candidate_metadata


def test_environment_override_has_highest_priority(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    result = resolve_runtime_path(
        "TEST_PATH",
        tmp_path / "default",
        tmp_path / "legacy",
        environ={"TEST_PATH": str(configured)},
    )
    assert result == configured.resolve()


def test_new_default_used_when_present(tmp_path: Path) -> None:
    default = tmp_path / "default"
    legacy = tmp_path / "legacy"
    default.mkdir()

    assert resolve_runtime_path("TEST_PATH", default, legacy, environ={}) == default.resolve()


def test_new_and_legacy_conflict_stops_instead_of_merging(tmp_path: Path) -> None:
    default = tmp_path / "default"
    legacy = tmp_path / "legacy"
    default.mkdir()
    legacy.mkdir()

    with pytest.raises(RuntimeError, match="자동 병합하지 않으므로"):
        resolve_runtime_path("TEST_PATH", default, legacy, environ={})


def test_legacy_fallback_warns_when_only_legacy_exists(tmp_path: Path) -> None:
    default = tmp_path / "default"
    legacy = tmp_path / "legacy"
    legacy.mkdir()

    with pytest.warns(FutureWarning, match="기존 runtime 경로"):
        result = resolve_runtime_path("TEST_PATH", default, legacy, environ={})

    assert result == legacy.resolve()


def test_new_default_returned_when_neither_path_exists(tmp_path: Path) -> None:
    default = tmp_path / "default"
    legacy = tmp_path / "legacy"

    assert resolve_runtime_path("TEST_PATH", default, legacy, environ={}) == default.resolve()


# ---------------------------------------------------------------------------
# M3 §3.4 env var parsing helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "On"])
def test_env_bool_true_values(monkeypatch, value):
    monkeypatch.setenv("TEST_FLAG", value)
    assert _env_bool("TEST_FLAG", False) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "Off"])
def test_env_bool_false_values(monkeypatch, value):
    monkeypatch.setenv("TEST_FLAG", value)
    assert _env_bool("TEST_FLAG", True) is False


@pytest.mark.parametrize("value", ["", "garbage"])
def test_env_bool_invalid_values_raise(monkeypatch, value):
    # M4.1 Design.md §4.3 — `_env_bool` now delegates to the FieldSpec bool
    # parser, which is stricter than the pre-M4.1 "silently False" behavior:
    # unrecognized values raise instead of guessing (env unset/"" is the only
    # falsy special case handled separately by `Settings.from_sources()`;
    # `_env_bool` itself, called directly with a garbage/empty string, raises).
    monkeypatch.setenv("TEST_FLAG", value)
    with pytest.raises(ValueError):
        _env_bool("TEST_FLAG", True)


def test_env_bool_unset_uses_default(monkeypatch):
    monkeypatch.delenv("TEST_FLAG", raising=False)
    assert _env_bool("TEST_FLAG", True) is True
    assert _env_bool("TEST_FLAG", False) is False


def test_env_enum_valid_value(monkeypatch):
    monkeypatch.setenv("TEST_ENUM", "b")
    assert _env_enum("TEST_ENUM", "a", frozenset({"a", "b"})) == "b"


def test_env_enum_unset_uses_default(monkeypatch):
    monkeypatch.delenv("TEST_ENUM", raising=False)
    assert _env_enum("TEST_ENUM", "a", frozenset({"a", "b"})) == "a"


def test_env_enum_invalid_value_raises(monkeypatch):
    monkeypatch.setenv("TEST_ENUM", "bogus")
    with pytest.raises(ValueError):
        _env_enum("TEST_ENUM", "a", frozenset({"a", "b"}))


def test_mmr_vector_source_default_is_stored():
    # Phase 2 채택 결과: 기본값은 "stored"다(gate 통과, Design.md §13.3 rollback
    # 스위치는 SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE=embed).
    assert config.MMR_VECTOR_SOURCE == "stored"


def test_routing_signal_override_default_is_true():
    # Phase 3 채택 결과: 기본값은 True다(gate 통과, Design.md §13.3 rollback
    # 스위치는 SIMPLE_QNA_RAG_ROUTING_SIGNAL_OVERRIDE=0).
    assert config.ROUTING_SIGNAL_OVERRIDE is True


def test_answer_template_mode_default_is_default():
    assert config.ANSWER_TEMPLATE_MODE == "default"


def test_bm25_tokenizer_default_is_whitespace():
    assert config.BM25_TOKENIZER == "whitespace"


# ---------------------------------------------------------------------------
# candidate ID regex (Design.md §3.1) via build_candidate_metadata()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate_id",
    [
        "m3-p0-baseline-check",
        "m3-p1-evaluator-v2",
        "m3-p2a-stored-vector",
        "m3-p3a-signal-override",
        "m3-p3c-two-stage",
        "m3-p4-intent-ab",
        "m3-p5a-char2gram",
        "m3-final",
    ],
)
def test_candidate_id_regex_accepts_valid_ids(candidate_id):
    result = build_candidate_metadata(candidate_id)
    assert result["candidate_id"] == candidate_id


@pytest.mark.parametrize(
    "candidate_id",
    [
        "m3-p7-out-of-range",
        "m3-p2",  # 세그먼트 없음
        "m3-p2a",  # 세그먼트 없음
        "m3-p2-r2-old-style",  # 'r2' round 표기 금지 규칙과 무관하게 세그먼트는 있지만
        "M3-P2A-STORED-VECTOR",  # 대문자 금지
        "m2-p2a-stored-vector",  # m3 prefix 필수
        "",
    ],
)
def test_candidate_id_regex_rejects_invalid_ids(candidate_id):
    if candidate_id == "m3-p2-r2-old-style":
        # 이 값 자체는 정규식상 유효한 형태(segment 존재)이므로 실제로는
        # 통과한다 — "r2/r3 표기 금지"는 사람이 지키는 명명 규칙이지 정규식
        # 계약이 아니다. 이 케이스는 별도로 확인한다.
        result = build_candidate_metadata(candidate_id)
        assert result["candidate_id"] == candidate_id
        return
    with pytest.raises(ValueError):
        build_candidate_metadata(candidate_id)


def test_candidate_id_none_returns_null_block():
    result = build_candidate_metadata(None)
    assert result == {
        "candidate_id": None,
        "label": None,
        "phase": None,
        "baseline_ref": "evaluation/baselines/m2_initial.json",
        "baseline_dataset_sha256": "61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a",
        "notes": None,
    }


def test_candidate_id_final_has_phase_six():
    result = build_candidate_metadata("m3-final")
    assert result["phase"] == 6


def test_candidate_id_phase_extracted_correctly():
    assert build_candidate_metadata("m3-p3a-signal-override")["phase"] == 3
