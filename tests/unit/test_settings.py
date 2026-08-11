"""M4.1 §4.5 — Settings valid/boundary/conflict fixtures."""

import pytest

from simple_qna_rag.settings import FIELD_SPECS, MODEL_VALIDATORS, Settings, SettingsError


def test_from_sources_no_args_succeeds_with_all_defaults():
    settings = Settings.from_sources()
    assert settings.CHUNK_SIZE == 1000
    assert settings.CHUNK_OVERLAP == 200
    assert settings.MMR_VECTOR_SOURCE == "stored"


def test_settings_is_frozen():
    settings = Settings.from_sources()
    with pytest.raises(Exception):
        settings.CHUNK_SIZE = 5


def test_settings_forbids_extra_fields():
    with pytest.raises(SettingsError):
        Settings.from_sources(base_environ={"SIMPLE_QNA_RAG_NOT_A_FIELD": "x"})


@pytest.mark.parametrize(
    "env_alias,raw,attr,expected",
    [
        ("SIMPLE_QNA_RAG_CHUNK_SIZE", "500", "CHUNK_SIZE", 500),
        ("SIMPLE_QNA_RAG_RETRIEVAL_K", "8", "RETRIEVAL_K", 8),
        ("SIMPLE_QNA_RAG_MMR_LAMBDA", "0.0", "MMR_LAMBDA", 0.0),
        ("SIMPLE_QNA_RAG_MMR_LAMBDA", "1.0", "MMR_LAMBDA", 1.0),
        ("SIMPLE_QNA_RAG_USE_MMR", "0", "USE_MMR", False),
        ("SIMPLE_QNA_RAG_USE_MMR", "yes", "USE_MMR", True),
        ("SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE", "embed", "MMR_VECTOR_SOURCE", "embed"),
        ("SIMPLE_QNA_RAG_BM25_TOKENIZER", "char2gram", "BM25_TOKENIZER", "char2gram"),
    ],
)
def test_valid_overrides_apply(env_alias, raw, attr, expected):
    settings = Settings.from_sources(base_environ={env_alias: raw})
    assert getattr(settings, attr) == expected


def test_empty_string_env_value_treated_as_unset():
    settings = Settings.from_sources(base_environ={"SIMPLE_QNA_RAG_CHUNK_SIZE": ""})
    assert settings.CHUNK_SIZE == 1000


@pytest.mark.parametrize(
    "env_alias,raw",
    [
        ("SIMPLE_QNA_RAG_CHUNK_SIZE", "not-an-int"),
        ("SIMPLE_QNA_RAG_MMR_LAMBDA", "not-a-float"),
        ("SIMPLE_QNA_RAG_USE_MMR", "maybe"),
        ("SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE", "bogus"),
        ("SIMPLE_QNA_RAG_BM25_TOKENIZER", "bogus"),
        ("SIMPLE_QNA_RAG_ANSWER_TEMPLATE_MODE", "bogus"),
    ],
)
def test_invalid_values_raise_settings_error_exit_2(env_alias, raw):
    with pytest.raises(SettingsError) as exc_info:
        Settings.from_sources(base_environ={env_alias: raw})
    assert exc_info.value.exit_code == 2


@pytest.mark.parametrize(
    "env_alias,raw",
    [
        ("SIMPLE_QNA_RAG_CHUNK_SIZE", "0"),
        ("SIMPLE_QNA_RAG_CHUNK_SIZE", "-1"),
        ("SIMPLE_QNA_RAG_RETRIEVAL_K", "0"),
        ("SIMPLE_QNA_RAG_MMR_LAMBDA", "-0.1"),
        ("SIMPLE_QNA_RAG_MMR_LAMBDA", "1.1"),
        ("SIMPLE_QNA_RAG_MMR_VECTOR_COSINE_FLOOR", "1.5"),
    ],
)
def test_boundary_violations_raise_settings_error(env_alias, raw):
    with pytest.raises(SettingsError):
        Settings.from_sources(base_environ={env_alias: raw})


@pytest.mark.parametrize(
    "overrides",
    [
        {"SIMPLE_QNA_RAG_CHUNK_OVERLAP": "1000", "SIMPLE_QNA_RAG_CHUNK_SIZE": "1000"},
        {"SIMPLE_QNA_RAG_MMR_K": "200", "SIMPLE_QNA_RAG_MMR_FETCH_K": "100"},
        {"SIMPLE_QNA_RAG_RERANKER_TOP_K": "100", "SIMPLE_QNA_RAG_RRF_TOP_K": "50"},
    ],
)
def test_cross_field_conflicts_raise_settings_error(overrides):
    with pytest.raises(SettingsError):
        Settings.from_sources(base_environ=overrides)


def test_unknown_env_key_exit_2():
    with pytest.raises(SettingsError) as exc_info:
        Settings.from_sources(base_environ={"SIMPLE_QNA_RAG_BOGUS_FIELD": "1"})
    assert exc_info.value.exit_code == 2


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "On"])
def test_bool_truth_table_true(value):
    settings = Settings.from_sources(base_environ={"SIMPLE_QNA_RAG_USE_MMR": value})
    assert settings.USE_MMR is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "Off"])
def test_bool_truth_table_false(value):
    settings = Settings.from_sources(base_environ={"SIMPLE_QNA_RAG_USE_MMR": value})
    assert settings.USE_MMR is False


def test_bool_truth_table_invalid_raises():
    with pytest.raises(SettingsError):
        Settings.from_sources(base_environ={"SIMPLE_QNA_RAG_USE_MMR": "sure"})


def test_model_validators_use_default_settings_all_pass():
    settings = Settings.from_sources()
    for mv in MODEL_VALIDATORS:
        mv.callable(settings)  # must not raise


def test_all_49_field_defaults_pass_validators():
    assert len(FIELD_SPECS) == 49
    Settings.from_sources()  # exercises every field's default through validation
