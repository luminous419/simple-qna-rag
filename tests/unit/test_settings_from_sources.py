"""M4.1 M2-03 — base<-env<-CLI merge semantics (Design.md §4.3, §5.3)."""

from simple_qna_rag.settings import Settings


def test_base_environ_only_key_is_preserved():
    settings = Settings.from_sources(base_environ={"SIMPLE_QNA_RAG_CHUNK_SIZE": "777"})
    assert settings.CHUNK_SIZE == 777


def test_cli_overrides_win_over_base_environ_same_key():
    settings = Settings.from_sources(
        base_environ={"SIMPLE_QNA_RAG_CHUNK_SIZE": "777"},
        cli_overrides={"SIMPLE_QNA_RAG_CHUNK_SIZE": "4200"},
    )
    assert settings.CHUNK_SIZE == 4200


def test_cli_overrides_do_not_erase_unrelated_base_environ_keys():
    settings = Settings.from_sources(
        base_environ={
            "SIMPLE_QNA_RAG_CHUNK_SIZE": "777",
            "SIMPLE_QNA_RAG_RETRIEVAL_K": "9",
        },
        cli_overrides={"SIMPLE_QNA_RAG_CHUNK_SIZE": "4200"},
    )
    assert settings.CHUNK_SIZE == 4200
    assert settings.RETRIEVAL_K == 9


def test_m3_rollback_flags_survive_unrelated_overrides():
    settings = Settings.from_sources(
        base_environ={
            "SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE": "embed",
            "SIMPLE_QNA_RAG_ROUTING_SIGNAL_OVERRIDE": "0",
            "SIMPLE_QNA_RAG_ROUTING_CORPUS_TOPIC_HINT": "1",
            "SIMPLE_QNA_RAG_ANSWER_TEMPLATE_MODE": "intent",
        },
        cli_overrides={"SIMPLE_QNA_RAG_RETRIEVAL_K": "9"},
    )
    assert settings.MMR_VECTOR_SOURCE == "embed"
    assert settings.ROUTING_SIGNAL_OVERRIDE is False
    assert settings.ROUTING_CORPUS_TOPIC_HINT is True
    assert settings.ANSWER_TEMPLATE_MODE == "intent"
    assert settings.RETRIEVAL_K == 9


def test_from_env_is_alias_for_from_sources_base_environ_only():
    a = Settings.from_env({"SIMPLE_QNA_RAG_CHUNK_SIZE": "321"})
    b = Settings.from_sources(base_environ={"SIMPLE_QNA_RAG_CHUNK_SIZE": "321"})
    assert a.CHUNK_SIZE == b.CHUNK_SIZE == 321
