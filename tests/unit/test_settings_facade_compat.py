"""M4.1 M3-01 — `config.py` legacy facade preserves pre-M4.1 name/type/value
contracts for all 41 fields plus the 42nd compat symbol `resolve_runtime_path`.

The pre-M4.1 `config.py` snapshot (name -> (type, default value)) is
reproduced here from the last commit before M4.1 rewrote the module, since
git history — not a live import — is the authoritative source for "what did
the facade look like before".
"""

import inspect
from pathlib import Path

from simple_qna_rag import config

_STR_TYPED_PATH_FIELDS = {
    "STATIC_DIR",
    "TEMPLATES_DIR",
    "VECTORSTORE_PATH",
    "DATA_DIR",
    "INTENT_MODEL_PATH",
}

_PRE_M41_SCALAR_DEFAULTS = {
    "COLLECTION_NAME": "document_collection",
    "EMBEDDING_MODEL_NAME": "BAAI/bge-m3",
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 200,
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_MODEL": "gpt-oss:20b",
    "RETRIEVAL_K": 4,
    "USE_MMR": True,
    "MMR_FETCH_K": 100,
    "MMR_K": 20,
    "MMR_LAMBDA": 0.5,
    "NORMALIZE_EMBEDDINGS": True,
    "USE_HYBRID_SEARCH": True,
    "BM25_TOP_K": 50,
    "DENSE_TOP_K": 50,
    "RRF_TOP_K": 50,
    "RRF_CONSTANT": 60,
    "USE_RERANKER": True,
    "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
    "RERANKER_TOP_K": 10,
    "USE_WEB_SEARCH": True,
    "WEB_SEARCH_MAX_RESULTS": 3,
    "WEB_SEARCH_TIMEOUT": 10,
    "WEB_SEARCH_REGION": "kr-kr",
    "MMR_VECTOR_SOURCE": "stored",
    "MMR_VECTOR_VALIDATION_SAMPLE": 3,
    "MMR_VECTOR_COSINE_FLOOR": 0.99,
    "MMR_EMBED_CACHE_MAX_ITEMS": 2048,
    "ROUTING_SIGNAL_OVERRIDE": True,
    "ROUTING_CORPUS_TOPIC_HINT": False,
    "ROUTING_CORPUS_TOPIC_HINT_MAX_ITEMS": 25,
    "ANSWER_TEMPLATE_MODE": "default",
    "INTENT_CONFIDENCE_FLOOR": 0.0,
    "BM25_TOKENIZER": "whitespace",
}


def test_project_root_is_path_and_unchanged_location():
    assert isinstance(config.PROJECT_ROOT, Path)
    assert config.PROJECT_ROOT == Path(__file__).resolve().parents[2]


def test_path_backed_facade_fields_are_str_typed():
    for name in _STR_TYPED_PATH_FIELDS:
        value = getattr(config, name)
        assert type(value) is str, f"{name} should be str, got {type(value)}"


def test_scalar_facade_defaults_match_pre_m41_snapshot():
    for name, expected in _PRE_M41_SCALAR_DEFAULTS.items():
        actual = getattr(config, name)
        assert actual == expected, f"{name}: expected {expected!r}, got {actual!r}"
        assert type(actual) is type(expected)


def test_prompt_template_is_non_empty_str():
    assert isinstance(config.PROMPT_TEMPLATE, str)
    assert "{context}" in config.PROMPT_TEMPLATE
    assert "{question}" in config.PROMPT_TEMPLATE


def test_resolve_runtime_path_signature_preserved():
    sig = inspect.signature(config.resolve_runtime_path)
    assert list(sig.parameters) == ["env_name", "default_path", "legacy_path", "environ"]
    assert sig.parameters["environ"].kind == inspect.Parameter.KEYWORD_ONLY
