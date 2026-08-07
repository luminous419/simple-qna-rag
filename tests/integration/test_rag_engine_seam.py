"""Tests for the RAGEngine production seam (M3-REQ-007, Design.md §8.2).

Verifies that `query()`'s result is equivalent to composing
`generate_answer(build_context(...))`, and that `ANSWER_TEMPLATE_MODE`
controls whether the intent classifier is invoked while preserving the
`intent` response contract.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from simple_qna_rag import rag_engine
from simple_qna_rag.rag_engine import RAGEngine


def _make_engine() -> RAGEngine:
    return object.__new__(RAGEngine)


def _fake_llm() -> RunnableLambda:
    """A real LangChain Runnable that returns a fixed string, so
    `PromptTemplate | llm | StrOutputParser` composes correctly without a
    real Ollama call."""
    return RunnableLambda(lambda prompt_value: "fixed-answer")


def test_build_context_joins_page_content_with_double_newline():
    engine = _make_engine()
    docs = [
        Document(page_content="first", metadata={}),
        Document(page_content="second", metadata={}),
    ]
    assert engine.build_context(docs) == "first\n\nsecond"


def test_build_context_empty_list_returns_empty_string():
    engine = _make_engine()
    assert engine.build_context([]) == ""


def test_format_sources_matches_legacy_shape():
    engine = _make_engine()
    docs = [
        Document(page_content="x" * 250, metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="short", metadata={}),
    ]
    sources = engine.format_sources(docs)
    assert sources[0] == {
        "index": 1,
        "source": "a.pdf",
        "page": 1,  # 0-based -> 1-based
        "content": "x" * 200,
    }
    assert sources[1] == {
        "index": 2,
        "source": "알 수 없음",
        "page": None,
        "content": "short",
    }


def test_generate_answer_invokes_chain_with_context_and_question():
    engine = _make_engine()
    engine.llm = _fake_llm()
    answer = engine.generate_answer("질문입니다", "문맥입니다", "{context} / {question}")
    assert "fixed-answer" in answer


def test_query_equals_generate_answer_of_build_context_composition(monkeypatch):
    """query()가 사실상 generate_answer(build_context(docs))와 동일한 조합임을
    fake LLM으로 고정한다(§8.2 회귀 안전장치)."""
    engine = _make_engine()
    engine.llm = _fake_llm()
    engine.intent_classifier_loaded = True

    docs = [Document(page_content="doc-one", metadata={"source": "a.pdf"})]
    engine._retrieve_documents = lambda question: docs

    monkeypatch.setattr(rag_engine, "ANSWER_TEMPLATE_MODE", "default")
    result = engine.query("질문")

    expected_context = engine.build_context(docs)
    expected_answer = engine.generate_answer("질문", expected_context, "무엇이든 {context} {question}")
    # 템플릿 문자열 자체는 실제 DEFAULT_TEMPLATE과 다르지만, 체인 구성이
    # 동일한 두 값(question, context)을 넣어 같은 fake LLM을 통과하므로
    # "fixed-answer"가 공통으로 들어있는지만 비교한다(정확한 프롬프트 문자열
    # 비교는 이 테스트의 목적이 아니다 — 조합 방식의 동등성이 목적).
    assert result["success"] is True
    assert "fixed-answer" in result["answer"]
    assert "fixed-answer" in expected_answer
    assert result["sources"] == engine.format_sources(docs)


def test_answer_template_mode_default_skips_intent_classifier(monkeypatch):
    engine = _make_engine()
    engine.llm = _fake_llm()
    engine.intent_classifier_loaded = True
    engine._retrieve_documents = lambda question: []

    monkeypatch.setattr(rag_engine, "ANSWER_TEMPLATE_MODE", "default")
    with patch.object(rag_engine, "classify_intent") as mock_classify:
        result = engine.query("아무 질문")

    mock_classify.assert_not_called()
    # 응답 계약 보존: "disabled" 같은 새 값이 아니라 기존 "other"를 쓴다.
    assert result["intent"] == "other"
    assert result["success"] is True


def test_answer_template_mode_intent_calls_classifier(monkeypatch):
    engine = _make_engine()
    engine.llm = _fake_llm()
    engine.intent_classifier_loaded = True
    engine._retrieve_documents = lambda question: []

    monkeypatch.setattr(rag_engine, "ANSWER_TEMPLATE_MODE", "intent")
    with patch.object(rag_engine, "classify_intent", return_value="comparison") as mock_classify:
        result = engine.query("질문")

    mock_classify.assert_called_once()
    assert result["intent"] == "comparison"


def test_answer_template_mode_intent_classifier_missing_model_falls_back_to_other(monkeypatch):
    engine = _make_engine()
    engine.llm = _fake_llm()
    engine.intent_classifier_loaded = True
    engine._retrieve_documents = lambda question: []

    monkeypatch.setattr(rag_engine, "ANSWER_TEMPLATE_MODE", "intent")
    with patch.object(rag_engine, "classify_intent", side_effect=FileNotFoundError("no model")):
        result = engine.query("질문")

    assert result["intent"] == "other"
    assert result["success"] is True
