"""agent.py `_decide_tool()` 신호-우선 판정 정책 테스트 (M3-REQ-004,
Design.md §7.4). 모델/네트워크를 쓰지 않는다 — `_llm_decide_tool()`만
stub하거나(§7.4 신호 stub 12칸), 실제 `classify_explicit_signal()` 경로를
그대로 통과시킨다(§7.4 단순화 Cycle 1 실제 classifier S1~S12).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from simple_qna_rag import agent


# ---------------------------------------------------------------------------
# 신호 stub 12칸 — classify_explicit_signal() 자체를 stub하고
# _llm_decide_tool()도 stub한다.
# ---------------------------------------------------------------------------


def _with_signal(monkeypatch, signal):
    monkeypatch.setattr(agent, "ROUTING_SIGNAL_OVERRIDE", True)
    monkeypatch.setattr(agent, "classify_explicit_signal", lambda q: signal)


class TestSignalStubMatrix:
    def test_1_web_llm_returns_web_search_with_query(self, monkeypatch):
        _with_signal(monkeypatch, agent.ExplicitSignal.WEB)
        with patch.object(agent, "_llm_decide_tool", return_value=("web_search", "환율")) as stub:
            result = agent._decide_tool("질문")
        assert result == ("web_search", "환율")
        stub.assert_called_once()

    def test_2_web_llm_returns_document_qa_route_is_corrected(self, monkeypatch):
        _with_signal(monkeypatch, agent.ExplicitSignal.WEB)
        with patch.object(agent, "_llm_decide_tool", return_value=("document_qa", "질문")):
            tool_name, query = agent._decide_tool("질문")
        assert tool_name == "web_search"
        assert query == agent.extract_web_search_query("질문")

    def test_3_web_llm_no_tool_still_web_search(self, monkeypatch):
        _with_signal(monkeypatch, agent.ExplicitSignal.WEB)
        with patch.object(agent, "_llm_decide_tool", return_value=(None, None)):
            tool_name, query = agent._decide_tool("질문")
        assert tool_name == "web_search"
        assert query == agent.extract_web_search_query("질문")

    def test_4_web_llm_empty_query_is_backfilled(self, monkeypatch):
        _with_signal(monkeypatch, agent.ExplicitSignal.WEB)
        with patch.object(agent, "_llm_decide_tool", return_value=("web_search", "")):
            tool_name, query = agent._decide_tool("질문")
        assert tool_name == "web_search"
        assert query == agent.extract_web_search_query("질문")

    def test_5_web_llm_exception_still_web_search(self, monkeypatch):
        _with_signal(monkeypatch, agent.ExplicitSignal.WEB)
        with patch.object(agent, "_llm_decide_tool", side_effect=RuntimeError("boom")):
            tool_name, query = agent._decide_tool("질문")
        assert tool_name == "web_search"
        assert query == agent.extract_web_search_query("질문")

    def test_6_document_llm_never_called(self, monkeypatch):
        _with_signal(monkeypatch, agent.ExplicitSignal.DOCUMENT)
        with patch.object(agent, "_llm_decide_tool") as stub:
            result = agent._decide_tool("질문")
        assert result == ("document_qa", "질문")
        stub.assert_not_called()

    def test_7_document_llm_would_raise_but_never_called(self, monkeypatch):
        _with_signal(monkeypatch, agent.ExplicitSignal.DOCUMENT)
        with patch.object(agent, "_llm_decide_tool", side_effect=RuntimeError("boom")) as stub:
            result = agent._decide_tool("질문")
        assert result == ("document_qa", "질문")
        stub.assert_not_called()

    def test_8_none_llm_document_qa_passthrough(self, monkeypatch):
        _with_signal(monkeypatch, agent.ExplicitSignal.NONE)
        with patch.object(agent, "_llm_decide_tool", return_value=("document_qa", "질문")):
            result = agent._decide_tool("질문")
        assert result == ("document_qa", "질문")

    def test_9_none_llm_no_tool_passthrough(self, monkeypatch):
        _with_signal(monkeypatch, agent.ExplicitSignal.NONE)
        with patch.object(agent, "_llm_decide_tool", return_value=(None, None)):
            result = agent._decide_tool("질문")
        assert result == (None, None)

    def test_10_none_llm_exception_propagates(self, monkeypatch):
        _with_signal(monkeypatch, agent.ExplicitSignal.NONE)
        with patch.object(agent, "_llm_decide_tool", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                agent._decide_tool("질문")

    def test_11_flag_off_uses_llm_decision_regardless_of_signal(self, monkeypatch):
        monkeypatch.setattr(agent, "ROUTING_SIGNAL_OVERRIDE", False)
        monkeypatch.setattr(agent, "classify_explicit_signal", lambda q: agent.ExplicitSignal.WEB)
        with patch.object(agent, "_llm_decide_tool", return_value=("document_qa", "질문")) as stub:
            result = agent._decide_tool("아무 질문")
        assert result == ("document_qa", "질문")
        stub.assert_called_once()

    def test_12_signal_classifier_exception_is_isolated_as_none(self, monkeypatch):
        monkeypatch.setattr(agent, "ROUTING_SIGNAL_OVERRIDE", True)

        def _boom(q):
            raise RuntimeError("signal classifier bug")

        monkeypatch.setattr(agent, "classify_explicit_signal", _boom)
        with patch.object(agent, "_llm_decide_tool", return_value=("document_qa", "질문")) as stub:
            result = agent._decide_tool("질문")
        assert result == ("document_qa", "질문")
        stub.assert_called_once()


# ---------------------------------------------------------------------------
# S1~S12 — 단순화 Cycle 1 실제 classifier (classify_explicit_signal은 stub
# 하지 않는다. _llm_decide_tool만 stub한다.)
# ---------------------------------------------------------------------------


@pytest.fixture
def signal_override_on(monkeypatch):
    monkeypatch.setattr(agent, "ROUTING_SIGNAL_OVERRIDE", True)


class TestRealClassifierMatrix:
    def test_s1_web_command_llm_exception_still_web(self, signal_override_on, monkeypatch):
        q = "웹에서 검색해줘"
        with patch.object(agent, "_llm_decide_tool", side_effect=RuntimeError("boom")):
            tool_name, query = agent._decide_tool(q)
        assert tool_name == "web_search"
        assert query == agent.extract_web_search_query(q)

    def test_s2_direct_command_llm_no_tool_still_web(self, signal_override_on, monkeypatch):
        q = "구글링해서 알려줘"
        with patch.object(agent, "_llm_decide_tool", return_value=(None, None)):
            tool_name, query = agent._decide_tool(q)
        assert tool_name == "web_search"
        assert query == agent.extract_web_search_query(q)

    def test_s3_unicode_boundary_llm_exception_still_web(self, signal_override_on, monkeypatch):
        q = "질문:웹에서 검색해줘"
        with patch.object(agent, "_llm_decide_tool", side_effect=RuntimeError("boom")):
            tool_name, query = agent._decide_tool(q)
        assert tool_name == "web_search"
        assert query == agent.extract_web_search_query(q)

    def test_s4_negation_document_scope_llm_not_called(self, signal_override_on, monkeypatch):
        q = "웹 검색은 하지 말고 문서로 답해줘"
        with patch.object(agent, "_llm_decide_tool") as stub:
            result = agent._decide_tool(q)
        assert result == ("document_qa", q)
        stub.assert_not_called()

    def test_s5_quoted_document_scope_llm_not_called(self, signal_override_on, monkeypatch):
        q = '"웹 검색"의 뜻을 문서에서 알려줘'
        with patch.object(agent, "_llm_decide_tool") as stub:
            result = agent._decide_tool(q)
        assert result == ("document_qa", q)
        stub.assert_not_called()

    def test_s6_document_scope_llm_not_called(self, signal_override_on, monkeypatch):
        q = "이 문서에서 관련 내용을 찾아줘"
        with patch.object(agent, "_llm_decide_tool") as stub:
            result = agent._decide_tool(q)
        assert result == ("document_qa", q)
        stub.assert_not_called()

    def test_s7_general_response_predicate_llm_result_passes_through(self, signal_override_on, monkeypatch):
        q = "웹검색으로 최신 환율 알려줘"
        with patch.object(agent, "_llm_decide_tool", return_value=("document_qa", q)):
            result = agent._decide_tool(q)
        assert result == ("document_qa", q)

    def test_s8_relative_clause_minimal_pair_1_exception_propagates(self, signal_override_on, monkeypatch):
        q = "웹검색에서 사용하는 API 구조 알려줘"
        with patch.object(agent, "_llm_decide_tool", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                agent._decide_tool(q)

    def test_s9_relative_clause_minimal_pair_2_no_tool_passes_through(self, signal_override_on, monkeypatch):
        q = "구글에서 사용하는 검색 기술 알려줘"
        with patch.object(agent, "_llm_decide_tool", return_value=(None, None)):
            result = agent._decide_tool(q)
        assert result == (None, None)

    def test_s10_search_topic_question_llm_result_passes_through(self, signal_override_on, monkeypatch):
        q = "웹검색 방법 알려줘"
        with patch.object(agent, "_llm_decide_tool", return_value=("document_qa", q)):
            result = agent._decide_tool(q)
        assert result == ("document_qa", q)

    def test_s11_compound_word_boundary_exception_propagates(self, signal_override_on, monkeypatch):
        q = "freewebsearch 사용법 알려줘"
        with patch.object(agent, "_llm_decide_tool", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                agent._decide_tool(q)

    def test_s12_flag_off_llm_decision_used_regardless(self, monkeypatch):
        monkeypatch.setattr(agent, "ROUTING_SIGNAL_OVERRIDE", False)
        q = "웹에서 검색해줘"  # would be WEB if override were on
        with patch.object(agent, "_llm_decide_tool", return_value=("document_qa", q)) as stub:
            result = agent._decide_tool(q)
        assert result == ("document_qa", q)
        stub.assert_called_once()


# ---------------------------------------------------------------------------
# route_query() 수준의 기존 폴백 4경로 무회귀(NONE 신호) + WEB 실패 재시도
# ---------------------------------------------------------------------------


class TestRouteQueryFallbackPreserved:
    def test_none_signal_llm_exception_falls_back_to_keyword_router(self, signal_override_on, monkeypatch):
        monkeypatch.setattr(agent, "USE_WEB_SEARCH", True)
        with patch.object(agent, "_llm_decide_tool", side_effect=RuntimeError("ollama down")), patch(
            "simple_qna_rag.agent.keyword_fallback_route",
            return_value={"answer": "폴백", "sources": [], "success": True, "search_type": "document_qa"},
        ) as mock_fallback:
            result = agent.route_query("RAG에서 MMR이 뭐야?")
        mock_fallback.assert_called_once()
        assert result["search_type"] == "document_qa"

    def test_none_signal_llm_no_tool_falls_back_to_keyword_router(self, signal_override_on, monkeypatch):
        monkeypatch.setattr(agent, "USE_WEB_SEARCH", True)
        with patch.object(agent, "_llm_decide_tool", return_value=(None, None)), patch(
            "simple_qna_rag.agent.keyword_fallback_route",
            return_value={"answer": "폴백", "sources": [], "success": True, "search_type": "document_qa"},
        ) as mock_fallback:
            result = agent.route_query("RAG에서 MMR이 뭐야?")
        mock_fallback.assert_called_once()

    def test_web_signal_search_failure_retries_document_qa(self, signal_override_on, monkeypatch):
        monkeypatch.setattr(agent, "USE_WEB_SEARCH", True)
        with patch.object(agent, "_llm_decide_tool", return_value=("web_search", "환율")), patch(
            "simple_qna_rag.agent.web_search_tool"
        ) as mock_web_tool, patch("simple_qna_rag.agent.rag_tool") as mock_rag_tool:
            mock_web_tool.func.return_value = {"success": False, "answer": "실패", "sources": []}
            mock_rag_tool.func.return_value = {
                "answer": "문서 답변",
                "sources": [],
                "success": True,
                "intent": "explanation",
            }
            result = agent.route_query("웹에서 검색해줘")

        mock_rag_tool.func.assert_called_once_with("웹에서 검색해줘")
        assert result["search_type"] == "document_qa"
