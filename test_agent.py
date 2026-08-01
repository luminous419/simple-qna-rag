#!/usr/bin/env python3
"""
agent.py의 route_query() 오케스트레이션 테스트 (pytest, LLM/Ollama 호출 없음)

_decide_tool()과 실제 도구 실행 함수(web_search_tool.func/rag_tool.func)를 mock으로
대체하여, Agent 성공/실패 시 분기(웹검색 성공, 웹검색 실패 후 문서QA 재시도, 도구 선택
실패, Agent 예외 발생 시 폴백)만 검증합니다. 실제 도구 "선택 정확도"는
test_agent_routing.py(라이브 LLM 필요)에서 다룹니다.
"""

from unittest.mock import MagicMock, patch

import agent


def _web_result(success=True):
    return {
        "answer": "웹 검색 결과",
        "sources": [{"index": 1, "source": "https://example.com"}],
        "success": success,
        "search_type": "web_search",
    }


def _rag_result():
    return {"answer": "문서 기반 답변", "sources": [], "success": True, "intent": "explanation"}


class TestRouteQueryWebSearchDisabled:
    def test_bypasses_agent_entirely_when_disabled(self):
        mock_engine = MagicMock()
        mock_engine.query.return_value = _rag_result()
        with patch("agent.USE_WEB_SEARCH", False), patch(
            "agent.get_rag_engine", return_value=mock_engine
        ) as mock_get_engine, patch("agent._decide_tool") as mock_decide:
            result = agent.route_query("아무 질문")

        mock_decide.assert_not_called()
        mock_get_engine.assert_called_once()
        assert result["search_type"] == "document_qa"


class TestRouteQueryToolSelection:
    def test_web_search_tool_selected_and_succeeds(self):
        with patch("agent.USE_WEB_SEARCH", True), patch(
            "agent._decide_tool", return_value=("web_search", "정제된 검색어")
        ), patch.object(
            agent.web_search_tool, "func", return_value=_web_result(success=True)
        ) as mock_web, patch.object(agent.rag_tool, "func") as mock_rag:
            result = agent.route_query("오늘 날씨 웹에서 찾아줘")

        mock_web.assert_called_once_with("정제된 검색어")
        mock_rag.assert_not_called()
        assert result["search_type"] == "web_search"
        assert result["success"] is True

    def test_web_search_failure_falls_back_to_document_qa(self):
        with patch("agent.USE_WEB_SEARCH", True), patch(
            "agent._decide_tool", return_value=("web_search", "정제된 검색어")
        ), patch.object(
            agent.web_search_tool, "func", return_value=_web_result(success=False)
        ) as mock_web, patch.object(
            agent.rag_tool, "func", return_value=_rag_result()
        ) as mock_rag:
            result = agent.route_query("오늘 날씨 웹에서 찾아줘")

        mock_web.assert_called_once()
        # document_qa 재시도는 LLM이 재작성한 검색어가 아니라 원본 질문을 그대로 써야 함
        mock_rag.assert_called_once_with("오늘 날씨 웹에서 찾아줘")
        assert result["search_type"] == "document_qa"

    def test_document_qa_tool_selected(self):
        with patch("agent.USE_WEB_SEARCH", True), patch(
            "agent._decide_tool", return_value=("document_qa", "RAG에서 MMR이 뭐야?")
        ), patch.object(
            agent.rag_tool, "func", return_value=_rag_result()
        ) as mock_rag, patch.object(agent.web_search_tool, "func") as mock_web:
            result = agent.route_query("RAG에서 MMR이 뭐야?")

        mock_rag.assert_called_once_with("RAG에서 MMR이 뭐야?")
        mock_web.assert_not_called()
        assert result["search_type"] == "document_qa"


class TestRouteQueryFallback:
    def test_decide_tool_exception_falls_back_to_keyword_router(self):
        with patch("agent.USE_WEB_SEARCH", True), patch(
            "agent._decide_tool", side_effect=RuntimeError("ollama down")
        ), patch(
            "agent.keyword_fallback_route", return_value=_web_result()
        ) as mock_fallback:
            result = agent.route_query("아무 질문")

        mock_fallback.assert_called_once_with("아무 질문")
        assert result["search_type"] == "web_search"

    def test_no_tool_selected_falls_back_to_keyword_router(self):
        with patch("agent.USE_WEB_SEARCH", True), patch(
            "agent._decide_tool", return_value=(None, None)
        ), patch(
            "agent.keyword_fallback_route", return_value=_web_result()
        ) as mock_fallback:
            result = agent.route_query("아무 질문")

        mock_fallback.assert_called_once_with("아무 질문")
        assert result["search_type"] == "web_search"


class TestRouteQueryFallbackDocumentQaRetry:
    """
    agent.py가 키워드 라우터로 폴백한 뒤, 그 키워드 라우터가 고른 웹검색마저 실패하는
    이중 장애 조합에서도 document_qa로 최종 복구되는지 검증한다. keyword_fallback_route를
    mock으로 대체하지 않고 query_router.py의 실제 재시도 로직까지 통과시킨다.
    """

    def test_decide_tool_exception_with_keyword_web_search_failure_retries_document_qa(self):
        mock_engine = MagicMock()
        mock_engine.query.return_value = _rag_result()
        with patch("agent.USE_WEB_SEARCH", True), patch(
            "query_router.USE_WEB_SEARCH", True
        ), patch(
            "agent._decide_tool", side_effect=RuntimeError("ollama down")
        ), patch(
            "query_router.search_and_format",
            return_value={
                "answer": "실패",
                "sources": [],
                "success": False,
                "search_type": "web_search",
            },
        ) as mock_search, patch(
            "query_router.get_rag_engine", return_value=mock_engine
        ):
            result = agent.route_query("오늘 날씨 웹에서 찾아줘")

        mock_search.assert_called_once()
        mock_engine.query.assert_called_once_with("오늘 날씨 웹에서 찾아줘")
        assert result["search_type"] == "document_qa"
        assert result["success"] is True

    def test_no_tool_selected_with_keyword_web_search_failure_retries_document_qa(self):
        mock_engine = MagicMock()
        mock_engine.query.return_value = _rag_result()
        with patch("agent.USE_WEB_SEARCH", True), patch(
            "query_router.USE_WEB_SEARCH", True
        ), patch(
            "agent._decide_tool", return_value=(None, None)
        ), patch(
            "query_router.search_and_format",
            return_value={
                "answer": "실패",
                "sources": [],
                "success": False,
                "search_type": "web_search",
            },
        ) as mock_search, patch(
            "query_router.get_rag_engine", return_value=mock_engine
        ):
            result = agent.route_query("웹검색으로 아무거나 찾아줘")

        mock_search.assert_called_once()
        mock_engine.query.assert_called_once_with("웹검색으로 아무거나 찾아줘")
        assert result["search_type"] == "document_qa"
        assert result["success"] is True
