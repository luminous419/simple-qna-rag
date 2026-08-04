#!/usr/bin/env python3
"""
query_router.py 단위 테스트 (pytest, Ollama/벡터스토어 등 외부 의존성 없음)

get_rag_engine()과 search_and_format()을 mock으로 대체하여 키워드 라우팅 분기와
검색어 정제 로직만 검증합니다. agent.py는 이 라우터를 폴백으로만 사용하므로,
여기서는 키워드 라우터 자체의 정확성만 다룹니다.
"""

from unittest.mock import MagicMock, patch

from simple_qna_rag import query_router


def _mock_rag_result():
    return {"answer": "문서 기반 답변", "sources": [], "success": True, "intent": "explanation"}


def _mock_web_result():
    return {"answer": "웹 검색 결과", "sources": [], "success": True, "search_type": "web_search"}


class TestRouteQueryKeywordDetection:
    def test_web_search_keyword_routes_to_web_search(self):
        with patch(
            "simple_qna_rag.query_router.search_and_format", return_value=_mock_web_result()
        ) as mock_search, patch("simple_qna_rag.query_router.get_rag_engine") as mock_get_engine:
            result = query_router.route_query("웹검색으로 Python을 찾아줘")

        mock_search.assert_called_once()
        mock_get_engine.assert_not_called()
        assert result["search_type"] == "web_search"

    def test_no_keyword_routes_to_document_qa(self):
        mock_engine = MagicMock()
        mock_engine.query.return_value = _mock_rag_result()
        with patch(
            "simple_qna_rag.query_router.get_rag_engine", return_value=mock_engine
        ) as mock_get_engine, patch("simple_qna_rag.query_router.search_and_format") as mock_search:
            result = query_router.route_query("RAG에서 MMR이 뭐야?")

        mock_get_engine.assert_called_once()
        mock_search.assert_not_called()
        assert result["search_type"] == "document_qa"

    def test_search_query_strips_keyword_boilerplate(self):
        with patch(
            "simple_qna_rag.query_router.search_and_format", return_value=_mock_web_result()
        ) as mock_search, patch("simple_qna_rag.query_router.get_rag_engine"):
            query_router.route_query("웹검색으로 Python을 찾아줘")

        called_query = mock_search.call_args[0][0]
        assert "웹검색" not in called_query
        assert "Python" in called_query

    def test_bare_search_keyword_without_web_context_still_triggers_web_search(self):
        """
        알려진 한계: '검색해줘'만으로도 웹검색으로 분류됨 (예: '이 문서에서 검색해줘'도
        웹검색으로 오분류될 수 있음). 키워드 라우터는 이 한계를 그대로 갖고 있으며,
        agent.py가 이를 보완하기 위한 LLM 기반 1차 라우팅을 제공한다.
        """
        with patch(
            "simple_qna_rag.query_router.search_and_format", return_value=_mock_web_result()
        ) as mock_search, patch("simple_qna_rag.query_router.get_rag_engine") as mock_get_engine:
            result = query_router.route_query("이 문서에서 관련 내용 검색해줘")

        mock_search.assert_called_once()
        mock_get_engine.assert_not_called()
        assert result["search_type"] == "web_search"


class TestRouteQueryWebSearchFailureFallback:
    def test_web_search_failure_falls_back_to_document_qa(self):
        mock_engine = MagicMock()
        mock_engine.query.return_value = _mock_rag_result()
        with patch(
            "simple_qna_rag.query_router.search_and_format",
            return_value={
                "answer": "실패",
                "sources": [],
                "success": False,
                "search_type": "web_search",
            },
        ) as mock_search, patch(
            "simple_qna_rag.query_router.get_rag_engine", return_value=mock_engine
        ) as mock_get_engine:
            result = query_router.route_query("웹검색으로 Python을 찾아줘")

        mock_search.assert_called_once()
        # document_qa 재시도는 정제된 검색어가 아니라 원본 질문을 그대로 써야 함
        mock_engine.query.assert_called_once_with("웹검색으로 Python을 찾아줘")
        mock_get_engine.assert_called_once()
        assert result["search_type"] == "document_qa"


class TestRouteQueryFeatureFlag:
    def test_web_search_disabled_ignores_keywords(self):
        mock_engine = MagicMock()
        mock_engine.query.return_value = _mock_rag_result()
        with patch("simple_qna_rag.query_router.USE_WEB_SEARCH", False), patch(
            "simple_qna_rag.query_router.get_rag_engine", return_value=mock_engine
        ) as mock_get_engine, patch("simple_qna_rag.query_router.search_and_format") as mock_search:
            result = query_router.route_query("웹검색으로 Python을 찾아줘")

        mock_get_engine.assert_called_once()
        mock_search.assert_not_called()
        assert result["search_type"] == "document_qa"
