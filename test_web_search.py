#!/usr/bin/env python3
"""
web_search.py 단위 테스트 (pytest, 네트워크 호출 없음)

DDGS를 mock으로 대체하여 포맷팅/실패 처리 로직만 검증합니다.
실제 DuckDuckGo 검색 품질을 확인하려면 `python web_search.py`를 직접 실행하세요.
"""

from unittest.mock import MagicMock, patch

from web_search import format_search_results, search_and_format, search_web

SAMPLE_RESULTS = [
    {"href": "https://example.com/1", "title": "제목1", "body": "요약1"},
    {"href": "https://example.com/2", "title": "제목2", "body": "요약2"},
]


def _mock_ddgs(results=None, side_effect=None):
    """DDGS() as ddgs: ddgs.text(...) 형태를 흉내내는 context manager mock"""
    mock_ddgs_instance = MagicMock()
    if side_effect is not None:
        mock_ddgs_instance.__enter__.return_value.text.side_effect = side_effect
    else:
        mock_ddgs_instance.__enter__.return_value.text.return_value = results or []
    return mock_ddgs_instance


class TestSearchWeb:
    def test_returns_formatted_results(self):
        with patch("web_search.DDGS", return_value=_mock_ddgs(SAMPLE_RESULTS)):
            results = search_web("테스트 쿼리", max_results=2)

        assert len(results) == 2
        assert results[0] == {
            "url": "https://example.com/1",
            "title": "제목1",
            "summary": "요약1",
        }

    def test_missing_fields_get_defaults(self):
        with patch("web_search.DDGS", return_value=_mock_ddgs([{}])):
            results = search_web("테스트 쿼리")

        assert results[0]["url"] == ""
        assert results[0]["title"] == "제목 없음"
        assert results[0]["summary"] == "요약 없음"

    def test_exception_returns_empty_list(self):
        with patch("web_search.DDGS", side_effect=RuntimeError("network error")):
            results = search_web("테스트 쿼리")

        assert results == []


class TestFormatSearchResults:
    def test_empty_results(self):
        assert format_search_results([]) == "검색 결과가 없습니다."

    def test_formats_markdown_links(self):
        formatted = format_search_results(
            [{"url": "https://example.com", "title": "제목", "summary": "요약"}]
        )
        assert "[제목](https://example.com)" in formatted
        assert "요약" in formatted


class TestSearchAndFormat:
    def test_success_shape(self):
        with patch("web_search.DDGS", return_value=_mock_ddgs(SAMPLE_RESULTS)):
            result = search_and_format("테스트 쿼리", max_results=2)

        assert result["success"] is True
        assert result["search_type"] == "web_search"
        assert len(result["sources"]) == 2
        assert result["sources"][0]["source"] == "https://example.com/1"
        assert result["sources"][0]["index"] == 1

    def test_no_results_returns_failure(self):
        with patch("web_search.DDGS", return_value=_mock_ddgs([])):
            result = search_and_format("테스트 쿼리")

        assert result["success"] is False
        assert result["sources"] == []
