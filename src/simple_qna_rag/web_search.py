#!/usr/bin/env python3
"""
웹검색 모듈

DuckDuckGo를 사용하여 웹 검색을 수행하고 결과를 포맷팅합니다.
"""

import time
from typing import List, Dict, Optional
from ddgs import DDGS

from simple_qna_rag.config import (
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_TIMEOUT,
    WEB_SEARCH_REGION
)
from simple_qna_rag.observability.logging import log_event
from simple_qna_rag.observability.deadline import current_deadline


def search_web(query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
    """
    웹 검색 실행

    Args:
        query: 검색 쿼리
        max_results: 최대 검색 결과 수 (기본값: config.WEB_SEARCH_MAX_RESULTS)

    Returns:
        List[Dict]: 검색 결과 리스트
            [
                {
                    "url": str,
                    "title": str,
                    "summary": str
                },
                ...
            ]
    """
    if max_results is None:
        max_results = WEB_SEARCH_MAX_RESULTS

    t0 = time.perf_counter()
    try:
        # DuckDuckGo 검색 실행
        # 주의: text()의 timelimit은 검색 결과의 최신성 필터(d/w/m/y)이며 요청 타임아웃이
        # 아님. 실제 HTTP 요청 타임아웃은 DDGS() 생성자의 timeout 인자로 설정한다.
        deadline = current_deadline()
        remaining = deadline.remaining() if deadline is not None else float(WEB_SEARCH_TIMEOUT)
        if remaining <= 0:
            log_event("web_search", stage="web_search", duration_ms=0.0, error_code="timeout")
            return []
        with DDGS(timeout=min(float(WEB_SEARCH_TIMEOUT), remaining)) as ddgs:
            results = list(ddgs.text(
                query=query,
                region=WEB_SEARCH_REGION,
                max_results=max_results,
                timelimit=None  # 결과 최신성 필터 없음 (요청 타임아웃과는 무관)
            ))

        # 결과 포맷팅
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = {
                "url": result.get("href", ""),
                "title": result.get("title", "제목 없음"),
                "summary": result.get("body", "요약 없음")
            }
            formatted_results.append(formatted_result)

        # M4.1 REPLACE(Design.md §6.1) — query/title/summary 원문은 로그에
        # 남기지 않는다. duration만 구조화 이벤트로 발행한다(REQ-004.1).
        duration_ms = (time.perf_counter() - t0) * 1000
        log_event("web_search", stage="web_search", duration_ms=duration_ms)
        return formatted_results

    except Exception:
        # M4.1 REPLACE(Design.md §6.1) — 예외 원문/traceback은 로그에 남기지 않는다.
        duration_ms = (time.perf_counter() - t0) * 1000
        log_event("web_search", stage="web_search", duration_ms=duration_ms, error_code="upstream")
        return []


def format_search_results(results: List[Dict[str, str]]) -> str:
    """
    검색 결과를 사용자에게 보여줄 형식으로 포맷팅

    Args:
        results: search_web()의 반환값

    Returns:
        str: 포맷팅된 검색 결과 문자열
    """
    if not results:
        return "검색 결과가 없습니다."

    formatted_text = "웹 검색 결과입니다:\n\n"

    for i, result in enumerate(results, 1):
        formatted_text += f"**{i}. [{result['title']}]({result['url']})**\n"
        formatted_text += f"{result['summary']}\n\n"

    return formatted_text.strip()


def search_and_format(query: str, max_results: Optional[int] = None) -> Dict[str, any]:
    """
    웹 검색 실행 및 결과 포맷팅 (통합 함수)

    Args:
        query: 검색 쿼리
        max_results: 최대 검색 결과 수

    Returns:
        dict: {
            "answer": str,  # 포맷팅된 답변
            "sources": List[Dict],  # 출처 정보
            "success": bool,
            "search_type": str  # "web_search"
        }
    """
    results = search_web(query, max_results)

    if not results:
        return {
            "answer": "웹 검색 결과를 가져오는 데 실패했습니다. 다시 시도해주세요.",
            "sources": [],
            "success": False,
            "search_type": "web_search"
        }

    # 답변 포맷팅
    answer = format_search_results(results)

    # 출처 정보 포맷팅 (rag_engine.query()와 호환되는 형식)
    sources = []
    for i, result in enumerate(results, 1):
        sources.append({
            "index": i,
            "source": result["url"],
            "title": result["title"],
            "page": None,  # 웹 검색은 페이지 번호가 없음
            "content": result["summary"][:200]  # 처음 200자만
        })

    return {
        "answer": answer,
        "sources": sources,
        "success": True,
        "search_type": "web_search"
    }
