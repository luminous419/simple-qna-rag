#!/usr/bin/env python3
"""
Query Router (키워드 기반 라우터)

사용자 질문을 분석하여 적절한 도구(웹검색 또는 문서 QA)를 선택하고 실행합니다.
키워드 기반의 간단하고 빠른 라우팅 방식을 사용합니다.
"""

from typing import Dict, Any

from simple_qna_rag.config import USE_WEB_SEARCH
from simple_qna_rag.observability.logging import log_event
from simple_qna_rag.rag_engine import get_rag_engine
from simple_qna_rag.web_search import search_and_format


# 웹검색 키워드 정의
WEB_SEARCH_KEYWORDS = [
    '웹검색', '인터넷검색', '웹에서', '인터넷에서', '온라인에서',
    '검색해줘', '찾아줘', '알아봐줘'
]


def extract_web_search_query(question: str) -> str:
    """질문에서 웹검색 키워드/조사를 제거해 검색어를 추출하는 순수 함수
    (M3-REQ-004 §7.4 — 기존 `route_query()`의 로직을 동작 변경 없이 그대로
    이동한 것). agent.py의 `_decide_tool()`이 명시 WEB 신호에서 LLM이
    실패했을 때 검증된 폴백으로 이 함수를 호출한다.

    빈 문자열이 나올 수 있는 입력에 대해서는 원본 질문으로 되돌리는 기존
    동작을 유지한다.
    """
    search_query = question

    # 키워드와 조사를 함께 제거
    keywords_with_particles = [
        '웹검색으로', '웹검색해서', '웹검색으로서', '웹검색',
        '인터넷검색으로', '인터넷검색해서', '인터넷검색으로서', '인터넷검색',
        '웹에서', '인터넷에서', '온라인에서',
        '검색해줘', '찾아줘', '알아봐줘'
    ]

    for keyword in keywords_with_particles:
        search_query = search_query.replace(keyword, ' ').strip()

    # 남은 불필요한 조사 제거
    search_query = search_query.replace('으로', ' ').replace('를', ' ').replace('을', ' ').strip()

    # 여러 공백을 하나로
    search_query = ' '.join(search_query.split())

    return search_query if search_query else question


def route_query(question: str) -> Dict[str, Any]:
    """
    질문을 라우팅하여 적절한 처리 수행

    키워드 기반 라우팅을 사용합니다.

    Args:
        question: 사용자 질문

    Returns:
        dict: {
            "answer": str,
            "sources": List[Dict],
            "success": bool,
            "search_type": str  # "web_search" or "document_qa"
        }
    """
    # 웹검색 기능이 비활성화되어 있으면 RAG만 사용
    if not USE_WEB_SEARCH:
        rag_engine = get_rag_engine()
        result = rag_engine.query(question)
        result["search_type"] = "document_qa"
        return result

    # 키워드 기반 간단한 라우팅
    question_lower = question.lower()
    use_web_search = any(keyword in question_lower for keyword in WEB_SEARCH_KEYWORDS)

    if use_web_search:
        # M4.1 REPLACE(Design.md §6.1) — 추출된 검색어 원문은 로그에 남기지 않는다.
        log_event("routing", decision="web_search", confidence=1.0)
        search_query = extract_web_search_query(question)

        result = search_and_format(search_query)
        if not result.get("success"):
            log_event("routing", decision="web_search_failed_retry_document_qa", confidence=1.0)
            rag_engine = get_rag_engine()
            result = rag_engine.query(question)
            result["search_type"] = "document_qa"
        return result
    else:
        log_event("routing", decision="document_qa", confidence=1.0)
        rag_engine = get_rag_engine()
        result = rag_engine.query(question)
        result["search_type"] = "document_qa"
        return result
