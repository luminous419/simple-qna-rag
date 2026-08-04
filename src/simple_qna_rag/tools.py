#!/usr/bin/env python3
"""
LangChain Tools 정의

Agent가 사용할 도구들을 정의합니다:
1. web_search_tool: 웹 검색 도구
2. rag_tool: 문서 기반 질의응답 도구

주의: 두 함수 모두 answer 문자열이 아닌 전체 결과 dict(answer/sources/success 등)를
반환합니다. agent.py가 LLM에게는 도구 "선택"만 맡기고 실제 실행은 이 함수들을 직접
호출해 sources를 포함한 구조화된 결과를 그대로 사용하기 때문입니다. (표준 AgentExecutor의
ReAct 루프처럼 도구 관찰 결과를 LLM이 다시 요약하게 하면, 이미 완결된 RAG 답변의 포맷이
깨지고 LLM 호출이 중복됩니다.)
"""

from typing import Dict, Any
from langchain.tools import Tool

from simple_qna_rag.web_search import search_and_format
from simple_qna_rag.rag_engine import get_rag_engine


def web_search_function(query: str) -> Dict[str, Any]:
    """
    웹 검색 도구 함수

    Args:
        query: 검색 쿼리

    Returns:
        dict: search_and_format()의 반환값 (answer/sources/success/search_type)
    """
    return search_and_format(query)


def rag_function(query: str) -> Dict[str, Any]:
    """
    문서 기반 질의응답 도구 함수

    Args:
        query: 사용자 질문

    Returns:
        dict: rag_engine.query()의 반환값 (answer/sources/success/intent)
    """
    rag_engine = get_rag_engine()
    return rag_engine.query(query)


# 웹 검색 도구 정의
web_search_tool = Tool(
    name="web_search",
    description=(
        "인터넷에서 정보를 검색합니다. "
        "사용자가 '웹검색', '인터넷검색', '웹에서 찾아줘', '인터넷에서 검색해줘'와 같은 "
        "키워드를 사용하거나, 최신 정보나 실시간 정보가 필요한 경우에 사용합니다. "
        "검색 결과는 URL, 제목, 요약을 포함하며 최대 3개까지 반환됩니다."
    ),
    func=web_search_function
)


# 문서 기반 질의응답 도구 정의
rag_tool = Tool(
    name="document_qa",
    description=(
        "등록된 문서 데이터베이스에서 관련 정보를 검색하여 답변을 생성합니다. "
        "일반적인 질의응답이나 문서 내용에 대한 질문에 사용합니다. "
        "웹검색 키워드가 없고, 문서 내용을 기반으로 답변해야 하는 경우에 사용합니다."
    ),
    func=rag_function
)


# 도구 리스트
tools = [web_search_tool, rag_tool]


def get_tools():
    """
    Agent가 사용할 도구 리스트 반환

    Returns:
        List[Tool]: 도구 리스트
    """
    return tools


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 60)
    print("LangChain Tools 테스트")
    print("=" * 60)

    print("\n[도구 목록]")
    for tool in tools:
        print(f"\n이름: {tool.name}")
        print(f"설명: {tool.description}")

    # 웹 검색 도구 테스트
    print("\n" + "=" * 60)
    print("웹 검색 도구 테스트")
    print("=" * 60)
    test_query = "Python"
    print(f"\n질문: {test_query}")
    try:
        result = web_search_tool.func(test_query)
        print(f"\n결과:\n{result['answer'][:300]}...")
        print(f"출처 수: {len(result['sources'])}")
    except Exception as e:
        print(f"오류: {e}")
