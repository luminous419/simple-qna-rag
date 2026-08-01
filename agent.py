#!/usr/bin/env python3
"""
LLM 기반 Agent 라우터

ChatOllama의 tool calling 기능으로 웹검색/문서QA 중 하나를 선택하고,
tools.py에 정의된 실제 실행 함수를 직접 호출하여 구조화된 결과(answer/sources)를 반환합니다.

설계 노트 (왜 표준 AgentExecutor의 ReAct 루프를 쓰지 않는가):
web_search/document_qa 두 도구는 이미 완결된 최종 답변(Markdown 표 포맷, sources 포함)을
반환합니다. AgentExecutor는 도구 실행 후 그 관찰(observation)을 바탕으로 LLM을 한 번 더
호출해 최종 답변을 "합성"하는데, 이 경우 이미 완성된 RAG 답변이 재요약되며 포맷이 깨지고
LLM 호출이 불필요하게 중복됩니다. 따라서 여기서는 LLM에게 "어느 도구를 쓸지"만 맡기고
(단발성 라우팅), 실제 실행 결과는 재가공 없이 그대로 클라이언트에 전달합니다.

검증 결과 (2026-08-01, gpt-oss:20b):
- 별도 지침 없이 두 도구를 bind_tools()하면, 웹검색 키워드가 있는 질문은 올바르게
  web_search를 호출하지만 순수 문서QA성 질문에는 도구를 호출하지 않고 모델이 자체
  지식으로 답하려는 경향이 있었음 (RAG 근거 없는 답변 위험).
- "반드시 둘 중 하나를 호출해야 하며 스스로 답하지 말라"는 시스템 프롬프트를 추가하자
  샘플 질문 4개 모두 기대한 도구로 라우팅됨. 아래 SYSTEM_PROMPT는 이 요구사항을 반드시
  포함해야 함.
"""

from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, USE_WEB_SEARCH
from query_router import route_query as keyword_fallback_route
from rag_engine import get_rag_engine
from tools import rag_tool, web_search_tool

SYSTEM_PROMPT = (
    "당신은 web_search와 document_qa 두 도구 중 하나를 선택하는 라우터입니다. "
    "사용자의 모든 질문에 대해 반드시 둘 중 하나의 도구를 호출해야 하며, "
    "스스로의 지식으로 직접 답변해서는 안 됩니다. "
    "최신 정보, 실시간 정보, 인터넷/웹 검색을 명시적으로 요청하는 질문이면 web_search를, "
    "그 외 일반 개념 설명, 비교, 절차 안내 등 문서 기반 질의응답이면 document_qa를 "
    "호출하세요.\n"
    "web_search를 호출할 때는 검색 엔진에 넣기 좋은 핵심 키워드만 추출한 검색어를 인자로 "
    "전달하세요 (예: '오늘 서울 날씨 좀 웹에서 검색해줘' -> '서울 오늘 날씨'). "
    "document_qa를 호출할 때는 사용자의 질문을 요약하거나 바꾸지 말고 그대로 인자로 "
    "전달하세요."
)

_router_llm = None


def _get_router_llm():
    """라우팅 전용 ChatOllama 인스턴스 (지연 초기화, 싱글톤).

    rag_engine.py의 답변 생성용 OllamaLLM과는 별개의 인스턴스입니다.
    답변 생성은 기존 방식(텍스트 완성 + 프롬프트 템플릿)을 그대로 사용하고,
    이 인스턴스는 오직 "어느 도구를 쓸지" 판단에만 사용됩니다.
    """
    global _router_llm
    if _router_llm is None:
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )
        _router_llm = llm.bind_tools([web_search_tool, rag_tool])
    return _router_llm


def _decide_tool(question: str) -> tuple[Optional[str], Optional[str]]:
    """
    LLM에게 도구 선택을 맡기고 (도구 이름, 도구에 전달할 검색어)를 반환.

    검색어는 tool call의 인자값(예: {'__arg1': '서울 오늘 날씨'})에서 추출합니다.
    web_search는 이 정제된 검색어를 그대로 사용해야 DuckDuckGo 검색 품질이 나옵니다
    (원본 질문 전체를 그대로 검색하면 '검색해줘' 같은 불용어 때문에 결과 품질이 떨어짐).
    도구를 선택하지 못하면 (None, None)을 반환합니다.
    """
    llm = _get_router_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    response = llm.invoke(messages)
    if not response.tool_calls:
        return None, None

    call = response.tool_calls[0]
    args = call.get("args") or {}
    query = next(iter(args.values()), question) if args else question
    return call["name"], query


def route_query(question: str) -> Dict[str, Any]:
    """
    LLM 기반 Agent 라우팅 (웹검색 vs 문서 QA)

    - USE_WEB_SEARCH=False면 Agent를 거치지 않고 바로 문서 QA로 처리
    - Agent 호출 자체가 실패하거나 도구를 선택하지 못하면 키워드 기반 라우터
      (query_router.py)로 폴백
    - 웹검색 도구 실행이 실패(success=False)하면 문서 QA로 재시도

    Args:
        question: 사용자 질문

    Returns:
        dict: {
            "answer": str,
            "sources": List[Dict],
            "success": bool,
            "search_type": str,  # "web_search" or "document_qa"
            "intent": str,       # document_qa 경로일 때만 존재
        }
    """
    if not USE_WEB_SEARCH:
        result = get_rag_engine().query(question)
        result["search_type"] = "document_qa"
        return result

    try:
        tool_name, tool_query = _decide_tool(question)
    except Exception as e:
        print(f"⚠️  Agent 라우팅 실패, 키워드 라우터로 폴백: {e}")
        return keyword_fallback_route(question)

    if tool_name == "web_search":
        print(f"\n🤖 Agent 선택: web_search (검색어: '{tool_query}')")
        result = web_search_tool.func(tool_query)
        if not result.get("success"):
            print("⚠️  웹검색 실패, document_qa로 재시도")
            # document_qa는 항상 원본 질문을 그대로 사용 (LLM이 재작성한 검색어가
            # 아니라 rag_engine/intent_classifier가 학습된 자연어 질문 형태를 기대함)
            result = rag_tool.func(question)
            result["search_type"] = "document_qa"
        return result

    if tool_name == "document_qa":
        print("\n🤖 Agent 선택: document_qa")
        result = rag_tool.func(question)
        result["search_type"] = "document_qa"
        return result

    print("⚠️  Agent가 도구를 선택하지 못함, 키워드 라우터로 폴백")
    return keyword_fallback_route(question)


if __name__ == "__main__":
    # 테스트 코드 (키워드 없이도 웹검색/문서QA가 올바르게 갈리는지 확인)
    test_questions = [
        "오늘 서울 날씨 좀 웹에서 검색해줘",
        "RAG에서 MMR이 뭐야?",
        "최신 파이썬 버전이 몇이야? 인터넷에서 찾아줘",
        "FAISS와 Elasticsearch를 비교해줘",
    ]

    print("=" * 60)
    print("Agent Router 테스트")
    print("=" * 60)

    for question in test_questions:
        print(f"\n질문: {question}")
        try:
            result = route_query(question)
            print(f"검색 타입: {result.get('search_type', 'unknown')}")
            print(f"성공: {result['success']}")
            print(f"출처 수: {len(result.get('sources', []))}")
        except Exception as e:
            print(f"오류: {e}")
            import traceback

            traceback.print_exc()
        print("-" * 60)
