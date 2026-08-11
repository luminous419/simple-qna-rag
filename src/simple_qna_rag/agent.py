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

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from simple_qna_rag.config import (
    DATA_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    UPSTREAM_CONNECT_TIMEOUT_SECONDS,
    ROUTING_CORPUS_TOPIC_HINT,
    ROUTING_CORPUS_TOPIC_HINT_MAX_ITEMS,
    ROUTING_SIGNAL_OVERRIDE,
    USE_WEB_SEARCH,
)
from simple_qna_rag.observability.logging import log_event
from simple_qna_rag.observability.deadline import current_deadline, ollama_call_client
from simple_qna_rag.observability.metrics import record_fallback, record_stage_duration, record_stage_error
from simple_qna_rag.query_router import extract_web_search_query
from simple_qna_rag.query_router import route_query as keyword_fallback_route
from simple_qna_rag.rag_engine import get_rag_engine
from simple_qna_rag.routing_signals import (
    ExplicitSignal,
    build_corpus_topic_hint,
    classify_explicit_signal,
    is_loopback_endpoint,
)
from simple_qna_rag.tools import rag_tool, web_search_tool

_BASE_SYSTEM_PROMPT = (
    "당신은 web_search와 document_qa 두 도구 중 하나를 선택하는 라우터입니다. "
    "사용자의 모든 질문에 대해 반드시 둘 중 하나의 도구를 호출해야 하며, "
    "스스로의 지식으로 직접 답변해서는 안 됩니다. "
    "판단이 애매하면 document_qa를 선택하세요. "
    "질문에 연도(2024/2025), 정책·법령·기업·제품·지수 이름이 들어 있다는 이유만으로 "
    "web_search를 선택하지 마세요. 등록 문서에는 최근 연도의 보고서와 정책 자료가 "
    "포함돼 있습니다.\n"
    "web_search는 다음 두 경우에만 선택하세요: "
    "① 사용자가 웹/인터넷/온라인 검색을 명시적으로 요청했거나, "
    "② 오늘의 날씨·현재 시세·실시간 지수·환율·경기 결과·속보·특정 기업의 최근 발표된 "
    "실적/뉴스처럼 지금 이 순간 또는 가장 최근에 발표된 값이 필요한 질문일 때. "
    "그 외 일반 개념 설명, 비교, 절차 안내 등 문서 기반 질의응답은 "
    "document_qa를 호출하세요.\n"
    "web_search를 호출할 때는 검색 엔진에 넣기 좋은 핵심 키워드만 추출한 검색어를 인자로 "
    "전달하세요 (예: '오늘 서울 날씨 좀 웹에서 검색해줘' -> '서울 오늘 날씨'). "
    "document_qa를 호출할 때는 사용자의 질문을 요약하거나 바꾸지 말고 그대로 인자로 "
    "전달하세요."
)


def _build_system_prompt() -> str:
    """M3-NFR-003: corpus 파일명을 프롬프트에 넣는 hint는
    `ROUTING_CORPUS_TOPIC_HINT`와 loopback endpoint 조건을 모두 만족할 때만
    조립한다. 디렉터리 접근 실패는 hint 없이 기본 프롬프트로 강등한다."""
    if not ROUTING_CORPUS_TOPIC_HINT:
        return _BASE_SYSTEM_PROMPT
    if not is_loopback_endpoint(OLLAMA_BASE_URL):
        return _BASE_SYSTEM_PROMPT
    try:
        file_names = sorted(p.name for p in Path(DATA_DIR).iterdir() if p.is_file())
    except OSError:
        return _BASE_SYSTEM_PROMPT
    hint = build_corpus_topic_hint(file_names, ROUTING_CORPUS_TOPIC_HINT_MAX_ITEMS)
    if not hint:
        return _BASE_SYSTEM_PROMPT
    return _BASE_SYSTEM_PROMPT + "\n" + hint


SYSTEM_PROMPT = _build_system_prompt()


def get_router_prompt_sha256() -> str:
    """현재 `SYSTEM_PROMPT`의 SHA-256(§3.3 routing 리포트 `router_prompt_sha256`).
    corpus topic hint 활성 여부에 따라 프롬프트가 달라지므로, 이 값으로 어떤
    프롬프트가 실제로 쓰였는지 리포트에서 식별할 수 있다."""
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


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


def _llm_decide_tool(question: str) -> tuple[Optional[str], Optional[str]]:
    """
    LLM에게 도구 선택을 맡기고 (도구 이름, 도구에 전달할 검색어)를 반환.

    검색어는 tool call의 인자값(예: {'__arg1': '서울 오늘 날씨'})에서 추출합니다.
    web_search는 이 정제된 검색어를 그대로 사용해야 DuckDuckGo 검색 품질이 나옵니다
    (원본 질문 전체를 그대로 검색하면 '검색해줘' 같은 불용어 때문에 결과 품질이 떨어짐).
    도구를 선택하지 못하면 (None, None)을 반환합니다.

    M3 이전에는 이 함수 이름이 `_decide_tool()`이었다. M3-REQ-004가 명시 신호
    (WEB/DOCUMENT)의 우선순위를 "모델 가용성과 무관"하게 지키도록 요구하므로,
    신호 판정을 LLM보다 먼저 수행하는 `_decide_tool()`이 이 함수를 감싼다
    (Design.md §7.4).
    """
    if current_deadline() is not None:
        tools = [
            {"type": "function", "function": {"name": "web_search", "description": "Search web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "document_qa", "description": "Answer from documents", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
        ]
        with ollama_call_client(
            host=OLLAMA_BASE_URL, connect_timeout=UPSTREAM_CONNECT_TIMEOUT_SECONDS
        ) as client:
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
                tools=tools, stream=False, options={"temperature": 0.1},
            )
        message = response["message"] if isinstance(response, dict) else response.message
        calls = message.get("tool_calls", []) if isinstance(message, dict) else message.tool_calls
        if not calls:
            return None, None
        call = calls[0]
        function = call.get("function", call) if isinstance(call, dict) else call.function
        name = function.get("name") if isinstance(function, dict) else function.name
        args = function.get("arguments", {}) if isinstance(function, dict) else function.arguments
        query = next(iter(args.values()), question) if args else question
        return name, query
    llm = _get_router_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    response = llm.invoke(messages)
    if not response.tool_calls:
        return None, None

    call = response.tool_calls[0]
    args = call.get("args") or {}
    query = next(iter(args.values()), question) if args else question
    return call["name"], query


def _log_signal_error(exc: Exception) -> None:
    # M4.1 REPLACE(Design.md §6.1) — 예외 원문은 로그에 남기지 않는다.
    log_event("routing", decision="signal_classification_error_fallback_to_llm", confidence=0.0)


def _log_llm_error(exc: Exception) -> None:
    log_event("routing", decision="llm_error_fallback_to_keyword_query", confidence=0.0)


def _decide_tool(question: str) -> tuple[Optional[str], Optional[str]]:
    """M3-REQ-004 §7.4: 명시 신호를 LLM 호출 **이전에** 결정론적으로 판정한다.

    - `ROUTING_SIGNAL_OVERRIDE=False`(기본값)이면 `_llm_decide_tool(question)`만
      호출한다 — M2와 완전히 동일한 rollback 경로(Design.md §13.3).
    - DOCUMENT 신호는 LLM을 호출하지 않고 `("document_qa", question)`을
      반환한다(문서 경로의 tool query 계약은 항상 원본 질문이므로 LLM이
      기여할 여지가 없다).
    - WEB 신호는 검색어 품질을 위해 LLM을 먼저 시도하되, 예외·no-tool·다른
      도구 선택·빈 query 등 어떤 이유로든 LLM이 web_search를 정하지 못하면
      `extract_web_search_query()`로 결정론적으로 보완한다. LLM 예외를
      keyword fallback으로 흘리지 않는다.
    - NONE 신호는 기존 계약을 완전히 그대로 유지한다(예외 전파,
      `(None, None)` 반환 포함) — `route_query()`의 기존 폴백 4경로가 M2와
      동일하게 동작한다.
    """
    signal = ExplicitSignal.NONE
    if ROUTING_SIGNAL_OVERRIDE:
        try:
            signal = classify_explicit_signal(question)
        except Exception as exc:  # noqa: BLE001 - 신호 분류 결함이 라우팅 전체를 막지 않게 한다
            _log_signal_error(exc)
            signal = ExplicitSignal.NONE

    if signal is ExplicitSignal.DOCUMENT:
        return "document_qa", question

    if signal is ExplicitSignal.WEB:
        try:
            llm_name, llm_query = _llm_decide_tool(question)
        except Exception as exc:  # noqa: BLE001 - 예외를 keyword fallback으로 흘리지 않는다
            _log_llm_error(exc)
            return "web_search", extract_web_search_query(question)
        if llm_name == "web_search" and llm_query:
            return "web_search", llm_query
        return "web_search", extract_web_search_query(question)

    return _llm_decide_tool(question)


def _call_rag_tool(question: str, metrics_registry: Any) -> Dict[str, Any]:
    """`rag_tool.func`(== `tools.rag_function`)를 호출한다. registry가 있을
    때만 `metrics_registry` kwarg를 전달해, registry가 없는 기존 호출자
    (CLI, 기존 테스트의 `assert_called_once_with(question)`)의 호출 시그니처를
    그대로 보존한다."""
    if metrics_registry is None:
        return rag_tool.func(question)
    return rag_tool.func(question, metrics_registry=metrics_registry)


def route_query(question: str, *, metrics_registry: Any = None) -> Dict[str, Any]:
    """
    LLM 기반 Agent 라우팅 (웹검색 vs 문서 QA)

    - USE_WEB_SEARCH=False면 Agent를 거치지 않고 바로 문서 QA로 처리
    - Agent 호출 자체가 실패하거나 도구를 선택하지 못하면 키워드 기반 라우터
      (query_router.py)로 폴백
    - 웹검색 도구 실행이 실패(success=False)하면 문서 QA로 재시도

    Args:
        question: 사용자 질문
        metrics_registry: `app.state.metrics_registry`(선택, REQ-004.1).
            web/server.py의 `/rag` 핸들러만 실제로 전달한다 — CLI/키워드
            폴백 경로는 항상 None이며, 이 경우 아래 계측 호출은 전부 no-op.

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
        result = get_rag_engine().query(question, metrics_registry=metrics_registry)
        result["search_type"] = "document_qa"
        return result

    t0 = time.perf_counter()
    try:
        tool_name, tool_query = _decide_tool(question)
    except Exception:
        record_stage_duration(metrics_registry, "routing", time.perf_counter() - t0)
        record_stage_error(metrics_registry, "routing", "internal")
        log_event("routing", decision="agent_routing_error_fallback_to_keyword", confidence=0.0)
        return keyword_fallback_route(question)
    record_stage_duration(metrics_registry, "routing", time.perf_counter() - t0)

    if tool_name == "web_search":
        # M4.1 REPLACE(Design.md §6.1) — 검색어 원문은 로그에 남기지 않는다.
        log_event("routing", decision="web_search", confidence=1.0)
        t0 = time.perf_counter()
        result = web_search_tool.func(tool_query)
        record_stage_duration(metrics_registry, "web_search", time.perf_counter() - t0)
        if not result.get("success"):
            log_event("routing", decision="web_search_failed_retry_document_qa", confidence=1.0)
            record_fallback(metrics_registry, "web_search", "other")
            # document_qa는 항상 원본 질문을 그대로 사용 (LLM이 재작성한 검색어가
            # 아니라 rag_engine/intent_classifier가 학습된 자연어 질문 형태를 기대함)
            result = _call_rag_tool(question, metrics_registry)
            result["search_type"] = "document_qa"
        return result

    if tool_name == "document_qa":
        log_event("routing", decision="document_qa", confidence=1.0)
        result = _call_rag_tool(question, metrics_registry)
        result["search_type"] = "document_qa"
        return result

    log_event("routing", decision="no_tool_selected_fallback_to_keyword", confidence=0.0)
    return keyword_fallback_route(question)
