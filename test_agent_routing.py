#!/usr/bin/env python3
"""
agent.py 라우팅 회귀 테스트 (실제 gpt-oss:20b tool calling 호출 필요)

Ollama 서버와 실제 LLM 호출이 필요하므로 기본적으로 스킵됩니다. 로컬에서 Ollama가
실행 중일 때 아래처럼 명시적으로 실행하세요:

    RUN_LIVE_LLM_TESTS=1 pytest test_agent_routing.py -v

Agent 시스템 프롬프트나 모델(OLLAMA_MODEL)을 바꿀 때마다 이 테스트로 라우팅 정확도
회귀 여부를 확인합니다. rag_engine/web_search를 실제로 실행하지는 않고(비용이 크므로),
LLM의 도구 "선택" 결정만 검증합니다.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM_TESTS") != "1",
    reason="실제 Ollama LLM 호출이 필요합니다. RUN_LIVE_LLM_TESTS=1로 명시적으로 실행하세요.",
)

from agent import _decide_tool  # noqa: E402

# (질문, 기대 도구) 라우팅 회귀 세트
ROUTING_CASES = [
    # 명시적 웹검색 키워드
    ("오늘 서울 날씨 좀 웹에서 검색해줘", "web_search"),
    ("최신 파이썬 버전이 몇이야? 인터넷에서 찾아줘", "web_search"),
    ("비트코인 시세를 온라인에서 검색해줘", "web_search"),
    ("삼성전자 주가를 웹검색으로 알아봐줘", "web_search"),
    # 키워드 없이 최신/실시간성이 드러나는 질문
    ("지금 서울 기온이 몇 도야?", "web_search"),
    ("오늘 환율이 어떻게 돼?", "web_search"),
    # 순수 문서 QA (설명)
    ("RAG에서 MMR이 뭐야?", "document_qa"),
    ("임베딩이 뭔지 설명해줘", "document_qa"),
    ("리랭커의 역할이 뭐야?", "document_qa"),
    # 비교
    ("FAISS와 Elasticsearch를 비교해줘", "document_qa"),
    ("BM25와 Dense Retrieval의 차이점은?", "document_qa"),
    # 절차
    ("Python에서 FAISS 설치하는 방법을 알려줘", "document_qa"),
    ("벡터스토어를 만드는 절차를 단계별로 설명해줘", "document_qa"),
    # 예/아니오
    ("LangChain은 무료로 사용할 수 있나요?", "document_qa"),
    ("Ollama는 로컬에서 실행되나요?", "document_qa"),
    # 키워드 라우터가 오분류하는 엣지 케이스 (Agent가 개선해야 하는 지점)
    ("이 문서에서 관련 내용을 찾아줘", "document_qa"),
]

# 최소 통과 기준. 100%를 요구하지 않는 이유는 LLM 출력의 확률적 변동을 감안한 것이며,
# 이 기준 밑으로 떨어지면 프롬프트/모델 변경이 라우팅 품질을 유의미하게 해쳤다는 신호로 본다.
MIN_ACCURACY = 0.8


def test_routing_regression_accuracy():
    correct = 0
    failures = []

    for question, expected_tool in ROUTING_CASES:
        tool_name, _ = _decide_tool(question)
        if tool_name == expected_tool:
            correct += 1
        else:
            failures.append((question, expected_tool, tool_name))

    accuracy = correct / len(ROUTING_CASES)
    print(f"\n라우팅 정확도: {correct}/{len(ROUTING_CASES)} ({accuracy:.0%})")

    if failures:
        detail = "\n".join(
            f"  - {q!r}: 기대={expected}, 실제={actual}" for q, expected, actual in failures
        )
        print(f"\n라우팅 실패 {len(failures)}건:\n{detail}")

    assert accuracy >= MIN_ACCURACY, (
        f"라우팅 정확도 {accuracy:.0%}가 최소 기준 {MIN_ACCURACY:.0%}에 미달 "
        f"({len(failures)}/{len(ROUTING_CASES)}건 실패)"
    )
