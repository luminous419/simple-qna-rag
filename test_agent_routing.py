#!/usr/bin/env python3
"""
agent.py 라우팅 회귀 테스트 (실제 gpt-oss:20b tool calling 호출 필요)

Ollama 서버와 실제 LLM 호출이 필요하므로 기본적으로 스킵됩니다. 로컬에서 Ollama가
실행 중일 때 아래처럼 명시적으로 실행하세요:

    RUN_LIVE_LLM_TESTS=1 pytest test_agent_routing.py -v

Agent 시스템 프롬프트나 모델(OLLAMA_MODEL)을 바꿀 때마다 이 테스트로 라우팅 정확도
회귀 여부를 확인합니다. rag_engine/web_search를 실제로 실행하지는 않고(비용이 크므로),
LLM의 도구 "선택" 결정만 검증합니다.

정답의 원천은 evaluation/datasets/golden.jsonl입니다(tags에 "routing_regression"이
붙은 16개 사례). 과거에는 이 파일에 하드코딩된 (질문, 기대 도구) 16쌍 리스트가
유일한 정답 소스였지만, Phase 5에서 골든셋으로 이관해 evaluation.routing의
evaluate_routing()과 정답 원천을 하나로 통합했습니다(M2-REQ-007).
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM_TESTS") != "1",
    reason="실제 Ollama LLM 호출이 필요합니다. RUN_LIVE_LLM_TESTS=1로 명시적으로 실행하세요.",
)

from agent import _decide_tool  # noqa: E402
from evaluation.dataset import load_jsonl  # noqa: E402
from evaluation.routing import evaluate_routing  # noqa: E402

GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "evaluation" / "datasets" / "golden.jsonl"

# 최소 통과 기준. 100%를 요구하지 않는 이유는 LLM 출력의 확률적 변동을 감안한 것이며,
# 이 기준 밑으로 떨어지면 프롬프트/모델 변경이 라우팅 품질을 유의미하게 해쳤다는 신호로 본다.
MIN_ACCURACY = 0.8


def _load_routing_regression_cases():
    cases = load_jsonl(GOLDEN_DATASET_PATH)
    selected = [case for case in cases if "routing_regression" in case.tags]
    assert len(selected) == 16
    return selected


def test_routing_regression_accuracy():
    cases = _load_routing_regression_cases()
    result = evaluate_routing(cases, _decide_tool)

    accuracy = result["accuracy"]
    print(
        f"\n라우팅 정확도: {result['correct_count']}/{result['total_cases']} "
        f"({accuracy:.0%})"
    )

    if result["failures"]:
        detail = "\n".join(
            f"  - {f['question']!r}: 기대={f['expected_route']}, "
            f"실제={f['actual_route']} ({f['failure_type']}"
            + (f", error={f['error']}" if f["error"] else "")
            + ")"
            for f in result["failures"]
        )
        print(f"\n라우팅 실패 {len(result['failures'])}건:\n{detail}")

    assert accuracy >= MIN_ACCURACY, (
        f"라우팅 정확도 {accuracy:.0%}가 최소 기준 {MIN_ACCURACY:.0%}에 미달 "
        f"({len(result['failures'])}/{result['total_cases']}건 실패)"
    )
