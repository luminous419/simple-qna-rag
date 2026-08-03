"""
Routing evaluator (M2-REQ-007).

offline/live 두 모드가 동일한 evaluate_routing() 집계 코어를 공유한다.
- offline: 외부 LLM 없이 고정 응답(mock)을 사용해 파싱/집계/오류 분류/리포팅
  경로만 검증한다. 이 모드의 accuracy는 실제 라우팅 품질을 의미하지 않는다.
- live: 실제 agent._decide_tool()을 호출해 골든셋(또는 --tag/--limit로 선택한
  부분집합)의 실제 라우팅 정확도를 측정한다. RUN_LIVE_LLM_TESTS=1 없이는
  실행되지 않는다.

이 모듈은 corpus/vectorstore를 전혀 사용하지 않는다 — Retrieval/Answer 리포트와
달리 reproducibility 필드는 항상 null이고 reproducibility_note에 이유를 남긴다
(evaluation.reporting.build_not_applicable_reproducibility_metadata()).

이 모듈은 import 시점에 agent.py/rag_engine.py/Ollama를 로드하지 않는다.
`agent._decide_tool`은 --mode live이고 RUN_LIVE_LLM_TESTS=1일 때만 main() 내부에서
지연 import된다.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from evaluation.dataset import DatasetError, load_jsonl
from evaluation.metrics import mean_median, percentile, precision_recall_f1
from evaluation.reporting import (
    build_metadata,
    build_not_applicable_reproducibility_metadata,
    write_report,
)
from evaluation.schema import GoldenCase, Route

SCHEMA_VERSION = "1.0.0"

_VALID_ROUTES = {route.value for route in Route}
_LABELS = [Route.DOCUMENT_QA.value, Route.WEB_SEARCH.value]


def _offline_mock_decide_tool(question: str) -> tuple[Optional[str], Optional[str]]:
    """오프라인 모드 전용 고정 응답(mock).

    실제 LLM을 호출하지 않고 질문 내용과 무관하게 항상 같은 값을 반환한다.
    목적은 CLI의 파싱/집계/오류 분류/리포팅 경로가 네트워크나 모델 없이도
    동작하는지 검증하는 것뿐이다(M2-REQ-007 "offline: 외부 LLM 없이 고정
    응답/mock을 사용..."). 이 함수가 만들어내는 accuracy는 실제 라우팅 품질을
    나타내지 않는다 — 실제 품질 측정은 --mode live에서 agent._decide_tool()로만
    수행한다.
    """
    return Route.DOCUMENT_QA.value, question


def evaluate_routing(
    cases: list[GoldenCase],
    decide_tool: Callable[[str], tuple[Optional[str], Optional[str]]],
    *,
    measure_latency: bool = True,
) -> dict:
    """offline/live 공통 Routing 평가 집계 코어.

    cases는 이미 호출부(main() 또는 테스트)가 tag/limit으로 필터링을 마친
    리스트여야 한다 — 이 함수 자체는 필터링을 하지 않는다.

    각 사례에 대해 decide_tool(question)을 호출하고 다음 세 실패 유형을
    confusion matrix/precision/recall/F1 계산에서 제외한 뒤 별도로 기록한다
    (evaluation.metrics.precision_recall_f1()은 labels 밖의 값을 받으면
    KeyError를 던지므로, "도구 미선택"/"예외"/"알 수 없는 route"는 애초에
    y_true/y_pred에 넣지 않는다):

    - "exception": decide_tool()이 예외를 던짐
    - "no_tool": decide_tool()이 (None, None)을 반환함
    - "unknown_route": decide_tool()이 반환한 도구 이름이 evaluation.schema.Route
      enum 값("document_qa", "web_search")에 없음

    이 세 유형에 속하지 않는 사례만 y_true/y_pred에 포함되며, 그중 예측이
    expected_route와 다르면 "mismatch"로 기록한다. 개별 사례의 실패는 다음
    사례 평가를 막지 않는다(전체 중단 금지).

    cases가 비어 있으면(빈 입력 또는 --tag/--limit 필터 결과 0건) 이를 성공
    accuracy로 조용히 처리하지 않기 위해 ValueError를 던진다(§6.5).

    반환 dict:
    {
        "total_cases": int,
        "success_count": int,       # decide_tool()이 예외 없이 실행된 사례 수
        "failure_count": int,       # decide_tool()이 예외를 던진 사례 수(exception_count와 동일)
        "excluded_count": int,      # no_tool_count + unknown_route_count
                                     # (success_count의 부분집합, confusion matrix 계산에서 제외됨)
        "correct_count": int,       # 예측이 expected_route와 일치한 사례 수
        "accuracy": float,          # correct_count / total_cases
        "no_tool_count": int,
        "unknown_route_count": int,
        "exception_count": int,
        "precision_recall_f1": {    # evaluation.metrics.precision_recall_f1()의 반환값 그대로
            "document_qa": {"precision":.., "recall":.., "f1":..},
            "web_search": {"precision":.., "recall":.., "f1":..},
            "confusion_matrix": {...},
        },
        "latency_ms": {
            "measured": bool,       # measure_latency 값. False면 아래 세 값은 항상 None
                                     # (0ms와 명확히 구분하기 위함)
            "mean": float | None,
            "median": float | None,
            "p95": float | None,
        },
        "failures": [
            {
                "id": str,
                "question": str,
                "expected_route": str,
                "actual_route": str | None,
                "failure_type": "exception" | "no_tool" | "unknown_route" | "mismatch",
                "error": str | None,
            },
            ...
        ],
    }
    """
    if not cases:
        raise ValueError(
            "평가할 사례가 없습니다: cases 리스트가 비어 있습니다. "
            "--tag/--limit 필터가 golden.jsonl의 어떤 사례와도 일치하지 않았을 "
            "수 있습니다. 필터 조건을 완화하거나 사례를 추가하세요."
        )

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies: list[float] = []
    failures: list[dict] = []

    no_tool_count = 0
    unknown_route_count = 0
    exception_count = 0
    correct_count = 0

    for case in cases:
        expected = case.expected_route.value
        start = time.perf_counter() if measure_latency else None

        try:
            tool_name, _tool_query = decide_tool(case.question)
        except Exception as exc:  # noqa: BLE001 - 사례별 예외를 기록하고 다음 사례를 계속 평가해야 함
            if measure_latency:
                latencies.append((time.perf_counter() - start) * 1000)
            exception_count += 1
            failures.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "expected_route": expected,
                    "actual_route": None,
                    "failure_type": "exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if measure_latency:
            latencies.append((time.perf_counter() - start) * 1000)

        if tool_name is None:
            no_tool_count += 1
            failures.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "expected_route": expected,
                    "actual_route": None,
                    "failure_type": "no_tool",
                    "error": None,
                }
            )
            continue

        if tool_name not in _VALID_ROUTES:
            unknown_route_count += 1
            failures.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "expected_route": expected,
                    "actual_route": tool_name,
                    "failure_type": "unknown_route",
                    "error": None,
                }
            )
            continue

        y_true.append(expected)
        y_pred.append(tool_name)
        if tool_name == expected:
            correct_count += 1
        else:
            failures.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "expected_route": expected,
                    "actual_route": tool_name,
                    "failure_type": "mismatch",
                    "error": None,
                }
            )

    total_cases = len(cases)
    success_count = total_cases - exception_count
    excluded_count = no_tool_count + unknown_route_count

    metrics = precision_recall_f1(y_true, y_pred, labels=list(_LABELS))

    if measure_latency:
        mean_latency, median_latency = mean_median(latencies)
        p95_latency = percentile(latencies, 95)
    else:
        mean_latency, median_latency, p95_latency = None, None, None

    return {
        "total_cases": total_cases,
        "success_count": success_count,
        "failure_count": exception_count,
        "excluded_count": excluded_count,
        "correct_count": correct_count,
        "accuracy": correct_count / total_cases,
        "no_tool_count": no_tool_count,
        "unknown_route_count": unknown_route_count,
        "exception_count": exception_count,
        "precision_recall_f1": metrics,
        "latency_ms": {
            "measured": measure_latency,
            "mean": mean_latency,
            "median": median_latency,
            "p95": p95_latency,
        },
        "failures": failures,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation.routing")
    parser.add_argument("--dataset", type=Path, required=True, help="골든셋 JSONL 경로")
    parser.add_argument("--output", type=Path, required=True, help="리포트 출력 디렉터리")
    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        required=True,
        help="offline: 고정 응답(mock)으로 파이프라인만 검증. "
        "live: 실제 agent._decide_tool() 호출(RUN_LIVE_LLM_TESTS=1 필요)",
    )
    parser.add_argument("--limit", type=int, default=None, help="평가할 사례 수 상한(원본 순서 유지)")
    parser.add_argument("--tag", type=str, default=None, help="이 태그를 가진 사례만 평가")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # live opt-in 확인은 dataset 로드나 agent import보다 먼저 수행한다 —
    # opt-in 없이는 어떤 외부 모델 import/호출도 일어나서는 안 된다(§6.3).
    if args.mode == "live" and os.environ.get("RUN_LIVE_LLM_TESTS") != "1":
        print(
            "live 모드는 실제 Ollama LLM(agent._decide_tool())을 호출합니다. "
            "명시적 opt-in 없이는 실행하지 않습니다.",
            file=sys.stderr,
        )
        print(
            "다음 조치: 로컬에 Ollama가 준비돼 있다면 RUN_LIVE_LLM_TESTS=1 환경변수를 "
            "설정한 뒤 다시 실행하세요. 예:\n"
            f"  RUN_LIVE_LLM_TESTS=1 python -m evaluation.routing "
            f"--dataset {args.dataset} --output {args.output} --mode live",
            file=sys.stderr,
        )
        return 2

    try:
        cases = load_jsonl(args.dataset)
    except DatasetError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        print(
            "다음 조치: --dataset 경로가 올바른 golden.jsonl을 가리키는지, "
            "파일이 유효한지 확인한 뒤 다시 실행하세요.",
            file=sys.stderr,
        )
        return 1

    if args.tag:
        cases = [c for c in cases if args.tag in c.tags]
    if args.limit is not None:
        cases = cases[: args.limit]

    if not cases:
        print(
            f"오류: --tag={args.tag!r} --limit={args.limit!r} 필터를 적용한 뒤 "
            f"남은 사례가 없습니다.",
            file=sys.stderr,
        )
        print(
            "다음 조치: --tag/--limit 조건을 완화하거나 golden.jsonl에 해당 조건을 "
            "만족하는 사례를 추가한 뒤 다시 실행하세요.",
            file=sys.stderr,
        )
        return 1

    if args.mode == "live":
        from agent import _decide_tool as decide_tool  # 지연 import (live + opt-in 전용)
    else:
        decide_tool = _offline_mock_decide_tool

    try:
        result = evaluate_routing(cases, decide_tool)
    except ValueError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    effective_argv = argv if argv is not None else sys.argv[1:]
    command = ["python", "-m", "evaluation.routing", *effective_argv]
    reproducibility = build_not_applicable_reproducibility_metadata(
        "routing은 corpus/vectorstore를 사용하지 않음"
    )
    payload = build_metadata(
        args.dataset,
        command,
        {
            "mode": args.mode,
            "tag_filter": args.tag,
            "limit": args.limit,
            **result,
            **reproducibility,
        },
    )

    json_path, md_path = write_report(payload, args.output, "routing")
    print(
        f"라우팅 정확도: {result['accuracy']:.1%} "
        f"({result['correct_count']}/{result['total_cases']})",
        file=sys.stderr,
    )
    if result["failures"]:
        print(f"실패 {len(result['failures'])}건 (상세는 리포트 참고)", file=sys.stderr)
    print(f"리포트 생성: {json_path}")
    print(f"리포트 생성: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
