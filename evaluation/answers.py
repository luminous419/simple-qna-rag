"""
Answer evaluator와 사람 검토 worksheet (M2-REQ-008~010, M2-REQ-015, Phase 6).

실제 `RAGEngine.query()` 경로를 그대로 사용해 assertion coverage, abstention
정확도, source 일치율, intent 정확도, latency를 자동 평가하고, 사람이 채점할
Markdown worksheet를 만든다.

이 모듈은 `rag_engine.py`, `prompt_templates.py`, 골든셋, 공통 평가 모듈
(`evaluation/schema.py`, `evaluation/dataset.py`, `evaluation/metrics.py`,
`evaluation/reporting.py`)을 수정하지 않고 순수 소비자로만 import한다.

`get_rag_engine()`은 이 모듈의 어떤 함수도 import 시점에 호출하지 않는다 —
`_get_engine()`을 통해 `evaluate_answers()` 실행 시점에만 지연 호출된다
(M2-NFR-003, 상위 규칙 문서 §3.3).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

from simple_qna_rag import config
from evaluation.dataset import DatasetError, load_jsonl
from evaluation.metrics import assertion_coverage, dedupe_preserve_order, mean_median, percentile
from evaluation.reporting import (
    build_metadata,
    build_reproducibility_metadata,
    escape_markdown_table_cell,
    write_report,
)
from evaluation.schema import AnswerAssertion, GoldenCase, is_answer_eval_eligible, normalize_source_id

SCHEMA_VERSION = "1.0.0"

# 실제 프롬프트 템플릿(prompt_templates.py)에 존재하는 두 공식 거절 문구를 모두
# 인식한다. explanation/comparison/procedure/other/uncertain 템플릿은
# "제공된 문서에서 관련 정보를 찾을 수 없습니다"를, yesno 템플릿은
# "제공된 문서만으로는 확실한 답변이 어렵습니다"를 사용한다
# (prompt_templates.py 참고). 두 템플릿 세트가 이후 바뀌면 이 목록도 함께
# 갱신해야 한다(coupling risk).
ABSTENTION_PHRASES = (
    "제공된 문서에서 관련 정보를 찾을 수 없습니다",
    "제공된 문서만으로는 확실한 답변이 어렵습니다",
)

ASSERTION_COVERAGE_LIMITATION_NOTE = (
    "assertion coverage(핵심 문구 포함 여부)는 답변에 특정 사실이 언급됐는지만 "
    "확인할 뿐, 답변 전체의 진실성(faithfulness)이나 문맥 왜곡 여부를 보증하지 "
    "않습니다. 사람 검토 worksheet의 Faithfulness 점수로 반드시 보완해야 합니다."
)


def _get_engine():
    """RAGEngine 싱글톤을 지연 로드한다. 테스트는 이 함수 자체를 monkeypatch해
    fake engine을 주입한다 — 모듈 import나 evaluate_answers() 호출 이전 시점에는
    rag_engine을 import하지 않는다."""
    from simple_qna_rag.rag_engine import get_rag_engine

    return get_rag_engine()


def _detect_abstention(answer: str) -> bool:
    """두 공식 거절 문구를 NFC 정규화 후 원문 그대로 포함하는지 확인한다.

    assertion_coverage()와 달리 casefold()는 적용하지 않는다 — 한국어는
    대소문자 개념이 없고, 영문 대소문자 변경이 다른 의미의 문구로 오인식될
    위험을 피하기 위해 원문 표기를 그대로 비교한다."""
    normalized = unicodedata.normalize("NFC", answer)
    return any(unicodedata.normalize("NFC", phrase) in normalized for phrase in ABSTENTION_PHRASES)


def _extract_returned_source_ids(sources: list[dict]) -> tuple[list[str], int]:
    """RAGEngine.query()/rag_tool.func()가 실제로 반환하는 sources는 문자열
    리스트가 아니라 {"index":.., "source":.., "page":.., "content":..} 형태의
    dict 리스트다(rag_engine.py:456-465). "source" 키가 없거나 값이 문자열이
    아니거나 빈 문자열인 항목은 건너뛰고 skipped 카운트에 반영한다(레거시로
    문자열을 바로 섞어 보내는 것은 지원하지 않음 — 항상 dict 리스트를 기대한다).

    Returns: (source_ids, skipped_count)
    """
    ids: list[str] = []
    skipped = 0
    for entry in sources:
        source = entry.get("source") if isinstance(entry, dict) else None
        if isinstance(source, str) and source:
            ids.append(source)
        else:
            skipped += 1
    return ids, skipped


def _source_match(returned_sources: list[str], relevant_sources: list[str]) -> dict | None:
    """"source 일치율"의 정의를 명시적으로 고정한다. 인자는 이미
    _extract_returned_source_ids()를 거친 순수 문자열 리스트여야 한다.
    relevant_sources가 비어 있으면(예: 순수 abstention 사례) 이 사례는 source
    평가에서 제외하고 None을 반환한다 — 호출부가 이를 source 평가 제외 수에
    반영한다. 양쪽 다 normalize_source_id() + dedupe_preserve_order()를 거친
    뒤 비교한다(Retrieval 지표와 동일한 정규화/중복 제거 함수 재사용).
    """
    if not relevant_sources:
        return None
    returned = set(dedupe_preserve_order([normalize_source_id(s) for s in returned_sources]))
    relevant = set(dedupe_preserve_order([normalize_source_id(s) for s in relevant_sources]))
    return {
        "source_any_hit": bool(returned & relevant),
        "source_recall": len(returned & relevant) / len(relevant),
    }


def _abstention_confusion(flags: list[tuple[bool, bool]]) -> dict:
    """"abstention 정확도"를 Answer 평가 대상 중 실제로 채점 가능했던(질의가
    성공한) 사례 전체에 대한 이진 분류로 정의한다. flags는 사례마다
    (expected_abstention, predicted_abstention) 튜플이다.

        TP: expected=True,  predicted=True   (정상적으로 거절)
        TN: expected=False, predicted=False  (정상적으로 답변)
        FP: expected=False, predicted=True   (답변 가능한데 잘못 거절)
        FN: expected=True,  predicted=False  (거절해야 하는데 답변을 시도)

    accuracy = (TP+TN)/N. N=0이면 accuracy=None이고 별도 필드
    abstention_accuracy_excluded_reason에 사유를 기록한다."""
    tp = sum(1 for e, p in flags if e and p)
    tn = sum(1 for e, p in flags if not e and not p)
    fp = sum(1 for e, p in flags if not e and p)
    fn = sum(1 for e, p in flags if e and not p)
    n = len(flags)
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": (tp + tn) / n if n > 0 else None,
        "abstention_accuracy_excluded_reason": None if n > 0 else "Answer 평가 대상 사례 없음(질의 성공 사례 0건)",
    }


def _fence_for(text: str) -> str:
    """답변 자체에 3중 backtick(또는 그보다 긴 연속 backtick) 코드블록이
    포함되면 고정된 3틱 fence로는 worksheet 구조가 깨질 수 있다. 답변에
    등장하는 가장 긴 연속 backtick 길이보다 1 긴 fence를 동적으로 생성한다
    (표준 CommonMark 규칙): fence_length = max(longest_backtick_run + 1, 3)."""
    longest_run = 0
    current = 0
    for ch in text:
        if ch == "`":
            current += 1
            longest_run = max(longest_run, current)
        else:
            current = 0
    return "`" * max(longest_run + 1, 3)


def _summarize_assertions(assertions: list[AnswerAssertion]) -> list[str]:
    """각 AnswerAssertion의 any_of를 " / "로 이어 사람이 읽을 수 있는 한 줄
    요약으로 만든다. worksheet의 "기대 핵심 사실" 섹션에만 쓰인다."""
    return [" / ".join(a.any_of) for a in assertions]


def _safe_inline(text: str | None) -> str:
    """Markdown 구조를 깨뜨릴 수 있는 줄바꿈/연속 공백을 단일 공백으로
    접어 한 줄 필드(질문, source, assertion 요약 등)에 안전하게 넣는다.
    파이프(|)는 표 셀이 아닌 일반 문단에서는 구조를 깨지 않으므로 그대로
    둔다."""
    if text is None:
        return ""
    return " ".join(text.split())


def evaluate_answers(
    dataset_path: Path,
    output_dir: Path,
    limit: int | None = None,
    tag: str | None = None,
) -> dict:
    """골든셋에서 `is_answer_eval_eligible()`을 만족하는 사례만 실제
    `RAGEngine.query()`로 평가하고, assertion/abstention/source/intent/latency
    자동 점수와 사람 검토 worksheet를 만든다.

    처리 순서:
      1. dataset 로드(오류는 DatasetError로 즉시 중단, 호출부에 전파).
      2. tag/limit을 원본 순서 그대로, 결정론적으로 적용.
      3. `is_answer_eval_eligible()`(공통 schema 함수, 재정의하지 않음)로 대상 판정.
      4. corpus/vectorstore 재현성 메타데이터를 먼저 계산(artifact 누락을
         모델 로드 이전에 빠르게 감지).
      5. 대상이 있을 때만 `_get_engine()`을 지연 호출.
      6. 사례별로 질의하고 실패(예외 또는 success=False)는 기록 후 계속,
         정상 답변만 assertion/abstention/source/intent 채점에 반영.
      7. 집계 후 JSON/Markdown 리포트와 review worksheet를 생성.

    CLI(`main()`)는 argparse의 `_positive_int` type으로 0/음수 `--limit`을 이미
    거부하지만, 이 함수를 Python에서 직접 호출하면 이 계층을 거치지 않는다.
    `limit < 1`을 여기서도 검증해야 `considered[:limit]`이 음수 slice(마지막
    |limit|개를 제외)로 조용히 재해석되지 않는다(M2_Phase_4_5_6_code_review_result.md
    재리뷰 P3).
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit은 1 이상이어야 합니다: {limit!r}")

    command = [
        "python",
        "-m",
        "evaluation.answers",
        "--dataset",
        str(dataset_path),
        "--output",
        str(output_dir),
    ]
    if limit is not None:
        command += ["--limit", str(limit)]
    if tag is not None:
        command += ["--tag", tag]

    cases: list[GoldenCase] = load_jsonl(dataset_path)

    considered = cases
    if tag is not None:
        considered = [c for c in considered if tag in c.tags]
    if limit is not None:
        considered = considered[:limit]

    eligible = [c for c in considered if is_answer_eval_eligible(c)]

    repro = build_reproducibility_metadata(Path(config.DATA_DIR), Path(config.VECTORSTORE_PATH))

    engine = _get_engine() if eligible else None

    case_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    latency_values_ms: list[float] = []

    assertion_passed_sum = 0
    assertion_total_sum = 0
    assertion_cases_scored = 0
    assertion_cases_excluded_no_assertion = 0
    success_count = 0
    failure_count = 0

    abstention_flags: list[tuple[bool, bool]] = []

    source_any_hit_flags: list[bool] = []
    source_recall_values: list[float] = []
    source_excluded_count = 0
    source_skipped_entries_total = 0

    intent_evaluated = 0
    intent_correct = 0

    for case in eligible:
        expected_intent = case.expected_intent.value if case.expected_intent else None
        assertion_summary = _summarize_assertions(case.answer_assertions)

        start = time.perf_counter()
        error: str | None = None
        result: dict | None = None
        try:
            result = engine.query(case.question)
        except Exception as exc:  # noqa: BLE001 - 개별 사례 실패는 기록 후 계속한다.
            error = str(exc)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latency_values_ms.append(elapsed_ms)

        if error is None and result is not None and not result.get("success", False):
            error = str(result.get("answer") or "success=False(원인 미상)")

        if error is not None:
            failure_count += 1
            failures.append({"id": case.id, "question": case.question, "error": error})
            case_results.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "status": "failure",
                    "error": error,
                    "latency_ms": elapsed_ms,
                    "intent_expected": expected_intent,
                    "intent_actual": None,
                    "intent_match": None,
                    "assertion_passed": None,
                    "assertion_total": None,
                    "expect_abstention": case.expect_abstention,
                    "predicted_abstention": None,
                    "abstention_match": None,
                    "source_any_hit": None,
                    "source_recall": None,
                    "source_excluded_reason": "질의 실패로 채점 제외",
                    "returned_sources": [],
                    "answer_assertions_summary": assertion_summary,
                    "answer": None,
                }
            )
            continue

        success_count += 1
        answer = result.get("answer", "")

        passed, total = assertion_coverage(answer, case.answer_assertions)
        assertion_passed_sum += passed
        assertion_total_sum += total
        if case.answer_assertions:
            assertion_cases_scored += 1
        else:
            assertion_cases_excluded_no_assertion += 1

        predicted_abstention = _detect_abstention(answer)
        abstention_flags.append((case.expect_abstention, predicted_abstention))
        abstention_match = case.expect_abstention == predicted_abstention

        raw_sources = result.get("sources", []) or []
        returned_ids, skipped = _extract_returned_source_ids(raw_sources)
        source_skipped_entries_total += skipped
        match = _source_match(returned_ids, case.relevant_sources)
        if match is None:
            source_excluded_count += 1
            source_any_hit: bool | None = None
            source_recall: float | None = None
            source_excluded_reason: str | None = "relevant_sources 없음"
        else:
            source_any_hit = match["source_any_hit"]
            source_recall = match["source_recall"]
            source_any_hit_flags.append(source_any_hit)
            source_recall_values.append(source_recall)
            source_excluded_reason = None

        actual_intent = result.get("intent")
        if expected_intent is not None:
            intent_evaluated += 1
            intent_match: bool | None = actual_intent == expected_intent
            if intent_match:
                intent_correct += 1
        else:
            intent_match = None

        returned_sources_display = [
            {"source": entry.get("source"), "page": entry.get("page")}
            for entry in raw_sources
            if isinstance(entry, dict) and isinstance(entry.get("source"), str) and entry.get("source")
        ]

        case_results.append(
            {
                "id": case.id,
                "question": case.question,
                "status": "success",
                "error": None,
                "latency_ms": elapsed_ms,
                "intent_expected": expected_intent,
                "intent_actual": actual_intent,
                "intent_match": intent_match,
                "assertion_passed": passed,
                "assertion_total": total,
                "expect_abstention": case.expect_abstention,
                "predicted_abstention": predicted_abstention,
                "abstention_match": abstention_match,
                "source_any_hit": source_any_hit,
                "source_recall": source_recall,
                "source_excluded_reason": source_excluded_reason,
                "returned_sources": returned_sources_display,
                "answer_assertions_summary": assertion_summary,
                "answer": answer,
            }
        )

    abstention_summary = _abstention_confusion(abstention_flags)
    abstention_summary["evaluated_count"] = len(abstention_flags)

    latency_mean, latency_median = mean_median(latency_values_ms)
    latency_summary = {
        "mean_ms": latency_mean,
        "median_ms": latency_median,
        "p95_ms": percentile(latency_values_ms, 95) if latency_values_ms else None,
        "count": len(latency_values_ms),
    }

    extra = {
        "case_counts": {
            "total_considered": len(considered),
            "eligible": len(eligible),
            "excluded_non_eligible": len(considered) - len(eligible),
            "success": success_count,
            "failure": failure_count,
        },
        "assertion": {
            # cases_scored는 "성공했고 answer_assertions가 1개 이상 있는" 사례 수다.
            # 이전에는 success_count를 그대로 썼는데, Answer 평가 대상에는
            # expect_abstention=True이고 assertion이 없는 사례도 포함되므로
            # cases_scored가 실제로 assertion을 채점한 사례 수보다 커 보이는
            # 문제가 있었다(M2_Phase_4_5_6_code_review_result.md P2).
            # cases_scored + cases_excluded_no_assertion + cases_excluded_failure
            # == len(eligible) 불변식이 성립해야 한다.
            "cases_scored": assertion_cases_scored,
            "cases_excluded_no_assertion": assertion_cases_excluded_no_assertion,
            "cases_excluded_failure": failure_count,
            "assertions_total": assertion_total_sum,
            "assertions_passed": assertion_passed_sum,
            "pass_rate": (assertion_passed_sum / assertion_total_sum) if assertion_total_sum > 0 else None,
            "limitation_note": ASSERTION_COVERAGE_LIMITATION_NOTE,
        },
        "abstention": abstention_summary,
        "source": {
            "evaluated_count": len(source_any_hit_flags),
            "excluded_count": source_excluded_count,
            "skipped_entries_total": source_skipped_entries_total,
            "any_hit_rate": (
                sum(source_any_hit_flags) / len(source_any_hit_flags) if source_any_hit_flags else None
            ),
            "mean_recall": (
                sum(source_recall_values) / len(source_recall_values) if source_recall_values else None
            ),
        },
        "intent": {
            "evaluated_count": intent_evaluated,
            "excluded_count": len(eligible) - intent_evaluated,
            "correct_count": intent_correct,
            "accuracy": (intent_correct / intent_evaluated) if intent_evaluated > 0 else None,
        },
        "latency_ms": latency_summary,
        "failures": failures,
        "case_results": case_results,
        "tag_filter": tag,
        "limit": limit,
    }

    payload = build_metadata(dataset_path, command, extra)
    payload.update(repro)

    json_path, md_path = write_report(payload, output_dir, "answers", render_markdown=_render_answers_markdown)
    worksheet_path = output_dir / f"{json_path.stem}_worksheet.md"
    write_review_worksheet(case_results, worksheet_path)

    output = dict(payload)
    output["report_json_path"] = str(json_path)
    output["report_markdown_path"] = str(md_path)
    output["worksheet_path"] = str(worksheet_path)
    return output


def _render_answers_markdown(payload: dict) -> str:
    """사람이 assertion pass rate/abstention 정확도/source hit·recall/intent
    정확도/latency/실패 수를 JSON을 열지 않고 바로 읽을 수 있는 Markdown을
    만든다(M2_Phase_4_5_6_code_review_result.md P1 — 공통 `_render_markdown()`은
    dict/list 필드를 렌더링하지 않는다). 사례별 상세는 별도 review worksheet와
    JSON에 있으므로 여기서는 집계 결과와 실패 사례 목록만 다룬다."""
    lines: list[str] = ["# answers", ""]
    generated_at = payload.get("generated_at_utc")
    if generated_at:
        lines.append(f"생성 시각(UTC): {generated_at}")
    lines.append("")

    counts = payload.get("case_counts") or {}
    lines.append("## 요약")
    lines.append("")
    lines.append(
        f"- 대상 사례: considered={counts.get('total_considered')} eligible={counts.get('eligible')} "
        f"(제외 {counts.get('excluded_non_eligible')}) · success={counts.get('success')} "
        f"failure={counts.get('failure')}"
    )
    lines.append("")

    assertion = payload.get("assertion") or {}
    pass_rate = assertion.get("pass_rate")
    pass_rate_display = "N/A" if pass_rate is None else f"{pass_rate:.1%}"
    lines.append("## Assertion")
    lines.append("")
    lines.append(
        f"- pass rate: {pass_rate_display} ({assertion.get('assertions_passed')}/{assertion.get('assertions_total')}) · "
        f"채점 사례 {assertion.get('cases_scored')}건 "
        f"(assertion 없음 제외 {assertion.get('cases_excluded_no_assertion')}건, "
        f"질의 실패 제외 {assertion.get('cases_excluded_failure')}건)"
    )
    lines.append(f"- 한계: {assertion.get('limitation_note')}")
    lines.append("")

    abstention = payload.get("abstention") or {}
    acc = abstention.get("accuracy")
    acc_display = "N/A" if acc is None else f"{acc:.1%}"
    lines.append("## Abstention")
    lines.append("")
    lines.append(
        f"- accuracy: {acc_display} (평가 대상 {abstention.get('evaluated_count')}건) · "
        f"TP={abstention.get('true_positive')} TN={abstention.get('true_negative')} "
        f"FP={abstention.get('false_positive')} FN={abstention.get('false_negative')}"
    )
    if abstention.get("abstention_accuracy_excluded_reason"):
        lines.append(f"- 제외 사유: {abstention.get('abstention_accuracy_excluded_reason')}")
    lines.append("")

    source = payload.get("source") or {}
    any_hit_rate = source.get("any_hit_rate")
    mean_recall = source.get("mean_recall")
    lines.append("## Source")
    lines.append("")
    lines.append(
        f"- any_hit_rate: {'N/A' if any_hit_rate is None else f'{any_hit_rate:.1%}'} · "
        f"mean_recall: {'N/A' if mean_recall is None else f'{mean_recall:.3f}'} "
        f"(평가 대상 {source.get('evaluated_count')}건, relevant_sources 없어 제외 {source.get('excluded_count')}건, "
        f"source entry 스킵 {source.get('skipped_entries_total')}건)"
    )
    lines.append("")

    intent = payload.get("intent") or {}
    intent_acc = intent.get("accuracy")
    lines.append("## Intent")
    lines.append("")
    lines.append(
        f"- accuracy: {'N/A' if intent_acc is None else f'{intent_acc:.1%}'} "
        f"({intent.get('correct_count')}/{intent.get('evaluated_count')}, 제외 {intent.get('excluded_count')}건)"
    )
    lines.append("")

    latency = payload.get("latency_ms") or {}
    lines.append("## Latency (ms)")
    lines.append("")
    mean_ms = latency.get("mean_ms")
    if mean_ms is None:
        lines.append("- 데이터 없음")
    else:
        lines.append(f"- mean: {mean_ms:.1f} · median: {latency.get('median_ms'):.1f} · p95: {latency.get('p95_ms'):.1f}")
    lines.append("")

    failures = payload.get("failures") or []
    lines.append(f"## 실패 사례 ({len(failures)}건)")
    lines.append("")
    if failures:
        lines.append("| id | 질문 | 오류 |")
        lines.append("|---|---|---|")
        for f in failures:
            case_id = escape_markdown_table_cell(f.get("id"))
            question = escape_markdown_table_cell(f.get("question"))
            error = escape_markdown_table_cell(f.get("error"))
            lines.append(f"| {case_id} | {question} | {error} |")
    else:
        lines.append("(없음)")
    lines.append("")
    lines.append(
        "사례별 상세 자동 점수와 답변 원문은 사람 검토용 worksheet"
        "(같은 디렉터리의 `answers_<timestamp>_worksheet.md`)와 같은 이름의 `.json` 리포트를 참고하세요."
    )
    return "\n".join(lines) + "\n"


def _positive_int(value: str) -> int:
    """--limit용 argparse type. 0/음수는 `cases[:limit]`이 Python 음수 slice로
    조용히 재해석되는 것을 막기 위해 parser 오류(exit 2)로 거부한다
    (M2_Phase_4_5_6_code_review_result.md P3)."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"--limit은 1 이상의 정수여야 합니다: {value!r}")
    return parsed


def _render_case_section(result: dict) -> str:
    lines: list[str] = [f"## {_safe_inline(result.get('id'))}", ""]
    lines.append(f"**질문**: {_safe_inline(result.get('question'))}")
    lines.append("")

    if result.get("status") != "success":
        lines.append(f"**자동 점수**: 질의 실패 — {_safe_inline(result.get('error'))}")
    else:
        passed = result.get("assertion_passed")
        total = result.get("assertion_total")
        abstention_match = result.get("abstention_match")
        any_hit = result.get("source_any_hit")
        recall = result.get("source_recall")
        recall_display = "N/A(제외)" if recall is None else f"{recall:.2f}"
        any_hit_display = "N/A(제외)" if any_hit is None else str(any_hit)
        intent_match = result.get("intent_match")
        intent_display = "N/A(대상 아님)" if intent_match is None else str(intent_match)
        lines.append(
            "**자동 점수**: "
            f"assertion {passed}/{total} · "
            f"abstention 일치 {abstention_match} · "
            f"source_any_hit {any_hit_display} · "
            f"source_recall {recall_display} · "
            f"intent 일치 {intent_display}"
        )
    lines.append("")

    lines.append("**반환 출처**:")
    returned_sources = result.get("returned_sources") or []
    if returned_sources:
        for src in returned_sources:
            page = src.get("page")
            page_display = "N/A" if page is None else str(page)
            lines.append(f"- {_safe_inline(src.get('source'))} (page {page_display})")
    else:
        lines.append("- (없음)")
    lines.append("")

    lines.append("**기대 핵심 사실**:")
    assertions_summary = result.get("answer_assertions_summary") or []
    if assertions_summary:
        for item in assertions_summary:
            lines.append(f"- {_safe_inline(item)}")
    else:
        lines.append("- (없음)")
    lines.append("")

    answer_text = result.get("answer")
    if answer_text is None:
        answer_text = f"(질의 실패로 답변 없음: {result.get('error') or '원인 미상'})"
    fence = _fence_for(answer_text)
    lines.append("**모델 답변**:")
    lines.append(fence)
    lines.append(answer_text)
    lines.append(fence)
    lines.append("")

    lines.append("**Faithfulness (1-5)**: _(빈칸)_")
    lines.append("**Relevance (1-5)**: _(빈칸)_")
    lines.append("**Completeness (1-5)**: _(빈칸)_")
    lines.append("**Citation correctness (1-5)**: _(빈칸)_")
    lines.append("**Reviewer note**: _(빈칸)_")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def write_review_worksheet(results: list[dict], output_path: Path) -> Path:
    """사람이 채점할 Markdown worksheet를 만든다. 표의 한 셀에 답변 전체를
    넣지 않고 사례마다 별도 section으로 렌더링한다(구조가 줄바꿈/파이프/코드
    블록이 섞인 LLM 답변에 깨지지 않도록). `sources[].content`(검색 chunk
    원문)는 results 자체에 담겨 오지 않으므로 여기서도 절대 노출되지 않는다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "# Answer evaluator 사람 검토 worksheet",
        "",
        "자동 채점(assertion/abstention/source/intent)은 참고용입니다. "
        "Faithfulness/Relevance/Completeness/Citation correctness/Reviewer note는 "
        "사람이 직접 채점합니다. 검색 chunk 원문은 이 문서에 포함되지 않습니다.",
        "",
    ]
    if not results:
        header.append("(Answer 평가 대상 사례가 없습니다.)")
        header.append("")

    body = "\n".join(header) + "\n".join(_render_case_section(r) for r in results)
    output_path.write_text(body, encoding="utf-8")
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation.answers")
    parser.add_argument("--dataset", type=Path, required=True, help="golden.jsonl 경로")
    parser.add_argument("--output", type=Path, required=True, help="리포트를 저장할 디렉터리")
    parser.add_argument("--limit", type=_positive_int, default=None, help="평가 대상 상한(원본 순서 기준, 1 이상)")
    parser.add_argument("--tag", type=str, default=None, help="이 tag를 가진 사례만 평가")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = evaluate_answers(args.dataset, args.output, limit=args.limit, tag=args.tag)
    except DatasetError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        if exc.kind == "io":
            print(
                "다음 조치: --dataset 경로가 올바른지, 접근 권한이 있는지, UTF-8로 "
                "저장됐는지 확인한 뒤 다시 실행하세요.",
                file=sys.stderr,
            )
        else:
            print(
                "다음 조치: golden.jsonl의 해당 줄/사례를 수정한 뒤 다시 실행하세요.",
                file=sys.stderr,
            )
        return 1
    except FileNotFoundError as exc:
        print(f"오류: 필수 리소스를 찾을 수 없습니다: {exc}", file=sys.stderr)
        print(
            "다음 조치: runtime/documents/ 디렉터리와 runtime/vectorstore/index.faiss·index.pkl이 존재하는지 "
            "확인하세요. vectorstore가 없다면 `simple-qna-rag-index`를 실행해 "
            "생성한 뒤 다시 실행하세요.",
            file=sys.stderr,
        )
        return 1
    except RuntimeError as exc:
        print(f"오류: RAG 엔진 초기화에 실패했습니다: {exc}", file=sys.stderr)
        print(
            "다음 조치: Ollama가 실행 중인지, vectorstore가 준비돼 있는지 확인한 뒤 "
            "다시 실행하세요.",
            file=sys.stderr,
        )
        return 1

    summary = {
        "case_counts": result["case_counts"],
        "assertion": result["assertion"],
        "abstention": result["abstention"],
        "source": result["source"],
        "intent": result["intent"],
        "latency_ms": result["latency_ms"],
        "report_json_path": result["report_json_path"],
        "report_markdown_path": result["report_markdown_path"],
        "worksheet_path": result["worksheet_path"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
