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
from evaluation import answer_rules as ar
from evaluation.dataset import DatasetError, load_jsonl
from evaluation.metrics import assertion_coverage, dedupe_preserve_order, mean_median, percentile
from evaluation.reporting import (
    build_candidate_metadata,
    build_metadata,
    build_reproducibility_metadata,
    build_warmup_metadata,
    escape_markdown_table_cell,
    write_report,
)
from evaluation.schema import AnswerAssertion, GoldenCase, is_answer_eval_eligible, normalize_source_id

SCHEMA_VERSION = "1.2.0"

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


def _intent_summary(template_mode: str, evaluated: int, correct: int, eligible_count: int) -> dict:
    """`ANSWER_TEMPLATE_MODE="default"`에서는 `RAGEngine.query()`가
    `classify_intent()`를 아예 호출하지 않고 응답 계약 보존을 위해 `intent="other"`만
    반환한다(Design.md §8.6, rag_engine.py). 이 상태에서 `expected_intent`와
    비교해 채점하면 비활성 classifier가 실패한 classifier로 오독되는 잘못된 0%
    accuracy가 나온다(Code_Review_Iteration_2.md M2) — 그래서 default 모드에서는
    intent 채점을 전부 제외한다. "intent" 모드에서는 기존 채점 계약(expected_intent가
    있는 사례만 채점)을 그대로 유지한다."""
    if template_mode == "default":
        return {
            "evaluated_count": 0,
            "excluded_count": eligible_count,
            "correct_count": 0,
            "accuracy": None,
            "intent_excluded_reason": "ANSWER_TEMPLATE_MODE=default (classifier 비활성)",
        }
    return {
        "evaluated_count": evaluated,
        "excluded_count": eligible_count - evaluated,
        "correct_count": correct,
        "accuracy": (correct / evaluated) if evaluated > 0 else None,
        "intent_excluded_reason": None,
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
    *,
    candidate_id: str | None = None,
    evaluator_profile: str = "v2",
    expect_variants_sha256: str | None = None,
    warmup_cases: int = 0,
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
    if warmup_cases < 0:
        raise ValueError(f"warmup_cases는 0 이상이어야 합니다: {warmup_cases!r}")
    if evaluator_profile not in ("v2", "v2-no-variants"):
        raise ValueError(f"evaluator_profile은 'v2' 또는 'v2-no-variants'여야 합니다: {evaluator_profile!r}")

    # 호출 시점의 config를 그대로 읽는다 — 테스트는 `config.ANSWER_TEMPLATE_MODE`를
    # monkeypatch해 두 모드를 각각 검증한다(Design.md §8.6).
    template_mode = config.ANSWER_TEMPLATE_MODE

    # §5.5 fail-closed: 공식 profile "v2"는 변형 표가 필수다. VariantTableError는
    # 그대로 호출부(main())까지 전파되어 exit 2로 변환된다.
    variants = ar.load_reviewed_variants(
        required=(evaluator_profile == "v2"),
        expect_sha256=expect_variants_sha256,
    )
    official = evaluator_profile == "v2" and variants is not None

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
    if candidate_id is not None:
        command += ["--candidate-id", candidate_id]
    command += ["--evaluator-profile", evaluator_profile]
    if expect_variants_sha256 is not None:
        command += ["--expect-variants-sha256", expect_variants_sha256]
    if warmup_cases:
        command += ["--warmup-cases", str(warmup_cases)]

    cases: list[GoldenCase] = load_jsonl(dataset_path)

    considered = cases
    if tag is not None:
        considered = [c for c in considered if tag in c.tags]
    if limit is not None:
        considered = considered[:limit]

    eligible = [c for c in considered if is_answer_eval_eligible(c)]

    if warmup_cases > len(eligible):
        raise ValueError(
            f"warmup_cases({warmup_cases})가 평가 대상 사례 수({len(eligible)})보다 큽니다"
        )

    repro = build_reproducibility_metadata(Path(config.DATA_DIR), Path(config.VECTORSTORE_PATH))

    engine = _get_engine() if eligible else None

    warmup_executed = 0
    warmup_succeeded = 0
    warmup_failed = 0
    warmup_case_ids: list[str] = []
    warmup_engine_id = id(engine) if engine is not None else None
    if engine is not None and warmup_cases > 0:
        for case in eligible[:warmup_cases]:
            warmup_executed += 1
            warmup_case_ids.append(case.id)
            try:
                engine.query(case.question)
                warmup_succeeded += 1
            except Exception:  # noqa: BLE001 - warm-up 실패는 집계만 하고 계속한다
                warmup_failed += 1

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

    assertion_v2_passed_sum = 0
    assertion_v2_total_sum = 0
    assertion_v2_cases_scored = 0
    assertion_v2_fixed_vs_v1: list[dict] = []
    assertion_v2_regressed_vs_v1: list[dict] = []
    abstention_v2_flags: list[tuple[bool, bool]] = []
    abstention_v2_fixed_vs_v1: list[dict] = []
    abstention_v2_regressed_vs_v1: list[dict] = []

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

        passed_v2, total_v2, _per_assertion_v2 = ar.assertion_coverage_v2(
            case.id, answer, case.answer_assertions, variants
        )
        if case.answer_assertions:
            assertion_v2_cases_scored += 1
            assertion_v2_passed_sum += passed_v2
            assertion_v2_total_sum += total_v2
            if passed < total and total_v2 > 0 and passed_v2 == total_v2:
                assertion_v2_fixed_vs_v1.append({"id": case.id, "v1": f"{passed}/{total}", "v2": f"{passed_v2}/{total_v2}"})
            if passed_v2 < passed:
                assertion_v2_regressed_vs_v1.append(
                    {"id": case.id, "v1": f"{passed}/{total}", "v2": f"{passed_v2}/{total_v2}"}
                )

        predicted_abstention_v2 = ar.detect_abstention_v2(answer)
        abstention_v2_flags.append((case.expect_abstention, predicted_abstention_v2))
        abstention_match_v2 = case.expect_abstention == predicted_abstention_v2
        if (not abstention_match) and abstention_match_v2:
            abstention_v2_fixed_vs_v1.append({"id": case.id})
        if abstention_match and not abstention_match_v2:
            abstention_v2_regressed_vs_v1.append({"id": case.id})

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
        if template_mode == "default":
            # classifier가 호출되지 않아 채점 대상이 아니다 — _intent_summary()가
            # 집계 단계에서 전부 제외 처리한다.
            intent_match: bool | None = None
        elif expected_intent is not None:
            intent_evaluated += 1
            intent_match = actual_intent == expected_intent
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
                "assertion_passed_v2": passed_v2,
                "assertion_total_v2": total_v2,
                "expect_abstention": case.expect_abstention,
                "predicted_abstention": predicted_abstention,
                "abstention_match": abstention_match,
                "predicted_abstention_v2": predicted_abstention_v2,
                "abstention_match_v2": abstention_match_v2,
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

    abstention_v2_summary = _abstention_confusion(abstention_v2_flags)
    abstention_v2_summary["evaluated_count"] = len(abstention_v2_flags)
    abstention_v2_summary["fixed_vs_v1"] = abstention_v2_fixed_vs_v1
    abstention_v2_summary["regressed_vs_v1"] = abstention_v2_regressed_vs_v1

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
        "evaluator_versions": {
            "assertion": "v1+v2",
            "abstention": "v1+v2",
            "rules_fingerprint": ar.rules_fingerprint(variants),
            "evaluator_profile": evaluator_profile,
            "reviewed_variants_loaded": variants is not None,
            "reviewed_variants_sha256": variants.sha256 if variants is not None else None,
            "official": official,
        },
        "assertion_v2": {
            "cases_scored": assertion_v2_cases_scored,
            "assertions_total": assertion_v2_total_sum,
            "assertions_passed": assertion_v2_passed_sum,
            "pass_rate": (assertion_v2_passed_sum / assertion_v2_total_sum) if assertion_v2_total_sum > 0 else None,
            "fixed_vs_v1": assertion_v2_fixed_vs_v1,
            "regressed_vs_v1": assertion_v2_regressed_vs_v1,
        },
        "abstention_v2": abstention_v2_summary,
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
        "intent": _intent_summary(template_mode, intent_evaluated, intent_correct, len(eligible)),
        "latency_ms": latency_summary,
        "failures": failures,
        "case_results": case_results,
        "tag_filter": tag,
        "limit": limit,
        "measured_case_count": success_count + failure_count,
        "warmup": build_warmup_metadata(
            requested_cases=warmup_cases,
            executed_cases=warmup_executed,
            succeeded_cases=warmup_succeeded,
            failed_cases=warmup_failed,
            case_ids=warmup_case_ids,
            same_process=True,
            engine_object_id_matches=(warmup_engine_id is None or warmup_engine_id == (id(engine) if engine else None)),
        ),
    }

    payload = build_metadata(dataset_path, command, extra)
    payload.update(repro)
    payload["schema_version"] = SCHEMA_VERSION
    payload["candidate"] = build_candidate_metadata(candidate_id)
    payload["answer_template_mode"] = template_mode

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
    lines.append("## Assertion (v1 규칙)")
    lines.append("")
    lines.append(
        f"- pass rate: {pass_rate_display} ({assertion.get('assertions_passed')}/{assertion.get('assertions_total')}) · "
        f"채점 사례 {assertion.get('cases_scored')}건 "
        f"(assertion 없음 제외 {assertion.get('cases_excluded_no_assertion')}건, "
        f"질의 실패 제외 {assertion.get('cases_excluded_failure')}건)"
    )
    lines.append(f"- 한계: {assertion.get('limitation_note')}")
    lines.append("")

    ev = payload.get("evaluator_versions") or {}
    assertion_v2 = payload.get("assertion_v2") or {}
    pass_rate_v2 = assertion_v2.get("pass_rate")
    pass_rate_v2_display = "N/A" if pass_rate_v2 is None else f"{pass_rate_v2:.1%}"
    official_note = "" if ev.get("official") else " — 실험 profile, M3 gate 판정 불가"
    lines.append(f"## Assertion (v2 규칙){official_note}")
    lines.append("")
    lines.append(
        f"- pass rate: {pass_rate_v2_display} "
        f"({assertion_v2.get('assertions_passed')}/{assertion_v2.get('assertions_total')}) · "
        f"채점 사례 {assertion_v2.get('cases_scored')}건 · "
        f"v1 대비 개선 {len(assertion_v2.get('fixed_vs_v1', []))}건 · "
        f"v1 대비 회귀 {len(assertion_v2.get('regressed_vs_v1', []))}건"
    )
    lines.append(
        "- 한계: v2는 표면 표현 차이를 줄여 false negative를 낮출 뿐, 의미 기반 correctness나 "
        "faithfulness를 보증하지 않는다. 부정 표현이 붙은 문장도 v1과 동일하게 일치로 판정될 수 있다."
    )
    lines.append(
        f"- rules_fingerprint: {ev.get('rules_fingerprint')} · profile: {ev.get('evaluator_profile')} · "
        f"official: {ev.get('official')} · reviewed_variants_sha256: {ev.get('reviewed_variants_sha256')}"
    )
    lines.append("")

    abstention = payload.get("abstention") or {}
    acc = abstention.get("accuracy")
    acc_display = "N/A" if acc is None else f"{acc:.1%}"
    lines.append("## Abstention (v1 규칙)")
    lines.append("")
    lines.append(
        f"- accuracy: {acc_display} (평가 대상 {abstention.get('evaluated_count')}건) · "
        f"TP={abstention.get('true_positive')} TN={abstention.get('true_negative')} "
        f"FP={abstention.get('false_positive')} FN={abstention.get('false_negative')}"
    )
    if abstention.get("abstention_accuracy_excluded_reason"):
        lines.append(f"- 제외 사유: {abstention.get('abstention_accuracy_excluded_reason')}")
    lines.append("")

    abstention_v2 = payload.get("abstention_v2") or {}
    acc_v2 = abstention_v2.get("accuracy")
    acc_v2_display = "N/A" if acc_v2 is None else f"{acc_v2:.1%}"
    lines.append(f"## Abstention (v2 규칙){official_note}")
    lines.append("")
    lines.append(
        f"- accuracy: {acc_v2_display} (평가 대상 {abstention_v2.get('evaluated_count')}건) · "
        f"TP={abstention_v2.get('true_positive')} TN={abstention_v2.get('true_negative')} "
        f"FP={abstention_v2.get('false_positive')} FN={abstention_v2.get('false_negative')} · "
        f"v1 대비 개선 {len(abstention_v2.get('fixed_vs_v1', []))}건 · "
        f"v1 대비 회귀 {len(abstention_v2.get('regressed_vs_v1', []))}건"
    )
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
    lines.append(f"- answer_template_mode: {payload.get('answer_template_mode')}")
    lines.append(
        f"- accuracy: {'N/A' if intent_acc is None else f'{intent_acc:.1%}'} "
        f"({intent.get('correct_count')}/{intent.get('evaluated_count')}, 제외 {intent.get('excluded_count')}건)"
    )
    if intent.get("intent_excluded_reason"):
        lines.append(f"- 제외 사유: {intent.get('intent_excluded_reason')}")
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
    parser.add_argument("--candidate-id", type=str, default=None, help="M3 candidate ID(§3.1 정규식)")
    parser.add_argument(
        "--evaluator-profile",
        choices=["v2", "v2-no-variants"],
        default="v2",
        help="공식(v2)은 변형 표 필수(fail-closed). v2-no-variants는 실험 전용, official=false",
    )
    parser.add_argument(
        "--expect-variants-sha256", type=str, default=None, help="변형 표 SHA-256 고정(불일치 시 exit 2)"
    )
    parser.add_argument(
        "--warmup-cases",
        type=int,
        default=0,
        help="동일 process/engine에서 먼저 실행하고 집계에서 제외할 앞쪽 사례 수(0 이상)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.warmup_cases < 0:
        parser.error("--warmup-cases는 0 이상이어야 합니다")

    try:
        result = evaluate_answers(
            args.dataset,
            args.output,
            limit=args.limit,
            tag=args.tag,
            candidate_id=args.candidate_id,
            evaluator_profile=args.evaluator_profile,
            expect_variants_sha256=args.expect_variants_sha256,
            warmup_cases=args.warmup_cases,
        )
    except ar.VariantTableError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        print(
            "다음 조치: evaluation/answer_variants.json이 있는지, schema가 올바른지, "
            "--expect-variants-sha256과 실제 해시가 일치하는지 확인하세요. "
            "변형 표 없이 정규화만 검증하려면 --evaluator-profile v2-no-variants를 명시하세요.",
            file=sys.stderr,
        )
        return 2
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
        "assertion_v2": result["assertion_v2"],
        "abstention": result["abstention"],
        "abstention_v2": result["abstention_v2"],
        "evaluator_versions": result["evaluator_versions"],
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
