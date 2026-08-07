"""Rescore a stored `evaluation/answers.py` report with v1/v2 rules, without
touching any model, vectorstore, or network (M3-REQ-006, Design.md §5.8).

Only `case_results[].answer` from the stored report is read; the dataset is
re-loaded to get each case's `answer_assertions`/`expect_abstention`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation import answer_rules as ar
from evaluation.dataset import DatasetError, load_jsonl
from evaluation.reporting import write_report
from evaluation.schema import GoldenCase


class RescoreError(RuntimeError):
    """사람이 읽을 오류 + CLI에서 exit 2로 변환한다."""


def rescore_report(
    report_path: Path,
    dataset_path: Path,
    *,
    variants: ar.VariantTable | None,
) -> dict:
    """저장된 answers 리포트를 읽어 v1/v2로 재채점한다. 원본 리포트는
    수정하지 않는다."""
    try:
        original = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RescoreError(f"리포트를 읽을 수 없습니다: {report_path}: {exc}") from exc

    case_results = original.get("case_results")
    if not isinstance(case_results, list):
        raise RescoreError(f"리포트에 case_results가 없습니다: {report_path}")

    try:
        golden_cases: list[GoldenCase] = load_jsonl(dataset_path)
    except DatasetError as exc:
        raise RescoreError(f"dataset을 읽을 수 없습니다: {exc}") from exc
    golden_by_id = {c.id: c for c in golden_cases}

    fixed_vs_v1: list[dict] = []
    regressed_vs_v1: list[dict] = []
    assertions_total = 0
    assertions_passed = 0
    cases_scored = 0

    abst_tp = abst_tn = abst_fp = abst_fn = 0
    abst_fixed: list[dict] = []
    abst_regressed: list[dict] = []

    rescored_cases: list[dict] = []

    for result in case_results:
        case_id = result.get("id")
        golden = golden_by_id.get(case_id)
        if result.get("status") != "success" or "answer" not in result or result["answer"] is None:
            rescored_cases.append({**result})
            continue
        if golden is None:
            rescored_cases.append({**result})
            continue

        answer = result["answer"]
        answer_norm = ar.normalize_text(answer)

        # assertion v1 (reproduced with the same semantics as v1: NFC + casefold substring)
        import unicodedata

        norm_answer_v1 = unicodedata.normalize("NFC", answer).casefold()
        v1_passed = 0
        for assertion in golden.answer_assertions:
            if any(unicodedata.normalize("NFC", p).casefold() in norm_answer_v1 for p in assertion.any_of):
                v1_passed += 1
        v1_total = len(golden.answer_assertions)

        v2_passed, v2_total, per_assertion = ar.assertion_coverage_v2(
            case_id, answer, golden.answer_assertions, variants
        )

        if golden.answer_assertions:
            cases_scored += 1
            assertions_total += v2_total
            assertions_passed += v2_passed
            if v1_passed < v1_total and v2_passed == v2_total and v2_total > 0:
                fixed_vs_v1.append({"id": case_id, "v1": f"{v1_passed}/{v1_total}", "v2": f"{v2_passed}/{v2_total}"})
            if v2_passed < v1_passed:
                regressed_vs_v1.append(
                    {"id": case_id, "v1": f"{v1_passed}/{v1_total}", "v2": f"{v2_passed}/{v2_total}"}
                )

        v1_abst = ar.detect_abstention_v1(answer)
        v2_abst = ar.detect_abstention_v2(answer)
        expected_abst = golden.expect_abstention
        if expected_abst and v2_abst:
            abst_tp += 1
        elif not expected_abst and not v2_abst:
            abst_tn += 1
        elif not expected_abst and v2_abst:
            abst_fp += 1
        elif expected_abst and not v2_abst:
            abst_fn += 1

        if (v1_abst != expected_abst) and (v2_abst == expected_abst):
            abst_fixed.append({"id": case_id})
        if (v1_abst == expected_abst) and (v2_abst != expected_abst):
            abst_regressed.append({"id": case_id})

        rescored_cases.append(
            {
                **result,
                "assertion_passed_v2": v2_passed,
                "assertion_total_v2": v2_total,
                "assertion_per_v2": per_assertion,
                "predicted_abstention_v2": v2_abst,
                "abstention_match_v2": v2_abst == expected_abst,
            }
        )

    abst_n = abst_tp + abst_tn + abst_fp + abst_fn

    payload = {
        "schema_version": "1.1.0",
        "source_report": str(report_path),
        "dataset_path": str(dataset_path),
        "evaluator_versions": {
            "assertion": "v1+v2",
            "abstention": "v1+v2",
            "rules_fingerprint": ar.rules_fingerprint(variants),
            "reviewed_variants_loaded": variants is not None,
            "reviewed_variants_sha256": variants.sha256 if variants is not None else None,
        },
        "assertion_v2": {
            "cases_scored": cases_scored,
            "assertions_total": assertions_total,
            "assertions_passed": assertions_passed,
            "pass_rate": (assertions_passed / assertions_total) if assertions_total > 0 else None,
            "fixed_vs_v1": fixed_vs_v1,
            "regressed_vs_v1": regressed_vs_v1,
        },
        "abstention_v2": {
            "true_positive": abst_tp,
            "true_negative": abst_tn,
            "false_positive": abst_fp,
            "false_negative": abst_fn,
            "accuracy": ((abst_tp + abst_tn) / abst_n) if abst_n > 0 else None,
            "evaluated_count": abst_n,
            "fixed_vs_v1": abst_fixed,
            "regressed_vs_v1": abst_regressed,
        },
        "case_results": rescored_cases,
    }
    return payload


def _render_rescore_markdown(payload: dict) -> str:
    lines = ["# rescore", ""]
    ev = payload.get("evaluator_versions", {})
    lines.append(f"- source_report: {payload.get('source_report')}")
    lines.append(f"- rules_fingerprint: {ev.get('rules_fingerprint')}")
    lines.append("")
    a = payload.get("assertion_v2", {})
    lines.append("## Assertion (v2)")
    lines.append("")
    rate = a.get("pass_rate")
    lines.append(
        f"- pass rate: {'N/A' if rate is None else f'{rate:.1%}'} "
        f"({a.get('assertions_passed')}/{a.get('assertions_total')}, {a.get('cases_scored')}건)"
    )
    lines.append(f"- fixed_vs_v1: {len(a.get('fixed_vs_v1', []))}건")
    lines.append(f"- regressed_vs_v1: {len(a.get('regressed_vs_v1', []))}건")
    lines.append("")
    b = payload.get("abstention_v2", {})
    lines.append("## Abstention (v2)")
    lines.append("")
    acc = b.get("accuracy")
    lines.append(f"- accuracy: {'N/A' if acc is None else f'{acc:.1%}'} (평가 대상 {b.get('evaluated_count')}건)")
    lines.append(f"- fixed_vs_v1: {len(b.get('fixed_vs_v1', []))}건")
    lines.append(f"- regressed_vs_v1: {len(b.get('regressed_vs_v1', []))}건")
    lines.append("")
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation.rescore")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", type=Path, default=None)
    parser.add_argument("--no-variants", action="store_true", help="변형 표 없이 정규화만 사용")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    variants = None
    if not args.no_variants:
        try:
            variants = ar.load_reviewed_variants(args.variants)
        except ar.VariantTableError as exc:
            print(f"오류: {exc}", file=sys.stderr)
            return 2

    try:
        payload = rescore_report(args.report, args.dataset, variants=variants)
    except RescoreError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2

    json_path, md_path = write_report(payload, args.output, "rescore", render_markdown=_render_rescore_markdown)
    print(json.dumps({"report_json_path": str(json_path), "report_markdown_path": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
