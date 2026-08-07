"""Intent Classifier 효용 대조 실험 — paired blind harness (M3-REQ-007,
Design.md §8).

Fixes the retrieved context once per case and runs both the current
intent-per-template path (variant A) and a fixed default-template path
(variant B) through it, so any answer difference is attributable to the
template alone (M3-NFR-004 — no retrieval/prompt logic is reimplemented
here; `RAGEngine.build_context()`/`format_sources()`/`generate_answer()`
and `_retrieve_documents()` are the only entry points used).

This module does not call `get_rag_engine()` at import time.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation import answer_rules as ar
from evaluation.dataset import DatasetError, load_jsonl
from evaluation.metrics import dedupe_preserve_order
from evaluation.schema import is_answer_eval_eligible, normalize_source_id
from simple_qna_rag.intent_classifier import classify_intent
from simple_qna_rag.prompt_templates import DEFAULT_TEMPLATE, get_template_by_intent

SCHEMA_VERSION = "1.0.0"
DEFAULT_SEED = "m3-intent-ab"

VARIANT_INTENT = "intent"
VARIANT_DEFAULT = "default"


def _get_engine():
    from simple_qna_rag.rag_engine import get_rag_engine

    return get_rag_engine()


def _extract_source_ids(sources: list[dict]) -> list[str]:
    ids = []
    for entry in sources:
        source = entry.get("source") if isinstance(entry, dict) else None
        if isinstance(source, str) and source:
            ids.append(source)
    return ids


def _score_variant(answer: str, case, variants: ar.VariantTable | None) -> dict:
    passed, total, _ = ar.assertion_coverage_v2(case.id, answer, case.answer_assertions, variants)
    abstention_v1 = ar.detect_abstention_v1(answer)
    abstention_v2 = ar.detect_abstention_v2(answer)
    return {
        "assertion_passed_v2": passed,
        "assertion_total_v2": total,
        "predicted_abstention_v1": abstention_v1,
        "abstention_match_v1": abstention_v1 == case.expect_abstention,
        "predicted_abstention_v2": abstention_v2,
        "abstention_match_v2": abstention_v2 == case.expect_abstention,
    }


def _source_scores(returned_ids: list[str], relevant_sources: list[str]) -> dict | None:
    if not relevant_sources:
        return None
    returned = set(dedupe_preserve_order([normalize_source_id(s) for s in returned_ids]))
    relevant = set(dedupe_preserve_order([normalize_source_id(s) for s in relevant_sources]))
    return {
        "source_any_hit": bool(returned & relevant),
        "source_recall": len(returned & relevant) / len(relevant),
    }


def run_experiment(
    dataset_path: Path,
    output_dir: Path,
    *,
    warmup_cases: int = 0,
    seed: str = DEFAULT_SEED,
    variants: ar.VariantTable | None = None,
) -> dict:
    """29건(is_answer_eval_eligible) 각각에 대해 context를 1회 고정하고 두
    variant(intent/default)를 같은 context로 실행한다. worksheet/key/
    context_snapshot 파일을 만들고 집계 payload를 반환한다."""
    try:
        cases = load_jsonl(dataset_path)
    except DatasetError:
        raise
    eligible = [c for c in cases if is_answer_eval_eligible(c)]

    output_dir.mkdir(parents=True, exist_ok=True)

    if not eligible:
        raise ValueError("Intent A/B 평가 대상 사례가 없습니다(is_answer_eval_eligible==True인 사례 0건)")

    engine = _get_engine()

    if warmup_cases:
        for case in eligible[:warmup_cases]:
            try:
                engine._retrieve_documents(case.question)
            except Exception:  # noqa: BLE001 - warm-up 실패는 무시하고 계속한다
                pass

    checkpoint_path = output_dir / "intent_ab_checkpoint.json"
    context_snapshot: dict[str, Any] = {}
    case_results: list[dict] = []
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("dataset_path") == str(dataset_path) and checkpoint.get("seed") == seed:
            context_snapshot = checkpoint.get("context_snapshot", {})
            case_results = checkpoint.get("case_results", [])
    results_by_id = {entry["id"]: entry for entry in case_results}

    def save_checkpoint() -> None:
        checkpoint_path.write_text(
            json.dumps(
                {
                    "dataset_path": str(dataset_path),
                    "seed": seed,
                    "context_snapshot": context_snapshot,
                    "case_results": list(results_by_id.values()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    rng_key = random.Random(seed)

    for case in eligible:
        existing = results_by_id.get(case.id)
        if existing and existing.get("status") == "success" and set(existing.get("answers", {})) == {
            VARIANT_INTENT,
            VARIANT_DEFAULT,
        }:
            continue
        status = "success"
        error = None
        docs = []
        try:
            docs = engine._retrieve_documents(case.question)
        except Exception as exc:  # noqa: BLE001
            status = "failure"
            error = f"{type(exc).__name__}: {exc}"

        entry: dict[str, Any] = existing or {
            "id": case.id,
            "question": case.question,
            "status": status,
            "error": error,
        }
        entry.update(status=status, error=error)

        if status == "success":
            if case.id in context_snapshot:
                context = context_snapshot[case.id]["context"]
                sources = context_snapshot[case.id]["sources"]
            else:
                context = engine.build_context(docs)
                sources = engine.format_sources(docs)
                context_snapshot[case.id] = {"context": context, "sources": sources}
                save_checkpoint()

            answers = entry.setdefault("answers", {})
            for variant_name, template_str_fn in (
                (VARIANT_INTENT, lambda: get_template_by_intent(_safe_classify_intent(case.question))),
                (VARIANT_DEFAULT, lambda: DEFAULT_TEMPLATE),
            ):
                if variant_name in answers:
                    continue
                template_str = template_str_fn()
                answer = engine.generate_answer(case.question, context, template_str)
                score = _score_variant(answer, case, variants)
                source_score = _source_scores(_extract_source_ids(sources), case.relevant_sources)
                answers[variant_name] = {
                    "answer": answer,
                    **score,
                    "source_any_hit": source_score["source_any_hit"] if source_score else None,
                    "source_recall": source_score["source_recall"] if source_score else None,
                }
                results_by_id[case.id] = entry
                save_checkpoint()
            entry["answers"] = answers
            entry["intent_label"] = _safe_classify_intent(case.question)

        results_by_id[case.id] = entry
        save_checkpoint()

    case_results = [results_by_id[case.id] for case in eligible]

    context_path = output_dir / "context_snapshot.json"
    context_path.write_text(json.dumps(context_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    worksheet_path, key_path = write_blind_worksheet(case_results, output_dir, seed=seed)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "seed": seed,
        "case_counts": {
            "eligible": len(eligible),
            "success": sum(1 for c in case_results if c["status"] == "success"),
            "failure": sum(1 for c in case_results if c["status"] == "failure"),
        },
        "case_results": case_results,
        "context_snapshot_path": str(context_path),
        "worksheet_path": str(worksheet_path),
        "key_path": str(key_path),
    }
    checkpoint_path.unlink(missing_ok=True)
    return payload


def _safe_classify_intent(question: str) -> str:
    try:
        return classify_intent(question)
    except Exception:  # noqa: BLE001 - Intent A/B는 실제 classify_intent() 경로 자체를 비교 대상으로 삼는다
        return "other"


def _fence_for(text: str) -> str:
    longest_run = 0
    current = 0
    for ch in text:
        if ch == "`":
            current += 1
            longest_run = max(longest_run, current)
        else:
            current = 0
    return "`" * max(longest_run + 1, 3)


def write_blind_worksheet(case_results: list[dict], output_dir: Path, *, seed: str) -> tuple[Path, Path]:
    """variant 정체·순서를 가린 사람 검토용 worksheet와, 그 매핑을 담은
    key 파일(Git 제외)을 만든다(§8.3). 순서는 `random.Random(f"{seed}:{case_id}")`
    로 결정해 완전히 재현 가능하다."""
    lines = [
        "# Intent A/B 사람 검토 worksheet",
        "",
        "자동 채점(assertion/abstention/source)은 참고용입니다. 아래 5줄만 채점하세요. "
        "출력 1/2의 variant 정체는 이 문서에 없습니다(key 파일에만 있음).",
        "",
    ]
    key_cases = []

    for entry in case_results:
        case_id = entry["id"]
        lines.append(f"## {case_id}")
        lines.append("")
        lines.append(f"**질문**: {entry['question']}")
        lines.append("")

        if entry["status"] != "success":
            lines.append(f"(질의 실패: {entry.get('error')})")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        rng = random.Random(f"{seed}:{case_id}")
        slots = [VARIANT_INTENT, VARIANT_DEFAULT]
        rng.shuffle(slots)
        slot1_variant, slot2_variant = slots

        for slot_index, variant_name in enumerate((slot1_variant, slot2_variant), start=1):
            answer_text = entry["answers"][variant_name]["answer"]
            fence = _fence_for(answer_text)
            lines.append(f"**출력 {slot_index}**:")
            lines.append(fence)
            lines.append(answer_text)
            lines.append(fence)
            lines.append("")

        lines.append(
            "<!-- 아래 5줄만 수정하세요. 값은 0 또는 1, 선호는 1/2/tie -->"
        )
        lines.append("- 출력1_형식적합성: _")
        lines.append("- 출력1_핵심사실보존: _")
        lines.append("- 출력2_형식적합성: _")
        lines.append("- 출력2_핵심사실보존: _")
        lines.append("- 선호: _")
        lines.append("- 검토메모:")
        lines.append("")
        lines.append("---")
        lines.append("")

        key_cases.append(
            {
                "id": case_id,
                "slot1": slot1_variant,
                "slot2": slot2_variant,
                "intent_label": entry.get("intent_label"),
            }
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    worksheet_path = output_dir / f"intent_ab_{ts}_worksheet.md"
    key_path = output_dir / f"intent_ab_{ts}_key.json"

    worksheet_path.write_text("\n".join(lines), encoding="utf-8")
    key_path.write_text(
        json.dumps({"seed": seed, "cases": key_cases}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return worksheet_path, key_path


# ---------------------------------------------------------------------------
# worksheet 파서 + 집계 (§8.4)
# ---------------------------------------------------------------------------

_FIELD_RE = re.compile(
    r"-\s*출력1_형식적합성:\s*(\S+)\s*\n"
    r"-\s*출력1_핵심사실보존:\s*(\S+)\s*\n"
    r"-\s*출력2_형식적합성:\s*(\S+)\s*\n"
    r"-\s*출력2_핵심사실보존:\s*(\S+)\s*\n"
    r"-\s*선호:\s*(\S+)"
)
_HEADING_RE = re.compile(r"^## (\S+)\s*$", re.MULTILINE)
_ALLOWED_BINARY = {"0", "1"}
_ALLOWED_PREFERENCE = {"1", "2", "tie"}


def _parse_worksheet(worksheet_text: str) -> dict[str, dict]:
    """case_id -> {"complete": bool, "output1_format":.., ..., "preference":..}."""
    headings = list(_HEADING_RE.finditer(worksheet_text))
    parsed: dict[str, dict] = {}
    for index, match in enumerate(headings):
        case_id = match.group(1)
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(worksheet_text)
        block = worksheet_text[start:end]
        field_match = _FIELD_RE.search(block)
        if field_match is None:
            parsed[case_id] = {"complete": False}
            continue
        o1f, o1c, o2f, o2c, pref = field_match.groups()
        values_ok = (
            o1f in _ALLOWED_BINARY
            and o1c in _ALLOWED_BINARY
            and o2f in _ALLOWED_BINARY
            and o2c in _ALLOWED_BINARY
            and pref in _ALLOWED_PREFERENCE
        )
        if not values_ok:
            parsed[case_id] = {"complete": False}
            continue
        parsed[case_id] = {
            "complete": True,
            "output1_format": int(o1f),
            "output1_facts": int(o1c),
            "output2_format": int(o2f),
            "output2_facts": int(o2c),
            "preference": pref,
        }
    return parsed


def aggregate_worksheet(worksheet_path: Path, key_path: Path) -> dict:
    """채점된 worksheet와 key 파일을 결합해 요구사항 §4.2 결정 근거를
    만든다. 값이 비었거나(`_`) 허용값이 아닌 사례는 `incomplete`로 표시하고
    집계에서 제외한다(부분 채점을 성공으로 취급하지 않음)."""
    worksheet_text = worksheet_path.read_text(encoding="utf-8")
    key_payload = json.loads(key_path.read_text(encoding="utf-8"))
    key_by_case = {c["id"]: c for c in key_payload["cases"]}

    parsed = _parse_worksheet(worksheet_text)

    preferred_intent = 0
    preferred_default = 0
    tie = 0
    incomplete = 0
    format_intent_sum = 0
    format_default_sum = 0
    facts_intent_sum = 0
    facts_default_sum = 0
    scored_cases = 0

    for case_id, key_entry in key_by_case.items():
        result = parsed.get(case_id)
        if result is None or not result.get("complete"):
            incomplete += 1
            continue
        scored_cases += 1

        slot_variant = {"1": key_entry["slot1"], "2": key_entry["slot2"]}
        slot_scores = {
            "1": (result["output1_format"], result["output1_facts"]),
            "2": (result["output2_format"], result["output2_facts"]),
        }
        for slot, variant in slot_variant.items():
            fmt, facts = slot_scores[slot]
            if variant == VARIANT_INTENT:
                format_intent_sum += fmt
                facts_intent_sum += facts
            else:
                format_default_sum += fmt
                facts_default_sum += facts

        pref = result["preference"]
        if pref == "tie":
            tie += 1
        else:
            preferred_variant = slot_variant[pref]
            if preferred_variant == VARIANT_INTENT:
                preferred_intent += 1
            else:
                preferred_default += 1

    n_scored = scored_cases
    margin_pp = ((preferred_intent - preferred_default) / n_scored * 100) if n_scored > 0 else None

    return {
        "schema_version": SCHEMA_VERSION,
        "worksheet_path": str(worksheet_path),
        "key_path": str(key_path),
        "counts": {
            "preferred_intent": preferred_intent,
            "preferred_default": preferred_default,
            "tie": tie,
            "incomplete": incomplete,
            "scored_cases": n_scored,
            "total_cases": len(key_by_case),
        },
        "margin_pp": margin_pp,
        "axis_sums": {
            "format_intent": format_intent_sum,
            "format_default": format_default_sum,
            "facts_intent": facts_intent_sum,
            "facts_default": facts_default_sum,
        },
        "decision": _decide(margin_pp, incomplete),
    }


def _decide(margin_pp: float | None, incomplete: int) -> str:
    """요구사항 §4.2: margin_pp >= 20.0이고 판단 불가 사례가 없어야 '유지
    가능'이다. 그 외(동률/미달/incomplete>0)는 보수적으로 '입증되지 않음'."""
    if margin_pp is None or incomplete > 0:
        return "unproven"
    if margin_pp >= 20.0:
        return "retain_candidate"
    return "unproven"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"0 이상이어야 합니다: {value!r}")
    return parsed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation.intent_ab")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="paired blind 실험 실행")
    run_parser.add_argument("--dataset", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--warmup-cases", type=_positive_int, default=0)
    run_parser.add_argument("--seed", type=str, default=DEFAULT_SEED)

    agg_parser = sub.add_parser("aggregate", help="채점된 worksheet 집계")
    agg_parser.add_argument("--worksheet", type=Path, required=True)
    agg_parser.add_argument("--key", type=Path, required=True)
    agg_parser.add_argument("--output", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            payload = run_experiment(
                args.dataset, args.output, warmup_cases=args.warmup_cases, seed=args.seed
            )
        except (DatasetError, ValueError, FileNotFoundError, RuntimeError) as exc:
            print(f"오류: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(payload["case_counts"], ensure_ascii=False, indent=2))
        print(f"worksheet: {payload['worksheet_path']}", file=sys.stderr)
        return 0

    if args.command == "aggregate":
        try:
            result = aggregate_worksheet(args.worksheet, args.key)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            print(f"오류: {exc}", file=sys.stderr)
            return 2
        args.output.mkdir(parents=True, exist_ok=True)
        out_path = args.output / "intent_ab_decision.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
