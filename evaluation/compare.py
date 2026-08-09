"""Compare two evaluator reports and judge Requirement §4.1 promotion gates
as pure functions (M3-REQ-003, Design.md §5.8).

`M3_GATES`/`evaluate_gates()` are the single source of truth for the M3
promotion thresholds; `evaluation/baseline.py` reuses `evaluate_gates()`
rather than redefining thresholds (Design.md §5.1).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reduction:
    base: float
    ratio: float

    def satisfied(self, value: float) -> bool:
        return value <= self.base * (1 - self.ratio)


@dataclass(frozen=True)
class Gate:
    name: str
    op: str  # "<=" | ">=" | "=="
    threshold: float
    also: Reduction | None = None
    source: str | None = None

    def evaluate(self, value: float | None) -> bool | None:
        if value is None:
            return None
        if self.op == "<=":
            ok = value <= self.threshold
        elif self.op == ">=":
            ok = value >= self.threshold
        elif self.op == "==":
            ok = value == self.threshold
        else:
            raise ValueError(f"지원하지 않는 op: {self.op}")
        if self.also is not None:
            ok = ok and self.also.satisfied(value)
        return ok


@dataclass(frozen=True)
class CountGate:
    name: str
    op: str  # ">=" | "=="
    required_correct: int
    denominator: int

    def evaluate(self, correct: int | None) -> bool | None:
        if correct is None:
            return None
        lhs = Fraction(correct, self.denominator)
        rhs = Fraction(self.required_correct, self.denominator)
        if self.op == ">=":
            return lhs >= rhs
        if self.op == "==":
            return lhs == rhs
        raise ValueError(f"지원하지 않는 op: {self.op}")


M3_GATES: list[Gate | CountGate] = [
    Gate("retrieval_latency_mean_ms", "<=", 8420.0, also=Reduction(16840.0, 0.50)),
    Gate("retrieval_latency_p95_ms", "<=", 13570.0, also=Reduction(22610.0, 0.40)),
    Gate("mmr_latency_mean_ms", "<=", 2869.862, also=Reduction(14349.31, 0.80)),
    Gate("recall@10", ">=", 0.9524),
    Gate("recall@5", ">=", 0.9286),
    Gate("mrr@10", ">=", 0.96),
    Gate("ndcg@10", ">=", 0.93),
    CountGate("routing_accuracy_median", ">=", 69, denominator=76),
    CountGate("routing_document_route_recall_median", ">=", 54, denominator=61),
    CountGate("routing_web_recall_each_run", "==", 15, denominator=15),
    Gate("source_any_hit", "==", 1.0),
    Gate("source_mean_recall", ">=", 0.93),
    Gate("answer_latency_mean_s", "<=", 61.03),
    Gate("answer_latency_p95_s", "<=", 82.37),
]

GATE_SPEC_VERSION = "m3-4.1"


def _get(payload: dict, path: str) -> Any:
    node: Any = payload
    for part in path.split("/"):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _retrieval_gate_inputs(retrieval: dict | None) -> dict:
    if not retrieval:
        return {}
    warmup_ok = _get(retrieval, "warmup/performed")
    fallback_count = _get(retrieval, "mmr_instrumentation/fallback_case_count")
    latency_blocked = warmup_ok is False or (fallback_count is not None and fallback_count > 0)
    latency = retrieval.get("latency_ms") or {}
    mmr_stage = _get(retrieval, "stage_summary/mmr/latency_ms_mean")
    metrics = retrieval.get("metrics") or {}
    return {
        "retrieval_latency_mean_ms": None
        if latency_blocked
        else latency.get("mean_ms", latency.get("mean")),
        "retrieval_latency_p95_ms": None
        if latency_blocked
        else latency.get("p95_ms", latency.get("p95")),
        "mmr_latency_mean_ms": None if latency_blocked else mmr_stage,
        "recall@10": metrics.get("recall@10"),
        "recall@5": metrics.get("recall@5"),
        "mrr@10": metrics.get("mrr@10"),
        "ndcg@10": metrics.get("ndcg@10"),
    }


def _routing_gate_inputs(routing: dict | None) -> dict:
    if not routing:
        return {}
    aggregate = routing.get("aggregate")
    per_run = routing.get("per_run")
    if aggregate is not None:
        accuracy_correct = _get(routing, "aggregate/correct_count/median")
        doc_correct = _get(routing, "aggregate/document_route_correct/median")
    else:
        accuracy_correct = routing.get("correct_count")
        # single-run reports already carry the raw count at the top level
        # (evaluate_routing() -> build_routing_payload() via **result) — use
        # it directly instead of deriving from precision_recall_f1's recall
        # ratio, which CountGate can't compare against a correct/denominator
        # pair (CR-I2-MAJ-02).
        doc_correct = routing.get("document_route_correct")

    web_recall_ok: bool | None = None
    if per_run:
        web_recall_ok = all(run.get("web_search_correct") == 15 for run in per_run)
    elif "web_search_correct" in routing:
        web_recall_ok = routing.get("web_search_correct") == 15

    return {
        "routing_accuracy_median": accuracy_correct,
        "routing_document_route_recall_median": doc_correct,
        "routing_web_recall_each_run": 15 if web_recall_ok else (None if web_recall_ok is None else 0),
    }


def _answers_gate_inputs(answers: dict | None) -> dict:
    if not answers:
        return {}
    warmup_ok = _get(answers, "warmup/performed")
    blocked = warmup_ok is False
    latency = answers.get("latency_ms") or {}
    source = answers.get("source") or {}
    return {
        "source_any_hit": source.get("any_hit_rate"),
        "source_mean_recall": source.get("mean_recall"),
        "answer_latency_mean_s": None if blocked else (
            latency.get("mean_ms") / 1000.0 if latency.get("mean_ms") is not None else None
        ),
        "answer_latency_p95_s": None if blocked else (
            latency.get("p95_ms") / 1000.0 if latency.get("p95_ms") is not None else None
        ),
    }


def evaluate_gates(payload: dict) -> dict:
    """`payload`는 {"retrieval": {...}|None, "routing": {...}|None,
    "answers": {...}|None} 형태다(각 evaluator의 리포트 dict를 그대로 넣는다).
    개별 evaluator 리포트가 없으면 해당 gate는 `pass=null`(판정 불가)이 된다.
    """
    inputs: dict[str, Any] = {}
    inputs.update(_retrieval_gate_inputs(payload.get("retrieval")))
    inputs.update(_routing_gate_inputs(payload.get("routing")))
    inputs.update(_answers_gate_inputs(payload.get("answers")))

    items = []
    for gate in M3_GATES:
        value = inputs.get(gate.name)
        if isinstance(gate, CountGate):
            passed = gate.evaluate(value)
            items.append(
                {
                    "id": gate.name,
                    "metric": value,
                    "threshold_correct": gate.required_correct,
                    "denominator": gate.denominator,
                    "comparison": gate.op,
                    "pass": passed,
                    "value": (value / gate.denominator) if isinstance(value, int) else None,
                }
            )
        else:
            passed = gate.evaluate(value)
            items.append(
                {
                    "id": gate.name,
                    "metric": value,
                    "threshold": gate.threshold,
                    "comparison": gate.op,
                    "pass": passed,
                }
            )

    # overall_pass is True only if every gate resolved to True. A gate that
    # is None (not yet proven / blocked by warmup or fallback) or False both
    # keep overall_pass False — "판정 불가" is not a pass.
    overall_pass = all(item["pass"] is True for item in items)

    return {"spec_version": GATE_SPEC_VERSION, "overall_pass": overall_pass, "items": items}


# ---------------------------------------------------------------------------
# compare_reports (fingerprint check + metric delta + per-case diff)
# ---------------------------------------------------------------------------


class CompareError(RuntimeError):
    pass


_FINGERPRINT_FIELDS = (
    "dataset_sha256",
    "corpus_manifest_sha256",
)


def _vectorstore_fp(payload: dict) -> tuple[Any, Any]:
    fp = payload.get("vectorstore_fingerprint") or {}
    return fp.get("index_faiss_sha256"), fp.get("index_pkl_sha256")


def compare_reports(baseline: dict, candidate: dict, *, kind: str) -> dict:
    """baseline/candidate는 이미 로드된 리포트 dict다(§5.8). fingerprint가
    다르면 `comparable=False`가 되고 호출부(CLI)가 이를 exit 3으로 변환한다."""
    diffs = []
    for field in _FINGERPRINT_FIELDS:
        b_val = baseline.get(field) or _get(baseline, f"execution/{field}") or _get(
            baseline, f"reproducibility/{field}"
        )
        c_val = candidate.get(field)
        if b_val is not None and c_val is not None and b_val != c_val:
            diffs.append({"field": field, "baseline": b_val, "candidate": c_val})

    b_faiss, b_pkl = _vectorstore_fp(baseline) if "vectorstore_fingerprint" in baseline else (
        _get(baseline, "reproducibility/vectorstore_fingerprint/index_faiss_sha256"),
        _get(baseline, "reproducibility/vectorstore_fingerprint/index_pkl_sha256"),
    )
    c_faiss, c_pkl = _vectorstore_fp(candidate)
    if b_faiss and c_faiss and b_faiss != c_faiss:
        diffs.append({"field": "index_faiss_sha256", "baseline": b_faiss, "candidate": c_faiss})
    if b_pkl and c_pkl and b_pkl != c_pkl:
        diffs.append({"field": "index_pkl_sha256", "baseline": b_pkl, "candidate": c_pkl})

    comparable = not diffs

    result: dict[str, Any] = {"kind": kind, "comparable": comparable, "fingerprint_diff": diffs}

    if kind == "retrieval":
        b_metrics = baseline.get("retrieval", baseline).get("metrics") if isinstance(
            baseline.get("retrieval", baseline), dict
        ) else {}
        c_metrics = candidate.get("metrics") or {}
        b_metrics = b_metrics or (baseline.get("metrics") or {})
        deltas = {}
        for key in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"):
            b_val = b_metrics.get(key)
            c_val = c_metrics.get(key)
            deltas[key] = {"baseline": b_val, "candidate": c_val, "delta": (
                c_val - b_val if isinstance(b_val, (int, float)) and isinstance(c_val, (int, float)) else None
            )}
        result["metric_deltas"] = deltas

    return result


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation.compare")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--kind", choices=["retrieval", "routing", "answers", "baseline"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        baseline = _load_json(args.baseline)
        candidate = _load_json(args.candidate)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2

    result = compare_reports(baseline, candidate, kind=args.kind)

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "compare.json"
    out_path.write_text(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))

    return 0 if result["comparable"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
