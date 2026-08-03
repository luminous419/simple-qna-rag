"""
Retrieval evaluator (M2-REQ-005, M2-REQ-006).

production `RAGEngine._retrieve_documents()`를 그대로 호출해 Recall@K/MRR@10/nDCG@10과
단계별(BM25/Dense/RRF/MMR/Reranker/전체) latency, 후보 문서 수를 측정한다. 검색
알고리즘 자체는 여기서 재구현하지 않는다 — `rag_engine.RAGEngine._retrieve_documents()`가
유일한 검색 로직이다(상위 계획 §3.4, §3.5).

이 모듈은 import 시점에 `get_rag_engine()`을 호출하지 않는다(M2-NFR-003). 모델/
벡터스토어 로딩은 `evaluate_retrieval()` 안에서 실제로 평가 대상 사례가 있을 때만
발생한다.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import config
from evaluation.dataset import DatasetError, load_jsonl
from evaluation.metrics import (
    dedupe_preserve_order,
    mean_median,
    mrr_at_k,
    ndcg_at_k,
    normalize_relevance_grades,
    percentile,
    recall_at_k,
)
from evaluation.reporting import (
    build_metadata,
    build_reproducibility_metadata,
    escape_markdown_table_cell,
    write_report,
)
from evaluation.schema import normalize_source_id

SCHEMA_VERSION = "1.0.0"

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)

# MRR@10/nDCG@10은 M2-REQ-005가 고정한 값이다 — Recall과 달리 k_values 인자로
# 바꾸지 않는다(k_values는 Recall@K의 K 목록만 제어한다).
_MRR_NDCG_K = 10


def _apply_tag_and_limit(cases, tag: str | None, limit: int | None):
    """tag/limit을 원본 순서를 유지한 채 결정론적으로 적용한다.

    CLI(`main()`)는 argparse의 `_positive_int` type으로 0/음수 `--limit`을 이미
    거부하지만, `evaluate_retrieval()`을 Python에서 직접 호출하면 이 계층을
    거치지 않는다. `limit < 1`을 여기서도 검증해야 `cases[:limit]`이 음수
    slice(마지막 |limit|개를 제외)로 조용히 재해석되지 않는다
    (M2_Phase_4_5_6_code_review_result.md 재리뷰 P3)."""
    if limit is not None and limit < 1:
        raise ValueError(f"limit은 1 이상이어야 합니다: {limit!r}")
    selected = [c for c in cases if tag in c.tags] if tag is not None else list(cases)
    if limit is not None:
        selected = selected[:limit]
    return selected


def evaluate_retrieval(
    dataset_path: Path,
    output_dir: Path,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    limit: int | None = None,
    tag: str | None = None,
) -> dict:
    """golden.jsonl의 `relevant_sources`가 있는 사례에 대해 production
    `RAGEngine._retrieve_documents(question, trace=RetrievalTrace())`를 호출하고
    Recall@K/MRR@10/nDCG@10, 단계별 latency/candidate_count를 집계해 JSON/Markdown
    리포트를 쓴 뒤 같은 payload(dict)를 반환한다.

    `get_rag_engine()`은 평가 대상 사례가 최소 1건 있을 때만, 그리고 이 함수
    내부에서만 호출한다(지연 로딩). dataset/schema 오류(`DatasetError`)와 엔진
    초기화/재현성 메타데이터 오류(`FileNotFoundError`, `RuntimeError`)는 즉시
    전파해 호출부(`main()`)가 처리하게 한다. 개별 사례의 검색 실패는 이 함수
    안에서 failure로 기록하고 나머지 사례 평가를 계속한다.
    """
    command = [
        "python", "-m", "evaluation.retrieval",
        "--dataset", str(dataset_path),
        "--output", str(output_dir),
    ]
    if limit is not None:
        command += ["--limit", str(limit)]
    if tag is not None:
        command += ["--tag", tag]

    all_cases = load_jsonl(dataset_path)
    selected_cases = _apply_tag_and_limit(all_cases, tag, limit)

    eligible_cases = [c for c in selected_cases if c.relevant_sources]
    recall_mrr_excluded_count = len(selected_cases) - len(eligible_cases)
    ndcg_eligible_count = sum(1 for c in selected_cases if c.relevance_grades)
    ndcg_excluded_count = len(selected_cases) - ndcg_eligible_count

    engine = None
    success_count = 0
    failure_count = 0
    recall_lists: dict[int, list[float]] = {k: [] for k in k_values}
    mrr_list: list[float] = []
    ndcg_list: list[float] = []
    latencies: list[float] = []
    stage_latency_by_name: dict[str, list[float]] = {}
    stage_candidates_by_name: dict[str, list[int]] = {}
    case_results: list[dict] = []

    for case in eligible_cases:
        if engine is None:
            # 지연 로딩: 평가할 사례가 실제로 있을 때만, 그리고 이 함수 내부에서만
            # get_rag_engine()을 호출한다(M2-NFR-003). import는 여기서 처음
            # 이뤄지므로 evaluation.retrieval 모듈 자체를 import하는 것만으로는
            # rag_engine의 모델/vectorstore가 로드되지 않는다.
            from rag_engine import RetrievalTrace, get_rag_engine

            engine = get_rag_engine()

        trace = RetrievalTrace()
        t0 = time.perf_counter()
        try:
            docs = engine._retrieve_documents(case.question, trace=trace)
        except Exception as exc:  # noqa: BLE001 - 개별 사례 실패는 기록 후 계속 진행한다
            failure_count += 1
            case_results.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": [dataclasses.asdict(s) for s in trace.stages],
                }
            )
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000

        raw_ids = [normalize_source_id(doc.metadata.get("source", "") or "") for doc in docs]
        ranked_ids = dedupe_preserve_order(raw_ids)
        relevant_ids = set(
            dedupe_preserve_order([normalize_source_id(s) for s in case.relevant_sources])
        )
        grades = normalize_relevance_grades(case.relevance_grades)

        case_metrics: dict[str, float | None] = {}
        for k in k_values:
            recall = recall_at_k(ranked_ids, relevant_ids, k)
            case_metrics[f"recall@{k}"] = recall
            recall_lists[k].append(recall)

        mrr_value = mrr_at_k(ranked_ids, relevant_ids, _MRR_NDCG_K)
        case_metrics[f"mrr@{_MRR_NDCG_K}"] = mrr_value
        mrr_list.append(mrr_value)

        if case.relevance_grades:
            ndcg_value = ndcg_at_k(ranked_ids, grades, _MRR_NDCG_K)
            case_metrics[f"ndcg@{_MRR_NDCG_K}"] = ndcg_value
            ndcg_list.append(ndcg_value)
        else:
            case_metrics[f"ndcg@{_MRR_NDCG_K}"] = None

        for stage_trace in trace.stages:
            stage_latency_by_name.setdefault(stage_trace.name, []).append(stage_trace.latency_ms)
            stage_candidates_by_name.setdefault(stage_trace.name, []).append(
                stage_trace.candidate_count
            )

        latencies.append(elapsed_ms)
        success_count += 1
        case_results.append(
            {
                "id": case.id,
                "question": case.question,
                "success": True,
                "ranked_source_ids": ranked_ids,
                "relevant_source_ids": sorted(relevant_ids),
                "metrics": case_metrics,
                "latency_ms": elapsed_ms,
                "trace": [dataclasses.asdict(s) for s in trace.stages],
            }
        )

    recall_agg = {
        f"recall@{k}": (sum(values) / len(values) if values else None)
        for k, values in recall_lists.items()
    }
    mrr_mean = sum(mrr_list) / len(mrr_list) if mrr_list else None
    ndcg_mean = sum(ndcg_list) / len(ndcg_list) if ndcg_list else None

    latency_mean, latency_median = mean_median(latencies)
    latency_p95 = percentile(latencies, 95)

    stage_summary = {}
    for name, lat_values in stage_latency_by_name.items():
        cand_values = stage_candidates_by_name[name]
        cand_mean, cand_median = mean_median([float(c) for c in cand_values])
        lat_mean, lat_median = mean_median(lat_values)
        stage_summary[name] = {
            "count": len(lat_values),
            "latency_ms_mean": lat_mean,
            "latency_ms_median": lat_median,
            "latency_ms_p95": percentile(lat_values, 95),
            "candidate_count_mean": cand_mean,
            "candidate_count_median": cand_median,
        }

    case_counts = {
        "total": len(selected_cases),
        "success": success_count,
        "failure": failure_count,
        "excluded": recall_mrr_excluded_count,
    }

    # Retrieval 리포트는 corpus/vectorstore를 실제로 사용하므로 이 필드들이
    # non-null이어야 한다(M2-REQ-010). data_dir/vectorstore_path가 없으면
    # FileNotFoundError가 그대로 전파되어 main()이 사람이 읽을 오류로 변환한다.
    reproducibility = build_reproducibility_metadata(
        Path(config.DATA_DIR), Path(config.VECTORSTORE_PATH)
    )

    vectorstore_document_count = None
    if engine is not None:
        # opportunistic: engine이 이미 초기화된 경우에만 시도하고, 이 값 하나를
        # 위해 별도로 모델/벡터스토어를 로드하지 않는다.
        try:
            vectorstore_document_count = len(engine.vectorstore.docstore._dict)
        except Exception:
            vectorstore_document_count = None

    extra = {
        "case_counts": case_counts,
        "recall_mrr_excluded_count": recall_mrr_excluded_count,
        "ndcg_excluded_count": ndcg_excluded_count,
        "k_values": list(k_values),
        "mrr_k": _MRR_NDCG_K,
        "ndcg_k": _MRR_NDCG_K,
        "metrics": {
            **recall_agg,
            f"mrr@{_MRR_NDCG_K}": mrr_mean,
            f"ndcg@{_MRR_NDCG_K}": ndcg_mean,
        },
        "latency_ms": {
            "mean": latency_mean,
            "median": latency_median,
            "p95": latency_p95,
        },
        "stage_summary": stage_summary,
        "vectorstore_document_count": vectorstore_document_count,
        "case_results": case_results,
    }
    extra.update(reproducibility)

    payload = build_metadata(dataset_path, command, extra)
    write_report(payload, output_dir, "retrieval", render_markdown=_render_retrieval_markdown)
    return payload


def _render_retrieval_markdown(payload: dict) -> str:
    """사람이 Recall@K/MRR@10/nDCG@10/latency/stage 요약/실패 사례를 JSON을 열지
    않고 바로 읽을 수 있는 Markdown을 만든다(M2_Phase_4_5_6_code_review_result.md
    P1 — 공통 `_render_markdown()`은 dict/list 필드를 렌더링하지 않는다)."""
    lines: list[str] = ["# retrieval", ""]
    generated_at = payload.get("generated_at_utc")
    if generated_at:
        lines.append(f"생성 시각(UTC): {generated_at}")
    lines.append(f"dataset: {payload.get('dataset_path')}")
    lines.append(f"command: `{' '.join(payload.get('command') or [])}`")
    lines.append("")

    counts = payload.get("case_counts") or {}
    lines.append("## 요약")
    lines.append("")
    lines.append(
        f"- 사례: total={counts.get('total')} success={counts.get('success')} "
        f"failure={counts.get('failure')} excluded(relevant_sources 없음)={counts.get('excluded')}"
    )
    lines.append(f"- nDCG 제외(relevance_grades 없음): {payload.get('ndcg_excluded_count')}")
    lines.append(f"- vectorstore 문서 수: {payload.get('vectorstore_document_count')}")
    lines.append("")

    metrics = payload.get("metrics") or {}
    lines.append("## Retrieval 품질 지표")
    lines.append("")
    lines.append("| 지표 | 값 |")
    lines.append("|---|---|")
    for key in sorted(metrics.keys()):
        value = metrics[key]
        value_display = "N/A" if value is None else f"{value:.3f}"
        lines.append(f"| {key} | {value_display} |")
    lines.append("")

    latency = payload.get("latency_ms") or {}
    lines.append("## End-to-end Latency (ms)")
    lines.append("")
    mean, median, p95 = latency.get("mean"), latency.get("median"), latency.get("p95")
    if mean is None:
        lines.append("- 데이터 없음(성공한 사례가 없음)")
    else:
        lines.append(f"- mean: {mean:.1f} · median: {median:.1f} · p95: {p95:.1f}")
    lines.append("")

    stage_summary = payload.get("stage_summary") or {}
    lines.append("## Stage별 요약")
    lines.append("")
    if stage_summary:
        lines.append("| stage | count | latency_ms mean | latency_ms p95 | candidate_count mean |")
        lines.append("|---|---|---|---|---|")
        for name in sorted(stage_summary.keys()):
            s = stage_summary[name]
            lines.append(
                f"| {name} | {s.get('count')} | {s.get('latency_ms_mean'):.1f} | "
                f"{s.get('latency_ms_p95'):.1f} | {s.get('candidate_count_mean'):.1f} |"
            )
    else:
        lines.append("(없음)")
    lines.append("")

    failures = [c for c in (payload.get("case_results") or []) if not c.get("success", True)]
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
    lines.append("전체 사례별 순위·trace는 같은 이름의 `.json` 리포트를 참고하세요.")
    return "\n".join(lines) + "\n"


def _positive_int(value: str) -> int:
    """--limit용 argparse type. 0/음수는 `cases[:limit]`이 Python 음수 slice로
    조용히 재해석되는 것을 막기 위해 parser 오류(exit 2)로 거부한다
    (M2_Phase_4_5_6_code_review_result.md P3)."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"--limit은 1 이상의 정수여야 합니다: {value!r}")
    return parsed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.retrieval",
        description=(
            "production RAGEngine._retrieve_documents()를 사용해 Retrieval 품질"
            "(Recall@K/MRR@10/nDCG@10)과 단계별 latency를 측정합니다."
        ),
    )
    parser.add_argument("--dataset", type=Path, required=True, help="golden.jsonl 경로")
    parser.add_argument(
        "--output", type=Path, required=True, help="JSON/Markdown 리포트를 저장할 디렉터리"
    )
    parser.add_argument(
        "--limit", type=_positive_int, default=None, help="평가할 최대 사례 수(원본 순서 기준 앞에서부터, 1 이상)"
    )
    parser.add_argument("--tag", type=str, default=None, help="이 tag를 가진 사례만 평가")
    return parser


def main(argv: list[str] | None = None) -> int:
    """`--help`는 rag_engine을 호출하지 않고 argparse 자체가 exit(0)로 종료한다.
    옵션 오류는 argparse의 표준 SystemExit(2) 경로를 그대로 사용한다."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        payload = evaluate_retrieval(
            dataset_path=args.dataset,
            output_dir=args.output,
            limit=args.limit,
            tag=args.tag,
        )
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
                f"다음 조치: {args.dataset}의 위 줄/사례를 수정한 뒤 다시 실행하세요.",
                file=sys.stderr,
            )
        return 1
    except FileNotFoundError as exc:
        print(f"오류: 필수 리소스를 찾을 수 없습니다: {exc}", file=sys.stderr)
        print(
            "다음 조치: data/ 와 vectorstore/ 가 준비돼 있는지 확인하세요. vectorstore가 "
            "없다면 `python document_register.py`를 실행해 data/의 문서를 등록/색인한 뒤 "
            "다시 실행하세요.",
            file=sys.stderr,
        )
        return 2
    except RuntimeError as exc:
        print(f"오류: RAG 엔진 초기화에 실패했습니다: {exc}", file=sys.stderr)
        print(
            "다음 조치: vectorstore/ 와 data/ 가 올바르게 준비돼 있는지, Ollama가 실행 "
            "중이고 필요한 모델이 설치돼 있는지 확인하세요. vectorstore가 없다면 "
            "`python document_register.py`를 실행해 문서를 등록/색인한 뒤 다시 실행하세요.",
            file=sys.stderr,
        )
        return 2

    counts = payload["case_counts"]
    print(
        f"Retrieval 평가 완료: total={counts['total']} success={counts['success']} "
        f"failure={counts['failure']} excluded={counts['excluded']} "
        f"ndcg_excluded={payload['ndcg_excluded_count']}",
        file=sys.stderr,
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
