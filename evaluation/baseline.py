"""
통합 baseline runner (M2-REQ-009, M2-REQ-010, Phase 7).

dataset validation -> Retrieval -> live Routing -> Answer 평가를 정해진 순서로
실행하고, 단계별 결과와 재현성 metadata를 하나의 통합 JSON/Markdown 리포트로
만든다. 이 모듈은 evaluator(retrieval/routing/answers)의 내부 알고리즘을
재구현하지 않고 각 evaluator의 공개 API(`evaluate_retrieval()`,
`evaluate_routing()`, `evaluate_answers()`)만 호출한다(상위 규칙 문서 §3.2).

이 모듈은 import 시점에 `agent.py`/`rag_engine.py`/Ollama를 로드하지 않는다.
`agent._decide_tool`은 `--skip-routing`이 아니고 opt-in(RUN_LIVE_LLM_TESTS=1)을
통과했을 때만 `run_baseline()` 내부에서 `_resolve_decide_tool()`을 통해
지연 import된다. `run_baseline()`은 항상 live Routing/Answer를 시도하는
API이므로(§4.1) `run_baseline()` 자신도 호출 즉시 opt-in을 검사해
`RuntimeError`를 던진다 — CLI `main()`이 더 친절한 안내 메시지와 함께 같은
검사를 먼저 수행해 정상 경로에서는 이 예외가 발생하지 않지만, `run_baseline()`을
CLI 없이 Python에서 직접 호출하는 경우에도 opt-in 없이 실제 모델이 호출되지
않도록 하는 방어선이다(M2_Phase_7_8_code_review_result.md P2). `run_baseline()`은
`sys.exit()`을 호출하지 않고 결과 dict를 반환하거나(정상/부분 실패) `ValueError`
(잘못된 `limit`)/`RuntimeError`(opt-in 없음)만 던진다(§4.5).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.answers import evaluate_answers
from evaluation.dataset import DatasetError, load_jsonl, validate_composition
from evaluation.reporting import (
    build_metadata,
    build_not_applicable_reproducibility_metadata,
    escape_markdown_table_cell,
    write_report,
)
from evaluation.retrieval import evaluate_retrieval
from evaluation.routing import evaluate_routing, render_routing_markdown

SCHEMA_VERSION = "1.0.0"

_BASE_LIMITATIONS = [
    "assertion coverage(핵심 문구 포함 여부)는 답변의 진실성(faithfulness) 전체를 "
    "보증하지 않습니다. 사람 검토 worksheet(answers 단계 산출물)로 반드시 보완해야 합니다.",
    "corpus_manifest_sha256/vectorstore_fingerprint는 파일 내용 해시 기반 지문일 뿐, "
    "vectorstore/가 현재 config.py 설정으로 실제로 생성됐다는 provenance(생성 이력)를 "
    "보장하지 않습니다.",
    "Routing 단계는 corpus/vectorstore를 사용하지 않으므로 재현성 fingerprint가 "
    "not_applicable(null)입니다 — top-level fingerprint는 Retrieval 단계 값만 반영합니다.",
]


def _stage_success(name: str, **fields) -> dict:
    return {"stage": name, "status": "success", **fields}


def _stage_failed(
    name: str,
    error_type: str,
    message: str,
    *,
    path: str | None = None,
    case_id: str | None = None,
    next_action: str,
    **fields,
) -> dict:
    return {
        "stage": name,
        "status": "failed",
        "error_type": error_type,
        "message": message,
        "path": path,
        "case_id": case_id,
        "next_action": next_action,
        **fields,
    }


def _stage_skipped(name: str, reason: str) -> dict:
    return {"stage": name, "status": "skipped", "reason": reason}


def _stage_not_run(name: str, reason: str) -> dict:
    return {"stage": name, "status": "not_run", "reason": reason}


def _resolve_decide_tool():
    """live Routing 평가에 쓸 실제 `agent._decide_tool()`을 지연 import한다.

    `--skip-routing`이 아니고(main()의) RUN_LIVE_LLM_TESTS=1 opt-in을 통과한
    뒤에만 호출된다. 오프라인 테스트는 이 함수 자체를 monkeypatch해 실제
    `agent.py`/`rag_engine.py` import(및 그로 인한 모델/Ollama 초기화)를 피한다."""
    from agent import _decide_tool

    return _decide_tool


def run_baseline(
    dataset_path: Path,
    output_dir: Path,
    *,
    skip_routing: bool = False,
    skip_answers: bool = False,
    limit: int | None = None,
    tag: str | None = None,
) -> dict:
    """validate -> retrieval -> routing -> answers 순서로 실행한다.

    `sys.exit()`을 호출하지 않는다 — 항상 결과 dict를 반환하거나(정상/부분
    실패 모두 dict), `limit`이 잘못된 경우 `ValueError`를, `RUN_LIVE_LLM_TESTS=1`
    opt-in이 없는 경우 `RuntimeError`를 던진다(§4.4/§4.8, Python API 계약).
    opt-in 검사는 CLI `main()`도 별도로 수행하지만(더 친절한 안내 메시지 포함),
    `run_baseline()`을 CLI 없이 직접 호출하는 경로도 안전해야 하므로 이 함수
    자신도 어떤 evaluator도 호출하기 전에 동일한 검사를 한다
    (M2_Phase_7_8_code_review_result.md P2). 최종 성공 여부는 반환 dict의
    `overall_success` 키로 판단한다. CLI `main()`이 이 값을 보고 exit code를
    결정한다.

    단계 순서: dataset 로드 -> composition validation -> (실패 시 이후 단계는
    모두 `not_run`으로 표시하고 즉시 반환) -> Retrieval(스킵 옵션 없음) ->
    (skip_routing이 아니면) live Routing -> (skip_answers가 아니면) Answer.
    Retrieval이 실패해도 Routing/Answer는 각자 독립적으로 계속 시도한다
    (Routing/Answer 둘 다 corpus/vectorstore 접근을 Retrieval의 성공 여부와
    무관하게 자체적으로 수행하기 때문이다) — 다만 이 경우 top-level
    corpus_manifest/vectorstore_fingerprint는 계산할 방법이 없으므로 null로
    남고 그 사유가 reproducibility_note에 기록된다.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit은 1 이상이어야 합니다: {limit!r}")

    if os.environ.get("RUN_LIVE_LLM_TESTS") != "1":
        raise RuntimeError(
            "통합 baseline은 실제 Retrieval/Ollama LLM/vectorstore를 사용합니다. "
            "RUN_LIVE_LLM_TESTS=1 환경변수 없이는 run_baseline()을 실행할 수 없습니다. "
            "CLI를 통해 실행한다면 opt-in 안내가 이미 main()에서 먼저 출력됩니다."
        )

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)

    start_time = time.perf_counter()
    stage_durations: dict[str, float] = {}
    stages: dict[str, dict] = {}

    command = [
        "python", "-m", "evaluation.baseline",
        "--dataset", str(dataset_path),
        "--output", str(output_dir),
    ]
    if skip_routing:
        command.append("--skip-routing")
    if skip_answers:
        command.append("--skip-answers")
    if limit is not None:
        command += ["--limit", str(limit)]
    if tag is not None:
        command += ["--tag", tag]

    options = {"skip_routing": skip_routing, "skip_answers": skip_answers, "limit": limit, "tag": tag}

    # ---- Stage 1: validate ----
    t0 = time.perf_counter()
    try:
        all_cases = load_jsonl(dataset_path)
    except DatasetError as exc:
        stage_durations["validate"] = time.perf_counter() - t0
        stages["validate"] = _stage_failed(
            "validate", type(exc).__name__, str(exc),
            path=str(dataset_path), case_id=exc.case_id,
            next_action="--dataset 경로와 golden.jsonl 내용을 확인한 뒤 다시 실행하세요.",
        )
        stages["retrieval"] = _stage_not_run("retrieval", "dataset를 로드하지 못해 실행하지 않음")
        stages["routing"] = _stage_not_run("routing", "dataset를 로드하지 못해 실행하지 않음")
        stages["answers"] = _stage_not_run("answers", "dataset를 로드하지 못해 실행하지 않음")
        stage_durations["retrieval"] = 0.0
        stage_durations["routing"] = 0.0
        stage_durations["answers"] = 0.0
        total_duration = time.perf_counter() - start_time
        base_extra = {
            "options": options,
            "overall_success": False,
            "stages": stages,
            "corpus_manifest": None,
            "corpus_manifest_sha256": None,
            "vectorstore_fingerprint": None,
            "reproducibility_note": "dataset 로드 실패로 재현성 metadata를 계산하지 않음",
            "reproducibility_limitations": list(_BASE_LIMITATIONS),
            "fingerprint_invariant": {"checked": False, "ok": None},
            "total_duration_seconds": total_duration,
            "stage_duration_seconds": stage_durations,
        }
        if exc.kind == "io":
            # dataset 파일 자체를 읽을 수 없다 — build_metadata()는 내부적으로
            # dataset_path.read_bytes()를 호출하므로 여기서는 리포트 파일을
            # 만들지 않고 결과 dict만 반환한다(기존 evaluator main()들이 이
            # 상황에서 리포트를 쓰지 않는 것과 동일한 정책).
            return {
                "schema_version": SCHEMA_VERSION,
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "command": command,
                "dataset_path": str(dataset_path),
                "dataset_sha256": None,
                **base_extra,
            }
        payload = build_metadata(dataset_path, command, base_extra)
        write_report(payload, output_dir, "baseline", render_markdown=_render_baseline_markdown)
        return payload

    validation_report = validate_composition(all_cases)
    stage_durations["validate"] = time.perf_counter() - t0

    if not validation_report.is_valid:
        stages["validate"] = _stage_failed(
            "validate", "CompositionValidationError",
            f"골든셋 구성 검증 실패: {len(validation_report.errors)}건",
            path=str(dataset_path), case_id=None,
            next_action="evaluation/datasets/golden.jsonl 구성을 보완한 뒤 다시 실행하세요.",
            errors=validation_report.errors,
            validation_report=validation_report.to_dict(),
        )
        stages["retrieval"] = _stage_not_run("retrieval", "dataset validation 실패로 실행하지 않음")
        stages["routing"] = _stage_not_run("routing", "dataset validation 실패로 실행하지 않음")
        stages["answers"] = _stage_not_run("answers", "dataset validation 실패로 실행하지 않음")
        stage_durations["retrieval"] = 0.0
        stage_durations["routing"] = 0.0
        stage_durations["answers"] = 0.0
        total_duration = time.perf_counter() - start_time
        extra = {
            "options": options,
            "overall_success": False,
            "stages": stages,
            "corpus_manifest": None,
            "corpus_manifest_sha256": None,
            "vectorstore_fingerprint": None,
            "reproducibility_note": "dataset validation 실패로 이후 단계를 실행하지 않아 재현성 metadata를 계산하지 않음",
            "reproducibility_limitations": list(_BASE_LIMITATIONS),
            "fingerprint_invariant": {"checked": False, "ok": None},
            "total_duration_seconds": total_duration,
            "stage_duration_seconds": stage_durations,
        }
        payload = build_metadata(dataset_path, command, extra)
        write_report(payload, output_dir, "baseline", render_markdown=_render_baseline_markdown)
        return payload

    stages["validate"] = _stage_success(
        "validate",
        total=validation_report.total,
        by_category=validation_report.by_category,
        validation_report=validation_report.to_dict(),
    )

    # ---- Stage 2: retrieval (스킵 옵션 없음) ----
    t0 = time.perf_counter()
    retrieval_output_dir = output_dir / "retrieval"
    retrieval_payload: dict | None = None
    try:
        retrieval_payload = evaluate_retrieval(dataset_path, retrieval_output_dir, limit=limit, tag=tag)
    except DatasetError as exc:
        stages["retrieval"] = _stage_failed(
            "retrieval", type(exc).__name__, str(exc),
            path=str(dataset_path), case_id=exc.case_id,
            next_action="dataset 내용을 수정한 뒤 다시 실행하세요.",
        )
    except FileNotFoundError as exc:
        stages["retrieval"] = _stage_failed(
            "retrieval", type(exc).__name__, str(exc),
            next_action=(
                "data/ 와 vectorstore/ 가 준비돼 있는지 확인하세요. vectorstore가 없다면 "
                "`python document_register.py`를 실행해 data/의 문서를 등록/색인한 뒤 "
                "다시 실행하세요."
            ),
        )
    except Exception as exc:  # noqa: BLE001 - 엔진 초기화 등 예상 밖 실패도 기록 후 다음 단계를 계속한다
        stages["retrieval"] = _stage_failed(
            "retrieval", type(exc).__name__, str(exc),
            next_action="Ollama가 실행 중인지, vectorstore/data가 올바른지 확인한 뒤 다시 실행하세요.",
        )
    else:
        failing_cases = [
            {"id": c.get("id"), "question": c.get("question"), "error": c.get("error")}
            for c in (retrieval_payload.get("case_results") or [])
            if not c.get("success", True)
        ]
        stages["retrieval"] = _stage_success(
            "retrieval",
            report_json_path=retrieval_payload.get("report_json_path"),
            report_markdown_path=retrieval_payload.get("report_markdown_path"),
            case_counts=retrieval_payload.get("case_counts"),
            metrics=retrieval_payload.get("metrics"),
            latency_ms=retrieval_payload.get("latency_ms"),
            failures=failing_cases,
        )
    stage_durations["retrieval"] = time.perf_counter() - t0

    if stages["retrieval"]["status"] == "success":
        top_corpus_manifest = retrieval_payload.get("corpus_manifest")
        top_corpus_manifest_sha256 = retrieval_payload.get("corpus_manifest_sha256")
        top_vectorstore_fingerprint = retrieval_payload.get("vectorstore_fingerprint")
        top_reproducibility_note = retrieval_payload.get("reproducibility_note")
    else:
        top_corpus_manifest = None
        top_corpus_manifest_sha256 = None
        top_vectorstore_fingerprint = None
        top_reproducibility_note = (
            "Retrieval 단계가 실패해 top-level corpus_manifest/vectorstore_fingerprint를 "
            "생성할 수 없습니다."
        )

    # ---- Stage 3: routing ----
    if skip_routing:
        stages["routing"] = _stage_skipped("routing", "사용자가 --skip-routing으로 명시적으로 제외함")
        stage_durations["routing"] = 0.0
    else:
        t0 = time.perf_counter()
        filtered_cases = list(all_cases)
        if tag is not None:
            filtered_cases = [c for c in filtered_cases if tag in c.tags]
        if limit is not None:
            filtered_cases = filtered_cases[:limit]

        try:
            decide_tool = _resolve_decide_tool()
            routing_result = evaluate_routing(filtered_cases, decide_tool, measure_latency=True)
        except ValueError as exc:
            stages["routing"] = _stage_failed(
                "routing", type(exc).__name__, str(exc),
                next_action="--tag/--limit 필터를 완화하거나 조건을 만족하는 사례를 golden.jsonl에 추가하세요.",
            )
        except Exception as exc:  # noqa: BLE001 - agent/Ollama 초기화 실패 등도 기록 후 다음 단계를 계속한다
            stages["routing"] = _stage_failed(
                "routing", type(exc).__name__, str(exc),
                next_action="Ollama가 실행 중인지, agent 설정이 올바른지 확인한 뒤 다시 실행하세요.",
            )
        else:
            # evaluate_routing() 자체는 성공했다 — 이제 결과를 report로 남기는
            # 단계다. M2_Phase_7_8_code_review_result.md P1: 이 report 작성
            # (metadata 생성 + write_report())도 실패할 수 있고(디스크 공간,
            # 권한 등), 이전에는 이 try 블록 밖에 있어서 예외가 run_baseline()
            # 밖으로 그대로 전파돼 Answer 단계가 실행되지 않고 이미 성공한
            # Retrieval 결과도 최종 report에 남지 않았다. evaluate_routing()의
            # 성공(evaluation_status)과 report 파일 생성 성공(report_status)을
            # 구분해 기록하되, report는 필수 산출물이므로 report 작성이
            # 실패하면 단계 최종 상태는 failed로 남긴다.
            try:
                routing_output_dir = output_dir / "routing"
                routing_command = [
                    "python", "-m", "evaluation.routing",
                    "--dataset", str(dataset_path), "--output", str(routing_output_dir), "--mode", "live",
                ]
                if tag is not None:
                    routing_command += ["--tag", tag]
                if limit is not None:
                    routing_command += ["--limit", str(limit)]
                reproducibility = build_not_applicable_reproducibility_metadata(
                    "routing은 corpus/vectorstore를 사용하지 않음"
                )
                routing_payload = build_metadata(
                    dataset_path, routing_command,
                    {"mode": "live", "tag_filter": tag, "limit": limit, **routing_result, **reproducibility},
                )
                json_path, md_path = write_report(
                    routing_payload, routing_output_dir, "routing", render_markdown=render_routing_markdown
                )
            except Exception as exc:  # noqa: BLE001 - report 작성 실패도 단계 실패로 기록하고 Answer는 계속 실행한다
                stages["routing"] = _stage_failed(
                    "routing", type(exc).__name__, str(exc),
                    path=str(output_dir / "routing"),
                    next_action="output 디렉터리의 쓰기 권한과 디스크 여유 공간을 확인한 뒤 다시 실행하세요.",
                    evaluation_status="success",
                    report_status="failed",
                    total_cases=routing_result.get("total_cases"),
                    correct_count=routing_result.get("correct_count"),
                    accuracy=routing_result.get("accuracy"),
                )
            else:
                stages["routing"] = _stage_success(
                    "routing",
                    report_json_path=str(json_path),
                    report_markdown_path=str(md_path),
                    total_cases=routing_result.get("total_cases"),
                    correct_count=routing_result.get("correct_count"),
                    accuracy=routing_result.get("accuracy"),
                    precision_recall_f1=routing_result.get("precision_recall_f1"),
                    latency_ms=routing_result.get("latency_ms"),
                    failures=routing_result.get("failures"),
                )
        stage_durations["routing"] = time.perf_counter() - t0

    # ---- Stage 4: answers ----
    answers_payload: dict | None = None
    if skip_answers:
        stages["answers"] = _stage_skipped("answers", "사용자가 --skip-answers로 명시적으로 제외함")
        stage_durations["answers"] = 0.0
    else:
        t0 = time.perf_counter()
        answers_output_dir = output_dir / "answers"
        try:
            answers_payload = evaluate_answers(dataset_path, answers_output_dir, limit=limit, tag=tag)
        except DatasetError as exc:
            stages["answers"] = _stage_failed(
                "answers", type(exc).__name__, str(exc),
                path=str(dataset_path), case_id=exc.case_id,
                next_action="dataset 내용을 수정한 뒤 다시 실행하세요.",
            )
        except FileNotFoundError as exc:
            stages["answers"] = _stage_failed(
                "answers", type(exc).__name__, str(exc),
                next_action=(
                    "data/ 디렉터리와 vectorstore/index.faiss·index.pkl이 존재하는지 확인하세요. "
                    "vectorstore가 없다면 `python document_register.py`를 실행해 생성한 뒤 "
                    "다시 실행하세요."
                ),
            )
        except Exception as exc:  # noqa: BLE001 - RAG 엔진 초기화 실패 등도 기록 후 baseline 결과를 보존한다
            stages["answers"] = _stage_failed(
                "answers", type(exc).__name__, str(exc),
                next_action="Ollama가 실행 중인지, vectorstore가 준비돼 있는지 확인한 뒤 다시 실행하세요.",
            )
        else:
            stages["answers"] = _stage_success(
                "answers",
                report_json_path=answers_payload.get("report_json_path"),
                report_markdown_path=answers_payload.get("report_markdown_path"),
                worksheet_path=answers_payload.get("worksheet_path"),
                case_counts=answers_payload.get("case_counts"),
                assertion=answers_payload.get("assertion"),
                abstention=answers_payload.get("abstention"),
                source=answers_payload.get("source"),
                intent=answers_payload.get("intent"),
                latency_ms=answers_payload.get("latency_ms"),
                failures=answers_payload.get("failures"),
            )
        stage_durations["answers"] = time.perf_counter() - t0

    # ---- fingerprint invariant (§4.7) ----
    fingerprint_invariant = {
        "checked": False,
        "ok": None,
        "corpus_manifest_sha256_match": None,
        "vectorstore_fingerprint_match": None,
        "retrieval_corpus_manifest_sha256": top_corpus_manifest_sha256,
        "answers_corpus_manifest_sha256": None,
        "retrieval_vectorstore_fingerprint": top_vectorstore_fingerprint,
        "answers_vectorstore_fingerprint": None,
    }
    fingerprint_mismatch = False
    if (
        stages["retrieval"]["status"] == "success"
        and stages["answers"]["status"] == "success"
        and answers_payload is not None
    ):
        answers_corpus_sha = answers_payload.get("corpus_manifest_sha256")
        answers_vs_fp = answers_payload.get("vectorstore_fingerprint")
        corpus_match = answers_corpus_sha == top_corpus_manifest_sha256
        vs_match = answers_vs_fp == top_vectorstore_fingerprint
        fingerprint_invariant.update(
            {
                "checked": True,
                "ok": bool(corpus_match and vs_match),
                "corpus_manifest_sha256_match": corpus_match,
                "vectorstore_fingerprint_match": vs_match,
                "answers_corpus_manifest_sha256": answers_corpus_sha,
                "answers_vectorstore_fingerprint": answers_vs_fp,
            }
        )
        fingerprint_mismatch = not fingerprint_invariant["ok"]

    stage_ok = (
        stages["validate"]["status"] == "success"
        and stages["retrieval"]["status"] == "success"
        and stages["routing"]["status"] in ("success", "skipped")
        and stages["answers"]["status"] in ("success", "skipped")
    )
    overall_success = bool(stage_ok and not fingerprint_mismatch)

    total_duration = time.perf_counter() - start_time

    limitations = list(_BASE_LIMITATIONS)
    if top_reproducibility_note:
        limitations.append(top_reproducibility_note)
    if fingerprint_mismatch:
        limitations.append(
            "Answer 단계가 독립적으로 계산한 corpus_manifest_sha256/vectorstore_fingerprint가 "
            "Retrieval 단계 값과 다릅니다 — baseline 실행 도중 data/ 또는 vectorstore/가 "
            "변경되었을 가능성이 있습니다. 두 값과 각 단계 결과는 모두 아래 보존돼 있습니다."
        )

    extra = {
        "options": options,
        "overall_success": overall_success,
        "stages": stages,
        "corpus_manifest": top_corpus_manifest,
        "corpus_manifest_sha256": top_corpus_manifest_sha256,
        "vectorstore_fingerprint": top_vectorstore_fingerprint,
        "reproducibility_note": top_reproducibility_note,
        "reproducibility_limitations": limitations,
        "fingerprint_invariant": fingerprint_invariant,
        "total_duration_seconds": total_duration,
        "stage_duration_seconds": stage_durations,
    }
    payload = build_metadata(dataset_path, command, extra)
    write_report(payload, output_dir, "baseline", render_markdown=_render_baseline_markdown)
    return payload


def _render_baseline_markdown(payload: dict) -> str:
    """사람이 전체 성공 여부, 단계별 핵심 지표/latency, 실패한 단계와 주요
    실패 사례, fingerprint와 해석 한계를 JSON을 열지 않고 바로 읽을 수 있는
    Markdown을 만든다(retrieval/routing/answers의 `_render_*_markdown()` 스타일을
    따른다)."""
    lines: list[str] = ["# baseline", ""]
    generated_at = payload.get("generated_at_utc")
    if generated_at:
        lines.append(f"생성 시각(UTC): {generated_at}")
    lines.append(f"command: `{' '.join(payload.get('command') or [])}`")
    lines.append(f"dataset: {payload.get('dataset_path')} (sha256={payload.get('dataset_sha256')})")
    lines.append(f"git commit: {payload.get('git_commit')} (dirty={payload.get('git_dirty')})")
    lines.append("")

    overall = payload.get("overall_success")
    lines.append(f"## 전체 결과: {'성공' if overall else '실패'}")
    lines.append("")
    stages = payload.get("stages") or {}
    durations = payload.get("stage_duration_seconds") or {}
    lines.append("| 단계 | 상태 | 소요 시간(초) |")
    lines.append("|---|---|---|")
    for name in ("validate", "retrieval", "routing", "answers"):
        stage = stages.get(name) or {}
        dur = durations.get(name)
        dur_display = "N/A" if dur is None else f"{dur:.2f}"
        lines.append(f"| {name} | {escape_markdown_table_cell(stage.get('status'))} | {dur_display} |")
    lines.append("")
    total_dur = payload.get("total_duration_seconds")
    if total_dur is not None:
        lines.append(f"총 실행 시간: {total_dur:.2f}초")
    lines.append("")

    # Retrieval
    retrieval = stages.get("retrieval") or {}
    lines.append("## Retrieval")
    lines.append("")
    if retrieval.get("status") == "success":
        counts = retrieval.get("case_counts") or {}
        lines.append(
            f"- 사례: total={counts.get('total')} success={counts.get('success')} "
            f"failure={counts.get('failure')} excluded={counts.get('excluded')}"
        )
        metrics = retrieval.get("metrics") or {}
        for key in sorted(metrics.keys()):
            value = metrics[key]
            value_display = "N/A" if value is None else f"{value:.3f}"
            lines.append(f"- {key}: {value_display}")
        latency = retrieval.get("latency_ms") or {}
        if latency.get("mean") is not None:
            lines.append(
                f"- latency mean/median/p95(ms): {latency.get('mean'):.1f}/"
                f"{latency.get('median'):.1f}/{latency.get('p95'):.1f}"
            )
        if retrieval.get("report_json_path"):
            lines.append(f"- 상세 리포트: {retrieval.get('report_json_path')}")
    else:
        lines.append(f"- 상태: {retrieval.get('status')} — {retrieval.get('message') or retrieval.get('reason')}")
    lines.append("")

    # Routing
    routing = stages.get("routing") or {}
    lines.append("## Routing")
    lines.append("")
    if routing.get("status") == "success":
        accuracy = routing.get("accuracy") or 0.0
        lines.append(
            f"- accuracy: {accuracy:.1%} ({routing.get('correct_count')}/{routing.get('total_cases')})"
        )
        pr = routing.get("precision_recall_f1") or {}
        for route in ("document_qa", "web_search"):
            m = pr.get(route) or {}
            lines.append(
                f"- {route}: precision={m.get('precision', 0.0):.3f} "
                f"recall={m.get('recall', 0.0):.3f} f1={m.get('f1', 0.0):.3f}"
            )
        failures = routing.get("failures") or []
        lines.append(f"- 실패 {len(failures)}건")
        if routing.get("report_json_path"):
            lines.append(f"- 상세 리포트: {routing.get('report_json_path')}")
    else:
        lines.append(f"- 상태: {routing.get('status')} — {routing.get('message') or routing.get('reason')}")
    lines.append("")

    # Answers
    answers = stages.get("answers") or {}
    lines.append("## Answers")
    lines.append("")
    if answers.get("status") == "success":
        assertion = answers.get("assertion") or {}
        pass_rate = assertion.get("pass_rate")
        lines.append(
            f"- assertion pass rate: {'N/A' if pass_rate is None else f'{pass_rate:.1%}'} "
            f"({assertion.get('assertions_passed')}/{assertion.get('assertions_total')})"
        )
        abstention = answers.get("abstention") or {}
        acc = abstention.get("accuracy")
        lines.append(f"- abstention accuracy: {'N/A' if acc is None else f'{acc:.1%}'}")
        source = answers.get("source") or {}
        any_hit = source.get("any_hit_rate")
        lines.append(f"- source any_hit_rate: {'N/A' if any_hit is None else f'{any_hit:.1%}'}")
        intent = answers.get("intent") or {}
        intent_acc = intent.get("accuracy")
        lines.append(f"- intent accuracy: {'N/A' if intent_acc is None else f'{intent_acc:.1%}'}")
        latency = answers.get("latency_ms") or {}
        if latency.get("mean_ms") is not None:
            lines.append(f"- latency mean(ms): {latency.get('mean_ms'):.1f}")
        failures = answers.get("failures") or []
        lines.append(f"- 실패 {len(failures)}건")
        if answers.get("report_json_path"):
            lines.append(f"- 상세 리포트: {answers.get('report_json_path')}")
    else:
        lines.append(f"- 상태: {answers.get('status')} — {answers.get('message') or answers.get('reason')}")
    lines.append("")

    # 실패한 단계와 주요 실패 사례(단계 자체 실패 + 성공한 단계 내 개별 사례 실패 모두 포함)
    lines.append("## 실패한 단계와 주요 실패 사례")
    lines.append("")
    any_failure_content = False
    for name in ("validate", "retrieval", "routing", "answers"):
        stage = stages.get(name) or {}
        status = stage.get("status")
        case_failures = stage.get("failures") or []
        if status != "failed" and not case_failures:
            continue
        any_failure_content = True
        lines.append(f"### {name} ({status})")
        if status == "failed":
            lines.append(f"- 오류 유형: {escape_markdown_table_cell(stage.get('error_type'))}")
            lines.append(f"- 메시지: {escape_markdown_table_cell(stage.get('message'))}")
            lines.append(f"- 다음 조치: {escape_markdown_table_cell(stage.get('next_action'))}")
        if case_failures:
            lines.append("")
            lines.append(f"사례별 실패 {len(case_failures)}건 (상위 20건까지 표시):")
            lines.append("")
            lines.append("| id | 질문 | 오류 |")
            lines.append("|---|---|---|")
            for f in case_failures[:20]:
                lines.append(
                    f"| {escape_markdown_table_cell(f.get('id'))} | "
                    f"{escape_markdown_table_cell(f.get('question'))} | "
                    f"{escape_markdown_table_cell(f.get('error'))} |"
                )
        lines.append("")
    if not any_failure_content:
        lines.append("(실패한 단계나 사례 없음)")
        lines.append("")

    # Fingerprint
    lines.append("## 재현성 fingerprint")
    lines.append("")
    lines.append(f"- corpus_manifest_sha256: {payload.get('corpus_manifest_sha256')}")
    vs_fp = payload.get("vectorstore_fingerprint") or {}
    lines.append(f"- vectorstore index.faiss sha256: {vs_fp.get('index_faiss_sha256')}")
    lines.append(f"- vectorstore index.pkl sha256: {vs_fp.get('index_pkl_sha256')}")
    fp_invariant = payload.get("fingerprint_invariant") or {}
    if fp_invariant.get("checked"):
        lines.append(f"- Retrieval/Answer fingerprint 일치: {fp_invariant.get('ok')}")
        if not fp_invariant.get("ok"):
            lines.append(
                f"  - corpus_manifest_sha256 일치: {fp_invariant.get('corpus_manifest_sha256_match')} "
                f"(retrieval={fp_invariant.get('retrieval_corpus_manifest_sha256')}, "
                f"answers={fp_invariant.get('answers_corpus_manifest_sha256')})"
            )
            lines.append(f"  - vectorstore_fingerprint 일치: {fp_invariant.get('vectorstore_fingerprint_match')}")
    else:
        lines.append(
            "- Retrieval/Answer fingerprint 비교: 수행되지 않음"
            "(Answer 단계 skip/미실행/실패 또는 Retrieval 실패)"
        )
    lines.append("")

    lines.append("## 알려진 해석 한계")
    lines.append("")
    for note in payload.get("reproducibility_limitations") or []:
        lines.append(f"- {note}")
    if payload.get("reproducibility_note"):
        lines.append(f"- {payload.get('reproducibility_note')}")
    lines.append("")

    lines.append("각 단계의 전체 사례별 상세는 위 상세 리포트 경로 또는 같은 이름의 `.json`을 참고하세요.")
    return "\n".join(lines) + "\n"


def _positive_int(value: str) -> int:
    """--limit용 argparse type. 0/음수는 slice가 조용히 음수로 재해석되는 것을
    막기 위해 parser 오류(exit 2)로 거부한다(retrieval.py/routing.py/answers.py와
    동일한 패턴)."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"--limit은 1 이상의 정수여야 합니다: {value!r}")
    return parsed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.baseline",
        description=(
            "dataset validation -> Retrieval -> live Routing -> Answer 평가를 순서대로 "
            "실행하고 통합 리포트를 생성합니다. 실제 Ollama LLM과 vectorstore를 사용하므로 "
            "RUN_LIVE_LLM_TESTS=1 환경변수가 필요합니다."
        ),
    )
    parser.add_argument("--dataset", type=Path, required=True, help="golden.jsonl 경로")
    parser.add_argument("--output", type=Path, required=True, help="리포트를 저장할 디렉터리")
    parser.add_argument("--skip-routing", action="store_true", help="Routing 평가를 제외합니다")
    parser.add_argument("--skip-answers", action="store_true", help="Answer 평가를 제외합니다")
    parser.add_argument(
        "--limit", type=_positive_int, default=None, help="평가 대상 상한(원본 순서 기준, 1 이상)"
    )
    parser.add_argument("--tag", type=str, default=None, help="이 tag를 가진 사례만 평가")
    return parser


def main(argv: list[str] | None = None) -> int:
    """`--help`는 어떤 live dependency도 초기화하지 않고 argparse 자체가
    exit(0)로 종료한다. live opt-in 확인은 dataset 로드보다도 먼저 수행한다
    (§4.8) — opt-in 없이는 어떤 모델/vectorstore/Ollama 접근도 일어나지 않는다."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if os.environ.get("RUN_LIVE_LLM_TESTS") != "1":
        print(
            "통합 baseline은 실제 Retrieval/Ollama LLM/vectorstore를 사용합니다. "
            "명시적 opt-in 없이는 실행하지 않습니다.",
            file=sys.stderr,
        )
        print(
            "다음 조치: 로컬에 Ollama와 data/, vectorstore/가 준비돼 있다면 "
            "RUN_LIVE_LLM_TESTS=1 환경변수를 설정한 뒤 다시 실행하세요. vectorstore가 "
            "없다면 먼저 `python document_register.py`로 data/의 문서를 등록/색인하세요. 예:\n"
            f"  RUN_LIVE_LLM_TESTS=1 python -m evaluation.baseline "
            f"--dataset {args.dataset} --output {args.output}",
            file=sys.stderr,
        )
        return 2

    try:
        payload = run_baseline(
            args.dataset,
            args.output,
            skip_routing=args.skip_routing,
            skip_answers=args.skip_answers,
            limit=args.limit,
            tag=args.tag,
        )
    except ValueError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2

    overall_success = payload.get("overall_success")
    print(f"Baseline 실행 완료: overall_success={overall_success}", file=sys.stderr)
    for name, stage in (payload.get("stages") or {}).items():
        print(f"  - {name}: {stage.get('status')}", file=sys.stderr)
    print(json.dumps({"overall_success": overall_success}, ensure_ascii=False))
    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
