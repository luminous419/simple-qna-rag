"""
리포트 생성과 재현성 메타데이터 (M2-REQ-010).

build_metadata()는 config.py(순수 상수 모듈, 부수효과 없음)만 읽는다. build_corpus_manifest()/
build_vectorstore_fingerprint()는 각각 data_dir/vectorstore_path의 파일을 바이트 단위로 읽어
해시만 계산하므로 모델이나 FAISS 역직렬화가 필요 없다(개발 계획 §3.6).
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from simple_qna_rag import config
from evaluation.schema import normalize_source_id

SCHEMA_VERSION = "1.0.0"

_REPO_ROOT = Path(__file__).resolve().parent.parent


class CorpusManifestError(Exception):
    """정규화된 source_id가 같은 서로 다른 실제 파일이 data_dir 안에 있을 때 발생한다
    (개발 계획 §3.3 "source ID 유일성"). collisions는 충돌한 source_id -> 실제 경로
    목록 매핑이다."""

    def __init__(self, message: str, collisions: dict[str, list[Path]]) -> None:
        self.collisions = collisions
        super().__init__(message)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=_REPO_ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=_REPO_ROOT,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def _active_retrieval_config() -> dict:
    """config.py의 USE_HYBRID_SEARCH/USE_MMR/USE_RERANKER 중 활성화된 파이프라인에
    해당하는 값만 포함한다(개발 계획 §3.6 "활성 파이프라인에 해당하는 값")."""
    cfg: dict = {
        "use_hybrid_search": config.USE_HYBRID_SEARCH,
        "use_mmr": config.USE_MMR,
        "use_reranker": config.USE_RERANKER,
        "retrieval_k": config.RETRIEVAL_K,
    }
    if config.USE_HYBRID_SEARCH:
        cfg["bm25_top_k"] = config.BM25_TOP_K
        cfg["dense_top_k"] = config.DENSE_TOP_K
        cfg["rrf_top_k"] = config.RRF_TOP_K
    if config.USE_MMR:
        cfg["mmr_k"] = config.MMR_K
        cfg["mmr_lambda"] = config.MMR_LAMBDA
    if config.USE_RERANKER:
        cfg["reranker_top_k"] = config.RERANKER_TOP_K
    return cfg


def build_metadata(dataset_path: Path, command: list[str], extra: dict) -> dict:
    """M2-REQ-010이 요구하는 공통 메타데이터를 만든다. extra의 키(예: case_counts,
    집계 지표, 사례별 결과)는 반환 dict에 그대로 병합된다 — 어떤 필드가 들어가는지는
    호출하는 evaluator(Phase 4~7)가 결정한다."""
    dataset_bytes = dataset_path.read_bytes()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "python_version": platform.python_version(),
        "command": command,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "reranker_model": config.RERANKER_MODEL if config.USE_RERANKER else None,
        "ollama_model": config.OLLAMA_MODEL,
        "retrieval_config": _active_retrieval_config(),
    }
    metadata.update(extra)
    return metadata


def _render_markdown(payload: dict, name: str) -> str:
    """평면적인(스칼라) 필드만 bullet list로 보여주는 최소 렌더러. dict/list 값은
    구조가 evaluator마다 달라 여기서 일반화하지 않고 같은 이름의 .json을 참고하도록
    안내한다 — Phase 4~7에서 사례별 표/요약이 필요해지면 각 evaluator가 이 함수를
    확장하거나 별도 렌더러를 추가한다."""
    lines = [f"# {name}", ""]
    generated_at = payload.get("generated_at_utc")
    if generated_at:
        lines.append(f"생성 시각(UTC): {generated_at}")
        lines.append("")
    lines.append("## 메타데이터")
    lines.append("")
    skipped: list[str] = []
    for key in sorted(payload.keys()):
        value = payload[key]
        if isinstance(value, (dict, list)):
            skipped.append(key)
            continue
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    if skipped:
        lines.append(
            f"복잡한 구조를 가진 필드({', '.join(skipped)})는 같은 이름의 `.json` 리포트를 참고하세요."
        )
    else:
        lines.append("자세한 내용은 같은 이름의 `.json` 리포트를 참고하세요.")
    return "\n".join(lines) + "\n"


def escape_markdown_table_cell(value: object) -> str:
    """evaluator별 Markdown renderer(retrieval/routing/answers)가 실패 사례 등
    자유 텍스트(질문, 오류 메시지, 임의의 route 문자열)를 표 셀에 넣을 때 공통으로
    쓰는 헬퍼. 줄바꿈/연속 공백은 단일 공백으로 접고, `\\`와 `|`를 이스케이프해
    Markdown table delimiter가 값 내용 때문에 깨지지 않게 한다
    (M2_Phase_4_5_6_code_review_result.md 재리뷰 P2). `\\`를 먼저 이스케이프한
    뒤 `|`를 이스케이프해야, 방금 삽입한 이스케이프용 백슬래시가 다시
    이스케이프되지 않는다."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    return text


def write_report(
    payload: dict,
    output_dir: Path,
    name: str,
    render_markdown: Optional[Callable[[dict], str]] = None,
) -> tuple[Path, Path]:
    """output_dir/{name}_{utc_timestamp}.json, .md를 생성하고 두 경로를 반환한다.
    JSON은 sort_keys=True, ensure_ascii=False, indent=2로 직렬화한다.

    같은 name으로 짧은 간격 안에 반복 호출되거나(빠른 재실행) 여러 프로세스가
    동시에 같은 name으로 호출해도 기존 리포트를 덮어쓰지 않는다
    (M2_Phase3_code_review_result.md P1) — timestamp는 마이크로초까지 포함하고,
    실제 파일 생성은 배타적 생성("x" 모드)으로 수행해 이미 같은 경로가 존재하면
    FileExistsError를 받아 suffix를 늘려 재시도한다. JSON 경로를 먼저 예약해
    "이 stem은 이제 이 호출의 것"임을 확정한 뒤에만 두 파일을 쓴다 — 둘 중 하나만
    쓰고 죽는 상황을 줄이기 위해 md까지 성공해야 최종 반환한다.

    render_markdown이 주어지면 그 콜백(payload -> Markdown 문자열)으로 .md 내용을
    만든다. 기본 `_render_markdown()`은 스칼라 필드만 나열하고 dict/list 값은
    JSON을 참고하라는 안내만 남기므로, 사람이 핵심 지표(Recall/MRR/nDCG, PR/F1,
    confusion matrix, latency, 실패 사례 등)를 직접 읽어야 하는 evaluator는 반드시
    evaluator별 렌더러를 전달해야 한다(M2_Phase_4_5_6_code_review_result.md P1)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
    md_text = render_markdown(payload) if render_markdown is not None else _render_markdown(payload, name)

    base_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = 0
    while True:
        stem = f"{name}_{base_timestamp}" if suffix == 0 else f"{name}_{base_timestamp}-{suffix}"
        json_path = output_dir / f"{stem}.json"
        md_path = output_dir / f"{stem}.md"
        try:
            with json_path.open("x", encoding="utf-8") as f:
                f.write(json_text)
        except FileExistsError:
            suffix += 1
            continue

        try:
            with md_path.open("x", encoding="utf-8") as f:
                f.write(md_text)
        except FileExistsError:
            # json_path는 이미 이 호출이 예약했지만 같은 stem의 md_path가 다른
            # 호출과 충돌했다 — json_path를 되돌리고 다음 suffix로 통째로 재시도한다.
            json_path.unlink(missing_ok=True)
            suffix += 1
            continue

        return json_path, md_path


def build_corpus_manifest(data_dir: Path) -> dict:
    """data_dir 하위 모든 파일(하위 디렉터리 포함)의 정규화 source_id/size_bytes/sha256을
    entries 배열로 반환한다. entries는 source_id 오름차순으로 정렬한 뒤 canonical
    JSON(sort_keys=True, ensure_ascii=False, separators=(",", ":"))으로 직렬화해
    SHA-256을 계산한 manifest_sha256도 함께 반환한다.

    이 함수는 document_register.py의 로더별 glob(**/*.pdf, **/*.txt)과 무관하게
    존재하는 모든 파일을 대상으로 한다 — 목적이 "실행 간 data/가 같은 파일을 봤는지"를
    확인하는 재현성 지문이지, 실제로 색인된 파일 목록이 아니기 때문이다.

    정규화된 source_id가 같은 서로 다른 파일이 있으면 CorpusManifestError를 던진다
    (개발 계획 §3.3).

    Returns: {"entries": [{"source_id":.., "size_bytes":.., "sha256":..}, ...], "manifest_sha256": ..}
    """
    if not data_dir.is_dir():
        # Path.rglob()은 존재하지 않는 디렉터리에도 예외 없이 빈 이터레이터를 반환하므로
        # (data/가 없으면 corpus_manifest가 조용히 빈 배열이 되는 위험), 명시적으로 확인한다.
        raise FileNotFoundError(f"data_dir가 존재하지 않습니다: {data_dir}")

    seen: dict[str, Path] = {}
    collisions: dict[str, list[Path]] = {}
    raw_entries: list[tuple[str, Path]] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        source_id = normalize_source_id(path.name)
        if source_id in seen:
            collisions.setdefault(source_id, [seen[source_id]])
            collisions[source_id].append(path)
        else:
            seen[source_id] = path
        raw_entries.append((source_id, path))

    if collisions:
        raise CorpusManifestError(
            f"정규화된 source_id가 충돌하는 파일이 있습니다: {sorted(collisions)}",
            collisions,
        )

    entries = [
        {
            "source_id": source_id,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for source_id, path in raw_entries
    ]
    entries.sort(key=lambda e: e["source_id"])

    canonical = json.dumps(entries, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    manifest_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"entries": entries, "manifest_sha256": manifest_sha256}


def build_vectorstore_fingerprint(vectorstore_path: Path) -> dict:
    """index.faiss/index.pkl 두 파일의 SHA-256만 계산한다. FAISS를 역직렬화하지
    않으므로 임베딩 모델 로드 비용이 없다. 파일이 없으면 FileNotFoundError를 그대로
    던진다."""
    faiss_path = vectorstore_path / "index.faiss"
    pkl_path = vectorstore_path / "index.pkl"
    return {
        "index_faiss_sha256": hashlib.sha256(faiss_path.read_bytes()).hexdigest(),
        "index_pkl_sha256": hashlib.sha256(pkl_path.read_bytes()).hexdigest(),
    }


def build_reproducibility_metadata(data_dir: Path, vectorstore_path: Path) -> dict:
    """corpus/vectorstore를 실제로 사용하는 평가(Retrieval, Answer, 통합 baseline)가
    공통으로 호출한다. data_dir/vectorstore_path가 없으면 FileNotFoundError를
    그대로 던진다 — 호출부(각 evaluator의 main())가 기존 오류 처리 정책(사람이 읽을
    오류 + exit(2))으로 잡는다."""
    manifest = build_corpus_manifest(data_dir)
    fingerprint = build_vectorstore_fingerprint(vectorstore_path)
    return {
        "corpus_manifest": manifest["entries"],
        "corpus_manifest_sha256": manifest["manifest_sha256"],
        "vectorstore_fingerprint": fingerprint,
        "reproducibility_note": None,
    }


_CANDIDATE_ID_RE = re.compile(r"^m3-(?:final|p[0-6][a-z]?(?:-[a-z0-9]+)+)$")


def build_candidate_metadata(
    candidate_id: str | None,
    label: str | None = None,
    baseline_ref: str = "evaluation/baselines/m2_initial.json",
    baseline_dataset_sha256: str = "61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a",
    notes: str | None = None,
) -> dict:
    """M3-REQ-001 candidate block (Design.md §3.1, §3.3). `candidate_id`가
    주어지면 §3.1의 정규식을 만족해야 한다(그렇지 않으면 ValueError로 즉시
    실패 — 잘못된 candidate ID로 리포트를 만들지 않기 위함). `candidate_id`가
    `None`이면(candidate 개념이 없는 실행) 모든 필드가 `null`인 block을
    반환한다."""
    if candidate_id is not None and not _CANDIDATE_ID_RE.match(candidate_id):
        raise ValueError(f"candidate_id가 §3.1 정규식과 맞지 않습니다: {candidate_id!r}")
    phase = None
    if candidate_id is not None and candidate_id != "m3-final":
        # "m3-p<phase>..." -> phase 정수 추출
        digits = candidate_id[len("m3-p"):].split("-", 1)[0]
        phase = int(digits[0]) if digits and digits[0].isdigit() else None
    elif candidate_id == "m3-final":
        phase = 6
    return {
        "candidate_id": candidate_id,
        "label": label,
        "phase": phase,
        "baseline_ref": baseline_ref,
        "baseline_dataset_sha256": baseline_dataset_sha256,
        "notes": notes,
    }


def build_warmup_metadata(
    *,
    requested_cases: int,
    executed_cases: int,
    succeeded_cases: int,
    failed_cases: int,
    case_ids: list[str],
    same_process: bool = True,
    engine_object_id_matches: bool = True,
) -> dict:
    """M3-NFR-002 warm-up metadata block (Design.md §3.3, §4.4).
    `performed = executed_cases > 0 and failed_cases == 0`."""
    performed = executed_cases > 0 and failed_cases == 0
    return {
        "requested_cases": requested_cases,
        "executed_cases": executed_cases,
        "succeeded_cases": succeeded_cases,
        "failed_cases": failed_cases,
        "case_ids": case_ids,
        "same_process": same_process,
        "engine_object_id_matches": engine_object_id_matches,
        "discarded_from_metrics": True,
        "performed": performed,
    }


def build_not_applicable_reproducibility_metadata(reason: str) -> dict:
    """Routing처럼 corpus/vectorstore를 쓰지 않는 evaluator가 호출한다. 모든
    리포트가 evaluator 종류와 무관하게 같은 key 집합을 갖도록, corpus/vectorstore
    관련 필드는 None으로 채우고 reproducibility_note에 이유를 남긴다."""
    return {
        "corpus_manifest": None,
        "corpus_manifest_sha256": None,
        "vectorstore_fingerprint": None,
        "reproducibility_note": reason,
    }
