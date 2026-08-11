"""M4.1 typed Settings single source (Design.md §4).

`FIELD_SPECS` is the single declarative source for every product setting —
41 fields, no hand-duplicated field list anywhere else. `Settings` is built
from `FIELD_SPECS` via `pydantic.create_model` (frozen, extra="forbid").
`Settings.from_sources()`/`Settings.from_env()` are the only construction
paths product code should use; direct `Settings(**kwargs)` bypasses the
base<-env<-CLI merge algorithm and is reserved for tests that supply every
required field explicitly.

`config.py` projects a legacy facade over this module (REQ-002.3) and must
not duplicate any default/validation logic defined here.
"""

from __future__ import annotations

import os
import math
import typing
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Mapping

import pydantic
from pydantic import ConfigDict, create_model, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

# Package root, computed independently of Settings/env (mirrors
# config.py's original PROJECT_ROOT calculation) so path-field parsers can
# close over a fixed value without needing other field values at parse time.
_PACKAGE_ROOT: Path = Path(__file__).resolve().parents[2]

ENV_PREFIX = "SIMPLE_QNA_RAG_"


class SettingsError(Exception):
    """Raised for any Settings load failure. `exit_code` is 2 per REQ-002.4."""

    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


# ---------------------------------------------------------------------------
# FieldSpec / ModelValidatorSpec (Design §4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    name: str
    annotation: Any
    default: object = PydanticUndefined
    default_factory: Callable[[], object] | None = None
    env_alias: str | None = None
    parser: Callable[[str], object] | None = None
    derive: Callable[[Mapping[str, object]], object] | None = None
    derived_from: tuple[str, ...] = ()
    validators: tuple[Callable[[object], object], ...] = ()
    consumers: tuple[str, ...] = ()
    facade_type: type | None = None
    facade_adapter: Callable[[object], object] | None = None


@dataclass(frozen=True)
class ModelValidatorSpec:
    callable: Callable[[Any], Any]
    constraint: str
    related_fields: tuple[str, ...]
    default_rendering: Callable[[Mapping[str, object]], str]


# ---------------------------------------------------------------------------
# Shared parsers / validators
# ---------------------------------------------------------------------------

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _parse_bool(raw: str) -> bool:
    value = raw.strip().casefold()
    if value in _BOOL_TRUE:
        return True
    if value in _BOOL_FALSE:
        return False
    raise ValueError(f"invalid bool value: {raw!r}")


def _parse_runtime_path(raw: str, project_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _make_path_parser() -> Callable[[str], Path]:
    def _parse(raw: str) -> Path:
        return _parse_runtime_path(raw, _PACKAGE_ROOT)

    return _parse


def _check_positive(v: object) -> object:
    if v <= 0:  # type: ignore[operator]
        raise ValueError("must be > 0")
    return v


def _check_unit_interval(v: object) -> object:
    if not (0 <= v <= 1):  # type: ignore[operator]
        raise ValueError("must be within [0, 1]")
    return v


def _make_range_validator(
    lo: float, hi: float, *, lo_inclusive: bool = True,
    hi_inclusive: bool = True, finite: bool = False,
) -> Callable[[object], object]:
    def _check(value: object) -> object:
        number = float(value)  # validators are only attached to numeric fields
        if finite and not math.isfinite(number):
            raise ValueError("must be finite")
        lower_ok = number >= lo if lo_inclusive else number > lo
        upper_ok = number <= hi if hi_inclusive else number < hi
        if not lower_ok or not upper_ok:
            raise ValueError("value outside allowed range")
        return value

    return _check


_VALIDATOR_LABELS: dict[Callable[[object], object], str] = {
    _check_positive: ">0",
    _check_unit_interval: "0<=x<=1",
}


# ---------------------------------------------------------------------------
# Cross-field model validators (Design §4.1, MODEL_VALIDATORS)
# ---------------------------------------------------------------------------


def _check_chunk_overlap_lt_chunk_size(self: Any) -> Any:
    if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be < CHUNK_SIZE")
    return self


def _check_mmr_k_le_fetch_k(self: Any) -> Any:
    if self.MMR_K > self.MMR_FETCH_K:
        raise ValueError("MMR_K must be <= MMR_FETCH_K")
    return self


def _check_reranker_le_rrf(self: Any) -> Any:
    if self.RERANKER_TOP_K > self.RRF_TOP_K:
        raise ValueError("RERANKER_TOP_K must be <= RRF_TOP_K")
    return self


def _check_queue_timeout_lt_execution_timeout(self: Any) -> Any:
    if self.QUERY_QUEUE_TIMEOUT_SECONDS >= self.QUERY_EXECUTION_TIMEOUT_SECONDS:
        raise ValueError("QUERY_QUEUE_TIMEOUT_SECONDS must be < QUERY_EXECUTION_TIMEOUT_SECONDS")
    return self


MODEL_VALIDATORS: tuple[ModelValidatorSpec, ...] = (
    ModelValidatorSpec(
        callable=_check_chunk_overlap_lt_chunk_size,
        constraint="CHUNK_OVERLAP < CHUNK_SIZE",
        related_fields=("CHUNK_OVERLAP", "CHUNK_SIZE"),
        default_rendering=lambda d: f"{d['CHUNK_OVERLAP']} < {d['CHUNK_SIZE']}",
    ),
    ModelValidatorSpec(
        callable=_check_mmr_k_le_fetch_k,
        constraint="MMR_K <= MMR_FETCH_K",
        related_fields=("MMR_K", "MMR_FETCH_K"),
        default_rendering=lambda d: f"{d['MMR_K']} <= {d['MMR_FETCH_K']}",
    ),
    ModelValidatorSpec(
        callable=_check_reranker_le_rrf,
        constraint="RERANKER_TOP_K <= RRF_TOP_K",
        related_fields=("RERANKER_TOP_K", "RRF_TOP_K"),
        default_rendering=lambda d: f"{d['RERANKER_TOP_K']} <= {d['RRF_TOP_K']}",
    ),
    ModelValidatorSpec(
        callable=_check_queue_timeout_lt_execution_timeout,
        constraint="QUERY_QUEUE_TIMEOUT_SECONDS < QUERY_EXECUTION_TIMEOUT_SECONDS",
        related_fields=("QUERY_QUEUE_TIMEOUT_SECONDS", "QUERY_EXECUTION_TIMEOUT_SECONDS"),
        default_rendering=lambda d: (
            f"{d['QUERY_QUEUE_TIMEOUT_SECONDS']} < {d['QUERY_EXECUTION_TIMEOUT_SECONDS']}"
        ),
    ),
)


# ---------------------------------------------------------------------------
# Prompt template default (verbatim from the pre-M4.1 config.py literal)
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE_DEFAULT = """당신은 문서 기반 RAG 시스템의 답변을 생성하는 AI 어시스턴트입니다.

 [역할 및 규칙]

 1. 먼저 질문과 제공된 문맥(Context)을 분석합니다.

 2. 주어진 문맥(Context)의 정보만을 사용하여 답변합니다.
  - 답변은 명확하고 구체적이어야 합니다.
  - 답변의 양식은 표로 정리하기 적합한지 여부에 따라 아래 규칙을 따릅니다.

 3. 답변 내용을 표로 정리하기 적합한지 판단합니다.
  - 표로 정리하기 적합한지 판단하근 기준은, 여러 개의 항목을 공통된 기준(열/컬럼)으로 비교 및 정리하기에 적합한지 여부입니다.
  - 답변 본문이 한 줄 이하 이거나 혹은 데이터 행이 1개 이하일 때에는 표로 정리하지 않습니다.
  - 이 판단 과정은 답변에 드러내지 마세요. (생각을 말로 설명하지 마세요.)

 4. 표로 정리하기에 적합한 답변 내용은 다음 형식의 Markdown 표로 본문을 작성합니다.
  - 첫 줄: 한 문장 핵심 요약. 만약 사용자가 '예', '아니오'로 답변할 수 있는 질문을 했다면 '예' 또는 '아니오' 형태의 답변을 제일 서두에 표현할 것.
  - 그 아래: 표 (헤더 1행 + 데이터 행 2개 이상)
  - 예시 해더: | 항목 | 설명 | 참고 | 와 같은 형태
  - 불필요한 서론 없이 간결하게 작성
  - 표 내용 후 맨 아래에 결론으로 내용을 요약

 5. 표로 정리하기에 부적합한 답변 내용은 억지로 표를 만들지 말고, 자연스러운 문단 또는 bullet/번호 리스트 형식으로 답변합니다.
  - 추상적인 개념 설명, 스토리형 질문은 표로 정리하지 말고 자연스러운 평문으로 답합니다.
  - '표로 답변하기 어렵다'와 같은 메타 설명은 하지 않습니다.

 6. 항상 제공된 컨텍스트를 우선적으로 사용하고, 근거 없는 내용을 상상하여 추가하지 않습니다.

 7. 문맥에 답이 없으면 '제공된 문서에서 관련 정보를 찾을 수 없습니다' 라고 답변합니다.

 8. 가능한 경우 출처를 인용합니다.

 문맥 (Context):
 {context}

 질문 (Question): {question}

 답변 (Answer):"""


# ---------------------------------------------------------------------------
# FIELD_SPECS — the 41-field single source (Field_Spec_Inventory.md)
# ---------------------------------------------------------------------------

_path_parser = _make_path_parser()

FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="PROJECT_ROOT",
        annotation=Path,
        default_factory=lambda: _PACKAGE_ROOT,
        consumers=("config.py 내부 파생 필드", "test_cli_entrypoints.py"),
    ),
    FieldSpec(
        name="STATIC_DIR",
        annotation=Path,
        derive=lambda v: v["PROJECT_ROOT"] / "web" / "static",
        derived_from=("PROJECT_ROOT",),
        consumers=("web/server.py",),
        facade_type=str,
        facade_adapter=str,
    ),
    FieldSpec(
        name="TEMPLATES_DIR",
        annotation=Path,
        derive=lambda v: v["PROJECT_ROOT"] / "web" / "templates",
        derived_from=("PROJECT_ROOT",),
        consumers=("web/server.py",),
        facade_type=str,
        facade_adapter=str,
    ),
    FieldSpec(
        name="VECTORSTORE_PATH",
        annotation=Path,
        default_factory=lambda: _PACKAGE_ROOT / "runtime" / "vectorstore",
        env_alias="SIMPLE_QNA_RAG_VECTORSTORE_DIR",
        parser=_path_parser,
        consumers=("cli/index_documents.py", "rag_engine.py"),
        facade_type=str,
        facade_adapter=str,
    ),
    FieldSpec(
        name="DATA_DIR",
        annotation=Path,
        default_factory=lambda: _PACKAGE_ROOT / "runtime" / "documents",
        env_alias="SIMPLE_QNA_RAG_DOCUMENTS_DIR",
        parser=_path_parser,
        consumers=("agent.py", "cli/index_documents.py"),
        facade_type=str,
        facade_adapter=str,
    ),
    FieldSpec(
        name="INTENT_MODEL_PATH",
        annotation=Path,
        default_factory=lambda: _PACKAGE_ROOT / "models" / "intent_classifier",
        env_alias="SIMPLE_QNA_RAG_MODEL_DIR",
        parser=_path_parser,
        consumers=("intent_classifier.py",),
        facade_type=str,
        facade_adapter=str,
    ),
    FieldSpec(
        name="COLLECTION_NAME",
        annotation=str,
        default="document_collection",
        env_alias="SIMPLE_QNA_RAG_COLLECTION_NAME",
        parser=str,
        consumers=(),
    ),
    FieldSpec(
        name="EMBEDDING_MODEL_NAME",
        annotation=str,
        default="BAAI/bge-m3",
        env_alias="SIMPLE_QNA_RAG_EMBEDDING_MODEL_NAME",
        parser=str,
        consumers=("cli/index_documents.py", "rag_engine.py"),
    ),
    FieldSpec(
        name="CHUNK_SIZE",
        annotation=int,
        default=1000,
        env_alias="SIMPLE_QNA_RAG_CHUNK_SIZE",
        parser=int,
        validators=(_check_positive,),
        consumers=("cli/index_documents.py",),
    ),
    FieldSpec(
        name="CHUNK_OVERLAP",
        annotation=int,
        default=200,
        env_alias="SIMPLE_QNA_RAG_CHUNK_OVERLAP",
        parser=int,
        consumers=("cli/index_documents.py",),
    ),
    FieldSpec(
        name="OLLAMA_BASE_URL",
        annotation=str,
        default="http://localhost:11434",
        env_alias="SIMPLE_QNA_RAG_OLLAMA_BASE_URL",
        parser=str,
        consumers=("agent.py", "rag_engine.py"),
    ),
    FieldSpec(
        name="OLLAMA_MODEL",
        annotation=str,
        default="gpt-oss:20b",
        env_alias="SIMPLE_QNA_RAG_OLLAMA_MODEL",
        parser=str,
        consumers=("agent.py", "rag_engine.py"),
    ),
    FieldSpec(
        name="RETRIEVAL_K",
        annotation=int,
        default=4,
        env_alias="SIMPLE_QNA_RAG_RETRIEVAL_K",
        parser=int,
        validators=(_check_positive,),
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="USE_MMR",
        annotation=bool,
        default=True,
        env_alias="SIMPLE_QNA_RAG_USE_MMR",
        parser=_parse_bool,
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="MMR_FETCH_K",
        annotation=int,
        default=100,
        env_alias="SIMPLE_QNA_RAG_MMR_FETCH_K",
        parser=int,
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="MMR_K",
        annotation=int,
        default=20,
        env_alias="SIMPLE_QNA_RAG_MMR_K",
        parser=int,
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="MMR_LAMBDA",
        annotation=float,
        default=0.5,
        env_alias="SIMPLE_QNA_RAG_MMR_LAMBDA",
        parser=float,
        validators=(_check_unit_interval,),
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="NORMALIZE_EMBEDDINGS",
        annotation=bool,
        default=True,
        env_alias="SIMPLE_QNA_RAG_NORMALIZE_EMBEDDINGS",
        parser=_parse_bool,
        consumers=("cli/index_documents.py", "rag_engine.py"),
    ),
    FieldSpec(
        name="USE_HYBRID_SEARCH",
        annotation=bool,
        default=True,
        env_alias="SIMPLE_QNA_RAG_USE_HYBRID_SEARCH",
        parser=_parse_bool,
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="BM25_TOP_K",
        annotation=int,
        default=50,
        env_alias="SIMPLE_QNA_RAG_BM25_TOP_K",
        parser=int,
        validators=(_check_positive,),
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="DENSE_TOP_K",
        annotation=int,
        default=50,
        env_alias="SIMPLE_QNA_RAG_DENSE_TOP_K",
        parser=int,
        validators=(_check_positive,),
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="RRF_TOP_K",
        annotation=int,
        default=50,
        env_alias="SIMPLE_QNA_RAG_RRF_TOP_K",
        parser=int,
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="RRF_CONSTANT",
        annotation=int,
        default=60,
        env_alias="SIMPLE_QNA_RAG_RRF_CONSTANT",
        parser=int,
        validators=(_check_positive,),
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="USE_RERANKER",
        annotation=bool,
        default=True,
        env_alias="SIMPLE_QNA_RAG_USE_RERANKER",
        parser=_parse_bool,
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="RERANKER_MODEL",
        annotation=str,
        default="BAAI/bge-reranker-v2-m3",
        env_alias="SIMPLE_QNA_RAG_RERANKER_MODEL",
        parser=str,
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="RERANKER_TOP_K",
        annotation=int,
        default=10,
        env_alias="SIMPLE_QNA_RAG_RERANKER_TOP_K",
        parser=int,
        validators=(_check_positive,),
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="USE_WEB_SEARCH",
        annotation=bool,
        default=True,
        env_alias="SIMPLE_QNA_RAG_USE_WEB_SEARCH",
        parser=_parse_bool,
        consumers=("agent.py", "query_router.py"),
    ),
    FieldSpec(
        name="WEB_SEARCH_MAX_RESULTS",
        annotation=int,
        default=3,
        env_alias="SIMPLE_QNA_RAG_WEB_SEARCH_MAX_RESULTS",
        parser=int,
        validators=(_check_positive,),
        consumers=("web_search.py",),
    ),
    FieldSpec(
        name="WEB_SEARCH_TIMEOUT",
        annotation=int,
        default=10,
        env_alias="SIMPLE_QNA_RAG_WEB_SEARCH_TIMEOUT",
        parser=int,
        validators=(_check_positive,),
        consumers=("web_search.py",),
    ),
    FieldSpec(
        name="WEB_SEARCH_REGION",
        annotation=str,
        default="kr-kr",
        env_alias="SIMPLE_QNA_RAG_WEB_SEARCH_REGION",
        parser=str,
        consumers=("web_search.py",),
    ),
    FieldSpec(
        name="PROMPT_TEMPLATE",
        annotation=str,
        default=_PROMPT_TEMPLATE_DEFAULT,
        env_alias="SIMPLE_QNA_RAG_PROMPT_TEMPLATE",
        parser=str,
        consumers=(),
    ),
    FieldSpec(
        name="MMR_VECTOR_SOURCE",
        annotation=Literal["embed", "stored"],
        default="stored",
        env_alias="SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE",
        parser=str,
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="MMR_VECTOR_VALIDATION_SAMPLE",
        annotation=int,
        default=3,
        env_alias="SIMPLE_QNA_RAG_MMR_VECTOR_VALIDATION_SAMPLE",
        parser=int,
        validators=(_check_positive,),
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="MMR_VECTOR_COSINE_FLOOR",
        annotation=float,
        default=0.99,
        env_alias="SIMPLE_QNA_RAG_MMR_VECTOR_COSINE_FLOOR",
        parser=float,
        validators=(_check_unit_interval,),
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="MMR_EMBED_CACHE_MAX_ITEMS",
        annotation=int,
        default=2048,
        env_alias="SIMPLE_QNA_RAG_MMR_EMBED_CACHE_MAX_ITEMS",
        parser=int,
        validators=(_check_positive,),
        consumers=(),
    ),
    FieldSpec(
        name="ROUTING_SIGNAL_OVERRIDE",
        annotation=bool,
        default=True,
        env_alias="SIMPLE_QNA_RAG_ROUTING_SIGNAL_OVERRIDE",
        parser=_parse_bool,
        consumers=("agent.py",),
    ),
    FieldSpec(
        name="ROUTING_CORPUS_TOPIC_HINT",
        annotation=bool,
        default=False,
        env_alias="SIMPLE_QNA_RAG_ROUTING_CORPUS_TOPIC_HINT",
        parser=_parse_bool,
        consumers=("agent.py",),
    ),
    FieldSpec(
        name="ROUTING_CORPUS_TOPIC_HINT_MAX_ITEMS",
        annotation=int,
        default=25,
        env_alias="SIMPLE_QNA_RAG_ROUTING_CORPUS_TOPIC_HINT_MAX_ITEMS",
        parser=int,
        validators=(_check_positive,),
        consumers=("agent.py",),
    ),
    FieldSpec(
        name="ANSWER_TEMPLATE_MODE",
        annotation=Literal["intent", "default"],
        default="default",
        env_alias="SIMPLE_QNA_RAG_ANSWER_TEMPLATE_MODE",
        parser=str,
        consumers=("rag_engine.py",),
    ),
    FieldSpec(
        name="INTENT_CONFIDENCE_FLOOR",
        annotation=float,
        default=0.0,
        env_alias="SIMPLE_QNA_RAG_INTENT_CONFIDENCE_FLOOR",
        parser=float,
        validators=(_check_unit_interval,),
        consumers=(),
    ),
    FieldSpec(
        name="BM25_TOKENIZER",
        annotation=Literal["whitespace", "char2gram", "bge-subword"],
        default="whitespace",
        env_alias="SIMPLE_QNA_RAG_BM25_TOKENIZER",
        parser=str,
        consumers=("tests/unit/test_config.py",),
    ),
    FieldSpec(name="QUERY_CONCURRENCY_LIMIT", annotation=int, default=1,
              env_alias="SIMPLE_QNA_RAG_QUERY_CONCURRENCY_LIMIT", parser=int,
              validators=(_make_range_validator(1, 2),), consumers=("web/concurrency.py",)),
    FieldSpec(name="QUERY_QUEUE_LIMIT", annotation=int, default=4,
              env_alias="SIMPLE_QNA_RAG_QUERY_QUEUE_LIMIT", parser=int,
              validators=(_make_range_validator(0, 64),), consumers=("web/concurrency.py",)),
    FieldSpec(name="QUERY_QUEUE_TIMEOUT_SECONDS", annotation=float, default=5.0,
              env_alias="SIMPLE_QNA_RAG_QUERY_QUEUE_TIMEOUT_SECONDS", parser=float,
              validators=(_make_range_validator(0, 30, lo_inclusive=False, finite=True),),
              consumers=("web/concurrency.py",)),
    FieldSpec(name="QUERY_EXECUTION_TIMEOUT_SECONDS", annotation=float, default=90.0,
              env_alias="SIMPLE_QNA_RAG_QUERY_EXECUTION_TIMEOUT_SECONDS", parser=float,
              validators=(_make_range_validator(1, 600, finite=True),),
              consumers=("web/concurrency.py",)),
    FieldSpec(name="SHUTDOWN_GRACE_SECONDS", annotation=float, default=30.0,
              env_alias="SIMPLE_QNA_RAG_SHUTDOWN_GRACE_SECONDS", parser=float,
              validators=(_make_range_validator(0, 120, finite=True),), consumers=("web/server.py",)),
    FieldSpec(name="MAX_REQUEST_BODY_BYTES", annotation=int, default=16384,
              env_alias="SIMPLE_QNA_RAG_MAX_REQUEST_BODY_BYTES", parser=int,
              validators=(_make_range_validator(256, 1_048_576),), consumers=("web/body_limit.py",)),
    FieldSpec(name="MAX_QUESTION_CHARS", annotation=int, default=4000,
              env_alias="SIMPLE_QNA_RAG_MAX_QUESTION_CHARS", parser=int,
              validators=(_make_range_validator(1, 32_000),), consumers=("web/server.py",)),
    FieldSpec(name="UPSTREAM_CONNECT_TIMEOUT_SECONDS", annotation=float, default=5.0,
              env_alias="SIMPLE_QNA_RAG_UPSTREAM_CONNECT_TIMEOUT_SECONDS", parser=float,
              validators=(_make_range_validator(0, 30, lo_inclusive=False, finite=True),),
              consumers=("observability/deadline.py",)),
)

assert len(FIELD_SPECS) == 49, f"FIELD_SPECS must have 49 fields, got {len(FIELD_SPECS)}"


# ---------------------------------------------------------------------------
# Topological sort + Settings model construction (Design §4.1/§4.3)
# ---------------------------------------------------------------------------


def _topo_sorted(specs: tuple[FieldSpec, ...]) -> tuple[FieldSpec, ...]:
    by_name = {s.name: s for s in specs}
    state: dict[str, int] = {}
    order: list[FieldSpec] = []

    def visit(spec: FieldSpec) -> None:
        current = state.get(spec.name)
        if current == 1:
            return
        if current == 0:
            raise ValueError(f"circular derived_from dependency at {spec.name!r}")
        state[spec.name] = 0
        for dep in spec.derived_from:
            visit(by_name[dep])
        state[spec.name] = 1
        order.append(spec)

    for spec in specs:
        visit(spec)
    return tuple(order)


def _field_definitions(specs: tuple[FieldSpec, ...]) -> dict[str, tuple[Any, FieldInfo]]:
    result: dict[str, tuple[Any, FieldInfo]] = {}
    for s in specs:
        if s.default_factory is not None:
            info = FieldInfo(default_factory=s.default_factory)
        else:
            info = FieldInfo(default=s.default)
        result[s.name] = (s.annotation, info)
    return result


def _validator_namespace(specs: tuple[FieldSpec, ...]) -> dict[str, Any]:
    ns: dict[str, Any] = {}
    for s in specs:
        for i, fn in enumerate(s.validators):
            ns[f"_check_{s.name}_{i}"] = field_validator(s.name)(fn)
    for i, mv in enumerate(MODEL_VALIDATORS):
        ns[f"_check_model_{i}"] = model_validator(mode="after")(mv.callable)
    return ns


Settings = create_model(
    "Settings",
    __config__=ConfigDict(frozen=True, extra="forbid"),
    __validators__=_validator_namespace(FIELD_SPECS),
    **_field_definitions(FIELD_SPECS),
)


def _from_sources(
    cls: type,
    base_environ: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, str] | None = None,
) -> Any:
    base = dict(os.environ if base_environ is None else base_environ)
    merged = {**base, **(cli_overrides or {})}
    values: dict[str, object] = {}
    for spec in _topo_sorted(FIELD_SPECS):
        if spec.env_alias is None:
            values[spec.name] = spec.derive(values) if spec.derive else spec.default_factory()
            continue
        raw = merged.get(spec.env_alias)
        if raw is None or raw == "":
            values[spec.name] = (
                spec.default_factory() if spec.default_factory is not None else spec.default
            )
            continue
        try:
            values[spec.name] = spec.parser(raw)
        except Exception as exc:  # noqa: BLE001 - normalized into SettingsError below
            raise SettingsError(f"{spec.env_alias}: invalid value {raw!r}", exit_code=2) from exc

    known = {s.env_alias for s in FIELD_SPECS if s.env_alias}
    unknown = sorted(k for k in merged if k.startswith(ENV_PREFIX) and k not in known)
    if unknown:
        raise SettingsError(f"unknown keys: {unknown}", exit_code=2)

    try:
        return cls.model_validate(values)
    except pydantic.ValidationError as exc:
        raise SettingsError(str(exc), exit_code=2) from exc


def _from_env(cls: type, environ: Mapping[str, str] | None = None) -> Any:
    return cls.from_sources(base_environ=environ)


Settings.from_sources = classmethod(_from_sources)
Settings.from_env = classmethod(_from_env)


# ---------------------------------------------------------------------------
# Cache seam (Design §4.3)
# ---------------------------------------------------------------------------

_settings_cache: Any = None


def get_settings() -> Any:
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = Settings.from_env()
    return _settings_cache


def reset_settings_cache() -> None:
    global _settings_cache
    _settings_cache = None


def set_settings_for_process(settings: Any) -> None:
    """Populate the process-wide cache with an already-validated Settings
    instance (used by CLI entry points after `from_sources(cli_overrides=...)`
    succeeds, so `config.py`'s later lazy `get_settings()` returns the same
    override-applied instance instead of recomputing from `os.environ` alone)."""
    global _settings_cache
    _settings_cache = settings


# ---------------------------------------------------------------------------
# facade value projection (Design §4.4)
# ---------------------------------------------------------------------------


def facade_value(settings: Any, spec: FieldSpec) -> object:
    raw = getattr(settings, spec.name)
    return spec.facade_adapter(raw) if spec.facade_adapter is not None else raw


# ---------------------------------------------------------------------------
# check-config disclosure policy (Design §5.2/§5.3, REQ-002.5) — declared per
# field from `FieldSpec.annotation` rather than hand-maintained, so a future
# field cannot silently start printing raw values. Any field whose annotation
# is not a closed/bounded domain (bool, int, float, or a `Literal[...]` enum)
# is treated as free-form user content and MUST NOT be printed verbatim —
# this is the class of field that leaked credential URLs and prompt text in
# CR-I1-MAJ-01. `Path` fields keep the pre-existing exists-only redaction.
# ---------------------------------------------------------------------------

_BOUNDED_VALUE_ANNOTATIONS = (bool, int, float)


def _is_literal_annotation(annotation: Any) -> bool:
    return typing.get_origin(annotation) is Literal


def field_disclosure(spec: FieldSpec) -> str:
    """Return the disclosure class for a field: "path" | "value" | "redacted"."""
    if spec.annotation is Path:
        return "path"
    if spec.annotation in _BOUNDED_VALUE_ANNOTATIONS or _is_literal_annotation(spec.annotation):
        return "value"
    return "redacted"


def check_config_payload(settings: Any) -> dict[str, object]:
    """Build the `--check-config` JSON payload (REQ-002.5): closed-domain
    fields (bool/int/float/Literal enum) are safe to print verbatim because
    pydantic already constrains them to a small declared set of values with
    no user-controlled free text. Every other field — including all `Path`
    fields and every plain `str` field (URLs, prompts, model names, tokens)
    — is redacted to non-reversible metadata only.
    """
    payload: dict[str, object] = {}
    for spec in FIELD_SPECS:
        value = getattr(settings, spec.name)
        disclosure = field_disclosure(spec)
        if disclosure == "value":
            payload[spec.name] = value
        elif disclosure == "path":
            payload[spec.name] = {"redacted": True, "kind": "path", "exists": value.exists()}
        else:
            payload[spec.name] = {
                "redacted": True,
                "kind": "string",
                "length": len(value) if isinstance(value, str) else None,
            }
    return payload


# ---------------------------------------------------------------------------
# Declarative table generators (Design §4.1.1) — used by
# scripts/generate_field_spec.py to produce docs/generated/settings_field_spec.md
# ---------------------------------------------------------------------------


def _default_field_values(specs: tuple[FieldSpec, ...]) -> dict[str, object]:
    values: dict[str, object] = {}
    for spec in _topo_sorted(specs):
        values[spec.name] = (
            spec.derive(values)
            if spec.derive
            else spec.default_factory()
            if spec.default_factory is not None
            else spec.default
        )
    return values


def _default_passes(specs: tuple[FieldSpec, ...]) -> dict[str, bool]:
    """Whether each field's default value passes its own validators and every
    related MODEL_VALIDATORS constraint, computed against the full default set."""
    defaults = _default_field_values(specs)
    namespace = SimpleNamespace(**defaults)
    passes: dict[str, bool] = {s.name: True for s in specs}

    for spec in specs:
        for validator in spec.validators:
            try:
                validator(defaults[spec.name])
            except ValueError:
                passes[spec.name] = False

    for mv in MODEL_VALIDATORS:
        try:
            mv.callable(namespace)
        except ValueError:
            for name in mv.related_fields:
                passes[name] = False

    return passes


def _to_markdown_table(header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        cells = [str(c) if c != () and c is not None else "" for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_derive_column(spec: FieldSpec) -> str:
    if spec.derive is None:
        return "—"
    return f"derived_from={spec.derived_from}"


def _render_validators_column(spec: FieldSpec) -> str:
    labels = [_VALIDATOR_LABELS.get(fn, fn.__name__) for fn in spec.validators]
    related_models = [mv.constraint for mv in MODEL_VALIDATORS if spec.name in mv.related_fields]
    labels.extend(f"model: {c}" for c in related_models)
    return ", ".join(labels) if labels else "없음"


def render_field_specs_table(specs: tuple[FieldSpec, ...]) -> str:
    passes = _default_passes(specs)
    rows = []
    for i, spec in enumerate(specs, start=1):
        default_display = (
            "—" if spec.default is PydanticUndefined and spec.default_factory is None else
            "(factory)" if spec.default_factory is not None else
            repr(spec.default)
        )
        rows.append(
            (
                i,
                f"`{spec.name}`",
                getattr(spec.annotation, "__name__", str(spec.annotation)),
                default_display,
                "있음" if spec.default_factory is not None else "—",
                spec.env_alias or "None",
                spec.parser.__name__ if getattr(spec.parser, "__name__", None) else ("—" if spec.parser is None else "parser"),
                _render_derive_column(spec),
                _render_validators_column(spec),
                ", ".join(spec.consumers) if spec.consumers else "—",
                "PASS" if passes[spec.name] else "FAIL",
                spec.facade_type.__name__ if spec.facade_type is not None else "None",
                spec.facade_adapter.__name__ if spec.facade_adapter is not None else "None",
            )
        )
    header = (
        "#", "name", "annotation", "default", "default_factory", "env_alias", "parser",
        "derive/derived_from", "validators", "consumers", "default_pass", "facade_type",
        "facade_adapter",
    )
    return _to_markdown_table(header, rows)


def render_model_validators_table(
    validators: tuple[ModelValidatorSpec, ...], specs: tuple[FieldSpec, ...]
) -> str:
    index_of = {s.name: i + 1 for i, s in enumerate(specs)}
    defaults = _default_field_values(specs)
    namespace = SimpleNamespace(**defaults)
    rows = []
    for n, mv in enumerate(validators, start=1):
        related = ", ".join(f"#{index_of[name]}" for name in mv.related_fields)
        default_value = mv.default_rendering(defaults)
        try:
            mv.callable(namespace)
            verdict = "PASS"
        except ValueError:
            verdict = "FAIL"
        rows.append((n, mv.constraint, related, default_value, verdict))
    return _to_markdown_table(("#", "제약", "관련 필드", "default 값", "판정"), rows)
