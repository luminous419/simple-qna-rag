"""Answer evaluator v1/v2 순수 규칙 (M3-REQ-006, Design.md §5).

이 모듈은 모델·파일 I/O를 쓰지 않는다(단, `load_reviewed_variants()`만
`answer_variants.json`을 읽는다 — 그 자체는 부수효과 없는 순수 파일 읽기다).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ASSERTION_EVALUATOR_VERSION = "v2"
ABSTENTION_DETECTOR_VERSION = "v2"

DEFAULT_VARIANTS_PATH = Path(__file__).resolve().parent / "answer_variants.json"

# ---------------------------------------------------------------------------
# §5.2 정규화 파이프라인
# ---------------------------------------------------------------------------

_FULLWIDTH_RE = re.compile(r"[！-～]")
_FULLWIDTH_SPACE_RE = re.compile(r"[　   ]")
_DASH_RE = re.compile(r"[‐-―−]")
_SINGLE_QUOTE_RE = re.compile(r"[‘’]")
_DOUBLE_QUOTE_RE = re.compile(r"[“”]")

_BACKTICK_RE = re.compile(r"`")
_TILDE_RE = re.compile(r"~")
_BOLD_ITALIC_RUN_RE = re.compile(r"\*{2,}")

_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_PP_RE = re.compile(r"(?:%\s*p\b|%\s*포인트|퍼센트\s*포인트|\bpp\b|%p)", re.IGNORECASE)
_PCT_RE = re.compile(r"(?:%|퍼센트)")
_JOIN_NUMBER_UNIT_RE = re.compile(r"(?<=\d)\s+(?=[⟪])")
_SPLIT_ASCII_SEPARATOR_RE = re.compile(r"(?<=[0-9A-Za-z])[_\-](?=[0-9A-Za-z])")
_COLLAPSE_WS_RE = re.compile(r"\s+")

_PP_SENTINEL = "⟪pp⟫"
_PCT_SENTINEL = "⟪pct⟫"


def normalize_text(text: str) -> str:
    """§5.2의 10단계를 정확한 순서로 적용한다. answer/assertion phrase 양쪽에
    동일하게 적용해야 한다(비대칭 처리 금지)."""
    t = unicodedata.normalize("NFC", text)

    t = _FULLWIDTH_RE.sub(lambda m: chr(ord(m.group(0)) - 0xFEE0), t)
    t = _FULLWIDTH_SPACE_RE.sub(" ", t)
    t = _DASH_RE.sub("-", t)
    t = _SINGLE_QUOTE_RE.sub("'", t)
    t = _DOUBLE_QUOTE_RE.sub('"', t)

    t = _BACKTICK_RE.sub("", t)
    t = _TILDE_RE.sub("", t)
    t = _BOLD_ITALIC_RUN_RE.sub("", t)

    t = t.casefold()

    t = _THOUSANDS_RE.sub("", t)

    t = _PP_RE.sub(_PP_SENTINEL, t)
    t = _PCT_RE.sub(_PCT_SENTINEL, t)

    t = _JOIN_NUMBER_UNIT_RE.sub("", t)

    t = _SPLIT_ASCII_SEPARATOR_RE.sub(" ", t)

    t = _COLLAPSE_WS_RE.sub(" ", t).strip()
    return t


def assertion_hit(answer_norm: str, phrase_norm: str) -> bool:
    """§5.4 assertion 매칭 규칙. 숫자로 시작/끝나는 phrase는 인접 숫자 경계
    lookaround로 오탐(예: `10.7%` != `0.7%`)을 막는다."""
    if not phrase_norm:
        return False
    pattern = re.escape(phrase_norm)
    if phrase_norm[0].isdigit():
        pattern = r"(?<![0-9.])" + pattern
    if phrase_norm[-1].isdigit():
        pattern = pattern + r"(?![0-9.])"
    return re.search(pattern, answer_norm) is not None


class VariantTableError(RuntimeError):
    """§5.5 fail-closed 정책. 공식 `v2` profile은 이 예외를 잡아 exit 2로
    변환해야 한다(CLI 계층의 책임, 순수 함수 계층은 sys.exit()을 호출하지
    않는다)."""


@dataclass(frozen=True)
class VariantEntry:
    case_id: str
    assertion_index: int
    add_any_of: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class VariantTable:
    schema_version: str
    entries: tuple[VariantEntry, ...]
    sha256: str
    raw_bytes: bytes

    def variants_for(self, case_id: str, assertion_index: int) -> tuple[str, ...]:
        extra: list[str] = []
        for entry in self.entries:
            if entry.case_id == case_id and entry.assertion_index == assertion_index:
                extra.extend(entry.add_any_of)
        return tuple(extra)


_REQUIRED_VARIANT_KEYS = {"case_id", "assertion_index", "add_any_of", "rationale"}


def load_reviewed_variants(
    path: Path | None = None,
    *,
    required: bool = True,
    expect_sha256: str | None = None,
) -> VariantTable | None:
    """검토된 scoped 변형 표를 로드한다(§5.5).

    - `required=True`이고 파일이 없거나 JSON/schema가 깨졌거나
      `expect_sha256`과 실제 해시가 다르면 `VariantTableError`를 던진다
      (fail-closed, 공식 profile `v2`용).
    - `required=False`이면 파일이 없을 때 `None`을 반환한다(실험 profile
      `v2-no-variants`용). 단, 존재하는데 깨진 파일은 `required` 값과
      무관하게 여전히 오류다(§5.5 표: "동일하게 exit 2").
    """
    target = path or DEFAULT_VARIANTS_PATH
    if not target.exists():
        if required:
            raise VariantTableError(f"변형 표 파일이 없습니다: {target}")
        return None

    raw_bytes = target.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VariantTableError(f"변형 표 JSON 파싱 실패: {target}: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0.0":
        raise VariantTableError(f"변형 표 schema_version이 지원되지 않습니다: {target}")

    raw_entries = payload.get("variants")
    if not isinstance(raw_entries, list):
        raise VariantTableError(f"변형 표에 'variants' 배열이 없습니다: {target}")

    entries: list[VariantEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict) or not _REQUIRED_VARIANT_KEYS.issubset(item.keys()):
            raise VariantTableError(f"변형 표 항목에 필수 키가 없습니다: {item!r}")
        add_any_of = item["add_any_of"]
        if not isinstance(add_any_of, list) or not add_any_of:
            raise VariantTableError(f"add_any_of는 비어 있지 않은 배열이어야 합니다: {item!r}")
        if not isinstance(item["rationale"], str) or not item["rationale"].strip():
            raise VariantTableError(f"rationale이 비어 있습니다: {item!r}")
        entries.append(
            VariantEntry(
                case_id=item["case_id"],
                assertion_index=item["assertion_index"],
                add_any_of=tuple(add_any_of),
                rationale=item["rationale"],
            )
        )

    if expect_sha256 is not None and sha256 != expect_sha256:
        raise VariantTableError(
            f"변형 표 SHA-256이 기대값과 다릅니다: expected={expect_sha256} actual={sha256}"
        )

    return VariantTable(
        schema_version=payload["schema_version"],
        entries=tuple(entries),
        sha256=sha256,
        raw_bytes=raw_bytes,
    )


def assertion_coverage_v2(
    case_id: str,
    answer: str,
    assertions: list,
    variants: VariantTable | None,
) -> tuple[int, int, list[dict]]:
    """(통과 수, 전체 수, per_assertion 상세). `assertions`는
    `evaluation.schema.AnswerAssertion` 리스트다."""
    answer_norm = normalize_text(answer)
    per_assertion: list[dict] = []
    passed = 0
    for index, assertion in enumerate(assertions):
        phrases = list(assertion.any_of)
        if variants is not None:
            phrases = phrases + list(variants.variants_for(case_id, index))
        hit = any(assertion_hit(answer_norm, normalize_text(p)) for p in phrases)
        if hit:
            passed += 1
        per_assertion.append({"index": index, "passed": hit})
    return passed, len(assertions), per_assertion


# ---------------------------------------------------------------------------
# §5.6 abstention detector v1/v2
# ---------------------------------------------------------------------------

ABSTENTION_LITERAL_PHRASES = (
    "제공된 문서에서 관련 정보를 찾을 수 없습니다",
    "제공된 문서만으로는 확실한 답변이 어렵습니다",
)

SCOPE_TOKENS = ("제공된 문서", "문서", "문맥", "자료", "문서 모음", "context")
INFO_TOKENS = ("정보", "내용", "언급", "자료", "데이터", "근거", "기재", "설명", "답변")
ABSENCE_TOKENS = (
    "찾을 수 없",
    "없습니다",
    "없음",
    "존재하지 않",
    "포함되어 있지 않",
    "포함되지 않",
    "확인할 수 없",
    "언급되지 않",
    "나와 있지 않",
    "확인되지 않",
    "제공되지 않",
    "기재되어 있지 않",
    "찾지 못",
)


def detect_abstention_v1(answer: str) -> bool:
    """M2 규칙의 동결 사본 — 두 공식 거절 문구를 NFC 정규화 후 원문 포함
    여부로만 판정한다. answers.py의 기존 `_detect_abstention()`과 동일한
    의미를 유지한다."""
    normalized = unicodedata.normalize("NFC", answer)
    return any(unicodedata.normalize("NFC", p) in normalized for p in ABSTENTION_LITERAL_PHRASES)


def _first_index(text: str, tokens: tuple[str, ...]) -> int | None:
    best: int | None = None
    for token in tokens:
        idx = text.find(token)
        if idx != -1 and (best is None or idx < best):
            best = idx
    return best


def _last_index(text: str, tokens: tuple[str, ...]) -> int | None:
    best: int | None = None
    for token in tokens:
        idx = text.rfind(token)
        if idx != -1 and (best is None or idx > best):
            best = idx
    return best


def detect_abstention_v2(answer: str) -> bool:
    """§5.6의 L1~L3 규칙."""
    n_full = normalize_text(answer)
    if any(normalize_text(p) in n_full for p in ABSTENTION_LITERAL_PHRASES):
        return True

    prose = "\n".join(line for line in answer.splitlines() if not line.strip().startswith("|"))

    for seg in re.split(r"[.!?\n]+", normalize_text(prose)):
        s = _first_index(seg, SCOPE_TOKENS)
        i = _first_index(seg, INFO_TOKENS)
        a = _last_index(seg, ABSENCE_TOKENS)
        if s is not None and i is not None and a is not None and s < i < a:
            return True
    return False


# ---------------------------------------------------------------------------
# §3.2 rules fingerprint
# ---------------------------------------------------------------------------

_NORMALIZATION_STEPS = [
    "nfc",
    "translate_lookalike",
    "strip_markdown",
    "casefold",
    "strip_thousands",
    "canonical_pp",
    "canonical_pct",
    "join_number_unit",
    "split_ascii_separator",
    "collapse_ws",
]


def rules_fingerprint(variants: VariantTable | None) -> str:
    """SHA-256(canonical JSON of rule table) — §3.2."""
    payload = {
        "assertion_version": ASSERTION_EVALUATOR_VERSION,
        "abstention_version": ABSTENTION_DETECTOR_VERSION,
        "normalization_steps": _NORMALIZATION_STEPS,
        "abstention_scope_tokens": list(SCOPE_TOKENS),
        "abstention_info_tokens": list(INFO_TOKENS),
        "abstention_absence_tokens": list(ABSENCE_TOKENS),
        "abstention_literal_phrases": list(ABSTENTION_LITERAL_PHRASES),
        "reviewed_variants_sha256": variants.sha256 if variants is not None else None,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
