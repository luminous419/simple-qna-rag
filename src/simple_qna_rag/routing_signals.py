"""Explicit web/document routing signal classification (M3-REQ-004,
Design.md §7.2 — routing simplification cycle 1, the sole normative
implementation contract).

`classify_explicit_signal()` is a pure function: no model, network, or file
I/O. It implements exactly the three-step decision described in Design.md
§7.2 — WEB command grammar, then DOCUMENT scope token, then NONE (delegated
to the LLM). Earlier iterations' `SOURCE_PARTICLE`/`TOPIC_HEAD`/word-distance
heuristics are intentionally not implemented; recall gaps in the NONE bucket
are resolved by the LLM router, not by more regex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit


class ExplicitSignal(str, Enum):
    WEB = "web"
    DOCUMENT = "document"
    NONE = "none"


@dataclass(frozen=True)
class SignalDecision:
    """`classify_explicit_signal_detail()`의 반환값. `signal`은
    `classify_explicit_signal()`과 동일한 값이고, `rule`은 어떤 grammar가
    판정을 결정했는지(`"direct_command"` | `"channel_search_command"` |
    `"document_scope"` | `"none"`)를 사람이 읽을 수 있게 남긴다."""

    signal: ExplicitSignal
    rule: str


# ---------------------------------------------------------------------------
# §7.2 step 1 — 인용 마스킹, 부정/금지 cue
# ---------------------------------------------------------------------------

_QUOTE_SPAN_RE = re.compile(r'"[^"]*"|`[^`]*`')

_PROHIBITION_CUES = ("하지 말고", "없이", "말고")


def _mask_quoted_spans(text: str) -> str:
    """`"..."`/`` `...` `` 구간 안의 문자를 같은 길이의 공백으로 치환해
    이후 판정에서 완전히 제외한다(위치는 보존해 나머지 문자열의 인덱스가
    바뀌지 않도록 한다)."""
    return _QUOTE_SPAN_RE.sub(lambda m: " " * len(m.group(0)), text)


def _has_prohibition_cue(text: str) -> bool:
    return any(cue in text for cue in _PROHIBITION_CUES)


# ---------------------------------------------------------------------------
# §7.2 step 2 — WEB command grammar (두 문형)
# ---------------------------------------------------------------------------

# CHANNEL 지시어와 WEB_FUSED(채널+검색이 한 어구로 붙은 융합형) 표현을 하나의
# 집합으로 다룬다 — 둘 다 "채널·융합 표현에 조사가 직접 결합"하는 grammar 2의
# 채널 명사가 될 수 있다(§7.2). WEB_FUSED 세 개는 grammar 1(직접 검색 명령)의
# 대상이기도 하다.
_WEB_FUSED = ("웹검색", "웹서치", "구글링")
_CHANNEL_TOKENS = ("웹", "인터넷", "온라인", "구글", "포털", "검색엔진", "web", "google")
_CHANNEL_OR_FUSED = sorted(set(_WEB_FUSED) | set(_CHANNEL_TOKENS), key=len, reverse=True)

# 왼쪽 Unicode 경계: 문자열 시작, 공백, 문장부호·괄호·따옴표 뒤는 경계로
# 인정하되 영숫자/한글이 곧바로 이어지는 복합어 내부(예: freewebsearch)는
# 경계로 인정하지 않는다. `\w`는 Unicode 기본이므로 한글도 포함된다.
_LEFT_BOUNDARY = r"(?<!\w)"

_CHANNEL_ALTERNATION = "|".join(re.escape(t) for t in _CHANNEL_OR_FUSED)

# grammar 1 — 직접 검색 명령: 융합형 표현에 명령형 어미(하다 활용 "해")가
# 중간 명사구 없이 바로 결합한다. "해" 뒤에 어떤 연결형이 오든(예: "해서
# 알려줘") 융합형 직후에 "해"가 바로 붙기만 하면 성립한다.
_DIRECT_COMMAND_RE = re.compile(
    _LEFT_BOUNDARY + r"(?:" + "|".join(re.escape(t) for t in _WEB_FUSED) + r")해"
)

# grammar 2 — 채널 지정 검색 명령: 채널·융합 표현 + (에서|으로|로)가 직접
# 결합한다(중간에 공백·다른 글자 없음).
_CHANNEL_PARTICLE_RE = re.compile(
    _LEFT_BOUNDARY + r"(?:" + _CHANNEL_ALTERNATION + r")(?:에서|으로|로)",
    re.IGNORECASE,
)

# 검색 행위 술어 집합. "일반 응답 동사"(알려/답해/보여)는 여기 포함되지
# 않는다 — grammar 2는 문장의 마지막 술어가 이 술어들로 시작할 때만 성립한다.
# "알아보"/"알아봐"는 알아보다의 규칙/불규칙(모음축약: 보+아→봐) 활용형을
# 모두 포괄하기 위해 둘 다 둔다.
_SEARCH_ACTION_STEMS = ("검색", "찾", "조회", "확인", "알아보", "알아봐")
_SEARCH_ACTION_TAIL_RE = re.compile(r"^(?:" + "|".join(_SEARCH_ACTION_STEMS) + r")")

_TRAILING_PUNCTUATION = "?!.。！？"


def _last_predicate(text: str) -> str:
    """문장 전체의 마지막 어절(공백 기준)을 trailing 문장부호 없이 반환한다.
    관형절 안에서 채널+조사가 등장해도 마지막 술어는 항상 문장 맨 끝의
    실제 서술어이므로, 이 값 하나만 검사하면 관형절 우회를 막을 수 있다."""
    stripped = text.strip().rstrip(_TRAILING_PUNCTUATION)
    if not stripped:
        return ""
    return stripped.split()[-1] if stripped.split() else ""


def _is_search_action_predicate(word: str) -> bool:
    return bool(_SEARCH_ACTION_TAIL_RE.match(word))


def _matches_direct_command(masked_text: str) -> bool:
    return _DIRECT_COMMAND_RE.search(masked_text) is not None


def _matches_channel_search_command(masked_text: str) -> bool:
    if _CHANNEL_PARTICLE_RE.search(masked_text) is None:
        return False
    return _is_search_action_predicate(_last_predicate(masked_text))


# ---------------------------------------------------------------------------
# §7.2 step 3 — DOCUMENT scope 신호
# ---------------------------------------------------------------------------

# 문서/백서/보고서 뒤에 위치·출처를 나타내는 조사가 직접 붙거나("문서에서",
# "백서에 따르면") "모음"/"들"이 끼어드는 형태("문서 모음에", "문서들에")만
# 인정한다. 조사 없이 목적격("문서를")·수식("문서 전체")으로만 쓰인 경우는
# 문서 범위 지시가 아니다(예: "PDF 문서를 텍스트로 변환" — NONE 유지).
_DOCUMENT_SCOPE_RE = re.compile(
    r"(?:문서(?:들)?(?:\s*모음)?|백서|보고서)(?:에서|에|으로|로|으로부터)(?:\s*따르면)?"
)


def classify_explicit_signal(question: str) -> ExplicitSignal:
    """WEB_COMMAND -> DOCUMENT_SCOPE -> NONE 순서의 순수 판정(§7.2)."""
    return classify_explicit_signal_detail(question).signal


def classify_explicit_signal_detail(question: str) -> SignalDecision:
    masked = _mask_quoted_spans(question)

    if not _has_prohibition_cue(masked):
        if _matches_direct_command(masked):
            return SignalDecision(ExplicitSignal.WEB, "direct_command")
        if _matches_channel_search_command(masked):
            return SignalDecision(ExplicitSignal.WEB, "channel_search_command")

    if _DOCUMENT_SCOPE_RE.search(masked):
        return SignalDecision(ExplicitSignal.DOCUMENT, "document_scope")

    return SignalDecision(ExplicitSignal.NONE, "none")


# ---------------------------------------------------------------------------
# §7.3 — corpus topic hint (loopback 조건부)
# ---------------------------------------------------------------------------

_COPY_SUFFIX_RE = re.compile(r"\s*복사본\s*$")
_WHITESPACE_UNDERSCORE_RE = re.compile(r"[\s_]+")


def is_loopback_endpoint(base_url: str) -> bool:
    """`base_url`의 host가 loopback(`localhost`/`127.0.0.1`/`::1`)이면
    True. 파싱할 수 없는 값은 안전 기본값으로 False를 반환한다(§7.3,
    M3-NFR-003 — 판단 불가 시 hint를 억제하는 쪽이 안전)."""
    try:
        host = urlsplit(base_url).hostname
    except ValueError:
        return False
    if host is None:
        return False
    return host.lower() in {"localhost", "127.0.0.1", "::1"}


def build_corpus_topic_hint(file_names: list[str], max_items: int = 25) -> str:
    """`file_names`(정렬된 파일명 목록)에서 확장자·"복사본"·중복 공백/
    언더스코어를 정리한 주제 힌트 문자열을 만든다. 입력이 비어 있으면 빈
    문자열을 반환한다. 순수 함수이며 디렉터리 접근은 호출부의 책임이다."""
    seen: dict[str, None] = {}
    for raw in sorted(file_names):
        name = Path(raw).stem
        name = _COPY_SUFFIX_RE.sub("", name)
        name = _WHITESPACE_UNDERSCORE_RE.sub(" ", name).strip()
        if not name:
            continue
        if name not in seen:
            seen[name] = None

    items = list(seen.keys())[:max_items]
    if not items:
        return ""

    prefix = "등록 문서 주제: "
    body = ", ".join(items)
    hint = prefix + body
    if len(hint) > 1200:
        hint = hint[:1197] + "..."
    return hint
