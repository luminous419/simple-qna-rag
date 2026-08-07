"""Tests for src/simple_qna_rag/routing_signals.py (M3-REQ-004,
Design.md §7.2 — routing simplification cycle 1 normative grammar).

No model/network — `classify_explicit_signal()` and `is_loopback_endpoint()`
are pure functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_qna_rag.routing_signals import (
    ExplicitSignal,
    build_corpus_topic_hint,
    classify_explicit_signal,
    is_loopback_endpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "evaluation" / "datasets" / "golden.jsonl"

WEB_EXACT_SET = {
    "ws-000",
    "ws-002",
    "ws-005",
    "ws-007",
    "ws-009",
    "rr-ws-python-version-001",
    "rr-ws-bitcoin-price-001",
    "rr-ws-samsung-stock-001",
}

DOCUMENT_EXACT_SET = {
    "bd-000",
    "ua-000",
    "ua-001",
    "ua-002",
    "ua-003",
    "ua-004",
    "ua-005",
    "ua-006",
    "dq-agent-arch-001",
    "dq-agent-vs-model-001",
    "dq-kb-price-outlook-001",
    "dq-kb-gangnam-001",
}


def _load_golden_cases() -> list[dict]:
    cases = []
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))
    return cases


# ---------------------------------------------------------------------------
# 골든 76건 exact set — WEB 8 / DOCUMENT 12 / NONE 56
# ---------------------------------------------------------------------------


class TestGoldenExactSet:
    @pytest.fixture(scope="class")
    def golden_cases(self) -> list[dict]:
        cases = _load_golden_cases()
        assert len(cases) == 76
        return cases

    def test_web_exact_set_no_false_positive_no_false_negative(self, golden_cases):
        got_web = {c["id"] for c in golden_cases if classify_explicit_signal(c["question"]) == ExplicitSignal.WEB}
        assert got_web == WEB_EXACT_SET

    def test_document_exact_set_no_false_positive_no_false_negative(self, golden_cases):
        got_doc = {
            c["id"] for c in golden_cases if classify_explicit_signal(c["question"]) == ExplicitSignal.DOCUMENT
        }
        assert got_doc == DOCUMENT_EXACT_SET

    def test_none_bucket_is_exactly_the_complement(self, golden_cases):
        got_none = {
            c["id"] for c in golden_cases if classify_explicit_signal(c["question"]) == ExplicitSignal.NONE
        }
        expected_none = {c["id"] for c in golden_cases} - WEB_EXACT_SET - DOCUMENT_EXACT_SET
        assert got_none == expected_none
        assert len(expected_none) == 56

    def test_three_sets_partition_all_76_cases(self, golden_cases):
        assert WEB_EXACT_SET.isdisjoint(DOCUMENT_EXACT_SET)
        assert len(WEB_EXACT_SET) + len(DOCUMENT_EXACT_SET) + 56 == 76


# ---------------------------------------------------------------------------
# 문형 양성 사례
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "오늘 날씨를 웹에서 검색해줘",
        "최신 환율을 인터넷에서 찾아줘",
        "온라인에서 검색해줘",
        "구글에서 조회해줘",
    ],
)
def test_channel_specified_search_command_is_web(question):
    assert classify_explicit_signal(question) == ExplicitSignal.WEB


@pytest.mark.parametrize(
    "question",
    [
        "웹검색해줘",
        "구글링해줘",
        "구글링해서 알려줘",
        "웹서치해줘",
    ],
)
def test_direct_search_command_is_web(question):
    assert classify_explicit_signal(question) == ExplicitSignal.WEB


# ---------------------------------------------------------------------------
# 일반 응답 술어 / 관형절 / 검색 주제 → NONE (LLM 위임)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "웹검색으로 최신 환율 알려줘",
        "인터넷에서 답해줘",
        "웹검색으로 알려줘",
    ],
)
def test_general_response_predicate_is_not_search_action_command(question):
    assert classify_explicit_signal(question) == ExplicitSignal.NONE


@pytest.mark.parametrize(
    "question",
    [
        "웹검색에서 사용하는 API 구조 알려줘",
        "구글에서 사용하는 검색 기술 알려줘",
    ],
)
def test_relative_clause_minimal_pair_does_not_bypass_last_predicate_check(question):
    """관형절(-는) 안에 채널+조사가 있어도 문장 전체의 마지막 술어가 일반
    응답 동사면 NONE이어야 한다(필수 회귀 테스트, Requirement §5)."""
    assert classify_explicit_signal(question) == ExplicitSignal.NONE


@pytest.mark.parametrize(
    "question",
    [
        "웹검색 방법 알려줘",
        "구글링 기능 알려줘",
    ],
)
def test_search_topic_question_without_adjacency_is_none(question):
    assert classify_explicit_signal(question) == ExplicitSignal.NONE


def test_compound_word_boundary_excludes_bare_channel_match():
    assert classify_explicit_signal("freewebsearch 사용법 알려줘") == ExplicitSignal.NONE
    assert classify_explicit_signal("websocket 설정을 알려줘") == ExplicitSignal.NONE
    assert classify_explicit_signal("googleapis 사용법을 알려줘") == ExplicitSignal.NONE


# ---------------------------------------------------------------------------
# 인용·부정 전처리
# ---------------------------------------------------------------------------


def test_quoted_web_command_is_masked_out():
    assert classify_explicit_signal('"웹에서 검색해줘"라는 문구를 설명해줘') == ExplicitSignal.NONE


def test_quoted_span_does_not_hide_a_real_document_scope_signal():
    assert classify_explicit_signal('"웹 검색"의 뜻을 문서에서 알려줘') == ExplicitSignal.DOCUMENT


@pytest.mark.parametrize(
    "question",
    [
        "웹 검색은 하지 말고 문서로 답해줘",
        "인터넷 없이 문서에서만 찾아줘",
    ],
)
def test_prohibition_cue_suppresses_web_signal(question):
    result = classify_explicit_signal(question)
    assert result != ExplicitSignal.WEB


# ---------------------------------------------------------------------------
# Unicode 왼쪽 경계
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "질문:웹에서 검색해줘",
        "(구글링해서 알려줘)",
        "\"웹에서 검색해줘\" 아 잠깐, 그냥 웹에서 검색해줘",
    ],
)
def test_left_boundary_after_punctuation_is_recognized(question):
    assert classify_explicit_signal(question) == ExplicitSignal.WEB


# ---------------------------------------------------------------------------
# DOCUMENT 신호
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "이 문서에서 관련 내용을 찾아줘",
        "제공된 문서에서 답을 찾아줘",
        "이 문서들에 나온 내용을 알려줘",
        "이 문서 모음에서 확인해줘",
        "백서에 따르면 어떤 내용인가요",
        "보고서에서 언급된 내용은 무엇인가요",
    ],
)
def test_document_scope_tokens_are_recognized(question):
    assert classify_explicit_signal(question) == ExplicitSignal.DOCUMENT


def test_bare_document_word_without_scope_particle_is_not_a_signal():
    # "PDF 문서를 텍스트로 변환" 같은 일반 기술 설명은 문서 범위 지시가
    # 아니다(목적격 조사 "를"만 있고 위치/출처 조사가 없음).
    assert classify_explicit_signal("PDF 문서를 텍스트로 변환하는 방법은 무엇인가요?") == ExplicitSignal.NONE


def test_weak_signal_material_word_is_not_a_hard_rule():
    # bd-002 "관련된 자료가 있으면 알려줘" — '자료'는 하드 규칙에 넣지 않고
    # LLM에 위임한다(Design.md O9).
    assert classify_explicit_signal("관련된 자료가 있으면 알려줘") == ExplicitSignal.NONE


# ---------------------------------------------------------------------------
# 우선순위: WEB 명령 문형 vs DOCUMENT 범위 지시가 동시에 관측되면 WEB 우선
# ---------------------------------------------------------------------------


def test_web_command_wins_over_document_scope_when_both_present():
    assert classify_explicit_signal("웹검색으로 이 문서 내용을 확인해줘") == ExplicitSignal.WEB


# ---------------------------------------------------------------------------
# is_loopback_endpoint()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://[::1]:11434",
        "https://LOCALHOST:8080",
    ],
)
def test_is_loopback_endpoint_true_cases(url):
    assert is_loopback_endpoint(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:11434",
        "http://192.168.0.10:11434",
        "not a url",
        "",
    ],
)
def test_is_loopback_endpoint_false_cases(url):
    assert is_loopback_endpoint(url) is False


# ---------------------------------------------------------------------------
# build_corpus_topic_hint()
# ---------------------------------------------------------------------------


def test_build_corpus_topic_hint_strips_extension_and_copy_suffix():
    hint = build_corpus_topic_hint(["2025 KB 부동산 보고서 복사본.pdf", "LangGraph 개요.pdf"])
    assert "2025 KB 부동산 보고서" in hint
    assert "복사본" not in hint
    assert ".pdf" not in hint


def test_build_corpus_topic_hint_dedupes_and_respects_max_items():
    names = [f"파일{i}.txt" for i in range(30)]
    hint = build_corpus_topic_hint(names, max_items=5)
    assert hint.count("파일") == 5


def test_build_corpus_topic_hint_empty_input_returns_empty_string():
    assert build_corpus_topic_hint([]) == ""


def test_build_corpus_topic_hint_collapses_whitespace_and_underscores():
    hint = build_corpus_topic_hint(["a__b   c.txt"])
    assert "a b c" in hint
