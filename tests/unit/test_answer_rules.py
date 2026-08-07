"""Tests for evaluation/answer_rules.py (M3-REQ-006, Design.md §5).

No model/network/vectorstore access. Fixtures for the 8 assertion + 3
abstention false negatives use answer text copied verbatim from the approved
M2 report (evaluation/reports/m2_full/answers/answers_20260804T145621300637Z.json,
cited by evaluation/answer_variants.json's `review.evidence_report`) so the
replay is self-contained and does not depend on that gitignored file existing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import answer_rules as ar
from evaluation.schema import AnswerAssertion

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# §5.2 normalization pipeline, step by step
# ---------------------------------------------------------------------------


def test_nfc_and_fullwidth_ascii() -> None:
    assert ar.normalize_text("ＡＢＣ") == "abc"


def test_fullwidth_and_special_spaces_collapse() -> None:
    assert ar.normalize_text("a　b c d e") == "a b c d e"


def test_dash_variants_normalize_to_hyphen() -> None:
    # Dash lookalikes canonicalize to "-" (step 2), then since "TF-IDF" is
    # all-ASCII-alnum around the hyphen, step 9 (split_ascii_separator)
    # turns it into a space — this is the same behavior as `chunk_size` ->
    # `chunk size` and is intentional, not a bug.
    assert ar.normalize_text("TF‑IDF") == "tf idf"
    # Korean text around the dash is not ASCII, so step 9 does not apply
    # and the canonicalized hyphen from step 2 survives.
    assert ar.normalize_text("검색‐생성") == "검색-생성"


def test_curly_quotes_normalize_to_straight() -> None:
    assert ar.normalize_text("‘a’") == "'a'"
    assert ar.normalize_text("“b”") == '"b"'


def test_strip_markdown_removes_backtick_tilde_and_bold_runs() -> None:
    assert ar.normalize_text("`chunk_size`") == "chunk size"
    assert ar.normalize_text("**핵심 요약**") == "핵심 요약"
    assert ar.normalize_text("~strike~") == "strike"


def test_strip_markdown_keeps_single_asterisk() -> None:
    assert ar.normalize_text("2*3") == "2*3"


def test_casefold_applied() -> None:
    assert ar.normalize_text("ABC") == "abc"


def test_strip_thousands_separator() -> None:
    assert ar.normalize_text("1,000") == ar.normalize_text("1000")


def test_thousands_separator_requires_three_digit_group() -> None:
    # (?=\d{3}\b) means a comma not followed by exactly a 3-digit group
    # boundary is left untouched.
    assert "," in ar.normalize_text("1,0000")


def test_canonical_pp_and_pct_sentinels() -> None:
    assert ar.normalize_text("1.0%p") == ar.normalize_text("1.0 pp")
    assert ar.normalize_text("1.0% 포인트") == ar.normalize_text("1.0%p")
    assert ar.normalize_text("0.7%") == ar.normalize_text("0.7 %")


def test_join_number_unit_absorbs_space_before_sentinel() -> None:
    a = ar.normalize_text("0.7 %")
    b = ar.normalize_text("0.7%")
    assert a == b
    assert ar._PCT_SENTINEL in a


def test_split_ascii_separator_only_between_ascii_alnum() -> None:
    assert ar.normalize_text("chunk_size") == ar.normalize_text("chunk size")
    assert ar.normalize_text("chunk-overlap") == ar.normalize_text("chunk overlap")


def test_split_ascii_separator_does_not_apply_to_korean() -> None:
    assert ar.normalize_text("비_공개") != ar.normalize_text("비 공개")


def test_collapse_whitespace_and_strip() -> None:
    assert ar.normalize_text("  a   b\n\nc  ") == "a b c"


def test_sentinel_characters_not_present_in_ordinary_input() -> None:
    # translate_lookalike (step 2) must not touch the sentinel brackets
    # themselves, and ordinary text must never contain them accidentally.
    assert "⟪" not in ar.normalize_text("hello world")
    assert "⟫" not in ar.normalize_text("hello world")


# ---------------------------------------------------------------------------
# §5.3 반례 C1-C10 (금지되는 정규화)
# ---------------------------------------------------------------------------


def test_c1_pct_vs_pp_never_collapse() -> None:
    phrase = ar.normalize_text("1.0%")
    answer = ar.normalize_text("1.0%p")
    assert not ar.assertion_hit(answer, phrase)


def test_c2_number_boundary_lookaround() -> None:
    phrase = ar.normalize_text("0.7%")
    answer = ar.normalize_text("10.7%")
    assert not ar.assertion_hit(answer, phrase)


def test_c3_number_substring_boundary() -> None:
    phrase = ar.normalize_text("0.7%")
    answer = ar.normalize_text("0.07%")
    assert not ar.assertion_hit(answer, phrase)


def test_c4_leading_sign_preserved() -> None:
    phrase = ar.normalize_text("-1.0pp")
    answer = ar.normalize_text("1.0pp")
    assert not ar.assertion_hit(answer, phrase)


def test_c5_negation_not_detected_known_limitation() -> None:
    # v2 does not do negation detection; this is a known limitation shared
    # with v1, not a regression.
    phrase = ar.normalize_text("증가")
    answer = ar.normalize_text("증가하지 않았다")
    assert ar.assertion_hit(answer, phrase)


def test_c6_single_asterisk_preserved() -> None:
    phrase = ar.normalize_text("2*3")
    answer = ar.normalize_text("23")
    assert not ar.assertion_hit(answer, phrase)


def test_c7_separator_becomes_space_not_deleted() -> None:
    phrase = ar.normalize_text("chunk size")
    answer = ar.normalize_text("chunksize")
    assert not ar.assertion_hit(answer, phrase)


def test_c8_korean_underscore_not_split() -> None:
    phrase = ar.normalize_text("비 공개")
    answer = ar.normalize_text("비_공개")
    assert not ar.assertion_hit(answer, phrase)


def test_c9_thousands_boundary_condition() -> None:
    phrase = ar.normalize_text("1000")
    answer = ar.normalize_text("1,0000")
    assert not ar.assertion_hit(answer, phrase)


def test_c10_sentinel_collision_safe() -> None:
    phrase = ar.normalize_text("⟪pp⟫")
    answer = ar.normalize_text("some text ⟪pp⟫ more text")
    assert ar.assertion_hit(answer, phrase)


# ---------------------------------------------------------------------------
# §5.6 abstention v2 반례 A1-A9
# ---------------------------------------------------------------------------


def test_a1_v1_literal_phrase() -> None:
    assert ar.detect_abstention_v2("제공된 문서에서 관련 정보를 찾을 수 없습니다") is True


def test_a2_v1_literal_yesno_phrase() -> None:
    assert ar.detect_abstention_v2("제공된 문서만으로는 확실한 답변이 어렵습니다") is True


def test_a3_scope_info_absence_order() -> None:
    text = "문맥에 2025년 노벨 경제학상 수상자에 관한 언급이 존재하지 않습니다."
    assert ar.detect_abstention_v2(text) is True


def test_a4_scope_info_absence_order_2() -> None:
    text = "제공된 문서에서는 관련하여 확인할 수 있는 정보가 없습니다."
    assert ar.detect_abstention_v2(text) is True


def test_a5_table_row_excluded() -> None:
    text = "| **문서 길이 고려 여부** | **없음** – 모든 단어가 동등하게 처리되는 문서를 전제로 함 |"
    assert ar.detect_abstention_v2(text) is False


def test_a6_table_row_excluded_2() -> None:
    text = "| 회담 주최국 | 해당 없음 – 문서에는 다른 나라가 주최한 회담 언급이 없습니다. |"
    assert ar.detect_abstention_v2(text) is False


def test_a7_bare_negative_without_scope_info_order() -> None:
    text = "제공된 문서에 없는 내용은 추측하지 않습니다."
    assert ar.detect_abstention_v2(text) is False


def test_a8_missing_scope_token() -> None:
    text = "관련 정보가 없는 경우에는 웹 검색이 필요합니다."
    assert ar.detect_abstention_v2(text) is False


def test_a9_partial_refusal_known_limitation() -> None:
    text = "2025년 성장률은 0.7%이며, 추가 통계는 문서에 정보가 없습니다."
    assert ar.detect_abstention_v2(text) is True


# ---------------------------------------------------------------------------
# 11 false negative replay (M3 Requirement §4.2 acceptance criterion),
# with literal answer text from the approved M2 report.
# ---------------------------------------------------------------------------

_REAL_ANSWERS = {
    "dq-rag-001": (
        "**핵심 요약**\nRAG는 사용자의 질문을 벡터화해 대규모 문서 데이터베이스에서 "
        "가장 관련성 높은 정보를 검색한 뒤, 그 내용을 언어 모델에 프롬프트로 제공하여 "
        "보다 정확하고 상세한 답변을 생성하는 두 단계(검색-생성) 시스템입니다.\n\n"
        "### 1. RAG의 목표와 개념\n"
        "- **핵심 아이디어**: “검색(Information Retrieval) + 생성(Generation)”"
        "의 결합으로, LLM이 직접 기억하고 있지 않은 사실도 활용할 수 있게 됩니다."
    ),
    "dq-langgraph-reducer-001": (
        "**핵심 요약**\nLangGraph에서 Reducer는 두 개의 메시지 리스트(`left`와 `right`)를 "
        "받아 자동으로 하나의 리스트에 병합해 주는 역할을 합니다."
    ),
    "dq-sparse-vs-dense-001": (
        "| 항목 | Sparse Retriever | Dense Retriever |\n"
        "|------|------------------|-----------------|\n"
        "| **표현 방식** | 이산적인 키워드 벡터 (TF-IDF, BM25 등) | 연속 고차원 임베딩 벡터 "
        "(딥러닝 기반) |"
    ),
    "dq-econ-growth-revision-001": (
        "| 항목 | 기존 전망치 | 수정치 |\n"
        "|------|------------|-------|\n"
        "| **한국 경제성장률** | 1.7 % | 0.7 % |\n"
        "| **변화량** | - | -1.0 pp (하향 조정) |\n"
        "수정치가 기존 전망치를 1.0 pp 낮추어 0.7 %로 설정되었습니다."
    ),
    "dq-textsplit-steps-001": (
        "분할을 위한 파라미터를 정의합니다 (`chunk_size`, `chunk_overlap`).\n"
        "- `chunk_size`: 한 청크당 포함될 토큰 수(예: 1000)\n"
        "- `chunk_overlap`: 인접 청크 간에 겹치는 토큰 수(예: 50)"
    ),
    "dq-realestate-procedure-001": (
        "**단계 2: 구청에 거래 허가 신청서 제출**\n"
        "- **수행 내용:** 해당 지역 구청(또는 시·군)으로 “주택거래 허가 신청”을 합니다.\n"
        "**단계 3: 구청의 허가 승인 및 공고 확인**\n"
        "- **수행 내용:** 구청에서 “주택거래 허가”를 승인하면, 해당 부동산에 대한 "
        "거래가 정식으로 가능해집니다.\n"
        "**단계 4: 실거주 의무 이행**\n"
        "- **수행 내용:** 매매 이후 실제로 해당 주택에 거주하기 시작합니다."
    ),
}

_REAL_ASSERTIONS = {
    "dq-rag-001": [["정보 검색과 생성을 통합", "검색과 생성을 통합"]],
    "dq-langgraph-reducer-001": [["list에 메시지를 추가", "메시지를 추가"]],
    "dq-sparse-vs-dense-001": [
        ["이산적인 키워드", "키워드 벡터"],
        ["연속적인 벡터", "고차원 벡터"],
    ],
    "dq-econ-growth-revision-001": [["0.7%"], ["1.0%p", "1.0% 포인트", "1.0%포인트"]],
    "dq-textsplit-steps-001": [
        ["단위 크기 선정", "chunk size", "청크 크기"],
        ["청크 오버랩", "chunk overlap"],
    ],
    "dq-realestate-procedure-001": [["구청 허가"], ["실거주 의무"]],
}

_REAL_ABSTENTION_ANSWERS = {
    "ua-001": (
        "| 항목 | 제공된 문서 |\n|------|------------|\n| 애플의 자체 개발 LLM 이름 | "
        "**관련 정보 없음** |\n\n제공된 문서에서는 애플이 자체 개발한 LLM 이름을 "
        "확인할 수 있는 정보가 없습니다."
    ),
    "ua-002": (
        "**핵심 요약**\n문맥에 2025년 노벨 경제학상 수상자에 관한 언급이 존재하지 "
        "않습니다.\n\n**결론**\n제공된 문서에서는 2025년 노벨 경제학상 수상자에 대한 "
        "정보를 찾을 수 없습니다."
    ),
    "ua-004": "제공된 문서에서 대한민국 2025년 최저임금에 관한 정보가 포함되어 있지 않습니다.",
}


@pytest.fixture(scope="module")
def variants() -> ar.VariantTable:
    table = ar.load_reviewed_variants()
    assert table is not None
    return table


@pytest.mark.parametrize("case_id", sorted(_REAL_ASSERTIONS))
def test_all_assertion_false_negatives_become_true_positive(case_id: str, variants: ar.VariantTable) -> None:
    answer = _REAL_ANSWERS[case_id]
    answer_norm = ar.normalize_text(answer)
    for index, any_of in enumerate(_REAL_ASSERTIONS[case_id]):
        phrases = list(any_of) + list(variants.variants_for(case_id, index))
        hit = any(ar.assertion_hit(answer_norm, ar.normalize_text(p)) for p in phrases)
        assert hit, f"{case_id}[{index}] still a false negative under v2"


@pytest.mark.parametrize("case_id", sorted(_REAL_ABSTENTION_ANSWERS))
def test_all_abstention_false_negatives_become_true_positive(case_id: str) -> None:
    answer = _REAL_ABSTENTION_ANSWERS[case_id]
    assert ar.detect_abstention_v1(answer) is False  # confirms these were v1 false negatives
    assert ar.detect_abstention_v2(answer) is True


def test_assertion_coverage_v2_uses_answer_assertion_objects(variants: ar.VariantTable) -> None:
    assertions = [AnswerAssertion(any_of=x) for x in _REAL_ASSERTIONS["dq-econ-growth-revision-001"]]
    passed, total, per_assertion = ar.assertion_coverage_v2(
        "dq-econ-growth-revision-001",
        _REAL_ANSWERS["dq-econ-growth-revision-001"],
        assertions,
        variants,
    )
    assert (passed, total) == (2, 2)
    assert all(item["passed"] for item in per_assertion)


# ---------------------------------------------------------------------------
# rules_fingerprint
# ---------------------------------------------------------------------------


def test_rules_fingerprint_stable_for_same_variants(variants: ar.VariantTable) -> None:
    assert ar.rules_fingerprint(variants) == ar.rules_fingerprint(variants)


def test_rules_fingerprint_changes_with_variant_table(tmp_path: Path, variants: ar.VariantTable) -> None:
    payload = json.loads(ar.DEFAULT_VARIANTS_PATH.read_text(encoding="utf-8"))
    payload["variants"].append(
        {
            "case_id": "dq-fake-001",
            "assertion_index": 0,
            "add_any_of": ["다른 표현"],
            "rationale": "테스트용 추가 항목",
        }
    )
    other_path = tmp_path / "other_variants.json"
    other_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    other_table = ar.load_reviewed_variants(other_path)

    assert ar.rules_fingerprint(variants) != ar.rules_fingerprint(other_table)


def test_rules_fingerprint_none_variants_differs_from_loaded() -> None:
    assert ar.rules_fingerprint(None) != ar.rules_fingerprint(ar.load_reviewed_variants())


# ---------------------------------------------------------------------------
# fail-closed variant table loading (§5.5)
# ---------------------------------------------------------------------------


def test_load_reviewed_variants_missing_file_required_raises(tmp_path: Path) -> None:
    with pytest.raises(ar.VariantTableError):
        ar.load_reviewed_variants(tmp_path / "does_not_exist.json", required=True)


def test_load_reviewed_variants_missing_file_not_required_returns_none(tmp_path: Path) -> None:
    assert ar.load_reviewed_variants(tmp_path / "does_not_exist.json", required=False) is None


def test_load_reviewed_variants_bad_json_raises_even_when_not_required(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ar.VariantTableError):
        ar.load_reviewed_variants(bad, required=False)
    with pytest.raises(ar.VariantTableError):
        ar.load_reviewed_variants(bad, required=True)


def test_load_reviewed_variants_unsupported_schema_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "variants.json"
    path.write_text(json.dumps({"schema_version": "9.9.9", "variants": []}), encoding="utf-8")
    with pytest.raises(ar.VariantTableError):
        ar.load_reviewed_variants(path, required=True)


def test_load_reviewed_variants_missing_required_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "variants.json"
    payload = {
        "schema_version": "1.0.0",
        "variants": [{"case_id": "x", "assertion_index": 0, "add_any_of": ["y"]}],  # no rationale
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ar.VariantTableError):
        ar.load_reviewed_variants(path, required=True)


def test_load_reviewed_variants_empty_add_any_of_raises(tmp_path: Path) -> None:
    path = tmp_path / "variants.json"
    payload = {
        "schema_version": "1.0.0",
        "variants": [
            {"case_id": "x", "assertion_index": 0, "add_any_of": [], "rationale": "r"}
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ar.VariantTableError):
        ar.load_reviewed_variants(path, required=True)


def test_load_reviewed_variants_sha256_mismatch_raises(tmp_path: Path, variants: ar.VariantTable) -> None:
    with pytest.raises(ar.VariantTableError):
        ar.load_reviewed_variants(ar.DEFAULT_VARIANTS_PATH, required=True, expect_sha256="0" * 64)


def test_load_reviewed_variants_sha256_match_succeeds(variants: ar.VariantTable) -> None:
    table = ar.load_reviewed_variants(ar.DEFAULT_VARIANTS_PATH, required=True, expect_sha256=variants.sha256)
    assert table is not None


def test_variant_scope_is_case_and_assertion_index_only(variants: ar.VariantTable) -> None:
    # A variant scoped to (dq-rag-001, 0) must not apply to a different case
    # or a different assertion index of the same case.
    assert variants.variants_for("dq-rag-001", 1) == ()
    assert variants.variants_for("some-other-case", 0) == ()
