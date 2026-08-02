"""
골든 데이터셋 로더 및 검증기 (M2-REQ-002, M2-REQ-004).

data/, vectorstore/, 모델을 전혀 사용하지 않는다 — import나
`python -m evaluation.dataset validate` 실행에 Ollama나 벡터스토어가
필요해서는 안 된다(M2-NFR-003). CI가 이 전제로 돌아간다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from evaluation.schema import Category, GoldenCase, Intent, is_answer_eval_eligible

KOREAN_PATTERN = re.compile(r"[가-힣]")

MIN_TOTAL_CASES = 60
MIN_DOCUMENT_QA = 40
MIN_WEB_SEARCH = 10
MIN_BOUNDARY_OR_UNANSWERABLE = 10
MIN_DOCUMENT_QA_WITH_SOURCES = 30
MIN_DOCUMENT_QA_WITH_ASSERTIONS = 20
MIN_EXPECT_ABSTENTION = 5
MIN_KOREAN_RATIO = 0.8
MIN_PER_INTENT = 5
REQUIRED_INTENTS = (Intent.EXPLANATION, Intent.COMPARISON, Intent.PROCEDURE, Intent.YESNO)


class DatasetError(Exception):
    """dataset.py 전용 오류. case_id/line_number가 있으면 메시지 앞에 위치를 붙인다.

    kind는 CLI가 "다음 조치" 안내를 오류 종류에 맞게 고르는 데 쓰인다
    (M2_Phase1_code_review_result.md P3) — "content"(기본값)는 golden.jsonl의
    특정 줄/사례를 고쳐야 하는 경우(JSON 문법, 스키마 위반, 중복 id)이고,
    "io"는 파일 자체를 찾거나 읽을 수 없는 경우(파일 없음, 디렉터리, 권한,
    인코딩)라 "줄/사례 수정" 안내가 맞지 않는다."""

    def __init__(
        self,
        message: str,
        *,
        case_id: str | None = None,
        line_number: int | None = None,
        kind: str = "content",
    ) -> None:
        self.case_id = case_id
        self.line_number = line_number
        self.kind = kind
        location = []
        if line_number is not None:
            location.append(f"line {line_number}")
        if case_id is not None:
            location.append(f"id={case_id}")
        prefix = f"[{', '.join(location)}] " if location else ""
        super().__init__(f"{prefix}{message}")


def load_jsonl(path: Path) -> list[GoldenCase]:
    """UTF-8 JSON Lines 파일을 GoldenCase 리스트로 로드한다.

    빈 줄은 건너뛴다. 파싱/스키마/중복 id 오류는 물론 파일 열기·읽기·인코딩
    오류까지 모두 DatasetError로 변환되어 CLI가 항상 사람이 읽을 수 있는
    오류만 노출한다(M2-REQ-016, design_review.md 1차 P2).
    """
    if not path.exists():
        raise DatasetError(
            f"데이터셋 파일을 찾을 수 없습니다: {path}. "
            f"경로를 확인하거나 Phase 2 골든셋 작성을 먼저 완료하세요.",
            kind="io",
        )

    cases: list[GoldenCase] = []
    seen_ids: dict[str, int] = {}

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue

                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise DatasetError(
                        f"JSON 파싱 실패: {exc.msg}. 이 줄의 JSON 문법을 확인하세요.",
                        line_number=line_number,
                    ) from exc

                try:
                    case = GoldenCase.model_validate(payload)
                except ValidationError as exc:
                    detail = "; ".join(
                        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                        for err in exc.errors()
                    )
                    raise DatasetError(
                        f"스키마 검증 실패: {detail}",
                        case_id=payload.get("id") if isinstance(payload, dict) else None,
                        line_number=line_number,
                    ) from exc

                if case.id in seen_ids:
                    raise DatasetError(
                        f"중복된 id입니다. 이전 등장 위치: line {seen_ids[case.id]}. "
                        f"모든 사례는 파일 전체에서 고유한 id를 가져야 합니다.",
                        case_id=case.id,
                        line_number=line_number,
                    )
                seen_ids[case.id] = line_number

                cases.append(case)
    except UnicodeDecodeError as exc:
        raise DatasetError(
            f"UTF-8로 디코딩할 수 없습니다: {exc}. 파일이 UTF-8로 저장됐는지 확인하세요.",
            kind="io",
        ) from exc
    except OSError as exc:
        raise DatasetError(
            f"파일을 열거나 읽을 수 없습니다: {exc}. 경로와 권한을 확인하세요.",
            kind="io",
        ) from exc

    return cases


@dataclass
class ValidationReport:
    total: int
    by_category: dict[str, int] = field(default_factory=dict)
    document_qa_with_relevant_sources: int = 0
    document_qa_with_answer_assertions: int = 0
    total_with_expect_abstention: int = 0
    korean_ratio: float = 0.0
    answer_eval_case_count: int = 0
    answer_eval_cases_without_intent: int = 0
    intent_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_category": self.by_category,
            "document_qa_with_relevant_sources": self.document_qa_with_relevant_sources,
            "document_qa_with_answer_assertions": self.document_qa_with_answer_assertions,
            "total_with_expect_abstention": self.total_with_expect_abstention,
            "korean_ratio": round(self.korean_ratio, 4),
            "answer_eval_case_count": self.answer_eval_case_count,
            "answer_eval_cases_without_intent": self.answer_eval_cases_without_intent,
            "intent_counts": self.intent_counts,
            "valid": self.is_valid,
            "errors": self.errors,
        }


def _is_korean(question: str) -> bool:
    return bool(KOREAN_PATTERN.search(question))


def validate_composition(cases: list[GoldenCase]) -> ValidationReport:
    """M2-REQ-002의 9개 구성 규칙을 검사한다. cases는 이미 load_jsonl()을
    통과해 필드 유효성이 보장된 상태여야 한다."""

    report = ValidationReport(total=len(cases))
    errors: list[str] = []

    for cat in Category:
        report.by_category[cat.value] = sum(1 for c in cases if c.category == cat)

    # intent 최소 수량은 실제로 intent 정확도가 측정되는 Answer 평가 대상
    # 사례에서만 집계한다(design_review.md 2차 P1) — web_search나 Retrieval
    # 전용 사례에 intent만 붙여도 구성 검증이 통과해버리는 문제를 막는다.
    answer_eval_cases = [c for c in cases if is_answer_eval_eligible(c)]
    report.answer_eval_case_count = len(answer_eval_cases)
    report.answer_eval_cases_without_intent = sum(
        1 for c in answer_eval_cases if c.expected_intent is None
    )
    for intent in Intent:
        report.intent_counts[intent.value] = sum(
            1 for c in answer_eval_cases if c.expected_intent == intent
        )

    document_qa_cases = [c for c in cases if c.category == Category.DOCUMENT_QA]
    report.document_qa_with_relevant_sources = sum(
        1 for c in document_qa_cases if c.relevant_sources
    )
    report.document_qa_with_answer_assertions = sum(
        1 for c in document_qa_cases if c.answer_assertions
    )
    report.total_with_expect_abstention = sum(1 for c in cases if c.expect_abstention)

    korean_count = sum(1 for c in cases if _is_korean(c.question))
    report.korean_ratio = korean_count / len(cases) if cases else 0.0

    if report.total < MIN_TOTAL_CASES:
        errors.append(
            f"전체 사례 수가 {report.total}개로 최소 {MIN_TOTAL_CASES}개에 미달합니다."
        )
    if report.by_category[Category.DOCUMENT_QA.value] < MIN_DOCUMENT_QA:
        errors.append(
            f"document_qa 사례가 {report.by_category[Category.DOCUMENT_QA.value]}개로 "
            f"최소 {MIN_DOCUMENT_QA}개에 미달합니다."
        )
    if report.by_category[Category.WEB_SEARCH.value] < MIN_WEB_SEARCH:
        errors.append(
            f"web_search 사례가 {report.by_category[Category.WEB_SEARCH.value]}개로 "
            f"최소 {MIN_WEB_SEARCH}개에 미달합니다."
        )
    boundary_or_unanswerable = (
        report.by_category[Category.BOUNDARY.value]
        + report.by_category[Category.UNANSWERABLE.value]
    )
    if boundary_or_unanswerable < MIN_BOUNDARY_OR_UNANSWERABLE:
        errors.append(
            f"boundary+unanswerable 사례가 {boundary_or_unanswerable}개로 "
            f"최소 {MIN_BOUNDARY_OR_UNANSWERABLE}개에 미달합니다."
        )
    if report.document_qa_with_relevant_sources < MIN_DOCUMENT_QA_WITH_SOURCES:
        errors.append(
            f"relevant_sources가 있는 document_qa 사례가 "
            f"{report.document_qa_with_relevant_sources}개로 최소 "
            f"{MIN_DOCUMENT_QA_WITH_SOURCES}개에 미달합니다."
        )
    if report.document_qa_with_answer_assertions < MIN_DOCUMENT_QA_WITH_ASSERTIONS:
        errors.append(
            f"answer_assertions가 있는 document_qa 사례가 "
            f"{report.document_qa_with_answer_assertions}개로 최소 "
            f"{MIN_DOCUMENT_QA_WITH_ASSERTIONS}개에 미달합니다."
        )
    if report.total_with_expect_abstention < MIN_EXPECT_ABSTENTION:
        errors.append(
            f"expect_abstention=true 사례가 {report.total_with_expect_abstention}개로 "
            f"최소 {MIN_EXPECT_ABSTENTION}개에 미달합니다."
        )
    if report.korean_ratio < MIN_KOREAN_RATIO:
        errors.append(
            f"한국어 질문 비율이 {report.korean_ratio:.1%}로 최소 "
            f"{MIN_KOREAN_RATIO:.0%}에 미달합니다."
        )
    for intent in REQUIRED_INTENTS:
        count = report.intent_counts[intent.value]
        if count < MIN_PER_INTENT:
            errors.append(
                f"Answer 평가 대상(answer_assertions 또는 expect_abstention=true) 중 "
                f"intent={intent.value} 사례가 {count}개로 최소 {MIN_PER_INTENT}개에 "
                f"미달합니다."
            )

    report.errors = errors
    return report


def main(argv: list[str] | None = None) -> int:
    """유효한 subcommand 또는 인자 없음(no-argv) 경로에서는 항상 int를
    반환한다(design_review.md 1차 P3) — 그래야 테스트가
    pytest.raises(SystemExit) 없이 반환값만으로 종료 코드를 검증할 수 있다.
    다만 존재하지 않는 subcommand나 잘못된 옵션(예: main(["bogus"]))은
    argparse 자체의 문법 오류 처리 경로를 타므로 표준 SystemExit(2)를 그대로
    던진다 — "모든 경로에서 int를 반환한다"는 이전 설명은 부정확했다
    (design_review.md 2차 P3)."""
    parser = argparse.ArgumentParser(prog="python -m evaluation.dataset")
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser(
        "validate", help="골든 데이터셋 schema/구성 검증"
    )
    validate_parser.add_argument("path", type=Path, help="golden.jsonl 경로")

    args = parser.parse_args(argv)

    if args.command == "validate":
        return _run_validate(args.path)

    parser.print_help(sys.stderr)
    return 2


def _run_validate(path: Path) -> int:
    try:
        cases = load_jsonl(path)
    except DatasetError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        if exc.kind == "io":
            print(
                "다음 조치: 파일 경로가 올바른지, 접근 권한이 있는지, UTF-8로 "
                "저장됐는지 확인한 뒤 다시 실행하세요.",
                file=sys.stderr,
            )
        else:
            print(
                "다음 조치: 위 줄/사례를 golden.jsonl에서 수정한 뒤 다시 실행하세요.",
                file=sys.stderr,
            )
        return 1

    report = validate_composition(cases)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))

    if not report.is_valid:
        print(f"\n검증 실패: {len(report.errors)}건", file=sys.stderr)
        for err in report.errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            "다음 조치: evaluation/datasets/golden.jsonl에 부족한 카테고리/필드를 "
            "가진 사례를 추가하거나 기존 사례를 수정한 뒤 다시 실행하세요.",
            file=sys.stderr,
        )
        return 1

    print("\n검증 통과.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
