"""
평가 지표 계산 함수 (M2-REQ-005~008, M2-REQ-011).

이 모듈은 data/, vectorstore/, 모델을 전혀 사용하지 않는다 — 순수 계산 함수만 담아
모델/vectorstore/네트워크 없이 단위 테스트할 수 있어야 한다(M2-NFR-003, M2-REQ-011
"메트릭 단위 테스트는 모델/vectorstore/Ollama/네트워크를 사용하면 안 된다").
"""

from __future__ import annotations

import math
import statistics
import unicodedata

from evaluation.schema import AnswerAssertion, normalize_source_id

SCHEMA_VERSION = "1.0.0"


def dedupe_preserve_order(ids: list[str]) -> list[str]:
    """최초 등장 순서를 유지한 채 중복 제거. recall_at_k/mrr_at_k/ndcg_at_k가 모두 이
    함수의 출력을 입력으로 받아야 하며, 그래야 세 지표에서 "k"가 동일하게 "top-k 고유
    source"를 의미한다(개발 계획 §3.5 — 중복 제거를 metric 함수 밖 호출부에서 한 번만
    수행). 이 함수 자체는 정규화를 하지 않으므로 호출부가 이미 normalize_source_id()를
    거친 id 리스트를 전달한다고 가정한다."""
    seen: set[str] = set()
    result: list[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _require_positive_k(k: int) -> None:
    """k < 1이면 ValueError를 던진다. 음수 k는 Python의 음수 slice 규칙 때문에
    ranked_ids[:k]가 "마지막 |k|개를 제외한 결과"로 조용히 재해석되어 의미 없는
    값을 낸다(M2_Phase3_code_review_result.md P3) — recall_at_k/mrr_at_k/
    ndcg_at_k 세 함수가 동일한 계약을 쓴다."""
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 합니다: {k!r}")


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """len({ranked_ids[:k]} ∩ relevant_ids) / len(relevant_ids).

    ranked_ids는 이미 dedupe_preserve_order()를 거친 리스트여야 한다(내부에서 다시
    중복을 제거하지 않는다). relevant_ids가 비어 있으면 이 사례는 애초에 Retrieval
    평가 대상이 아니므로(relevant_sources가 없는 사례는 호출부가 걸러야 한다)
    ValueError를 던진다. k < 1도 ValueError."""
    _require_positive_k(k)
    if not relevant_ids:
        raise ValueError("relevant_ids가 비어 있습니다 — 평가 대상이 아닌 사례는 호출부에서 걸러야 합니다")
    hit = len(set(ranked_ids[:k]) & relevant_ids)
    return hit / len(relevant_ids)


def mrr_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """top-k 안에서 처음 등장하는 관련 문서의 1/rank. k 안에 관련 문서가 없으면 0.0.

    recall_at_k와 동일한 이유로 relevant_ids가 비어 있거나 k < 1이면 ValueError를
    던진다."""
    _require_positive_k(k)
    if not relevant_ids:
        raise ValueError("relevant_ids가 비어 있습니다 — 평가 대상이 아닌 사례는 호출부에서 걸러야 합니다")
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _dcg(ranked_ids: list[str], relevance_grades: dict[str, int], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        grade = relevance_grades.get(doc_id, 0)
        gain = (2**grade) - 1
        dcg += gain / math.log2(rank + 1)
    return dcg


def ndcg_at_k(ranked_ids: list[str], relevance_grades: dict[str, int], k: int) -> float:
    """개발 계획 §3.5에서 고정한 nDCG 공식. gain은 2**grade - 1(표준 graded gain).

    Precondition: relevance_grades는 normalize_relevance_grades()를 거친, key가 이미
    normalize_source_id()로 정규화된 mapping이어야 한다 — ranked_ids의 각 id와 같은
    정규화 형태가 아니면 relevance_grades.get(doc_id, 0)이 항상 0을 반환해 관련
    문서도 비관련으로 취급되고, 이 함수 자체는 그 사실을 알 방법이 없어 조용히 틀린
    값을 낸다(개발 계획 §3.5 "grade key 정규화 책임" 참고).

    relevance_grades가 비어 있으면 idcg가 0이 되어 0.0을 반환한다(예외를 던지지
    않음) — Phase 1 schema가 all-zero/빈 grade 사례를 이미 골든셋 저작 시점에
    막아주지만, 이 함수 자체는 임의의 입력에 대해 방어적으로 동작해야 한다.
    k < 1이면 recall_at_k/mrr_at_k와 동일하게 ValueError."""
    _require_positive_k(k)
    dcg = _dcg(ranked_ids, relevance_grades, k)
    ideal_order = sorted(relevance_grades, key=relevance_grades.get, reverse=True)
    idcg = _dcg(ideal_order, relevance_grades, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_recall_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """라벨별 precision/recall/F1과 confusion matrix를 반환한다.

    {
        <label>: {"precision": .., "recall": .., "f1": ..},
        ...
        "confusion_matrix": {<actual>: {<predicted>: count, ...}, ...},
    }

    분모가 0인 경우(해당 라벨을 한 번도 예측하지 않았거나, 실제로 한 번도 나타나지
    않은 경우) precision/recall/f1은 0.0으로 정의한다(0으로 나누지 않음).
    y_true/y_pred의 각 값은 반드시 labels 안에 있어야 한다(그렇지 않으면 KeyError —
    "모든 예측이 한 route"처럼 labels 밖의 값(예: 도구 미선택/예외)은 호출부가
    별도로 집계하고 이 함수에는 전달하지 않는다)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true와 y_pred의 길이가 다릅니다")

    confusion: dict[str, dict[str, int]] = {actual: {pred: 0 for pred in labels} for actual in labels}
    for actual, predicted in zip(y_true, y_pred):
        confusion[actual][predicted] += 1

    result: dict = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        result[label] = {"precision": precision, "recall": recall, "f1": f1}

    result["confusion_matrix"] = confusion
    return result


def percentile(values: list[float], p: float) -> float | None:
    """nearest-rank 방식의 백분위수(0 <= p <= 100). 외부 라이브러리의 보간 방식
    차이를 피하기 위해 직접 구현한다(M2-NFR-001 결정론). values가 비어 있으면
    None을 반환한다(M2-REQ-011 "빈 latency 목록" 케이스).

    p가 [0, 100] 밖이면 values가 비어 있는지와 무관하게 ValueError를 던진다 —
    이전에는 범위를 벗어난 p가 조용히 clamp되어(p=-1이 p=0처럼, p=101이 p=100처럼
    동작) 잘못된 호출이 정상 결과처럼 보였다(M2_Phase3_code_review_result.md P2)."""
    if not (0 <= p <= 100):
        raise ValueError(f"p는 0~100 사이여야 합니다: {p!r}")
    if not values:
        return None
    sorted_values = sorted(values)
    n = len(sorted_values)
    rank = math.ceil(p / 100 * n)
    rank = max(1, min(rank, n))
    return sorted_values[rank - 1]


def mean_median(values: list[float]) -> tuple[float | None, float | None]:
    """(평균, 중앙값). values가 비어 있으면 (None, None)."""
    if not values:
        return None, None
    return statistics.mean(values), statistics.median(values)


def assertion_coverage(answer: str, assertions: list[AnswerAssertion]) -> tuple[int, int]:
    """(통과한 assertion 수, 전체 assertion 수). 각 AnswerAssertion은 any_of 중
    하나라도 answer에 포함되면 통과한다. 비교 전 Unicode NFC 정규화와 casefold()를
    적용한다(M2-REQ-008 "문자열 비교는 Unicode NFC와 대소문자 정규화를 적용해야
    합니다"). assertions가 비어 있으면 (0, 0)을 반환한다."""
    normalized_answer = unicodedata.normalize("NFC", answer).casefold()
    passed = 0
    for assertion in assertions:
        if any(
            unicodedata.normalize("NFC", phrase).casefold() in normalized_answer
            for phrase in assertion.any_of
        ):
            passed += 1
    return passed, len(assertions)


def normalize_relevance_grades(grades: dict[str, int]) -> dict[str, int]:
    """{normalize_source_id(k): v for k, v in grades.items()}이되, 정규화 후 서로
    다른 raw key가 충돌하면 ValueError를 던진다.

    Phase 1 schema(GoldenCase.relevance_grades)는 골든셋 저작 시점에 이미 이런
    충돌을 막아주지만, 이 함수는 GoldenCase를 거치지 않은 임의의 dict도 받을 수
    있으므로 방어적으로 재검사한다(개발 계획 §3.5)."""
    normalized: dict[str, int] = {}
    seen_raw: dict[str, str] = {}
    for raw_key, grade in grades.items():
        norm_key = normalize_source_id(raw_key)
        if norm_key in seen_raw:
            raise ValueError(
                f"relevance_grades 정규화 후 key가 충돌합니다: {seen_raw[norm_key]!r}와 "
                f"{raw_key!r}가 모두 {norm_key!r}로 정규화됩니다"
            )
        seen_raw[norm_key] = raw_key
        normalized[norm_key] = grade
    return normalized
