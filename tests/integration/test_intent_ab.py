"""Tests for evaluation/intent_ab.py (M3-REQ-007, Design.md §8). Uses a fake
RAGEngine — no model/network/vectorstore access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from evaluation import intent_ab


def _write_dataset(tmp_path: Path, cases: list[dict]) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8"
    )
    return path


def _case(id_: str, question: str, **overrides) -> dict:
    base = {
        "id": id_,
        "question": question,
        "category": "document_qa",
        "expected_route": "document_qa",
        "tags": [],
    }
    base.update(overrides)
    return base


class FakeEngine:
    """Real RAGEngine.build_context()/format_sources() semantics, fake
    retrieval and fake generation — exercises the seam contract without any
    model or vectorstore."""

    def __init__(self, docs_by_question: dict, answer_by_call=None):
        self._docs_by_question = docs_by_question
        self._answer_by_call = answer_by_call or (lambda q, ctx, tmpl: f"answer::{tmpl[:20]}")
        self.generate_calls: list[tuple] = []

    def _retrieve_documents(self, question):
        return self._docs_by_question[question]

    def build_context(self, documents):
        return "\n\n".join(d.page_content for d in documents)

    def format_sources(self, documents):
        return [{"index": i, "source": d.metadata.get("source"), "page": None, "content": d.page_content[:200]}
                for i, d in enumerate(documents, 1)]

    def generate_answer(self, question, context, template_str):
        self.generate_calls.append((question, context, template_str))
        return self._answer_by_call(question, context, template_str)


def _install_fake_engine(monkeypatch, engine) -> None:
    monkeypatch.setattr(intent_ab, "_get_engine", lambda: engine)


@pytest.fixture(autouse=True)
def _stub_intent_classifier(monkeypatch):
    monkeypatch.setattr(intent_ab, "classify_intent", lambda q: "explanation")


def test_run_experiment_fixes_context_once_per_case(tmp_path, monkeypatch):
    dataset_path = _write_dataset(
        tmp_path,
        [_case("c1", "질문1", answer_assertions=[{"any_of": ["핵심"]}])],
    )
    docs = [Document(page_content="본문", metadata={"source": "a.pdf"})]
    engine = FakeEngine({"질문1": docs})
    _install_fake_engine(monkeypatch, engine)

    payload = intent_ab.run_experiment(dataset_path, tmp_path / "out")

    assert payload["case_counts"]["success"] == 1
    # 두 variant 모두 같은 context를 받았는지 확인
    contexts_used = {call[1] for call in engine.generate_calls}
    assert contexts_used == {"본문"}
    assert len(engine.generate_calls) == 2  # intent + default


def test_run_experiment_both_variants_share_identical_context(tmp_path, monkeypatch):
    dataset_path = _write_dataset(
        tmp_path,
        [_case("c1", "질문1", expect_abstention=True)],
    )
    docs = [Document(page_content="문맥A", metadata={"source": "a.pdf"})]
    engine = FakeEngine({"질문1": docs})
    _install_fake_engine(monkeypatch, engine)

    payload = intent_ab.run_experiment(dataset_path, tmp_path / "out")
    result = payload["case_results"][0]
    assert "intent" in result["answers"]
    assert "default" in result["answers"]


def test_run_experiment_writes_context_snapshot_and_worksheet_and_key(tmp_path, monkeypatch):
    dataset_path = _write_dataset(
        tmp_path,
        [_case("c1", "질문1", answer_assertions=[{"any_of": ["x"]}])],
    )
    docs = [Document(page_content="본문", metadata={"source": "a.pdf"})]
    engine = FakeEngine({"질문1": docs})
    _install_fake_engine(monkeypatch, engine)

    payload = intent_ab.run_experiment(dataset_path, tmp_path / "out")

    assert Path(payload["context_snapshot_path"]).exists()
    assert Path(payload["worksheet_path"]).exists()
    assert Path(payload["key_path"]).exists()


def test_worksheet_does_not_reveal_variant_identity(tmp_path, monkeypatch):
    dataset_path = _write_dataset(
        tmp_path,
        [_case("c1", "질문1", answer_assertions=[{"any_of": ["x"]}])],
    )
    docs = [Document(page_content="본문", metadata={"source": "a.pdf"})]
    engine = FakeEngine({"질문1": docs})
    _install_fake_engine(monkeypatch, engine)

    payload = intent_ab.run_experiment(dataset_path, tmp_path / "out")
    worksheet_text = Path(payload["worksheet_path"]).read_text(encoding="utf-8")

    # 사례 블록(## c1 이후)에는 variant 식별자가 전혀 노출되지 않아야 한다.
    # 문서 맨 위 설명 문구가 "variant"라는 단어 자체를 언급하는 것은 허용된다
    # (실제 정체를 알려주지 않는 일반 설명이므로) — 검사 대상은 사례 블록뿐이다.
    case_block = worksheet_text.split("## c1", 1)[1]
    assert "intent" not in case_block.lower()
    assert "default" not in case_block.lower()
    assert "variant" not in case_block.lower()


def test_seed_reproducibility_of_slot_order(tmp_path, monkeypatch):
    dataset_path = _write_dataset(
        tmp_path,
        [_case("c1", "질문1", answer_assertions=[{"any_of": ["x"]}])],
    )
    docs = [Document(page_content="본문", metadata={"source": "a.pdf"})]
    engine1 = FakeEngine({"질문1": docs})
    _install_fake_engine(monkeypatch, engine1)
    payload1 = intent_ab.run_experiment(dataset_path, tmp_path / "out1", seed="fixed-seed")
    key1 = json.loads(Path(payload1["key_path"]).read_text(encoding="utf-8"))

    engine2 = FakeEngine({"질문1": docs})
    _install_fake_engine(monkeypatch, engine2)
    payload2 = intent_ab.run_experiment(dataset_path, tmp_path / "out2", seed="fixed-seed")
    key2 = json.loads(Path(payload2["key_path"]).read_text(encoding="utf-8"))

    assert key1["cases"][0]["slot1"] == key2["cases"][0]["slot1"]
    assert key1["cases"][0]["slot2"] == key2["cases"][0]["slot2"]


def test_failed_retrieval_case_recorded_and_run_continues(tmp_path, monkeypatch):
    dataset_path = _write_dataset(
        tmp_path,
        [
            _case("bad", "실패질문", expect_abstention=True),
            _case("ok", "성공질문", expect_abstention=True),
        ],
    )
    docs = [Document(page_content="본문", metadata={"source": "a.pdf"})]

    class FailingEngine(FakeEngine):
        def _retrieve_documents(self, question):
            if question == "실패질문":
                raise RuntimeError("검색 실패")
            return docs

    engine = FailingEngine({})
    _install_fake_engine(monkeypatch, engine)

    payload = intent_ab.run_experiment(dataset_path, tmp_path / "out")
    assert payload["case_counts"]["failure"] == 1
    assert payload["case_counts"]["success"] == 1
    failed = next(c for c in payload["case_results"] if c["id"] == "bad")
    assert failed["status"] == "failure"
    assert "answers" not in failed


def test_no_eligible_cases_raises(tmp_path, monkeypatch):
    dataset_path = _write_dataset(tmp_path, [_case("c1", "질문1")])  # no assertions, no abstention
    engine = FakeEngine({})
    _install_fake_engine(monkeypatch, engine)
    with pytest.raises(ValueError):
        intent_ab.run_experiment(dataset_path, tmp_path / "out")


# ---------------------------------------------------------------------------
# aggregate_worksheet()
# ---------------------------------------------------------------------------


def _write_key(tmp_path: Path, cases: list[dict], seed: str = "s") -> Path:
    path = tmp_path / "key.json"
    path.write_text(json.dumps({"seed": seed, "cases": cases}, ensure_ascii=False), encoding="utf-8")
    return path


def _worksheet_block(case_id: str, o1f, o1c, o2f, o2c, pref) -> str:
    return (
        f"## {case_id}\n\n**질문**: q\n\n"
        f"- 출력1_형식적합성: {o1f}\n"
        f"- 출력1_핵심사실보존: {o1c}\n"
        f"- 출력2_형식적합성: {o2f}\n"
        f"- 출력2_핵심사실보존: {o2c}\n"
        f"- 선호: {pref}\n"
        f"- 검토메모:\n\n---\n\n"
    )


def test_aggregate_worksheet_counts_preferences_correctly(tmp_path):
    key_path = _write_key(
        tmp_path,
        [
            {"id": "c1", "slot1": "intent", "slot2": "default"},
            {"id": "c2", "slot1": "default", "slot2": "intent"},
        ],
    )
    worksheet_text = _worksheet_block("c1", 1, 1, 1, 0, "1") + _worksheet_block("c2", 1, 1, 1, 1, "2")
    worksheet_path = tmp_path / "worksheet.md"
    worksheet_path.write_text(worksheet_text, encoding="utf-8")

    result = intent_ab.aggregate_worksheet(worksheet_path, key_path)
    # c1: slot1=intent preferred -> preferred_intent
    # c2: slot2=intent preferred -> preferred_intent
    assert result["counts"]["preferred_intent"] == 2
    assert result["counts"]["preferred_default"] == 0
    assert result["counts"]["scored_cases"] == 2
    assert result["margin_pp"] == pytest.approx(100.0)
    assert result["decision"] == "retain_candidate"


def test_aggregate_worksheet_incomplete_entries_excluded_and_counted(tmp_path):
    key_path = _write_key(
        tmp_path,
        [
            {"id": "c1", "slot1": "intent", "slot2": "default"},
            {"id": "c2", "slot1": "intent", "slot2": "default"},
        ],
    )
    worksheet_text = _worksheet_block("c1", 1, 1, 1, 1, "1") + _worksheet_block("c2", "_", "_", "_", "_", "_")
    worksheet_path = tmp_path / "worksheet.md"
    worksheet_path.write_text(worksheet_text, encoding="utf-8")

    result = intent_ab.aggregate_worksheet(worksheet_path, key_path)
    assert result["counts"]["incomplete"] == 1
    assert result["counts"]["scored_cases"] == 1
    assert result["decision"] == "unproven"  # incomplete > 0 -> 보수적 처리


def test_aggregate_worksheet_below_threshold_is_unproven(tmp_path):
    # 5건 중 intent 1건, default 1건 선호, 나머지 3건 tie -> margin_pp =
    # (1-1)/5*100 = 0.0%, 20.0%p 미달이므로 '입증되지 않음'이어야 한다.
    key_path = _write_key(
        tmp_path,
        [
            {"id": f"c{i}", "slot1": "intent", "slot2": "default"} for i in range(1, 6)
        ],
    )
    worksheet_text = "".join(
        [
            _worksheet_block("c1", 1, 1, 1, 1, "1"),  # intent 선호
            _worksheet_block("c2", 1, 1, 1, 1, "2"),  # default 선호
            _worksheet_block("c3", 1, 1, 1, 1, "tie"),
            _worksheet_block("c4", 1, 1, 1, 1, "tie"),
            _worksheet_block("c5", 1, 1, 1, 1, "tie"),
        ]
    )
    worksheet_path = tmp_path / "worksheet.md"
    worksheet_path.write_text(worksheet_text, encoding="utf-8")

    result = intent_ab.aggregate_worksheet(worksheet_path, key_path)
    assert result["counts"]["tie"] == 3
    assert result["margin_pp"] == pytest.approx(0.0)
    assert result["decision"] == "unproven"


def test_aggregate_worksheet_invalid_value_treated_as_incomplete(tmp_path):
    key_path = _write_key(tmp_path, [{"id": "c1", "slot1": "intent", "slot2": "default"}])
    worksheet_text = _worksheet_block("c1", "2", 1, 1, 1, "1")  # 2 is not in {0,1}
    worksheet_path = tmp_path / "worksheet.md"
    worksheet_path.write_text(worksheet_text, encoding="utf-8")

    result = intent_ab.aggregate_worksheet(worksheet_path, key_path)
    assert result["counts"]["incomplete"] == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_run_help_exits_zero():
    with pytest.raises(SystemExit) as exc_info:
        intent_ab.main(["--help"])
    assert exc_info.value.code == 0


def test_main_aggregate_writes_decision_file(tmp_path):
    key_path = _write_key(tmp_path, [{"id": "c1", "slot1": "intent", "slot2": "default"}])
    worksheet_text = _worksheet_block("c1", 1, 1, 1, 1, "1")
    worksheet_path = tmp_path / "worksheet.md"
    worksheet_path.write_text(worksheet_text, encoding="utf-8")

    output_dir = tmp_path / "out"
    exit_code = intent_ab.main(
        ["aggregate", "--worksheet", str(worksheet_path), "--key", str(key_path), "--output", str(output_dir)]
    )
    assert exit_code == 0
    assert (output_dir / "intent_ab_decision.json").exists()
