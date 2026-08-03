"""rag_engine.py의 Retrieval 계측과 evaluation/retrieval.py 단위 테스트 (M2 Phase 4).

이 파일은 두 계층을 검증한다.

1. `RAGEngine._retrieve_documents()`의 trace 계측이 기존 검색 결과를 전혀 바꾸지
   않는지(characterization test, 4개 config 분기 전부) — 실제 임베딩/reranker
   모델 대신 결정론적 가짜 문서를 반환하는 더미로 monkeypatch한다.
2. `evaluation.retrieval.evaluate_retrieval()`/`main()`이 fake RAGEngine을
   주입받아 Recall/MRR/nDCG 집계, dedupe, 제외 규칙, CLI 오류 처리를 올바르게
   수행하는지.

실제 data/, vectorstore/, Ollama, Hugging Face 모델은 어디에서도 사용하지 않는다
(M2-NFR-003).
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import evaluation.retrieval as retrieval_module
import rag_engine
from evaluation.retrieval import evaluate_retrieval, main
from rag_engine import RAGEngine, RetrievalStageTrace, RetrievalTrace


# ---------------------------------------------------------------------------
# 공용 헬퍼
# ---------------------------------------------------------------------------


def _make_engine() -> RAGEngine:
    """실제 프로세스 전역 RAGEngine 싱글톤과 분리된 새 인스턴스를 만든다.

    `RAGEngine.__new__`는 클래스 수준 싱글톤(`_instance`)을 반환하므로 그대로
    `RAGEngine()`을 호출하면 다른 테스트/모듈과 상태를 공유하게 된다.
    `object.__new__`로 우회해 이 테스트 전용의 독립된 인스턴스를 만든다.
    """
    return object.__new__(RAGEngine)


def _docs(*sources: str) -> list[Document]:
    return [
        Document(page_content=f"content-{i}", metadata={"source": s})
        for i, s in enumerate(sources)
    ]


def _stub_mmr(question, documents, top_k=20, lambda_mult=0.5):
    return list(documents[:top_k])


def _stub_rerank(query, documents, top_k=5):
    return list(documents[:top_k])


class _RecordingRetriever:
    """bm25/dense retriever를 흉내내는 결정론적 더미. 호출 인자를 기록해 stage
    wiring이 실제 검색 호출을 건너뛰지 않는지도 확인할 수 있게 한다."""

    def __init__(self, docs: list[Document]):
        self._docs = docs
        self.calls: list[tuple] = []

    def invoke(self, question, top_k=None):
        self.calls.append((question, top_k))
        return list(self._docs)


def _configure_branch(monkeypatch, *, hybrid: bool, mmr: bool, reranker: bool):
    monkeypatch.setattr(rag_engine, "USE_HYBRID_SEARCH", hybrid)
    monkeypatch.setattr(rag_engine, "USE_MMR", mmr)
    monkeypatch.setattr(rag_engine, "USE_RERANKER", reranker)
    monkeypatch.setattr(rag_engine, "BM25_TOP_K", 50)
    monkeypatch.setattr(rag_engine, "DENSE_TOP_K", 50)
    monkeypatch.setattr(rag_engine, "RRF_TOP_K", 20)
    monkeypatch.setattr(rag_engine, "RRF_CONSTANT", 60)
    monkeypatch.setattr(rag_engine, "MMR_K", 20)
    monkeypatch.setattr(rag_engine, "MMR_LAMBDA", 0.5)
    monkeypatch.setattr(rag_engine, "RERANKER_TOP_K", 10)


# ---------------------------------------------------------------------------
# 1. RAGEngine._retrieve_documents() 4-분기 characterization test
# ---------------------------------------------------------------------------


class TestRetrieveDocumentsFourBranches:
    """개발 계획 §3.4의 4개 검색 분기 각각에서 trace=None과 trace=RetrievalTrace()가
    완전히 동일한 문서 리스트(순서 포함)를 반환하는지 고정한다(필수, 선택 아님)."""

    def _hybrid_engine(self, monkeypatch):
        _configure_branch(monkeypatch, hybrid=True, mmr=True, reranker=True)
        engine = _make_engine()
        engine.bm25_retriever = _RecordingRetriever(_docs("a.pdf", "b.pdf", "c.pdf"))
        engine.dense_retriever = _RecordingRetriever(_docs("b.pdf", "d.pdf"))
        engine._apply_mmr = _stub_mmr
        engine._rerank_documents = _stub_rerank
        return engine

    def test_hybrid_mmr_reranker_same_result_with_and_without_trace(self, monkeypatch):
        engine = self._hybrid_engine(monkeypatch)
        without_trace = engine._retrieve_documents("질문")
        with_trace = engine._retrieve_documents("질문", trace=RetrievalTrace())
        assert without_trace == with_trace
        assert [d.metadata["source"] for d in without_trace] == [
            d.metadata["source"] for d in with_trace
        ]

    def test_hybrid_mmr_reranker_stage_names_and_order(self, monkeypatch):
        engine = self._hybrid_engine(monkeypatch)
        trace = RetrievalTrace()
        docs = engine._retrieve_documents("질문", trace=trace)
        names = [s.name for s in trace.stages]
        assert names == ["bm25", "dense", "rrf", "mmr", "reranker", "total"]
        assert names.count("total") == 1
        assert trace.stages[-1].name == "total"
        assert trace.stages[-1].candidate_count == len(docs)
        for s in trace.stages:
            assert s.candidate_count >= 0
            assert s.latency_ms >= 0
        # 실제 검색 호출이 건너뛰어지지 않았는지도 확인한다.
        assert engine.bm25_retriever.calls == [("질문", 50)]
        assert engine.dense_retriever.calls == [("질문", None)]

    def _mmr_only_engine(self, monkeypatch):
        _configure_branch(monkeypatch, hybrid=False, mmr=True, reranker=False)
        engine = _make_engine()
        engine.dense_retriever = _RecordingRetriever(_docs("x.pdf", "y.pdf"))
        engine._apply_mmr = _stub_mmr
        engine._rerank_documents = _stub_rerank
        return engine

    def test_mmr_only_same_result_with_and_without_trace(self, monkeypatch):
        engine = self._mmr_only_engine(monkeypatch)
        without_trace = engine._retrieve_documents("질문")
        with_trace = engine._retrieve_documents("질문", trace=RetrievalTrace())
        assert without_trace == with_trace

    def test_mmr_only_stage_names(self, monkeypatch):
        engine = self._mmr_only_engine(monkeypatch)
        trace = RetrievalTrace()
        docs = engine._retrieve_documents("질문", trace=trace)
        names = [s.name for s in trace.stages]
        # MMR-only 분기는 `dense_retriever` 자체가 이미 search_type="mmr"로
        # 구성돼 있으므로(_setup_retriever), 별도의 "mmr" stage 이름이 아니라
        # "dense" 하나로 기록된다(개발 계획 §3.4 참조 구현과 동일).
        assert names == ["dense", "total"]
        assert names.count("total") == 1
        assert trace.stages[-1].candidate_count == len(docs)

    def _reranker_only_engine(self, monkeypatch):
        _configure_branch(monkeypatch, hybrid=False, mmr=False, reranker=True)
        engine = _make_engine()
        engine.dense_retriever = _RecordingRetriever(_docs("p.pdf", "q.pdf", "r.pdf"))
        engine._rerank_documents = _stub_rerank
        return engine

    def test_reranker_only_same_result_with_and_without_trace(self, monkeypatch):
        engine = self._reranker_only_engine(monkeypatch)
        without_trace = engine._retrieve_documents("질문")
        with_trace = engine._retrieve_documents("질문", trace=RetrievalTrace())
        assert without_trace == with_trace

    def test_reranker_only_stage_names(self, monkeypatch):
        engine = self._reranker_only_engine(monkeypatch)
        trace = RetrievalTrace()
        docs = engine._retrieve_documents("질문", trace=trace)
        names = [s.name for s in trace.stages]
        assert names == ["dense", "reranker", "total"]
        assert names.count("total") == 1
        assert trace.stages[-1].candidate_count == len(docs)

    def _plain_engine(self, monkeypatch):
        _configure_branch(monkeypatch, hybrid=False, mmr=False, reranker=False)
        engine = _make_engine()
        engine.dense_retriever = _RecordingRetriever(_docs("m.pdf", "n.pdf"))
        return engine

    def test_plain_similarity_same_result_with_and_without_trace(self, monkeypatch):
        engine = self._plain_engine(monkeypatch)
        without_trace = engine._retrieve_documents("질문")
        with_trace = engine._retrieve_documents("질문", trace=RetrievalTrace())
        assert without_trace == with_trace

    def test_plain_similarity_stage_names(self, monkeypatch):
        engine = self._plain_engine(monkeypatch)
        trace = RetrievalTrace()
        docs = engine._retrieve_documents("질문", trace=trace)
        names = [s.name for s in trace.stages]
        assert names == ["dense", "total"]
        assert names.count("total") == 1
        assert trace.stages[-1].candidate_count == len(docs)


class TestTraceZeroCostWhenDisabled:
    """M2-REQ-006: trace=None이면 RetrievalStageTrace 객체가 전혀 생성되지
    않는다(비활성 시 zero-cost)."""

    def test_trace_none_creates_no_stage_objects(self, monkeypatch):
        _configure_branch(monkeypatch, hybrid=True, mmr=True, reranker=True)
        engine = _make_engine()
        engine.bm25_retriever = _RecordingRetriever(_docs("a.pdf", "b.pdf"))
        engine.dense_retriever = _RecordingRetriever(_docs("b.pdf", "c.pdf"))
        engine._apply_mmr = _stub_mmr
        engine._rerank_documents = _stub_rerank

        calls: list[tuple] = []
        original = rag_engine.RetrievalStageTrace

        def _spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(rag_engine, "RetrievalStageTrace", _spy)

        engine._retrieve_documents("질문", trace=None)
        assert calls == []

        engine._retrieve_documents("질문", trace=RetrievalTrace())
        assert len(calls) == 6  # bm25, dense, rrf, mmr, reranker, total

    def test_default_call_without_trace_argument_is_unaffected(self, monkeypatch):
        """기존 호출부(RAGEngine.query() 등)와 동일하게 trace 인자를 아예 넘기지
        않아도 그대로 동작해야 한다(M2-REQ-012, 시그니처 확장이 기존 호출자를
        깨뜨리지 않음)."""
        _configure_branch(monkeypatch, hybrid=False, mmr=False, reranker=False)
        engine = _make_engine()
        engine.dense_retriever = _RecordingRetriever(_docs("a.pdf"))
        docs = engine._retrieve_documents("질문")
        assert [d.metadata["source"] for d in docs] == ["a.pdf"]


# ---------------------------------------------------------------------------
# 2. evaluation.retrieval.evaluate_retrieval()/main() 단위 테스트
# ---------------------------------------------------------------------------


def _case(id_: str, question: str, *, relevant_sources=None, relevance_grades=None,
          tags=None, **overrides) -> dict:
    d = {
        "id": id_,
        "question": question,
        "category": "document_qa",
        "expected_route": "document_qa",
        "tags": tags or [],
    }
    if relevant_sources is not None:
        d["relevant_sources"] = relevant_sources
    if relevance_grades is not None:
        d["relevance_grades"] = relevance_grades
    d.update(overrides)
    return d


def _write_dataset(tmp_path: Path, cases: list[dict]) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
        encoding="utf-8",
    )
    return path


class FakeRetrievalEngine:
    """`evaluate_retrieval()`에 주입되는 fake RAGEngine. 실제 모델/벡터스토어를
    전혀 사용하지 않고, `responses[question]`으로 미리 정해둔 문서 리스트(또는
    예외 인스턴스)를 그대로 반환한다."""

    def __init__(self, responses: dict, *, vectorstore_document_count: int | None = None):
        self._responses = responses
        self.calls: list[str] = []
        if vectorstore_document_count is not None:
            self.vectorstore = SimpleNamespace(
                docstore=SimpleNamespace(
                    _dict={f"id{i}": None for i in range(vectorstore_document_count)}
                )
            )

    def _retrieve_documents(self, question, trace=None):
        self.calls.append(question)
        outcome = self._responses[question]
        if isinstance(outcome, Exception):
            raise outcome
        docs = list(outcome)
        if trace is not None:
            trace.stages.append(RetrievalStageTrace("dense", 1.5, len(docs)))
            trace.stages.append(RetrievalStageTrace("total", 2.0, len(docs)))
        return docs


def _install_fake_engine(monkeypatch, engine) -> None:
    monkeypatch.setattr(rag_engine, "get_rag_engine", lambda: engine)


_FAKE_REPRODUCIBILITY = {
    "corpus_manifest": [{"source_id": "a.pdf", "size_bytes": 1, "sha256": "x" * 64}],
    "corpus_manifest_sha256": "y" * 64,
    "vectorstore_fingerprint": {"index_faiss_sha256": "f" * 64, "index_pkl_sha256": "g" * 64},
    "reproducibility_note": None,
}


def _mock_reproducibility(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval_module,
        "build_reproducibility_metadata",
        lambda *a, **k: dict(_FAKE_REPRODUCIBILITY),
    )


class TestEvaluateRetrievalDedupeAndMetrics:
    def test_dedupe_applied_once_and_shared_across_all_metrics(self, tmp_path, monkeypatch):
        _mock_reproducibility(monkeypatch)
        question = "중복 소스 질문"
        dataset_path = _write_dataset(
            tmp_path, [_case("c1", question, relevant_sources=["B.pdf"])]
        )
        docs = _docs("A.pdf", "A.pdf", "A.pdf", "B.pdf")
        engine = FakeRetrievalEngine({question: docs})
        _install_fake_engine(monkeypatch, engine)

        payload = evaluate_retrieval(dataset_path, tmp_path / "reports")
        result = payload["case_results"][0]

        # 중복 제거 후 순위는 [a.pdf, b.pdf] 여야 하고(B는 dedup 후 2위),
        # Recall/MRR/nDCG 모두 이 동일한 순위를 사용해야 한다.
        assert result["ranked_source_ids"] == ["a.pdf", "b.pdf"]
        assert result["metrics"]["recall@1"] == 0.0
        assert result["metrics"]["recall@3"] == 1.0
        assert result["metrics"]["mrr@10"] == pytest.approx(0.5)

    def test_multiple_relevant_sources_partial_recall(self, tmp_path, monkeypatch):
        _mock_reproducibility(monkeypatch)
        question = "다중 정답 질문"
        dataset_path = _write_dataset(
            tmp_path,
            [_case("c1", question, relevant_sources=["a.pdf", "b.pdf", "c.pdf"])],
        )
        engine = FakeRetrievalEngine({question: _docs("a.pdf", "z.pdf")})
        _install_fake_engine(monkeypatch, engine)

        payload = evaluate_retrieval(dataset_path, tmp_path / "reports")
        result = payload["case_results"][0]
        assert result["metrics"]["recall@10"] == pytest.approx(1 / 3)

    def test_empty_search_results_gives_zero_scores_not_error(self, tmp_path, monkeypatch):
        _mock_reproducibility(monkeypatch)
        question = "검색결과 없음"
        dataset_path = _write_dataset(
            tmp_path, [_case("c1", question, relevant_sources=["a.pdf"])]
        )
        engine = FakeRetrievalEngine({question: []})
        _install_fake_engine(monkeypatch, engine)

        payload = evaluate_retrieval(dataset_path, tmp_path / "reports")
        result = payload["case_results"][0]
        assert result["success"] is True
        assert result["ranked_source_ids"] == []
        assert result["metrics"]["recall@1"] == 0.0
        assert result["metrics"]["mrr@10"] == 0.0

    def test_case_with_relevance_grades_gets_ndcg_score(self, tmp_path, monkeypatch):
        _mock_reproducibility(monkeypatch)
        question = "등급 있는 질문"
        dataset_path = _write_dataset(
            tmp_path,
            [
                _case(
                    "c1",
                    question,
                    relevant_sources=["a.pdf"],
                    relevance_grades={"a.pdf": 3},
                )
            ],
        )
        engine = FakeRetrievalEngine({question: _docs("a.pdf")})
        _install_fake_engine(monkeypatch, engine)

        payload = evaluate_retrieval(dataset_path, tmp_path / "reports")
        result = payload["case_results"][0]
        assert result["metrics"]["ndcg@10"] == pytest.approx(1.0)
        assert payload["ndcg_excluded_count"] == 0
        assert payload["metrics"]["ndcg@10"] == pytest.approx(1.0)

    def test_case_without_relevance_grades_excluded_from_ndcg(self, tmp_path, monkeypatch):
        _mock_reproducibility(monkeypatch)
        question = "등급 없는 질문"
        dataset_path = _write_dataset(
            tmp_path, [_case("c1", question, relevant_sources=["a.pdf"])]
        )
        engine = FakeRetrievalEngine({question: _docs("a.pdf")})
        _install_fake_engine(monkeypatch, engine)

        payload = evaluate_retrieval(dataset_path, tmp_path / "reports")
        result = payload["case_results"][0]
        assert result["metrics"]["ndcg@10"] is None
        assert payload["ndcg_excluded_count"] == 1
        assert payload["metrics"]["ndcg@10"] is None
        # Recall/MRR 대상에서는 제외되지 않아야 한다.
        assert result["metrics"]["recall@1"] == 1.0
        assert payload["case_counts"]["excluded"] == 0

    def test_case_without_relevant_sources_excluded_entirely(self, tmp_path, monkeypatch):
        _mock_reproducibility(monkeypatch)
        dataset_path = _write_dataset(tmp_path, [_case("c1", "출처 없는 질문")])
        engine = FakeRetrievalEngine({})
        _install_fake_engine(monkeypatch, engine)

        payload = evaluate_retrieval(dataset_path, tmp_path / "reports")
        assert payload["case_counts"]["total"] == 1
        assert payload["case_counts"]["excluded"] == 1
        assert payload["case_counts"]["success"] == 0
        assert payload["case_counts"]["failure"] == 0
        assert payload["ndcg_excluded_count"] == 1
        assert payload["case_results"] == []
        assert engine.calls == []  # 검색 자체가 호출되지 않아야 한다

    def test_individual_case_failure_is_recorded_and_run_continues(self, tmp_path, monkeypatch):
        _mock_reproducibility(monkeypatch)
        dataset_path = _write_dataset(
            tmp_path,
            [
                _case("c1", "실패할 질문", relevant_sources=["a.pdf"]),
                _case("c2", "성공할 질문", relevant_sources=["b.pdf"]),
            ],
        )
        engine = FakeRetrievalEngine(
            {
                "실패할 질문": RuntimeError("검색 실패"),
                "성공할 질문": _docs("b.pdf"),
            }
        )
        _install_fake_engine(monkeypatch, engine)

        payload = evaluate_retrieval(dataset_path, tmp_path / "reports")
        assert payload["case_counts"]["failure"] == 1
        assert payload["case_counts"]["success"] == 1
        failed = next(r for r in payload["case_results"] if r["id"] == "c1")
        succeeded = next(r for r in payload["case_results"] if r["id"] == "c2")
        assert failed["success"] is False
        assert "검색 실패" in failed["error"]
        assert succeeded["success"] is True
        assert succeeded["metrics"]["recall@1"] == 1.0

    def test_tag_and_limit_filters_are_deterministic(self, tmp_path, monkeypatch):
        _mock_reproducibility(monkeypatch)
        cases = [
            _case("c1", "q1", relevant_sources=["a.pdf"], tags=["keep"]),
            _case("c2", "q2", relevant_sources=["a.pdf"], tags=["skip"]),
            _case("c3", "q3", relevant_sources=["a.pdf"], tags=["keep"]),
            _case("c4", "q4", relevant_sources=["a.pdf"], tags=["keep"]),
        ]
        dataset_path = _write_dataset(tmp_path, cases)
        engine = FakeRetrievalEngine({f"q{i}": _docs("a.pdf") for i in range(1, 5)})
        _install_fake_engine(monkeypatch, engine)

        payload = evaluate_retrieval(dataset_path, tmp_path / "reports", tag="keep", limit=2)
        ids = [r["id"] for r in payload["case_results"]]
        assert ids == ["c1", "c3"]  # 원본 순서에서 'keep' 태그 사례 중 앞의 2개

    def test_vectorstore_document_count_opportunistic(self, tmp_path, monkeypatch):
        _mock_reproducibility(monkeypatch)
        question = "질문"
        dataset_path = _write_dataset(
            tmp_path, [_case("c1", question, relevant_sources=["a.pdf"])]
        )
        engine = FakeRetrievalEngine({question: _docs("a.pdf")}, vectorstore_document_count=42)
        _install_fake_engine(monkeypatch, engine)

        payload = evaluate_retrieval(dataset_path, tmp_path / "reports")
        assert payload["vectorstore_document_count"] == 42

    def test_vectorstore_document_count_none_when_engine_never_initialized(
        self, tmp_path, monkeypatch
    ):
        _mock_reproducibility(monkeypatch)
        dataset_path = _write_dataset(tmp_path, [_case("c1", "출처 없는 질문")])
        engine = FakeRetrievalEngine({})
        _install_fake_engine(monkeypatch, engine)

        payload = evaluate_retrieval(dataset_path, tmp_path / "reports")
        assert payload["vectorstore_document_count"] is None
        assert engine.calls == []

    def test_reproducibility_fields_are_non_null(self, tmp_path, monkeypatch):
        _mock_reproducibility(monkeypatch)
        question = "질문"
        dataset_path = _write_dataset(
            tmp_path, [_case("c1", question, relevant_sources=["a.pdf"])]
        )
        engine = FakeRetrievalEngine({question: _docs("a.pdf")})
        _install_fake_engine(monkeypatch, engine)

        payload = evaluate_retrieval(dataset_path, tmp_path / "reports")
        assert payload["corpus_manifest"] is not None
        assert payload["corpus_manifest_sha256"] is not None
        assert payload["vectorstore_fingerprint"] is not None

    def test_stage_summary_aggregates_latency_and_candidate_counts(self, tmp_path, monkeypatch):
        _mock_reproducibility(monkeypatch)
        question = "질문"
        dataset_path = _write_dataset(
            tmp_path, [_case("c1", question, relevant_sources=["a.pdf"])]
        )
        engine = FakeRetrievalEngine({question: _docs("a.pdf", "b.pdf")})
        _install_fake_engine(monkeypatch, engine)

        payload = evaluate_retrieval(dataset_path, tmp_path / "reports")
        assert "dense" in payload["stage_summary"]
        assert "total" in payload["stage_summary"]
        assert payload["stage_summary"]["dense"]["count"] == 1
        assert payload["stage_summary"]["total"]["candidate_count_mean"] == pytest.approx(2.0)

    def test_writes_json_and_markdown_report(self, tmp_path, monkeypatch):
        _mock_reproducibility(monkeypatch)
        question = "질문"
        dataset_path = _write_dataset(
            tmp_path, [_case("c1", question, relevant_sources=["a.pdf"])]
        )
        engine = FakeRetrievalEngine({question: _docs("a.pdf")})
        _install_fake_engine(monkeypatch, engine)

        output_dir = tmp_path / "reports"
        evaluate_retrieval(dataset_path, output_dir)
        json_files = list(output_dir.glob("retrieval_*.json"))
        md_files = list(output_dir.glob("retrieval_*.md"))
        assert len(json_files) == 1
        assert len(md_files) == 1


# ---------------------------------------------------------------------------
# 3. CLI (`main()`) 테스트
# ---------------------------------------------------------------------------


class TestMainCLI:
    def test_help_exits_zero_without_importing_rag_engine(self, monkeypatch):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "rag_engine":
                pytest.fail("rag_engine이 --help 경로에서 import 되었습니다")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded)

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_missing_required_options_is_non_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_missing_dataset_file_exits_1_with_guidance(self, tmp_path, capsys):
        missing = tmp_path / "nope.jsonl"
        code = main(["--dataset", str(missing), "--output", str(tmp_path / "out")])
        captured = capsys.readouterr()
        assert code == 1
        assert "오류" in captured.err
        assert "조치" in captured.err

    def test_engine_init_failure_exits_non_zero_with_document_register_guidance(
        self, tmp_path, monkeypatch, capsys
    ):
        dataset_path = _write_dataset(
            tmp_path, [_case("c1", "q1", relevant_sources=["a.pdf"])]
        )

        def _boom():
            raise RuntimeError("RAG 엔진 초기화 실패")

        monkeypatch.setattr(rag_engine, "get_rag_engine", _boom)

        code = main(["--dataset", str(dataset_path), "--output", str(tmp_path / "out")])
        captured = capsys.readouterr()
        assert code != 0
        assert "document_register.py" in captured.err

    def test_missing_corpus_via_reproducibility_exits_non_zero_with_guidance(
        self, tmp_path, monkeypatch, capsys
    ):
        # relevant_sources가 없어 엔진이 초기화되지 않는 사례만 있는 데이터셋 —
        # build_reproducibility_metadata() 단계에서만 실패가 발생하는 경로를
        # 별도로 검증한다.
        dataset_path = _write_dataset(tmp_path, [_case("c1", "출처 없는 질문")])

        def _raise_fnf(*args, **kwargs):
            raise FileNotFoundError("data/ 디렉터리가 없습니다")

        monkeypatch.setattr(retrieval_module, "build_reproducibility_metadata", _raise_fnf)

        code = main(["--dataset", str(dataset_path), "--output", str(tmp_path / "out")])
        captured = capsys.readouterr()
        assert code != 0
        assert "document_register.py" in captured.err

    def test_successful_run_returns_zero_and_prints_metrics(self, tmp_path, monkeypatch, capsys):
        _mock_reproducibility(monkeypatch)
        question = "질문"
        dataset_path = _write_dataset(
            tmp_path, [_case("c1", question, relevant_sources=["a.pdf"])]
        )
        engine = FakeRetrievalEngine({question: _docs("a.pdf")})
        _install_fake_engine(monkeypatch, engine)

        code = main(["--dataset", str(dataset_path), "--output", str(tmp_path / "out")])
        assert code == 0
        captured = capsys.readouterr()
        metrics = json.loads(captured.out)
        assert "recall@1" in metrics
        assert "mrr@10" in metrics
        assert "ndcg@10" in metrics
