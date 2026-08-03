"""evaluation/reporting.py 단위 테스트 (M2 Phase 3, M2-REQ-010).

build_corpus_manifest()/build_vectorstore_fingerprint()는 파일 바이트만 읽으므로
tmp_path의 더미 파일로 검증한다 — 실제 data/, vectorstore/, 모델을 사용하지 않는다.
"""

from __future__ import annotations

import hashlib
import json

import pytest

import config
import evaluation.reporting as reporting
from evaluation.reporting import (
    CorpusManifestError,
    build_corpus_manifest,
    build_metadata,
    build_not_applicable_reproducibility_metadata,
    build_reproducibility_metadata,
    build_vectorstore_fingerprint,
    escape_markdown_table_cell,
    write_report,
)


def _make_dataset_file(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text('{"id": "a"}\n', encoding="utf-8")
    return path


class TestEscapeMarkdownTableCell:
    """M2_Phase_4_5_6_code_review_result.md 재리뷰 P2: 실패 사례 표에 들어가는
    자유 텍스트(질문, 오류 메시지, 임의의 route 문자열)가 pipe/backslash/줄바꿈을
    포함해도 표 열 구조를 깨뜨리지 않아야 한다."""

    def test_pipe_is_escaped(self):
        assert escape_markdown_table_cell("a | b") == "a \\| b"

    def test_backslash_is_escaped_before_pipe_so_no_double_escaping(self):
        # 원문에 이미 "\|"가 있으면 "\\|"(백슬래시 이스케이프 + 파이프 이스케이프)가
        # 돼야지, 파이프까지 다시 이스케이프해 "\\\\|"가 되면 안 된다.
        assert escape_markdown_table_cell("a\\|b") == "a\\\\\\|b"

    def test_newlines_and_repeated_whitespace_collapse_to_single_space(self):
        assert escape_markdown_table_cell("line1\nline2   line3") == "line1 line2 line3"

    def test_none_becomes_empty_string(self):
        assert escape_markdown_table_cell(None) == ""

    def test_non_string_value_is_stringified(self):
        assert escape_markdown_table_cell(404) == "404"


class TestBuildMetadata:
    def test_contains_required_m2_req_010_fields(self, tmp_path):
        dataset_path = _make_dataset_file(tmp_path)
        metadata = build_metadata(dataset_path, ["python", "-m", "evaluation.retrieval"], {})
        for key in [
            "schema_version",
            "generated_at_utc",
            "git_commit",
            "git_dirty",
            "dataset_path",
            "dataset_sha256",
            "python_version",
            "command",
            "embedding_model",
            "reranker_model",
            "ollama_model",
            "retrieval_config",
        ]:
            assert key in metadata

    def test_dataset_sha256_matches_actual_file_hash(self, tmp_path):
        dataset_path = _make_dataset_file(tmp_path)
        metadata = build_metadata(dataset_path, [], {})
        expected = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        assert metadata["dataset_sha256"] == expected

    def test_extra_dict_is_merged_in(self, tmp_path):
        dataset_path = _make_dataset_file(tmp_path)
        metadata = build_metadata(dataset_path, [], {"case_counts": {"total": 5}})
        assert metadata["case_counts"] == {"total": 5}

    def test_missing_dataset_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_metadata(tmp_path / "nope.jsonl", [], {})

    def test_retrieval_config_omits_inactive_pipeline_keys(self, tmp_path, monkeypatch):
        dataset_path = _make_dataset_file(tmp_path)
        monkeypatch.setattr(config, "USE_HYBRID_SEARCH", False)
        monkeypatch.setattr(config, "USE_MMR", False)
        monkeypatch.setattr(config, "USE_RERANKER", False)
        metadata = build_metadata(dataset_path, [], {})
        cfg = metadata["retrieval_config"]
        assert "bm25_top_k" not in cfg
        assert "mmr_k" not in cfg
        assert "reranker_top_k" not in cfg
        assert metadata["reranker_model"] is None

    def test_retrieval_config_includes_active_pipeline_keys(self, tmp_path, monkeypatch):
        dataset_path = _make_dataset_file(tmp_path)
        monkeypatch.setattr(config, "USE_HYBRID_SEARCH", True)
        monkeypatch.setattr(config, "USE_MMR", True)
        monkeypatch.setattr(config, "USE_RERANKER", True)
        metadata = build_metadata(dataset_path, [], {})
        cfg = metadata["retrieval_config"]
        assert "bm25_top_k" in cfg
        assert "mmr_k" in cfg
        assert "reranker_top_k" in cfg
        assert metadata["reranker_model"] is not None


class TestWriteReport:
    def test_creates_json_and_markdown_with_matching_stem(self, tmp_path):
        json_path, md_path = write_report({"generated_at_utc": "2026-01-01T00:00:00Z", "x": 1}, tmp_path, "retrieval")
        assert json_path.exists()
        assert md_path.exists()
        assert json_path.stem == md_path.stem
        assert json_path.name.startswith("retrieval_")
        assert json_path.suffix == ".json"
        assert md_path.suffix == ".md"

    def test_json_content_is_sorted_and_not_ascii_escaped(self, tmp_path):
        payload = {"b": 1, "a": "한글"}
        json_path, _ = write_report(payload, tmp_path, "answer")
        text = json_path.read_text(encoding="utf-8")
        assert text.index('"a"') < text.index('"b"')
        assert "한글" in text
        assert json.loads(text) == payload

    def test_markdown_lists_scalar_fields_and_notes_complex_ones(self, tmp_path):
        payload = {"generated_at_utc": "2026-01-01T00:00:00Z", "case_counts": {"total": 5}, "note": "ok"}
        _, md_path = write_report(payload, tmp_path, "routing")
        text = md_path.read_text(encoding="utf-8")
        assert "note" in text
        assert "case_counts" in text  # 안내 문구에 필드 이름은 언급됨

    def test_render_markdown_callback_overrides_default_renderer(self, tmp_path):
        """M2_Phase_4_5_6_code_review_result.md P1: evaluator가 render_markdown을
        넘기면 기본 스칼라 전용 렌더러 대신 그 콜백의 출력을 그대로 .md에 쓴다."""
        payload = {"generated_at_utc": "2026-01-01T00:00:00Z", "metrics": {"recall@10": 0.5}}
        _, md_path = write_report(
            payload, tmp_path, "retrieval", render_markdown=lambda p: f"# custom\nrecall@10={p['metrics']['recall@10']}"
        )
        text = md_path.read_text(encoding="utf-8")
        assert text == "# custom\nrecall@10=0.5"

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "a" / "b" / "reports"
        json_path, _ = write_report({}, nested, "baseline")
        assert json_path.exists()

    def test_consecutive_calls_with_same_name_do_not_overwrite(self, tmp_path):
        """M2_Phase3_code_review_result.md P1: 같은 name으로 짧은 간격 안에
        반복 호출해도 서로 다른 경로를 받고, 두 payload 모두 디스크에 남아야 한다
        (이전에는 초 단위 timestamp가 같으면 두 번째 호출이 첫 번째 결과를
        조용히 덮어썼다)."""
        json_path1, md_path1 = write_report({"run": 1}, tmp_path, "retrieval")
        json_path2, md_path2 = write_report({"run": 2}, tmp_path, "retrieval")

        assert json_path1 != json_path2
        assert md_path1 != md_path2
        assert json.loads(json_path1.read_text(encoding="utf-8")) == {"run": 1}
        assert json.loads(json_path2.read_text(encoding="utf-8")) == {"run": 2}

    def test_many_rapid_consecutive_calls_all_preserved(self, tmp_path):
        paths = [write_report({"run": i}, tmp_path, "retrieval") for i in range(20)]
        json_paths = [p[0] for p in paths]
        assert len(set(json_paths)) == 20
        for i, (json_path, _) in enumerate(paths):
            assert json.loads(json_path.read_text(encoding="utf-8")) == {"run": i}

    def test_existing_file_at_target_path_forces_retry(self, tmp_path, monkeypatch):
        """timestamp 정밀도와 무관하게, 동시 실행 등으로 정확히 같은 경로가 이미
        존재하는 상황도 배타적 생성 + suffix 재시도로 견뎌야 한다."""
        import datetime as datetime_module

        import evaluation.reporting as reporting_module

        fixed_now = datetime_module.datetime(2026, 1, 1, tzinfo=datetime_module.timezone.utc)

        class _FixedDatetime(datetime_module.datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_now

        monkeypatch.setattr(reporting_module, "datetime", _FixedDatetime)

        expected_timestamp = fixed_now.strftime("%Y%m%dT%H%M%S%fZ")
        # 미리 같은 이름의 파일을 만들어 첫 시도가 충돌하도록 강제한다.
        (tmp_path / f"retrieval_{expected_timestamp}.json").write_text("preexisting", encoding="utf-8")
        (tmp_path / f"retrieval_{expected_timestamp}.md").write_text("preexisting", encoding="utf-8")

        json_path, md_path = write_report({"run": "new"}, tmp_path, "retrieval")

        assert json_path.name == f"retrieval_{expected_timestamp}-1.json"
        assert md_path.name == f"retrieval_{expected_timestamp}-1.md"
        assert json.loads(json_path.read_text(encoding="utf-8")) == {"run": "new"}
        # 기존 파일은 그대로 보존돼야 한다.
        assert (tmp_path / f"retrieval_{expected_timestamp}.json").read_text(encoding="utf-8") == "preexisting"


class TestBuildCorpusManifest:
    def test_entries_and_manifest_sha256(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "a.pdf").write_bytes(b"hello")
        (data_dir / "b.txt").write_bytes(b"world")

        manifest = build_corpus_manifest(data_dir)
        source_ids = [e["source_id"] for e in manifest["entries"]]
        assert source_ids == sorted(source_ids)
        assert {"a.pdf", "b.txt"} == set(source_ids)
        for entry in manifest["entries"]:
            assert entry["size_bytes"] > 0
            assert len(entry["sha256"]) == 64

    def test_manifest_sha256_is_deterministic_for_same_content(self, tmp_path):
        for name in ("d1", "d2"):
            d = tmp_path / name
            d.mkdir()
            (d / "a.pdf").write_bytes(b"same content")
        m1 = build_corpus_manifest(tmp_path / "d1")
        m2 = build_corpus_manifest(tmp_path / "d2")
        assert m1["manifest_sha256"] == m2["manifest_sha256"]

    def test_manifest_sha256_changes_when_file_content_changes(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "a.pdf").write_bytes(b"version 1")
        m1 = build_corpus_manifest(data_dir)
        (data_dir / "a.pdf").write_bytes(b"version 2 (changed)")
        m2 = build_corpus_manifest(data_dir)
        assert m1["manifest_sha256"] != m2["manifest_sha256"]

    def test_manifest_reflects_added_file(self, tmp_path):
        """M2_Phase3_code_review_result.md P2: 파일 추가 시 manifest_sha256이
        바뀌고, entries의 source_id 차집합으로 추가된 파일을 식별할 수 있어야
        한다(개발 계획 §3.6)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "a.pdf").write_bytes(b"hello")
        before = build_corpus_manifest(data_dir)

        (data_dir / "b.txt").write_bytes(b"world")
        after = build_corpus_manifest(data_dir)

        assert after["manifest_sha256"] != before["manifest_sha256"]
        before_ids = {e["source_id"] for e in before["entries"]}
        after_ids = {e["source_id"] for e in after["entries"]}
        assert after_ids - before_ids == {"b.txt"}
        assert before_ids - after_ids == set()

    def test_manifest_reflects_removed_file(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "a.pdf").write_bytes(b"hello")
        (data_dir / "b.txt").write_bytes(b"world")
        before = build_corpus_manifest(data_dir)

        (data_dir / "b.txt").unlink()
        after = build_corpus_manifest(data_dir)

        assert after["manifest_sha256"] != before["manifest_sha256"]
        before_ids = {e["source_id"] for e in before["entries"]}
        after_ids = {e["source_id"] for e in after["entries"]}
        assert before_ids - after_ids == {"b.txt"}
        assert after_ids - before_ids == set()

    def test_content_change_identifiable_via_entries_sha256(self, tmp_path):
        """집계 hash(manifest_sha256)만으로는 어떤 파일이 바뀌었는지 알 수 없다는
        문제(개발 계획 §3.6 4차 리뷰 P1)를 entries 배열 자체로 확인한다."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "a.pdf").write_bytes(b"version 1")
        (data_dir / "b.txt").write_bytes(b"unchanged")
        before = {e["source_id"]: e for e in build_corpus_manifest(data_dir)["entries"]}

        (data_dir / "a.pdf").write_bytes(b"version 2 (changed)")
        after = {e["source_id"]: e for e in build_corpus_manifest(data_dir)["entries"]}

        assert before["a.pdf"]["sha256"] != after["a.pdf"]["sha256"]
        assert before["b.txt"]["sha256"] == after["b.txt"]["sha256"]

    def test_same_basename_different_subdirectories_raises(self, tmp_path):
        data_dir = tmp_path / "data"
        (data_dir / "sub1").mkdir(parents=True)
        (data_dir / "sub2").mkdir(parents=True)
        (data_dir / "sub1" / "x.pdf").write_bytes(b"v1")
        (data_dir / "sub2" / "x.pdf").write_bytes(b"v2")
        with pytest.raises(CorpusManifestError) as exc_info:
            build_corpus_manifest(data_dir)
        assert "x.pdf" in exc_info.value.collisions
        assert len(exc_info.value.collisions["x.pdf"]) == 2

    def test_case_only_difference_raises(self, tmp_path):
        # 같은 디렉터리에 대소문자만 다른 두 파일을 만들면 macOS(APFS 기본값 등)
        # 대소문자 비구분 파일시스템에서는 실제로 한 파일로 합쳐져 테스트 자체가
        # 무의미해진다 — 그래서 서로 다른 하위 디렉터리를 사용해 실제로 두 파일이
        # 디스크에 공존하게 하고, normalize_source_id()가 그 둘을 같은 source_id로
        # 정규화하는지만 검증한다(파일시스템 대소문자 구분 여부와 무관).
        data_dir = tmp_path / "data"
        (data_dir / "sub1").mkdir(parents=True)
        (data_dir / "sub2").mkdir(parents=True)
        (data_dir / "sub1" / "X.pdf").write_bytes(b"v1")
        (data_dir / "sub2" / "x.pdf").write_bytes(b"v2")
        with pytest.raises(CorpusManifestError):
            build_corpus_manifest(data_dir)

    def test_nfc_nfd_only_difference_raises(self, tmp_path):
        import unicodedata

        # 같은 이유로 서로 다른 하위 디렉터리를 사용한다 — macOS는 파일명을 저장 시점에
        # NFD로 정규화하는 경우가 있어 같은 디렉터리에서는 NFC/NFD 두 표기가 하나로
        # 합쳐질 수 있다.
        data_dir = tmp_path / "data"
        (data_dir / "sub1").mkdir(parents=True)
        (data_dir / "sub2").mkdir(parents=True)
        nfc_name = unicodedata.normalize("NFC", "가나다.pdf")
        nfd_name = unicodedata.normalize("NFD", "가나다.pdf")
        (data_dir / "sub1" / nfc_name).write_bytes(b"v1")
        (data_dir / "sub2" / nfd_name).write_bytes(b"v2")
        with pytest.raises(CorpusManifestError):
            build_corpus_manifest(data_dir)

    def test_missing_data_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_corpus_manifest(tmp_path / "nope")


class TestBuildVectorstoreFingerprint:
    def test_correct_sha256_for_both_files(self, tmp_path):
        (tmp_path / "index.faiss").write_bytes(b"faiss-bytes")
        (tmp_path / "index.pkl").write_bytes(b"pkl-bytes")
        fp = build_vectorstore_fingerprint(tmp_path)
        assert fp["index_faiss_sha256"] == hashlib.sha256(b"faiss-bytes").hexdigest()
        assert fp["index_pkl_sha256"] == hashlib.sha256(b"pkl-bytes").hexdigest()

    def test_missing_vectorstore_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_vectorstore_fingerprint(tmp_path / "nope")

    def test_missing_one_file_raises(self, tmp_path):
        (tmp_path / "index.faiss").write_bytes(b"only-faiss")
        with pytest.raises(FileNotFoundError):
            build_vectorstore_fingerprint(tmp_path)


class TestBuildReproducibilityMetadata:
    def _make_corpus_and_vectorstore(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "a.pdf").write_bytes(b"hello")
        vs_dir = tmp_path / "vectorstore"
        vs_dir.mkdir()
        (vs_dir / "index.faiss").write_bytes(b"faiss")
        (vs_dir / "index.pkl").write_bytes(b"pkl")
        return data_dir, vs_dir

    def test_returns_non_null_fields(self, tmp_path):
        data_dir, vs_dir = self._make_corpus_and_vectorstore(tmp_path)
        result = build_reproducibility_metadata(data_dir, vs_dir)
        assert result["corpus_manifest"] is not None
        assert result["corpus_manifest_sha256"] is not None
        assert result["vectorstore_fingerprint"] is not None
        assert result["reproducibility_note"] is None

    def test_missing_data_dir_raises(self, tmp_path):
        _, vs_dir = self._make_corpus_and_vectorstore(tmp_path)
        with pytest.raises(FileNotFoundError):
            build_reproducibility_metadata(tmp_path / "nope", vs_dir)

    def test_missing_vectorstore_raises(self, tmp_path):
        data_dir, _ = self._make_corpus_and_vectorstore(tmp_path)
        with pytest.raises(FileNotFoundError):
            build_reproducibility_metadata(data_dir, tmp_path / "nope")


class TestBuildNotApplicableReproducibilityMetadata:
    def test_returns_null_fields_with_reason(self):
        result = build_not_applicable_reproducibility_metadata("routing은 corpus/vectorstore를 사용하지 않음")
        assert result["corpus_manifest"] is None
        assert result["corpus_manifest_sha256"] is None
        assert result["vectorstore_fingerprint"] is None
        assert result["reproducibility_note"] == "routing은 corpus/vectorstore를 사용하지 않음"


class TestReportSchemaByEvaluatorKind:
    """개발 계획 §3.6: Retrieval/Answer 스타일 리포트는 재현성 필드가 모두 non-null,
    Routing 스타일 리포트는 모두 null + reproducibility_note가 채워져야 한다."""

    def test_retrieval_style_payload_has_non_null_reproducibility_fields(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "a.pdf").write_bytes(b"x")
        vs_dir = tmp_path / "vectorstore"
        vs_dir.mkdir()
        (vs_dir / "index.faiss").write_bytes(b"y")
        (vs_dir / "index.pkl").write_bytes(b"z")

        dataset_path = _make_dataset_file(tmp_path)
        payload = build_metadata(dataset_path, ["python", "-m", "evaluation.retrieval"], {})
        payload.update(build_reproducibility_metadata(data_dir, vs_dir))

        assert payload["corpus_manifest"] is not None
        assert payload["corpus_manifest_sha256"] is not None
        assert payload["vectorstore_fingerprint"] is not None
        assert payload["reproducibility_note"] is None

    def test_routing_style_payload_has_null_reproducibility_fields(self, tmp_path):
        dataset_path = _make_dataset_file(tmp_path)
        payload = build_metadata(dataset_path, ["python", "-m", "evaluation.routing"], {})
        payload.update(
            build_not_applicable_reproducibility_metadata("routing은 corpus/vectorstore를 사용하지 않음")
        )

        assert payload["corpus_manifest"] is None
        assert payload["corpus_manifest_sha256"] is None
        assert payload["vectorstore_fingerprint"] is None
        assert payload["reproducibility_note"] is not None
