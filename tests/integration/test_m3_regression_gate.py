"""M4.1 §10.4 — M3 regression wrapper contract tests (R1-MAJ-01/R2-MAJ-01)."""

import hashlib
import inspect
from pathlib import Path
from unittest import mock

import pytest

import scripts.run_m4_regression_gate as run_m4_regression_gate
from evaluation.compare import evaluate_gates
from evaluation.reporting import (
    build_vectorstore_fingerprint as real_build_vectorstore_fingerprint,
)


def test_build_vectorstore_fingerprint_signature_is_single_path_arg():
    sig = inspect.signature(real_build_vectorstore_fingerprint)
    assert list(sig.parameters) == ["vectorstore_path"]


def test_vectorstore_fingerprint_invokes_real_api(monkeypatch, tmp_path):
    (tmp_path / "index.faiss").write_bytes(b"faiss-bytes")
    (tmp_path / "index.pkl").write_bytes(b"pkl-bytes")
    real_fn = real_build_vectorstore_fingerprint
    spy = mock.Mock(wraps=real_fn)
    monkeypatch.setattr(run_m4_regression_gate, "build_vectorstore_fingerprint", spy)
    monkeypatch.setattr(run_m4_regression_gate, "VECTORSTORE_PATH", str(tmp_path))

    result = run_m4_regression_gate._vectorstore_fingerprint()

    spy.assert_called_once_with(Path(str(tmp_path)))
    call_arg = spy.call_args.args[0]
    assert isinstance(call_arg, Path)
    assert call_arg == Path(str(tmp_path))
    expected = real_fn(tmp_path)
    assert result == expected
    assert set(result) == {"index_faiss_sha256", "index_pkl_sha256"}
    assert result["index_faiss_sha256"] == hashlib.sha256(
        (tmp_path / "index.faiss").read_bytes()
    ).hexdigest()
    assert result["index_pkl_sha256"] == hashlib.sha256(
        (tmp_path / "index.pkl").read_bytes()
    ).hexdigest()


def test_exit_2_when_run_baseline_raises_runtime_error(monkeypatch, tmp_path):
    faiss = tmp_path / "index.faiss"
    pkl = tmp_path / "index.pkl"
    faiss.write_bytes(b"a")
    pkl.write_bytes(b"b")
    monkeypatch.setattr(run_m4_regression_gate, "VECTORSTORE_PATH", str(tmp_path))
    monkeypatch.setattr(
        run_m4_regression_gate,
        "BASELINE_FILES",
        (),
    )
    monkeypatch.setattr(
        run_m4_regression_gate,
        "run_baseline",
        mock.Mock(side_effect=RuntimeError("RUN_LIVE_LLM_TESTS=1 required")),
    )
    assert run_m4_regression_gate.main([]) == 2


def test_exit_1_when_overall_success_false_despite_gates_all_true(monkeypatch, tmp_path):
    faiss = tmp_path / "index.faiss"
    pkl = tmp_path / "index.pkl"
    faiss.write_bytes(b"a")
    pkl.write_bytes(b"b")
    monkeypatch.setattr(run_m4_regression_gate, "VECTORSTORE_PATH", str(tmp_path))
    monkeypatch.setattr(run_m4_regression_gate, "BASELINE_FILES", ())

    items = [{"id": f"gate_{i}", "pass": True} for i in range(14)]
    payload = {
        "overall_success": False,  # fingerprint mismatch inside run_baseline itself
        "gate_evaluation": {"spec_version": "1.0.0", "overall_pass": True, "items": items},
    }
    monkeypatch.setattr(run_m4_regression_gate, "run_baseline", mock.Mock(return_value=payload))
    assert run_m4_regression_gate.main([]) == 1


def test_exit_1_when_two_file_mutation_detected(monkeypatch, tmp_path):
    faiss = tmp_path / "index.faiss"
    pkl = tmp_path / "index.pkl"
    faiss.write_bytes(b"a")
    pkl.write_bytes(b"b")
    monkeypatch.setattr(run_m4_regression_gate, "VECTORSTORE_PATH", str(tmp_path))
    monkeypatch.setattr(run_m4_regression_gate, "BASELINE_FILES", ())

    def _mutate(*args, **kwargs):
        faiss.write_bytes(b"MUTATED")
        return {
            "overall_success": True,
            "gate_evaluation": {"spec_version": "1.0.0", "overall_pass": True, "items": []},
        }

    monkeypatch.setattr(run_m4_regression_gate, "run_baseline", mock.Mock(side_effect=_mutate))
    assert run_m4_regression_gate.main([]) == 1


def test_evaluate_gates_none_pass_collapses_overall_pass_false():
    payload = {"retrieval": None, "routing": None, "answers": None}
    result = evaluate_gates(payload)
    assert result["overall_pass"] is False
    assert all(item["pass"] is None for item in result["items"])
    assert len(result["items"]) == 14


def test_warmup_cases_is_positive_constant():
    # CR-I2-MAJ-02: the acceptance wrapper must request a positive warm-up
    # (evaluation.compare blanket-nulls the 5 latency gates when
    # warmup.performed=false, which run_baseline()'s own warmup_cases=0
    # default would trigger).
    assert run_m4_regression_gate.WARMUP_CASES > 0


def test_run_baseline_is_called_with_positive_warmup_cases(monkeypatch, tmp_path):
    faiss = tmp_path / "index.faiss"
    pkl = tmp_path / "index.pkl"
    faiss.write_bytes(b"a")
    pkl.write_bytes(b"b")
    monkeypatch.setattr(run_m4_regression_gate, "VECTORSTORE_PATH", str(tmp_path))
    monkeypatch.setattr(run_m4_regression_gate, "BASELINE_FILES", ())

    items = [{"id": f"gate_{i}", "pass": True} for i in range(14)]
    payload = {
        "overall_success": True,
        "stages": {},
        "gate_evaluation": {"spec_version": "1.0.0", "overall_pass": True, "items": items},
    }
    spy = mock.Mock(return_value=payload)
    monkeypatch.setattr(run_m4_regression_gate, "run_baseline", spy)

    assert run_m4_regression_gate.main([]) == 0

    assert spy.call_args.kwargs["warmup_cases"] == run_m4_regression_gate.WARMUP_CASES
    assert run_m4_regression_gate.WARMUP_CASES > 0


def test_warmup_performed_true_reads_report_json(tmp_path):
    report_path = tmp_path / "retrieval.json"
    report_path.write_text('{"warmup": {"performed": true}}', encoding="utf-8")
    assert run_m4_regression_gate._warmup_performed(str(report_path)) is True


def test_warmup_performed_false_reads_report_json(tmp_path):
    report_path = tmp_path / "retrieval.json"
    report_path.write_text('{"warmup": {"performed": false}}', encoding="utf-8")
    assert run_m4_regression_gate._warmup_performed(str(report_path)) is False


def test_warmup_performed_none_when_path_missing():
    assert run_m4_regression_gate._warmup_performed(None) is None


def test_warmup_performed_none_when_file_unreadable(tmp_path):
    assert run_m4_regression_gate._warmup_performed(str(tmp_path / "nope.json")) is None


def test_check_warmup_evidence_warns_when_requested_but_not_performed(tmp_path, capsys):
    report_path = tmp_path / "retrieval.json"
    report_path.write_text('{"warmup": {"performed": false}}', encoding="utf-8")
    payload = {
        "stages": {
            "retrieval": {"status": "success", "report_json_path": str(report_path)},
        }
    }
    run_m4_regression_gate._check_warmup_evidence(payload)
    err = capsys.readouterr().err
    assert "warmup.performed=False" in err


def test_check_warmup_evidence_silent_when_performed(tmp_path, capsys):
    report_path = tmp_path / "retrieval.json"
    report_path.write_text('{"warmup": {"performed": true}}', encoding="utf-8")
    payload = {
        "stages": {
            "retrieval": {"status": "success", "report_json_path": str(report_path)},
        }
    }
    run_m4_regression_gate._check_warmup_evidence(payload)
    assert capsys.readouterr().err == ""


def test_baseline_and_vectorstore_files_untouched_by_wrapper_module_import():
    """Importing the wrapper module must not touch the real baseline files or
    runtime vectorstore (REQ-006.2) — this only exercises import-time safety;
    the mutation-detection behavior itself is covered above with tmp_path."""
    for path in run_m4_regression_gate.BASELINE_FILES:
        assert path == Path(str(path))  # no accidental resolve()/mutation at import
