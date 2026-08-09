"""CR-I2-MAJ-01 — vectorstore preflight canonical 2-file existence/hash check."""

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import preflight_vectorstore as pv  # noqa: E402


def test_check_vectorstore_returns_sha256_for_both_canonical_files(tmp_path):
    faiss = tmp_path / "index.faiss"
    pkl = tmp_path / "index.pkl"
    faiss.write_bytes(b"faiss-bytes")
    pkl.write_bytes(b"pkl-bytes")

    result = pv.check_vectorstore(tmp_path)

    assert result == {
        "index.faiss": hashlib.sha256(b"faiss-bytes").hexdigest(),
        "index.pkl": hashlib.sha256(b"pkl-bytes").hexdigest(),
    }


def test_check_vectorstore_raises_when_index_faiss_missing(tmp_path):
    (tmp_path / "index.pkl").write_bytes(b"pkl-bytes")

    with pytest.raises(FileNotFoundError, match="index.faiss"):
        pv.check_vectorstore(tmp_path)


def test_check_vectorstore_raises_when_index_pkl_missing(tmp_path):
    (tmp_path / "index.faiss").write_bytes(b"faiss-bytes")

    with pytest.raises(FileNotFoundError, match="index.pkl"):
        pv.check_vectorstore(tmp_path)


def test_check_vectorstore_raises_when_both_missing(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        pv.check_vectorstore(tmp_path)
    assert "index.faiss" in str(exc_info.value)
    assert "index.pkl" in str(exc_info.value)


def test_main_exit_0_and_prints_hashes_when_both_present(tmp_path, capsys):
    (tmp_path / "index.faiss").write_bytes(b"a")
    (tmp_path / "index.pkl").write_bytes(b"b")

    exit_code = pv.main([str(tmp_path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "index.faiss sha256=" in out
    assert "index.pkl sha256=" in out


def test_main_exit_1_when_missing(tmp_path, capsys):
    exit_code = pv.main([str(tmp_path)])

    assert exit_code == 1
    assert "preflight 실패" in capsys.readouterr().err


def test_main_exit_2_on_wrong_arg_count():
    assert pv.main([]) == 2
    assert pv.main(["a", "b"]) == 2
