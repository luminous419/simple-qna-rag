#!/usr/bin/env python3
"""CR-I2-MAJ-01 preflight — verify the two canonical vectorstore files exist
at the runner-local (checkout-external) path before the M3 live regression
gate reads them, and report their sha256 so a missing/mutated artifact fails
fast with a clear message instead of deep inside
`run_m4_regression_gate._vectorstore_fingerprint()`.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

CANONICAL_FILES = ("index.faiss", "index.pkl")


def check_vectorstore(vectorstore_dir: Path) -> dict[str, str]:
    """`vectorstore_dir`에 `CANONICAL_FILES`가 모두 존재하는지 확인하고 각
    파일의 sha256을 반환한다. 하나라도 없으면 `FileNotFoundError`를 던진다."""
    missing = []
    hashes: dict[str, str] = {}
    for name in CANONICAL_FILES:
        path = vectorstore_dir / name
        if not path.is_file():
            missing.append(str(path))
            continue
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if missing:
        raise FileNotFoundError(
            "vectorstore preflight 실패 — 누락된 canonical 파일: " + ", ".join(missing)
        )
    return hashes


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("사용법: preflight_vectorstore.py <vectorstore_dir>", file=sys.stderr)
        return 2

    vectorstore_dir = Path(argv[0])
    try:
        hashes = check_vectorstore(vectorstore_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for name in CANONICAL_FILES:
        print(f"{name} sha256={hashes[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
