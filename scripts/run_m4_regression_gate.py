#!/usr/bin/env python3
"""M4.1 §10.1 — M3 14-gate regression wrapper (thin, no assembly logic).

Reuses `evaluation.baseline.run_baseline()` (which already assembles
retrieval/routing/answers -> `evaluate_gates()` -> `write_report()`) instead
of re-implementing gate assembly. The only new logic here is: (1) resolving
the vectorstore path through the legacy facade `config.VECTORSTORE_PATH`
(`str`) and normalizing to `Path` at a single call boundary
(`_vectorstore_fingerprint()`), (2) verifying the M3 baseline files and
runtime vectorstore are byte-identical before and after the run
(REQ-006.2), and (3) requesting a positive same-process warm-up
(CR-I2-MAJ-02) so `evaluation.compare._retrieval_gate_inputs()`/
`_answers_gate_inputs()` don't blanket-null the 5 latency gates on
`warmup.performed=false` — `run_baseline()`'s own default (`warmup_cases=0`)
exists for callers that don't need the latency gates (e.g. quick smoke runs)
and is intentionally left unchanged; this wrapper is the one call site that
promises Requirement §4's 14/14-gate acceptance evidence, so it must opt in
explicitly rather than inherit that default.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.baseline import run_baseline
from evaluation.reporting import build_vectorstore_fingerprint
from simple_qna_rag.config import VECTORSTORE_PATH  # legacy facade, str (Design.md §4.2)

BASELINE_FILES = (
    Path("evaluation/baselines/m3_initial.json"),
    Path("evaluation/baselines/m3_initial.md"),
)

# CR-I2-MAJ-02: must be a positive int — a 14/14-gate acceptance run without a
# warm-up leaves the 5 latency gates permanently `pass=None` (see module
# docstring). 3 matches the M3 final/final-approved acceptance runs'
# `--warmup-cases 3`.
WARMUP_CASES = 3
assert WARMUP_CASES > 0, f"WARMUP_CASES must be positive to exercise the latency gates: {WARMUP_CASES!r}"


def _hash_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vectorstore_fingerprint() -> dict:
    # M5-01: str(legacy facade) -> Path normalization happens exactly once,
    # inside this helper.
    return build_vectorstore_fingerprint(Path(VECTORSTORE_PATH))


def _warmup_performed(report_json_path: str | None) -> bool | None:
    """CR-I2-MAJ-02: read back a stage's own report JSON and return its
    `warmup.performed` flag — `None` if the path is missing or unreadable.
    `payload["stages"][name]` itself doesn't carry the warmup block (only
    summary fields), so this is the only way to verify from the wrapper's
    return value that a *requested* positive warm-up was *actually*
    performed, rather than trusting the request alone."""
    if not report_json_path:
        return None
    try:
        report = json.loads(Path(report_json_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return (report.get("warmup") or {}).get("performed")


def _check_warmup_evidence(payload: dict) -> None:
    for stage_name in ("retrieval", "answers"):
        stage = (payload.get("stages") or {}).get(stage_name) or {}
        if stage.get("status") != "success":
            continue
        performed = _warmup_performed(stage.get("report_json_path"))
        if performed is not True:
            print(
                f"경고: warmup_cases={WARMUP_CASES}(양수)를 요청했지만 {stage_name} 단계 리포트의 "
                f"warmup.performed={performed!r}입니다 — 관련 latency gate가 UNKNOWN(None)으로 "
                "판정되어 overall_pass가 False가 될 수 있습니다.",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    pre_baseline = {p: _hash_bytes(p) for p in BASELINE_FILES}
    pre_vectorstore = _vectorstore_fingerprint()
    immutable_ok = False
    try:
        payload = run_baseline(
            dataset_path=Path("evaluation/datasets/golden.jsonl"),
            output_dir=Path("evaluation/reports/m4_regression"),
            warmup_cases=WARMUP_CASES,
        )
    except RuntimeError as exc:  # RUN_LIVE_LLM_TESTS=1 opt-in missing
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        post_baseline = {p: _hash_bytes(p) for p in BASELINE_FILES}
        post_vectorstore = _vectorstore_fingerprint()
        immutable_ok = pre_baseline == post_baseline and pre_vectorstore == post_vectorstore

    if not immutable_ok:
        print("baseline/vectorstore mutated during regression run", file=sys.stderr)
        return 1

    _check_warmup_evidence(payload)

    overall_success = payload["overall_success"]
    overall_pass = payload["gate_evaluation"]["overall_pass"]
    return 0 if (overall_success is True and overall_pass is True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
