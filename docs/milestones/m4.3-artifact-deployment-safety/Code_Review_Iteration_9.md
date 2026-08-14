# M4.3 Artifact & Deployment Safety — Code Review Iteration 9

Reviewer: Fresh Codex independent code review worker

Baseline: `84f6b407c9dd6d2de262c4d8f708618d11b37766`

Reviewed state: current uncommitted remediation relative to that baseline

Verdict: **PASS — 10.0/10** (`CRITICAL 0`, `MAJOR 0`, `MINOR 0`, `TRIVIAL 0`)

## Scope and conclusion

I independently reviewed the current uncommitted remediation for
`Code_Review_Iteration_8.md` finding CR-I8-MAJ-01 and the claims in
`Code_Review_Iteration_8_Remediation.md`. I did not edit implementation,
commit, push, merge, or invoke any Native Linux, Ollama, DDGS, protected/live,
self-hosted, or image gate.

CR-I8-MAJ-01 is closed. Ordinary Label binding remains byte-exact: the code
compares the unmodified Label with `_unicode_escape` renderings of individual
decoded Subject RDN values and contains no case-folding, whitespace stripping,
or Unicode normalization in that comparison. Compatibility is a dictionary
lookup by the exact certificate SHA-256 followed by exact string equality with
one Label; wrong digests, mutated legacy Labels, and case/leading/trailing-space
ordinary-Label mutations are rejected.

I independently walked all three documented compatibility boundaries using
the scanner's own strict grammar and certificate decoder: the venv top-level
`certifi==2026.07.22`, the venv pip-vendored `certifi==2025.10.05`, and the
repository-default Python's genuine pip-vendored `certifi==2023.07.22`. The
union of genuine Label deviations was exactly equal to the eight-entry
`_CERTIFICATE_LABEL_COMPATIBILITY` table: no required pair was absent and no
extra pair was present. The 2023.07.22 boundary contributes the five entries
identified in Iteration 8, while the complete bundle passes end to end under
the repository-default interpreter.

## Rechecked security and release properties

- Issuer, Subject, Serial, MD5, SHA1, SHA256, and Label remain bound to the
  exact immediately following certificate; grammar-valid metadata-smuggling
  probes remain rejected.
- `_decode_certificate()` retains secure `mkstemp` creation, explicit file
  descriptor ownership transfer, close-on-transfer-failure, and unlink in
  `finally`. Cleanup errors other than `FileNotFoundError` propagate, so the
  verification path remains fail-closed.
- Singleton retry tests retain strong object references and use `is`/`is not`;
  the successful returned engine is the exact successfully initialized
  candidate, not an integer-ID reuse artifact.
- Requirement, Plan, Traceability, Design, workflow, evidence assembler, and
  baseline checker have no diff from `84f6b407`. `M4.1_BLOCKED=true`, protected
  M3 live `NOT_RUN`/`SKIPPED`, `operational_status=BLOCKED`, and
  `overall_release_ready=false` therefore remain protected.

## Verification evidence

| Check | Result |
|---|---|
| `venv/bin/python -m pytest -q tests/unit/test_scan_image_layers.py tests/unit/test_rag_engine_singleton.py` | **PASS: 112 passed** |
| `python3 -m pytest -q tests/unit/test_scan_image_layers.py tests/unit/test_rag_engine_singleton.py` | **PASS: 112 passed**, 2 unrelated environment warnings |
| Independent three-bundle deviation enumeration | **PASS:** union exactly equals all 8 allowlist pairs |
| Compact wrong-digest, mutated-legacy-Label, and ordinary case/whitespace tests under venv Python | **PASS: 13 passed** |
| Same compact direct tests under repository-default `python3` | **PASS: 13 passed** |
| `git diff --check 84f6b407` before this report | **PASS** |
| Protected/live/self-hosted/image gates | **NOT RUN**, by scope |

## Gate

Severity count is `CRITICAL 0 / MAJOR 0 / MINOR 0 / TRIVIAL 0`; score is
**10.0/10**. The required PASS gate is `CRITICAL=0`, `MAJOR=0`, and score
`>=9.7`, so Code Review Iteration 9 is **PASS**.
