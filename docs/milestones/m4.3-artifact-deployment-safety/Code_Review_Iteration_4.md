# M4.3 Artifact & Deployment Safety — Code Review Iteration 4

Reviewer: Fresh Codex independent code review worker
Reviewed revision: `1e7fbac4ecda5217a2a315cb5e54621708624edb`
Exact range: `765bde38058b958ffc22e8459524f0bb7b23f0c2..1e7fbac4ecda5217a2a315cb5e54621708624edb`
Verdict: **PASS — 9.7/10** (`CRITICAL 0`, `MAJOR 0`, `MINOR 2`, `TRIVIAL 0`)

## Scope and conclusion

I fully read `milestone_dev_orchestration_guide.md`,
[Code Review Iteration 3](Code_Review_Iteration_3.md),
[Hosted CI Remediation Iteration 2](Hosted_CI_Remediation_Iteration_2.md),
[Implementation Report](Implementation_Report.md), and
[Traceability](Traceability.md), then independently inspected the exact eight-file
diff. Both Iteration 3 MAJOR findings are closed. The CA exception now requires a
genuine regular tar member, a trusted path, complete consumption of an ASCII byte
stream containing one or more certificate blocks and only permitted whitespace,
and successful OpenSSL X.509 parsing. Every non-regular trusted-path member remains
forbidden by the generic `.pem`/`.crt` rules.

The remediation does not alter the protected workflow or release-state logic.
`M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`, and
`overall_release_ready=false` remain mandatory. No Native Linux/Ollama/DDGS,
protected live, self-hosted runner, or environment-approval action was executed or
changed.

## Closure verification

### CR-I3-MAJ-01 — closed

`is_verified_ca_bundle()` first rejects non-ASCII, then applies a full-input grammar
whose only accepted language is one-or-more matching `CERTIFICATE` blocks separated
or surrounded by space, tab, CR, or LF. It finally passes the entire accepted text to
`SSLContext.load_verify_locations()`, so syntactically shaped but non-X.509 material
also fails. Independent inspection and focused execution confirmed rejection of
prepended, appended, and interleaved secret text; complete and incomplete private-key
material; mixed or mismatched labels; malformed delimiters/base64; zero blocks; and
non-ASCII bytes. A genuine certificate and a two-certificate whitespace-separated
bundle remain accepted.

This is strict full consumption, not a label search: any secret byte outside a valid
certificate body prevents the regular expression fullmatch. The subsequent OpenSSL
load is a real structural X.509 oracle rather than acceptance based only on base64
shape.

### CR-I3-MAJ-02 — closed

The scan entry point passes `TarInfo.isfile()` to `classify_member()`, and the CA
exception requires that value to be true before content is read or trusted. Symlink,
hardlink, character/block device, FIFO, and directory members therefore cannot receive
the exception; a trusted-looking `.pem` or `.crt` path falls through to the credential
patterns. This conservative policy never resolves or follows link targets, so dangling,
absolute, relative-traversal, cyclic, or untrusted targets cannot create a pass. The
new `.crt` pattern closes the equivalent suffix gap.

Layer traversal is still additive and independent. Whiteout entries themselves are
skipped, but an earlier-layer credential is retained in the violation history; duplicate
credentials and repeated whiteouts do not suppress it. Ordinary member traversal
continues to fail closed after normalization.

## Findings

### CR-I4-MIN-01 — the end-to-end hardlink regression has a non-specific oracle

In `tests/unit/test_scan_image_layers.py`,
`test_scan_flags_hardlink_bypass_end_to_end` asserts only that some credential exists
and `forbidden_count >= 1`. Its fixture also includes
`app/secrets/key.pem`, which independently satisfies the `.pem` rule. Consequently the
test would still pass if the trusted-looking hardlink itself were mistakenly exempted.
The direct hardlink unit test does assert the exact link classification and protects the
current implementation, so this is not a remaining bypass and is MINOR rather than
MAJOR. Strengthen the scan-level oracle to assert a violation whose `member` is exactly
`etc/ssl/certs/innocent.pem` (and ideally assert both expected member records).

### CR-I4-MIN-02 — the remediation range is not `git diff --check` clean

`git diff --check 765bde38058b958ffc22e8459524f0bb7b23f0c2..1e7fbac` reports trailing
whitespace on lines 3–5 and an extra blank line at EOF in
`Code_Review_Iteration_3.md`. These are documentation-only formatting defects and do
not affect scanner or CI behavior, but the exact reviewed commit does not satisfy the
stated whitespace check. This review does not modify that existing document.

## Container smoke and dependency lock

The hosted workflow invokes `python scripts/container_smoke.py` after installing the
locked dependencies and the editable project. In the project virtual environment, the
new repository-root `sys.path` insertion makes the bare script import
`tests.support.mock_ollama`; execution proceeds past the former
`ModuleNotFoundError` and reaches the expected `docker_run_failed` result for a
deliberately nonexistent image. The change is import wiring only and leaves all smoke
checks and `compute_all_ok()` unchanged.

One portability caveat was observed outside the controlled hosted environment: a
system interpreter with an unrelated installed regular package named `tests` can still
shadow this repository's namespace-package `tests` directory. That does not reproduce
in the locked project virtual environment used for the focused check, and hosted run
31609022196 had not yet reached the smoke step at review time, so it is not rated as a
hosted closure failure. A direct bare-script regression test or path-based support-module
import would make the guarantee environment-independent.

The lock diff changes only the generated header path plus
`charset-normalizer` 3.4.9 to 3.5.0 and its hashes. A clean linux/amd64
`python:3.11-slim` container with `uv==0.8.15` resolved 103 packages twice and
`compile_lock.sh --verify` passed with no drift.

## Verification evidence

| Check | Result |
|---|---|
| `pytest -q tests/unit/test_scan_image_layers.py tests/unit/test_container_smoke_contract.py tests/unit/test_dependency_lock.py tests/unit/test_dependency_snapshot.py` | **64 passed** |
| Project-venv bare `container_smoke.py --image nonexistent:image` | Former import error absent; expected `docker_run_failed` |
| linux/amd64 `python:3.11-slim`, `uv==0.8.15`, `compile_lock.sh --verify` | **PASS**, 103 packages, two identical resolutions |
| `git diff --check 765bde3..1e7fbac` | **FAIL**, documentation-only whitespace noted in CR-I4-MIN-02 |
| [Hosted run 31609022196](https://github.com/luminous419/simple-qna-rag/actions/runs/31609022196) | In progress at review time; lock verification already **SUCCESS**, protected M3 live **SKIPPED**, container smoke not yet reached |

## Gate

Severity count is `CRITICAL 0 / MAJOR 0 / MINOR 2 / TRIVIAL 0`; score is
**9.7/10**. The required gate is `CRITICAL=0`, `MAJOR=0`, and score >=9.7, so
Code Review Iteration 4 is **PASS**. This is a pre-merge code-quality result only;
it does not convert pending hosted evidence into PASS or change any operational/release
readiness state.
