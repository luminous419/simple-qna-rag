# M4.3 Artifact & Deployment Safety — Code Review Iteration 3

Reviewer: Fresh Codex independent code review worker
Reviewed revision: `765bde38058b958ffc22e8459524f0bb7b23f0c2`
Exact range: `5b91840699ad268ab27cbe05ad8ad4a8bb1957d9..765bde38058b958ffc22e8459524f0bb7b23f0c2`
Verdict: **FAIL — 8.8/10** (`CRITICAL 0`, `MAJOR 2`, `MINOR 0`, `TRIVIAL 0`)

## Scope and conclusion

I read `milestone_dev_orchestration_guide.md`, the M4.3
`Code_Review_Iteration_2.md`, `Acceptance_Report.md`,
`Hosted_CI_Remediation_Iteration_1.md`, `Implementation_Report.md`, and
`Traceability.md`, and independently reviewed the exact six-file remediation diff.
The Linux lock remediation is correct and narrowly scoped: a linux/amd64 Python 3.11
container resolves 103 packages and `compile_lock.sh --verify` passes, while the lock
diff changes only the generated-output path plus the expected releases of `langsmith`
0.10.17→0.10.18, `sqlalchemy` 2.0.51→2.0.52, and `typing-inspection`
0.4.3→0.4.4 with their hashes.

The OCI scanner remediation does not meet its stated strict path-and-content contract.
Two independently exploitable allowlist gaps remain: arbitrary non-PEM secret bytes can
be appended to a valid certificate, and trusted-looking symlink/hardlink members bypass
content and target validation entirely. Both are MAJOR because they let forbidden image
material evade the release-blocking scanner; therefore the >=9.7 PASS threshold is not
met.

## Findings

### CR-I3-MAJ-01 — `is_verified_ca_bundle` accepts a certificate plus arbitrary secret data

`is_verified_ca_bundle()` finds only `BEGIN` labels and checks that every label it found
is `CERTIFICATE`, then delegates the whole string to
`SSLContext.load_verify_locations()`. OpenSSL accepts valid certificate PEM surrounded
or followed by unrelated text, so this is not a strict “pure certificate blocks only”
parser. Direct probes against the reviewed revision returned `True` for each of:

```text
REAL_CA_CERT + "API_TOKEN=supersecret\n"
REAL_CA_CERT + "not-a-pem secret payload\n"
REAL_CA_CERT + "-----END PRIVATE KEY-----\n"
```

Consequently an attacker can place a valid public CA followed by an application secret
under an allowlisted `.pem`/`.crt` path and receive no violation. Existing mixed-bundle
coverage tests only a second recognized `BEGIN RSA PRIVATE KEY` block; it does not test
prefix/suffix junk, unmatched end markers, secret/key-value text, comments/whitespace
policy, or exact full-input consumption.

Required remediation: parse the entire ASCII input into an explicitly defined sequence
of complete, matching `BEGIN CERTIFICATE`/`END CERTIFICATE` blocks (permitting only the
minimal separators intentionally supported), reject every unconsumed byte or mismatched
delimiter, and then structurally validate every certificate. Add negative tests for
secret text before/between/after certificates, unmatched/mismatched delimiters, and a
valid certificate followed by private-key material that lacks a matching `BEGIN` line.

### CR-I3-MAJ-02 — symlink and hardlink candidates are allowed by path alone

`scan()` passes both `TarInfo.issym()` and `TarInfo.islnk()` as `is_symlink=True`, and
`classify_member()` immediately returns clean for either kind at a trusted-looking path.
It never examines `member.linkname`, normalizes the target, requires an allowlisted
target, resolves the referenced member, or verifies certificate bytes. A layer can
therefore contain `app/secrets/key.pem` plus an allowlisted hardlink such as
`etc/ssl/certs/innocent.pem`, or an allowlisted symlink aimed at an arbitrary app secret,
and the link entry itself is exempt despite the stated requirement that both narrow path
and strict certificate parsing are mandatory. The direct contract probe
`classify_member("etc/ssl/certs/innocent.pem", is_symlink=True)` returned `None`.

The new positive test fixes only the expected `/usr/lib/ssl/cert.pem` symlink shape and
contains no adversarial targets. There are no tests for relative/absolute traversal,
normalization, dangling links, trusted-to-untrusted links, link cycles, hardlinks, or
links to private-key/malformed/mixed content.

Required remediation: remove the path-only exemption. Either conservatively classify
all `.pem`/`.crt` links as credentials, or resolve links within the layer under a bounded,
cycle-safe, normalized policy and require both the final target path and referenced bytes
to satisfy the same CA constraints. Add the adversarial symlink/hardlink matrix above and
prove failures remain closed when a target cannot be resolved or read.

## Other reviewed properties

- Path normalization for ordinary members rejects the tested
  `etc/ssl/certs/../../../app/secret.pem` shape after normalization; no ordinary-file
  prefix bypass was found.
- Private-key PEM with a recognized `BEGIN` label, malformed certificate PEM, a real CA
  outside a trusted path, `.env`, and `.pfx` remain rejected by the focused suite.
- Per-layer scanning still detects a credential in an earlier layer even when a later
  layer whiteouts it; the additive/deletion-history invariant is preserved.
- The exact diff does not change workflow approval, protected live execution, assembler,
  or release-state code. `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN` (the hosted job
  is skipped), and `overall_release_ready=false` remain mandatory and unchanged.

## Verification evidence

| Check | Result |
|---|---|
| linux/amd64 `python:3.11-slim`, `uv==0.8.15`, `bash scripts/compile_lock.sh --verify` | **PASS**, 103 packages, no drift |
| Host-native macOS lock verify | Expected platform mismatch: 102 packages and drift; not used as the Linux gate |
| `pytest -q tests/unit/test_scan_image_layers.py tests/unit/test_dependency_lock.py tests/unit/test_dependency_snapshot.py` | **30 passed** |
| Adversarial appended-secret and link probes | **Bypasses reproduced** as described above |
| `git diff --check 5b918406..765bde3` | **PASS** |
| [Hosted CI run 31606183756](https://github.com/luminous419/simple-qna-rag/actions/runs/31606183756) at review time | In progress; Python lock reproducibility step **SUCCESS**, frontend **SUCCESS**, container build/scan pending, protected M3 live **SKIPPED** |

No Native Linux/Ollama/DDGS, protected M3 live, M4.1 live, self-hosted runner, or
environment approval action was executed or changed.

## Gate

Severity count is `CRITICAL 0 / MAJOR 2 / MINOR 0 / TRIVIAL 0`; score is **8.8/10**.
The code-quality gate requires `CRITICAL=0`, `MAJOR=0`, and score >=9.7, so Iteration 3
is **FAIL**. The Linux lock hosted failure is closed, but the OCI scanner failure is not
safely closed until both bypasses and their negative regression tests are addressed.
