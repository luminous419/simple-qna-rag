# M4.3 Artifact & Deployment Safety — Code Review Iteration 6

Reviewer: Fresh Codex independent code review worker

Reviewed revision: `84f6b407c9dd6d2de262c4d8f708618d11b37766`

Exact range: `30419791a4bf984984ee191190e6ee8b2225b3f0..84f6b407c9dd6d2de262c4d8f708618d11b37766`

PR: #18

Verdict: **FAIL — 9.4/10** (`CRITICAL 0`, `MAJOR 1`, `MINOR 1`, `TRIVIAL 0`)

## Scope and conclusion

I fully read `milestone_dev_orchestration_guide.md` and the requested M4.3
Requirement, Plan, Traceability, Design, Code Review Iteration 5, Iteration 5
Remediation, and Implementation Report documents, then independently reviewed
the exact committed/pushed range above. I did not edit implementation code,
merge, or execute any Native Linux/Ollama/DDGS/live/self-hosted gate.

CR-I5-MAJ-02 is closed in code: a failed construction clears both singleton
layers, every `initialize()` attempt resets its artifact reason, the dedicated
`EngineArtifactError` constructor accepts exactly
`index_verification.REASONS`, and the server classifies only that exception
type. The regression tests cover artifact-to-ordinary, artifact-to-success,
ordinary-to-success, constructor allowlisting, per-attempt reset, and an
arbitrary reason-bearing server exception.

CR-I5-MAJ-01 is not closed. The seven fields are now structurally exact and
ordered and Serial/fingerprints have bounded grammars, but Issuer/Subject still
accept the very token, path, key-value, and private-material payload classes
the remediation claims to reject. The tests avoid those grammar-valid forms by
choosing a colon for Issuer/Subject, so they do not establish the advertised
field-specific non-smuggling property.

## Findings

### CR-I6-MAJ-01 — Issuer/Subject grammar still permits adversarial smuggling

**Severity:** MAJOR (pre-merge security scanner correctness; CR-I5-MAJ-01
remains open).

`scripts/scan_image_layers.py::_ISSUER_SUBJECT_VALUE` is
`[A-Za-z0-9 ()/,._=\\-]{1,256}`. That alphabet necessarily permits free-form
spaces, slash paths, underscores, and repeated `=` assignments. With all other
six genuine fields and the genuine certificate unchanged, direct deterministic
probes produced:

```text
Issuer `CN=API_TOKEN=supersecret`              -> True
Issuer `CN=../../etc/shadow`                   -> True
Issuer `CN=PRIVATE KEY`                        -> True
Issuer `CN=AWS_SECRET_ACCESS_KEY=ABCDEF`       -> True
```

These are not merely theoretical alternate DNs: they are exactly token,
path, key-value, and private-material payload classes named by the required
remediation. OpenSSL validates only the following certificate and does not bind
the comment metadata to it, so the valid X.509 block cannot authenticate these
Issuer/Subject strings.

The parametrized adversarial test uses `token: ...` and `Authorization: ...`
for Issuer/Subject. Those cases fail only because colon is excluded; the test
does not try grammar-valid equivalents using `=`/`/`/spaces. Thus it would pass
while the bypasses above remain open.

**Required fix:** do not treat an independently parseable DN-looking comment
as authenticated metadata. Parse the certificate and require the seven stanza
values (at least Issuer, Subject, Serial, and all three fingerprints) to equal
values derived from that exact certificate, with a canonical rendering policy;
alternatively discard comments and validate only fully consumed certificate
blocks if comment retention is unnecessary. Add the four exact probes above
for both Issuer and Subject, plus equivalent Label payloads that remain inside
its accepted alphabet. Retain the full installed certifi bundle and genuine
pip-vendored underscore positive fixtures.

### CR-I6-MIN-01 — fresh-identity oracle compares recyclable integer IDs

**Severity:** MINOR (test reliability).

`test_artifact_then_retry_success_uses_fresh_identity` and the ordinary-failure
variant retain only `id(self)` for the failed candidate. CPython may recycle an
address immediately after the failed candidate is released, so a genuinely
fresh object can receive the same integer ID and make this regression test
flaky. Retain the actual candidate objects in the spy and assert object
identity with `is not`; this also makes the oracle express the contract
directly. The implementation itself does clear `_instance` and `_initialized`,
so this is a test defect, not a reproduced singleton defect.

## Other reviewed properties

- `_CERTIFI_STANZA` names exactly seven fields in the required order; omission,
  duplication, reordering, and an eighth field are structurally rejected.
- Serial is decimal-only and bounded to 48 digits. MD5, SHA1, and SHA256 are
  lowercase colon-hex with exact 16/20/32-byte lengths.
- The pip-vendored Entrust 2048 fixture contains the genuine `CPS_2048`
  underscore shape and a complete certificate. The report claims it was copied
  from the built image, but no immutable image digest or raw scan/smoke receipt
  is committed, so that local-image provenance claim is not independently
  replayable from this range. Hosted CI cannot repair the grammar bypass.
- Failure clears `RAGEngine._instance`, `RAGEngine._initialized`, and leaves the
  module `_rag_engine` unset before raising. A subsequent successful attempt is
  cached normally.
- `EngineArtifactError` rejects every value outside the exact public set and
  the server's general exception branch no longer reads `.reason`.
- The range stays within the stated scanner/engine/server/tests/documentation
  remediation boundary and does not modify OCI union/link resolution or CI
  approval logic.

## Verification evidence

| Check | Result |
|---|---|
| `pytest -q tests/unit/test_scan_image_layers.py tests/unit/test_rag_engine_singleton.py tests/integration/test_health_endpoints.py` | **90 passed** |
| Grammar-valid Issuer adversarial probes | **Bypass reproduced:** all four returned `True` |
| `python scripts/check_markdown_links.py` | **PASS**, 122 files / 566 links / 0 failures |
| `git diff --check 30419791..84f6b407` | **PASS** |
| `gh pr checks 18` during review | frontend PASS; python/container/M4.3 deterministic pending; protected M3 live SKIPPED |

The full unit/integration suite was also started as a deterministic local check;
the focused 90-test result above is the completed, attributable regression
receipt used for this verdict. No protected or live test was invoked.

## Release-state and gate

`M4.1_BLOCKED=true` remains preserved. Protected M3 live remains
`NOT_RUN`/`SKIPPED`, and `overall_release_ready=false`; this review does not
alter any of those states.

Severity count is `CRITICAL 0 / MAJOR 1 / MINOR 1 / TRIVIAL 0`; score is
**9.4/10**. The required gate is `CRITICAL=0`, `MAJOR=0`, and score >=9.7, so
Code Review Iteration 6 is **FAIL**. One narrowly scoped remediation iteration
is allowed: close CR-I6-MAJ-01 with certificate-bound metadata validation and
repair the identity oracle in CR-I6-MIN-01, then obtain a fresh independent
review. No merge is allowed on this result.
