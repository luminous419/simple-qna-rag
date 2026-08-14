# M4.3 Hosted CI Remediation Iteration 6 — Fresh Codex Code Review

Reviewer: Codex (independent dispatched review)  
Baseline: `2418119`  
Reviewed working-tree scope: all 13 changed paths plus `Hosted_CI_Remediation_Iteration_6.md`  
Hosted evidence under diagnosis: run `31804369490`

## Verdict

**FAIL — 8.8 / 10.0**

- CRITICAL: **0**
- MAJOR: **2**
- MINOR: **1**
- Pre-merge code-quality Gate: **FAIL**
- Diagnostic commit/push Gate: **FAIL until the two bounded MAJOR remediations below are reviewed**
- Post-merge/hosted acceptance Gate: **NOT READY / NOT RUN for this working tree**

The proposed `versions/` mode correction is technically sound for completed publications and does not weaken member ownership, immutable-member, contained-open, or trust-before-pickle enforcement. The hosted root-cause attribution remains a strong but unconfirmed hypothesis: run `31804369490` proves a deterministic generic `engine_init_failed`, not that its underlying errno was `EACCES`; the next hosted run is still required after the code defects below are closed. The change cannot pass because one raw-`EACCES` route remains and the evidence tail is demonstrably not bounded by its advertised byte limit.

## Findings

### CR-HCIR6-MAJ-01 — `resolve_current()` still leaks raw `EACCES`, so the translation is not exhaustive

`resolve_current()` opens `INDEX_ROOT/current` directly with `os.open(..., dir_fd=root.fd)`. Its handler translates only `ENOENT` and `ELOOP`; `EACCES` is re-raised as a raw `PermissionError`. This is on the normal engine initialization path before `verify_version()`, and therefore reproduces the same collapse described by Iteration 6: `_load_vectorstore()` does not receive a `TrustBoundaryError`, `RAGEngine.initialize()` swallows the raw exception, and readiness remains the opaque `engine_init_failed`.

The new matrix covers `open_contained_root`, `open_subdir`, and `open_member`, but never the direct current-pointer open. Thus the report's “three entry points” model is incomplete. The three added translations themselves are bounded and semantically appropriate, and they do not weaken the dirfd/`O_NOFOLLOW` chain; the defect is the omitted fourth access.

**Exact bounded remediation:** in `resolve_current()` translate `errno.EACCES` from the direct `current` open to a new specific bounded reason such as `current_pointer_permission_denied`; add that literal to `REASONS`; add a mutation-strength test that patches the exact `os.open("current", ...)` call to raise `PermissionError(errno.EACCES, ...)` and asserts the typed reason propagates through the engine/artifact readiness chain. Do not broaden translation to unrelated errnos and do not change fallback-on-`ENOENT` semantics.

### CR-HCIR6-MAJ-02 — startup retention is neither byte-bounded nor always character-bounded

`_capture_container_logs()` still measures Python characters (`len(str)`) rather than encoded bytes. More seriously, it concatenates every matching startup line before calculating the remainder; if the preserved lines alone exceed `max_bytes`, it returns all of them unchanged. Independent deterministic probes produced:

```text
multibyte:       100 characters / 300 UTF-8 bytes with max_bytes=100
startup_overflow: 230 characters / 230 bytes with max_bytes=100
duplicate_startup: 270 characters / 270 bytes with max_bytes=100
```

This directly violates the prior approved review's explicit byte-bound cleanup requirement and Iteration 6's claim that the 16,000-byte cap is preserved. It also permits repeated or unusually long startup records to enlarge uploaded evidence and stderr beyond the contract. The existing positive test uses one short ASCII startup line, so it cannot detect either mutation.

**Exact bounded remediation:** operate on UTF-8 bytes; select at most one diagnostically relevant startup event (prefer the latest), truncate it and the recent tail within one strict `max_bytes` total, avoid duplicating overlapping bytes, and decode with a deterministic error policy. Add tests for multibyte text, a single startup line larger than the budget, repeated startup lines, and an overlap case, each asserting `len(result.encode("utf-8")) <= max_bytes` while retaining startup evidence when it can fit.

### CR-HCIR6-MIN-03 — lock seeding tests prove the copy, not resolver durability or the upgrade path

The implementation preserves uv `0.8.15`, exact header normalization, hashes, `--extra-index-url`, `unsafe-best-match`, and two-run body comparison. Seeding both scratch outputs from the committed lock is consistent with uv's existing-output preference and the reported hosted-equivalent Linux result is coherent. However, the new fake-uv tests only record that bytes existed before fake uv overwrote them; they do not prove that an unrequested transitive release remains pinned, that an incompatible direct constraint is re-resolved, or that the documented intentional removal/full-regeneration path works with real uv. Editing a hashed stanza by hand is also a poor documented upgrade path; removal or an explicit upgrade mode is the safe path.

**Bounded follow-up:** add a Linux/uv-0.8.15 deterministic fixture test (local package index or otherwise network-independent) with two versions of a transitive package: prove default rerun retains the committed version, a changed direct constraint resolves the compatible version, and a deliberate stanza removal or explicit upgrade option advances it with regenerated hashes. This is a MINOR because the supplied linux/amd64 reproduction supports the production behavior, but the regression test currently has weak mutation strength.

## Security and compatibility assessment

`os.chmod(index_root / "versions", 0o755)` after publication resolves cross-UID traversal for the completed tree even under an ambient `0o077` umask. It grants no write bit to group/other; published version directories remain `0o555`, members remain `0o444`, `expected_owner_uid` is still checked on opened members, and pickle construction still consumes only previously opened, hashed bytes. The change does not replace dirfd-relative opens, `O_NOFOLLOW`, `fstat`, hash validation, or trust-before-pickle. A crash before the new chmod may leave an incomplete restrictive tree, but no successful publication is returned before chmod/fsync completes, so that is not a weakening or a false-success path.

`engine_error_type` is optional only on the internal `startup` event, is derived as `type(exc).__name__`, and is revalidated against an anchored 64-character identifier grammar. Invalid values clamp to `unknown`; exception messages, traceback, paths, and URLs are not logged. `/health/ready` remains exactly the bounded public reason response, and the focused integration test verifies that a secret-bearing `PermissionError` message is absent. No readiness priority or public status changed.

The generated logging disposition change is justified: all four edits are line-number shifts caused by the allowlist/validator additions; callsite identities and classifications do not change, and both generated checks pass.

The protected surfaces are untouched by the diff: `.github/workflows/ci.yml`, producer assembly and baseline scripts, baseline artifacts, `requirements.txt`, and `requirements.lock` have no changes. `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`/`SKIPPED` semantics, `operational_status=BLOCKED`, and `overall_release_ready=false` remain intact. No Native Linux/Ollama/DDGS/live/self-hosted/environment approval gate was executed or changed in this review.

## Evidence reviewed and checks run

The run `31804369490` evidence and Iterations 5/6 form a coherent diagnostic chain: the service was live, readiness repeatedly returned `503 engine_init_failed`, and the old tail retained health traffic while evicting the earlier startup event. That evidence supports observability remediation and an EACCES hypothesis, but cannot retroactively prove the hidden exception type. Likewise, the reported container smoke PASS and linux/amd64 lock verification are plausible local deterministic evidence, not hosted proof for the unpushed working tree.

```text
venv/bin/python -m pytest -q
1320 passed, 1 skipped, 4 warnings in 171.44s

focused changed-area suite
127 passed, 3 warnings in 11.41s

independent log-bound counterexamples
3/3 reproduced the advertised bound violation
```

The full-suite count exactly matches the claimed `1320 passed, 1 skipped`. The claimed local container smoke PASS, deterministic positive exit 0, and negative exit 1 evidence is internally consistent with the unchanged five semantic booleans and exact negative reason, but was not rerun here because the full deterministic Python suite plus direct counterexamples were sufficient to decide this FAIL. No excluded gate, commit, push, PR mutation, or merge was performed.

## Gate classification and next step

This is a **pre-merge FAIL**. Fix only MAJ-01 and MAJ-02 with the bounded changes and mutation tests stated above, regenerate the logging callsite artifact only if line movement requires it, rerun the deterministic suite, and obtain a fresh independent code review. If that review passes, one diagnostic commit/push and hosted rerun may be authorized; hosted success must then be evaluated separately, while M4.1/protected-live blockers and `overall_release_ready=false` remain unchanged.
