# M4.3 Recovery Runbook

Diagnosis and rollback/backup-restore procedures for the M4.3 canonical
index lifecycle ([Design.md](../milestones/m4.3-artifact-deployment-safety/Design.md) §7.6).

## 1. Diagnostic table

| Symptom | Diagnostic command | Expected result | Recovery |
|---|---|---|---|
| readiness 503 `artifact_manifest_schema_invalid` | `simple-qna-rag-index-lifecycle verify --version <current>` | exit 1, `error_code` set | Roll back to previous version (§3 below) |
| readiness 503 `artifact_member_hash_mismatch` | same | exit 1 | Roll back; investigate storage/transfer corruption separately |
| readiness 503 `artifact_settings_mismatch` | same | exit 1, `error_code=settings_mismatch` | Confirm the intended embedding/chunk settings for this version; either fix settings or activate the version actually built for them |
| readiness 503 `artifact_test_embedding_seam_unavailable` | none (immediate) | — | Production is behaving correctly — the test seam is not physically present in this image; ensure `SIMPLE_QNA_RAG_EMBEDDING_PROVIDER`/`ALLOW_TEST_EMBEDDING` are unset in production |
| `activate`/`rollback` exit 3 | none (immediate) | `error_code=lock_timeout` | Check for a concurrently running lifecycle process before retrying |
| `build`/`import-legacy` fails mid-way (disk full) | `df -h <index-root mount>` | free space confirmed low | Free disk space; the `.staging/<op>` remnant is inactive and harmless — it is not automatically deleted. Clean it up explicitly and only after confirming no `activate`/`build` is in flight: `simple-qna-rag-index-lifecycle cleanup --apply --include-staging --index-root <root>` |
| Ollama outage | `python scripts/preflight_ollama.py <url> <model>` | connection failure reported | Restore Ollama connectivity; unrelated to index activation state |
| container start fails (read-only violation) | `docker logs <container>` | permission denied on a write path | Re-check volume owner/mode (deployment runbook §2) |

## 2. Rollback procedure

```text
1. Stop traffic to this instance, or confirm internal load-balancer/health-check exclusion.
2. Confirm the previous image digest and index version from the pre-deploy
   snapshot (deployment_runbook.md §6).
3. simple-qna-rag-index-lifecycle rollback --to-version <previous-id> --index-root <root>
   (if `verify` inside this call fails, STOP here — do not proceed further)
4. docker run <previous image digest>
5. Confirm readiness. If it fails, escalate via the operations channel —
   do not attempt further automatic mutation.
```
`rollback --to-previous` derives the previous version from
`activation_history/` automatically; prefer the explicit `--to-version` form
during an incident so the target is never ambiguous.

## 3. Backup / restore

Backup the immutable version directory (manifest included) as a unit:
```bash
tar czf backup-<version-id>.tar.gz -C <index-root>/versions <version-id>
```

Restore never activates untrusted pickle bytes directly. Extract into a
fresh staging directory, then let the normal publish path re-verify hash and
schema before the version is eligible for `activate`:
```bash
mkdir -p <index-root>/.staging/<new-op-id>
tar xzf backup-<version-id>.tar.gz -C <index-root>/.staging/<new-op-id> --strip-components=1
# the extracted files are re-verified by the same hash/schema check
# `verify_version()` uses; only a version that passes is eligible to
# `activate`. Restoring a version that fails verification means the backup
# itself is untrusted — do not force-activate it.
```

## 4. Escalation

If any verification step above fails a second time after the indicated
recovery action, stop and escalate to the operations channel with:
- the exact command and receipt/error_code observed,
- the current and previous `version_id`,
- the deployment identity snapshot from `deployment_runbook.md` §6.

Do not attempt additional mutation while escalation is pending — every
lifecycle mutation in this design is fail-closed and leaves `current`
unchanged on failure, so waiting is always safe.
