# M4.3 Deployment Runbook

Applies to the `deploy/Dockerfile` `production` stage image and the
`INDEX_ROOT` canonical version layout introduced by M4.3 ([Design.md](../milestones/m4.3-artifact-deployment-safety/Design.md)
§7). Native Linux/Ollama execution is an operator's manual responsibility —
this cycle does not run it and does not certify it.

## 1. Preflight

1. Confirm the image digest you intend to deploy (do not deploy by mutable
   tag alone):
   ```bash
   docker pull <registry>/<image>@sha256:<digest>
   ```
2. Confirm current index pointer/settings identity on the target host:
   ```bash
   simple-qna-rag-index-lifecycle list --index-root <host-index-root>
   ```
3. Ollama/model preflight is an **operator manual procedure**, not executed
   by this cycle:
   ```bash
   python scripts/preflight_ollama.py <ollama-base-url> <model-name>
   ```

## 2. Volume owner/mode

```bash
chown -R 10001:10001 <host-index-root>
find <host-index-root> -type d -exec chmod 0555 {} \;
```
(New/unpublished index roots start writable; the `chmod 0555` above only
applies cleanly once at least one version has been published and the
directory tree is otherwise stable — do not run it against a root with an
`activate`/`build` still in flight.)

## 3. Index verify/activate

```bash
simple-qna-rag-index-lifecycle verify --version <version-id> --index-root <host-index-root>
simple-qna-rag-index-lifecycle activate --to-version <version-id> --index-root <host-index-root>
```
`verify` must exit 0 before `activate` is attempted. Both commands print a
canonical JSON receipt; treat any `exit_code != 0` as "do not proceed."

## 4. Restart

```bash
docker compose restart app   # or: systemctl restart simple-qna-rag
```
Poll readiness with bounded exponential backoff (recommended: 1s, 2s, 4s,
8s, 16s, then stop and escalate):
```bash
curl -sf http://localhost:8000/health/ready
```

## 5. Smoke

```bash
curl -s -X POST http://localhost:8000/rag \
  -H 'Content-Type: application/json' \
  -d '{"question": "<a known corpus question>"}'
```
Expect HTTP 200 and a non-empty `answer` field.

## 6. Release identity record

Record the following snapshot **before** and **after** deployment, and
confirm both snapshots describe the same release identity:

| Field | Before | After |
|---|---|---|
| `image_digest` | `docker inspect --format '{{.Id}}' <image>` | |
| `current_version_id` | `simple-qna-rag-index-lifecycle list` `current` | |
| `settings_hash` | from the lifecycle receipt's `identity.settings_hash` | |
| `dependency_lock_sha256` | `python scripts/dependency_snapshot.py` | |

A deployment that changes any of these fields unexpectedly (i.e. not the
field you intended to change) should be treated as a deployment defect and
rolled back per [recovery_runbook.md](recovery_runbook.md).

## 7. Standard container run flags

```bash
docker run --rm \
  --user 10001:10001 \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  -v "<host-index-root>:/app/runtime/index:ro" \
  -e SIMPLE_QNA_RAG_INDEX_ROOT=/app/runtime/index \
  -p 8000:8000 \
  <image>
```
Index mutation (`build`/`import-legacy`/`activate`/`rollback`/`cleanup`) is
never performed inside this read-only container — always run the lifecycle
CLI on the host or in a separate writable-volume operator container against
the same `<host-index-root>`.
