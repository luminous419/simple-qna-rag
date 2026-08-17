# Internal Deployment and Observability Plan

## 1. Purpose and release boundary

This plan turns the verified M4 hosted/OCI release into a controlled internal
deployment and establishes the first comparable operating baseline. The
certified surface is the hosted Python/frontend service and the OCI image.
Native Linux/Ollama, self-hosted runners, protected live gates, and public
internet exposure remain `NOT_ADOPTED`; this plan neither runs nor reclassifies
them. Consequently `hosted_release_ready` may be true while
`native_linux_release_ready`, `full_production_release_ready`, and
`overall_release_ready` remain false.

The initial evidence is split deliberately:

1. [M4 hosted/OCI initial baseline](../../evaluation/baselines/m4_hosted_oci_initial.md)
   is the immutable exact-SHA CI/OCI and container-smoke baseline.
2. Real-traffic operating values are populated only after an internal
   deployment completes the observation window in section 5. CI smoke timings
   must not be presented as user-query latency or availability.

## 2. Deployment unit and ownership

Deploy by immutable OCI digest, never by a mutable tag. The operator records
the merge SHA, workflow run, OCI image ID, index version, settings hash, and
dependency-lock hash in one deployment record. Follow the normative
[deployment runbook](deployment_runbook.md), including its exact-run baseline
verification, and use the [recovery runbook](recovery_runbook.md) for rollback.

The initial topology is one internal instance with the M4.2 defaults:
`concurrency_limit=1`, a bounded queue, a read-only application container, and
a separately managed read-only index mount. TLS, authentication, and
user-specific rate limiting are required at a reverse proxy before any traffic
outside the trusted internal network is allowed.

## 3. Promotion sequence

1. Select a successful `push` CI run whose `headSha` equals the intended
   `master` commit. Download `m4-baseline` and verify it with
   `check_m4_baseline.py --require-identity-binding` exactly as documented in
   the deployment runbook.
2. Pull or import the production image and pin its runtime OCI digest. Do not
   confuse that digest with `m4-baseline.image_digest`, which is the SHA-256 of
   the container-smoke evidence payload used by the assembler.
3. Snapshot the old image digest, index version, settings hash, dependency lock
   hash, and readiness state. Verify and activate the intended index version.
4. Start the container with the read-only, non-root, dropped-capability flags
   from the deployment runbook. Poll `/health/live` and `/health/ready` with
   bounded backoff; then issue one known-corpus smoke request.
5. Hold at one instance and default concurrency for the first observation
   window. Do not tune concurrency during baseline collection.
6. If readiness fails, identity drifts unexpectedly, or the smoke request
   fails, stop promotion and execute the explicit rollback procedure.

## 4. Collection contract

Collect process-local `/metrics`, structured JSON logs, health probes, and the
deployment identity record. Preserve only bounded labels; never add question,
answer, document content, authorization values, or unrestricted URLs as metric
labels or log fields.

| Signal | Source | Baseline field / calculation |
|---|---|---|
| request volume and HTTP result | `rag_requests_total` | rate and status/route counts |
| request latency | `rag_request_duration_seconds` | p50, p95, p99 per supported route |
| stage latency/errors | `rag_stage_duration_seconds`, `rag_stage_errors_total` | p95 and error count by bounded stage/code |
| terminal result | `rag_query_outcomes_total` | success, timeout, cancellation, internal-error counts |
| admission pressure | `rag_admission_rejected_total`, `rag_queue_depth` | overload/not-ready count and maximum queue depth |
| malformed/large input | `rag_input_rejected_total` | count by bounded rejection reason |
| fallback behavior | `rag_fallback_total` | count by bounded route/reason |
| logging contract failure | `logging_dropped_fields_total` | delta; investigate every non-zero delta |
| health | `/health/live`, `/health/ready` | success ratio and readiness reason changes |
| resources | container runtime | CPU and memory working-set p50/p95/max |

Counters are converted to deltas using the first and last scrape of the same
process lifetime. A restart starts a new segment; never subtract counters
across a reset. Histograms must be aggregated without averaging percentiles.

## 5. First real-traffic observation window

Use at least seven consecutive calendar days and at least 100 accepted RAG
queries. If either condition is not met, label the result `INSUFFICIENT_DATA`
and extend collection; do not manufacture a representative baseline. Record
zero-traffic intervals and every restart/deployment so selection bias is
visible.

The first window is descriptive, not a new release certification. Use these
initial guardrails for intervention:

- any readiness failure lasting more than 60 seconds: investigate and pause
  promotion;
- any internal-error response or `logging_dropped_fields_total` increase:
  investigate before declaring the window complete;
- sustained queue saturation or overload rejection: retain concurrency 1 and
  open a capacity finding rather than tuning ad hoc;
- identity mismatch or artifact verification failure: immediate rollback.

At window close, write a dated operating-baseline JSON and Markdown summary
next to `evaluation/baselines/m4_hosted_oci_initial.*`. Include the exact
deployment identity, UTC interval, request count, restart count, metric
summaries, missing-data reasons, and incidents. Secrets and raw user content
must not be copied into baseline artifacts.

## 6. Decision after the window

Keep the M4 hosted/OCI configuration when the window is stable. Create a
targeted corrective milestone for a repeated operational defect. Start an M5
candidate only when measured document volume, traffic, resource pressure, or
repeated user demand demonstrates the corresponding need. Native Linux/Ollama
remains outside this decision and is not a prerequisite for hosted/OCI
operation.
