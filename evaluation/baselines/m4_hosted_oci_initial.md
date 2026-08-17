# M4 Hosted/OCI Initial Operating Baseline

Captured on 2026-08-16 KST from the exact-`master` push workflow for commit
`07ebf2cb673a788d5e4338535dbfb22cdee4c0b9`.

## Result

**PASS — hosted CI/OCI pre-deployment baseline.** GitHub Actions run
[`31892502554`](https://github.com/luminous419/simple-qna-rag/actions/runs/31892502554)
completed successfully. Its downloaded `m4-baseline` passed the repository's
identity-bound checker for the exact SHA, run ID, attempt, workflow path, and
`push` event.

This is not a real-user traffic baseline. No internal traffic observation
window has run, so request latency, error rate, throughput, and resource
percentiles remain `NOT_COLLECTED`. Collection and promotion rules are defined
in the [internal deployment and observability plan](../../docs/operations/internal_deployment_observability_plan.md).

## Identity and readiness

| Field | Value |
|---|---|
| Git SHA | `07ebf2cb673a788d5e4338535dbfb22cdee4c0b9` |
| Workflow run / attempt | `31892502554` / `1` |
| Event | `push` |
| `m4-baseline.json` SHA-256 | `1a1f8c49d7374fe13f266b1ed8ad3a48fdf6ed58b3899d576f546840db66ec28` |
| deterministic status | `PASS` |
| hosted release ready | `true` |
| native Linux release ready | `false` |
| full production / overall ready | `false` / `false` |
| native Linux/Ollama and protected live gates | `NOT_ADOPTED` / not run |

## Hosted and OCI observations

The complete workflow took 1,039 seconds. Job elapsed times were Python tests
503 seconds, frontend tests 12 seconds, container 212 seconds, M4.3
deterministic 962 seconds, and evidence assembly 72 seconds; parallel jobs
therefore do not sum to workflow elapsed time.

The production container smoke reached readiness in 5.01 seconds with HTTP
200, passed the mock query, root-page, and static-asset checks, and stopped
gracefully in 1.51 seconds. The production test seam was sealed and the layer
scan found zero forbidden members. The runtime OCI image ID observed by the
smoke was
`sha256:4d521c5060bdca3b74e45a7dfc2d091ccbd7572b7864caebc21fd20a5087ce29`.

The M4 baseline field named `image_digest` is not that runtime OCI image ID;
it is the SHA-256 of `container_smoke.json` used in the evidence manifest:
`ca6be00ecf11109594db401bc018ee46a3125eb7714d10e7e507656555d7a2d1`.
Keeping these identities separate prevents an evidence payload hash from being
mistaken for a deployable registry digest.

## Next observation

After an internal deployment, collect at least seven consecutive calendar days
and 100 accepted RAG queries. Until both minima are met, the real-traffic
baseline remains `INSUFFICIENT_DATA` or `NOT_COLLECTED`; CI smoke values must
not be substituted for service latency or availability.
