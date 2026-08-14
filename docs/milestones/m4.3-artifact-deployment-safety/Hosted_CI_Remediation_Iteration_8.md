# M4.3 Artifact & Deployment Safety — Hosted CI Remediation Iteration 8

작성자: Claude Sonnet 5 (bounded remediation worker, iteration-cap reset로 승인됨)
대상: PR #18 (미push, 동일 브랜치 `agent/m4-3-artifact-deployment-safety`)
기준 리뷰: `Code_Review_Hosted_CI_Remediation_7.md` (FAIL — 9.3/10.0, CRITICAL 0 /
MAJOR 1 / MINOR 1)
기준 revision: `2418119`(Iteration 6 코드 변경 위) — **merge/commit/push는
수행하지 않는다** — fresh Codex 리뷰가 필요하다.

## 0. 범위와 결론 요약

이 iteration은 `Code_Review_Hosted_CI_Remediation_7.md`가 지적한 단일 MAJOR인
**CR-HCIR7-MAJ-01**(CR-HCIR6-MAJ-02의 잔여 결함)만 닫는다. MAJ-01(current
pointer permission fix, Iteration 7에서 이미 PASS)은 손대지 않았고,
CR-HCIR6-MIN-03(lock fixture)도 계속 명시적으로 제외한다.

CR-HCIR7-MAJ-01이 지적한 두 결함:

1. `_capture_container_logs()`가 `len(combined_bytes) <= max_bytes`
   early-return을 startup 선택보다 먼저 수행해, 반복되는 `startup` 이벤트가
   전체적으로는 예산 이하일 때(예: `startup` 2개, 70바이트, `max_bytes=1000`)
   dedup을 전혀 거치지 않고 둘 다 그대로 반환됐다 — "최신 1건만 보존" 계약이
   over-budget 경로에서만 적용되고 있었다.
2. `if startup and startup in tail` 검사는 tail 창이 `startup` 줄 **전체**를
   포함할 때만 중복을 제거했다. tail 창의 시작 바이트가 `startup` 줄
   **중간**에 걸리는 부분 overlap(review가 재현한
   `...startup"}\n_init_failed"}\nTAIL` 사례)에서는 겹치는 바이트가 그대로
   중복 출력됐다.

## 1. 수정 (`scripts/container_smoke.py::_capture_container_logs`)

### 1.1 결함 1 — under-budget dedup 누락

startup 줄 선택과 "최신 1건만 보존"을 byte-budget 조기 반환보다 **앞으로**
옮겼다. 이제 `combined`/`lines` 자체가 dedup된 상태에서만 budget 비교가
일어난다.

```python
lines = combined.splitlines(keepends=True)
startup_indices = [i for i, ln in enumerate(lines) if '"event": "startup"' in ln]
startup = lines[startup_indices[-1]] if startup_indices else ""
if len(startup_indices) > 1:
    keep = startup_indices[-1]
    lines = [ln for i, ln in enumerate(lines) if i not in startup_indices or i == keep]
    combined = "".join(lines)

combined_bytes = combined.encode("utf-8")
if len(combined_bytes) <= max_bytes:
    return combined
```

`startup_indices`(문자열 값이 아닌 위치 인덱스)로 유지·제거 대상을 구분해
동일한 텍스트를 가진 startup 줄이 여러 개여도 정확히 마지막 위치의 줄만
남긴다. 이 dedup은 이제 over-budget 여부와 무관하게 항상 먼저 수행되므로,
under-budget 로그에 반복 `startup`이 있어도 "최신 1건만 보존" 계약이 그대로
적용된다.

### 1.2 결함 2 — 부분 overlap 미제거

over-budget 경로의 tail 조립 로직을 byte-level 최대 suffix/prefix overlap
제거로 교체했다. 완전 포함(기존 계약)과 경계 부분 겹침(새 결함) 두 형태를
모두 다룬다.

```python
if startup_bytes and startup_bytes in tail_bytes:
    result_bytes = tail_bytes
else:
    overlap = 0
    max_check = min(len(startup_bytes), len(tail_bytes))
    for k in range(max_check, 0, -1):
        if startup_bytes[-k:] == tail_bytes[:k]:
            overlap = k
            break
    result_bytes = startup_bytes + tail_bytes[overlap:]
return result_bytes.decode("utf-8", errors="ignore")
```

- `startup_bytes in tail_bytes`(완전 포함, 기존 테스트가 요구하는 계약)이면
  `tail_bytes` 그대로 반환한다 — 이전 동작과 동일.
- 그렇지 않으면 `startup_bytes`의 suffix와 `tail_bytes`의 prefix가 일치하는
  최대 길이(`overlap`)를 바이트 단위로 찾아 그 만큼만 `tail_bytes`에서 잘라낸
  뒤 이어붙인다. overlap이 0이면(겹침 없음) 기존 `startup + tail` 동작과
  동일하다.
- 비교와 슬라이싱을 모두 **바이트 단위**로 수행한 뒤 결합 결과를 단 한 번만
  `errors="ignore"`로 decode한다 — 이전처럼 `startup`/`tail`을 각각 개별
  decode한 뒤 문자열로 비교하지 않으므로, 멀티바이트 절단 경계가 비교 자체를
  틀리게 만들 가능성이 구조적으로 사라진다.
- 예산 불변식은 그대로 유지된다: `len(result_bytes) = len(startup_bytes) +
  len(tail_bytes) - overlap <= len(startup_bytes) + len(tail_bytes) <=
  len(startup_bytes) + remaining = max_bytes` (overlap이 0이어도, 완전
  포함이어도, 부분 겹침이어도 항상 성립).

### 1.3 건드리지 않은 것

`docker logs --tail 200` 호출, 성공 경로에서 아무것도 채우지 않는 계약,
`max_bytes` 기본값(16000), `_run()`/예외 처리(`try/except Exception:
return ""`), 단일 oversized `startup` 이벤트를 head-preserve로 잘라내는
분기(§oversized), no-startup 시 plain tail로 폴백하는 분기는 전혀 바뀌지
않았다. MAJ-01(`resolve_current()`의 `EACCES` 번역과
`_load_vectorstore()`의 propagation, `src/simple_qna_rag/index/
verification.py`/`src/simple_qna_rag/rag_engine.py`)은 이번 iteration에서
diff가 없다.

## 2. 테스트 (`tests/unit/test_container_smoke_readiness_diagnostics.py`)

두 가지 mutation-strength 테스트를 추가했다. 각각 CR-HCIR7-MAJ-01이 제시한
반례를 그대로 재현하고, 이전 코드(§1의 수정 전 로직)에 대해 실제로 실패함을
별도 스크립트로 검증했다(§4.2).

- `test_capture_container_logs_dedupes_repeated_startup_events_even_under_budget`
  — `startup` 이벤트 2개(합계 70바이트), `max_bytes=1000`(review가 제시한
  정확한 반례). 결과에 `"event": "startup"`이 정확히 1회만 나타나고, 최신
  이벤트(`attempt=2`)만 남으며 이전 이벤트(`attempt=1`)는 완전히 사라져야
  함을 확인한다.
- `test_capture_container_logs_trims_partial_overlap_when_tail_starts_inside_startup_line`
  — lead noise로 전체 로그를 예산 초과 상태로 만든 뒤, `remaining`을
  `len(trail_noise) < remaining < len(startup_bytes) + len(trail_noise)`
  범위로 정확히 구성해 tail 창의 시작 지점이 `startup` 줄 중간(끝에서
  15바이트 지점)에 걸리도록 강제한다. 결과가 `startup_line + trail_noise`와
  정확히 일치해야 함을 assert한다 — overlap이 제거되지 않으면 review가
  재현한 것과 동일한 중복 fragment(`..._init_failed"}\n`)가 끼어들어 이
  등호 비교가 실패한다.

기존 계약을 유지하는지 확인하기 위해 기존 6개 테스트(멀티바이트,
oversized-single-startup, repeated-startup-over-budget,
avoids-duplicating-startup-in-tail-window, combines/truncates/no-startup
fallback/never-raises)를 그대로 재실행했고 전부 통과했다(회귀 없음, §4.1).

## 3. 닫힘 매핑(exact closure mapping)

| 발견 | 파일 | 수정 | 테스트 |
|---|---|---|---|
| CR-HCIR7-MAJ-01 (결함 1: under-budget dedup 누락) | `scripts/container_smoke.py` — `_capture_container_logs()` | startup 선택/dedup을 byte-budget 조기 반환보다 앞으로 이동 | `test_capture_container_logs_dedupes_repeated_startup_events_even_under_budget` |
| CR-HCIR7-MAJ-01 (결함 2: 부분 overlap 미제거) | `scripts/container_smoke.py` — `_capture_container_logs()` | byte-level 최대 suffix/prefix overlap 계산 후 tail에서 제거 | `test_capture_container_logs_trims_partial_overlap_when_tail_starts_inside_startup_line` |
| MAJ-01(current pointer EACCES, Iteration 7 PASS) | — | 이번 iteration에서 diff 없음(작업 지시에 따라 범위 제외) | — |
| CR-HCIR6-MIN-03(lock fixture) | — | 이번 iteration 범위에서 **명시적으로 제외**(작업 지시) | — |

CR-HCIR7-MAJ-01이 지적한 두 결함 모두 CRITICAL/MAJOR 재발 없이 정확히 리뷰가
요구한 범위(startup selection을 size early-return 이전으로, partial overlap
포함 dedup)로 닫혔다고 판단한다.

## 4. 재검증 결과

### 4.1 테스트/정적 검증

- 대상 파일 focused 테스트: `venv/bin/python -m pytest -q
  tests/unit/test_container_smoke_readiness_diagnostics.py
  tests/unit/test_index_verification.py tests/unit/test_rag_engine_singleton.py
  tests/integration/test_health_endpoints.py` — **58 passed**(신규 2건 포함,
  회귀 없음).
- 전체 로컬 결정론적 suite: `venv/bin/python -m pytest -q` —
  **1328 passed, 1 skipped, 4 warnings in 167.44s**. Iteration 7 리뷰 기준
  `1326 passed, 1 skipped` 대비 정확히 신규 2건(§2) 순수 추가 — 회귀 없음.
- `python scripts/generate_field_spec.py --check`: exit 0(변경 없음 — 이번
  iteration은 필드 스펙에 영향을 주는 로깅 스키마를 건드리지 않았다).
- `python scripts/logging_callsite_audit.py --check`: exit 0(변경 없음 —
  `docs/generated/logging_callsite_disposition.json`은 이전 세션의 Iteration 6
  작업에서 이미 갱신된 상태 그대로다).
- `python scripts/check_markdown_links.py`: 검사 파일 141개(tracked 137 +
  untracked 4), 링크 597개, 실패 0개.
- protected 경계(`git diff --exit-code -- .github/workflows/ci.yml
  scripts/scan_image_layers.py scripts/assemble_m4_evidence.py
  scripts/check_m4_baseline.py evaluation/baselines/m3_initial.*
  requirements.lock requirements.txt`): exit 0(변경 없음).

### 4.2 mutation-strength 독립 검증

새 두 테스트가 실제로 CR-HCIR7-MAJ-01의 결함을 잡아내는지, 수정 전 로직(§1
이전, `Hosted_CI_Remediation_Iteration_7.md`가 서술한 그대로의 코드)을 별도
스크립트로 재현해 동일한 assertion을 돌려 확인했다.

```text
test1 old-code startup count: 2 (기대: 1) -> 구 코드에서 실패 확인
test2 old-code result: '{"event": "startup", "reason": "engine_init_failed"}\n_init_failed"}\nTAIL\n'
       expected:        '{"event": "startup", "reason": "engine_init_failed"}\nTAIL\n'
       -> 구 코드에서 실패 확인(review가 보고한 것과 동일한 중복 fragment)
Both tests correctly fail against old buggy logic (mutation-strength confirmed).
```

두 테스트 모두 수정 전 코드에서는 실패하고 수정 후 코드에서는 통과함을
확인했다 — 리뷰가 요구한 mutation-strength 기준을 만족한다.

### 4.3 실제 컨테이너 검증

- `docker build --platform linux/amd64 --target production -f
  deploy/Dockerfile -t simple-qna-rag:iter8-repro .`: 성공(대부분 레이어
  캐시 hit — 이번 iteration은 `deploy/Dockerfile`이나 의존성을 건드리지
  않았다).
- `venv/bin/python scripts/container_smoke.py --image
  simple-qna-rag:iter8-repro`(실제 docker, 보안 플래그·negative control
  전부 포함, macOS `linux/amd64` 에뮬레이션):

```json
{
  "status": "PASS",
  "readiness_sequence": {
    "live": true, "ready": true,
    "ready_last_http_status": 200, "ready_last_reason": "ok",
    "ready_poll_elapsed_seconds": 6.07
  },
  "production_test_seam_sealed": true,
  "production_test_seam_seal_last_http_status": 503,
  "production_test_seam_seal_last_reason": "artifact_test_embedding_seam_unavailable",
  "root_page_ok": true, "mock_query_ok": true, "static_asset_ok": true
}
```

  6개 boolean 전부 true, `container_log_tail` 키 없음(성공 경로에서
  `_capture_container_logs()`가 아예 호출되지 않는 계약 그대로 유지) — §1의
  재작성이 정상 경로와 negative control 경로 어디에도 회귀를 일으키지
  않았다. 컨테이너/이미지 정리 후 leftover 컨테이너 없음을 확인했다.
- Native Linux/Ollama/DDGS/live/self-hosted, protected M3/M4.1 live gate,
  environment 승인 경계는 이 remediation에서도 실행·변경하지 않았다.
  `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`/workflow `SKIPPED`,
  `overall_release_ready=false`는 코드 diff가 닿지 않는 경로이므로 그대로
  보존된다.

## 5. 이 remediation이 건드리지 않은 것

`_poll_ready`의 `max_seconds`, `evaluate_readiness()`의 분기 순서/우선순위,
negative control의 판정 임계값, `compute_all_ok`/`_ALL_OK_KEYS`,
`resolve_current()`의 `EACCES`/`ENOENT`/`ELOOP`/기타 errno 처리와
`_load_vectorstore()`의 `TrustBoundaryError` propagation(Iteration 7에서
이미 PASS 판정, MAJ-01), `expected_owner_uid` 검사, dirfd/`O_NOFOLLOW` 체인,
immutable-member 계약, trust-before-pickle 동작, `docker logs --tail 200`
호출 자체, `_capture_container_logs`의 `max_bytes` 기본값(16000)과 실패
시에만 채우는 계약, oversized-single-startup 분기의 head-preserve
동작(§1.3), `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`,
`overall_release_ready=false` 산출 경로, `m3-live-regression-gate` 블록,
`.github/workflows/ci.yml`, `scripts/scan_image_layers.py`/
`assemble_m4_evidence.py`/`check_m4_baseline.py`, `requirements.lock`/
`requirements.txt`, CPU torch extra-index 시맨틱, `--generate-hashes`
계약, uv 0.8.15 고정, CR-HCIR6-MIN-03(lock fixture 강화 — 이번 작업
지시에 따라 명시적으로 제외), Native Linux/Ollama/DDGS/live/self-hosted
승인 경계, 위 결함과 무관한 어떤 파일도 수정하지 않았다.

## 6. 남은 작업

이 커밋은 아직 push되지 않았다. **merge/commit/push는 수행하지 않는다** —
fresh Codex 리뷰가 필요하다(작업 지시). 리뷰가 PASS하면 diagnostic
commit/push와 hosted 재실행이 다음 단계이며, hosted 성공 여부는 별도로
평가한다. CR-HCIR6-MIN-03은 이 리뷰가 PASS한 이후 별도 승인을 받아 후속
iteration으로 진행할 수 있다.
