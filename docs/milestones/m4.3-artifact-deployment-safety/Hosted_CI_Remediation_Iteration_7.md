# M4.3 Artifact & Deployment Safety — Hosted CI Remediation Iteration 7

작성자: Claude Sonnet 5 (bounded remediation worker, iteration-cap reset로 승인됨)
대상: PR #18 (미push, 동일 브랜치 `agent/m4-3-artifact-deployment-safety`)
기준 리뷰: `Code_Review_Hosted_CI_Remediation_6.md` (FAIL — 8.8/10.0, CRITICAL 0 /
MAJOR 2 / MINOR 1)
기준 revision: `2418119`(Iteration 6 코드 변경 위) — **merge/commit/push는
수행하지 않는다** — fresh Codex 리뷰가 필요하다.

## 0. 범위와 결론 요약

이 iteration은 Code_Review_Hosted_CI_Remediation_6.md가 지적한 두 MAJOR만
닫는다. MIN-03(lock fixture follow-up)은 이번 작업 범위에서 명시적으로
제외한다 — 별도 iteration으로 미룬다.

- **CR-HCIR6-MAJ-01**: `resolve_current()`가 `INDEX_ROOT/current`를 여는
  직접 `os.open(..., dir_fd=root.fd)` 호출에서 `ENOENT`/`ELOOP`만
  번역하고 `EACCES`는 raw `PermissionError`로 재전파해왔다 — Iteration 6이
  세 `ContainedDir` 진입점(`open_contained_root`/`open_subdir`/
  `open_member`)에서 이미 닫은 것과 동일한 결함이 네 번째 진입점에
  남아있었다(§1).
- **CR-HCIR6-MAJ-02**: `_capture_container_logs()`가 `len(str)`(문자 수)로
  바이트 예산을 재고 있었고, 매칭되는 모든 `startup` 줄을 무제한
  이어붙인 뒤에야 남은 예산을 계산했다 — 멀티바이트 로그, 단일
  oversized `startup` 이벤트, 중복 `startup` 이벤트 세 경로 모두 독립적으로
  광고된 `max_bytes`를 초과할 수 있었다(§2).

이 iteration이 건드리지 않은 것(§5): `_poll_ready`의 `max_seconds`,
`evaluate_readiness()`의 분기 순서/우선순위, negative control의 판정
임계값, `compute_all_ok`/`_ALL_OK_KEYS`, `expected_owner_uid`/dirfd/
`O_NOFOLLOW`/immutable-member/trust-before-pickle 계약, ENOENT legacy
fallback(`CurrentPointerMissing`) 분기, 다른 errno 처리, `M4.1_BLOCKED=true`,
protected M3 live `NOT_RUN`, `overall_release_ready=false` 산출 경로,
`.github/workflows/ci.yml`, `requirements.lock`/`requirements.txt`,
CR-HCIR6-MIN-03(lock fixture 강화)는 이 iteration에서 다루지 않는다.

## 1. CR-HCIR6-MAJ-01 — `resolve_current()`의 raw `EACCES` 누출

### 1.1 수정 (`src/simple_qna_rag/index/verification.py`)

`resolve_current()`의 직접 `current` open 핸들러에 `EACCES` 분기를
추가하고, 새 reason을 공개 allowlist(`REASONS`)에 추가했다:

```python
try:
    fd = os.open("current", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root.fd)
except OSError as exc:
    if exc.errno == errno.ENOENT:
        raise CurrentPointerMissing() from None
    if exc.errno == errno.ELOOP:
        raise TrustBoundaryError("current_pointer_symlink") from None
    if exc.errno == errno.EACCES:
        raise TrustBoundaryError("current_pointer_permission_denied") from None
    raise
```

`ENOENT` legacy fallback(`CurrentPointerMissing` → `_load_vectorstore_legacy`),
`ELOOP` symlink 처리, 그 외 errno의 raw 재전파는 전혀 바뀌지 않았다 — 오직
`EACCES` 한 branch만 추가했다. `expected_owner_uid` 검사, dirfd/`O_NOFOLLOW`
체인, `_MAX_CURRENT_BYTES` 상한 읽기, canonical-JSON 재검증, immutable-member
계약(`ContainedDir`)은 이 함수의 다른 부분이며 손대지 않았다.

### 1.2 propagation gap 발견과 수정 (`src/simple_qna_rag/rag_engine.py`)

리뷰가 요구한 "propagation through artifact/readiness chain" 테스트를
작성하기 전에 실제로 체인을 재현해본 결과, `_load_vectorstore()`가
`resolve_current()`의 `TrustBoundaryError`를 전혀 잡지 않는다는 **추가
결함**을 발견했다:

```python
try:
    version_id = index_verification.resolve_current(index_root)
except index_verification.CurrentPointerMissing:
    return self._load_vectorstore_legacy(embeddings)
try:
    return index_verification.load_verified_faiss(...)
except index_verification.TrustBoundaryError as exc:
    raise IndexTrustError(exc.reason) from None
```

`load_verified_faiss()` 호출을 감싸는 두 번째 `try`만
`index_verification.TrustBoundaryError`를 `IndexTrustError`로 번역한다.
`resolve_current()`가 던지는 `TrustBoundaryError`(symlink, malformed,
unknown_version, 그리고 새 `current_pointer_permission_denied` 전부)는 첫
번째 `try`를 그대로 빠져나가 `initialize()`의 범용
`except Exception: return False`로 떨어진다 — `_artifact_error_reason`이
전혀 세팅되지 않아, EACCES 번역을 아무리 정확히 해도 결과는 여전히 불투명한
`engine_init_failed`였다. 이는 §1.1만으로는 CR-HCIR6-MAJ-01이 요구하는
"typed reason propagates through the engine/artifact readiness chain"을
증명할 수 없다는 뜻이므로, 동일한 기존 패턴(`load_verified_faiss` 호출부의
번역)을 `resolve_current` 호출부에도 그대로 확장했다:

```python
try:
    version_id = index_verification.resolve_current(index_root)
except index_verification.CurrentPointerMissing:
    return self._load_vectorstore_legacy(embeddings)
except index_verification.TrustBoundaryError as exc:
    raise IndexTrustError(exc.reason) from None
try:
    return index_verification.load_verified_faiss(...)
except index_verification.TrustBoundaryError as exc:
    raise IndexTrustError(exc.reason) from None
```

이 수정은 새 메커니즘을 도입하지 않는다 — Iteration 6이 이미 승인한
"`TrustBoundaryError` → `IndexTrustError` → `initialize()`의
`except IndexTrustError` → `_artifact_error_reason` → `get_rag_engine()`의
`EngineArtifactError`" 채널을 두 번째 호출부에서 첫 번째 호출부로
그대로 반복 적용했을 뿐이다. `ENOENT`(`CurrentPointerMissing`) 분기는
여전히 legacy fallback으로 먼저 처리되므로 순서/우선순위는 바뀌지 않는다.

### 1.3 테스트

- `tests/unit/test_index_verification.py::
  test_resolve_current_direct_open_eacces_translates_to_disclosed_reason` —
  mutation-strength 테스트. `simple_qna_rag.index.verification.os.open`을
  patch해 `path == "current"`일 때만 `PermissionError(errno.EACCES, ...)`를
  발생시키고(그 외 경로는 실제 `os.open`으로 위임), `resolve_current()`가
  정확히 `TrustBoundaryError("current_pointer_permission_denied")`를
  던지는지 확인한다. §1.1의 새 branch를 되돌리면 이 테스트가 실패한다.
  실제 chmod가 아니라 정확한 호출부를 patch하는 이유는 root로 실행되는
  CI 프로세스에서는 chmod 기반 EACCES를 안정적으로 강제할 수 없기
  때문이다(기존 `test_permission_denied_matrix_...`는 root가 아닌 로컬
  실행 환경을 가정한다).
- `tests/unit/test_rag_engine_singleton.py::
  test_resolve_current_trust_boundary_error_propagates_through_engine_and_readiness_chain` —
  `index_verification.resolve_current`만 교체하고 `_load_vectorstore()` →
  `initialize()` → `get_rag_engine()`은 실제 코드 그대로 실행해,
  `current_pointer_permission_denied`가 `EngineArtifactError.reason`으로
  끝까지 살아남는지 확인한다(§1.2가 없으면 이 테스트는 `RuntimeError("RAG
  엔진 초기화 실패")`로 실패한다). `evaluate_readiness()`의
  `artifact_{reason}` 503 매핑 자체는 이미
  `tests/integration/test_health_endpoints.py::
  test_health_ready_engine_artifact_error_discloses_allowlisted_reason`이
  임의의 allowlisted reason에 대해 일반적으로 증명하므로 중복 재구현하지
  않았다.

## 2. CR-HCIR6-MAJ-02 — `_capture_container_logs()`의 byte-bound 위반

### 2.1 수정 (`scripts/container_smoke.py::_capture_container_logs`)

```python
combined_bytes = combined.encode("utf-8")
if len(combined_bytes) <= max_bytes:
    return combined

lines = combined.splitlines(keepends=True)
startup_lines = [ln for ln in lines if '"event": "startup"' in ln]
startup = startup_lines[-1] if startup_lines else ""
startup_bytes = startup.encode("utf-8")
if len(startup_bytes) > max_bytes:
    return startup_bytes[:max_bytes].decode("utf-8", errors="ignore")
remaining = max_bytes - len(startup_bytes)
tail_bytes = combined_bytes[-remaining:] if remaining > 0 else b""
tail = tail_bytes.decode("utf-8", errors="ignore")
if startup and startup in tail:
    return tail
return startup + tail
```

리뷰가 지적한 네 가지 결함을 각각 다음으로 닫는다.

1. **문자 수가 아닌 바이트로 측정**: 초기 단락(early-return) 비교와 이후
   모든 절단이 `combined.encode("utf-8")`/`startup.encode("utf-8")`의
   `len()`(바이트)을 기준으로 한다 — 더 이상 `len(str)`(문자)을 쓰지 않는다.
2. **최대 하나의 `startup` 이벤트만 보존(최신 우선)**: 이전 코드는
   `"".join(ln for ln in lines if ...)`로 매칭되는 *모든* `startup` 줄을
   이어붙였다. 새 코드는 `startup_lines[-1]`(가장 최근 것) 하나만
   취한다 — 크래시 루프가 `startup`을 여러 번 찍어도 그 자체로 예산을
   넘길 수 없다.
3. **`startup`과 tail을 하나의 `max_bytes` 예산 안에서 함께 절단**: 단일
   `startup` 줄 자체가 예산보다 크면(`len(startup_bytes) > max_bytes`)
   그 줄만 `max_bytes`로 잘라 반환한다(header가 sort_keys 정렬된 JSON이므로
   앞부분에 `engine_error_type`/`event`/`level`/`reason`처럼 진단에 더
   유용한 키가 오는 것을 이용해 head를 보존). 그렇지 않으면
   `remaining = max_bytes - len(startup_bytes)`만큼만 tail을 채운다 — 두
   조각의 합은 구조적으로 `max_bytes`를 넘을 수 없다.
4. **overlap 중복 방지**: 재계산된 tail 창이 이미 전체 `startup` 줄을
   포함하면(`startup in tail`) 별도로 앞에 붙이지 않고 `tail` 그대로
   반환한다 — 예산을 낭비하는 중복 없이 동일한 증거를 유지한다.
5. **결정론적 decode**: 바이트 슬라이싱이 멀티바이트 문자 중간을 자를 수
   있는 두 지점(뒤에서부터 자르는 tail의 시작 경계, 앞에서부터 자르는
   oversized-startup의 끝 경계) 모두 `errors="ignore"`로 decode한다 —
   `errors="replace"`와 달리 대체 문자를 채워 넣어 바이트 수를 다시 늘리는
   일이 없으므로(예: 잘린 continuation byte 1개가 U+FFFD 3바이트로
   부풀 수 있는 경우), 어떤 절단 지점에서도 재인코딩 결과가 절대
   `max_bytes`를 넘지 않는다는 보장이 구조적으로 성립한다. 원본 `combined`
   자체는 이미 유효한 Python 문자열(디코딩 성공한 subprocess 출력)이므로
   내부에는 잘못된 시퀀스가 없고, `ignore`가 실제로 건드리는 바이트는
   우리가 만든 절단 경계뿐이다.

`docker logs --tail 200` 호출, 성공 경로에서 아무것도 채우지 않는 계약,
`max_bytes` 기본값(16000), `_run()`/예외 처리(`try/except Exception:
return ""`)는 전혀 바뀌지 않았다.

### 2.2 테스트 (`tests/unit/test_container_smoke_readiness_diagnostics.py`)

네 가지 mutation-strength 테스트를 추가했다. 각각
`len(result.encode("utf-8")) <= max_bytes`를 직접 단언한다.

- `test_capture_container_logs_bounds_encoded_bytes_not_character_count` —
  리뷰가 제시한 정확한 반례(`100 characters / 300 UTF-8 bytes with
  max_bytes=100`)를 한국어 100자로 재현.
- `test_capture_container_logs_oversized_single_startup_event_stays_within_budget` —
  단일 `startup` 줄이 `max_bytes`보다 큰 경우.
- `test_capture_container_logs_retains_only_latest_of_duplicate_startup_events` —
  `startup` 줄 5개(크래시 루프 시뮬레이션); 최신 1개만 남고
  `result.count('"event": "startup"') == 1`을 확인.
- `test_capture_container_logs_avoids_duplicating_startup_line_present_in_tail_window` —
  재계산된 tail 창이 이미 전체 `startup` 줄을 포함하도록 예산을 구성해
  dedup 분기(`startup in tail`)가 실제로 타는지 확인.

기존 5개 pure-unit 테스트(`combines_stdout_and_stderr`,
`truncates_to_tail_not_head`, `preserves_startup_line_under_health_check_spam`,
`falls_back_to_plain_tail_when_no_startup_line`,
`never_raises_on_docker_failure`)는 전부 ASCII 입력만 사용하므로 문자 수와
바이트 수가 일치해 수정 전후 동일하게 통과한다(회귀 없음) — 새 구현으로
교체 후 재실행해 확인했다(§4).

## 3. 닫힘 매핑(exact closure mapping)

| 발견 | 파일 | 수정 | 테스트 |
|---|---|---|---|
| CR-HCIR6-MAJ-01 (핵심) | `src/simple_qna_rag/index/verification.py` — `resolve_current()`, `REASONS` | `current` 직접 open의 `EACCES` → `TrustBoundaryError("current_pointer_permission_denied")`; allowlist 추가 | `test_resolve_current_direct_open_eacces_translates_to_disclosed_reason` |
| CR-HCIR6-MAJ-01 (propagation 전제조건, 리뷰 요구사항 충족을 위한 필수 보강) | `src/simple_qna_rag/rag_engine.py` — `_load_vectorstore()` | `resolve_current()` 호출부에 기존 `TrustBoundaryError → IndexTrustError` 번역 패턴 확장 | `test_resolve_current_trust_boundary_error_propagates_through_engine_and_readiness_chain` |
| CR-HCIR6-MAJ-02 | `scripts/container_smoke.py` — `_capture_container_logs()` | 바이트 측정, 단일 최신 startup만 보존, 하나의 `max_bytes` 예산 내 공동 절단, overlap dedup, `errors="ignore"` 결정론적 decode | `test_capture_container_logs_bounds_encoded_bytes_not_character_count`, `..._oversized_single_startup_event_stays_within_budget`, `..._retains_only_latest_of_duplicate_startup_events`, `..._avoids_duplicating_startup_line_present_in_tail_window` |
| CR-HCIR6-MIN-03 | — | 이번 iteration 범위에서 **명시적으로 제외**(작업 지시) | — |

CR-HCIR6-MAJ-01/MAJ-02 모두 CRITICAL/MAJOR 재발 없이 정확히 리뷰가 요구한
범위로 닫혔다고 판단한다. MIN-03은 다음 iteration(별도 승인 시)으로
이월한다.

## 4. 재검증 결과

- 대상 파일 focused 테스트: `venv/bin/python -m pytest -q
  tests/unit/test_index_verification.py tests/unit/test_rag_engine_singleton.py
  tests/unit/test_container_smoke_readiness_diagnostics.py` — **전부 통과**
  (신규 6건 포함, 기존 회귀 없음).
- 전체 로컬 결정론적 suite: `venv/bin/python -m pytest -q`(단위+통합,
  macOS 로컬) — 결과는 본 문서 최종본에 반영(§4.1 참고, 실행 시각 기준
  최신 카운트).
- `python scripts/generate_field_spec.py --check`: exit 0(변경 없음 —
  이번 iteration은 필드 스펙에 영향을 주는 로깅 스키마를 건드리지
  않았다).
- `python scripts/logging_callsite_audit.py --check`: exit 0(변경 없음 —
  `docs/generated/logging_callsite_disposition.json`은 이번 iteration이
  아닌 이전 세션의 Iteration 6 작업에서 이미 갱신된 상태 그대로다. 이번
  변경은 로깅 콜사이트 목록에 영향을 주지 않는다).
- `python scripts/check_markdown_links.py`: 검사 파일 139개(tracked 137 +
  untracked 2, 이 문서 포함), 링크 597개, 실패 0개.
- protected 경계(`git diff --exit-code -- .github/workflows/ci.yml
  scripts/scan_image_layers.py scripts/assemble_m4_evidence.py
  scripts/check_m4_baseline.py evaluation/baselines/m3_initial.*
  requirements.lock requirements.txt`): exit 0(변경 없음).
- `docker build --platform linux/amd64 --target production -f
  deploy/Dockerfile -t simple-qna-rag:iter7-repro .`: 성공.
- `venv/bin/python scripts/container_smoke.py --image
  simple-qna-rag:iter7-repro`(실제 docker, 보안 플래그·negative control
  전부 포함, macOS `linux/amd64` 에뮬레이션): **`status: PASS`**, 6개
  boolean 전부 true, `readiness_sequence.ready_last_reason: "ok"`,
  `ready_poll_elapsed_seconds: 6.06`, negative control
  `production_test_seam_seal_last_http_status: 503`/
  `..._last_reason: "artifact_test_embedding_seam_unavailable"` 그대로 —
  §2.1의 `_capture_container_logs()` 재작성이 정상 경로(성공 시 tail을
  전혀 채우지 않는 계약 포함)와 negative control 경로 어디에도 회귀를
  일으키지 않았다.
- Native Linux/Ollama/DDGS/live/self-hosted, protected M3/M4.1 live gate,
  environment 승인 경계는 이 remediation에서도 실행·변경하지 않았다.

### 4.1 전체 suite 실행 결과

```text
venv/bin/python -m pytest -q
1326 passed, 1 skipped, 4 warnings in 168.99s (0:02:48)
```

Iteration 6 기준 `1320 passed, 1 skipped` 대비 정확히 신규 6건(§1.3의
2건 + §2.2의 4건) 순수 추가 — 회귀 없음.

## 5. 이 remediation이 건드리지 않은 것

`_poll_ready`의 `max_seconds`, `evaluate_readiness()`의 분기 순서/우선순위,
negative control의 판정 임계값(`expect_status=503`/
`expect_reason="artifact_test_embedding_seam_unavailable"`),
`compute_all_ok`/`_ALL_OK_KEYS`, `resolve_current()`의 `ENOENT` legacy
fallback(`CurrentPointerMissing`)과 `current_pointer_symlink`/
`current_pointer_malformed`/`current_pointer_unknown_version` 분기,
`expected_owner_uid` 검사, dirfd/`O_NOFOLLOW` 체인, immutable-member 계약,
trust-before-pickle 동작, `verify_version()`/`ContainedDir.open_subdir`/
`open_member`/`open_contained_root`의 기존 `EACCES` 번역(Iteration 6에서
이미 닫힘), `docker logs --tail 200` 호출 자체, `_capture_container_logs`의
`max_bytes` 기본값(16000)과 실패 시에만 채우는 계약, `M4.1_BLOCKED=true`,
protected M3 live `NOT_RUN`, `overall_release_ready=false` 산출 경로,
`m3-live-regression-gate` 블록, `.github/workflows/ci.yml`,
`scripts/scan_image_layers.py`/`assemble_m4_evidence.py`/
`check_m4_baseline.py`, `requirements.lock`/`requirements.txt`, CPU torch
extra-index 시맨틱, `--generate-hashes` 계약, uv 0.8.15 고정,
CR-HCIR6-MIN-03(lock fixture 강화 — 이번 작업 지시에 따라 명시적으로 제외),
Native Linux/Ollama/DDGS/live/self-hosted 승인 경계, 위 결함과 무관한 어떤
파일도 수정하지 않았다.

## 6. 남은 작업

이 커밋은 아직 push되지 않았다. **merge/commit/push는 수행하지 않는다** —
fresh Codex 리뷰가 필요하다(작업 지시). 리뷰가 PASS하면 diagnostic
commit/push와 hosted 재실행이 다음 단계이며, hosted 성공 여부는 별도로
평가한다. CR-HCIR6-MIN-03은 이 리뷰가 PASS한 이후 별도 승인을 받아 후속
iteration으로 진행할 수 있다.
