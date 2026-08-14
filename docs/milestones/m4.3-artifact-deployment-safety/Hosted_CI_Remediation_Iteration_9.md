# M4.3 Artifact & Deployment Safety — Hosted CI Remediation Iteration 9

작성자: Claude Sonnet 5 (bounded remediation worker)
대상: PR #18 (미push, 동일 브랜치 `agent/m4-3-artifact-deployment-safety`)
기준 커밋: `c99419f` (Iteration 6-8, hosted run
[31822051073](https://github.com/luminous419/simple-qna-rag/actions/runs/31822051073))
직전 실패: container job `94837374291`, step "Container security/mock
smoke" — 실제 Linux hosted runner에서 `current` pointer 파일이 mode `0600`으로
쓰여져 있어, 프로덕션 컨테이너를 구동하는 non-owner UID `10001`이 이를 열 수
없었다(`current_pointer_permission_denied`, 진단은 정확했으나 근본 결함은
남아 있었다 — Iteration 6/7이 이 reason을 readiness 경로가 정확히 보고하도록
고쳤을 뿐, `current`를 실제로 world-readable하게 쓰는 수정은 아직 없었다).

## 0. 인수한 상태와 범위

이전 release worker가 release 범위를 벗어나 남긴 미커밋 후보 수정이
`src/simple_qna_rag/index/lifecycle.py`와 `tests/unit/test_index_lifecycle.py`
두 파일에만 존재했다. 이번 iteration은 이 두 후보 수정을 **감사(audit)한 뒤
근본 결함을 좁게(narrow) 수정**하는 것으로 범위를 한정한다 — 다른 어떤 파일도
건드리지 않는다.

### 0.1 후보 수정 감사 결과

후보 diff는 `activate()`의 `current` pointer tmp 파일에 `os.chmod(tmp, 0o444)`를
추가하고, 이를 검증하는 회귀 테스트
(`test_activate_writes_current_world_readable_regardless_of_umask`)를
추가한 것이었다. `_publish()`가 이미 발행된 버전 파일에 `0o444`/디렉터리에
`0o555`를 사용하는 기존 패턴과 일관되어 보였고, 원인 분석
(`_write_fsync`의 기본 mode가 `0o600`이라는 것, container가 `--user
10001:10001`로 구동된다는 것 — `scripts/container_smoke.py`의
`build_docker_run_argv`)도 정확했다.

**그러나 `0o444`는 감사 중 발견한 실제 회귀를 유발했다.** 전체 결정론적
suite(`venv/bin/python -m pytest -q`)를 실행하자
`tests/integration/test_index_lifecycle_fault_injection.py::
test_crash_recovery_journal_reconciles_to_consistent_state`가
`PermissionError: [Errno 13] Permission denied`로 실패했다. 이 기존 테스트는
crash-recovery 재조정을 검증하기 위해 `activate()` 이후 `(tmp_path /
"current").write_bytes(...)`로 pointer 파일을 **의도적으로 제자리에서
덮어쓴다**(중단된 트랜지션을 재현하는 방식). `current`는 `_publish`의 버전
파일들과 달리 **불변(immutable) 아티팩트가 아니라 가변(mutable) 포인터**이며,
crash-recovery 경로가 이를 제자리에서 재작성하는 것은 정상 동작이다. `0o444`
(owner 포함 전원 read-only)는 이 정상 동작을 깨뜨린다 — 후보 수정이
컨테이너 결함은 고쳤지만 그 과정에서 별도의 회귀를 새로 유발한 것이다.

이 감사 결과에 따라 두 후보 파일의 수정 방향은 유지하되, 정확한 모드 값만
`0o444`에서 **`0o644`**(owner rw, world r)로 교정했다. `0o644`는 world-read
비트를 유지해 원래 결함(비-owner UID의 read)을 그대로 해결하면서, owner
write 비트를 남겨 crash-recovery 테스트가 요구하는 제자리 쓰기를 깨뜨리지
않는다.

## 1. 수정 (`src/simple_qna_rag/index/lifecycle.py::activate`)

```python
tmp = index_root / f".current.tmp.{os.getpid()}.{op_id}"
_write_fsync(tmp, canonical_json_bytes(
    {"schema_version": 1, "version_id": verified.version_id}) + b"\n")
os.chmod(tmp, 0o644)
os.replace(tmp, index_root / "current")
```

- `os.chmod`는 `os.replace`(atomic rename) **이전에** tmp 파일에 적용된다 —
  `current`라는 이름으로는 더 엄격한 모드로 보이는 순간이 전혀 없다.
- `os.replace`는 원본 파일 고유의 permission bit를 그대로 유지하므로(대상이
  이를 요구하지 않는다), rename 이후 `current`도 `0o644`를 그대로 유지한다.
- `_publish`의 `0o444`(파일)/`0o555`(디렉터리)는 건드리지 않았다 — 발행된
  버전 아티팩트의 불변성 계약은 이번 수정과 무관하다.

## 2. 테스트 (`tests/unit/test_index_lifecycle.py`)

기존 후보 테스트 `test_activate_writes_current_world_readable_regardless_of_umask`를
유지하되, 기대값을 `"444"`에서 `"644"`로 교정하고 `os.access` 기반 확인을
추가했다:

```python
def test_activate_writes_current_world_readable_regardless_of_umask(tmp_path):
    old_umask = os.umask(0o077)
    try:
        v1 = _publish(tmp_path)
        lifecycle.activate(tmp_path, v1, operation="activate", settings_snapshot=_SNAPSHOT)
    finally:
        os.umask(old_umask)
    current_path = tmp_path / "current"
    assert oct(current_path.stat().st_mode)[-3:] == "644"
    assert os.access(current_path, os.R_OK)
    assert os.access(current_path, os.W_OK)
```

- 제한적 umask(`0o077`)를 사용해, 관대한 호스트 umask가 `os.chmod` 수정의
  회귀를 가려버리는 것을 방지한다(`versions/` 디렉터리에 대한 Iteration 6
  테스트와 동일한 패턴).
- `os.access` 검증 두 줄은 mode bit 문자열 비교만으로는 놓칠 수 있는 실제
  read/write 가능 여부를 추가로 보증한다.
- 기존 `test_crash_recovery_journal_reconciles_to_consistent_state`(신규
  작성분 아님, 이번 iteration에서 diff 없음)는 그대로 두어 회귀 방지망
  역할을 하게 했다 — §4.2에서 mutation-strength 검증에 사용한다.

## 3. 닫힘 매핑(exact closure mapping)

| 발견 | 파일 | 수정 | 테스트 |
|---|---|---|---|
| Hosted job 94837374291, "Container security/mock smoke" — `current` pointer mode `0600`으로 인한 UID 10001 EACCES | `src/simple_qna_rag/index/lifecycle.py::activate` | tmp pointer 파일에 atomic rename 이전 `os.chmod(tmp, 0o644)` 추가 | `test_activate_writes_current_world_readable_regardless_of_umask` (신규/교정) |
| 감사 중 발견: 후보 수정의 `0o444`가 `test_crash_recovery_journal_reconciles_to_consistent_state`(기존 테스트)를 깨뜨림 | 동일 | 모드 값을 `0o444` → `0o644`로 교정(owner-writable 유지) | `test_crash_recovery_journal_reconciles_to_consistent_state` (기존, 회귀 확인 재사용) |

## 4. 재검증 결과

### 4.1 대상/전체 테스트

- 대상 파일 focused 테스트: `venv/bin/python -m pytest -q
  tests/unit/test_index_lifecycle.py
  tests/integration/test_index_lifecycle_fault_injection.py` — **40 passed**
  (신규 1건 포함, crash-recovery 회귀 없음).
- 전체 로컬 결정론적 suite: `venv/bin/python -m pytest -q` — **1329 passed,
  1 skipped, 4 warnings in 175.00s**. Iteration 8 리뷰 기준 `1328 passed, 1
  skipped` 대비 정확히 신규 1건(§2) 순수 추가 — 회귀 없음.
- `python scripts/generate_field_spec.py --check`: exit 0(변경 없음).
- `python scripts/logging_callsite_audit.py --check`: exit 0(변경 없음).
- protected 경계(`git diff --exit-code -- .github/workflows/ci.yml
  scripts/scan_image_layers.py scripts/assemble_m4_evidence.py
  scripts/check_m4_baseline.py evaluation/baselines/m3_initial.*
  requirements.lock requirements.txt deploy/Dockerfile`): exit 0(변경 없음).
- `git status --short`: 수정된 파일은 정확히 `src/simple_qna_rag/index/
  lifecycle.py`와 `tests/unit/test_index_lifecycle.py` 두 개뿐.

### 4.2 mutation-strength 독립 검증

두 개의 독립 mutant를 별도 스크립트로 적용해, 새 회귀 방지망(신규 테스트 +
기존 crash-recovery 테스트)이 정확히 올바른 수정만 통과시키는지 확인했다.
검증 후 원본 수정으로 완전히 복원(diff 동일성 확인)했다.

```text
Mutant A — os.chmod 호출 자체를 제거(원래 결함 재현):
  test_activate_writes_current_world_readable_regardless_of_umask
  -> FAILED (assert '600' == '644' / '644')
  결함 재도입을 신규 테스트가 정확히 잡아낸다.

Mutant B — os.chmod(tmp, 0o444)(후보가 원래 제출했던, 너무 제한적인 값):
  test_crash_recovery_journal_reconciles_to_consistent_state
  -> FAILED (PermissionError: [Errno 13] Permission denied)
  후보의 원래 결함(과도한 제한)을 기존 crash-recovery 테스트가 정확히
  잡아낸다.

복원 후 재검증:
  tests/unit/test_index_lifecycle.py
  tests/integration/test_index_lifecycle_fault_injection.py
  -> 40 passed, 3 warnings
```

두 mutant 모두 서로 다른 테스트에 의해 개별적으로 킬(kill)됨을 확인했다 —
`0o644`가 두 요구사항(비-owner world-read, owner-write 유지) 모두를 만족하는
유일하게 올바른 값이라는 것을 회귀 방지망이 실제로 강제한다는 의미다.

### 4.3 실제 컨테이너 검증(linux/amd64)

```
docker build --platform linux/amd64 --target production \
  -f deploy/Dockerfile -t simple-qna-rag:iter9-repro .
```
성공(대부분 레이어 캐시 hit — 이번 iteration은 `deploy/Dockerfile`이나
의존성을 건드리지 않았다).

```
venv/bin/python scripts/container_smoke.py \
  --image simple-qna-rag:iter9-repro --output /tmp/container_smoke_iter9.json
```

결과(macOS `linux/amd64` 에뮬레이션, 실제 docker, 보안 플래그·negative
control 전부 포함, `--user 10001:10001`로 구동):

```json
{
  "status": "PASS",
  "readiness_sequence": {
    "live": true,
    "ready": true,
    "ready_last_http_status": 200,
    "ready_last_reason": "ok",
    "ready_poll_elapsed_seconds": 6.04
  },
  "host_gateway_reachable": true,
  "mock_query_ok": true,
  "root_page_ok": true,
  "static_asset_ok": true,
  "production_test_seam_sealed": true,
  "production_test_seam_seal_last_http_status": 503,
  "production_test_seam_seal_last_reason": "artifact_test_embedding_seam_unavailable"
}
```

`ready_last_reason: "ok"`(이전 hosted 실패는 `current_pointer_permission_denied`
계열의 503으로 readiness가 수렴하지 못했다), 5개 `_ALL_OK_KEYS` boolean 전부
`true`, `container_log_tail` 키 없음(성공 경로 계약 유지). 컨테이너/이미지
정리 후(`docker rmi simple-qna-rag:iter9-repro`) leftover 컨테이너 없음을
`docker ps -a --filter ancestor=...`로 확인했다.

## 5. 이 remediation이 건드리지 않은 것

`_publish()`의 `0o444`/`0o555` 불변 아티팩트 계약, `versions/` 디렉터리
`0o755` 수정(Iteration 6), `_capture_container_logs()`의 dedup/overlap 로직
(Iteration 8), `resolve_current()`의 errno 분기와 `_load_vectorstore()`의
`TrustBoundaryError` propagation(`src/simple_qna_rag/index/verification.py`,
`src/simple_qna_rag/rag_engine.py` — 이번 iteration에서 diff 없음),
`expected_owner_uid` 검사, dirfd/`O_NOFOLLOW` 체인, `_write_transition_journal`/
`_append_history`/`_write_receipt_atomic`(계속 `0o600`, 비-owner 읽기가 필요
없다), `deploy/Dockerfile`, `.github/workflows/ci.yml`,
`scripts/scan_image_layers.py`/`assemble_m4_evidence.py`/
`check_m4_baseline.py`, `requirements.lock`/`requirements.txt`,
`m3-live-regression-gate` 블록과 `environment: m3-live-regression` 승인
경계. `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`/workflow `SKIPPED`,
`overall_release_ready=false`는 이 diff가 닿지 않는 `scripts/
assemble_m4_evidence.py`/`check_m4_baseline.py` 경로에서 산출되므로 그대로
보존된다 — 이번 iteration에서 그 값들을 재계산하거나 재실행하지 않았다.

## 6. 남은 작업

이 커밋은 아직 push되지 않았다(작업 지시에 따라 commit/push 수행하지 않음).
fresh 코드 리뷰가 필요하다. 리뷰가 PASS하면 commit/push와 hosted 재실행이
다음 단계이며, hosted job `94837374291`가 실패했던 정확한 지점("Container
security/mock smoke")이 실제 linux/amd64 컨테이너에서 재현 검증되었음을
§4.3이 보인다.
