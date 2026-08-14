# M4.3 Artifact & Deployment Safety — Code Review Iteration 1 Remediation

작성자: Claude Code Sonnet 5 (remediation worker)
기준 revision: `648e3ab` (`master`, M4.2 merge) — 이 세션의 모든 변경은
작업 트리에만 존재하며 **commit/push/PR을 수행하지 않았다**.
대상 리뷰: [Code_Review_Iteration_1.md](Code_Review_Iteration_1.md)
(판정 **FAIL — 7.8/10**, `CRITICAL 0 / MAJOR 3 / MINOR 1 / TRIVIAL 0`)

## 0. 범위

이 문서는 Code Review Iteration 1이 지적한 4개 finding
(`CR-I1-MAJ-01`, `CR-I1-MAJ-02`, `CR-I1-MAJ-03`, `CR-I1-MIN-01`)을
finding별로 코드 변경/신규 테스트/실제 실행 결과에 1:1로 매핑한다.
이전 구현 세션이 작성한 [Implementation_Report.md](Implementation_Report.md)의
§1~§8은 remediation 이전 시점의 기록으로 그대로 보존했고, 이 remediation의
요약은 그 문서의 §9에 추가했다. Design.md/Requirement.md/Plan.md 본문은
변경하지 않았다(finding들은 설계 위반이 아니라 구현 결함이었다).

이 remediation이 건드리지 않은 것: `scripts/orchestration_watchdog.py`의
의도적 readiness fix, protected `m3-live-regression-gate` workflow 블록,
`M4.1_BLOCKED=true`/M3 live `NOT_RUN`/`overall_release_ready=false` 산출
경로, Native Linux/Ollama/DDGS, self-hosted runner/environment 승인 경계.
이 4개 finding은 모두 `index/`, `container_smoke.py`, 그리고 그 테스트에만
있었다.

또한 저장소 루트에 있던 untracked `.transition` 파일을 삭제했다 — 이
파일은 이 milestone의 산출물이 아니라 리뷰가 CR-I1-MAJ-02를 재현하며 남긴
악성 샘플 JSON 잔재(`{"schema": "wrong", "phase": "pointer_committed",
"op_id": "../escaped", "operation": "delete", "pre_pointer": null,
"post_pointer": null, "recorded_at": "x"}`)였다. `grep -rn '\.transition'
tests/ src/ scripts/`로 모든 `.transition` 관련 테스트가 `tmp_path`
픽스처만 사용함을 확인했으므로(`tests/integration/
test_index_lifecycle_fault_injection.py`), 실제 산출물이나 테스트
의존성이 아니었다.

## 1. CR-I1-MAJ-01 — container smoke가 정적 자산을 검사하지 않고 항상 `false`를 기록

### 원본 결함

`scripts/container_smoke.py:149` 부근에서 `root_ok, static_ok = False,
False`로 초기화한 뒤 `static_ok`를 갱신하는 코드가 전혀 없이
`result["static_asset_ok"] = static_ok`로 그대로 기록했다. `run_smoke()`
자체의 `all_ok` 계산에도 `static_asset_ok`가 빠져 있어 다른 4개 필드만
True면 `status="PASS"`로 종료할 수 있었지만, `assemble_m4_evidence.py`의
`_verify_payloads`는 `static_asset_ok == True`를 요구하므로 실제 Docker
build/smoke가 완전히 성공해도 `m4-assemble`이 `PAYLOAD_INVALID`가 되어
deterministic gate가 필연적으로 실패했다.

### 수정

- `scripts/container_smoke.py`에 `STATIC_ASSET_PATH = "/static/app.js"`
  (production 이미지가 실제로 `COPY web/static/`하는 vendored 자산),
  `check_static_asset(port, *, timeout=5.0) -> bool` — 실제
  `GET http://127.0.0.1:<port>/static/app.js`를 호출해 (1) status==200,
  (2) `Content-Type`이 `text/javascript`/`application/javascript`로
  시작, (3) body가 비어있지 않음을 모두 확인해야 `True`를 반환한다.
  404(예: `COPY web/static/` 누락 회귀), 잘못된 content-type, 빈 body,
  connection 예외는 모두 `False`.
- `run_smoke()`의 `static_ok` 상수 스텁을 제거하고
  `result["static_asset_ok"] = check_static_asset(host_port)`로 실제
  호출 결과를 기록하도록 배선했다.
- `_ALL_OK_KEYS = (host_gateway_reachable, mock_query_ok, root_page_ok,
  static_asset_ok, production_test_seam_sealed)`와
  `compute_all_ok(result) -> bool`을 신설해 `all_ok` 계산 자체를
  테스트 가능한 독립 값으로 분리했다 — `static_asset_ok`가 이제
  `all_ok`/`status`/CI exit code에 실제로 반영된다.

### 신규 테스트 (`tests/unit/test_container_smoke_contract.py`, 4→15)

| 테스트 | 검증 내용 |
|---|---|
| `test_check_static_asset_true_for_200_with_js_content_type_and_body` | stubbed 200 + `text/javascript` + non-empty body → `True` |
| `test_check_static_asset_false_for_404_missing_copy_web_static` | stubbed `HTTPError(404)` → `False`(정확히 `COPY web/static/` 누락 회귀 시나리오) |
| `test_check_static_asset_false_for_wrong_content_type` | 200 + `text/html` → `False` |
| `test_check_static_asset_false_for_empty_body` | 200 + 빈 body → `False` |
| `test_check_static_asset_false_on_connection_error` | `OSError` → `False`(예외를 삼키지 않고 명시적으로 실패 처리) |
| `test_check_static_asset_requests_the_expected_path` | 호출 URL이 정확히 `STATIC_ASSET_PATH`인지 확인 |
| `test_compute_all_ok_true_when_every_field_true` | 5개 필드 모두 True → `True` |
| `test_compute_all_ok_false_when_static_asset_ok_false_even_if_others_true` | 나머지 4개가 True여도 `static_asset_ok=False`면 `False`(원본 finding의 정확한 회귀 오라클) |
| `test_compute_all_ok_false_when_static_asset_ok_missing_from_receipt` | 필드 자체가 없으면 `False`(fail-closed) |
| `test_main_exits_nonzero_when_static_asset_check_fails` | `run_smoke()`를 스텁해 "docker 성공+static 404" 시나리오를 재현 — `main()` exit code 1 확인 |
| `test_main_exits_zero_when_all_checks_including_static_asset_pass` | 전부 PASS → exit 0 |

기존 4개 argv-계약 테스트(`test_docker_run_argv_...`,
`test_reachability_probe_argv_...`, `test_negative_activation_argv_...`,
`test_docker_unavailable_short_circuits_to_skipped`)는 무변경.

### 실제 이미지 대상 실행에 대한 한계

이 remediation은 **로직**을 stubbed HTTP로 전수 검증했다 — 실제
`docker build --target production` 이미지에 대해 `container_smoke.py`
전체를 실행하는 것은 여전히 이 milestone의 기존 제약(호스트 arm64 +
Docker Desktop 디스크 소진, Implementation_Report.md §3/§4)에 막혀 있다.
`test_main_exits_nonzero_when_static_asset_check_fails`는 `run_smoke()`를
스텁해 "COPY web/static/ 누락" 시나리오를 CLI 계약 레벨에서 재현하지만,
실제 Dockerfile에서 그 줄을 제거하고 빌드하는 end-to-end 테스트는 아니다
— 그 최종 확인은 hosted CI(x86_64, 디스크 여유)에서 이뤄져야 한다(기존
Implementation_Report.md §8-3와 동일한 잔여 조건).

## 2. CR-I1-MAJ-02 — corrupt transition journal이 검증 없이 PASS receipt/history로 승격

### 원본 결함

`src/simple_qna_rag/index/lifecycle.py::_reconcile_pending_transition`이
`.transition` 파일을 `json.loads()`로만 파싱한 뒤 `record.get()`으로
raw 값을 그대로 `_append_history`/`ActivationReceipt`에 전달했다. JSON
object 여부, exact key set, `schema` 리터럴, `phase`/`operation` enum,
32-hex `op_id`, pointer 타입/정규식, timestamp 형식을 전혀 검증하지
않았다. 리뷰는 `{"schema": "wrong", "phase": "pointer_committed",
"op_id": "../escaped", "operation": "delete", "pre_pointer": null,
"post_pointer": null, "recorded_at": "x"}`를 빈 index root에 두고
`_reconcile_pending_transition()`을 호출해 `outcome="completed"`인
PASS receipt가 실제로 기록됨을 재현했다 — crash recovery라는 mutation
경계가 tampered/corrupted durable state를 fail-closed로 거부하지
않았다.

### 수정

`src/simple_qna_rag/index/lifecycle.py`에 순수 함수
`_parse_transition_journal(raw: bytes) -> _TransitionRecord`를
신설했다(파일 시스템을 건드리지 않는다 — grep-auditable):

1. UTF-8 디코드 + JSON object 파싱 실패 → `transition_journal_corrupt`
2. `set(doc) != {"schema", "phase", "op_id", "operation", "pre_pointer",
   "post_pointer", "recorded_at"}`(exact 7-key, 초과/누락 모두 거부)
3. `doc["schema"] != "m43-transition-journal-v1"` 거부
4. `doc["phase"] not in {"prepared", "pointer_committed"}` 거부
5. `op_id`가 `^[0-9a-f]{32}$`에 fullmatch하지 않으면 거부(32-hex
   고정 길이 정규식이므로 `"../escaped"`나 다른 어떤 traversal 문자열도
   구조적으로 통과할 수 없다 — path 조작 표면이 사라진다)
6. `doc["operation"] not in {"activate", "rollback"}` 거부
7. `pre_pointer`가 `None`이 아니면 `^[0-9a-f]{16}$`(기존
   `_VERSION_ID_RE`) fullmatch 필수
8. `post_pointer`는 항상 문자열이어야 하며 같은 16-hex 정규식 필수
   (null 거부 — 원본 finding의 재현 사례가 정확히 이 필드를 `null`로
   뒀다)
9. `recorded_at`이 `^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$`
   에 fullmatch하지 않으면 거부

모든 실패는 예외 없이 동일한 `TrustBoundaryError(
"transition_journal_corrupt")`이며, 이 검증은 `current` 읽기나 어떤
mutation보다 먼저 실행된다 — malformed journal은 절대로 부분적으로도
신뢰되지 않는다. `_reconcile_pending_transition`과
read-only `_diagnose_pending_transition` 둘 다 이 단일 parser를
재사용하도록 재작성했다(이전에는 두 함수가 각자 `json.loads` +
`.get()`을 중복 구현했다).

### 신규 테스트 (`tests/integration/test_index_lifecycle_fault_injection.py`, 4→24)

`_MALFORMED_JOURNAL_CASES`(18개 케이스)를
`test_malformed_transition_journal_rejected_without_any_mutation`으로
parametrize:

`not_json`, `not_object`, `missing_key`, `extra_key`, `wrong_schema`,
`invalid_phase_enum`, `invalid_operation_enum`, `op_id_wrong_length`,
`op_id_uppercase_hex`, `op_id_path_traversal`
(`"../../../../etc/passwd"`), `op_id_path_traversal_relative`
(`"../escaped"`), `pre_pointer_wrong_type`, `pre_pointer_malformed_hex`,
`post_pointer_null`, `post_pointer_wrong_type`,
`post_pointer_malformed_hex`, `recorded_at_wrong_type`,
`recorded_at_not_iso8601`.

각 케이스는 (a) `TrustBoundaryError("transition_journal_corrupt")`가
발생하고, (b) `current` 포인터 바이트가 호출 전과 동일하며, (c)
`.transition` 파일 자체는 삭제되지 않고 그대로 남아(운영자 조사용) 있고,
(d) 이 malformed 레코드가 담고 있던 op_id로 어떤 history 파일도 생성되지
않았음을 확인한다.

추가로:

- `test_reproduces_cr_i1_maj_02_original_finding_journal` — 리뷰 원문의
  정확한 재현 입력을 그대로 사용해(빈 index root, `schema:"wrong"`,
  `op_id:"../escaped"`, `operation:"delete"`, 양쪽 pointer `null`)
  `.last_activation_receipt.json`/`activation_history/`가 생성되지
  않았음을 직접 확인한다.
- `test_valid_journal_with_traversal_lookalike_op_id_prefix_still_rejected`
  — `op_id = "." * 32`(32자 길이는 맞지만 hex가 아님)로 정규식 경계를
  핀 고정한다.

### 실행 결과

```
venv/bin/python -m pytest tests/integration/test_index_lifecycle_fault_injection.py -q
24 passed
```

## 3. CR-I1-MAJ-03 — member size를 확인하기 전에 파일 전체를 무제한 메모리 적재

### 원본 결함

`src/simple_qna_rag/index/verification.py::_read_member_bytes`가
`os.read()`를 EOF까지 무한 루프로 반복해 모든 chunk를 리스트에 쌓고
`b"".join()`한 **뒤에야** manifest의 `size_bytes`와 비교했다. manifest가
선언한 크기가 작아도, 실제 `index.faiss`/`index.pkl` 파일이 그보다 훨씬
크거나(변조) 계속 자라는 파일이면 검증 단계에서 프로세스 메모리를 고갈시킬
수 있었다.

### 수정

`_read_member_bytes`를 `_read_bounded(fd, *, max_bytes) -> bytes`로
교체했다:

```python
def _read_bounded(fd: int, *, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total < max_bytes:
        chunk = os.read(fd, min(_MEMBER_READ_CHUNK, max_bytes - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)
```

`_read_and_verify_member`는 `max_bytes=expected["size_bytes"] + 1`을
전달한다 — 실제 파일이 아무리 크거나 계속 자라도 이 함수는 결코
`expected_size + 1`바이트를 초과해 메모리에 적재하지 않는다. 초과분은
곧바로 `len(data) != expected["size_bytes"]`로 `member_size_mismatch`가
된다(오버사이즈와 정확한-크기를 길이만으로 구분 가능하게 만드는 `+1`의
목적).

추가로 `src/simple_qna_rag/index/manifest.py`에
`MAX_MEMBER_SIZE_BYTES = 8 * 1024**3`(8 GiB)를 신설해
`_require_file_ref`의 `size_bytes` 검증에 상한을 추가했다 — manifest가
애초에 비합리적인 크기를 선언하는 것 자체를 parse 단계에서
`ManifestValueError("invalid_range:{key}.size_bytes")`로 거부한다(bounded
read가 그 선언값을 신뢰하기 전에).

### 신규 테스트 (`tests/unit/test_index_verification.py`)

| 테스트 | 오라클 |
|---|---|
| `test_read_bounded_stops_at_max_bytes_against_a_growing_source` | `os.pipe()` + 무한 background writer 스레드로 "계속 자라는 파일"을 시뮬레이션 — `_read_bounded`가 정확히 `max_bytes`에서 멈추고 블록되지 않음을 확인(growing-file 오라클) |
| `test_read_bounded_returns_short_data_on_genuine_eof` | 진짜 EOF(3바이트 파일)에서 짧은 데이터를 그대로 반환(short-read 오라클) |
| `test_read_bounded_never_requests_more_than_the_bound` | `os.read`를 spy로 감싸 실제 반환된 바이트 총합이 `max_bytes`를 넘지 않음을 확인(read-byte-count 오라클) |
| `test_verify_version_rejects_oversize_member_with_bounded_read` | 실제 published index의 `index.faiss`를 5MB 이상 오버사이즈로 변조 → `member_size_mismatch`가 나면서도 spy 총 read 바이트가 원본 파일 크기 + 5,000,000바이트보다 훨씬 작음을 확인(oversize 오라클) |
| `test_verify_version_rejects_short_member` | `index.pkl`을 1바이트 truncate → `member_size_mismatch` |

## 4. CR-I1-MIN-01 — manifest/current bounded read가 EOF를 확인하지 않음

### 원본 결함

`manifest.json`(`os.read(manifest_fd, MAX_MANIFEST_BYTES)`)과
`current`(`os.read(fd, 4096)`) 모두 단일 `os.read()` 호출로 읽었다. 정확히
read limit 안에서 완결되는 valid JSON 뒤에 추가 바이트가 있는 파일은,
그 추가 바이트가 다음 `os.read()` 호출에만 남아 parser에 전달되지 않으므로
oversized/non-canonical 파일을 완전히 거부하지 못했다(member
hash/settings binding은 계속 검증되므로 즉각적인 pickle trust 우회는
아니지만 strict canonical schema 계약 위반).

### 수정

두 읽기 모두 §3에서 신설한 `_read_bounded(fd, max_bytes=LIMIT + 1)`로
교체했다:

- `raw = _read_bounded(manifest_fd, max_bytes=MAX_MANIFEST_BYTES + 1)` →
  `len(raw) > MAX_MANIFEST_BYTES`면 즉시 `manifest_oversize`.
- `raw = _read_bounded(fd, max_bytes=_MAX_CURRENT_BYTES + 1)`
  (`_MAX_CURRENT_BYTES = 4096`) → 초과 시 `current_pointer_malformed`.

파싱 성공 후에는 **exact canonical-byte 비교**를 추가했다:

```python
canonical = canonical_json_bytes(manifest) + b"\n"
if raw != canonical and raw != canonical[:-1]:
    raise TrustBoundaryError("manifest_non_canonical")
```

(`current`도 동일한 패턴, 기존 reason `current_pointer_malformed` 재사용).
"canonical 또는 마지막 개행 1개만 없는 형태"만 허용한다 — production
writer(`lifecycle.py::_write_fsync(..., canonical_json_bytes(...) +
b"\n")`)가 항상 만드는 정확한 두 형태다. 이 검사는 self-hash는 그대로
통과하지만(재정렬된 key도 `derive_version_id`가 `sort_keys=True`로
재계산하므로 self-hash 자체는 불변) key 순서/공백이 다른 파일이나, 유효한
JSON 뒤에 JSON-legal whitespace만 추가된 파일(즉 `json.loads`는 성공하지만
raw bytes가 canonical과 다른 정확히 그 케이스)을 새로 거부한다 —
`REASONS` frozenset에 `manifest_oversize`/`manifest_non_canonical`을
추가했다.

`verification.py`에서 `manifest.py`의 `canonical_json_bytes`를 새로
import했다.

### 신규 테스트 (`tests/unit/test_index_verification.py`)

| 테스트 | 경계 |
|---|---|
| `test_verify_version_rejects_manifest_larger_than_max_bytes` | 정확히 `MAX_MANIFEST_BYTES + 1`바이트 파일 → `manifest_oversize`(크기 경계) |
| `test_verify_version_rejects_manifest_with_trailing_whitespace_within_limit` | limit 안에 완전히 들어가는, `json.loads`는 성공하지만 canonical과 다른 trailing whitespace → `manifest_non_canonical`(리뷰가 지적한 정확한 시나리오 — 이전엔 통과했을 케이스) |
| `test_verify_version_accepts_manifest_missing_only_the_trailing_newline` | canonical bytes에서 마지막 개행만 없는 형태는 여전히 허용(허용된 leniency 확인) |
| `test_current_pointer_rejects_file_larger_than_max_bytes` | `current`에서 동일한 크기 경계(`_MAX_CURRENT_BYTES + 1`) |
| `test_current_pointer_rejects_trailing_whitespace_within_limit` | `current`에서 동일한 canonical-byte 경계 |

기존 `test_current_pointer_trust_matrix`는 `json.dumps(...)`(공백 포함
비-canonical 형식) 대신 `canonical_json_bytes(...) + b"\n"`으로 `current`를
쓰도록 수정했다 — 실제 production writer와 같은 형식으로 맞춰, 새 canonical
검사가 이 테스트의 다른 의도(symlink/dangling-symlink/unknown-version 분기
확인)를 가리지 않게 했다. 이 테스트가 검증하려는 실패 모드는 canonical
여부가 아니라 symlink/존재하지 않는 버전이므로, 입력 자체는 legitimate
production 형식이어야 그 분기를 정확히 격리할 수 있다.

## 5. 재검증 요약

| 명령 | 결과 |
|---|---|
| `venv/bin/python -m pytest tests/unit/test_index_verification.py -q` | 19 passed(9→19, +10) |
| `venv/bin/python -m pytest tests/unit/test_index_manifest.py tests/unit/test_index_lifecycle.py tests/unit/test_index_lifecycle_cli.py tests/integration/test_index_lifecycle_fault_injection.py -q` | 63 passed(fault-injection 4→24 반영, 나머지 무변경) |
| `venv/bin/python -m pytest tests/unit/test_container_smoke_contract.py -q` | 15 passed(4→15, +11) |
| `venv/bin/python -m pytest tests/unit tests/integration -q` | **1173 passed, 1 skipped**(1132→1173, +41; 1개 skip은 M4.3 무관 pre-existing) |
| `npm test` | 9 passed(무변경) |
| `venv/bin/python -m compileall -q src scripts tests evaluation` | exit 0 |
| `venv/bin/python scripts/check_markdown_links.py` | 파일 113개, 링크 517개, 실패 0개(이 문서 자신을 포함한 최종 실행) |
| `git diff --check` | exit 0 |
| `venv/bin/python scripts/generate_field_spec.py --check` | exit 0(이 finding들은 settings field spec에 영향 없음) |
| `venv/bin/python scripts/logging_callsite_audit.py --check` | exit 0 |
| `venv/bin/python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303` | **exit 0**, 17개 node 전부 `success_count=10/10`(회귀 없음) |
| `venv/bin/python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 --inject-evidence-mismatch` | **exit 1**(negative control 기대 성공), `negative_control={"executed":true,"expected_to_fail":true,"actual_exit_code":1,"result":"REJECTED_AS_EXPECTED"}` |

실행하지 않은 것(작업 지시 경계 유지): Native Linux, Ollama, DDGS,
protected M3 live 14-gate, M4.1 live 14-gate, self-hosted
runner/environment 설정 변경, 실제 hosted GitHub Actions 실행(commit/push
없음), `docker build --target production`(호스트 arm64/디스크 제약은
remediation 이전과 동일 — Implementation_Report.md §3/§8-3 참조).

## 6. M4 release readiness에 대한 영향

이 4개 finding 수정은 모두 M4.3 내부 코드 결함 수정이며,
`M4.1_BLOCKED=true`/protected M3 live `NOT_RUN`/
`overall_release_ready=false` 산출 경로(`scripts/assemble_m4_evidence.py`,
`scripts/check_m4_baseline.py`)를 전혀 건드리지 않았다. 이 값들을
계산하거나 우회하는 코드 경로는 여전히 어디에도 없다(§5의
`assemble_payload_verification`/`baseline_strict_schema` acceptance node가
10/10으로 이를 재확인한다). **M4 전체 release readiness는 이 remediation
이후에도 여전히 BLOCKED다.**
