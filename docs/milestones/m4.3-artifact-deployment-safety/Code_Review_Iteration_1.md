# M4.3 Artifact & Deployment Safety — Code Review Iteration 1

검토자: Fresh Codex independent review worker (M4.3 code review iteration 1)  
기준 revision: `648e3abcca7c321e7f4dd13a7fbed1a4f1886c3e`  
판정: **FAIL — 7.8/10** (`CRITICAL 0`, `MAJOR 3`, `MINOR 1`, `TRIVIAL 0`)

## 검토 범위와 결론

`milestone_dev_orchestration_guide.md`, 이 디렉터리의 Requirement/Plan/Design/
Traceability, Design Review Iteration 1~6, Implementation Report를 읽고 기준 revision
이후 전체 working tree를 독립 검토했다. 구현은 trust-before-pickle의 재오픈 제거,
dirfd 기반 containment, pinned M3 provenance, producer receipt tagged schema,
watchdog `consumer_fenced` 종료, settings/readiness 호환성, protected M3 live job의
trigger/environment block 보존을 대체로 정확하게 구현했다. 특히
`scripts/assemble_m4_evidence.py:283-284,309-311`은 M3 live를 `NOT_RUN`, M4.1을
`BLOCKED`, `M4.1_BLOCKED=true`, `overall_release_ready=false`로 고정하고 checker도
이를 재계산한다.

그러나 hosted container gate가 현재 코드상 반드시 실패하고, crash recovery가
corrupt transition journal을 성공으로 확정할 수 있으며, 검증 전에 선언 크기와
무관하게 artifact 전체를 메모리에 적재하는 신뢰 경계 결함이 있다. Gate 규칙
(`CRITICAL=0`, `MAJOR=0`, score>=9.7)상 FAIL이다. 로컬 Docker 완주 실패는 보고된
host disk exhaustion을 hosted-CI evidence pending으로만 취급했으며 finding으로
세지 않았다.

## Findings

### MAJOR — CR-I1-MAJ-01: container smoke가 정적 자산을 검사하지 않고 항상 `false`를 기록해 hosted gate가 필연적으로 실패한다

- 위치: `scripts/container_smoke.py:149-168`, 특히 `static_ok = False` 이후 값을
  갱신하는 코드가 없고 `result["static_asset_ok"] = static_ok`를 기록한다.
- 영향: `run_smoke()`의 자체 `all_ok`에는 `static_asset_ok`가 빠져 있어
  `status="PASS"`로 종료할 수 있지만, `scripts/assemble_m4_evidence.py:46-52`는
  container payload에서 이 필드가 `True`여야만 producer를 `OK`로 인정한다.
  따라서 실제 Docker build/smoke가 모두 성공해도 `m4-assemble`은
  `PAYLOAD_INVALID`가 되고 deterministic gate는 FAIL한다. 이는 Design §7.5와
  §12가 명시한 정적 자산 회귀 검사를 구현하지 않은 것이다.
- 탐지 공백: `tests/unit/test_container_smoke_contract.py:10-47`의 4개 테스트는
  argv와 Docker 부재 skip만 검사해 `run_smoke`의 receipt 의미론을 전혀 실행하지
  않는다. Implementation Report §6의 축소가 실제 hosted-path 결함을 놓친 직접
  사례이므로 선택적 후속 수준이 아니다.
- remediation: 실제 정적 URL(렌더된 root가 참조하는 vendored asset)을 GET하여
  200/기대 content-type 또는 고정 marker를 검증하고 `static_ok`를 그 결과로
  설정한다. `static_asset_ok`를 `all_ok`에도 포함하고, HTTP를 stub한 단위 테스트와
  production 이미지에서 `COPY web/static` 제거 시 exit 1인 negative test를 추가한다.

### MAJOR — CR-I1-MAJ-02: corrupt transition journal이 검증 없이 PASS receipt/history로 승격된다

- 위치: `src/simple_qna_rag/index/lifecycle.py:338-365::_reconcile_pending_transition`.
  JSON object 여부, exact key set, schema, phase, 32-hex `op_id`, operation enum,
  pre/post pointer 타입/형식, timestamp를 검증하지 않은 채 `record.get()` 및 raw
  필드를 `_append_history`와 PASS receipt에 전달한다.
- 재현: 빈 index root에 schema `wrong`, phase `pointer_committed`, valid-length op-id,
  operation `delete`, `pre_pointer=null`, `post_pointer=null`인 `.transition`을 두고
  `_reconcile_pending_transition()`을 호출했다. 함수는
  `ReconcileReport(outcome='completed', ...)`를 반환하고 operation `delete`, null
  post-pointer인 `outcome="PASS"` receipt를 실제로 기록했다. 다음 history read는
  이 잘못된 row 때문에 신뢰 상태를 손상시킨다.
- 영향: crash recovery라는 mutation 경계가 corrupted/tampered durable state를
  fail-closed로 거부하지 않고 성공한 activation으로 공증한다. 잘못된 `op_id`는
  path 구성에도 직접 쓰여 예외/경로 조작 표면을 만든다. activation atomicity와
  audit evidence를 신뢰할 수 없게 하는 주요 결함이다.
- 탐지 공백: `tests/integration/test_index_lifecycle_fault_injection.py:96-180`은
  정상 형태 journal의 두 축약 crash 상태만 hand-write하며 malformed schema/
  enum/type/key/path cases를 테스트하지 않는다. Design의 넓은 crash/schema
  matrix 축소가 실제 detection gap으로 이어졌다.
- remediation: 별도 strict parser에서 exact schema와 모든 타입/정규식/enum을
  검증한 immutable record만 reconcile에 전달한다. invalid journal은 어떤
  history/receipt/current mutation도 없이 `transition_journal_corrupt`로 거부하고,
  unknown/extra/missing keys, null post, invalid phase/operation/op-id 및 traversal
  문자열을 parameterized negative tests로 고정한다.

### MAJOR — CR-I1-MAJ-03: member size를 확인하기 전에 파일 전체를 무제한 메모리 적재한다

- 위치: `src/simple_qna_rag/index/verification.py:141-165::_read_member_bytes` 및
  `_read_and_verify_member`.
- 영향: manifest가 선언한 `size_bytes`가 작아도 `index.faiss`/`index.pkl`의 EOF까지
  모든 chunk를 list에 누적하고 `b"".join()`한 뒤에야 크기를 비교한다. 신뢰되지
  않은 artifact가 거대한 regular file 또는 계속 커지는 파일이면 검증 단계에서
  프로세스 메모리를 고갈시킬 수 있다. Design §3.3은 선언 크기를 넘는 즉시
  `member_size_mismatch`로 중단한다고 명시했지만 구현이 그 bounded-read 계약을
  빠뜨렸다.
- 탐지 공백: `tests/unit/test_index_verification.py`는 hash mismatch만 다루고
  oversize/short-read/growing-file에서 bounded read를 검증하지 않는다.
- remediation: expected size를 reader에 전달해 최대 `size_bytes + 1`까지만 읽고,
  초과 즉시 거부하며 EOF 전 short size도 거부한다. 합리적인 artifact별 상한도
  manifest parser에서 적용하고, spy/fake fd로 실제 read byte 수가 상한을 넘지
  않음을 테스트한다.

### MINOR — CR-I1-MIN-01: manifest/current bounded read가 EOF를 확인하지 않는다

- 위치: `src/simple_qna_rag/index/verification.py:194-204` (`manifest.json` 한 번의
  `os.read(MAX_MANIFEST_BYTES)`), `:246-267` (`current` 한 번의 `os.read(4096)`).
- 영향: 정확히 read limit 안에서 완결되는 valid JSON 뒤의 추가 bytes가 다음 read에
  남는 특수 구성은 parser에 전달되지 않아 oversized/non-canonical file을 완전하게
  거부하지 못한다. member hashes/settings binding은 계속 검증되므로 즉각적인 pickle
  trust 우회는 아니지만 strict canonical schema 계약 위반이다.
- remediation: limit+1 bytes를 loop/read하여 초과를 명시적으로 거부하고 exact
  canonical bytes(+허용된 단일 newline)도 비교한다. 두 경계값에 대한 tests를 추가한다.

## 확인된 올바른 불변식

- `FAISS.load_local(...allow_dangerous_deserialization=True)`는 legacy loader 한 곳에만
  남고, versioned path는 검증된 bytes에서 직접 deserialize하여 재오픈하지 않는다.
- `resolve_current`는 genuine ENOENT만 legacy fallback 신호로 만들고 symlink는
  fail-closed한다.
- activate/rollback은 동일 `activate()` primitive를 공유하고 pointer replace 전
  transition journal과 parent fsync를 사용한다.
- OCI production stage는 UID/GID 10001, read-only runtime smoke flags, cap drop,
  no-new-privileges, test seam의 production 비포함 구조를 갖는다. 실제 image build/
  layer scan evidence는 hosted CI pending이다.
- `.github/workflows/ci.yml`의 protected `m3-live-regression-gate` block은 기존
  self-hosted labels, environment approval, trusted trigger 조건을 보존한다. 이 리뷰는
  live gate와 runner/environment 설정을 실행·변경하지 않았다.
- intentional `scripts/orchestration_watchdog.py::_classify_runner_error/run_loop` 변경은
  `consumer_fenced`를 bounded reason으로 exact-once journal 후 nonzero 종료하고 generic
  transient만 재시도한다. 관련 16개 watchdog tests가 통과했다.

## 검증 증거

- 안전한 선별 pytest: lifecycle/verification/fault injection/container contract/
  assembler/baseline/workflow/watchdog 총 **62 passed**.
- Implementation Report의 전체 로컬 결과(1132 passed, frontend 9 passed,
  deterministic repeat 10/10)는 문서와 산출 코드를 대조했다. Native Linux,
  Ollama, DDGS, protected M3/M4.1 live gate는 실행하지 않았다.
- `python scripts/check_markdown_links.py`: 아래 최종 실행 결과 참조.
- `git diff --check`: 아래 최종 실행 결과 참조.
- Docker build 미완주는 host disk exhaustion에 따른 hosted evidence pending으로 분류했다.

## 점수와 Gate

기본 구조와 보안 방향은 강하지만, 실제 hosted deterministic chain을 반드시 깨는
container smoke 결함과 crash-recovery trust corruption, unbounded pre-verification read는
release gate 이전에 수정돼야 한다. 축소된 테스트 수 자체가 감점 사유인 것은 아니나,
이번 세 finding은 축소가 핵심 negative/semantic 경로를 탐지하지 못했다는 구체적
증거다. 최종 점수 **7.8/10**, Gate **FAIL**.
