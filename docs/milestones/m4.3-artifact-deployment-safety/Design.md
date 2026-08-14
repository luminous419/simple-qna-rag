# M4.3 Artifact & Deployment Safety 상세 설계

상태: **상세 설계 — 구현 전, 독립 리뷰 Iteration 4 반영 (Iteration 5 개정)**
요구사항: [Requirement.md](Requirement.md)
계획: [Plan.md](Plan.md)
추적표: [Traceability.md](Traceability.md)
설계 리뷰: [Design_Review_Iteration_1.md](Design_Review_Iteration_1.md)(FAIL 5.8/10,
CRITICAL 1·MAJOR 7·MINOR 1 — §16 closure matrix 참조),
[Design_Review_Iteration_2.md](Design_Review_Iteration_2.md)(FAIL 8.4/10,
CRITICAL 0·MAJOR 5·MINOR 1 — §17 closure matrix 참조),
[Design_Review_Iteration_3.md](Design_Review_Iteration_3.md)(FAIL 8.2/10,
CRITICAL 0·MAJOR 5·MINOR 1 — §18 closure matrix 참조),
[Design_Review_Iteration_4.md](Design_Review_Iteration_4.md)(FAIL 9.1/10,
CRITICAL 0·MAJOR 2·MINOR 1 — 이번 개정이 전부 반영, §19 closure matrix 참조)
기준 revision: `648e3ab` (`master`, M4.2 merge)
작성자: Claude Code Sonnet 5 (상세 설계자, 개발 orchestration guide §"에이전트의 역할")

구현 상태: **구현 완료 — 로컬 결정론적 검증 PASS**. 이 설계(Iteration 6, DR-I6-MIN-01
포함)를 코드/테스트/workflow/runbook으로 구현했다. 실제 구현이 이 문서의 계획과
다른 지점(예: `build_manifest`의 identity-field 키 집합에서 `schema_version` 제외,
`_fd_relative_rmtree`의 디렉터리 재-chmod 추가, `run_m43_acceptance.py`의
`negative_control` 5-key→4-key 수정)은 [Implementation_Report.md](Implementation_Report.md)에
근거와 함께 기록했다 — 이 문서 본문은 설계 당시 계획을 그대로 보존하고 별도로
수정하지 않는다. hosted GitHub Actions receipt는 이 작업 트리가 commit/push되지
않았으므로 아직 존재하지 않는다(`NOT_RUN`) — M4.1 운영 `BLOCKED`, protected M3 live
`NOT_RUN`, `overall_release_ready=false`는 구현 이후에도 그대로 보존된다.

이 문서는 코드를 작성하지 않는다. 모든 심볼·상태 머신·syscall 순서·스키마·테스트는
Phase 1(prototype)에서 검증된 OS/Python 원시 동작을 근거로 하며, 구현 phase가
그대로 채택할 수 있는 수준까지 구체화한다. 검증한 원시 동작(§0.3)과 기존
648e3ab 코드의 실제 인용(§0.2)은 모두 이 세션에서 직접 읽거나 실행해 확인했다.

## 0. 범위, 근거와 설계 원칙

### 0.1 두 판정 경계 (Requirement §1 재확인)

이 설계는 **M4.3 deterministic cycle**만 PASS시킬 수 있다. **전체 M4 release
readiness**는 M4.1 post-merge live 14-gate와 protected M3 live gate가 실제
receipt로 해소될 때까지 `BLOCKED`로 유지된다. 아래 모든 심볼·job·스크립트는
이 분리를 구조적으로 강제한다 — `check_m4_baseline.py`(§9)가 이 불변식의
유일한 판정 지점이며, 다른 어떤 파일도 `overall_release_ready`를 계산하지
않는다.

### 0.2 648e3ab 기준 코드 인용 (Phase 0 동결 스냅샷)

- `src/simple_qna_rag/cli/index_documents.py::create_vectorstore`(L189-193,
  L250-252) — 기존 `VECTORSTORE_PATH`가 있으면 `shutil.rmtree`로 삭제 후
  `vectorstore.save_local(VECTORSTORE_PATH)`로 직접 덮어쓴다. 실패 시 이전
  index가 사라진다.
- `src/simple_qna_rag/rag_engine.py::_load_vectorstore`(L169-191) —
  `os.path.exists(VECTORSTORE_PATH)`만 확인한 뒤 바로
  `FAISS.load_local(VECTORSTORE_PATH, embeddings,
  allow_dangerous_deserialization=True)`를 호출한다. provenance/hash/symlink
  검증이 없다.
- `src/simple_qna_rag/settings.py::FIELD_SPECS` — `VECTORSTORE_PATH`(L275-284)는
  `_PACKAGE_ROOT/runtime/vectorstore` 기본값, `SIMPLE_QNA_RAG_VECTORSTORE_DIR`
  env alias, consumers `cli/index_documents.py, rag_engine.py`를 갖는
  `pydantic.create_model` 기반 frozen `Settings`의 한 필드다. `config.py`가
  `globals()[_spec.name] = facade_value(_settings, _spec)`로 레거시 상수를
  투영한다.
- `.github/workflows/ci.yml` — `python-tests`, `frontend-tests`(hosted,
  무조건), `m3-live-regression-gate`(protected, `runs-on: [self-hosted,
  ollama-m3]`, `environment: m3-live-regression`, trigger는 push-to-master
  또는 승인된 workflow_dispatch로만 제한)만 존재한다. `container`,
  `m43-deterministic`, `m4-assemble` job은 없다.
- `evaluation/baselines/m3_initial.json::reproducibility.vectorstore_fingerprint`
  — `index_faiss_sha256`/`index_pkl_sha256` 두 필드가 M3 승인 hash pair다
  (legacy import의 유일한 신뢰 근거, REQ-002.1).
- `scripts/orchestration_watchdog.py`의 working-tree delta(§11) — tracked base
  `e57fe1c` 대비 `inspect_run()`이 `payload["coordinator"]["terminal"]`을
  먼저 읽어 `task-list --run <run> --from <terminal> --brief --json`과
  `check --terminal <terminal> --run <run> --peek --json`으로 바꾼다(이전은
  `--run`만 사용하는 run-only 조회였다).
- `tests/unit/test_orchestration_watchdog.py`(9개 기존 케이스) — `FakeRunner`가
  `"task-list" in joined`/`"check" in joined` 부분 문자열만 확인하고, 정확한
  argv, terminal-scope 격리, runner 예외 전파는 검증하지 않는다.

### 0.3 Phase 1 prototype로 검증한 원시 동작 (설계 근거)

이 세션에서 저장소 밖 scratchpad에 분리된 1회성 스크립트로 직접 실행해
확인했다(구현물이 아니며 커밋 대상이 아니다):

1. `os.open(symlink_path, os.O_RDONLY | os.O_NOFOLLOW)` → `OSError(ELOOP)`.
   symlink 최종 경로 요소를 OS 수준에서 차단하는 `contained_open`(§3.2)의
   근거.
2. `json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",",
   ":"))`를 1000회 재직렬화해도 SHA-256이 동일 — canonical manifest(§2)의
   결정론성 근거(REQ-001.3, NFR-001).
3. `json.loads(text, parse_constant=lambda v: ...raise ValueError...)`가
   `NaN`/`Infinity`를 예외로 거부 — manifest parser의 non-finite 거부(REQ-001.4)
   근거이자 `run_m42_acceptance.py::harvest_node_receipt`가 이미 쓰는 동일
   패턴의 재사용.
4. 디렉터리 fd를 열어 `os.fsync(fd)`가 성공 — parent directory fsync(REQ-003.2)의
   구현 가능성 근거.

cross-device `os.rename` → `OSError(errno.EXDEV)`는 이 환경에서 두 번째
파일시스템을 만들 수 없어 직접 재현하지 않았다. 이는 POSIX 표준 동작이며
`shutil.move`의 fallback 코드에서도 이미 이 계약에 의존한다 — §4.5에서
`errno.EXDEV`를 명시적으로 catch하는 설계로 반영한다.

### 0.4 설계 원칙

1. **trust-before-pickle, reopen 없음**: `index/verification.py`는
   `FAISS.load_local`을 전혀 호출하지 않는다(§3.4, Iteration 1 CRIT-01
   반영). 검증된 version에서 pickle을 되살리는 유일한 코드 경로는
   `load_verified_faiss`가 `verify_version()`이 반환한 **동일한 검증
   bytes**(`VerifiedVersion.faiss_bytes`/`pkl_bytes`)에서 직접
   `faiss.deserialize_index`/`pickle.loads`를 호출하는 것뿐이며, 이
   경로는 검증 이후 파일/디렉터리를 다시 열지 않는다 — TOCTOU 창 자체가
   구조적으로 없다. `FAISS.load_local(..., allow_dangerous_deserialization=True)`
   호출은 코드베이스 전체에서 정확히 한 곳,
   `rag_engine.py::_load_vectorstore_legacy`(648e3ab와 바이트 동일,
   REQ-009.1)에만 남는다 — `grep -rn "load_local" src/` 결과가 1건이어야
   하는 것이 이 불변식의 감사 명령이다(§3.4, §10).
2. **단일 activation primitive 재사용**: activate와 rollback은 같은 함수
   `index/lifecycle.py::activate()`를 호출한다(§4.4). rollback 전용 pointer
   교체 코드는 만들지 않는다.
3. **호환성 브리지**(§5) — `VECTORSTORE_PATH`는 이름·기본값·consumers 그대로
   보존한다. 새 `INDEX_ROOT` 필드를 추가하고, `INDEX_ROOT/current`가 없으면
   `_load_vectorstore`가 648e3ab와 바이트 단위로 동일한 legacy 경로로
   폴백한다. 기존 M4.1/M4.2 테스트·self-hosted runner·`preflight_vectorstore.py`
   등 어떤 기존 소비자도 수정하지 않는다(REQ-009.1, NFR-004).
4. **receipt는 고정 vocabulary만 포함**한다 — 예외 원문, 절대경로, 문서 내용,
   secret은 어떤 receipt/manifest/로그에도 쓰지 않는다(M4.1 REPLACE 관례
   계승, REQ-004.2).
5. **fail-closed는 코드가 아니라 테스트가 증명**한다 — 모든 신규 negative
   control은 "동일 parser가 실제로 실패시키는" 것을 assert한다(문자열 매칭이
   아니라 exit code/예외 재현).

## 1. Symbol Inventory

### 1.1 신규 파일

| 파일 | 역할 |
|---|---|
| `src/simple_qna_rag/index/__init__.py` | 패키지 마커 |
| `src/simple_qna_rag/index/manifest.py` | canonical manifest schema, version ID 파생 (§2) |
| `src/simple_qna_rag/index/verification.py` | contained-open, trust-before-pickle (§3) |
| `src/simple_qna_rag/index/lifecycle.py` | staging/activate/rollback/cleanup 상태 머신, lock (§4) |
| `src/simple_qna_rag/cli/index_lifecycle.py` | `simple-qna-rag-index-lifecycle` CLI (§6) |
| `tests/support/simple_qna_rag_test_seam/__init__.py` | 패키지 마커 — **`src/` 밖**에 있으므로 production Dockerfile의 `COPY src/ ./src/`가 물리적으로 포함하지 않는다(§5.2-a, DR-I3-MAJ-02) |
| `tests/support/simple_qna_rag_test_seam/deterministic_embeddings.py` | 컨테이너 mock smoke용 결정론적 embedding test seam — `DeterministicTestEmbeddings`(§5.2-a). `src/`에서 이전(구 위치 `src/simple_qna_rag/deterministic_embeddings.py`는 DR-I2-MAJ-02가 만들었으나 DR-I3-MAJ-02가 production 물리적 부재 요구로 재배치) |
| `deploy/Dockerfile` | test/production multi-stage (§7.1) |
| `.dockerignore` | 빌드 컨텍스트 제외 목록 (§7.2) |
| `scripts/scan_image_layers.py` | OCI layer/traversal/whiteout 스캐너 (§7.4) |
| `scripts/container_smoke.py` | 컨테이너 보안/mock smoke 실행기 (§7.5) |
| `scripts/run_m43_acceptance.py` | M4.3 결정론적 acceptance repeat runner (§8.4) |
| `scripts/assemble_m4_evidence.py` | fresh-dir evidence 조립기 (§8.2) |
| `scripts/check_m4_baseline.py` | baseline state algebra 검사기 (§9.2) |
| `scripts/deploy_drill.py` | mock 배포/복구 drill 실행기 (§7.7) |
| `docs/operations/deployment_runbook.md` | 배포 runbook (§7.6) |
| `docs/operations/recovery_runbook.md` | 복구 runbook (§7.6) |
| `tests/unit/test_index_manifest.py` | manifest canonicalization/negative 테스트 |
| `tests/unit/test_index_verification.py` | contained-open/trust-boundary 테스트 |
| `tests/unit/test_index_lifecycle.py` | legacy import/retention 단위 테스트 |
| `tests/unit/test_pinned_baseline_provenance.py` | pinned baseline 상수 ↔ tracked M3 baseline 파일 대조 회귀 (§4.7, DR-I2-MIN-06) |
| `tests/unit/test_rag_engine_embeddings.py` | `_build_embeddings` provider 분기, 기본값이 huggingface임을 고정 (§5.2-a, DR-I2-MAJ-02) |
| `tests/integration/test_index_lifecycle_fault_injection.py` | staging/activate/rollback fault matrix, 100회 반복, crash history/receipt exact-once (§4.4-a-1, DR-I2-MAJ-01) |
| `tests/unit/test_index_lifecycle_cli.py` | lifecycle CLI exit code/receipt 테스트 |
| `tests/unit/test_scan_image_layers.py` | layer scanner positive/negative fixture |
| `tests/unit/test_container_smoke_contract.py` | container_smoke.py 파서 계약 테스트 (docker 없이) |
| `tests/unit/test_assemble_m4_evidence.py` | assembler negative control 전수 |
| `tests/unit/test_check_m4_baseline.py` | baseline enum/algebra 테스트 |
| `tests/unit/test_m43_acceptance_runner.py` | run_m43_acceptance.py 계약 테스트 |
| `tests/unit/test_deploy_drill.py` | mock drill 중단점 테스트 |

### 1.2 변경 파일 (MODIFIED)

| 파일 | 변경 요지 |
|---|---|
| `src/simple_qna_rag/settings.py` | `INDEX_ROOT`/`EMBEDDING_PROVIDER`/`ALLOW_TEST_EMBEDDING` FieldSpec과 2-키 test-seam validator 추가(§5.1) — 이 검증은 accidental-default-activation을 막는 convenience 계층일 뿐, production 활성화를 막는 신뢰 경계는 `_build_embeddings()`의 물리적 import 실패다(§5.2-a, DR-I3-MAJ-02). 기존 필드 무변경 |
| `src/simple_qna_rag/rag_engine.py` | `_load_vectorstore` 분기(§5.2): INDEX_ROOT/current 있으면 verified 경로, 없으면 기존 legacy 코드를 `_load_vectorstore_legacy`로 추출해 그대로 재사용. `_build_embeddings()` 헬퍼 추가(§5.2-a) — `TestEmbeddingSeamUnavailable` 예외를 `IndexTrustError`와 같은 방식으로 잡아 `artifact_error_reason="test_embedding_seam_unavailable"`을 설정(DR-I3-MAJ-02, §5.3 readiness 재사용) |
| `tests/unit/test_settings_inventory.py` | test-seam validator 2-key 게이트 negative case, 기본값 회귀 case 추가(§5.1) |
| `src/simple_qna_rag/observability/health.py` | `evaluate_readiness`에 옵션 인자 `artifact_error_reason: str \| None = None` 추가(§5.3). 기본값 유지로 기존 호출부 무변경 |
| `src/simple_qna_rag/web/server.py` | lifespan에서 `app.state.engine_artifact_reason` 설정 한 줄 추가(§5.3) |
| `.github/workflows/ci.yml` | `container`, `m43-deterministic`, `m4-assemble` job 추가, `python-tests`/`frontend-tests`에 evidence 발행 step 1개씩 추가. `m3-live-regression-gate` 블록은 텍스트 무변경(§8) |
| `scripts/orchestration_watchdog.py` | terminal-scoping은 이미 완료된 M4.3 readiness fix(§11.1) — 그 delta는 무변경. `_classify_runner_error` 헬퍼와 `main()` 호출부 추가(§11.1, DR-I2-MAJ-05 신규), `run_loop`의 `except Exception` 분기를 `consumer_fenced`(즉시 nonzero 종료)/generic transient(기존 재시도)로 나누는 제어 흐름 확장(§11.1, DR-I3-MAJ-05 신규) — 모두 기존 코드 삭제 없는 순수 추가/분기 확장 |
| `tests/unit/test_orchestration_watchdog.py` | exact argv/read-only peek/fail-closed 테스트 8종 추가(§11.2) |
| `docs/generated/settings_field_spec.md` | `generate_field_spec.py --check` 재생성 산출물 (구현 phase 실행) |

## 2. Index root 디렉터리 레이아웃과 권한 모델

```text
<INDEX_ROOT>/
  current                      # 일반 파일(심볼릭 링크 아님). {"schema_version":1,"version_id":"<16-hex>"}
  activation_history/          # 성공한 activate/rollback 1건당 1개의 불변 레코드 파일
    <op_id>.json                 # temp-write/fsync/atomic-rename/parent-fsync로 커밋
                                  # (§4.4-a-1, DR-I3-MAJ-01 — 더 이상 append-only
                                  # 단일 파일이 아니다)
  versions/
    <version-id>/               # publish 후 불변. dir 0o555, 내부 파일 0o444
      index.faiss
      index.pkl
      manifest.json
  .staging/
    <operation-id>/              # 작업 중에만 존재. 0o700. 실패 시 inactive로 잔류(§4.2)
  .lock                          # 0바이트, flock 대상. 콘텐츠 없음
```

- `current`을 symlink가 아닌 일반 파일로 정한 이유: REQ-001.4가 symlink를
  전 영역에서 거부 대상으로 두므로, activation pointer 자체를 symlink로
  만들면 "pointer는 예외"라는 별도 규칙이 필요해진다. 일반 파일 +
  `os.replace()` atomic rename은 symlink와 동일한 원자성을 제공하면서
  예외를 만들지 않는다.
- `version-id`는 `^[0-9a-f]{16}$` 정규식으로 고정한다(§2.2). `current` 파싱 시
  이 정규식을 만족하지 않으면 `verification.py`가 `current_pointer_malformed`로
  거부하고, 만족해도 `versions/<version-id>/`가 없으면
  `current_pointer_unknown_version`으로 거부한다 — pointer 파일 손상이
  path traversal로 이어지지 않는다.
- `versions/<id>/`는 publish 직후 `os.chmod(0o444)`(파일)/`os.chmod(0o555)`(디렉터리)로
  방어적 read-only화한다. 이는 "절대 재기록하지 않는다"는 설계 불변식의
  운영체제 수준 보강이며, 유일한 신뢰 근거는 아니다(진짜 근거는 파일이
  다시 열리지 않는다는 코드 구조 자체).
- `.staging/<operation-id>/`는 `uuid4().hex`로 명명해 최종 `version-id`와
  절대 겹치지 않게 한다(해시 계산 전에 디렉터리를 먼저 만들어야 하므로).

### 2.1 canonical manifest 스키마 (`index/manifest.py`)

```python
MANIFEST_SCHEMA_VERSION = 1

# 자기참조/실행 metadata로 제외되는 필드 (NFR-001)
EXCLUDED_FROM_IDENTITY = ("version_id", "created_at")

REQUIRED_KEYS = frozenset({
    "schema_version", "version_id", "created_at",
    "corpus_manifest_sha256",       # legacy_import면 None 허용
    "source_document_count", "chunk_count",
    "embedding_model_name", "embedding_model_revision", "embedding_provider",
    "normalize_embeddings", "chunk_size", "chunk_overlap",
    "faiss_index_type", "faiss_dimension",
    "settings_hash", "dependency_lock_sha256",
    "builder_git_sha", "builder_git_dirty",
    "index_faiss", "index_pkl",       # each: {"size_bytes": int, "sha256": <64-hex>}
    "source",                          # "build" | "legacy_import"
    "legacy_baseline_id",              # str | None
})
```

`build_manifest(identity_fields: dict, created_at: str) -> dict`:
1. `identity_fields`의 키 집합이 `REQUIRED_KEYS - EXCLUDED_FROM_IDENTITY`와
   정확히 같은지 확인(초과/누락 모두 `ManifestSchemaError`).
2. `version_id = derive_version_id(identity_fields)`.
3. `{**identity_fields, "version_id": version_id, "created_at": created_at,
   "schema_version": MANIFEST_SCHEMA_VERSION}`를 반환.

`derive_version_id(identity_fields: dict) -> str`:
```python
def derive_version_id(identity_fields: dict) -> str:
    payload = {"schema_version": MANIFEST_SCHEMA_VERSION, **identity_fields}
    raw = canonical_json_bytes(payload)
    return hashlib.sha256(raw).hexdigest()[:16]
```

`canonical_json_bytes(obj: dict) -> bytes`:
```python
def canonical_json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")
```
파일에 쓸 때는 `canonical_json_bytes(manifest) + b"\n"`(trailing newline
정책, `dependency_snapshot.py`의 `text + "\n"` 관례와 동일).

`parse_manifest(raw: bytes) -> dict` (strict reader, §5 gate의 근거):
1. `json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)` —
   `NaN`/`Infinity`/`-Infinity`는 `ManifestValueError("non_finite")`.
2. 최상위가 `dict`가 아니거나 키 집합이 `REQUIRED_KEYS`와 다르면
   (초과 키 포함) `ManifestSchemaError("unknown_or_missing_key")`.
3. `schema_version != MANIFEST_SCHEMA_VERSION` → `ManifestSchemaError`.
4. 타입 검사: `chunk_size`/`chunk_overlap`/`source_document_count`/
   `chunk_count`/`faiss_dimension`은 `int`이고 `>0`(chunk_overlap은 `>=0`);
   `normalize_embeddings`/`builder_git_dirty`는 `bool`; `embedding_provider`는
   `{"huggingface", "deterministic_test"}`의 원소인 `str`(§5.1 Settings
   enum과 동일 vocabulary, DR-I2-MAJ-02); `index_faiss`/
   `index_pkl`은 `{"size_bytes": int>=0, "sha256": 64-hex}` 형태만 허용
   (정규식 `^[0-9a-f]{64}$`); `settings_hash`/`dependency_lock_sha256`/
   `builder_git_sha`도 각각 고정 길이 hex 정규식(git sha는 40-hex).
5. `version_id`가 `^[0-9a-f]{16}$`을 만족하고, 동시에
   `derive_version_id({k: v for k, v in manifest.items() if k not in
   EXCLUDED_FROM_IDENTITY})`와 정확히 같은지 재계산 검증
   (`ManifestValueError("self_hash_mismatch")`) — manifest 파일 자체가
   변조돼도 자기 무결성이 깨진다.
6. `source == "legacy_import"`이면 `legacy_baseline_id`는 non-null 문자열이어야
   하고 `corpus_manifest_sha256`는 `None`이어야 한다(legacy는 corpus 원본을
   추적하지 않음). `source == "build"`이면 반대.

`round_trip_stable(manifest: dict, iterations: int = 100) -> bool` — 테스트
전용 헬퍼. `canonical_json_bytes(manifest)`를 100회 재직렬화해 매번 동일
바이트인지 확인한다(§0.3-2에서 1000회로 이미 원시 동작을 검증했으므로 100회는
회귀 게이트일 뿐 신규 위험 탐색이 아니다).

### 2.2 version ID 파생과 identity 필드 결정

`identity_fields`는 실제 스테이징된 파일에서 계산한다 — 절대 하드코딩하지
않는다(REQ-002.6 계열 원칙과 동일한 "매직 넘버 금지" 관례를 이 파일에도
적용):

| 필드 | 출처 |
|---|---|
| `corpus_manifest_sha256` | `evaluation.reporting.build_corpus_manifest(DATA_DIR)["manifest_sha256"]`(build만) |
| `source_document_count`/`chunk_count` | 실제 로드/분할 결과 길이 |
| `embedding_model_name` | `settings.EMBEDDING_MODEL_NAME` |
| `embedding_model_revision` | `HuggingFaceEmbeddings`가 노출하지 않으므로 `"unknown"` 고정값(모델 카드 revision API가 없음 — 잔여 위험으로 §11에 기록) |
| `embedding_provider` | `settings.EMBEDDING_PROVIDER`(§5.1, DR-I2-MAJ-02) — `_verify_settings_binding`(§3.3)이 이 필드로 host build와 런타임 provider 일치를 검증한다 |
| `normalize_embeddings`/`chunk_size`/`chunk_overlap` | 해당 settings 필드 |
| `faiss_index_type` | `type(index).__name__`(예: `"IndexFlatIP"`) |
| `faiss_dimension` | `index.d` |
| `settings_hash` | `hashlib.sha256(canonical_json_bytes(_settings_binding_snapshot())).hexdigest()`(§5.4) |
| `dependency_lock_sha256` | `dependency_snapshot.py::_lock_sha256_canonical(LOCK_FILE.read_text())`(재계산, 파일 읽기만 재사용) |
| `builder_git_sha`/`builder_git_dirty` | `evaluation.reporting._git_commit()`/`_git_dirty()` 재사용(이미 공개 함수는 아니므로 동일 로직을 `index/manifest.py`에 4줄 복제 — 기존 코드베이스가 `fingerprint.py`/`reporting.py` 사이에서도 같은 방식으로 작은 유틸을 중복하는 선례를 따름) |
| `index_faiss`/`index_pkl` | 스테이징된 실제 파일의 `os.stat().st_size`와 `hashlib.sha256(파일 바이트)` |
| `source`/`legacy_baseline_id` | 호출자(`build()` vs `import_legacy()`)가 전달 |

## 3. `index/verification.py` — contained-open과 trust-before-pickle

### 3.1 고정 reason vocabulary

`vector_index.py::VectorIndexValidationError.reason`(기존 M3 코드에 이미 있는
패턴, 148행 `except VectorIndexValidationError as exc: ... reason=exc.reason`)을
그대로 계승한다.

```python
class TrustBoundaryError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

REASONS = frozenset({
    "manifest_missing", "manifest_schema_invalid", "manifest_self_hash_mismatch",
    "version_dir_missing", "version_dir_not_directory", "version_dir_symlink",
    "extra_or_missing_member", "member_not_regular_file", "member_is_symlink",
    "member_mode_forbidden", "member_owner_forbidden",
    "member_size_mismatch", "member_hash_mismatch",
    "root_escape", "cross_device_staging",
    "settings_mismatch", "current_pointer_malformed",
    "current_pointer_unknown_version",
    "current_pointer_symlink",          # 신규(DR-I1-MAJ-02): symlink/dangling
                                         # current는 "없음"과 다른 trust failure
    "staging_entry_untrusted",          # 신규(DR-I1-MAJ-04): .staging 이름 불일치/
                                         # symlink/owner 위반 — 삭제 후보에서 제외
    # legacy_baseline_tampered/legacy_baseline_untracked는 DR-I2-MIN-06
    # 수정으로 제거됐다 — import_legacy가 더 이상 런타임에 baseline
    # 파일을 읽지 않고(§4.7) 코드 상수만 참조하므로 "런타임에 발견한
    # baseline 파일 손상"이라는 실행 경로 자체가 없다. source_dir가
    # 승인 pair와 다르면 기존 member_hash_mismatch로 거부된다.
    "lock_file_untrusted",              # 신규(DR-I1-MAJ-05): .lock이 기존에 symlink/
                                         # non-regular/wrong-owner로 존재
    "transition_journal_corrupt",       # 신규(DR-I1-MAJ-05): reconcile 시 저널이
                                         # 파싱 불가 — 수동 개입 필요
    "activation_history_record_corrupt",  # 신규(DR-I3-MAJ-01): rename으로 이미
                                         # 완결된(fsync 통과) activation_history/
                                         # 레코드 파일이 파싱 불가 — "진행 중 write"가
                                         # 아니라 사후 손상/변조이므로 조용히 건너뛰지
                                         # 않고 fail-closed한다(§4.4-a-1)
    "activation_history_schema_invalid",  # 신규(DR-I4-MAJ-01): 완결된 레코드의
                                         # key 집합이 정확히
                                         # `_HISTORY_REQUIRED_KEYS`가 아니거나,
                                         # 타입/`schema` 리터럴/`sequence`
                                         # non-negative-int/pointer 정규식 중
                                         # 하나라도 어긋난다(§4.4-a-1)
    "activation_history_filename_op_id_mismatch",  # 신규(DR-I4-MAJ-01):
                                         # `<op_id>.json` 파일 이름의 op_id와
                                         # 본문 `op_id` 필드가 다르다 —
                                         # rename 대상이 바뀌었거나 내용이
                                         # 사후 치환됐다는 신호(§4.4-a-1)
    "activation_history_operation_invalid",  # 신규(DR-I4-MAJ-01): `operation`이
                                         # `{"activate","rollback"}` enum 밖의
                                         # 값이다(§4.4-a-1)
    "activation_history_sequence_invalid",  # 신규(DR-I4-MAJ-01): `sequence`
                                         # 값에 중복이 있거나 `0..N-1` 연속
                                         # 정수가 아니다(gap) — ordering
                                         # oracle이 단순 정렬이 아니라
                                         # uniqueness/contiguity를 직접
                                         # 검증한다(§4.4-a-1)
    "activation_history_current_mismatch",  # 신규(DR-I4-MAJ-01): 최신
                                         # (sequence 최대) committed record의
                                         # `post_pointer`가 실제 `current`와
                                         # 다르다 — history가 current를 만든
                                         # operation과 다른 상태를 가리키면
                                         # `previous`를 신뢰하지 않고
                                         # fail-closed한다(§4.4-a-1)
    "test_embedding_seam_unavailable",  # 신규(DR-I3-MAJ-02): production 이미지에는
                                         # deterministic test embedding 모듈이 물리적으로
                                         # 없으므로 두 env var를 설정해도 import가
                                         # ModuleNotFoundError로 실패한다 — readiness가
                                         # 503 artifact_test_embedding_seam_unavailable로
                                         # fail-closed한다(§5.2-a)
})
```

### 3.2 `ContainedDir` — root-to-leaf dirfd 체인, symlink/TOCTOU 완전 차단 (DR-I1-CRIT-01 반영)

Iteration 1 CRIT-01은 이전 `contained_open`이 조상 디렉터리를 **path
기반으로 사전 검사**한 뒤 최종 요소만 `O_NOFOLLOW`로 여는 구조였음을
지적했다 — 사전 검사와 실제 open 사이, 그리고 여러 멤버를 같은
디렉터리에서 연속으로 열 때 그 디렉터리 자체가 검사 이후 교체될 수
있는 창이 남는다. 아래 설계는 **root부터 leaf까지 dirfd 체인을 열어
고정**하고, 한 디렉터리에서 여러 멤버를 열 때도 항상 같은 dirfd를
재사용해 "검사한 그 디렉터리"와 "실제로 읽는 그 디렉터리"가 fd
identity 수준에서 동일함을 강제한다.

```python
@dataclass
class ContainedDir:
    """열린 디렉터리 fd 하나를 감싼다. 이 fd가 가리키는 디렉터리는 생성 시점에
    O_DIRECTORY|O_NOFOLLOW로 고정됐으므로, 이후 이 fd를 통해 여는 모든 하위
    항목은 그 시점의 디렉터리 identity에 상대적이다 — 이름을 다시 path로
    풀어 재조회하지 않는다(TOCTOU 창 제거)."""
    fd: int

    def open_subdir(self, name: str) -> "ContainedDir":
        if "/" in name or name in (".", ".."):
            raise TrustBoundaryError("root_escape")
        try:
            fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                         dir_fd=self.fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise TrustBoundaryError("version_dir_symlink") from None
            if exc.errno == errno.ENOENT:
                raise TrustBoundaryError("version_dir_missing") from None
            if exc.errno == errno.ENOTDIR:
                raise TrustBoundaryError("version_dir_not_directory") from None
            raise
        return ContainedDir(fd)

    def open_member(self, name: str, *, allowed_mode_mask: int = 0o022,
                     expected_owner_uid: int | None = None,
                     missing_reason: str = "manifest_missing") -> int:
        """이 디렉터리 fd에 상대적으로 최종 파일 요소를 O_NOFOLLOW로 연다.
        같은 ContainedDir에서 연속으로 여러 멤버를 열어도 매번 같은 검증된
        디렉터리를 상대로 열리므로, 멤버 사이에 디렉터리가 교체될 수 없다."""
        if "/" in name or name in (".", ".."):
            raise TrustBoundaryError("root_escape")
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise TrustBoundaryError("member_is_symlink") from None
            if exc.errno == errno.ENOENT:
                raise TrustBoundaryError(missing_reason) from None
            raise
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            raise TrustBoundaryError("member_not_regular_file")
        if st.st_mode & allowed_mode_mask:
            os.close(fd)
            raise TrustBoundaryError("member_mode_forbidden")
        if expected_owner_uid is not None and st.st_uid != expected_owner_uid:
            os.close(fd)
            raise TrustBoundaryError("member_owner_forbidden")
        return fd

    def listdir(self) -> list[str]:
        return os.listdir(self.fd)   # os.listdir(fd)는 내부적으로 dup(fd)를
                                      # fdopendir에 넘기므로 self.fd는 열린
                                      # 채로 남는다 — CPython posixmodule 계약

    def close(self) -> None:
        os.close(self.fd)


def open_contained_root(root: Path) -> ContainedDir:
    """index_root 자체를 O_NOFOLLOW로 연다 — root 자체가 symlink 체인이면
    여기서 즉시 실패한다(예전 설계의 별도 realpath 이중 검사를 대체)."""
    try:
        fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise TrustBoundaryError("root_escape") from None
        raise
    return ContainedDir(fd)
```

- **root-to-leaf 보증**: `open_contained_root(index_root)` →
  `.open_subdir("versions")` → `.open_subdir(version_id)`가 만드는 세
  fd는 각각 O_NOFOLLOW로 그 시점에 고정된다. 어느 단계에서든 심볼릭
  링크면 그 즉시 실패하고, 성공한 fd는 이후 프로세스가 그 디렉터리를
  삭제·rename해도 커널이 유지한다(POSIX open-fd 불변식) — "검사 후
  다른 디렉터리로 교체" 공격이 구조적으로 불가능하다.
- **같은 디렉터리에서 여러 멤버를 열 때의 안전성**: `verify_version()`은
  `manifest.json`/`index.faiss`/`index.pkl` 세 멤버를 모두 **같은
  `version_dir: ContainedDir` 인스턴스**(즉 같은 fd)에서 연다(§3.3).
  이전 설계처럼 매번 `index_root`부터 path를 다시 걸어가지 않으므로
  세 번의 open 사이에 `versions/<id>`가 교체될 여지가 없다.
  `allowed_mode_mask=0o022`는 group/other write bit를 거부한다(owner
  write는 build 직후 파일에 남아있을 수 있으므로 publish 시
  `chmod 0o444`로 제거하고, verification은 이를 재확인하는 이중
  계층이다).
- owner 검사(`member_owner_forbidden`)는 컨테이너/self-hosted 배포에서만
  의미 있는 선택적 파라미터로 `verify_version()`이 넘긴다(§3.3) — CI
  hosted runner는 UID가 매 run 달라질 수 있으므로 CI에서는
  `expected_owner_uid=None`(검사 생략)로 호출하고, 컨테이너 런타임은
  `expected_owner_uid=10001`(§7.1)로 호출한다.
- **음수 테스트**: `tests/unit/test_index_verification.py`는 racer
  스레드가 `verify_version()` 실행 도중(각 `open_subdir`/`open_member`
  호출 사이) `versions/<id>` 디렉터리를 다른 디렉터리로 rename하거나
  개별 멤버를 symlink로 교체하는 fixture를 포함한다. dirfd 체인 덕분에
  이미 연 fd는 racer의 rename에 영향받지 않으므로, 이 테스트는 "racer가
  아무 효과도 못 낸다"(반환된 `VerifiedVersion`이 racer 이전 콘텐츠와
  동일)를 assert하는 방식으로 통과해야 한다(§10 `verification_trust`
  프로파일 노드가 이 케이스를 포함하도록 확장, §12).

### 3.3 `verify_version` — publish된 version 하나를 처음부터 재검증

```python
@dataclass(frozen=True)
class VerifiedVersion:
    version_id: str
    manifest: dict
    faiss_bytes: bytes
    pkl_bytes: bytes

def verify_version(index_root: Path, version_id: str, *,
                    settings_snapshot: dict,
                    expected_owner_uid: int | None = None) -> VerifiedVersion:
    if not re.fullmatch(r"[0-9a-f]{16}", version_id):
        raise TrustBoundaryError("current_pointer_malformed")
    root = open_contained_root(index_root)
    try:
        versions_dir = root.open_subdir("versions")
        try:
            version_dir = versions_dir.open_subdir(version_id)
        finally:
            versions_dir.close()
        try:
            entries = sorted(version_dir.listdir())
            if entries != ["index.faiss", "index.pkl", "manifest.json"]:
                raise TrustBoundaryError("extra_or_missing_member")

            manifest_fd = version_dir.open_member("manifest.json",
                                                    allowed_mode_mask=0o022,
                                                    missing_reason="manifest_missing")
            try:
                raw = os.read(manifest_fd, MAX_MANIFEST_BYTES)  # MAX_MANIFEST_BYTES=65536
            finally:
                os.close(manifest_fd)
            manifest = parse_manifest(raw)      # ManifestError -> 그대로 전파(같은 예외 계층)
            if manifest["version_id"] != version_id:
                raise TrustBoundaryError("manifest_self_hash_mismatch")

            # 세 멤버 모두 같은 version_dir fd(같은 ContainedDir 인스턴스)에서
            # 연다 — 세 open 사이에 이 디렉터리가 교체될 여지가 없다(§3.2).
            faiss_bytes = _read_and_verify_member(version_dir, "index.faiss",
                                                   manifest["index_faiss"], expected_owner_uid)
            pkl_bytes = _read_and_verify_member(version_dir, "index.pkl",
                                                 manifest["index_pkl"], expected_owner_uid)

            _verify_settings_binding(manifest, settings_snapshot)  # mismatch -> settings_mismatch
            return VerifiedVersion(version_id, manifest, faiss_bytes, pkl_bytes)
        finally:
            version_dir.close()
    finally:
        root.close()
```

`_read_and_verify_member(version_dir: ContainedDir, name, expected, expected_owner_uid)`는
`version_dir.open_member(name, ...)`으로 연 fd에서 끝까지 읽고(1MiB
청크 loop, 선언된 `size_bytes`를 넘어서면 즉시 `member_size_mismatch`로
중단해 무제한 읽기를 막는다) — 파일을 닫기 전에 — 실제 바이트 길이와
`hashlib.sha256(bytes).hexdigest()`를 manifest 값과 비교한다
(`member_size_mismatch`/`member_hash_mismatch`). 이 읽기가 §3.4에서
재사용할 `faiss_bytes`/`pkl_bytes` **그 자체**이므로 **파일을 두 번
열지 않는다** — 검증과 로드가 같은 fd 세션 안에서 이뤄져 TOCTOU 창이
없다(DR-I1-CRIT-01: 이전 설계는 이 문장만 주장하고 실제로는
`load_verified_faiss`가 경로를 다시 열었다 — §3.4에서 그 재오픈 자체를
제거해 이 문장이 코드와 일치하게 만든다).

`_verify_settings_binding`은 `manifest["embedding_model_name"]`,
`manifest["embedding_provider"]`, `manifest["normalize_embeddings"]`,
`manifest["chunk_size"]`, `manifest["chunk_overlap"]`을 현재
`settings_snapshot`(§5.4)과 정확히 비교한다 — 다른 임베딩 모델/provider로
만든 index를 다른 설정의 서비스가 잘못 로드하는 사고를 차단한다.
`embedding_provider` 비교(DR-I2-MAJ-02 신규)는 host에서
`deterministic_test`로 build한 index를 production 기본값
(`huggingface`)으로 기동한 서비스가 실수로 query하는 사고, 그리고
반대로 container smoke가 host build와 다른 provider로 기동됐을 때
증상이 model 초기화 실패가 아니라 `settings_mismatch` 503으로 바로
드러나게 만든다(§7.5 4단계).

### 3.4 `load_verified_faiss` — 검증 bytes에서 직접 구성, 재오픈 없음 (DR-I1-CRIT-01 수정)

Iteration 1 CRIT-01: 이전 설계는 `verify_version()`이 이미 읽어 hash까지
확인한 `faiss_bytes`/`pkl_bytes`를 버리고, `FAISS.load_local(str(version_dir),
...)`로 **같은 경로를 다시 열었다**. 검증과 역직렬화 사이에 파일/디렉터리
교체 창이 생기는 근본 원인이었다. 수정된 설계는 `FAISS.load_local`을
전혀 호출하지 않고, 검증된 bytes에서 **직접** FAISS 객체를 구성한다 —
재오픈이 없으므로 재오픈 TOCTOU도 없다.

```python
def load_verified_faiss(index_root: Path, version_id: str, *, embeddings,
                         settings_snapshot: dict,
                         expected_owner_uid: int | None = None) -> "FAISS":
    verified = verify_version(index_root, version_id,
                               settings_snapshot=settings_snapshot,
                               expected_owner_uid=expected_owner_uid)
    return _construct_faiss_from_verified_bytes(verified, embeddings)


def _construct_faiss_from_verified_bytes(verified: "VerifiedVersion", embeddings) -> "FAISS":
    """verify_version()이 반환한 bytes에서만 역직렬화한다 — 이 함수는
    os.open/open()을 한 번도 호출하지 않는다(grep 감사 가능: 이 함수
    본문에 파일시스템 호출이 없다). langchain FAISS.load_local과 동일한
    두 단계(네이티브 index 복원 + docstore pickle 복원)를 파일 대신
    bytes에서 수행하고, 같은 생성자로 FAISS 인스턴스를 만든다."""
    import numpy as np
    import faiss as faiss_native
    from langchain_community.vectorstores import FAISS

    native_index = faiss_native.deserialize_index(
        np.frombuffer(verified.faiss_bytes, dtype=np.uint8))
    # pickle.loads는 여전히 REQ-001.5의 신뢰 경계 안에 있다 — 이 시점에는
    # verify_version()이 이미 size/hash/settings_binding을 모두 통과시킨
    # bytes만 들어온다(REQ-001.4의 "load 0회"는 이제 "재오픈 0회"로 강화됨).
    docstore, index_to_docstore_id = pickle.loads(verified.pkl_bytes)
    return FAISS(embeddings.embed_query, native_index, docstore, index_to_docstore_id)
```

- `faiss.deserialize_index`/`faiss.serialize_index`는 FAISS 파이썬
  바인딩이 제공하는 표준 API로, `faiss.read_index(path)`의 순수 bytes
  버전이다(파일 I/O 없이 numpy uint8 배열을 파싱) — `_stage_candidate`의
  `_smoke_check`(§4.2)가 쓰는 `faiss.read_index`와 동일한 native 포맷을
  다른 입력 형태로 파싱할 뿐이므로 신규 파싱 표면이 아니다.
- `FAISS.load_local`을 참조하는 곳은 이제 코드베이스 전체에서
  `rag_engine.py::_load_vectorstore_legacy`(§5.2, 648e3ab 바이트 동일)
  단 한 곳뿐이다 — `grep -rn "load_local" src/simple_qna_rag/`가 이
  불변식의 회귀 감사 명령이며 §8.4의 `run_m43_acceptance.py`가 이를
  자동화한다(부정 케이스: manifest 위조 시 `verify_version`이 예외를
  던지므로 `_construct_faiss_from_verified_bytes` 호출 자체가 일어나지
  않는다 — mock/spy로 0-call을 assert).
- **race 음성 테스트**(§3.2 마지막 항목, §12): racer 스레드가
  `verify_version()` 반환 **이후** `versions/<id>` 디렉터리 전체를
  삭제하거나 다른 내용으로 교체해도 `load_verified_faiss`의 나머지
  실행은 이미 메모리에 있는 `verified.faiss_bytes`/`pkl_bytes`만 쓰므로
  결과가 달라지지 않는다 — 이 테스트는 `verify_version` 반환 직후
  `shutil.rmtree(version_dir)`를 실행한 뒤에도
  `_construct_faiss_from_verified_bytes`가 정상적으로 원래 콘텐츠로
  성공함을 assert해 "재오픈이 아예 없다"를 관찰 가능한 형태로 증명한다.

### 3.5 `resolve_current` — genuine absence만 legacy fallback (DR-I1-MAJ-02 수정)

Iteration 1 MAJ-02: 이전 설계는 `pointer_path.exists()`로 사전 확인했다.
`Path.exists()`는 symlink를 따라가 대상을 확인하므로, dangling
symlink(`current -> /nonexistent`)에서는 `False`를 반환해 "없음"으로
분류되고 legacy 경로로 fail-open downgrade됐다. 수정된 설계는
`exists()` 사전 검사를 완전히 제거하고, **dirfd-relative open 자체의
errno만으로** 부재와 trust 위반을 구분한다 — `ENOENT`(진짜 없음)와
`ELOOP`(symlink, dangling 포함)는 open 시스템 콜 수준에서 이미 다른
errno이므로 별도의 사전 판별이 필요 없다.

```python
class CurrentPointerMissing(Exception):
    """INDEX_ROOT/current 엔트리 자체가 없음(ENOENT) — legacy fallback
    신호(§5.2)다. symlink/dangling symlink/non-regular는 이 예외가
    아니라 TrustBoundaryError로 분류된다(trust 위반이지 부재가 아니다)."""

def resolve_current(index_root: Path) -> str:
    root = open_contained_root(index_root)
    try:
        try:
            fd = os.open("current", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root.fd)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                raise CurrentPointerMissing() from None
            if exc.errno == errno.ELOOP:
                # symlink든 dangling symlink든 여기서 동일하게 ELOOP다 —
                # "존재하지만 신뢰 경계를 위반함"으로 분류하고 legacy
                # fallback을 절대 허용하지 않는다(fail-closed).
                raise TrustBoundaryError("current_pointer_symlink") from None
            raise
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise TrustBoundaryError("current_pointer_malformed")
            if st.st_mode & 0o022:
                raise TrustBoundaryError("current_pointer_malformed")
            raw = os.read(fd, 4096)
        finally:
            os.close(fd)
    finally:
        root.close()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise TrustBoundaryError("current_pointer_malformed") from None
    if set(doc) != {"schema_version", "version_id"} or doc.get("schema_version") != 1:
        raise TrustBoundaryError("current_pointer_malformed")
    version_id = doc["version_id"]
    if not re.fullmatch(r"[0-9a-f]{16}", version_id):
        raise TrustBoundaryError("current_pointer_malformed")
    versions_dir = open_contained_root(index_root).open_subdir("versions")
    try:
        try:
            version_dir = versions_dir.open_subdir(version_id)
        except TrustBoundaryError as exc:
            if exc.reason in ("version_dir_missing", "version_dir_not_directory",
                               "version_dir_symlink"):
                raise TrustBoundaryError("current_pointer_unknown_version") from None
            raise
        version_dir.close()
    finally:
        versions_dir.close()
    return version_id
```

- **음성 테스트 매트릭스**(§12): `current`가 (a) 존재하지 않음 →
  `CurrentPointerMissing` → legacy loader 호출 **1회**(정상, 의도된
  fallback), (b) 유효 대상을 가리키는 symlink → `TrustBoundaryError`,
  legacy loader 호출 **0회**, (c) dangling symlink →
  `TrustBoundaryError`, legacy loader 호출 **0회**, (d) `versions/`
  바로 아래가 아니라 상위 디렉터리를 가리키는 symlink → 동일하게
  `TrustBoundaryError`. 이 4가지 케이스 각각을 spy로 감싼
  `_load_vectorstore_legacy`의 호출 횟수로 검증하는 것이
  `test_index_verification.py::test_current_pointer_trust_matrix`의
  계약이다 — "dangling/root/current symlink 각각이 legacy loader
  0-call임을 spy로 검증"(리뷰 수정안 원문)을 정확히 만족한다.

## 4. `index/lifecycle.py` — staging/activation/rollback/retention 상태 머신

### 4.1 전체 상태 머신

```text
             build()/import_legacy()
                    |
                    v
   [PENDING] --mkdir(.staging/<op>, 0o700)--> [STAGED_DIR]
                    |  write index.faiss/index.pkl + fsync(각 파일)
                    v
             [FILES_WRITTEN]
                    |  write manifest.json + fsync(manifest)
                    v
             [MANIFEST_WRITTEN]
                    |  fsync(.staging/<op> 디렉터리 fd)
                    v
             [STAGING_DURABLE]
                    |  hash 재검증 + faiss.read_index() 스모크(비-pickle)
                    v
             [SMOKE_OK] --os.rename(.staging/<op>, versions/<id>)--> [PUBLISHED]
                    |                                                    |
              실패(어느 단계든) -> .staging/<op> inactive로 잔류         fsync(versions/ 부모)
                    |                                                    v
                    v                                              [DURABLE_VERSION]
              [FAILED] (receipt 없음, current 불변)

   activate(version_id) / rollback(to_version_id)  ── 동일 함수 §4.4
             |
   acquire_index_lock() -> verify_version(재검증) -> write tmp pointer + fsync
             -> os.replace(tmp, current) -> fsync(index_root) -> receipt write
             (검증 실패/lock 실패 -> current 불변, receipt outcome=FAIL 또는 미작성)

   cleanup(dry_run|apply)
             |
   acquire_index_lock() -> candidates = versions - {current, previous, protected}
   dry_run: candidates만 출력, 삭제 없음
   apply: 락 유지한 채 candidate별 재검증(current/previous/protected 아님을
          재확인) 후 realpath-contained regular directory에 한해 rmtree
```

이 순서는 REQ-002.3의 문자열 그대로다: "새 디렉터리 생성, 파일 write, file
fsync, manifest write/fsync, directory fsync, hash/load smoke, immutable
destination rename". `[SMOKE_OK]`에서 `[PUBLISHED]`로 가는 `os.rename`
한 번이 전체 publish를 원자화한다(같은 파일시스템 내 디렉터리 rename은
POSIX에서 원자적).

### 4.2 staging 세부 (`_stage_candidate`)

```python
def _stage_candidate(index_root: Path, *, faiss_bytes: bytes, pkl_bytes: bytes,
                      identity_fields: dict) -> tuple[Path, dict]:
    _assert_same_filesystem(index_root)                      # §4.5
    staging_root = index_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    op_dir = staging_root / uuid.uuid4().hex
    op_dir.mkdir(mode=0o700)
    try:
        _write_fsync(op_dir / "index.faiss", faiss_bytes)
        _write_fsync(op_dir / "index.pkl", pkl_bytes)
        manifest = build_manifest(identity_fields, created_at=isoformat(utc_now()))
        _write_fsync(op_dir / "manifest.json", canonical_json_bytes(manifest) + b"\n")
        _fsync_dir(op_dir)
        _smoke_check(op_dir, manifest)                        # hash 재확인 + faiss.read_index
        return op_dir, manifest
    except Exception:
        # 실패 시 op_dir을 지우지 않는다 — REQ-002.5 "실패·취소·disk full 뒤
        # staging 잔여물은 inactive로 남아야" 한다. cleanup(§4.6)이 검증된
        # .staging 자식만 대상으로 명시적으로 처리한다.
        raise
```

`_write_fsync(path, data)` (DR-I1-MAJ-05 수정: 완전 쓰기 loop):
```python
def _write_fsync(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    # O_EXCL: 새 파일만 허용, 기존 파일 덮어쓰기 거부 — staging 디렉터리는
    # 항상 새로 만들어지므로 O_EXCL이 실패하면 그 자체가 버그 신호.
    try:
        view = memoryview(data)
        total = 0
        while total < len(view):
            n = os.write(fd, view[total:])
            if n == 0:
                raise OSError("short write returned 0 bytes")
            total += n
        os.fsync(fd)
    finally:
        os.close(fd)
```
이전 설계는 단일 `os.write(fd, data)` 호출만 가정했다 — POSIX는
`write()`가 요청한 전체 바이트를 쓴다고 보장하지 않으므로(특히 큰
`index.faiss`/`index.pkl` 페이로드나 파이프/네트워크 파일시스템에서),
short write가 발생하면 manifest의 size/hash가 실제 디스크 내용과
불일치하는 상태로 fsync가 성공해버릴 수 있었다. 위 loop는 `n == 0`(진행
불가)을 별도 실패로 분리해 무한 루프를 만들지 않는다.

`_fsync_dir(path)`: `fd = os.open(path, os.O_RDONLY); os.fsync(fd);
os.close(fd)` (§0.3-4로 검증).

`_smoke_check(op_dir, manifest)`:
1. `index.faiss`/`index.pkl`을 다시 읽어 size/hash가 manifest와 일치하는지
   재확인(방금 쓴 값과 다시 읽은 값의 독립 검증 — 파일시스템 캐시 버그/
   fsync 누락을 잡는 회귀 게이트).
2. `faiss.read_index(str(op_dir / "index.faiss"))`로 FAISS 네이티브
   포맷만 파싱한다(pickle이 아니므로 REQ-001.5 위반이 아니다) — 손상된
   바이너리를 activation 이전에 조기 발견.
3. `index.pkl`은 **역직렬화하지 않는다** — pickle 신뢰 경계는 activation
   시점의 `verify_version`/`load_verified_faiss`(§3)로 완전히 미룬다.

### 4.3 publish (`_publish`)

```python
def _publish(index_root: Path, op_dir: Path, manifest: dict) -> Path:
    version_id = manifest["version_id"]
    dest = index_root / "versions" / version_id
    if dest.exists():
        # 결정론적 재빌드가 같은 identity를 냄 — idempotent 단축 경로
        existing = parse_manifest((dest / "manifest.json").read_bytes())
        if existing["index_faiss"] != manifest["index_faiss"] or \
           existing["index_pkl"] != manifest["index_pkl"]:
            raise LifecycleError("version_collision_byte_mismatch")  # 발생하면 버그/공격 신호
        shutil.rmtree(op_dir)          # 이미 존재 -> 새 staging은 폐기, 재사용
        return dest
    try:
        os.rename(op_dir, dest)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise LifecycleError("cross_device_staging") from None
        raise
    for name in ("index.faiss", "index.pkl", "manifest.json"):
        os.chmod(dest / name, 0o444)
    os.chmod(dest, 0o555)
    _fsync_dir(index_root / "versions")
    _fsync_dir(index_root)
    return dest
```

### 4.4 단일 activation primitive (`activate`) — activate와 rollback이 공유 (DR-I1-MAJ-05 수정)

Iteration 1 MAJ-05는 두 가지를 지적했다: (1) pointer가 durable하게
바뀐 뒤(`_fsync_dir(index_root)`) `_append_history`가 실패하면 pointer는
이미 새 값인데 history/receipt가 없어 "previous"에 의존하는 rollback과
retention이 대칭성을 잃는다. (2) 재시작 시 이 불완전 전이를 결정론적으로
복구하는 절차가 없었다. 수정된 설계는 pointer replace **전에** 의도를
durable하게 기록하는 **transition journal**을 도입하고, 모든 lifecycle
진입점이 시작 시 이 journal을 재조정(reconcile)한다.

```python
@dataclass(frozen=True)
class ActivationReceipt:
    schema: str
    operation: str            # "activate" | "rollback" (호출자가 넘김, 로직은 동일)
    operation_id: str
    outcome: str               # "PASS" | "FAIL"
    exit_code: int
    error_code: str | None
    pre_pointer: str | None
    post_pointer: str | None
    started_at: str
    finished_at: str
    verifications: list[str]
    reconciled: bool = False   # 신규(§4.4-b): crash 후 reconcile이 사후 발행한 receipt면 True

def activate(index_root: Path, version_id: str, *, operation: str,
             settings_snapshot: dict, lock_timeout: float,
             expected_owner_uid: int | None = None) -> ActivationReceipt:
    op_id = uuid.uuid4().hex
    started = isoformat(utc_now())
    try:
        with acquire_index_lock(index_root, timeout=lock_timeout):
            _reconcile_pending_transition(index_root)   # 이전 crash 잔여물 우선 정리(§4.4-b)
            try:
                pre = _read_current_or_none(index_root)
            except (CurrentPointerMissing,):
                pre = None
            verified = verify_version(index_root, version_id,
                                       settings_snapshot=settings_snapshot,
                                       expected_owner_uid=expected_owner_uid)
            # 1) 저널에 "의도"를 pointer 교체 전에 durable 기록한다.
            _write_transition_journal(index_root, phase="prepared", op_id=op_id,
                                       operation=operation, pre_pointer=pre,
                                       post_pointer=verified.version_id)
            # 2) 검증된 pointer만 교체 — 유일하게 current를 건드리는 줄.
            tmp = index_root / f".current.tmp.{os.getpid()}.{op_id}"
            _write_fsync(tmp, canonical_json_bytes(
                {"schema_version": 1, "version_id": verified.version_id}) + b"\n")
            os.replace(tmp, index_root / "current")
            _fsync_dir(index_root)   # <- 이 줄이 성공하면 pointer는 durable(commit point)
            # 3) commit marker — pointer가 이미 durable하게 바뀌었음을 저널에 기록.
            _write_transition_journal(index_root, phase="pointer_committed", op_id=op_id,
                                       operation=operation, pre_pointer=pre,
                                       post_pointer=verified.version_id)
            # history/receipt는 둘 다 op_id 기준 exact-once 헬퍼다(§4.4-a,
            # DR-I2-MAJ-01) — 정상 경로와 §4.4-b reconcile 경로가 완전히
            # 같은 두 함수를 호출하므로 "정상 성공"과 "crash 후 사후 완결"이
            # 같은 idempotency 보증을 공유한다.
            _append_history(index_root, op_id=op_id, operation=operation,
                             pre_pointer=pre, post_pointer=verified.version_id)
            receipt = ActivationReceipt(
                schema="m43-lifecycle-receipt-v1", operation=operation,
                operation_id=op_id, outcome="PASS", exit_code=0, error_code=None,
                pre_pointer=pre, post_pointer=verified.version_id,
                started_at=started, finished_at=isoformat(utc_now()),
                verifications=["schema", "hash", "size", "settings_binding"])
            _write_receipt_atomic(index_root, receipt)
            _clear_transition_journal(index_root)   # 4) 전이 완결 — 저널 비움
            return receipt
    except LockTimeoutError:
        return _fail_receipt(op_id, operation, started, "lock_timeout", exit_code=3)
    except TrustBoundaryError as exc:
        return _fail_receipt(op_id, operation, started, exc.reason, exit_code=1)
```

- `rollback(index_root, to_version_id, ...)`는 `activate(index_root,
  to_version_id, operation="rollback", ...)`를 그대로 호출하는 1줄
  래퍼다 — "같은 activation primitive로 pointer만 교체"(REQ-003.4)를
  코드 재사용으로 리터럴하게 만족시킨다. `--to-version`은 필수 인자이며
  자동 선택 로직은 어디에도 없다(REQ-003.4 "실패 version을 자동 선택하지
  않는다"). 편의용 `--to-previous`는 CLI 계층(§6.2)에서 먼저
  `_read_current_or_none(index_root)`로 현재 pointer를 읽고,
  `_read_previous_from_history(index_root, current=current)`(§4.4-a-1,
  **DR-I4-MAJ-01** 수정 — 최신 committed record의 `pre_pointer`에서
  직접 도출하며 그 record의 `post_pointer == current`를 함께 검증)로
  `to_version_id`를 **lock 밖에서 미리** 해석한 뒤 동일한
  `activate()` 호출로 넘긴다 — lifecycle 함수 자체에는 "previous"
  개념이 없다. 이 사전 해석이
  `TrustBoundaryError("activation_history_current_mismatch")`를 던지면
  CLI는 `activate()`를 호출하지 않고 그 예외를 §6.4의 표준 오류
  변환 경로로 그대로 전파한다(exit 1, receipt `outcome=FAIL`).
- 실패 원자성: `current` 파일을 건드리는 코드는 `os.replace(tmp, ...)`
  단 한 줄이다. 그 앞 어디서 예외가 나도 `current`는 무변경이다.
  `_write_transition_journal(phase="prepared", ...)`이 그 앞에서
  실패해도 마찬가지로 `current`는 무변경이며 저널도 아직 없으므로
  reconcile할 것이 없다.

#### 4.4-a `_write_transition_journal`/`_clear_transition_journal`

```python
def _write_transition_journal(index_root: Path, *, phase: str, op_id: str,
                               operation: str, pre_pointer: str | None,
                               post_pointer: str) -> None:
    record = canonical_json_bytes({
        "schema": "m43-transition-journal-v1", "phase": phase, "op_id": op_id,
        "operation": operation, "pre_pointer": pre_pointer, "post_pointer": post_pointer,
        "recorded_at": isoformat(utc_now()),
    }) + b"\n"
    tmp = index_root / f".transition.tmp.{os.getpid()}.{op_id}"
    _write_fsync(tmp, record)
    os.replace(tmp, index_root / ".transition")   # 단일 슬롯 — 한 번에 진행 중 전이는 하나뿐
    _fsync_dir(index_root)

def _clear_transition_journal(index_root: Path) -> None:
    path = index_root / ".transition"
    if path.exists():
        os.unlink(path)
        _fsync_dir(index_root)
```
`.transition`은 `current`와 마찬가지로 root 바로 아래 단일 파일이며,
`acquire_index_lock`이 build/import/activate/rollback/cleanup을 전부
직렬화하므로 한 번에 진행 중인 전이는 최대 하나다 — 슬롯이 하나뿐이어도
동시성 문제가 없다.

#### 4.4-a-1 `_append_history`/`_write_receipt_atomic`/`_read_history_rows`/`_read_previous_from_history` — operation_id 기준 exact-once, per-record immutable file, strict schema, previous 대수 (DR-I2-MAJ-01 / DR-I3-MAJ-01 / DR-I4-MAJ-01 수정)

Iteration 2 MAJ-01은 두 함수를 op_id 기준 idempotent하게 만들었지만,
history 저장 형태 자체는 여전히 단일 `activation_history.jsonl`에 대한
`O_APPEND` + 단일 `os.write`였다. **Iteration 3 MAJ-01**은 이 가정
자체가 틀렸다고 지적했다 — `PIPE_BUF` 원자성은 pipe/FIFO 계약이지
regular file의 "요청한 바이트 전체가 한 syscall로 반영된다"는 보장이
아니다. short write나 newline 이전 crash가 만든 partial tail 뒤에
재시도 record가 그대로 이어붙으면 두 JSON이 하나의 malformed 줄로
합쳐진다. 그러면 `_read_history_op_ids`가 그 op_id를 "행 없음"으로
취급하고, 이후 재시도가 이 손상된 줄 **뒤**에 새 온전한 줄을 계속
추가해도 원래 op_id는 영구히 나타나지 않는다 — exact-once가 저장
계층에서 이미 깨진 상태였다.

수정된 설계는 append-only JSONL 저장 형태 자체를 버린다. history의
물리적 단위를 "파일 안의 한 줄"에서 **operation마다 하나씩 존재하는
불변 레코드 파일**로 바꾼다 — §4.2/§4.4가 이미 쓰는 것과 완전히 같은
primitive(임시 파일 **전체** 쓰기 + fsync + `os.replace` atomic rename +
부모 디렉터리 fsync)로 커밋한다. 이 primitive는 "부분적으로 쓰인
내용이 관찰 가능한 이름으로 나타나는" 경로 자체를 없앤다 — `os.replace`가
성공적으로 반환하기 **전**에는 목적지 이름(`<op_id>.json`)이 존재하지
않고, 반환한 **뒤**에는 그 이름의 내용이 이미 fsync를 통과한 완결
바이트뿐이다. "짧은 write 뒤에 이어붙는 재시도"라는 실패 모드 자체가
없다 — 서로 다른 operation은 서로 다른 파일 이름을 가지므로 같은
슬롯을 두 번 건드릴 여지가 없다.

```python
_HISTORY_RECORD_NAME_RE = re.compile(r"^[0-9a-f]{32}\.json$")   # uuid4().hex 형식(op_id)만 신뢰
_HISTORY_RECORD_SCHEMA = "m43-activation-history-record-v1"
# 완결된 레코드가 반드시 가져야 하는 key 집합 — 초과/누락 모두 거부한다
# (DR-I4-MAJ-01: 이전 설계는 이 exact-key/type 검사가 전혀 없었다).
_HISTORY_REQUIRED_KEYS = frozenset({
    "schema", "op_id", "sequence", "operation", "pre_pointer", "post_pointer",
    "recorded_at", "reconciled",
})
_HISTORY_OPERATION_ENUM = frozenset({"activate", "rollback"})
_VERSION_ID_RE = re.compile(r"^[0-9a-f]{16}$")

def _history_dir(index_root: Path) -> Path:
    return index_root / "activation_history"

def _history_record_path(index_root: Path, op_id: str) -> Path:
    return _history_dir(index_root) / f"{op_id}.json"


def _read_history_rows(index_root: Path) -> list[dict]:
    """activation_history/의 각 <op_id>.json은 rename **이후에만** 존재하므로
    이 함수가 여는 모든 파일은 이미 완결(fsync 통과)된 바이트다 — partial
    read가 구조적으로 불가능하다(구 JSONL 설계의 "손상된 후행 줄" 분기
    자체가 필요 없어졌다). rename 전 crash로 남은 `.tmp.<pid>.<op_id>`
    잔여물은 이름 정규식이 걸러내므로 아예 조회 대상이 아니다(§4.2의
    `.staging/<op>` 실패 잔여물 처리와 동일한 원칙 — 잔여물은 지우지
    않고 무시한다).

    **DR-I4-MAJ-01**: JSON 파싱이 성공해도 이 함수는 곧바로 dict를 신뢰하지
    않는다 — exact key/type schema, 파일 이름↔본문 `op_id` 일치, `operation`
    enum, `sequence`의 uniqueness/contiguity를 모두 fail-closed로 검증한
    뒤에만 정렬된 리스트를 반환한다. 이 검증들 중 하나라도 실패하면
    ordering/`previous` 계산이 손상된 입력 위에서 조용히 진행되는 대신
    즉시 `TrustBoundaryError`를 던진다."""
    history_dir = _history_dir(index_root)
    if not history_dir.is_dir():
        return []
    rows: list[dict] = []
    for name in sorted(os.listdir(history_dir)):
        if not _HISTORY_RECORD_NAME_RE.fullmatch(name):
            continue
        expected_op_id = name[:-len(".json")]
        raw = (history_dir / name).read_bytes()
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # 완결된(rename된) 파일인데도 파싱이 실패하면 이는 더 이상
            # "진행 중 write"가 아니라 fsync 이후 저장매체 손상/수동
            # 변조 같은 진짜 이상 상태다 — 조용히 건너뛰지 않고
            # fail-closed한다(구 설계의 "손상 줄은 무시" 정책과 의도적으로
            # 다르다: 그 정책은 "아직 안 끝난 write"를 가정할 수 있었지만
            # 이 파일은 이미 rename을 통과했으므로 그 가정이 성립하지 않는다).
            raise TrustBoundaryError("activation_history_record_corrupt") from None
        if not isinstance(row, dict) or set(row) != _HISTORY_REQUIRED_KEYS:
            raise TrustBoundaryError("activation_history_schema_invalid")
        if row["schema"] != _HISTORY_RECORD_SCHEMA:
            raise TrustBoundaryError("activation_history_schema_invalid")
        if not isinstance(row["op_id"], str) or row["op_id"] != expected_op_id:
            # 파일 이름의 op_id와 본문 op_id가 다르면 rename 대상이
            # 뒤바뀌었거나 내용이 사후 치환된 것 — "어느 쪽을 믿을지"를
            # 추측하지 않고 즉시 거부한다.
            raise TrustBoundaryError("activation_history_filename_op_id_mismatch")
        if not isinstance(row["sequence"], int) or isinstance(row["sequence"], bool) \
                or row["sequence"] < 0:
            raise TrustBoundaryError("activation_history_schema_invalid")
        if row["operation"] not in _HISTORY_OPERATION_ENUM:
            raise TrustBoundaryError("activation_history_operation_invalid")
        if row["pre_pointer"] is not None and (
                not isinstance(row["pre_pointer"], str)
                or not _VERSION_ID_RE.fullmatch(row["pre_pointer"])):
            raise TrustBoundaryError("activation_history_schema_invalid")
        if not isinstance(row["post_pointer"], str) or not _VERSION_ID_RE.fullmatch(row["post_pointer"]):
            raise TrustBoundaryError("activation_history_schema_invalid")
        if not isinstance(row["recorded_at"], str):
            raise TrustBoundaryError("activation_history_schema_invalid")
        if not isinstance(row["reconciled"], bool):
            raise TrustBoundaryError("activation_history_schema_invalid")
        rows.append(row)
    rows.sort(key=lambda r: r["sequence"])
    sequences = [r["sequence"] for r in rows]
    # unique-contiguous ordering oracle(DR-I4-MAJ-01): 정렬만으로는 duplicate
    # sequence나 gap을 잡지 못한다 — 정렬 후 `0..N-1` 연속 정수와 정확히
    # 같은지 별도로 검증한다.
    if len(set(sequences)) != len(sequences) or (sequences and sequences != list(range(len(sequences)))):
        raise TrustBoundaryError("activation_history_sequence_invalid")
    return rows


def _read_history_op_ids(index_root: Path) -> frozenset[str]:
    """파일 **이름**만으로 판단한다 — 내용을 파싱할 필요조차 없다.
    exact-once 검사가 존재 여부만으로 결정된다는 것 자체가 partial-content
    위험을 제거한 이 설계의 핵심이다."""
    history_dir = _history_dir(index_root)
    if not history_dir.is_dir():
        return frozenset()
    return frozenset(name[:-len(".json")] for name in os.listdir(history_dir)
                      if _HISTORY_RECORD_NAME_RE.fullmatch(name))


def _next_history_sequence(index_root: Path) -> int:
    rows = _read_history_rows(index_root)
    return (max((r["sequence"] for r in rows), default=-1)) + 1


def _append_history(index_root: Path, *, op_id: str, operation: str,
                     pre_pointer: str | None, post_pointer: str,
                     reconciled: bool = False) -> None:
    """op_id당 정확히 하나의 불변 레코드 파일만 존재하도록 보장한다
    (exact-once). 이미 `<op_id>.json`이 있으면 아무것도 하지 않고
    반환한다 — activate()의 정상 호출과 reconcile의 사후 호출이 이
    검사를 공유하므로, crash가 이 함수의 rename **성공 이후** 어디서
    일어나든 재시도가 두 번째 레코드를 만들 수 없다. 이 함수는 항상
    index lock을 쥔 채로만 호출되므로(§4.4, §4.4-b) `sequence` 재계산이
    다른 쓰기와 경쟁하지 않는다."""
    history_dir = _history_dir(index_root)
    history_dir.mkdir(mode=0o700, exist_ok=True)
    dest = _history_record_path(index_root, op_id)
    if dest.exists():
        return
    sequence = _next_history_sequence(index_root)
    record = canonical_json_bytes({
        "schema": "m43-activation-history-record-v1", "op_id": op_id, "sequence": sequence,
        "operation": operation, "pre_pointer": pre_pointer, "post_pointer": post_pointer,
        "recorded_at": isoformat(utc_now()), "reconciled": reconciled,
    }) + b"\n"
    tmp = history_dir / f".tmp.{os.getpid()}.{op_id}"
    try:
        os.unlink(tmp)   # 같은 프로세스 내 이전 실패 재시도로 남은 동일 이름 잔여물 제거
    except FileNotFoundError:
        pass
    _write_fsync(tmp, record)     # 임시 파일 전체 쓰기 + fsync(§4.2 primitive 재사용)
    os.replace(tmp, dest)          # atomic rename — 이 줄 이전 crash는 dest가 존재하지
                                    # 않으므로 "레코드 없음"과 관찰상 완전히 동일하다
    _fsync_dir(history_dir)        # parent fsync — rename 자체를 durable하게 만든다(§0.3-4 근거)


def _read_previous_from_history(index_root: Path, *, current: str | None) -> str | None:
    """**DR-I4-MAJ-01 수정**: previous는 "마지막에서 두 번째 레코드의
    `post_pointer`"가 아니라, **현재 `current`를 만든 바로 그 최신
    (sequence 최대) committed record의 `pre_pointer`**에서 직접 도출한다.
    이전 정의는 record가 정확히 하나뿐인 최초 activation/import 직후
    (`pre_pointer=A`, `post_pointer=B`)에 무조건 `None`을 반환해 요구되는
    `previous=A`를 잃었다 — 새 정의는 레코드가 하나여도 그 레코드의
    `pre_pointer`를 그대로 반환하므로 이 손실이 없다.

    이 함수는 **latest record의 `post_pointer == current`**를 함께
    검증한다(호출자가 넘긴 `current`와 대조) — history가 현재 pointer를
    만든 operation과 다른 상태를 가리키면(수동 조작, 부분 복구, 레코드
    누락) `previous`를 신뢰하지 않고 즉시 fail-closed한다. 레코드가
    하나도 없는데 `current`가 `None`이 아니면(history가 아예 없는데
    pointer는 이미 존재 — 정합성 위반) 마찬가지로 거부한다."""
    rows = _read_history_rows(index_root)
    if not rows:
        if current is not None:
            raise TrustBoundaryError("activation_history_current_mismatch")
        return None
    latest = rows[-1]
    if latest["post_pointer"] != current:
        raise TrustBoundaryError("activation_history_current_mismatch")
    return latest["pre_pointer"]


def _read_last_receipt_or_none(index_root: Path) -> dict | None:
    path = index_root / ".last_activation_receipt.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _write_receipt_atomic(index_root: Path, receipt: ActivationReceipt) -> None:
    """단일 슬롯(`index_root/.last_activation_receipt.json`)에 원자적으로
    쓰되, 기존 슬롯의 `operation_id`가 이미 이 receipt와 같으면 다시
    쓰지 않는다(exact-once, `_append_history`와 동일한 원칙). 이 함수는
    애초에 단일 슬롯을 temp-write/fsync/atomic-rename/parent-fsync로
    교체하는 구조였으므로(변경 없음) DR-I3-MAJ-01의 근본 원인(단일
    `os.write`에 대한 잘못된 PIPE_BUF 가정)이 처음부터 적용되지 않는다
    — 이 함수는 재설계 대상이 아니다."""
    existing = _read_last_receipt_or_none(index_root)
    if existing is not None and existing.get("operation_id") == receipt.operation_id:
        return
    payload = canonical_json_bytes(dataclasses.asdict(receipt)) + b"\n"
    tmp = index_root / f".receipt.tmp.{os.getpid()}.{receipt.operation_id}"
    _write_fsync(tmp, payload)
    os.replace(tmp, index_root / ".last_activation_receipt.json")
    _fsync_dir(index_root)
```

- **partial write/tail crash 시나리오가 구조적으로 사라짐(DR-I3-MAJ-01)**:
  구 JSONL 설계의 실패 모드는 "짧은 write가 만든 partial bytes에
  newline이 없으면 재시도 record가 바로 이어붙어 하나의 malformed
  line이 된다"였다. 새 설계에는 그 실패 모드에 대응하는 관찰 가능한
  상태 자체가 없다 — 진행 중 write는 항상 `.tmp.<pid>.<op_id>`라는
  별도 이름 아래에 있고, `_read_history_rows`/`_read_history_op_ids`
  둘 다 이 이름을 정규식으로 걸러내 절대 후보로 삼지 않는다. `<op_id>.json`이
  나타나는 유일한 방법은 `os.replace`의 성공적 반환뿐이며, 그 시점의
  내용은 이미 `_write_fsync`의 fsync를 통과한 것이다.
- **fault injection 확장**(§10, §12): "short write(0이 아닌 partial
  포함)", "newline 전 crash", "partial tail이 이미 있는 재시작"을 모두
  `test_crash_recovery_history_and_receipt_exact_once_matrix`에 반영한다
  — monkeypatch 대상은 `_write_fsync`(tmp 파일에 대한 짧은 write를
  주입)이며, 주입 지점에 따라 기대 결과가 다르다: (a) tmp write/fsync
  단계에서 crash → `<op_id>.json`이 존재하지 않음 → 재시작한 reconcile이
  `_append_history`를 다시 호출해 **처음부터 온전하게 커밋**(재시도가
  곧 복구), (b) `os.replace` 성공 직후·`_fsync_dir(history_dir)` 진입
  전 crash → `<op_id>.json`이 이미 존재 → 재시작한 reconcile은
  `dest.exists()` 검사에서 즉시 skip(exact-once, 물리적 재기록 0회),
  (c) 구 JSONL 잔여 파일(partial tail을 가진 `activation_history.jsonl`)을
  fixture로 남겨두는 회귀 케이스 → 새 저장 형태에는 이런 파일이 아예
  존재할 수 없음을 `_read_history_rows`가 디렉터리 존재 여부만 본다는
  사실로 확인한다(§4.4-a-1 코드 참조). 세 경우 모두 operation당 history
  레코드 파일 정확히 1개, 최신 receipt의 `operation_id`가 그 op_id와
  정확히 일치, `previous=pre_pointer`, `current=post_pointer`를 검증한다.
- 상태 대수(§4.4-b, §12): pointer replace 성공 후 (a) history append 후·
  receipt write 전, (b) receipt write 후·journal unlink 전, (c) journal
  unlink **자체**의 각 지점에서 crash를 주입하고 재시작(reconcile을
  1~3회 반복 호출)해도, operation당 history 레코드 파일이 정확히 1개,
  최신 receipt의 `operation_id`가 그 op_id와 정확히 일치, `previous`가
  `pre_pointer`와 일치, `current`가 `post_pointer`와 일치함을
  `test_index_lifecycle_fault_injection.py::
  test_crash_recovery_history_and_receipt_exact_once_matrix`가
  검증한다(§10, §12에 반영).
- **`previous` 대수 독립 oracle(DR-I4-MAJ-01 신규)**:
  `tests/unit/test_index_lifecycle.py::test_previous_history_algebra_matrix`가
  `_read_history_rows`/`_read_previous_from_history`만(락이나 `activate()`
  전체 상태 머신 없이 `activation_history/` 디렉터리를 직접 fixture로
  구성)을 대상으로 아래 케이스를 각각 독립 assert한다 — 이 테스트가
  DR-I4-MAJ-01의 정확한 재현·수정 증거다.
  1. **empty**(레코드 0개, `current=None`) → `previous is None`(예외 없음).
  2. **empty인데 current가 있음**(레코드 0개, `current="B"*16`을 그대로
     치환한 16-hex) → `TrustBoundaryError("activation_history_current_mismatch")`.
  3. **first `A→B`**(레코드 1개, `pre_pointer=A16hex`, `post_pointer=B16hex`,
     `current=B16hex`) → `previous == A16hex`(이전 설계는 여기서 무조건
     `None`을 반환했다 — 이 케이스가 DR-I4-MAJ-01의 핵심 재현이다).
  4. **second `B→C`**(레코드 2개: `sequence=0`인 `A→B`, `sequence=1`인
     `B→C`, `current=C16hex`) → `previous == B16hex`(최신 record의
     `pre_pointer`).
  5. **rollback `C→B`**(레코드 3개: `A→B`, `B→C`, `sequence=2`인
     `operation="rollback"`, `pre_pointer=C16hex`, `post_pointer=B16hex`,
     `current=B16hex`) → `previous == C16hex`(rollback 자체도 같은 대수를
     따른다 — "롤백해도 다시 앞으로 롤백할 수 있는 지점"이 정확히
     C가 된다).
  6. **sequence duplicate**(두 레코드가 모두 `sequence=0`) →
     `TrustBoundaryError("activation_history_sequence_invalid")`.
  7. **sequence gap**(레코드가 `sequence=0`과 `sequence=2`만 있고 1이
     없음) → `TrustBoundaryError("activation_history_sequence_invalid")`.
  8. **filename↔body op_id mismatch**(파일명 `<op_id_X>.json`인데 본문
     `op_id`가 다른 `op_id_Y`) →
     `TrustBoundaryError("activation_history_filename_op_id_mismatch")`.
  9. **operation enum 위반**(`operation="delete"`) →
     `TrustBoundaryError("activation_history_operation_invalid")`.
  10. **latest.post_pointer != current**(레코드는 정상이지만 `current`
      인자로 다른 16-hex를 넘김 — 수동 pointer 조작을 시뮬레이션) →
      `TrustBoundaryError("activation_history_current_mismatch")`.
  11. **crash window 재사용**: §4.4-a-1의
      `test_crash_recovery_history_and_receipt_exact_once_matrix`가
      주입하는 세 crash 지점(tmp write/fsync 전, `os.replace` 직후·
      parent-fsync 전, 구 JSONL 잔여 fixture) 각각의 재시작 후 상태에서
      **추가로** `_read_previous_from_history(index_root,
      current=<재시작 후 실제 current>)`를 호출해 케이스 3-5의 대수가
      crash 유무와 무관하게 항상 성립함을 같은 테스트 함수 안에서
      재확인한다(중복 테스트 파일을 만들지 않고 기존 crash fixture를
      재사용).
  이 11개 케이스 모두 `_read_history_rows`가 파일시스템 fixture만으로
  구동되므로 `acquire_index_lock`/전체 `activate()` 상태 머신을 거치지
  않는다 — history 대수 자체의 정확성과 lifecycle 통합은 서로 다른
  테스트 레이어가 각각 담당한다(빠른 단위 테스트로 대수를, 느린 통합
  테스트로 crash injection을 검증).

#### 4.4-b `_reconcile_pending_transition` — 재시작 시 결정론적 복구

```python
def _reconcile_pending_transition(index_root: Path) -> ReconcileReport | None:
    """모든 lifecycle 진입점이 lock을 잡은 직후 가장 먼저 호출한다. 이전
    프로세스가 activate() 도중 crash했다면 여기서 상태를 결정론적으로
    정리한 뒤에만 새 operation을 진행한다."""
    journal_path = index_root / ".transition"
    if not journal_path.is_file():
        return None   # 정상 종료 — 정리할 것 없음
    try:
        record = json.loads(journal_path.read_bytes().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise TrustBoundaryError("transition_journal_corrupt") from None
    phase = record.get("phase")
    pre, post = record.get("pre_pointer"), record.get("post_pointer")
    try:
        actual_current = _read_current_or_none(index_root)
    except CurrentPointerMissing:
        actual_current = None

    if phase == "prepared" and actual_current == pre:
        # os.replace(tmp, current) 또는 그 뒤 _fsync_dir가 durable하게
        # 완료됐다는 증거가 없다 — pointer는 이전 값 그대로다. 전체
        # operation을 ABORTED로 간주하고 저널만 지운다. history/receipt는
        # 쓰지 않는다(이 operation은 애초에 성공하지 않았다).
        os.unlink(journal_path)
        _fsync_dir(index_root)
        return ReconcileReport(outcome="aborted", op_id=record["op_id"])
    if actual_current == post:
        # os.replace + 그 뒤 fsync(index_root)까지는 확실히 durable하게
        # 완료됐다(그렇지 않고서야 current가 post일 수 없다). 남은 단계
        # (history append, receipt write)만 사후 완결한다 — 두 헬퍼 모두
        # op_id 기준 exact-once이므로(§4.4-a-1, DR-I2-MAJ-01) 이 reconcile이
        # 몇 번 반복 호출되든(재-crash로 인한 재시도 포함) 실제 물리적
        # append/write는 최초 1회만 일어난다.
        _append_history(index_root, op_id=record["op_id"], operation=record["operation"],
                         pre_pointer=pre, post_pointer=post, reconciled=True)
        receipt = ActivationReceipt(
            schema="m43-lifecycle-receipt-v1", operation=record["operation"],
            operation_id=record["op_id"], outcome="PASS", exit_code=0, error_code=None,
            pre_pointer=pre, post_pointer=post,
            started_at=record["recorded_at"], finished_at=isoformat(utc_now()),
            verifications=["schema", "hash", "size", "settings_binding"],
            reconciled=True)
        _write_receipt_atomic(index_root, receipt)
        os.unlink(journal_path)
        _fsync_dir(index_root)
        return ReconcileReport(outcome="completed", op_id=record["op_id"])
    # actual_current가 pre도 post도 아니면 다른 operation이 그 사이 성공적으로
    # 개입한 것이므로(이 lock을 잡고 있는 한 이론상 불가능하지만, 수동
    # 파일 조작 등 방어적 케이스) 자동 판단하지 않고 중단한다.
    raise TrustBoundaryError("transition_journal_corrupt")
```
- `_reconcile_pending_transition`은 `.transition`을 지우고 필요하면
  history/receipt를 쓰는 **mutating** 함수이므로, lock을 잡는
  `build`/`import-legacy`/`activate`/`rollback`/`cleanup`(dry-run
  포함, §4.5)만 lock 획득 직후 호출한다 — lock을 잡지 않는
  `verify`/`list`는 이 함수를 호출하지 않는다. 읽기 전용 진단이 필요한
  **읽기 전용** 헬퍼 `_diagnose_pending_transition(index_root) ->
  dict | None`을 쓴다 — `.transition`이 있으면 그 내용과
  `_read_current_or_none(index_root)`를 함께 읽어 `{"phase", "pre_pointer",
  "post_pointer", "actual_current", "would_reconcile_to": "aborted"|"completed"}`를
  **아무것도 쓰지 않고** 반환한다(lock 없이도 안전 — 다른 프로세스가
  같은 순간 lock을 잡고 진짜 reconcile을 수행 중이어도, 이 함수는 그저
  한 순간의 스냅샷을 읽어 보고할 뿐이다).
- 이 알고리즘이 `actual_current`만으로 `prepared`/`pointer_committed`
  두 phase를 사실상 동일하게 처리하는 이유: `phase="pointer_committed"`
  기록 자체가 durable fsync 이후에 일어나므로, 저널에 그 phase가
  남아있다는 것은 이미 `actual_current == post`임을 강하게 시사한다.
  그러나 `_write_transition_journal(phase="pointer_committed", ...)`
  **자신**도 crash할 수 있으므로, 이 함수는 phase 문자열을 신뢰하지
  않고 항상 `actual_current`(진짜 디스크 상태)와 `pre`/`post`를
  비교해서만 판단한다 — phase는 진단 정보일 뿐 판정 근거가 아니다.
- 상태 대수 검증(REQ-003 정량 기준과 §12 fault injection에 반영):
  crash를 pointer replace 전/후/중간의 각 fsync 지점에 주입해도
  `_reconcile_pending_transition` 이후에는 항상 (a) `current`가
  `pre` 또는 `post` 중 하나이고 (b) `activation_history/`의 최신
  (`sequence` 최대) 레코드 파일의 `post_pointer`와 `current`가
  일치하며 (c) 최신 receipt의
  `post_pointer`가 `current`와 일치함을 `test_index_lifecycle_fault_injection.py::
  test_crash_recovery_journal_reconciles_to_consistent_state`가 검증한다.

### 4.5 OS advisory lock (DR-I1-MAJ-05 수정: trusted dirfd에서 O_CREAT|O_NOFOLLOW)

Iteration 1 MAJ-05: 이전 설계는 `lock_path.touch(exist_ok=True)`로
파일을 만든 뒤(경로 기반) `os.open(lock_path, os.O_RDWR)`로 **다시**
열었다 — `touch`와 `open` 사이 `.lock`이 symlink로 교체되면 그 symlink
대상을 열게 된다. 수정된 설계는 root의 검증된 dirfd 하나로 생성과
열기를 **한 syscall**로 합치고, 연 뒤에는 regular-file/mode를
재확인한다(공격자가 이 코드가 처음 실행되기 전에 이미 `.lock`을
symlink나 world-writable 파일로 심어둔 경우까지 방어).

```python
@contextmanager
def acquire_index_lock(index_root: Path, *, timeout: float):
    root_fd = os.open(str(index_root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # O_CREAT|O_NOFOLLOW를 dir_fd 상대로 한 번에 — "만들고 나중에 다시
        # 연다"는 두 단계가 없으므로 그 사이 symlink 교체 창이 없다.
        fd = os.open(".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600,
                     dir_fd=root_fd)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise TrustBoundaryError("lock_file_untrusted")
            if st.st_mode & 0o077:
                raise TrustBoundaryError("lock_file_untrusted")
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise LockTimeoutError()
                    time.sleep(0.05)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    finally:
        os.close(root_fd)
```
`O_NOFOLLOW`가 걸린 `os.open(..., dir_fd=root_fd)`는 `.lock`이 symlink면
`ELOOP`로 즉시 실패한다(별도 except 없이 그대로 전파 — lock 획득
자체의 실패이므로 호출자가 `OSError`로 받는다; CLI(§6.2)는 이를
도메인 오류로 분류되지 않은 예기치 못한 실패로 exit 1 처리한다).
`timeout=0`이면 즉시 실패(REQ-003.1 "즉시 또는 bounded timeout 후 실패").
이 lock은 `build`/`import-legacy`/`activate`/`rollback`/`cleanup`(dry-run
포함)이 잡는다 — `cleanup`은 삭제하지 않는 dry-run 모드에서도
`current`/`previous`와 candidate 목록을 **같은 순간의 일관된 스냅샷**으로
보여줘야 하고, 그 스냅샷을 읽기 전에 §4.4-b의 pending transition
reconcile도 함께 수행해야 하므로 항상 lock 안에서 실행한다(§4.6-a).
`verify`/`list`는 순수 조회이고 스냅샷 일관성이 없어도 안전하므로 lock을
잡지 않는다 — publish가 원자적 rename이므로 lock 없는 reader가 절반만 쓰인
version을 볼 수 없다(디렉터리 엔트리가 나타나는 순간 이미 완전히 채워져
있다).

`_assert_same_filesystem(index_root)`(§4.2):
```python
def _assert_same_filesystem(index_root: Path) -> None:
    versions_dir = index_root / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    if os.stat(index_root / ".staging").st_dev != os.stat(versions_dir).st_dev \
            if (index_root / ".staging").exists() else False:
        raise LifecycleError("cross_device_staging")
```
1차 방어는 이 사전 `st_dev` 비교, 2차 방어는 `_publish`의 `EXDEV` catch(§4.3)다
— 두 겹으로 REQ-002.2 "cross-device activation은 거부"를 보강한다.

### 4.6 retention/cleanup (DR-I1-MAJ-04 수정: fd-relative delete + staging 정책)

Iteration 1 MAJ-04는 두 가지를 지적했다: (1) 삭제 전 `realpath`/
`is_symlink`/`is_dir` 검사가 path 기반이라 advisory lock만으로는 막지
못하는 비협조적 로컬 actor의 rename/symlink 교체가 검사와
`shutil.rmtree(target)` 사이에 끼어들 수 있다. (2) `.staging` 실패
잔여물을 "inactive로 보존"한다고 문서화했지만 실제 `cleanup()`은
`versions/`만 열거해 `.staging`을 전혀 다루지 않는다 — TTL/owner/
liveness/dry-run 정책이 없어 disk-full 실패가 반복되면 잔여물이
무제한 누적된다.

```python
@dataclass(frozen=True)
class CleanupReceipt:
    candidates: list[str]
    deleted: list[str]
    dry_run: bool
    staging_candidates: list[str] = ()
    staging_deleted: list[str] = ()
```

#### 4.6-a version 삭제 — fd-relative no-follow walk

```python
def cleanup(index_root: Path, *, apply: bool, protect: list[str],
            lock_timeout: float, include_staging: bool = False,
            staging_min_age_seconds: int = 3600) -> CleanupReceipt:
    with acquire_index_lock(index_root, timeout=lock_timeout):
        _reconcile_pending_transition(index_root)   # §4.4-b
        current = _read_current_or_none(index_root)
        previous = _read_previous_from_history(index_root, current=current)
        protected = {current, previous, *protect} - {None}
        root = open_contained_root(index_root)
        try:
            versions_dir = root.open_subdir("versions")
            try:
                all_versions = set(versions_dir.listdir())
                candidates = sorted(all_versions - protected)
                deleted = []
                if apply:
                    for version_id in candidates:
                        # TOCTOU 방지: apply 직전 락을 쥔 채로 재확인(경로가
                        # 아니라 다시 읽은 pointer/history 값과 비교)
                        current_now = _read_current_or_none(index_root)
                        previous_now = _read_previous_from_history(index_root, current=current_now)
                        if version_id in {current_now, previous_now}:
                            continue
                        _fd_relative_rmtree(versions_dir.fd, version_id)
                        deleted.append(version_id)
            finally:
                versions_dir.close()
            staging_candidates, staging_deleted = [], []
            if include_staging:
                staging_candidates, staging_deleted = _cleanup_staging(
                    root, apply=apply, min_age_seconds=staging_min_age_seconds)
        finally:
            root.close()
        return CleanupReceipt(candidates=candidates, deleted=deleted, dry_run=not apply,
                               staging_candidates=staging_candidates,
                               staging_deleted=staging_deleted)


def _fd_relative_rmtree(parent_fd: int, name: str) -> None:
    """openat/fstatat 기반 no-follow 재귀 삭제. 매 단계가 dir_fd 상대이므로
    "검사 후 rmtree(path)를 별도 syscall로 다시 연다"는 창이 없다 — 연
    fd 자체가 그 시점의 디렉터리 identity를 고정한다."""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise TrustBoundaryError("version_dir_symlink") from None
        raise
    try:
        for entry in os.listdir(fd):
            st = os.stat(entry, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                raise TrustBoundaryError("version_dir_symlink")
            if stat.S_ISDIR(st.st_mode):
                _fd_relative_rmtree(fd, entry)
            else:
                os.chmod(entry, 0o600, dir_fd=fd, follow_symlinks=False) \
                    if hasattr(os, "chmod") else None
                os.unlink(entry, dir_fd=fd)
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=parent_fd)
```
publish 시 `chmod 0o444/0o555`(§4.3)로 읽기 전용화한 파일/디렉터리도
소유 프로세스(운영자 UID)가 지우는 것 자체는 POSIX에서 디렉터리
쓰기 권한 문제이지 파일 모드 문제가 아니므로 별도 `chmod` 없이
`os.unlink`가 성공한다 — 위 코드의 조건부 `chmod` 시도는 방어적
여분이며 필수 경로는 아니다(구현 phase에서 실측 후 불필요하면 제거).

#### 4.6-b `.staging` retention — REQ-002.5의 명시적 정책

```python
STAGING_NAME_RE = re.compile(r"^[0-9a-f]{32}$")   # uuid4().hex 형식만 신뢰

def _cleanup_staging(root: ContainedDir, *, apply: bool,
                      min_age_seconds: int) -> tuple[list[str], list[str]]:
    staging_dir = root.open_subdir(".staging")
    try:
        now = time.time()
        candidates = []
        for name in staging_dir.listdir():
            st = os.stat(name, dir_fd=staging_dir.fd, follow_symlinks=False)
            if not STAGING_NAME_RE.fullmatch(name):
                continue   # 이름 불일치 항목은 후보에서 제외(삭제도 안 함) —
                           # 알 수 없는 항목을 조용히 지우지 않는다(fail-closed)
            if stat.S_ISLNK(st.st_mode):
                continue   # symlink는 절대 후보 아님
            if not stat.S_ISDIR(st.st_mode):
                continue
            if (now - st.st_mtime) < min_age_seconds:
                continue   # 진행 중이거나 방금 실패한 operation일 수 있음 —
                           # index-root 전역 lock이 build/import를 cleanup과
                           # 직렬화하므로 "현재 쓰는 중"인 항목은 없지만,
                           # min-age는 여전히 "막 실패해 재시도 예정"인
                           # 잔여물을 보호하는 2차 방어선이다.
            candidates.append(name)
        deleted = []
        if apply:
            for name in candidates:
                _fd_relative_rmtree(staging_dir.fd, name)
                deleted.append(name)
        return sorted(candidates), sorted(deleted)
    finally:
        staging_dir.close()
```
- `cleanup` CLI(§6.1)는 `--include-staging`(기본 off, `versions/`
  정리와 분리된 명시적 opt-in)과 `--staging-min-age-seconds`(기본
  3600)를 추가한다 — `--dry-run`/`--apply` 경계는 `.staging`에도
  동일하게 적용된다(REQ-002.5 "explicit dry-run-first 명령").
- 이름이 `^[0-9a-f]{32}$`와 다른 항목, symlink, 디렉터리가 아닌 항목은
  **절대 후보에 포함하지 않는다** — 삭제 실수보다 "정리 안 됨"이 항상
  안전한 실패 방향이다(다음 `cleanup --dry-run`이 계속 보여주므로
  운영자가 인지할 수 있다).
- `test_index_lifecycle.py::test_cleanup_staging_protects_unexpected_and_young_entries`가
  이름 불일치·symlink·young(방금 생성)·old(정리 대상) 4가지 조합을
  matrix로 검증한다(§12).

### 4.7 `import_legacy` — pinned legacy approval as code constants, 런타임 파일 의존 제거 (DR-I1-MAJ-03 / DR-I2-MIN-06 / DR-I3-MIN-06 / DR-I4-MIN-03 수정)

Iteration 1 MAJ-03: 이전 설계는 "임의 expected hash는 승인 근거가 될 수
없다"고 서술했지만, CLI가 `--baseline-json PATH`(§6.1 이전 버전)를
그대로 노출해 호출자가 자신의 hash를 담은 임의 JSON을 승인 근거로
제출할 수 있었다 — 서술과 인터페이스가 정면 충돌했다. Iteration 1
개정은 CLI 플래그를 제거했지만 여전히 `_REPO_ROOT / "evaluation" /
"baselines" / "m3_initial.json"`을 **런타임에 파일로 읽었다**. Iteration
2 MIN-06은 이 경로가 개발 checkout(`_REPO_ROOT`가 소스 트리를 가리킴)
바깥에서는 실재하지 않음을 지적했다 — production 이미지는 `pip install
--target /install`로 `site-packages`에 패키지를 설치하므로(§7.1),
설치된 `index/lifecycle.py::__file__`을 기준으로 조상 디렉터리를
거슬러 올라가도 `evaluation/`은 그 트리에 없다. 이미지가
`evaluation/baselines/m3_initial.json`을 별도 COPY하더라도 그 파일이
`site-packages` 트리 밖(`/app/evaluation/...`)에 있으면 같은 문제가
반복된다. 또한 `test_import_legacy_rejects_tampered_or_untracked_baseline`이라는
테스트 이름은 "git untracked" 여부를 검사하는 것처럼 읽히지만 실제
구현은 embedded SHA 불일치만 검사해 이름과 보장이 어긋났다.

수정된 설계는 **런타임에 어떤 파일도 읽지 않는다** — 승인된 hash pair
자체를 `index/lifecycle.py` 모듈 상수로 박아 넣어, 배포 형태(dev
checkout, `pip install --target`, 컨테이너 이미지)와 무관하게 항상
같은 값을 참조하게 만든다. "커밋된 baseline과 정확히 같은 값"이라는
보장은 런타임 파일 재확인이 아니라 **테스트 시점에 tracked 파일과
상수를 직접 대조하는 provenance 회귀 테스트**로 옮긴다 — 이 테스트는
저장소 checkout에서만 실행되므로 런타임 배포 위치 문제와 무관하다.

```python
# index/lifecycle.py 모듈 상수 — CLI 인자로 override되지 않고, 런타임에
# 어떤 파일도 열지 않는다. 이 값들은 승인된
# evaluation/baselines/m3_initial.json의
# reproducibility.vectorstore_fingerprint.{index_faiss_sha256,index_pkl_sha256}를
# 그대로 옮겨 적은 리터럴이다(DR-I4-MIN-03 — 이 세션이 tracked 파일을
# 직접 읽어 확인한 실제 값이며 더 이상 placeholder가 아니다. 구현 phase는
# §15의 `git diff --exit-code -- evaluation/baselines/m3_initial.*`로
# 이 상수 치환 자체가 tracked baseline 파일 바이트를 건드리지 않았음을
# 재확인한다).
_PINNED_M3_BASELINE_ID = "m3_initial"
_PINNED_M3_APPROVED_INDEX_FAISS_SHA256 = "c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820"
_PINNED_M3_APPROVED_INDEX_PKL_SHA256 = "3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00"


def _pinned_m3_approved_pair() -> dict:
    """production 호출은 이 함수를 인자 없이 호출한다 — 모듈 상수만
    반환하며 파일시스템 접근이 전혀 없다(grep 감사 가능: 이 함수 본문에
    os.open/open/read_bytes/read_text가 없다)."""
    return {"index_faiss_sha256": _PINNED_M3_APPROVED_INDEX_FAISS_SHA256,
            "index_pkl_sha256": _PINNED_M3_APPROVED_INDEX_PKL_SHA256,
            "baseline_id": _PINNED_M3_BASELINE_ID}


def import_legacy(index_root: Path, source_dir: Path, *,
                   _approved_override: dict | None = None) -> LifecycleReceipt:
    """`_approved_override`는 leading underscore로 표시된 test-only
    seam이며 `cli/index_lifecycle.py`의 argparse 정의 어디에도 이 값을
    채울 수 있는 플래그가 없다 — production CLI 경로는 항상
    `_pinned_m3_approved_pair()`의 상수만 신뢰한다."""
    approved = _approved_override or _pinned_m3_approved_pair()
    faiss_bytes = (source_dir / "index.faiss").read_bytes()
    pkl_bytes = (source_dir / "index.pkl").read_bytes()
    if hashlib.sha256(faiss_bytes).hexdigest() != approved["index_faiss_sha256"] or \
       hashlib.sha256(pkl_bytes).hexdigest() != approved["index_pkl_sha256"]:
        raise TrustBoundaryError("member_hash_mismatch")   # legacy 원본이 승인 pair와 불일치
    identity_fields = _legacy_identity_fields(faiss_bytes, pkl_bytes, approved["baseline_id"])
    op_dir, manifest = _stage_candidate(index_root, faiss_bytes=faiss_bytes,
                                         pkl_bytes=pkl_bytes, identity_fields=identity_fields)
    dest = _publish(index_root, op_dir, manifest)
    return LifecycleReceipt(operation="import_legacy", outcome="PASS",
                             target_version_id=manifest["version_id"])
```
- `source_dir`의 원본 `index.faiss`/`index.pkl`은 **읽기만** 한다 —
  `_stage_candidate`가 그 bytes를 `.staging/<op>/`에 복사해 쓰므로
  legacy 원본 파일 자체는 바이트 단위로 무변경이다(REQ-002.4).
- **배포 위치 문제가 구조적으로 사라진다(DR-I2-MIN-06)**: `import_legacy`의
  production 경로는 `_pinned_m3_approved_pair()`만 호출하고 이 함수는
  파일 I/O를 하지 않으므로, "이 파일이 이 배포 형태에서 실재하는가"라는
  질문 자체가 없다 — dev checkout, `pip install --target` 설치,
  컨테이너 이미지 어디서나 동일한 Python 리터럴이 이미 로드된 모듈에
  들어있다. `deploy/Dockerfile`(§7.1)의 production stage는 이 값을 위해
  `evaluation/`을 COPY할 필요가 없다(스캐너 forbidden 목록의
  `evaluation/reports/`와 무관하게, `evaluation/baselines/`조차 이미지에
  없어도 된다 — REQ-005.1 "test/evaluation/runtime data를 포함하지
  않는다"와 오히려 더 잘 맞는다).
- **provenance 회귀 테스트(이름을 실제 보장과 일치시킴, DR-I2-MIN-06,
  fixture seam 추가로 DR-I3-MIN-06 완결)**: 파싱 로직을
  `_parse_m3_baseline_fingerprint(raw: bytes) -> dict`(순수 함수 —
  `evaluation/baselines/m3_initial.json`과 같은 스키마의 bytes를 받아
  `{"index_faiss_sha256": ..., "index_pkl_sha256": ...}`를 반환하며,
  파일 경로를 전혀 알지 못한다)로 분리한다. 이 helper 덕분에 positive와
  negative 두 테스트가 **같은 파서**를 서로 다른 bytes에 적용해 서로
  다른 것을 증명한다:
  1. **positive**: `tests/unit/test_pinned_baseline_provenance.py::
     test_pinned_constants_match_tracked_m3_baseline_bytes`가 저장소
     checkout의 tracked 경로(`Path(__file__).resolve().parents[2] /
     "evaluation" / "baselines" / "m3_initial.json"`)를 **바이트
     그대로** 읽어 `_parse_m3_baseline_fingerprint`에 넘기고, 반환된
     두 hash가 `_PINNED_M3_APPROVED_INDEX_FAISS_SHA256`/
     `_PINNED_M3_APPROVED_INDEX_PKL_SHA256` 두 모듈 상수와 정확히
     같은지 `assert`한다 — tracked 원본 파일은 읽기만 하고 절대 쓰지
     않는다.
  2. **negative(신규, DR-I3-MIN-06 — "임시 복사본에서 1바이트 변조"라는
     §12 서술을 실제 주입 가능한 seam으로 구현)**: 같은 테스트 모듈의
     `test_tampered_baseline_copy_diverges_from_pinned_constants`가
     `tmp_path`에 tracked 파일의 **바이트 사본**을 만들고
     (`read_bytes()` 후 `write_bytes()`로 `tmp_path`에 기록 — tracked
     원본은 전혀 건드리지 않는다), 그 임시 사본 bytes에서
     `index_faiss_sha256` hex 문자열의 한 글자를 다른 hex 문자로
     바꾼 뒤, **같은** `_parse_m3_baseline_fingerprint(tampered_bytes)`를
     호출해 반환된 hash가 pinned 상수와 **다름**을 `assert`한다 — 이
     negative가 실패(즉 1바이트 변조에도 hash가 여전히 같게 파싱됨)하면
     비교 메커니즘 자체가 고장났다는 뜻이므로, positive 테스트의
     "통과"가 무의미하지 않음을 이 negative가 독립적으로 증명한다.
  두 테스트 모두 CI checkout에서 항상 실행되므로(§10 프로파일 노드로
  등록, `legacy_baseline_pin`) "이 파일이 정당하게 갱신됐는데 상수가
  갱신되지 않음", "상수가 파일과 무관하게 임의로 바뀜", "비교
  메커니즘 자체가 변조를 못 잡음" 세 가지를 모두 즉시 실패로
  검출한다 — 이것이 이전 이름
  `test_import_legacy_rejects_tampered_or_untracked_baseline`이 암시했던
  "untracked/tampered" 보장의 실제 구현이며, 이름을 실제 검사 내용과
  일치시켰다(git이 파일을 추적하는지 자체는 이 테스트가 검사하지
  않는다 — 저장소에 그 경로의 파일이 존재하지 않으면 positive 테스트
  자체가 `FileNotFoundError`로 실패하므로 "파일이 없어짐"도 별도
  분기 없이 이미 fail-closed다). 저장소 전체의 dirty 상태는 기존
  `git diff --check`(Plan §8)가 별도로 감사한다 — 두 검사는 서로 다른
  시점(commit-time diff vs 상수-파일 대응)을 담당하므로 중복이 아니다.
- **구현 Gate 필수 조건 — real constants closure(DR-I3-MIN-06 fixture seam
  CLOSED, DR-I4-MIN-03로 production trust root 상수 자체를 CLOSED)**:
  `_PINNED_M3_APPROVED_INDEX_FAISS_SHA256`/`_PINNED_M3_APPROVED_INDEX_PKL_SHA256`는
  이제 이 설계 문서 안에서도 승인 baseline의 **실제 SHA-256 값**
  (`c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820`/
  `3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00`,
  이 세션이 tracked `evaluation/baselines/m3_initial.json`을 직접 읽어
  확인)이다(아래 코드 인용) — `"0" * 64` placeholder는 남아 있지 않다.
  구현 phase는 이 리터럴을 그대로 코드에 옮겨 적는 것 외에 추가 작업이
  없으며, 그 옮겨 적는 작업 자체가 `evaluation/baselines/m3_initial.json`
  파일을 실수로 건드리지 않았는지는 여전히
  `git diff --exit-code -- evaluation/baselines/m3_initial.*`로
  확인해야 한다(§15 clean 검증 명령에 이미 포함 — **tracked baseline
  bytes 무변경 gate는 보존한다**, 이 명령이 0이 아닌 exit로 실패하면
  "상수만 옮겨 적었어야 하는데 원본 파일도 함께 바뀜"이라는 사고를
  구현 Gate 단계에서 차단한다). positive provenance 테스트
  (`test_pinned_constants_match_tracked_m3_baseline_bytes`)는 이제 이
  실제 상수 값과 tracked 파일 bytes의 재계산 결과가 그대로 일치함을
  assert한다 — placeholder 시절처럼 이 테스트가 항상 실패하는 상태가
  아니라, §10 acceptance profile 통과의 실질적 전제 조건으로 이미
  성립한다.
- `test_index_lifecycle.py::test_import_legacy_rejects_source_hash_mismatch`는
  `_approved_override`로 임의 승인 pair를 주입해 `source_dir`의 실제
  hash와 다르면 `TrustBoundaryError("member_hash_mismatch")`가 발생함을
  검증한다(이전 이름의 "tampered baseline" 부분은 provenance 테스트로
  이관됐으므로, 이 테스트는 순수하게 "source가 승인 pair와 다르면
  거부"만 검증하도록 이름과 범위를 좁혔다).
- `REASONS`(§3.1)에서 `legacy_baseline_tampered`/`legacy_baseline_untracked`는
  제거한다 — 런타임 파일 읽기가 없으므로 "런타임에 발견한 baseline
  파일 손상"이라는 실행 경로 자체가 존재하지 않는다(그 자리를
  `member_hash_mismatch`가 대신한다).

## 5. 호환성 브리지 — Settings와 `rag_engine.py`

### 5.1 `settings.py` 신규 FieldSpec

```python
FieldSpec(
    name="INDEX_ROOT",
    annotation=Path,
    default_factory=lambda: _PACKAGE_ROOT / "runtime" / "index",
    env_alias="SIMPLE_QNA_RAG_INDEX_ROOT",
    parser=_path_parser,
    consumers=("rag_engine.py", "index/lifecycle.py", "index/verification.py",
               "cli/index_lifecycle.py"),
    facade_type=str,
    facade_adapter=str,
),
FieldSpec(
    name="EMBEDDING_PROVIDER",
    annotation=str,
    default_factory=lambda: "huggingface",
    env_alias="SIMPLE_QNA_RAG_EMBEDDING_PROVIDER",
    parser=_enum_parser(("huggingface", "deterministic_test")),
    consumers=("rag_engine.py", "index/lifecycle.py"),
    facade_type=str,
    facade_adapter=str,
),
FieldSpec(
    name="ALLOW_TEST_EMBEDDING",
    annotation=bool,
    default_factory=lambda: False,
    env_alias="SIMPLE_QNA_RAG_ALLOW_TEST_EMBEDDING",
    parser=_bool_parser,
    consumers=("rag_engine.py",),
    facade_type=bool,
    facade_adapter=bool,
),
```
`FIELD_SPECS` 리스트에 추가만 한다 — `VECTORSTORE_PATH`(L275-284)는 한
글자도 바꾸지 않는다. `EMBEDDING_PROVIDER`/`ALLOW_TEST_EMBEDDING`은
DR-I2-MAJ-02(§7.5) container mock smoke를 위한 test seam이다. 구현
phase는 이 필드 추가 직후 `python scripts/generate_field_spec.py --check`가
drift를 보고하도록 하고, `--check` 없이(즉 재생성 모드로) 한 번
실행해 `docs/generated/settings_field_spec.md`를 갱신해야 한다(§8.5
clean 검증 명령에 이미 포함).

**두 층위의 방어(DR-I3-MAJ-02, 이전 개정과의 핵심 차이)**: Iteration 3
리뷰는 "이 두 env var가 production 배포 절차/runbook에서 설정되지
않는다"는 **운영 관례**만으로는 production에서 test seam이 활성화되지
못하게 막는 **경계**가 아니라고 지적했다 — 운영자가 실수로(또는 악의적으로)
production 배포에 두 env var를 설정하면, config validator만으로는
아무것도 이 활성화를 막지 못한다. 수정된 설계는 이 사실을 문서에서도
정직하게 두 층위로 분리한다:

1. **Layer 1 (accidental-default-activation 방지, 여기 §5.1)**: `Settings`
   생성 시점의 cross-field validator(`pydantic.model_validator`류, 기존
   `Settings`가 이미 쓰는 관례)가 `EMBEDDING_PROVIDER == "deterministic_test"`이면서
   `ALLOW_TEST_EMBEDDING is not True`이면
   `SettingsValidationError("test_embedding_provider_requires_explicit_allow")`를
   발생시킨다 — 두 env var를 **모두** 명시적으로 설정해야 다음 단계에
   진입한다는 실수 방지 게이트일 뿐이며, 두 env var를 실제로 모두
   설정한 호출자를 막지는 못한다. 이 예외는 기존 bootstrap 오류
   경로(§5.3 `bootstrap_error`)를 그대로 타므로 `/health/ready`가 503
   `settings_invalid`로 fail-closed한다.
2. **Layer 2 (production 활성화 자체를 막는 실제 신뢰 경계, §5.2-a)**:
   `DeterministicTestEmbeddings` 모듈이 `src/` 밖(`tests/support/simple_qna_rag_test_seam/`)에만
   존재하고 production Dockerfile(§7.1)은 `src/`만 COPY하므로, production
   이미지에는 이 모듈이 **물리적으로 없다**. Layer 1을 통과해 이
   분기에 도달해도 `importlib.import_module(...)`이 `ModuleNotFoundError`로
   실패하고, `_build_embeddings()`가 이를 `TestEmbeddingSeamUnavailable`로
   변환해 readiness를 503 `artifact_test_embedding_seam_unavailable`으로
   fail-closed한다(§5.2-a). 이것이 "production 경로에서 활성화 불가능함"의
   실제 증거이며, `container` CI job의 negative OCI test(§7.5 4-neg
   단계)가 실제 production 이미지로 이를 재현·검증한다.

`test_settings_inventory.py`(§13 회귀 대상)에 다음 두 케이스를
추가한다(Layer 1 검증): `test_deterministic_embedding_provider_without_allow_flag_rejected`(둘
중 `ALLOW_TEST_EMBEDDING`만 빠진 조합이 bootstrap 실패로 이어짐을
검증)와 `test_default_settings_never_activate_test_embedding_provider`
(두 env var 모두 미설정인 기본 상태에서 `EMBEDDING_PROVIDER ==
"huggingface"`이고 `ALLOW_TEST_EMBEDDING is False`임을 확인 — production
경로가 실수로 test seam을 상속하지 않는다는 회귀 고정). Layer 2
검증은 §5.2-a/§7.4/§7.5를 참조한다.

### 5.2 `rag_engine.py::_load_vectorstore` — MODIFIED

```python
def _load_vectorstore(self) -> "FAISS":
    embeddings = _build_embeddings()
    index_root = Path(INDEX_ROOT)
    try:
        version_id = verification.resolve_current(index_root)
    except verification.CurrentPointerMissing:
        return self._load_vectorstore_legacy(embeddings)
    try:
        return verification.load_verified_faiss(
            index_root, version_id, embeddings=embeddings,
            settings_snapshot=_settings_binding_snapshot(),
            expected_owner_uid=_container_expected_uid())
    except verification.TrustBoundaryError as exc:
        raise IndexTrustError(exc.reason) from None

def _load_vectorstore_legacy(self, embeddings) -> "FAISS":
    # 648e3ab L169-191과 바이트 단위로 동일 — M4.1/M4.2 계약 보존(REQ-009.1)
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(
            "벡터스토어가 존재하지 않습니다. "
            "먼저 simple-qna-rag-index를 실행하여 문서를 등록해주세요."
        )
    return FAISS.load_local(VECTORSTORE_PATH, embeddings,
                             allow_dangerous_deserialization=True)
```

`IndexTrustError(RuntimeError)`는 `.reason` 속성만 갖는다(예외 메시지에
경로/스택을 넣지 않음). `TestEmbeddingSeamUnavailable(RuntimeError)`도
동일하게 `.reason` 속성만 갖는다(§5.2-a, DR-I3-MAJ-02 신규).
`_build_embeddings()`가 `_load_vectorstore`의 첫 줄에서 호출되므로 이
예외는 `_load_vectorstore` 내부 어떤 try/except에도 걸리지 않고 그대로
`initialize()`까지 전파된다 — `RAGEngine.initialize()`(L131-167)는 기존
`except Exception: return False` 앞에 다음 두 분기를 **추가만** 한다:

```python
except IndexTrustError as exc:
    self._artifact_error_reason = exc.reason
    return False
except TestEmbeddingSeamUnavailable as exc:
    self._artifact_error_reason = exc.reason
    return False
except Exception:
    return False
```
두 예외가 같은 `_artifact_error_reason` 플러밍(§5.3)을 공유하므로,
readiness는 신뢰 경계 위반(`artifact_member_hash_mismatch` 등)과 test
seam 미봉인(`artifact_test_embedding_seam_unavailable`)을 같은 503
응답 형태(`f"artifact_{artifact_error_reason}"`)로 노출하되 `reason`
값 자체는 서로 다르므로 운영자가 진단 표(§7.6)에서 구분할 수 있다.

`_container_expected_uid()`는 환경변수 `SIMPLE_QNA_RAG_EXPECT_UID`(신규
선택적 env — Settings 필드로 만들지 않고 os.environ 직접 조회로 최소화;
컨테이너 엔트리포인트가 `10001`을 주입하고 CLI/CI에서는 미설정이므로
`None` 반환)를 읽는다. 이는 순수 배포 편의 파라미터이며 신뢰 경계
자체(§3.2)는 이 값이 없어도 symlink/hash 검증으로 이미 완전하다.

### 5.2-a `_build_embeddings` / `DeterministicTestEmbeddings` — 배포 embedding 선택과 test seam (DR-I2-MAJ-02 신규, DR-I3-MAJ-02로 물리적 봉인 재설계)

Iteration 3 MAJ-02: 이전 개정은 `DeterministicTestEmbeddings`를
`src/simple_qna_rag/deterministic_embeddings.py`에 두었다 — production
Dockerfile의 `production` stage가 `COPY src/ ./src/`를 실행하므로 이
모듈이 **production 이미지 안에 그대로 존재**했다. §5.1의 2-키 Settings
게이트는 "설정을 두 개 다 명시해야 한다"는 실수 방지책일 뿐, 그 두
env var를 실제로 설정할 수 있는 운영자/침해자를 막는 경계가 아니다 —
"production 경로에서 활성화 불가능하다"는 문서 주장과 "모듈이 이미지
안에 있고 env var 두 개면 import된다"는 실제 코드가 정면으로 모순됐다.

수정된 설계는 이 모듈을 **`src/` 밖으로 완전히 이동**한다 —
`tests/support/simple_qna_rag_test_seam/deterministic_embeddings.py`(§1.1
신규 경로). production Dockerfile(§7.1)의 `production` stage는
`src/`/`pyproject.toml`/`README.md`/`LICENSE`/`web/static`/`web/templates`만
COPY하고 `tests/`는 어떤 stage에서도 production 이미지에 들어가지
않으므로, 이 모듈은 production 이미지에 **물리적으로 존재하지
않는다** — "이 파일이 이미지 layer 안에 있는가"라는 질문 자체가
`docker save`로 뜬 tar에서 확인 가능한 사실이 된다(§7.4 layer scanner에
회귀 검사 추가).

```python
_TEST_SEAM_MODULE = "simple_qna_rag_test_seam.deterministic_embeddings"

class TestEmbeddingSeamUnavailable(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _build_embeddings():
    if EMBEDDING_PROVIDER == "deterministic_test":
        # Settings validator(§5.1 Layer 1)가 이미 ALLOW_TEST_EMBEDDING is
        # True를 강제했으므로 이 분기에 도달했다는 것 자체가 두 env var가
        # 모두 명시적으로 설정됐다는 증거다 — 여기서 다시 검사하지 않는다.
        # 실제 신뢰 경계(Layer 2)는 이 바로 다음 줄이다: production
        # 이미지에는 이 모듈이 존재하지 않으므로 import는 구조적으로
        # 실패한다(grep 감사 가능: production stage의 COPY 목록 어디에도
        # `tests/`가 없다, §7.1).
        try:
            module = importlib.import_module(_TEST_SEAM_MODULE)
        except ModuleNotFoundError as exc:
            raise TestEmbeddingSeamUnavailable("test_embedding_seam_unavailable") from exc
        return module.DeterministicTestEmbeddings()
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': NORMALIZE_EMBEDDINGS}
    )
```

`tests/support/simple_qna_rag_test_seam/deterministic_embeddings.py`(신규
파일, §1.1 — `src/` 밖):
```python
class DeterministicTestEmbeddings:
    """네트워크/모델 다운로드 없이 완전히 결정론적인 고정 차원 벡터를
    생성한다 — 텍스트 의미를 반영하지 않으므로 검색 품질 비교 용도가
    아니다. 오직 container_smoke.py(§7.5)의 hosted Linux 배관 검증
    (build -> activate -> serve -> query 200)을 임베딩 모델 다운로드나
    네트워크 접근 없이 재현하기 위한 test seam이다. LangChain의
    Embeddings 프로토콜(embed_documents/embed_query)만 구현한다. 이
    파일은 의도적으로 `src/` 밖(`tests/support/`)에 있다 —
    production 이미지가 `src/`만 COPY하므로(§7.1) 이 파일은 그 이미지
    안에 물리적으로 존재할 수 없다(DR-I3-MAJ-02)."""

    DIMENSION = 32  # 임의 고정값 — 실제 프로덕션 모델 차원과 무관(§2.2
                     # faiss_dimension은 이 provider로 만든 index 자신의
                     # index.d에서 그대로 읽으므로 불일치가 생기지 않는다)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = (digest * ((self.DIMENSION // len(digest)) + 1))[:self.DIMENSION]
        vec = [b / 255.0 for b in raw]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]  # 정규화 — cosine/IP 검색과 호환
```
- 완전히 결정론적(같은 text → 같은 벡터, 실행/환경 무관)이므로
  `container_smoke.py`가 host에서 만든 index를 컨테이너 안에서 그대로
  재현 가능하게 query할 수 있다 — 실제 `BAAI/bge-m3` 모델을 컨테이너
  안에 심을 필요가 없다(REQ-005.4 "Ollama model, corpus, index...가
  없어야 한다"와 정합 — embedding 모델 가중치도 이미지에 포함하지
  않는다는 원칙의 연장).
- **hosted smoke는 이 모듈이 없는 production 이미지를 그대로 쓴다
  (DR-I3-MAJ-02, "test image가 아니라 production 경로" 요구)**:
  §7.5의 `container_smoke.py`는 production 이미지를 재빌드하거나 다른
  "test 전용 이미지"를 따로 만들지 않는다. 대신 컨테이너를 기동할 때
  `tests/support/`를 **읽기 전용 bind mount**로 추가하고
  `PYTHONPATH`에 그 경로를 넣어 import 경로만 런타임에 확장한다 — 이미지
  자체의 파일 목록, `USER 10001:10001`, `--read-only`, `--cap-drop ALL`,
  `--security-opt no-new-privileges` 등 §7.3의 보안 경계는 그대로
  유지되므로 "동일 runtime filesystem/security boundary를 보존한 별도
  명시적 test harness"가 된다(마운트 자체도 `:ro`이므로 read-only rootfs
  원칙과 충돌하지 않는다). 정확한 mount/env 조합은 §7.5 4단계를 참조.
- **production 활성화 거부의 negative OCI test(DR-I3-MAJ-02 핵심
  신규)**: §7.5 4-neg 단계가 **같은** production 이미지를 harness
  mount/PYTHONPATH **없이** 두 env var만 설정해 기동하고,
  `/health/ready`가 503 `artifact_test_embedding_seam_unavailable`을
  반환함을 bounded polling으로 확인한다 — 200이 한 번이라도 관찰되면
  `production_test_seam_not_sealed`로 즉시 FAIL. 이것이 "production
  이미지에서 두 env를 설정해도 bootstrap이 거부된다"의 실행 가능한
  증거다.
- **`Settings` Layer 1과 물리적 Layer 2가 서로 다른 시점/메커니즘으로
  독립 고정**(§5.1의 두 테스트가 Layer 1, 아래가 Layer 2): (a)
  `tests/unit/test_rag_engine_embeddings.py::
  test_build_embeddings_default_uses_huggingface_provider`(env var
  미설정 상태에서 `_build_embeddings`가 `HuggingFaceEmbeddings`
  인스턴스를 반환하는지 `isinstance`로 확인 — 이 테스트는 실제
  HuggingFace 모델 다운로드를 유발하므로 기존 `python-tests`가 이미
  지불하는 비용과 동일 수준, 신규 비용 아님), (b)
  `tests/unit/test_rag_engine_embeddings.py::
  test_build_embeddings_raises_seam_unavailable_when_module_absent`(`sys.modules`/`sys.path`에서
  `simple_qna_rag_test_seam`을 제거한 상태를 monkeypatch로 재현해
  `EMBEDDING_PROVIDER="deterministic_test"`일 때
  `TestEmbeddingSeamUnavailable`이 실제로 발생함을 확인 — production
  이미지의 조건을 unit test 수준에서도 재현하는 회귀), (c) §7.4
  layer scanner fixture(아래)와 §7.5 4-neg 단계(컨테이너 수준 재현)가
  각각 정적/동적으로 이중 고정한다.

### 5.3 readiness 통합 — `health.py`/`web/server.py`

`evaluate_readiness`(MODIFIED, 기본값으로 하위호환):
```python
def evaluate_readiness(
    bootstrap_error: str | None, settings_error: str | None, engine_error: str | None,
    *, lifecycle: str | None = None, saturated: bool = False,
    orphaned: int = 0, concurrency_limit: int = 0,
    artifact_error_reason: str | None = None,   # 신규, 기본 None
) -> tuple[int, str]:
    if bootstrap_error is not None:
        return 503, "static_mount_failed"
    if settings_error is not None:
        return 503, "settings_invalid"
    if artifact_error_reason is not None:
        return 503, f"artifact_{artifact_error_reason}"
    if engine_error is not None:
        return 503, "engine_init_failed"
    ...  # 이하 무변경
```
`artifact_error_reason` 분기를 `engine_error`보다 먼저 두는 이유: 둘 다
`initialize()` 실패에서 나오지만(`engine_error`가 항상 함께 설정됨),
운영자가 더 구체적인 원인(신뢰 경계 위반 vs 일반 초기화 실패)을
readiness 응답만으로 구분해야 배포 runbook(§7.6)의 진단 표가 동작한다.
기존 호출부는 새 인자를 넘기지 않으므로 `artifact_error_reason=None`이
유지되고 기존 4개 readiness 테스트는 변경 없이 통과한다.

`web/server.py` lifespan(L339-366 부근, MODIFIED) — 기존
`app.state.engine_error = str(exc)` 옆줄에 한 줄만 추가:
```python
app.state.engine_artifact_reason = getattr(engine, "_artifact_error_reason", None)
```
그리고 `/health/ready` 핸들러(L492 부근)가 `evaluate_readiness(...,
artifact_error_reason=getattr(request.app.state, "engine_artifact_reason",
None))`로 넘긴다.

### 5.4 `_settings_binding_snapshot()`

```python
def _settings_binding_snapshot() -> dict:
    return {
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "embedding_provider": EMBEDDING_PROVIDER,  # DR-I2-MAJ-02
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
```
`rag_engine.py`가 이미 import하는 `config`(→`settings` facade) 상수만
사용한다 — 새 import 경계를 만들지 않는다.

## 6. Lifecycle CLI (`cli/index_lifecycle.py`)

새 진입점 `simple-qna-rag-index-lifecycle`(`pyproject.toml::[project.scripts]`에
1줄 추가, 기존 3개 진입점은 무변경).

### 6.1 subcommand/argv

```text
simple-qna-rag-index-lifecycle build
    [--documents-dir PATH] [--index-root PATH] [--lock-timeout-seconds N=10]
simple-qna-rag-index-lifecycle import-legacy
    --source-dir PATH [--index-root PATH] [--lock-timeout-seconds N=10]
simple-qna-rag-index-lifecycle verify --version VERSION_ID [--index-root PATH]
    [--reconcile-check]
simple-qna-rag-index-lifecycle activate --to-version VERSION_ID
    [--index-root PATH] [--lock-timeout-seconds N=10]
simple-qna-rag-index-lifecycle rollback (--to-version VERSION_ID | --to-previous)
    [--index-root PATH] [--lock-timeout-seconds N=10]
simple-qna-rag-index-lifecycle list [--index-root PATH]
simple-qna-rag-index-lifecycle cleanup (--dry-run | --apply)
    [--protect VERSION_ID ...] [--include-staging]
    [--staging-min-age-seconds N=3600] [--index-root PATH] [--lock-timeout-seconds N=10]
```
`--dry-run`이 `cleanup`의 기본값(인자 미지정 시)이며, `--apply`만 명시적
파괴적 동작이다(REQ-004.1). `--include-staging`도 기본 off인 opt-in이다
(§4.6-b).

의도적으로 없는 플래그(DR-I1-MAJ-03/DR-I2-MIN-06 반영): `import-legacy
--expected-hash`와 `import-legacy --baseline-json`. REQ-002.1이 "CLI가
제공한 임의 expected hash는 승인 근거가 될 수 없다"고 명시하므로,
신뢰 근거는 `index/lifecycle.py`에 고정된 승인 hash pair 코드
상수뿐이다(§4.7) — 런타임에 여는 baseline 파일 자체가 없으므로 그
경로를 사용자 입력으로 대체할 여지도 없다. CLI argparse 정의
어디에도 이 상수를 대체할 수 있는 옵션이 없다 — 이전 iteration에서
노출됐던 `--baseline-json PATH`는 완전히 제거됐다. 테스트가 다른
승인 pair를 주입해야 하는 경우는 `import_legacy(...,
_approved_override=...)` Python 함수 인자로만 접근하며, 이 파라미터는
CLI argparse 어디에도 연결되지 않는다.

### 6.2 exit code 표

| exit | 의미 | receipt 작성 |
|---:|---|---|
| 0 | PASS | 있음, `outcome=PASS` |
| 1 | 도메인 실패(trust boundary 거부, 해시/스키마 불일치, 검증 실패) | 있음, `outcome=FAIL` |
| 2 | 사용법/인자 오류(argparse 표준) | 없음 |
| 3 | lock 경쟁/timeout | 있음, `outcome=FAIL`, `error_code=lock_timeout` |

이는 `orchestration_state.py`/`orchestration_watchdog.py`가 이미 쓰는
"2=파싱/권한 오류, 그 외=도메인 결과" 관례를 그대로 잇되, lock 전용
3번을 신설해 §5의 §5 gate(lock 경쟁 mutation 1개 이하)를 CLI 결과
코드만으로 자동 판별 가능하게 한다.

### 6.3 receipt 스키마

```json
{
  "schema": "m43-lifecycle-receipt-v1",
  "operation": "build|import_legacy|verify|activate|rollback|cleanup",
  "operation_id": "<uuid4 hex>",
  "started_at": "2026-08-12T00:00:00Z",
  "finished_at": "2026-08-12T00:00:01Z",
  "outcome": "PASS|FAIL",
  "exit_code": 0,
  "error_code": null,
  "identity": {
    "builder_git_sha": "...", "builder_git_dirty": false,
    "settings_hash": "...", "dependency_lock_sha256": "..."
  },
  "source_version_id": null,
  "target_version_id": "<16-hex or null>",
  "pre_pointer": "<16-hex or null>",
  "post_pointer": "<16-hex or null>",
  "verifications": ["schema", "hash", "size", "settings_binding"],
  "artifact_sha256": "<sha256 over this object with this field absent>"
}
```
경로 필드는 전부 `index_root` 상대 경로만 포함한다(절대경로 없음, REQ-004.2).
`error_code`는 `verification.REASONS`(§3.1) ∪ `{"lock_timeout",
"cross_device_staging", "version_collision_byte_mismatch"}`의 고정
vocabulary에서만 나온다 — `str(exc)`를 그대로 넣는 코드 경로는 없다.

### 6.4 `CorpusManifestError`/lifecycle 오류 변환 (REQ-004.4)

```python
def main(argv=None) -> int:
    try:
        result = _dispatch(args)  # build/import-legacy/verify/activate/rollback/list/cleanup
    except (ManifestError, TrustBoundaryError, LifecycleError, LockTimeoutError) as exc:
        # traceback 없이 안정된 receipt(위 스키마)로 변환, exit는 §6.2 표
        receipt = _fail_receipt_from_exc(exc)
        _emit(receipt)
        return receipt["exit_code"]
    except (ValueError, argparse.ArgumentError):
        return 2
    _emit(result.receipt)
    return result.receipt["exit_code"]
```
`evaluation.dataset` 모듈이 이미 `CorpusManifestError`를 정의한다(§0.2 확인
결과 재확인 필요 시 `evaluation/reporting.py::build_corpus_manifest`가
직접 던진다) — `build` subcommand는 이 예외를 위 catch 목록에 포함해
동일하게 안정된 exit/receipt로 변환한다.

## 7. OCI production image (REQ-005)

### 7.1 `deploy/Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.11-slim

FROM ${PYTHON_IMAGE} AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

FROM base AS builder
RUN python -m pip install --no-cache-dir "uv==0.8.15"
COPY requirements.lock ./
RUN python -m pip install --require-hashes --no-cache-dir \
      -r requirements.lock \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      --target /install
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir --no-deps --target /install -e .

FROM base AS test
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY pyproject.toml README.md LICENSE requirements.lock ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY evaluation/ ./evaluation/
RUN python -c "from simple_qna_rag.web.server import app"
CMD ["python", "-c", "print('test stage import smoke only')"]

FROM base AS production
RUN groupadd -g 10001 app && \
    useradd -u 10001 -g app -M -s /usr/sbin/nologin app
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY web/static/ ./web/static/
COPY web/templates/ ./web/templates/
RUN mkdir -p /app/runtime/index && chown -R app:app /app/runtime
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["python", "-m", "uvicorn", "simple_qna_rag.web.server:app", \
            "--host", "0.0.0.0", "--port", "8000"]
```
- **DR-I1-MAJ-06 수정**: 이전 설계는 `web/`을 전혀 복사하지 않았다.
  `src/simple_qna_rag/settings.py:252-270`의 `PROJECT_ROOT/web/static`,
  `PROJECT_ROOT/web/templates`가 런타임 경로이고 M4.2 readiness는 static
  mount 실패를 503 `static_mount_failed`로 우선 처리하므로(§5.3
  `evaluate_readiness`의 `bootstrap_error` 분기가 이 값을 가장 먼저
  검사), static 자산 없는 이미지는 `/health/ready`조차 200을 낼 수
  없었다. `COPY web/static/ ./web/static/`와 `COPY web/templates/
  ./web/templates/`를 명시적 allowlist 항목으로 추가해 이 결함을
  닫는다.
- `web/static/vendor/**/*.map`은 `.dockerignore`(§7.2)가 여전히
  제외한다 — sourcemap은 런타임에 불필요하고 스캐너 forbidden 목록
  대상도 아니므로 `web/static/` 전체 COPY와 충돌하지 않는다(`.dockerignore`는
  빌드 컨텍스트 자체에서 그 파일들을 제외하므로 `COPY`가 애초에 볼 수
  없다).
- `scripts/scan_image_layers.py`(§7.4)의 `FORBIDDEN_PATTERNS`에는
  `web/static/`이나 `web/templates/`가 없다 — 이 두 경로는 정당한
  production 자산이므로 스캐너가 위반으로 잡지 않는다(스캐너
  fixture에 `web/static/style.css`류 clean 항목을 하나 추가해 이
  allowlist가 실제로 위반이 아님을 회귀 검증한다, §7.4 fixture 확장).

- `test` stage는 `docker build --target test`로만 빌드해 "설치 가능성 +
  import 가능성"을 증명한다(Plan §8의 `docker build --target test`
  명령과 1:1 대응). 실제 `pytest` 실행은 hosted `python-tests` job이
  담당하며(REQ-005.5 "실제 Ollama image test는 구성요소가 아니다"와
  같은 원칙으로 이미지 빌드와 테스트 실행 책임을 분리), 이 stage는
  빌드/스캐너 대상 확보 목적만 갖는다.
- `production` stage는 `builder`가 만든 `/install`(pip `--target`
  디렉터리)만 복사한다 — `uv`, pip 캐시, `requirements.lock` 원본,
  컴파일러 흔적이 없다(REQ-005.1 "builder, compiler, test/evaluation/
  runtime data를 포함하지 않는다").
- `USER 10001:10001`은 숫자 UID/GID로 고정한다(REQ-005.3 "고정 non-root
  UID/GID").
- `runtime/index`를 이미지 안에 만들어 두는 것은 빈 디렉터리 마운트
  포인트 준비일 뿐이다 — 실제 index 콘텐츠는 REQ-005.3 "documents/index는
  read-only mount"에 따라 런타임에 볼륨으로 주입한다(이미지에 index
  바이트를 굽지 않음 — REQ-005.4).
- **test embedding seam이 production 이미지에 물리적으로 없음
  (DR-I3-MAJ-02)**: `production` stage의 COPY 목록(`src/`,
  `pyproject.toml`, `README.md`, `LICENSE`, `web/static/`,
  `web/templates/`) 어디에도 `tests/`가 없다 — `tests/support/simple_qna_rag_test_seam/`(§5.2-a)는
  이 stage가 절대 참조하지 않는 경로이므로, 그 안의
  `DeterministicTestEmbeddings`는 production 이미지 layer 어디에도
  존재하지 않는다. 이는 `.dockerignore`(§7.2)의 방어적 이중화가 아니라
  이 `COPY` allowlist 자체가 유일한 신뢰 경계다(§7.2 원칙과 동일).
  `scripts/scan_image_layers.py`(§7.4)의 `FORBIDDEN_PATTERNS`에
  `simple_qna_rag_test_seam`을 추가해 이 구조적 부재를 이미지 tar
  수준에서도 정적으로 재확인한다(방어 이중화 — 1차는 COPY allowlist,
  2차는 layer scan, 3차는 §7.5 4-neg 단계의 런타임 negative test).

### 7.2 `.dockerignore`

```text
.git/
.env
.env.*
runtime/
evaluation/reports/
.pytest_cache/
__pycache__/
*.pyc
.idea/
.claude/
node_modules/
models/
venv/
.venv/
web/static/vendor/**/*.map
dependency_snapshot.json
```
이는 **성능/방어 이중화**이지, 유일한 신뢰 경계가 아니다 — 실제 신뢰
경계는 `production` stage의 명시적 `COPY` allowlist(`src/`,
`pyproject.toml`, `README.md`, `LICENSE`만; `COPY . .` 전체 복사는
어디에도 없다)다. `.dockerignore`가 항목 하나를 놓쳐도 `COPY`
allowlist가 잡는다는 것이 독립 리뷰(Plan §5 "container COPY/layer
현실성")가 확인해야 할 핵심 사실이다.

### 7.3 최소 권한 실행 계약 (REQ-005.3)

컨테이너 실행 표준 플래그(runbook/`container_smoke.py`/K8s 매니페스트
예시가 공유):
```bash
docker run --rm \
  --user 10001:10001 \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --add-host host.docker.internal:host-gateway \
  -v "<host-index-root>:/app/runtime/index:ro" \
  -e SIMPLE_QNA_RAG_INDEX_ROOT=/app/runtime/index \
  -p 8000:8000 \
  simple-qna-rag:m43-candidate
```
`--add-host host.docker.internal:host-gateway`(DR-I1-MAJ-07 수정)는
Docker Engine 20.10+가 지원하는 표준 특수 값으로, Linux 컨테이너 안에서
호스트로 나가는 게이트웨이 주소를 `host.docker.internal`이라는 이름에
바인딩한다 — macOS/Windows Docker Desktop에서는 이 이름이 기본
제공되지만 **Linux Engine**(GitHub-hosted `ubuntu-latest`가 쓰는 바로 그
환경)에서는 이 플래그 없이는 이름 자체가 존재하지 않는다. 이전 설계는
이 플래그 없이 `docker run` argv만 제시해 목표 hosted 환경에서 mock
Ollama 연결이 애초에 불가능했다.
`--read-only` + `index`가 `ro` 마운트이므로 index mutation은 별도 operator
command(§6 lifecycle CLI를 호스트 또는 별도 쓰기 가능 컨테이너에서 실행)로만
가능하다(REQ-005.3 마지막 문장).

### 7.4 `scripts/scan_image_layers.py`

```python
FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (".git/", "vcs_directory"), (".env", "env_file"),
    ("runtime/vectorstore/", "index_artifact"),
    ("runtime/documents/", "corpus_artifact"),
    ("runtime/index/versions/", "index_artifact"),
    ("models/intent_classifier/", "model_artifact"),
    (".ollama/", "ollama_data"),
    ("evaluation/reports/", "ci_report"),
    ("id_rsa", "credential"), (".pem", "credential"), (".pfx", "credential"),
    ("simple_qna_rag_test_seam", "test_embedding_seam"),  # 신규(DR-I3-MAJ-02):
    # DeterministicTestEmbeddings가 production layer 어디에도 없어야 한다
    # (§5.2-a, §7.1) — 이 패턴이 매치되면 production COPY allowlist가
    # 깨졌다는 신호다
)

def export_image(image: str, out_tar: Path) -> None:
    subprocess.run(["docker", "save", image, "-o", str(out_tar)], check=True)

def normalize_member_path(name: str) -> str:
    posix = posixpath.normpath(name.lstrip("/"))
    return posix

def classify_member(name: str) -> tuple[str, str] | None:
    """(category, matched_pattern) 반환, 없으면 None. traversal은
    normalize 후에도 '..'로 시작하면 별도 category 'path_traversal'."""
    norm = normalize_member_path(name)
    if norm.split("/", 1)[0] == "..":
        return ("path_traversal", name)
    for pattern, category in FORBIDDEN_PATTERNS:
        if pattern.rstrip("/") in norm:
            return (category, pattern)
    return None

def is_whiteout(name: str) -> bool:
    base = posixpath.basename(normalize_member_path(name))
    return base.startswith(".wh.")

def scan(image: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "image.tar"
        export_image(image, archive)
        with tarfile.open(archive) as outer:
            manifest = json.loads(outer.extractfile("manifest.json").read())
            layer_paths = [entry for cfg in manifest for entry in cfg["Layers"]]
            violations = []
            layer_reports = []
            for layer_path in layer_paths:
                members = []
                with tarfile.open(fileobj=outer.extractfile(layer_path)) as layer:
                    for member in layer.getmembers():
                        if is_whiteout(member.name):
                            continue   # 삭제 표시는 내용 검사 대상 아님
                        hit = classify_member(member.name)
                        if hit:
                            violations.append({"layer": layer_path, "member": member.name,
                                                "category": hit[0], "pattern": hit[1]})
                        members.append(member.name)
                layer_reports.append({"layer": layer_path, "member_count": len(members)})
            return {"schema": "m43-layer-scan-v1", "image": image,
                    "layers": layer_reports, "violations": violations,
                    "forbidden_count": len(violations)}
```
`layer.getmembers()`만 읽고 `extractall()`을 호출하지 않는다 — tar
member를 실제로 디스크에 풀지 않으므로 스캐너 자체가 zip-slip류 실행
위험을 갖지 않는다(스캐너는 관찰만 한다). positive/negative control
fixture는 `tests/unit/test_scan_image_layers.py`가 `tarfile.open(mode="w")`로
그때그때 만든다(§1.1의 fixture 파일을 저장소에 바이너리로 커밋하지
않음) — 하나는 `.git/HEAD`, `runtime/vectorstore/index.faiss`,
`id_rsa`를 포함하는 "forbidden" 레이어, 다른 하나는 `src/`,
`README.md`만 포함하는 "clean" 레이어, 세 번째는 `../../etc/passwd`
경로를 가진 "traversal" 레이어, 네 번째는 `.wh..wh..opq`(OCI opaque
whiteout)와 `.wh.runtime`(디렉터리 whiteout)만 포함하는 "whiteout-only"
레이어(위반 0건이어야 함을 검증), 다섯 번째는
`tests/support/simple_qna_rag_test_seam/deterministic_embeddings.py`
경로 하나만 포함하는 "test-seam-leak" 레이어(신규, DR-I3-MAJ-02 —
`test_embedding_seam` 위반 1건이 검출돼야 함을 검증해 §7.4
`FORBIDDEN_PATTERNS`의 새 항목이 실제로 작동함을 회귀 고정), 여섯
번째는 `web/static/style.css`만 포함하는 "clean-web-asset" 레이어
(DR-I1-MAJ-06 fixture 확장, 위반 0건).

### 7.5 `scripts/container_smoke.py` (DR-I2-MAJ-02 수정: exact bind/reachability + 컨테이너 내부 embedding runtime; DR-I3-MAJ-02 수정: 봉인된 production 이미지 + 별도 harness)

Iteration 2 MAJ-02: 이전 설계는 mock 서버의 정확한 bind 주소를 계약화하지
않았고(`http.server` 기본 동작에 암묵적으로 의존), 더 근본적으로
**컨테이너 안에서 실제로 embedding을 초기화할 방법이 없었다** —
production 이미지는 `BAAI/bge-m3` 모델 가중치를 포함하지 않고(REQ-005.4),
read-only rootfs와 제한된 `/tmp`에서 네트워크로 대형 모델을 받아
캐시할 경로도 정의되지 않았다. 따라서 host에서 real HuggingFace
embedding으로 만든 index를 컨테이너가 query하는 것은 애초에
구현 불가능했다. Iteration 2 개정은 (1) mock 서버의 bind 주소와
reachability 확인을 명시적 계약으로 만들고, (2)
`DeterministicTestEmbeddings` test seam을 host 빌드와 컨테이너 query
양쪽에서 같은 provider로 사용해 네트워크/모델 의존을 제거했다.

**Iteration 3 MAJ-02**: 그 test seam 모듈이 `src/`(production 이미지에
COPY되는 트리) 안에 있었기 때문에, "production 경로에서 활성화
불가능"이라는 문서 주장이 실제 이미지 구조와 모순됐다. §5.2-a가 이
모듈을 `tests/support/simple_qna_rag_test_seam/`으로 옮겨 production
이미지에서 물리적으로 제거했으므로, 이 스크립트는 이제 두 가지를
동시에 만족해야 한다: (a) 여전히 production 이미지 **그 자체**로
hosted Linux smoke(build → activate → serve → query 200)를 수행하되,
test seam은 **읽기 전용 bind mount + PYTHONPATH**로 런타임에만
주입한다(§5.2-a "test image가 아니라 production과 동일한 runtime
filesystem/security boundary를 보존한 별도 명시적 test harness"), (b)
harness 없이 같은 production 이미지에 두 env var만 설정하면 여전히
거부됨을 별도 negative 컨테이너로 증명한다(4-neg 단계, 신규).

```text
사용법: container_smoke.py --image <tag> --output <path>
```
절차:
1. `tests/support/mock_ollama.py`(신규, 테스트 지원 모듈) — 표준
   라이브러리 `http.server`만으로 `POST /api/generate`에 고정 텍스트를
   스트리밍 응답하는 스텁 서버를 `ThreadingHTTPServer(("0.0.0.0", 0),
   Handler)`로 기동한다 — **bind 주소는 `"0.0.0.0"`으로 명시**한다(암묵적
   기본값에 의존하지 않음, `run_json`류 exact-contract 관례와 동일).
   포트는 0(자동 할당)이며 실제 배정된 포트를 `server.server_port`로
   읽는다. 같은 핸들러가 `GET /mock/ping`에 `200 "pong"`을 반환하는
   경량 reachability probe 엔드포인트를 추가로 노출한다(app 수준
   로직과 무관하게 host-gateway 연결 자체만 확인하기 위함).
2. `<tmp-index-root>`에 `EMBEDDING_PROVIDER=deterministic_test`,
   `ALLOW_TEST_EMBEDDING=True`로 `index/lifecycle.py::build()`를
   **테스트 fixture corpus**(2~3개 짧은 txt, 저장소에 이미 있는
   `evaluation/datasets/`류 자산 재사용 또는 인라인 문자열)로 직접
   Python 함수 호출해 작은 INDEX_ROOT를 준비하고 `activate()`한다 —
   `container_smoke.py`는 host(CI checkout, 전체 저장소 트리 보유)에서
   실행되므로 `sys.path.insert(0, str(REPO_ROOT / "tests" / "support"))`
   후 `simple_qna_rag_test_seam.deterministic_embeddings`를 직접
   import해 `DeterministicTestEmbeddings`(§5.2-a)를 쓴다 — 네트워크나
   모델 다운로드가 전혀 없다(수초가 아니라 수 밀리초 내 완료).
3. **host-gateway reachability probe(신규, DR-I2-MAJ-02)**: `docker run
   -d ...`로 컨테이너를 기동한 직후, 앱이 준비될 때까지 기다리기 전에
   `docker exec <container_id> python -c "import urllib.request as u;
   u.urlopen('http://host.docker.internal:<port>/mock/ping',
   timeout=5).read()"`를 실행한다(이미지에 이미 있는 Python 인터프리터만
   사용 — 새 바이너리 의존을 추가하지 않는다). 0이 아닌 exit 또는
   예외면 `host_gateway_reachable=false`로 즉시 FAIL하고 이후 단계
   (readiness/rag 확인)는 진행하지 않는다 — 이 probe가 실패하는데
   `/rag` 실패만으로 원인을 추정하지 않기 위한 독립 진단 지점이다.
4. **같은 production 이미지**를 `docker run -d --user 10001:10001
   --read-only --tmpfs /tmp --security-opt no-new-privileges --cap-drop
   ALL --add-host host.docker.internal:host-gateway -v
   <tmp-index-root>:/app/runtime/index:ro -v
   <repo>/tests/support:/opt/m43-test-seam:ro -e
   PYTHONPATH=/opt/m43-test-seam -e
   SIMPLE_QNA_RAG_INDEX_ROOT=/app/runtime/index -e
   SIMPLE_QNA_RAG_EMBEDDING_PROVIDER=deterministic_test -e
   SIMPLE_QNA_RAG_ALLOW_TEST_EMBEDDING=1 -e
   SIMPLE_QNA_RAG_OLLAMA_BASE_URL=http://host.docker.internal:<port> -p
   <host-port>:8000 <image>`로 기동(§7.3과 동일한 `--add-host`,
   DR-I1-MAJ-07). **신규(DR-I3-MAJ-02)**: `-v
   <repo>/tests/support:/opt/m43-test-seam:ro`와 `-e
   PYTHONPATH=/opt/m43-test-seam`이 이번에 새로 추가된 harness 주입
   경로다 — production 이미지 자체는 재빌드하지 않고, `--read-only`
   rootfs 위에 **읽기 전용** 볼륨 하나를 더 마운트해 `import
   simple_qna_rag_test_seam.deterministic_embeddings`가 컨테이너 안에서도
   성공하게 만든다. `USER`/`--read-only`/`--cap-drop ALL`/
   `--security-opt no-new-privileges`는 §7.3과 완전히 동일하므로 "test
   image가 아니라 production과 동일한 runtime filesystem/security
   boundary를 보존한 별도 harness"라는 요구(§5.2-a)를 그대로 만족한다.
   컨테이너의 `EMBEDDING_PROVIDER`/`ALLOW_TEST_EMBEDDING`은 2단계에서
   host build에 쓴 것과 **정확히 같은 값**이어야 `_verify_settings_binding`(§3.3)이
   통과한다 — 다르면 readiness가 503 `artifact_settings_mismatch`로
   fail-closed하므로 이 스텝 자체가 "host/컨테이너 provider 불일치"
   회귀를 잡는 검증이기도 하다.
4-neg. **production 활성화 거부 negative control(신규, DR-I3-MAJ-02)**:
   4단계와 **완전히 같은 production 이미지**로 별도 컨테이너를
   `docker run -d --user 10001:10001 --read-only --tmpfs /tmp
   --security-opt no-new-privileges --cap-drop ALL -e
   SIMPLE_QNA_RAG_EMBEDDING_PROVIDER=deterministic_test -e
   SIMPLE_QNA_RAG_ALLOW_TEST_EMBEDDING=1 -p <neg-host-port>:8000
   <image>`로 기동한다 — **harness volume/`PYTHONPATH`는 의도적으로
   생략**한다(이 컨테이너가 4단계와 동일한 이미지 tag라는 사실 자체가
   "test 전용 이미지가 아니라 production 경로에서 거부됨"의 증거다).
   최대 10초 동안 1초 간격으로 `GET /health/ready`를 bounded polling해
   503과 body의 `reason == "artifact_test_embedding_seam_unavailable"`을
   확인한다 — polling 중 한 번이라도 200이 관찰되면 즉시
   `production_test_seam_not_sealed=true`로 FAIL. 확인 후 `docker stop
   -t 5`로 정리하고 `production_test_seam_sealed` bool을 receipt에
   기록한다(9단계).
5. `GET /health/live` == 200, `GET /health/ready` == 200(마운트된 index가
   유효하므로) 확인.
6. **정적 자산/루트 페이지 확인(DR-I1-MAJ-06)**: `GET /` == 200이고
   응답 본문이 `web/templates/`가 렌더링하는 HTML(고정 문자열 마커,
   예: `<title>`)을 포함하는지 확인. `GET /static/<실제 존재하는
   자산 1개, 예: style.css>` == 200과 `Content-Type` 확인. 이 두
   검사가 실패하면(예: 404) `static_asset_missing`으로 즉시 FAIL —
   §7.1에서 `web/static`/`web/templates`를 COPY하지 않으면 이 스텝이
   바로 이 실패를 재현한다(회귀 방지).
7. `POST /rag`에 mock corpus 관련 질문 1건 전송 — mock Ollama가 고정
   텍스트를 반환하고 embedding은 `DeterministicTestEmbeddings`이므로
   200과 고정 답변 포함 여부만 확인(실제 LLM/검색 품질 비교 아님,
   REQ-005.5 "mock query"의 문자 그대로 구현).
8. `docker stop -t 10 <container>`로 graceful stop, exit code와 stop
   소요시간 기록.
9. JSON receipt(`schema: "m43-container-smoke-v1"`)에
   `security_options_applied`, `readiness_sequence`, `host_gateway_reachable`,
   `root_page_ok`, `static_asset_ok`, `mock_query_ok`,
   `production_test_seam_sealed`(신규, DR-I3-MAJ-02 — 4-neg 단계의
   결과), `embedding_provider`(`"deterministic_test"` 리터럴),
   `graceful_stop_seconds`, `image_digest`(`docker inspect --format
   '{{.Id}}'`) 기록.
   `host_gateway_reachable`/`mock_query_ok`/`root_page_ok`/`static_asset_ok`/`production_test_seam_sealed`는
   모두 명시적 `bool`이며(§8.2 assembler의 semantic 검증이 이 필드들을
   직접 읽는다, DR-I1-MAJ-08/DR-I2-MAJ-03 참조), 다섯 중 하나라도
   `false`면 `container_smoke.py` 자신이 exit 1을 반환해 CI job을
   실패시킨다.

`docker` 바이너리가 없는 로컬 환경에서는 `shutil.which("docker") is None`이면
즉시 `{"status": "SKIPPED", "reason": "docker_unavailable"}`로 exit 0 —
GitHub-hosted `ubuntu-latest`에는 Docker가 사전 설치돼 있으므로 CI에서는
항상 실행된다(로컬 개발 편의를 위한 예외이며 CI Gate 판정에는 영향
없음 — CI job은 `docker` 존재를 전제하고 SKIPPED가 나오면 그 자체를
job 실패로 별도 assert한다).

**exact argv 계약**: `build_docker_run_argv(image, *, index_root, host_port,
mock_port, test_seam_dir) -> list[str]`, `build_negative_activation_argv(image,
*, neg_host_port) -> list[str]`(신규, DR-I3-MAJ-02), `build_reachability_probe_argv(container_id,
mock_port) -> list[str]`를 `container_smoke.py`의 순수 함수로 분리한다
— docker/subprocess를 실제로 호출하지 않고 이 함수들의 반환값만
비교하는 `tests/unit/test_container_smoke_contract.py::
test_docker_run_argv_includes_add_host_and_embedding_seam_env`,
`test_reachability_probe_argv_targets_mock_ping_via_host_gateway`,
`test_negative_activation_argv_omits_test_seam_mount_and_pythonpath`(신규
— `build_negative_activation_argv`의 반환 리스트에 `-v .../tests/support`나
`PYTHONPATH` 관련 문자열이 **하나도** 없음을 assert해 4-neg 단계가
harness를 실수로 함께 넘기는 회귀를 잡는다)가 `--add-host
host.docker.internal:host-gateway`, `SIMPLE_QNA_RAG_EMBEDDING_PROVIDER=deterministic_test`,
`SIMPLE_QNA_RAG_ALLOW_TEST_EMBEDDING=1`, `-v <repo>/tests/support:/opt/m43-test-seam:ro`,
`PYTHONPATH=/opt/m43-test-seam`(4단계에만 존재, 4-neg에는 부재)과
reachability probe의 정확한 `docker exec ... python -c ...` 형태를
회귀 고정한다(docker 없이 CI 어디서나 실행 가능).

### 7.6 `docs/operations/` runbook

`docs/operations/deployment_runbook.md` 목차:
1. Preflight — digest 확인(`docker pull <image>@sha256:...`), settings
   확인(`simple-qna-rag-index-lifecycle list --index-root <root>`로
   현재 pointer 확인), Ollama/model preflight는 **운영자 수동 절차**로
   `scripts/preflight_ollama.py <url> <model>` 명령만 기록하고 이 cycle에서
   실행하지 않았음을 명시(REQ-006.1).
2. Volume owner/mode — `chown -R 10001:10001 <host-index-root>`,
   `find <host-index-root> -type d -exec chmod 0555 {} \;`.
3. Index verify/activate — `simple-qna-rag-index-lifecycle verify --version
   <id>` → `activate --to-version <id>`.
4. Restart — `docker compose restart app`(또는 `systemctl restart
   simple-qna-rag`) 예시, readiness 폴링 루프(`curl -sf
   http://localhost:8000/health/ready`를 지수 백오프로 최대 N회).
5. Smoke — 고정 질문 1건 `/rag` 호출과 기대 status.
6. Release identity 기록 — `docs/operations/deployment_runbook.md`가
   요구하는 배포 전/후 스냅샷 표: `{image_digest, current_version_id,
   settings_hash, dependency_lock_sha256}`을 배포 전/후 각각 기록하고
   두 스냅샷이 "같은 release identity"인지 대조(REQ-006.2).

`docs/operations/recovery_runbook.md` 진단 표(REQ-006.3, 발췌):

| 증상 | 진단 명령 | 기대 결과 | 복구 순서 |
|---|---|---|---|
| readiness 503 `artifact_manifest_schema_invalid` | `simple-qna-rag-index-lifecycle verify --version <current>` | exit 1, `error_code` 확인 | 이전 version으로 rollback(§7.6-7) |
| readiness 503 `artifact_member_hash_mismatch` | 동일 | exit 1 | rollback, 손상 원인 조사(디스크/전송) |
| `activate` exit 3 | 없음(즉시 판별) | `error_code=lock_timeout` | 동시 실행 중인 lifecycle 프로세스 확인 후 재시도 |
| disk full 중 `build` 실패 | `df -h <index-root 마운트>` | 여유 공간 확보 | `.staging/<op>` 잔여물은 무해(inactive) — 필요 시 수동 `cleanup`은 아님, staging 정리는 별도 운영 절차(§4.2 주석) |
| Ollama outage | `scripts/preflight_ollama.py` | 연결 실패 보고 | Ollama 복구 후 재시도, index/activation과 무관 |
| container start 실패(읽기 전용 위반) | `docker logs <container>` | permission denied 등 | 볼륨 owner/mode(§7.6-2) 재확인 |

rollback 절차(REQ-006.4):
```text
1. traffic 중지 또는 내부 bind 확인(로드밸런서/헬스체크에서 이 인스턴스 제외)
2. 이전 image digest와 index version을 배포 전 스냅샷(§7.6-6)에서 확인
3. simple-qna-rag-index-lifecycle rollback --to-version <이전 id>
   (verify 실패 시 여기서 중단 — 더 진행하지 않음)
4. docker run 이전 image digest로 재기동
5. readiness 확인 — 실패 시 escalation(운영 채널)으로 중단, 추가 mutation 없음
```

backup/restore(REQ-006.5): `tar czf backup-<version-id>.tar.gz -C
<index-root>/versions <version-id>` 형태로 **불변 version 디렉터리
전체**(manifest 포함)를 백업한다. restore는 `tar xzf` 후 곧바로
`versions/`에 놓지 않고, `.staging/<새 op>/`로 풀어 `verify_version`과
동일한 해시 검사를 통과해야만(§4.3 publish와 동일 경로) `versions/`에
들어간다 — "untrusted restored pickle을 바로 활성화하지 않는다"를
코드 경로 재사용으로 보장한다(restore 전용 fast-path가 없다).

### 7.7 `scripts/deploy_drill.py`

```text
python scripts/deploy_drill.py --root <tmp> --repeat 3 --output drill.json
```
절차(mock, 실제 Docker/Ollama 불필요): 임시 `INDEX_ROOT`에 대해
`build → activate → (RAGEngine 재초기화 시뮬레이션: RAGEngine._initialized
= False; get_rag_engine() 재호출) → readiness 확인 → rollback(초기
버전 없음이므로 2회차부터) → readiness 확인`을 3회 반복하고, 마지막에
`fault injection` 4종(REQ-006.3 목록 중 결정론적으로 재현 가능한 것들 —
manifest 손상은 `manifest.json` 바이트 일부 뒤집기, disk full은
`os.statvfs` 대신 `_write_fsync`를 monkeypatch해 `OSError(ENOSPC)`를
주입, lock contention은 별도 프로세스/스레드가 lock을 잡은 채로
`activate` 시도, readiness failure는 settings_mismatch 유발)를 실행해
"검증 실패 뒤 후속 mutation 0"을 각 케이스에서 `current` 파일 해시로
확인한다. 최종 image/index identity가 시작 identity와 같음을
`drill.json`에 기록한다(REQ-006.6 RTO 계열 지표는 각 단계의 wall-clock
소요시간을 초 단위로 receipt에 남겨 대체한다).

## 8. 단일 workflow — `.github/workflows/ci.yml` (REQ-007)

### 8.1 신규/변경 job 배치

`python-tests`/`frontend-tests`는 기존 step을 모두 유지하고 각각 **마지막
step으로** 1개씩 추가한다:

```yaml
      - name: Write CI producer receipt (M4.3 evidence identity)
        if: success()
        run: >
          python scripts/write_ci_producer_receipt.py --job python-tests
          --output ci_producer_receipt.json
      - name: Upload producer receipt
        if: success()
        uses: actions/upload-artifact@v4
        with:
          name: m43-evidence-python-tests
          path: ci_producer_receipt.json
          if-no-files-found: error
          retention-days: 90
```
(`frontend-tests`도 `--job frontend-tests`, artifact 이름
`m43-evidence-frontend-tests`로 동일 패턴, 동일하게
`if-no-files-found: error`.) 모든 `upload-artifact` step에
`if-no-files-found: error`를 명시하는 것은 DR-I1-MAJ-08 수정의 일부다
— 기본값(`warn`)은 선언한 path가 하나도 없어도 step을 success로
남기므로, evidence가 통째로 비어도 job 자체는 초록불일 수 있었다.
`error`는 이 실패를 job 실패로 즉시 승격한다. `if: success()`이므로 그
job의 앞선 step 중 하나라도 실패하면 이 step 자체가 실행되지 않고,
따라서 evidence 파일이 애초에 업로드되지 않는다 — "자기보고 성공"이
아니라 "존재 자체가 GHA가 계산한 성공 신호"가 되도록 만든다(§8.2에서
`m4-assemble`이 그래도 `needs.*.result`를 1차 근거로 별도 사용하는
이유는 evidence 파일 부재만으로 job 실패를 구분할 수 없는 경우 — 예:
`upload-artifact` 자체 실패 — 를 이중으로 잡기 위함).

새 `container` job(§7.4/7.5 호출):
```yaml
  container:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build test-stage image
        run: docker build --target test -f deploy/Dockerfile .
      - name: Build production-stage image
        run: docker build --target production -f deploy/Dockerfile -t simple-qna-rag:${{ github.sha }} .
      - name: Scan OCI layers
        run: python scripts/scan_image_layers.py --image simple-qna-rag:${{ github.sha }} --output layer_scan.json
      - name: Container security/mock smoke
        run: python scripts/container_smoke.py --image simple-qna-rag:${{ github.sha }} --output container_smoke.json
      - name: Write CI producer receipt
        if: success()
        run: python scripts/write_ci_producer_receipt.py --job container --output ci_producer_receipt.json
      - name: Upload container evidence
        if: success()
        uses: actions/upload-artifact@v4
        with:
          name: m43-evidence-container
          path: |
            layer_scan.json
            container_smoke.json
            ci_producer_receipt.json
          if-no-files-found: error
          retention-days: 90
```

새 `m43-deterministic` job:
```yaml
  m43-deterministic:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11", cache: pip, cache-dependency-path: requirements.lock}
      - run: python -m pip install --require-hashes -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
      - run: python -m pip install -e . --no-deps
      - name: Run M4.3 deterministic acceptance
        run: python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 --output m43.json
      - name: Run M4.3 negative control (expected exit 1)
        run: |
          set +e
          python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 --inject-evidence-mismatch --output m43-negative.json
          code=$?
          if [ "$code" -ne 1 ]; then echo "negative control did not fail as expected (exit=$code)"; exit 1; fi
      - name: Write CI producer receipt
        if: success()
        run: python scripts/write_ci_producer_receipt.py --job m43-deterministic --output ci_producer_receipt.json
      - name: Upload deterministic evidence
        if: success()
        uses: actions/upload-artifact@v4
        with:
          name: m43-evidence-m43-deterministic
          path: |
            m43.json
            m43-negative.json
            ci_producer_receipt.json
          if-no-files-found: error
          retention-days: 90
```

새 `m4-assemble` job:
```yaml
  m4-assemble:
    needs: [python-tests, frontend-tests, container, m43-deterministic]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Download python-tests evidence
        uses: actions/download-artifact@v4
        continue-on-error: true
        with: {name: m43-evidence-python-tests, path: assemble/python-tests}
      - name: Download frontend-tests evidence
        uses: actions/download-artifact@v4
        continue-on-error: true
        with: {name: m43-evidence-frontend-tests, path: assemble/frontend-tests}
      - name: Download container evidence
        uses: actions/download-artifact@v4
        continue-on-error: true
        with: {name: m43-evidence-container, path: assemble/container}
      - name: Download m43-deterministic evidence
        uses: actions/download-artifact@v4
        continue-on-error: true
        with: {name: m43-evidence-m43-deterministic, path: assemble/m43-deterministic}
      - name: Assemble M4 evidence and baseline
        run: >
          python scripts/assemble_m4_evidence.py --fresh-dir assemble
          --expected-sha ${{ github.sha }}
          --expected-run-id ${{ github.run_id }}
          --expected-run-attempt ${{ github.run_attempt }}
          --expected-workflow-path .github/workflows/ci.yml
          --expected-event ${{ github.event_name }}
          --needs-result python-tests=${{ needs.python-tests.result }}
          --needs-result frontend-tests=${{ needs.frontend-tests.result }}
          --needs-result container=${{ needs.container.result }}
          --needs-result m43-deterministic=${{ needs.m43-deterministic.result }}
          --evidence python-tests=assemble/python-tests/ci_producer_receipt.json
          --evidence frontend-tests=assemble/frontend-tests/ci_producer_receipt.json
          --evidence container=assemble/container/ci_producer_receipt.json
          --evidence m43-deterministic=assemble/m43-deterministic/ci_producer_receipt.json
          --output assemble/m4-baseline.json
      - name: Check M4 baseline state algebra
        run: python scripts/check_m4_baseline.py --candidate assemble/m4-baseline.json --expect-operational-blocked
      - name: Upload M4 baseline
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: m4-baseline
          path: assemble/m4-baseline.json
          if-no-files-found: error
          retention-days: 90
```
`if: always()`와 `continue-on-error: true`(download 단계에서만)를 쓰는
이유: producer가 실패/스킵돼도 `m4-assemble`이 실행돼 "실패했다는
baseline"을 남겨야 하기 때문이다(REQ-007.3 "missing/skipped/cancelled/
expired/malformed evidence는 FAIL이다" — job 자체가 skip되면 baseline이
아예 생성되지 않아 조용히 사라지는 것을 막는다). `assemble_m4_evidence.py`
자체는 파일이 없으면 그 producer를 `MALFORMED`/`FAILED`로 기록하고
계속 진행한다(§8.2).

### 8.2 `scripts/assemble_m4_evidence.py` (DR-I1-MAJ-08 수정: 상세 evidence까지 검증; DR-I5-MAJ-01로 producer receipt exact tagged schema + job binding + duplicate-filename fail-closed 추가)

Iteration 1 MAJ-08: 이전 설계는 producer마다 최소 `ci_producer_receipt.json`
하나의 identity(schema/sha/run/attempt/path/event)만 확인했고,
`layer_scan.json`/`container_smoke.json`/`m43.json` 같은 상세 evidence는
"baseline에 파일명과 hash만 인용"할 뿐 실제로 열어 검증하지 않았다.
`needs.*.result == success`와 최소 receipt만 있으면 container scan이
위반을 발견했거나 smoke가 실패했어도 producer가 `OK`가 될 수 있었다
— 합성 PASS 경로였다. 수정된 설계는 각 producer receipt를 **payload
목록을 포함한 canonical envelope**로 확장하고, assembler가 그 payload
전체의 hash/size와 **semantic 상태**까지 같은 parser로 검증한 뒤에만
`OK`를 낸다.

**DR-I5-MAJ-01(Iteration 5 판정, DR-I4-MAJ-02와 동일 근본 문제의
재발 — [Design_Review_Iteration_5.md](Design_Review_Iteration_5.md)의
유일한 MAJOR)**: DR-I4-MAJ-02가 도입한 `payload_manifest_sha256` 결합은
producer receipt의 `payloads` 리스트를 이미 `{filename: entry}` dict로
축약한 **뒤**에만 계산됐다. 같은 filename이 두 번 선언되면
`declared[entry["filename"]] = entry`에서 마지막 entry가 앞 entry를
조용히 덮어써 duplicate 자체가 dict 축약 시점에 사라지고,
`_evaluate_producer`의 `{p["filename"]: p["sha256"] for p in
doc.get("payloads", [])}` 재축약도 같은 방식으로 duplicate를 삼켰다.
동시에 `_check_identity`는 필수 key가 **포함**됐는지만(`required <=
set(doc)`) 검사해 unknown top-level key를 가진 receipt를 통과시켰고,
`doc["job"]`(receipt가 스스로 선언한 job)을 호출자가 이 evidence slot에
배정한 job과 비교하지 않아 다른 job의 receipt를 그대로 옮겨 놓고
payload 내용만 맞추는 substitution이 통과할 수 있었다. 수정된 설계는
top-level receipt 문서와 `payloads` 리스트 양쪽을 dict/set으로 축약하기
**전에** exact tagged schema로 먼저 파싱하고, 그 파싱이 완전히 성공한
뒤에만 canonical `{filename: sha256}` mapping과 `payload_manifest_sha256`을
계산한다(아래 `_check_identity`/`_verify_payloads` 전면 재작성).

#### 8.2-a 확장된 producer receipt 스키마 (`ci_producer_receipt.json`)

```json
{
  "schema": "m43-producer-receipt-v1",
  "job": "container",
  "sha": "<GITHUB_SHA>", "run_id": "<GITHUB_RUN_ID>", "run_attempt": "<GITHUB_RUN_ATTEMPT>",
  "workflow_path": "<...>", "event_name": "<GITHUB_EVENT_NAME>",
  "semantic_status": "PASS",
  "payload_manifest_sha256": "<64-hex>",
  "payloads": [
    {"filename": "layer_scan.json", "sha256": "<64-hex>", "size_bytes": 1234},
    {"filename": "container_smoke.json", "sha256": "<64-hex>", "size_bytes": 987}
  ]
}
```
(`m43-deterministic` job의 receipt는 같은 스키마에 `payloads: [{"filename":
"m43.json", "sha256": "<64-hex>", "size_bytes": 2345}, {"filename":
"m43-negative.json", "sha256": "<64-hex>", "size_bytes": 2210}]`을 채운다.
**DR-I5-MAJ-01**: payload entry는 오직 identity 세 필드
(`filename`/`sha256`/`size_bytes`)만 갖는 exact schema다 — 이전 개정에
있던 `semantic_field`/`semantic_expected` 정보성 필드는 완전히
제거됐다. 그 필드들은 애초에 assembler 판정에 쓰이지 않았고(container의
semantic 판정은 §8.2-b `REQUIRED_PAYLOADS` spec이, m43 두 파일의 semantic
판정은 §8.2-c `_parse_and_verify_m43_payload`가 각각 payload 파일
bytes에서 독립 재계산했다), receipt의 exact-key schema 표면만 넓혀
"exact 검사"를 어렵게 만드는 죽은 정보였다 — 제거해도 검증 능력은
전혀 줄지 않는다.)

`write_ci_producer_receipt.py --job JOB --payload FILENAME [--payload
FILENAME ...]`이 각 payload 파일을 그 job의 마지막 step에서 실제로 읽어
`filename`/`sha256`/`size_bytes` 세 필드만 채운다 — `python-tests`/
`frontend-tests`는 `payloads: []`(상세 evidence 없음, identity만),
`container`는 `layer_scan.json`과 `container_smoke.json`,
`m43-deterministic`은 `m43.json`/`m43-negative.json`(§10.1) 두 파일을
선언한다. **semantic 판정은 오직 assembler 쪽 spec이 실제 payload 파일
bytes를 다시 읽어 계산**한다(`forbidden_count == 0`,
`host_gateway_reachable`/`mock_query_ok`/`root_page_ok`/`static_asset_ok`/
`production_test_seam_sealed` 모두 `true`, §7.5; m43 두 파일은 §8.2-c) —
receipt는 그 결과를 미리 알거나 자기 보고할 방법이 없다.
`semantic_status`는 이 job이 스스로 계산한 전체 판정(참고용 top-level
필드, `PASS`/`FAIL` enum으로 exact 검사됨, 아래 `_check_identity`)이며,
assembler는 이 필드를 **신뢰하지 않고 payload에서 직접 재계산**한다
(아래 `_verify_payloads`, m43 두 파일은 §8.2-c).

**`payload_manifest_sha256`(신규, DR-I4-MAJ-02)**: 이 job이 선언하는
전체 payload 집합의 identity를 단일 64-hex 값으로 고정한 canonical
manifest hash다.
```python
# write_ci_producer_receipt.py, assemble_m4_evidence.py,
# check_m4_baseline.py 세 스크립트가 각각 이 2줄을 독립적으로 정의한다
# — index/manifest.py::canonical_json_bytes를 import해 재사용하는 것은
# §2.2 builder_git_sha/dirty 계열이 이미 쓰는 "작은 유틸 중복" 선례와
# 같은 관례다(§0.4-5 "fail-closed는 코드가 아니라 테스트가 증명" —
# 세 스크립트가 서로 다른 시점에 같은 계산을 각자 수행해야 아래
# closure가 성립한다).
def _payload_manifest_sha256(payload_hashes: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload_hashes)).hexdigest()
```
`write_ci_producer_receipt.py`는 이 job이 선언하는 모든 payload를 다
쓴 **뒤** `{filename: sha256}` dict(방금 자신이 쓴 `payloads` 리스트에서
파생)로 `payload_manifest_sha256`을 계산해 receipt에 채운다 — 이 값이
"baseline copy 이전, producer 자신이 선언한 identity"다.

#### 8.2-b 알고리즘

CLI(§0.4에서 정한 `ci_acceptance_contract.py`의 flag 기반 fail-closed
관례를 계승):
```text
python scripts/assemble_m4_evidence.py --fresh-dir DIR
  --expected-sha SHA --expected-run-id ID --expected-run-attempt N
  --expected-workflow-path PATH --expected-event EVENT
  --needs-result JOB=RESULT [--needs-result JOB=RESULT ...]
  --evidence JOB=PATH [--evidence JOB=PATH ...]
  --output OUTPUT_PATH
```

```python
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_PRODUCERS = ("python-tests", "frontend-tests", "container", "m43-deterministic")
REQUIRED_PAYLOADS = {   # job -> {filename: (semantic_field(s), expected)}, 없으면 빈 dict
    "python-tests": {}, "frontend-tests": {},
    "container": {
        "layer_scan.json": ("forbidden_count", 0),
        "container_smoke.json": (
            ("host_gateway_reachable", "mock_query_ok", "root_page_ok", "static_asset_ok",
             "production_test_seam_sealed"),  # 신규(DR-I3-MAJ-02)
            (True, True, True, True, True)),
    },
    # m43-deterministic의 두 payload는 DR-I2-MAJ-03 typed parser
    # (`_parse_and_verify_m43_payload`, §8.2-c)로 검증한다 — 일반 field-
    # equality 튜플이 아니라 전용 마커 `"_typed_m43"`을 spec에 넣어
    # `_verify_payloads`가 분기하도록 한다(§8.2-b).
    "m43-deterministic": {
        "m43.json": ("_typed_m43", False),          # expect_negative=False
        "m43-negative.json": ("_typed_m43", True),   # expect_negative=True
    },
}

# DR-I5-MAJ-01: producer receipt를 dict/set으로 축약하기 전에 먼저 이
# exact tagged schema로 파싱한다. `RECEIPT_TOP_KEYS`/`PAYLOAD_ENTRY_KEYS`는
# `<=`(부분집합) 비교가 아니라 `==`(exact-set) 비교에만 쓰인다 — 초과 key든
# 누락 key든 모두 같은 reason으로 fail-closed 거부한다.
RECEIPT_SCHEMA = "m43-producer-receipt-v1"
RECEIPT_TOP_KEYS = frozenset({
    "schema", "job", "sha", "run_id", "run_attempt", "workflow_path",
    "event_name", "semantic_status", "payload_manifest_sha256", "payloads",
})
SEMANTIC_STATUS_ENUM = frozenset({"PASS", "FAIL"})
PAYLOAD_ENTRY_KEYS = frozenset({"filename", "sha256", "size_bytes"})
# 모든 job의 REQUIRED_PAYLOADS filename을 합친 전역 allowlist — job별
# exact-set 비교(아래 `_verify_payloads`) 이전에, receipt가 애초에 알려진
# payload 파일 이름만 선언하는지 먼저 좁힌다(filename allowlist).
KNOWN_PAYLOAD_FILENAMES = frozenset().union(*(set(v) for v in REQUIRED_PAYLOADS.values()))

def assemble(args) -> dict:
    fresh_dir = Path(args.fresh_dir).resolve()
    _assert_no_unexpected_entries(fresh_dir, expected_subdirs=REQUIRED_PRODUCERS)
    needs = dict(args.needs_result)
    evidence_paths = _group_by_job(args.evidence)
    producers = {}
    for job in REQUIRED_PRODUCERS:
        producers[job] = _evaluate_producer(job, needs.get(job), evidence_paths.get(job, []),
                                             fresh_dir, args)
    deterministic_status = "PASS" if all(p["status"] == "OK" for p in producers.values()) else "FAIL"
    return _build_baseline(producers, deterministic_status, args)

def _evaluate_producer(job, needs_result, paths, fresh_dir, args) -> dict:
    if needs_result != "success":
        return {"status": "FAILED_OR_SKIPPED", "needs_result": needs_result}
    if len(paths) == 0:
        return {"status": "MISSING"}
    if len(paths) > 1:
        return {"status": "DUPLICATE_PRODUCER", "count": len(paths)}
    receipt_path = _resolve_contained(paths[0], fresh_dir)
    if receipt_path is None:
        return {"status": "PATH_TRAVERSAL"}
    try:
        doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "MALFORMED"}
    ok, reason = _check_identity(doc, job, args)
    if not ok:
        return {"status": "IDENTITY_MISMATCH", "reason": reason}
    payload_dir = receipt_path.parent   # producer의 다른 evidence 파일도 같은 서브디렉터리
    ok, reason, payload_hashes = _verify_payloads(job, doc, payload_dir, fresh_dir)
    if not ok:
        return {"status": "PAYLOAD_INVALID", "reason": reason}
    # DR-I4-MAJ-02/DR-I5-MAJ-01: payload_hashes는 `_verify_payloads`가
    # (a) 각 entry를 exact-key/type/range/filename-allowlist로 검증하고,
    # (b) raw filename list 길이==unique filename set 길이로 duplicate를
    # (동일 hash든 상이 hash든) fail-closed 거부하고, (c) 그 뒤에야
    # canonical mapping으로 축약해 실제 파일로 hash/size/semantic을
    # 재검증한 뒤에만 반환하는 값이다 — 이 dict 자체가 "assembler
    # output"(assembler가 이 실행에서 직접 확인한 identity)이다.
    # `computed_manifest_sha256`은 그 assembler output에서 독립적으로
    # 계산한 canonical hash이고, `declared_manifest_sha256`은 producer
    # 단계가 자기 receipt에 선언한("baseline copy" 이전) 값이다 — 두 값이
    # 정확히 같아야만 OK를 낸다. 이 비교는 "같은 payload_hashes를 두 번
    # 해싱해 항상 같음을 확인"하는 항진명제가 아니다: receipt가
    # `payload_manifest_sha256`을 위조하거나 갱신을 빠뜨렸는데
    # `payloads`/실제 파일만 바꾼 경우(또는 그 반대) 여기서 걸린다.
    computed_manifest_sha256 = _payload_manifest_sha256(payload_hashes)
    declared_manifest_sha256 = doc["payload_manifest_sha256"]  # _check_identity가 이미 str 타입 보장
    if not _HEX64_RE.fullmatch(declared_manifest_sha256):
        return {"status": "PAYLOAD_INVALID", "reason": "payload_manifest_sha256_malformed"}
    if declared_manifest_sha256 != computed_manifest_sha256:
        return {"status": "PAYLOAD_INVALID", "reason": "payload_manifest_sha256_mismatch"}
    return {"status": "OK", "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "payload_hashes": payload_hashes,
            "payload_manifest_sha256": computed_manifest_sha256}

def _check_identity(doc, job, args) -> tuple[bool, str | None]:
    """doc을 dict/set으로 축약하기 전에 exact tagged schema로 먼저 검사한다
    (DR-I5-MAJ-01) — 여기를 통과한 뒤에야 `_verify_payloads`가 `payloads`를
    다룬다. `job`은 이 receipt가 채워야 하는 evidence slot(호출자인
    `_evaluate_producer`의 루프 변수, §8.2-b `REQUIRED_PRODUCERS` 순회
    항목)이지 receipt가 자기 보고한 값이 아니다."""
    if not isinstance(doc, dict):
        return False, "receipt_not_object"
    if set(doc) != RECEIPT_TOP_KEYS:
        # 초과 key와 누락 key를 하나의 reason으로 묶는다 — exact-set
        # 비교이므로 어느 쪽이든 "receipt가 선언한 identity 표면 자체가
        # 계약과 다르다"는 같은 결함이다.
        return False, "unknown_or_missing_top_level_key"
    if doc["schema"] != RECEIPT_SCHEMA:
        return False, "wrong_schema"
    # DR-I5-MAJ-01: receipt가 스스로 선언한 job이 호출자가 이 evidence
    # slot에 배정한 job과 정확히 같아야 한다 — 그렇지 않으면 다른 job의
    # receipt를 이 slot에 그대로 옮겨 놓고 payload 내용만 이 slot의
    # 요구에 맞추는 substitution이 통과한다(§8.3 "receipt job swap").
    if doc["job"] != job:
        return False, "receipt_job_mismatch"
    if not isinstance(doc["sha"], str) or doc["sha"] != args.expected_sha:
        return False, "cross_sha_mismatch"
    if not isinstance(doc["run_id"], (str, int)) or isinstance(doc["run_id"], bool) \
            or str(doc["run_id"]) != str(args.expected_run_id):
        return False, "cross_run_mismatch"
    if not isinstance(doc["run_attempt"], (str, int)) or isinstance(doc["run_attempt"], bool) \
            or str(doc["run_attempt"]) != str(args.expected_run_attempt):
        return False, "cross_run_attempt_mismatch"
    if not isinstance(doc["workflow_path"], str) or doc["workflow_path"] != args.expected_workflow_path:
        return False, "workflow_path_mismatch"
    if not isinstance(doc["event_name"], str) or doc["event_name"] != args.expected_event:
        return False, "event_mismatch"
    if doc["semantic_status"] not in SEMANTIC_STATUS_ENUM:
        return False, "semantic_status_invalid"
    if not isinstance(doc["payload_manifest_sha256"], str):
        return False, "payload_manifest_sha256_not_string"
    if not isinstance(doc["payloads"], list):
        return False, "payloads_not_list"
    return True, None

def _verify_payloads(job, doc, payload_dir, fresh_dir) -> tuple[bool, str | None, dict[str, str] | None]:
    """`doc["payloads"]`(이미 `_check_identity`가 list임을 보장)를 세 단계로
    검사한다.
    (1) entry 하나하나를 dict/set으로 축약하기 **전에** exact-key/type/
        range/filename-allowlist로 개별 검증한다(DR-I5-MAJ-01) — 실패하면
        즉시 typed FAIL이고 어떤 dict 변환도 일어나지 않는다.
    (2) raw filename list의 길이와 그 unique set의 길이를 비교해 duplicate
        filename을 거부한다 — 같은 filename이 동일 entry(같은 sha256/
        size_bytes)로 반복되든 서로 다른 entry(다른 sha256/size_bytes)로
        반복되든 이 길이 비교 하나로 둘 다 걸린다(마지막 entry가 앞
        entry를 조용히 덮어써 duplicate가 사라지는 경로를 dict 변환 전에
        원천 차단).
    (3) 여기까지 통과한 뒤에만 canonical `{filename: entry}` mapping으로
        축약하고, 그 mapping이 선언한 payload 하나하나가 (a) 실제로
        존재하고 (b) hash/size가 doc의 선언과 일치하며 (c) 그 payload
        내부의 semantic 필드가 REQUIRED_PAYLOADS의 기대값과 일치하는지
        재계산한다 — doc["semantic_status"]는 참고만 하고 신뢰하지 않는다.
        spec이 `("_typed_m43", expect_negative)` 형태이면 generic
        field-equality 대신 §8.2-c의 typed parser를 쓴다(DR-I2-MAJ-03)."""
    required_files = REQUIRED_PAYLOADS.get(job, {})
    raw_payloads = doc["payloads"]

    for entry in raw_payloads:
        if not isinstance(entry, dict) or set(entry) != PAYLOAD_ENTRY_KEYS:
            return False, "payload_entry_schema_invalid", None
        filename, sha256_value, size_bytes = entry["filename"], entry["sha256"], entry["size_bytes"]
        if not isinstance(filename, str) or filename not in KNOWN_PAYLOAD_FILENAMES:
            return False, "payload_entry_filename_not_allowlisted", None
        if not isinstance(sha256_value, str) or not _HEX64_RE.fullmatch(sha256_value):
            return False, "payload_entry_sha256_invalid", None
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            return False, "payload_entry_size_bytes_invalid", None

    raw_filenames = [entry["filename"] for entry in raw_payloads]
    if len(raw_filenames) != len(set(raw_filenames)):
        # DR-I5-MAJ-01: 동일 entry 반복(같은 hash)과 서로 다른 entry로
        # 선언된 반복(다른 hash) 모두 여기서 동일하게 거부된다 — 어느
        # 쪽이든 raw list 길이가 unique filename 개수보다 크므로 별도
        # 분기가 필요 없다.
        return False, "payload_duplicate_filename", None

    # 여기 도달한 시점의 raw_payloads는 이미 (1) exact-key/type/range/
    # allowlist를 통과했고 (2) filename이 unique함이 증명됐다 — 이제야
    # dict로 축약해도 안전하다.
    declared = {entry["filename"]: entry for entry in raw_payloads}
    if set(required_files) != set(declared):
        return False, "payload_set_mismatch", None
    for filename, spec in required_files.items():
        entry = declared[filename]
        target = _resolve_contained(str(payload_dir / filename), fresh_dir)
        if target is None or not target.is_file():
            return False, f"payload_missing:{filename}", None
        actual_bytes = target.read_bytes()
        if len(actual_bytes) != entry["size_bytes"]:
            return False, f"payload_size_mismatch:{filename}", None
        if hashlib.sha256(actual_bytes).hexdigest() != entry["sha256"]:
            return False, f"payload_hash_mismatch:{filename}", None
        if spec[0] == "_typed_m43":
            ok, reason = _parse_and_verify_m43_payload(actual_bytes, expect_negative=spec[1])
            if not ok:
                return False, f"{reason}:{filename}", None
            continue
        try:
            payload_doc = json.loads(actual_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False, f"payload_malformed:{filename}", None
        fields, expected = spec
        if isinstance(fields, tuple):
            actual = tuple(payload_doc.get(f) for f in fields)
        else:
            actual = payload_doc.get(fields)
        if actual != expected:
            return False, f"payload_semantic_mismatch:{filename}", None
    # (1)-(3) 통과 후에만 canonical mapping을 만든다 — 이 시점의 `declared`는
    # 이미 unique/allowlisted/type-checked/실제 파일 검증까지 끝난
    # 상태이므로, 여기서 만든 payload_hashes가 그대로 "assembler output"
    # identity가 된다(§8.2-b `_evaluate_producer`가 이 값으로
    # payload_manifest_sha256을 계산).
    payload_hashes = {filename: declared[filename]["sha256"] for filename in required_files}
    return True, None, payload_hashes

def _resolve_contained(path: str, fresh_dir: Path) -> Path | None:
    resolved = Path(path).resolve()
    if not str(resolved).startswith(str(fresh_dir) + os.sep):
        return None
    return resolved
```

#### 8.2-c `_parse_and_verify_m43_payload` — m43 typed payload parser, 독립 pinned node oracle (DR-I2-MAJ-03 신규, DR-I3-MAJ-03로 독립화)

§10.1이 정의한 `m43.json`/`m43-negative.json` 스키마를 assembler가
`m43.json.status == "PASS"` 한 필드가 아니라 **전체 semantic
completeness**로 재계산한다.

**Iteration 3 MAJ-03**: 이전 개정은 `PROFILE_NODE_IDS`를
`run_m43_acceptance.py`에서 직접 import해 assembler의 exact-set 검사
oracle로 썼다. 두 스크립트가 "우연히 다른 집합을 기대하는" drift는
막았지만, runner 자체의 버그가 필수 node를 constant와 output 양쪽에서
함께 누락하면(예: `PROFILE_NODE_IDS`에서 한 항목을 실수로 지우는 커밋)
assembler도 그 축소된 같은 constant를 import해 exact-set 검사를
통과시킨다 — producer 상세 결과를 consumer가 **독립** 재계산해야
한다는 closure 목적 자체가 무너진다. 수정된 설계는 assembler가 자기
자신의 **review-pinned 독립 상수**를 갖고, `run_m43_acceptance.py`의
`PROFILE_NODE_IDS`를 런타임 판정 경로에서 import하지 않는다 — 두
상수는 물리적으로 분리된 리터럴이며, 둘 사이의 의도적 동기화는 오직
테스트 시점의 cross-check 회귀 테스트로만 이뤄진다.

```python
# assembler 자신의 review-pinned 필수 node ID 집합 — run_m43_acceptance.py를
# import하지 않는다(DR-I3-MAJ-03: runner 버그가 constant와 output 양쪽에서
# node를 함께 누락해도 이 독립 상수는 영향받지 않는다). §10 PROFILE_NODE_IDS의
# key 목록을 review 시점에 그대로 옮겨 적은 리터럴이며, 갱신 시 반드시
# test_expected_node_ids_matches_producer_profile_node_ids가 두 상수의
# 일치를 재확인한다.
EXPECTED_M43_NODE_IDS = frozenset({
    "manifest_canonical", "manifest_negative", "verification_trust",
    "verification_reopen_race", "legacy_baseline_pin", "staging_fault",
    "activation_rollback", "crash_recovery_journal", "lock_untrusted_symlink",
    "legacy_import", "retention", "lock_contention", "layer_scanner",
    "container_static_and_connectivity", "embedding_provider_seam_guard",
    "assemble_payload_verification", "baseline_strict_schema",
})

M43_SCHEMA = "m43-acceptance-receipt-v1"
M43_SEED = 4303
M43_REPEAT = 10
M43_EXPECTED_COMMAND = (f"run_m43_acceptance.py --profile deterministic "
                        f"--repeat {M43_REPEAT} --seed {M43_SEED}")
M43_TOP_KEYS = frozenset({"schema", "profile", "seed", "repeat", "command",
                          "started_at", "finished_at", "nodes",
                          "negative_control", "status"})
M43_NODE_KEYS = frozenset({"repeat", "success_count", "status"})
M43_NEGATIVE_KEYS = frozenset({"executed", "expected_to_fail", "actual_exit_code", "result"})

def _parse_and_verify_m43_payload(raw: bytes, *, expect_negative: bool) -> tuple[bool, str | None]:
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, "m43_payload_malformed_json"
    if not isinstance(doc, dict) or set(doc) != M43_TOP_KEYS:
        return False, "m43_payload_key_mismatch"
    if doc.get("schema") != M43_SCHEMA or doc.get("profile") != "deterministic" or \
       doc.get("seed") != M43_SEED or doc.get("repeat") != M43_REPEAT:
        return False, "m43_identity_mismatch"
    if doc.get("command") != M43_EXPECTED_COMMAND:
        return False, "m43_command_mismatch"
    nodes = doc.get("nodes")
    if not isinstance(nodes, dict) or set(nodes) != EXPECTED_M43_NODE_IDS:
        return False, "m43_node_set_mismatch"
    for name, node in nodes.items():
        if not isinstance(node, dict) or set(node) != M43_NODE_KEYS:
            return False, f"m43_node_schema_mismatch:{name}"
        if node.get("repeat") != M43_REPEAT or node.get("success_count") != M43_REPEAT \
                or node.get("status") != "PASS":
            return False, f"m43_node_not_fully_passed:{name}"
    neg = doc.get("negative_control")
    if not isinstance(neg, dict) or set(neg) != M43_NEGATIVE_KEYS:
        return False, "m43_negative_control_schema_mismatch"
    if expect_negative:
        if neg.get("executed") is not True or neg.get("expected_to_fail") is not True or \
           neg.get("actual_exit_code") != 1 or neg.get("result") != "REJECTED_AS_EXPECTED" or \
           doc.get("status") != "REJECTED_AS_EXPECTED":
            return False, "m43_negative_control_not_rejected"
    else:
        if neg.get("executed") is not False or neg.get("expected_to_fail") is not None or \
           neg.get("actual_exit_code") is not None or neg.get("result") is not None or \
           doc.get("status") != "PASS":
            return False, "m43_positive_status_not_pass"
    return True, None
```
- **독립 node oracle(DR-I3-MAJ-03)**: `nodes`의 key 집합을 assembler
  자신의 `EXPECTED_M43_NODE_IDS`와 비교한다(초과/누락 모두 즉시 실패)
  — `run_m43_acceptance.py`의 `PROFILE_NODE_IDS`를 런타임에 import하지
  않으므로, runner의 버그가 그 constant를 잘못 줄여도(예: 한 node를
  실수로 삭제) assembler는 여전히 review-pinned 전체 집합을 기대해 그
  누락을 `m43_node_set_mismatch`로 잡는다 — 이전 설계가 놓쳤던 "runner
  버그가 constant와 output 양쪽에서 node를 함께 누락"을 이제 assembler가
  독립적으로 검출한다. 두 상수가 legitimate하게 함께 갱신돼야 하는
  경우(신규 node 추가)는
  `tests/unit/test_assemble_m4_evidence.py::test_expected_node_ids_matches_producer_profile_node_ids`가
  두 리터럴(assembler의 `EXPECTED_M43_NODE_IDS`, `run_m43_acceptance.py`의
  `PROFILE_NODE_IDS`)의 불일치를 즉시 실패시켜 "한쪽만 갱신하고 잊음"을
  막는다 — 이 테스트만 두 상수를 모두 import하며, production 판정
  경로(`_parse_and_verify_m43_payload`)는 이 테스트를 거치지 않고
  독립 상수만 쓴다.
- **command exact match(DR-I3-MAJ-03 신규)**: `doc["command"]`이
  assembler 자신의 `M43_REPEAT`/`M43_SEED` 상수로 조립한
  `M43_EXPECTED_COMMAND`와 정확히 다르면 즉시 `m43_command_mismatch`로
  실패한다 — 다른 명령으로 실행한 결과나 명령 문자열만 조작된 receipt를
  차단한다(이전 개정은 `command` 키의 **존재**만 top-level schema로
  확인하고 값은 검사하지 않았다).
- **repeat/seed exact match**: `doc["repeat"] != 10` 또는
  `doc["seed"] != 4303`이면 즉시 실패 — 다른 seed/repeat로 실행해
  결과를 재사용하는 시나리오를 차단한다.
- **각 node의 완전한 반복 성공**: `success_count == repeat == 10`이고
  `status == "PASS"`인 node만 통과 — node 하나라도 부분 성공(예:
  `success_count: 9`)이면 그 node 자체가 실패로 집계된다.
- **negative/positive receipt의 전체 identity 필드 검사(DR-I3-MAJ-03
  신규)**: 이전 개정은 `M43_NEGATIVE_KEYS`에 `expected_to_fail`/`actual_exit_code`
  키가 존재하는지만 schema로 확인하고 **값**은 검사하지 않았다.
  수정된 검사는 negative receipt(`expect_negative=True`)에서
  `expected_to_fail is True`와 `actual_exit_code == 1`을(거부됐다는
  증거), positive receipt(`expect_negative=False`)에서
  `expected_to_fail is None`과 `actual_exit_code is None`을(negative
  control이 아예 실행되지 않았다는 증거) 명시적으로 요구한다 —
  `m43-negative.json`이 존재하고 파싱 가능하다는 사실만으로는 통과하지
  않으며, 위조된 `expected_to_fail=false`나 positive receipt의 가짜
  exit code도 이제 거부된다.
- `tests/unit/test_assemble_m4_evidence.py`에 다음 negative case를
  추가한다(§8.3 표에 반영): 최소 `{"status": "PASS"}` JSON(§10.1 스키마
  없이 status만 있는 합성 파일) → `m43_payload_key_mismatch`; `nodes`에서
  한 node 누락 → `m43_node_set_mismatch`; `repeat`을 5로 변조 →
  `m43_identity_mismatch`; `seed`를 다른 값으로 변조 →
  `m43_identity_mismatch`; 한 node의 `success_count`를 9로 변조 →
  `m43_node_not_fully_passed`; `m43-negative.json`의 `negative_control.result`가
  `"TAMPERING_ACCEPTED_BUG"` → `m43_negative_control_not_rejected`;
  `payloads` entry 자체가 `filename`/`sha256` 키가 없는 malformed dict
  → `_verify_payloads`가 dict로 축약하기 전 entry별 exact-key 검사
  단계에서 즉시 `payload_entry_schema_invalid`로 typed FAIL(DR-I5-MAJ-01
  재작성 — `set(entry) != PAYLOAD_ENTRY_KEYS`가 dict comprehension보다
  먼저 실행되므로 `KeyError`/`TypeError`로 assembler 자체를 죽이는 경로가
  구조적으로 존재하지 않는다. DR-I2-MAJ-03 원문의 "malformed payloads
  entry는 dict comprehension의 KeyError/TypeError로 assembler 자체를
  죽일 수 있다" 지적은 이 exact-key 선검사로 닫힌다 — Iteration 2~4의
  방어적 `.get()` 스킵 방식은 duplicate filename을 조용히 삼키는
  DR-I5-MAJ-01의 원인이었으므로 더 이상 쓰지 않는다).
- **DR-I3-MAJ-03 신규 case**: `command`를 `"run_m43_acceptance.py
  --profile deterministic --repeat 5 --seed 4303"`(다른 repeat 값)로
  변조 → `m43_command_mismatch`; `run_m43_acceptance.py`의
  `PROFILE_NODE_IDS`에서만 한 node를 제거하고(monkeypatch로 시뮬레이션)
  `m43.json`의 `nodes`에서도 같은 node를 제거해 "constant와 output
  양쪽에서 함께 누락"을 재현 → assembler의 독립 `EXPECTED_M43_NODE_IDS`가
  여전히 그 node를 기대하므로 `m43_node_set_mismatch`(이 케이스가
  DR-I3-MAJ-03의 핵심 재현이다 — 이전 설계에서는 이 케이스가 조용히
  통과했다); `m43-negative.json`의 `negative_control.expected_to_fail`을
  `false`로 변조(다른 필드는 정상) → `m43_negative_control_not_rejected`;
  `m43.json`(positive)의 `negative_control.actual_exit_code`를 `0`
  대신 `1`로 위조 → `m43_positive_status_not_pass`; 두 독립 pinned
  상수(`assemble_m4_evidence.EXPECTED_M43_NODE_IDS`,
  `run_m43_acceptance.PROFILE_NODE_IDS`)를 직접 비교하는 별도 테스트
  `test_expected_node_ids_matches_producer_profile_node_ids`(negative
  case가 아니라 provenance 회귀 — 두 상수가 실제로 같은 집합인지
  확인).

`_assert_no_unexpected_entries`는 `fresh_dir` 하위 최상위 항목이
`REQUIRED_PRODUCERS`가 만든 4개 서브디렉터리(download-artifact가 만든
것) 외에 아무것도 없는지 확인한다("fresh empty assemble directory에
중복 없이" — download 단계에서 이미 격리되므로 이 검사는 CI YAML
구성 실수를 잡는 방어선이다). `_resolve_contained`를 receipt 경로와
payload 경로 양쪽에 동일하게 적용해 path traversal 검사를 파서 표면
전체에 일관되게 유지한다.

`python-tests`/`frontend-tests`가 쓰는 `ci_producer_receipt.json`은
`payloads: []`인 §8.2-a 스키마의 최소 형태다(상세 evidence 없음, identity만
검증 대상). `container` job은 `layer_scan.json`/`container_smoke.json`
두 payload를(assembler `REQUIRED_PAYLOADS` spec 기준 host_gateway_reachable
포함 5-field semantic 재계산 대상 — `production_test_seam_sealed` 포함,
DR-I2-MAJ-02/DR-I3-MAJ-02), `m43-deterministic` job은 `m43.json`/`m43-negative.json`
두 payload를(각각 `_typed_m43` 마커로 §8.2-c의 완전한 schema/identity/
node-set/negative-control 재계산 대상, DR-I2-MAJ-03) `payloads` 필드에
채워 넣는다 — `assemble_m4_evidence.py`가 읽는 최상위 파일은 여전히
`ci_producer_receipt.json` 하나뿐이지만(파서 표면을 좁게 유지,
REQ-007.5의 "동일 parser" 요구를 `_check_identity` + `_verify_payloads`
+ `_parse_and_verify_m43_payload` 세 함수로 만족), 그 안의 `payloads`
선언을 통해 상세 evidence의 실제 hash/size/semantic 상태까지 같은
실행에서 재계산·검증한다(DR-I1-MAJ-08, DR-I2-MAJ-02/03). 상세 evidence
파일 자체는 여전히 같은 artifact 안에 함께 올라가 사람이 나중에
내려받아 볼 수 있다.

### 8.3 negative control 목록과 대응

| 시나리오(REQ-007.5) | 재현 방법 | 기대 `producers[job].status` |
|---|---|---|
| 삭제 | `--evidence job=<존재하지 않는 경로>` | `MALFORMED`(read 실패) |
| 변조 | evidence JSON의 `sha` 필드를 다른 값으로 바꿔 저장 | `IDENTITY_MISMATCH: cross_sha_mismatch` |
| cross-run | `run_id` 다른 값 | `IDENTITY_MISMATCH: cross_run_mismatch` |
| cross-SHA | 위 표와 동일(변조 항목과 병합) | 동일 |
| duplicate producer | 같은 job에 `--evidence` 2회 | `DUPLICATE_PRODUCER` |
| skipped producer | `--needs-result job=skipped` | `FAILED_OR_SKIPPED` |
| path traversal | `--evidence job=../outside.json` | `PATH_TRAVERSAL` |
| stale artifact | `run_attempt` 값이 expected와 다름(재시도 run의 옛 artifact) | `IDENTITY_MISMATCH: cross_run_attempt_mismatch` |
| synthesized PASS(최소 receipt만) | `needs-result job=success`이지만 evidence 파일 자체가 없음(가짜 성공 신호만 있고 진짜 receipt가 없는 경우) | `MISSING` |
| synthesized PASS(상세 evidence 위조, DR-I1-MAJ-08) | `container` job의 receipt는 정상 identity를 갖지만 `layer_scan.json`의 실제 `forbidden_count`가 `1`인데 receipt의 `payloads[].sha256`은 `forbidden_count: 0`이던 옛 파일의 해시를 그대로 가리킴(파일이 사후 교체됨) | `PAYLOAD_INVALID: payload_hash_mismatch:layer_scan.json` |
| synthesized PASS(semantic 위조) | payload hash/size는 실제 파일과 일치하지만 `container_smoke.json` 내부의 `mock_query_ok`가 `false`(스모크 자체는 실패했는데 receipt의 `semantic_status`만 `"PASS"`로 자체 보고) | `PAYLOAD_INVALID: payload_semantic_mismatch:container_smoke.json` |
| payload 누락 | receipt는 `m43.json`을 선언하지만 그 파일이 실제 evidence 디렉터리에 없음(업로드 일부 누락) | `PAYLOAD_INVALID: payload_missing:m43.json` |
| payload 집합 불일치 | `container` receipt가 `layer_scan.json`만 선언하고 `container_smoke.json` 선언이 빠짐(REQUIRED_PAYLOADS와 다른 집합) | `PAYLOAD_INVALID: payload_set_mismatch` |
| m43 최소 위조 receipt(DR-I2-MAJ-03) | `m43.json`을 `{"status":"PASS"}`만 있는 파일로 대체(§10.1 스키마 없음) | `PAYLOAD_INVALID: m43_payload_key_mismatch:m43.json` |
| m43 node 누락(DR-I2-MAJ-03) | `nodes`에서 `crash_recovery_journal` 키 삭제 | `PAYLOAD_INVALID: m43_node_set_mismatch:m43.json` |
| m43 repeat/seed 변조(DR-I2-MAJ-03) | `repeat: 5` 또는 `seed: 1234`로 변조 | `PAYLOAD_INVALID: m43_identity_mismatch:m43.json` |
| m43 node 부분 성공(DR-I2-MAJ-03) | 한 node의 `success_count: 9`(repeat 10 미만) | `PAYLOAD_INVALID: m43_node_not_fully_passed:m43.json` |
| m43 negative control 미거부(DR-I2-MAJ-03) | `m43-negative.json`의 `negative_control.result`를 `"TAMPERING_ACCEPTED_BUG"`로 변조 | `PAYLOAD_INVALID: m43_negative_control_not_rejected:m43-negative.json` |
| m43 command 변조(DR-I3-MAJ-03) | `command`를 `--repeat 5`가 섞인 다른 문자열로 변조 | `PAYLOAD_INVALID: m43_command_mismatch:m43.json` |
| m43 runner constant+output 동시 누락(DR-I3-MAJ-03) | `PROFILE_NODE_IDS`와 `m43.json.nodes` 양쪽에서 같은 node를 함께 제거(runner 버그 재현) | `PAYLOAD_INVALID: m43_node_set_mismatch:m43.json`(assembler의 독립 `EXPECTED_M43_NODE_IDS`가 여전히 그 node를 기대) |
| m43 negative expected_to_fail 위조(DR-I3-MAJ-03) | `m43-negative.json`의 `negative_control.expected_to_fail`을 `false`로 변조 | `PAYLOAD_INVALID: m43_negative_control_not_rejected:m43-negative.json` |
| m43 positive receipt 가짜 exit code(DR-I3-MAJ-03) | `m43.json`(positive)의 `negative_control.actual_exit_code`를 `null`이 아닌 `1`로 위조 | `PAYLOAD_INVALID: m43_positive_status_not_pass:m43.json` |
| payload entry malformed(DR-I2-MAJ-03, DR-I5-MAJ-01로 reason 재정의) | `payloads[0]`에서 `"filename"` 키를 삭제한 dict(모자란 키); 별도 variant로 알 수 없는 extra key를 추가한 dict(초과 키) | `PAYLOAD_INVALID: payload_entry_schema_invalid`(assembler crash 없이 typed FAIL, dict 축약 전 거부) |
| host-gateway 미연결(DR-I2-MAJ-02) | `container_smoke.json`의 `host_gateway_reachable`을 `false`로 변조(파일 hash도 그 파일에 맞게 재계산 — hash mismatch가 아니라 semantic mismatch만 격리 검증) | `PAYLOAD_INVALID: payload_semantic_mismatch:container_smoke.json` |
| production test seam 미봉인(DR-I3-MAJ-02) | `container_smoke.json`의 `production_test_seam_sealed`를 `false`로 변조(파일 hash도 그 파일에 맞게 재계산) | `PAYLOAD_INVALID: payload_semantic_mismatch:container_smoke.json` |
| payload-manifest hash malformed(DR-I4-MAJ-02 신규) | receipt의 `payload_manifest_sha256`을 32-hex(잘못된 길이)로 변조 | `PAYLOAD_INVALID: payload_manifest_sha256_malformed` |
| payload-manifest hash mismatch(DR-I4-MAJ-02 신규) | `payloads[]`/실제 파일은 무변경, receipt의 `payload_manifest_sha256`만 무작위 다른 64-hex 값으로 변조(재계산 누락을 재현) | `PAYLOAD_INVALID: payload_manifest_sha256_mismatch` |
| unknown top-level key(DR-I5-MAJ-01 신규) | receipt에 계약에 없는 key(예: `"note": "..."`) 하나를 추가로 삽입 | `IDENTITY_MISMATCH: unknown_or_missing_top_level_key` |
| missing top-level key(DR-I5-MAJ-01 신규) | receipt에서 `event_name` key 자체를 삭제 | `IDENTITY_MISMATCH: unknown_or_missing_top_level_key`(같은 exact-set 비교가 초과/누락 모두 잡는다) |
| wrong schema literal(DR-I5-MAJ-01 신규) | `"schema": "m43-producer-receipt-v0"`(다른 버전 문자열)로 변조 | `IDENTITY_MISMATCH: wrong_schema` |
| receipt job swap(DR-I5-MAJ-01 신규, DR-I4-MAJ-02 원문 예시의 실제 재현) | `container` evidence slot(`--evidence container=...`)에 `"job": "m43-deterministic"`이라고 자칭하는 receipt를 두고 payload 내용만 `container` spec에 맞춤 | `IDENTITY_MISMATCH: receipt_job_mismatch` |
| semantic_status enum 위반(DR-I5-MAJ-01 신규) | `"semantic_status": "MAYBE"`(`PASS`/`FAIL` 외 값)로 변조 | `IDENTITY_MISMATCH: semantic_status_invalid` |
| payloads가 list가 아님(DR-I5-MAJ-01 신규) | `"payloads": {"layer_scan.json": {...}}`(list 대신 dict)로 변조 | `IDENTITY_MISMATCH: payloads_not_list` |
| payload filename allowlist 위반(DR-I5-MAJ-01 신규) | `container` receipt의 한 entry `filename`을 `"evil.json"`(어떤 job의 REQUIRED_PAYLOADS에도 없는 이름)으로 변조 | `PAYLOAD_INVALID: payload_entry_filename_not_allowlisted` |
| size_bytes에 bool 대입(DR-I5-MAJ-01 신규) | 한 payload entry의 `"size_bytes": true`(Python에서 `int`의 subclass라 타입 검사만으로는 통과할 수 있는 값) | `PAYLOAD_INVALID: payload_entry_size_bytes_invalid` |
| duplicate filename, 동일 hash(DR-I5-MAJ-01 신규, DR-I5-MAJ-01 원문 예시) | `container` receipt의 `payloads`에 `layer_scan.json` entry를 완전히 같은 `sha256`/`size_bytes`로 두 번 넣음(raw list 길이 3, unique filename 2) | `PAYLOAD_INVALID: payload_duplicate_filename` |
| duplicate filename, 상이 hash(DR-I5-MAJ-01 신규) | `container` receipt의 `payloads`에 `layer_scan.json` entry를 서로 다른 `sha256`/`size_bytes`로 두 번 넣음(마지막 entry만 실제 파일과 일치) | `PAYLOAD_INVALID: payload_duplicate_filename`(마지막 entry가 실제 파일과 일치해도 raw list 길이 검사가 dict 축약보다 먼저 실행되므로 거부) |

`tests/unit/test_assemble_m4_evidence.py`는 이 표의 37행(기존 27행 +
DR-I5-MAJ-01 신규 10행, `payload entry malformed` 행의 extra-key
variant를 포함하면 38개 파라미터화 case)을 각각 파라미터화된 테스트
케이스로 구현하고, 추가로 "정상 4개 producer 전부 OK"인 positive 케이스
1개와 §8.2-c의 독립 provenance 회귀
(`test_expected_node_ids_matches_producer_profile_node_ids`, 표에는
없음 — negative case가 아니라 두 pinned 상수의 일치 확인) 1개를 더해
총 40개 이상의 케이스로 `deterministic_status`가 표대로 `FAIL`(37행에서
파생된 38개 negative case) 또는 `PASS`(1개)가 되는지 확인한다. `_check_identity`/`_verify_payloads`의
새 reason들은 각각
`test_check_identity_rejects_unknown_or_missing_top_level_key`,
`test_check_identity_rejects_wrong_schema_literal`,
`test_check_identity_rejects_receipt_job_mismatch`,
`test_check_identity_rejects_invalid_semantic_status_enum`,
`test_check_identity_rejects_non_list_payloads`,
`test_verify_payloads_rejects_malformed_payload_entry_schema`(모자란 키/초과
키 두 variant를 한 테스트에서 파라미터화),
`test_verify_payloads_rejects_unknown_filename_not_allowlisted`,
`test_verify_payloads_rejects_bool_as_int_size_bytes`,
`test_verify_payloads_rejects_duplicate_filename_same_hash`,
`test_verify_payloads_rejects_duplicate_filename_different_hash`
10개 함수로 1:1 대응한다(DR-I4-MAJ-02가 남긴 same-count filename
substitution/extra+omission 상쇄/cross-job filename swap 5개
baseline-checker-level negative oracle은 §9.2에 그대로 보존되며, 이
행들은 그보다 앞선 assembler 단계에서 원본 receipt 자체의 duplicate/
schema/job identity를 닫는다). positive 케이스의 각 producer
receipt는 `payload_manifest_sha256`을 그 receipt가 선언한
`payload_hashes`에서 올바르게 재계산한 값으로 채운다 — 그렇지 않으면
새 `payload_manifest_sha256_mismatch` 검사 자체가 positive 케이스를
막아버리므로, 이 필드를 빠뜨리면 fixture 헬퍼가 즉시 실패해 드러난다.

## 9. M4 baseline (REQ-008)

### 9.1 스키마

```json
{
  "schema": "m4-baseline-v1",
  "schema_version": "1.0.0",
  "generated_at": "2026-08-12T00:00:00Z",
  "git_sha": "<expected-sha>",
  "workflow_run": {
    "run_id": "...", "run_attempt": "...",
    "workflow_path": ".github/workflows/ci.yml", "event_name": "..."
  },
  "m3_fingerprint_reference": "evaluation/baselines/m3_initial.json",
  "dependency_snapshot_sha256": "<dependency_snapshot.json의 lock_sha256_canonical>",
  "settings_hash": "<§5.4 스냅샷의 sha256>",
  "image_digest": "<container evidence의 docker inspect Id, 없으면 null>",
  "m43_deterministic_receipt_sha256": "<m43-deterministic evidence 내 m43.json sha256>",
  "producers": { "python-tests": {"status": "OK", ...}, "...": "..." },
  "gates": {
    "python_tests": "PASS", "frontend_tests": "PASS",
    "container": "PASS", "m43_deterministic": "PASS",
    "m3_live_regression": "NOT_RUN",
    "m41_operational": "BLOCKED"
  },
  "deterministic_status": "PASS",
  "operational_status": "BLOCKED",
  "M4.1_BLOCKED": true,
  "overall_release_ready": false
}
```
`gates.python_tests`/`frontend_tests`/`container`/`m43_deterministic`은
`producers[job].status == "OK"`이면 `"PASS"`, 그 외
(`FAILED_OR_SKIPPED`/`MISSING`/`DUPLICATE_PRODUCER`/`IDENTITY_MISMATCH`/
`PATH_TRAVERSAL`/`MALFORMED`)는 전부 `"FAIL"`로 접힌다(enum이 6종뿐이므로
"어느 실패든 FAIL"로 정규화 — 상세 사유는 `producers` 아래에 그대로
남는다). `gates.m3_live_regression`은 이번 cycle에서 항상 `"NOT_RUN"`
리터럴이고, `gates.m41_operational`은 항상 `"BLOCKED"` 리터럴이다 —
`assemble_m4_evidence.py`가 이 두 값을 계산하는 로직 자체를 갖지 않고
상수로 박아 넣는다(REQ-008.4 "protected live evidence 미실행 상태에서...
release-ready 주장을 금지"를 코드가 값을 계산하려는 시도조차 하지
않는 방식으로 강제).

### 9.2 `scripts/check_m4_baseline.py` (DR-I1-MIN-09 / DR-I2-MAJ-04 수정: strict schema + producer→gate 재계산; DR-I3-MAJ-04로 producer variant tagged-union schema 추가; DR-I4-MAJ-02로 payload exact filename set + manifest hash identity 추가)

Iteration 1 MIN-09: 이전 설계는 `candidate["gates"].items()`처럼 **존재하는**
키만 순회해 enum을 검사했다 — `python_tests` 같은 필수 gate가 통째로
빠져도 `.items()`는 그냥 짧은 dict를 순회할 뿐 별도 issue를 내지
않았고, `deterministic_status`도 producer 상태에서 재계산하지 않고
candidate가 자기 보고한 값을 그대로 신뢰했다. Iteration 1 개정은
exact-key 검사와 `gates`→`deterministic_status` 재계산은 추가했지만,
Iteration 2 MAJ-04는 그 재계산이 여전히 candidate가 자기 보고한
**`gates` 자체**를 입력으로 삼는다는 것을 지적했다 — `producers`
필드는 exact-key/schema/status를 전혀 검사하지 않으므로
`producers={}`이거나 모든 producer가 `MISSING`이어도 `gates`
네 필드를 `"PASS"`로 써넣으면 checker를 통과한다. Iteration 2 개정은
`producers`도 `gates`와 같은 수준으로 exact-key/enum 검사하고, **gate
값 자체를 producer status에서 다시 계산**한 뒤에야 그 재계산된 값을
candidate의 `gates`와 비교하도록 만들었다.

**Iteration 3 MAJ-04**: 그 producer-level 검사가 각 producer entry에
`"status" in entry`만 확인했다 — dict이고 status 키만 있으면 통과하므로,
네 entry를 모두 `{"status": "OK"}`로 만든 synthetic candidate는 receipt
hash, `needs_result`, payload hash 등 어떤 metadata도 없이 네 gate와
`deterministic_status=PASS`를 통과시켰다. §8.2-b가 실제로 만드는 producer
결과는 **status별로 필요한 필드가 다른 tagged union**(`OK`는
`receipt_sha256`/`payload_hashes`, `FAILED_OR_SKIPPED`는 `needs_result`,
`IDENTITY_MISMATCH`/`PAYLOAD_INVALID`는 `reason`, `DUPLICATE_PRODUCER`는
`count`, `MISSING`/`PATH_TRAVERSAL`/`MALFORMED`는 `status`뿐)인데도, 이전
개정은 그 구조를 전혀 모델링하지 않았다. 수정된 설계는 이 tagged union을
`PRODUCER_STATUS_SCHEMA`로 명문화하고, 각 producer entry의 key 집합을 그
status에 대응하는 exact set과 비교한 뒤에야 `OK` entry의 `receipt_sha256`(64-hex
SHA-256 문자열)과 `payload_hashes`(64-hex 값을 가진 dict, 그 job이 선언한
payload 개수와 일치)를 형식 검증한다 — 최소-status-only candidate,
success metadata 누락/추가/타입 오류, failure variant에 success-only
필드가 섞인 candidate를 모두 여기서 거부한다.

**Iteration 4 MAJ-02**: 그 `payload_hashes` 검사가 **개수**와 값의
형식(64-hex)만 확인했다 — `container`에 `{"a": "<64hex>", "b":
"<64hex>"}`, `m43-deterministic`에 임의 두 filename을 넣어도 개수(2개)와
형식(64-hex)만 맞으면 통과했다. 즉 baseline candidate가 assembler가
실제로 검증한 `layer_scan.json`/`container_smoke.json` 또는
`m43.json`/`m43-negative.json`이라는 **filename identity**를 보존한다는
보장이 없었다(same-count filename substitution, extra+omission 상쇄,
cross-job filename 교환이 모두 통과). 수정된 설계는 count 상수
(`PRODUCER_EXPECTED_PAYLOAD_COUNT`)를 job별 review-pinned **exact
filename set**(`PRODUCER_EXPECTED_PAYLOAD_FILENAMES`)으로 교체해
`set(payload_hashes) == PRODUCER_EXPECTED_PAYLOAD_FILENAMES[job]`을
검사하고, `OK` entry에 §8.2-a/§8.2-b가 새로 추가한
`payload_manifest_sha256`(receipt가 선언한 canonical payload-manifest
hash를 assembler가 자신이 재검증한 `payload_hashes`와 대조해 일치를
확인한 뒤에만 실어 보내는 값, §8.2-b)도 포함시켜, checker가 그 값을
`payload_hashes`에서 **독립적으로 재계산**한 값과 다시 대조한다 —
assembler output(이 실행에서 재검증된 `payload_hashes`)과 baseline
copy(candidate에 실려온 `payload_manifest_sha256`)의 identity를
checker 시점에서 한 번 더 결합해 확인한다.

**DR-I5-MAJ-01과 이 checker의 관계**: 이 절의 `check()`는 원본 producer
receipt(`ci_producer_receipt.json`)를 전혀 열지 않는다 — 입력은 오직
`assemble_m4_evidence.py::assemble()`(§8.2-b)이 만든
`assemble/m4-baseline.json`(`--candidate`)뿐이고, 그 JSON의
`producers[job].payload_hashes`/`payload_manifest_sha256`은
`_evaluate_producer`의 `OK` 분기가 반환한 값을 `_build_baseline`이 그대로
옮겨 적은 것이다(§8.2-b, `assemble()`은 `producers[job] =
_evaluate_producer(...)`만 호출하고 그 반환 dict를 가공하지 않는다).
`_evaluate_producer`의 `OK` 분기는 `_verify_payloads`가 이미 (a)
entry exact-key/type/range/allowlist, (b) raw filename list==unique set
길이(duplicate fail-closed), (c) 실제 파일 hash/size/semantic 재검증을
모두 통과시킨 뒤에 반환한 canonical mapping이 아니면 절대 도달하지 않는
분기다 — 즉 `check_m4_baseline.py`가 관찰하는 `payload_hashes`는 항상
"assembler가 이 실행에서 독립적으로 검증한 exact identity"이고, 원본
receipt의 duplicate/unknown-key/schema/job 위반이 조금이라도 있으면
그 job은 `producers[job].status != "OK"`가 되어 애초에 `payload_hashes`
필드 자체가 candidate에 실리지 않는다(`PRODUCER_STATUS_SCHEMA`의 실패
variant들, 아래 참조). 따라서 이 checker의 §9.2 로직 자체는
DR-I5-MAJ-01로 인해 변경되지 않았다 — 이 절이 이미 전제하던 "assembler
output과 baseline copy의 결합"이 이제 원본 receipt 단계에서부터 exact함이
증명됐을 뿐이다(§20 참조).

```python
GATE_ENUM = frozenset({"NOT_RUN", "SKIPPED", "UNKNOWN", "BLOCKED", "PASS", "FAIL"})
REQUIRED_TOP_KEYS = frozenset({
    "schema", "schema_version", "generated_at", "git_sha", "workflow_run",
    "m3_fingerprint_reference", "dependency_snapshot_sha256", "settings_hash",
    "image_digest", "m43_deterministic_receipt_sha256", "producers", "gates",
    "deterministic_status", "operational_status", "M4.1_BLOCKED",
    "overall_release_ready",
})
REQUIRED_GATE_KEYS = frozenset({
    "python_tests", "frontend_tests", "container", "m43_deterministic",
    "m3_live_regression", "m41_operational",
})
DETERMINISTIC_GATE_KEYS = frozenset({
    "python_tests", "frontend_tests", "container", "m43_deterministic",
})
# producers dict의 key(§8.2 REQUIRED_PRODUCERS)와 gates dict의 key는
# 이름 규칙이 다르다(하이픈 vs 언더스코어) — 이 매핑이 유일한 변환
# 지점이다.
PRODUCER_TO_GATE_KEY = {
    "python-tests": "python_tests", "frontend-tests": "frontend_tests",
    "container": "container", "m43-deterministic": "m43_deterministic",
}
REQUIRED_PRODUCER_KEYS = frozenset(PRODUCER_TO_GATE_KEY)
# assemble_m4_evidence.py::_evaluate_producer(§8.2-b)가 낼 수 있는 전체 status enum.
PRODUCER_STATUS_ENUM = frozenset({
    "OK", "MISSING", "FAILED_OR_SKIPPED", "DUPLICATE_PRODUCER",
    "IDENTITY_MISMATCH", "PATH_TRAVERSAL", "MALFORMED", "PAYLOAD_INVALID",
})
# 신규(DR-I3-MAJ-04): §8.2-b `_evaluate_producer`가 실제로 만드는 tagged
# union — status마다 요구되는 key 집합이 다르다. "status" 하나만 있는
# 최소 위조가 여기서 걸린다.
PRODUCER_STATUS_SCHEMA = {
    "OK": frozenset({"status", "receipt_sha256", "payload_hashes", "payload_manifest_sha256"}),
    "MISSING": frozenset({"status"}),
    "FAILED_OR_SKIPPED": frozenset({"status", "needs_result"}),
    "DUPLICATE_PRODUCER": frozenset({"status", "count"}),
    "IDENTITY_MISMATCH": frozenset({"status", "reason"}),
    "PATH_TRAVERSAL": frozenset({"status"}),
    "MALFORMED": frozenset({"status"}),
    "PAYLOAD_INVALID": frozenset({"status", "reason"}),
}
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
# job이 §8.2 REQUIRED_PAYLOADS에서 선언하는 **정확한 filename 집합**을
# 참조한다(DR-I4-MAJ-02: count만으로는 same-count substitution/
# cross-job swap을 잡지 못한다 — §8.2-b `REQUIRED_PAYLOADS`의 key
# 집합과 review-pinned로 동기화된 리터럴이며, checker가 payload 원본을
# 다시 파싱하지는 않는다는 원래 책임 경계는 그대로 유지한다).
PRODUCER_EXPECTED_PAYLOAD_FILENAMES = {
    "python-tests": frozenset(), "frontend-tests": frozenset(),
    "container": frozenset({"layer_scan.json", "container_smoke.json"}),
    "m43-deterministic": frozenset({"m43.json", "m43-negative.json"}),
}


def _payload_manifest_sha256(payload_hashes: dict[str, str]) -> str:
    """§8.2-a/§8.2-b와 정확히 같은 2줄 정의 — 세 스크립트가 독립적으로
    이 계산을 수행해야 assembler output과 baseline copy의 identity
    결합이 의미를 갖는다(DR-I4-MAJ-02, §8.2-a 주석 참조)."""
    return hashlib.sha256(canonical_json_bytes(payload_hashes)).hexdigest()

def check(candidate: dict, *, expect_operational_blocked: bool) -> tuple[bool, list[str]]:
    issues = []
    if not isinstance(candidate, dict):
        return False, ["candidate_not_object"]
    top_keys = set(candidate)
    if top_keys != REQUIRED_TOP_KEYS:
        issues.append(f"top_level_key_mismatch:missing={sorted(REQUIRED_TOP_KEYS - top_keys)}"
                       f",extra={sorted(top_keys - REQUIRED_TOP_KEYS)}")
        return False, issues   # 최상위 스키마가 틀리면 이하 필드 접근 자체가 안전하지 않다

    gates = candidate["gates"]
    if not isinstance(gates, dict) or set(gates) != REQUIRED_GATE_KEYS:
        issues.append("gate_key_set_mismatch")
        return False, issues
    for name, value in gates.items():
        if not isinstance(value, str) or value not in GATE_ENUM:
            issues.append(f"unknown_gate_enum:{name}={value!r}")
    if issues:
        return False, issues

    # DR-I2-MAJ-04: producers도 exact-key/enum으로 독립 검사한다 —
    # producers={} 또는 전부 MISSING이어도 여기서 즉시 걸린다.
    producers = candidate["producers"]
    if not isinstance(producers, dict) or set(producers) != REQUIRED_PRODUCER_KEYS:
        issues.append("producer_key_set_mismatch")
        return False, issues
    expected_gates_from_producers = {}
    for job, gate_key in PRODUCER_TO_GATE_KEY.items():
        entry = producers[job]
        if not isinstance(entry, dict) or "status" not in entry:
            issues.append(f"producer_schema_invalid:{job}")
            continue
        status = entry["status"]
        if status not in PRODUCER_STATUS_ENUM:
            issues.append(f"producer_status_unknown:{job}={status!r}")
            continue
        # DR-I3-MAJ-04: status별 tagged union의 key 집합을 exact로 검사한다
        # — {"status":"OK"} 같은 최소 위조, success 필드 누락/과다,
        # failure variant에 success-only 필드가 섞인 candidate가 모두
        # 여기서 걸린다.
        if set(entry) != PRODUCER_STATUS_SCHEMA[status]:
            issues.append(f"producer_variant_schema_mismatch:{job}:status={status}:"
                           f"keys={sorted(entry)}")
            continue
        if status == "OK":
            receipt_sha = entry["receipt_sha256"]
            if not isinstance(receipt_sha, str) or not _HEX64_RE.fullmatch(receipt_sha):
                issues.append(f"producer_receipt_sha256_malformed:{job}")
                continue
            payload_hashes = entry["payload_hashes"]
            expected_filenames = PRODUCER_EXPECTED_PAYLOAD_FILENAMES[job]
            # DR-I4-MAJ-02: count가 아니라 **exact filename set** 비교다 —
            # same-count filename substitution({"a.json","b.json"}처럼
            # 이름만 다른 2개), extra+omission 상쇄(하나 빼고 하나 더한
            # 같은 개수), cross-job filename 교환(container에 m43.json/
            # m43-negative.json이 들어옴)이 모두 여기서 걸린다 — 세
            # 케이스 모두 "집합이 다르다"는 하나의 검사로 통일된다.
            if not isinstance(payload_hashes, dict) or set(payload_hashes) != expected_filenames:
                issues.append(f"producer_payload_filename_set_mismatch:{job}")
                continue
            if any(not isinstance(k, str) or not isinstance(v, str) or not _HEX64_RE.fullmatch(v)
                   for k, v in payload_hashes.items()):
                issues.append(f"producer_payload_hashes_malformed:{job}")
                continue
            # DR-I4-MAJ-02: baseline copy(candidate가 실어온
            # payload_manifest_sha256)를 assembler output(candidate가 실어온
            # payload_hashes)에서 독립적으로 재계산한 값과 결합해 확인한다
            # — filename 집합이 같아도 hash 값 자체가 payload_hashes와
            # payload_manifest_sha256 사이에서 서로 다르게 조작되면 여기서
            # 걸린다.
            manifest_sha = entry["payload_manifest_sha256"]
            if not isinstance(manifest_sha, str) or not _HEX64_RE.fullmatch(manifest_sha):
                issues.append(f"producer_payload_manifest_sha256_malformed:{job}")
                continue
            if manifest_sha != _payload_manifest_sha256(payload_hashes):
                issues.append(f"producer_payload_manifest_sha256_mismatch:{job}")
                continue
        expected_gates_from_producers[gate_key] = "PASS" if status == "OK" else "FAIL"
    if issues:
        return False, issues

    # gate 값을 producer status에서 재계산한 뒤에야 candidate가 self-report한
    # gates와 대조한다 — "producer=MISSING인데 gate=PASS"가 여기서 걸린다.
    for gate_key, expected in expected_gates_from_producers.items():
        if gates.get(gate_key) != expected:
            issues.append(f"gate_producer_algebra_mismatch:{gate_key}:"
                           f"producers_imply={expected},gates_say={gates.get(gate_key)!r}")
    if issues:
        return False, issues

    # deterministic_status는 이제 candidate의 gates가 아니라 producer에서
    # 재계산한 expected_gates_from_producers **로컬 변수**에서만 도출한다
    # (DR-I2-MAJ-04 "오직 재계산된 지역 값에서 도출") — 위 algebra 비교를
    # 통과했으므로 이 값은 candidate.gates와도 이미 일치하지만, 참조
    # 대상 자체를 producer 파생 값으로 고정해 향후 이 함수가 실수로
    # candidate 자기 보고를 다시 신뢰하는 회귀를 구조적으로 막는다.
    expected_deterministic = "PASS" if all(
        expected_gates_from_producers[gate_key] == "PASS"
        for gate_key in DETERMINISTIC_GATE_KEYS) else "FAIL"
    if candidate.get("deterministic_status") != expected_deterministic:
        issues.append("deterministic_status_algebra_mismatch")

    expected_operational = "PASS" if (gates["m41_operational"] == "PASS" and
                                       gates["m3_live_regression"] == "PASS") else "BLOCKED"
    if candidate.get("operational_status") != expected_operational:
        issues.append("operational_status_algebra_mismatch")

    expected_ready = (expected_deterministic == "PASS" and
                       expected_operational == "PASS")
    if candidate.get("overall_release_ready") != expected_ready:
        issues.append("overall_release_ready_algebra_mismatch")

    for bool_key in ("M4.1_BLOCKED", "overall_release_ready"):
        if not isinstance(candidate.get(bool_key), bool):
            issues.append(f"non_boolean_field:{bool_key}={candidate.get(bool_key)!r}")

    if expect_operational_blocked:
        if candidate.get("operational_status") != "BLOCKED" or \
           candidate.get("M4.1_BLOCKED") is not True or \
           candidate.get("overall_release_ready") is not False:
            issues.append("expected_operational_blocked_not_satisfied")
    return (not issues, issues)
```
모든 비교는 `!=`/`is not True`/`is not False`/`isinstance`의
**명시적** 형태다 — `if not candidate.get("overall_release_ready"):`
같은 truthy 검사는 어디에도 쓰지 않는다(REQ-008.3 "truthy/누락/default
PASS로 처리하지 않는다"를 코드 스타일 규칙으로 못박음 — 누락된 키는
최상위 exact-key 검사에서 이미 걸러진다).

- **exact-key 스키마가 먼저다**: `top_keys != REQUIRED_TOP_KEYS`이면
  이후 어떤 필드도 읽지 않고 즉시 `False`를 반환한다 — 필드 누락으로
  인한 `KeyError`가 검사기 자신을 죽여 "예외 없이 종료 = 통과"로
  오인되는 경로를 없앤다. `gates`/`producers` 키 집합도 동일하게
  exact-match다 — `python_tests`가 빠지면(리뷰가 지적한 정확한 예시)
  `gate_key_set_mismatch`로, `producers`가 `{}`이면
  `producer_key_set_mismatch`로 즉시 실패한다.
- **재계산이 자기 보고를 두 단계로 대체(DR-I2-MAJ-04)**: (1) `gates`
  값 자체를 candidate의 `producers[job].status`에서 다시 계산해
  candidate의 `gates`와 비교하고, (2) `deterministic_status`/
  `operational_status`/`overall_release_ready`는 그 **producer 파생
  로컬 값**(candidate의 `gates`가 아니라)에서만 도출해 candidate의
  자기 보고와 비교한다 — `producers={}` 또는 모든 producer가
  `MISSING`이면서 `gates`만 `"PASS"`로 조작된 candidate는 (1) 단계의
  `gate_producer_algebra_mismatch`에서 이미 걸리므로 (2) 단계에
  도달하지 못한다. assembler가 올바르게 계산했더라도 checker가
  producer 원본부터 독립적으로 같은 결론에 도달해야만 통과한다
  (REQ-008 "hosted assembler만 신뢰하면 안 된다"는 리뷰 원문 취지를
  구조적으로 반영).
- **producer variant tagged-union schema(DR-I3-MAJ-04 신규)**: (1)/(2)
  단계에 도달하기 **전에** 각 producer entry의 key 집합을
  `PRODUCER_STATUS_SCHEMA[status]`와 exact 비교한다 — 리뷰가 지적한
  정확한 예시("네 entry를 모두 `{"status":"OK"}`로 만든 candidate")는
  `OK` variant가 `receipt_sha256`/`payload_hashes`/`payload_manifest_sha256`을
  요구하므로 `producer_variant_schema_mismatch`에서 즉시 걸리고,
  `gate_producer_algebra_mismatch` 단계까지 가지도 못한다. `OK` entry는
  추가로 `receipt_sha256`이 64-hex 문자열인지, `payload_hashes`가 그
  job의 review-pinned exact filename set(§8.2
  `PRODUCER_EXPECTED_PAYLOAD_FILENAMES`, `REQUIRED_PAYLOADS`와 동기화)과
  정확히 같은 key 집합을 가진 64-hex 값 dict인지도 검사한다 —
  "receipt hash, `needs_result`, payload hashes/reasons가 전혀 없어도
  통과"하던 이전 결함을 구조적으로 닫는다.
- **payload identity: exact filename set + manifest hash 결합(DR-I4-MAJ-02
  신규)**: `payload_hashes`의 **개수**만 확인하던 이전 검사는 `container`에
  `{"a.json": "<64hex>", "b.json": "<64hex>"}`(다른 이름, 같은 개수)를
  넣거나 `m43-deterministic`에 `container`의 filename을 그대로 옮겨
  적어도 통과시켰다. 수정된 검사는 (1) `set(payload_hashes) ==
  PRODUCER_EXPECTED_PAYLOAD_FILENAMES[job]`로 filename 집합 자체를
  exact 비교하고, (2) `OK` entry의 `payload_manifest_sha256`이 64-hex이며
  **그 `payload_hashes`에서 checker가 직접 재계산한 canonical hash와
  정확히 같은지**(`_payload_manifest_sha256(payload_hashes)`)를
  추가로 요구한다. (1)은 same-count filename substitution/extra+omission
  상쇄/cross-job filename 교환을, (2)는 `payload_hashes`와
  `payload_manifest_sha256`이 서로 다른 시점·다른 소스에서 각각
  독립적으로 조작돼 서로 어긋나는 경우를 닫는다 — assembler가 이
  실행에서 실제로 재검증한 identity("assembler output", §8.2-b)와
  candidate에 실려온 identity("baseline copy")가 checker 시점에도 다시
  한 번 결합된다.
- `test_check_m4_baseline.py`는 (a) 각 필수 top-level/gate/producer 키의
  개별 누락, (b) 예상치 못한 extra key 추가, (c) `M4.1_BLOCKED`/
  `overall_release_ready`에 `"true"`(문자열)/`1`/`null` 대입, (d)
  `deterministic_status="PASS"`인데 `gates.container="FAIL"`인 모순
  candidate, (e) `producers={}`(빈 dict)이면서 `gates` 네 필드가 모두
  `"PASS"`인 candidate(DR-I2-MAJ-04 원문 예시) → `producer_key_set_mismatch`,
  (f) 모든 producer가 `status="MISSING"`이면서 `gates`가 전부 `"PASS"`인
  candidate → 4개의 `gate_producer_algebra_mismatch`, (g) 한 producer만
  `status="OK"`이고 나머지는 `MISSING`인데 `gates`가 전부 `"PASS"`인
  candidate(부분 위조) → 나머지 3개 gate에서 `gate_producer_algebra_mismatch`,
  (i) 네 producer 모두 `{"status": "OK"}`만 있는 candidate(DR-I3-MAJ-04
  원문 예시, `receipt_sha256`/`payload_hashes`/`payload_manifest_sha256`
  누락) → `producer_variant_schema_mismatch`(4건), (j) `OK` entry에
  `receipt_sha256`을 32-hex(잘못된 길이)로 대입 →
  `producer_receipt_sha256_malformed`, (k) `container`의 `payload_hashes`가
  `{"layer_scan.json": "<64hex>"}` 1개 항목만 있음(2개
  filename이 모두 필요) → `producer_payload_filename_set_mismatch`,
  (l) `MISSING` entry에 success-only 필드 `receipt_sha256`이 섞여
  들어감 → `producer_variant_schema_mismatch`(failure variant에
  success 필드 혼입), (n, DR-I4-MAJ-02 신규) `container`의
  `payload_hashes`를 `{"foo.json": "<64hex>", "bar.json": "<64hex>"}`로
  대체(개수는 2로 동일, filename만 다름 — same-count substitution) →
  `producer_payload_filename_set_mismatch`, (o, DR-I4-MAJ-02 신규)
  `container`의 `payload_hashes`에서 `layer_scan.json`을 빼고
  `extra.json`을 추가(개수는 여전히 2 — omission과 extra가 상쇄) →
  `producer_payload_filename_set_mismatch`, (p, DR-I4-MAJ-02 신규)
  `m43-deterministic`의 `payload_hashes`를 `container`의 filename 집합
  (`{"layer_scan.json", "container_smoke.json"}`)으로 교체(cross-job
  filename swap) → `producer_payload_filename_set_mismatch`, (q,
  DR-I4-MAJ-02 신규) `OK` entry의 `payload_manifest_sha256`을 32-hex
  (잘못된 길이)로 대입 → `producer_payload_manifest_sha256_malformed`,
  (r, DR-I4-MAJ-02 신규) `payload_hashes`는 정상 filename 집합·형식을
  유지한 채 그 안의 한 값(64-hex)만 다른 유효한 64-hex로 바꾸고
  `payload_manifest_sha256`은 이전 값 그대로 남겨 둠(재계산 없이
  `payload_hashes`만 사후 변조) → `producer_payload_manifest_sha256_mismatch`,
  (m) 정상 candidate 1개를 포함해 최소 24개 이상의
  파라미터화 케이스로 이 알고리즘을 검증한다(MIN-09 "모든 key별
  omission, extra key, wrong type, null, truthy string negative cases",
  DR-I2-MAJ-04 "producer omission/extra/status mismatch 및
  producer=MISSING + gate=PASS", DR-I3-MAJ-04 "tagged union exact
  schema + receipt/payload identity metadata", DR-I4-MAJ-02 "exact
  payload filename set + assembler output/baseline copy manifest hash
  결합"을 문자 그대로 구현).

exit code: 문제 없으면 0, `issues`가 있으면 1(`issues` 목록을 stderr에
JSON으로 출력).

## 10. `run_m43_acceptance.py`

`run_m42_acceptance.py`의 `PROFILE_NODE_IDS`/`collect_profile_nodes`
관례(§0에서 확인)를 그대로 잇는다.

```python
PROFILE_NODE_IDS = MappingProxyType({
    "manifest_canonical": ("tests/unit/test_index_manifest.py::test_canonical_round_trip_100x",),
    "manifest_negative": ("tests/unit/test_index_manifest.py::test_schema_and_hash_rejection_matrix",),
    "verification_trust": ("tests/unit/test_index_verification.py::test_symlink_owner_mode_toctou_matrix",
                            "tests/unit/test_index_verification.py::test_current_pointer_trust_matrix"),
    "verification_reopen_race": (  # 신규(DR-I1-CRIT-01)
        "tests/unit/test_index_verification.py::test_verify_then_load_uses_captured_bytes_no_reopen",
        "tests/unit/test_index_verification.py::test_racer_between_member_opens_has_no_effect",
    ),
    "legacy_baseline_pin": (  # 신규(DR-I1-MAJ-03), 재작성(DR-I2-MIN-06)
        "tests/unit/test_pinned_baseline_provenance.py::test_pinned_constants_match_tracked_m3_baseline_bytes",
        "tests/unit/test_index_lifecycle.py::test_import_legacy_rejects_source_hash_mismatch",
    ),
    "staging_fault": ("tests/integration/test_index_lifecycle_fault_injection.py::test_staging_fault_matrix_preserves_current",),
    "activation_rollback": ("tests/integration/test_index_lifecycle_fault_injection.py::test_activate_rollback_100x",),
    "crash_recovery_journal": (  # 신규(DR-I1-MAJ-05), 확장(DR-I2-MAJ-01)
        "tests/integration/test_index_lifecycle_fault_injection.py::test_crash_recovery_journal_reconciles_to_consistent_state",
        "tests/integration/test_index_lifecycle_fault_injection.py::test_crash_recovery_history_and_receipt_exact_once_matrix",
    ),
    "lock_untrusted_symlink": (  # 신규(DR-I1-MAJ-05)
        "tests/integration/test_index_lifecycle_fault_injection.py::test_preexisting_lock_symlink_rejected",
    ),
    "legacy_import": ("tests/unit/test_index_lifecycle.py::test_legacy_import_hash_and_byte_preservation",),
    "retention": ("tests/unit/test_index_lifecycle.py::test_cleanup_dry_run_then_apply_protects_current_and_previous",
                  "tests/unit/test_index_lifecycle.py::test_cleanup_staging_protects_unexpected_and_young_entries"),  # 신규(DR-I1-MAJ-04)
    "lock_contention": ("tests/integration/test_index_lifecycle_fault_injection.py::test_concurrent_lock_contention_bounded",),
    "layer_scanner": ("tests/unit/test_scan_image_layers.py::test_positive_negative_traversal_whiteout_fixtures",),
    "container_static_and_connectivity": (  # 신규(DR-I1-MAJ-06/07), 확장(DR-I2-MAJ-02/DR-I3-MAJ-02)
        "tests/unit/test_container_smoke_contract.py::test_docker_run_argv_includes_add_host_and_embedding_seam_env",
        "tests/unit/test_container_smoke_contract.py::test_reachability_probe_argv_targets_mock_ping_via_host_gateway",
        "tests/unit/test_container_smoke_contract.py::test_negative_activation_argv_omits_test_seam_mount_and_pythonpath",
    ),
    "embedding_provider_seam_guard": (  # 신규(DR-I2-MAJ-02), 확장(DR-I3-MAJ-02)
        "tests/unit/test_settings_inventory.py::test_deterministic_embedding_provider_without_allow_flag_rejected",
        "tests/unit/test_rag_engine_embeddings.py::test_build_embeddings_default_uses_huggingface_provider",
        "tests/unit/test_rag_engine_embeddings.py::test_build_embeddings_raises_seam_unavailable_when_module_absent",
    ),
    "assemble_payload_verification": (  # 신규(DR-I1-MAJ-08), 확장(DR-I2-MAJ-03/DR-I3-MAJ-03)
        "tests/unit/test_assemble_m4_evidence.py::test_payload_hash_size_and_semantic_negative_matrix",
        "tests/unit/test_assemble_m4_evidence.py::test_m43_typed_payload_negative_matrix",
        "tests/unit/test_assemble_m4_evidence.py::test_expected_node_ids_matches_producer_profile_node_ids",
    ),
    "baseline_strict_schema": (  # 신규(DR-I1-MIN-09), 확장(DR-I2-MAJ-04)
        "tests/unit/test_check_m4_baseline.py::test_strict_schema_and_algebra_recompute_matrix",
    ),
})
```
`main()`은 `--profile {deterministic,live}`(`live`는 m42와 동일하게
`{"status": "NOT_RUN", ..., "M4.1_BLOCKED": True}`만 쓰고 exit 0),
`--repeat`(기본 10), `--seed`(기본 4303), `--output`,
`--inject-evidence-mismatch`(help=SUPPRESS)를 갖는다.

#### 10.1 `m43.json`/`m43-negative.json` acceptance receipt 스키마 (DR-I2-MAJ-03 신규)

Iteration 2 MAJ-03이 지적한 근본 원인은 이 출력 스키마 자체가 이전
개정에 없었다는 것이다 — assembler(§8.2)가 검증할 "상세 semantic
필드"가 `status` 하나뿐이었던 이유는 `run_m43_acceptance.py`가
그 외 어떤 필드도 약속하지 않았기 때문이다. 아래는 `--profile
deterministic` 실행(정상/negative 두 모드 공통)이 반드시 채우는
필드다.

```json
{
  "schema": "m43-acceptance-receipt-v1",
  "profile": "deterministic",
  "seed": 4303,
  "repeat": 10,
  "command": "run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303",
  "started_at": "2026-08-12T00:00:00Z",
  "finished_at": "2026-08-12T00:00:05Z",
  "nodes": {
    "manifest_canonical": {"repeat": 10, "success_count": 10, "status": "PASS"},
    "...": "PROFILE_NODE_IDS의 모든 키가 여기 정확히 1개씩 대응"
  },
  "negative_control": {
    "executed": false, "expected_to_fail": null,
    "actual_exit_code": null, "result": null
  },
  "status": "PASS"
}
```
- `nodes`의 key 집합은 실행 시점 `PROFILE_NODE_IDS`의 key 집합과
  **정확히 같아야** 한다(exact set — 초과/누락 모두 스크립트 자체
  버그로 간주해 `main()`이 `RuntimeError`로 즉시 중단, receipt 미작성).
  각 node 값은 그 node에 대응하는 test id들을 `--repeat`번 반복
  실행한 성공 횟수(`success_count`)이며, `success_count == repeat`일
  때만 `status: "PASS"`, 그 외 `"FAIL"`이다(node 하나라도 100%
  미만이면 이미 node 자체가 FAIL로 기록되므로, top-level `status`도
  아래 규칙에 따라 자동으로 `FAIL`이 된다).
- top-level `status`는 (a) 모든 node가 `PASS`이고 (b)
  `negative_control.executed is False`일 때만(즉 `--inject-evidence-mismatch`가
  **없는** 정상 실행) `"PASS"`다. `--inject-evidence-mismatch` 모드에서는
  `negative_control.executed = True`이고, 변조된 identity가 올바르게
  거부되면(§10 본문의 `_check_identity` 재현) `negative_control.result =
  "REJECTED_AS_EXPECTED"`, top-level `status = "REJECTED_AS_EXPECTED"`,
  프로세스 exit code = 1(Plan §8 "negative command는 exit 1이 기대
  성공"과 일치). 반대로 변조가 잘못 통과되면(회귀) `negative_control.result
  = "TAMPERING_ACCEPTED_BUG"`, top-level `status =
  "TAMPERING_ACCEPTED_BUG"`, exit code = 0 — CI YAML(§8.1)의 `if [ "$code"
  -ne 1 ]; then ... exit 1; fi` 검사가 이 exit code 0을 job 실패로
  승격한다(CI YAML 자체는 변경하지 않고 §10 스크립트 계약만 §10.1로
  구체화한 것).
- `m43-negative.json`(negative control 모드의 출력)은 `negative_control.executed
  = true`, `negative_control.expected_to_fail = true`,
  `negative_control.actual_exit_code = 1`, `negative_control.result =
  "REJECTED_AS_EXPECTED"`, top-level `status = "REJECTED_AS_EXPECTED"`를
  갖는다 — 이것이 §8.2 typed parser가 `m43-negative.json`에 요구하는
  "성공"의 정확한 형태다(정상 통과 = 변조가 거부됨).
- `tests/unit/test_m43_acceptance_runner.py`가 이 스키마를 채우는
  `main()`의 두 모드(정상/negative) 각각에 대해 필드 존재/타입/값
  matrix를 검증한다(§13 회귀 대상에 반영).

`collect_profile_nodes`는 `run_m42_acceptance.py`와 동일한 로직(15줄,
`pytest --collect-only -q`로 실제 존재/이름 일치를 재확인)을 이 스크립트
안에 복제한다 — 기존 코드베이스가 `_git_commit`/`_git_dirty`를
`fingerprint.py`/`reporting.py` 양쪽에 두는 것과 같은 "작은 CLI 유틸은
스크립트 간 import 결합보다 복제를 우선한다"는 기존 선례를 따른다.

`--inject-evidence-mismatch` 동작: 실제 `assemble_m4_evidence.py`의
`_check_identity`를 import해(스크립트↔스크립트 import는 `scripts/`가
하나의 패키지가 아니므로 `importlib.util.spec_from_file_location`으로
로드하거나, 더 간단히 `sys.path.insert(0, "scripts")` 후 `import
assemble_m4_evidence`로 직접 import — `test_orchestration_watchdog.py`가
이미 `sys.path.insert(0, ".../scripts"); import orchestration_watchdog`
패턴을 쓰는 선례를 그대로 재사용) 정상 producer receipt 하나의 `sha`
필드를 변조한 뒤 `_check_identity`가 이를 거부하는지 확인한다. 이
검사가 실패(즉 변조된 receipt가 통과)하면 `status="FAIL"`,
`diagnostic="tampered_identity_accepted"`로 기록하고, 정상 거부되면
그것이 **기대한 성공**이므로 `run_m43_acceptance.py` 프로세스 자체는
`exit 1`을 반환한다(Plan §8 "negative command는 exit 1과 retained
failure evidence가 기대 성공"과 동일하게, negative-control 모드에서는
"올바르게 거부됨"이 스크립트의 exit 1로 인코딩된다 — `run_m42_acceptance.py`의
`--inject-conservation-mismatch` 모드가 이미 이 관례를 확립했다).

`_receipt_is_complete`류 완전성 검사도 m42와 동일 원칙(모든
`PROFILE_NODE_IDS` 항목이 `--repeat`번 전부 성공해야 `PASS`)으로
재사용한다.

## 11. `scripts/orchestration_watchdog.py` readiness fix — 테스트와 commit scope (REQ-009.2/3/4)

### 11.1 현재 상태 재확인과 bounded-reason classifier 추가 (DR-I2-MAJ-05로 범위 수정, DR-I3-MAJ-05로 run_loop 제어 흐름까지 확장)

`git diff scripts/orchestration_watchdog.py`(§0.2에 전문 인용)의
terminal-scoping 부분(task-list/check를 `--from`/`--terminal`로
coordinator identity에 bind)은 이미 correct하고 self-contained하다 —
이 부분은 **추가 코드 변경을 요구하지 않는다**.

Iteration 2 MAJ-05는 이 주장이 놓친 별도 지점을 지적했다: 현재
`run_json`(L22-26)이 던지는 `RuntimeError`의 메시지는 `f"command failed
({proc.returncode}): {' '.join(command)}: {proc.stderr.strip()}"` —
**전체 커맨드라인과 원본 CLI stderr를 그대로 문자열에 담는다**.
`main()`의 except 블록(L128-130)과 `run_loop`의 except 블록(L98-99)은
이 `str(exc)`를 각각 stdout(`error` 필드)과 journal(`error` 필드)에
**가공 없이** 그대로 쓴다 — Orca가 `consumer_fenced`처럼 구조화된
거부를 stderr에 실어 보내도, 그 실패가 "고정 vocabulary 안의 bounded
reason"인지 "예측하지 못한 임의 CLI 출력"인지 호출자가 구분할 방법이
없다(§0.4-4 "receipt는 고정 vocabulary만 포함"이 lifecycle receipt에는
이미 적용됐지만 watchdog 실패 경로에는 아직 없었다).

**Iteration 3 MAJ-05**: bounded-reason classifier 자체는 정확했지만,
`run_loop`(L94-101)은 그 classifier가 반환한 reason이
`consumer_fenced`이든 다른 무엇이든 **똑같이** journal에 기록하고
`time.sleep(interval)` 뒤 다음 반복에서 `check_once`를 다시 호출했다.
`consumer_fenced`는 "이 프로세스가 이 run의 coordinator terminal
소유권을 이미 잃었다"는 뜻이므로, 같은 권한 없이 계속 재시도하는 것은
안전한 복구가 아니라 거부된 권한으로 API를 계속 두드리는 것이다 —
매 interval마다 동일 journal 행이 계속 append되어 저장량도 bounded하지
않고, 실행 프로세스는 정상 종료(exit 0) 경로에 그대로 남는다. 수정된
설계는 `run_loop`의 예외 처리 분기 자체를 두 갈래로 나눈다:
`consumer_fenced`(terminal ownership loss)는 journal에 **단 한 번**
bounded reason을 기록한 뒤 `run_loop`을 즉시 nonzero로 종료하고,
그 외 generic transient CLI 실패만 기존과 같이 `interval`-paced 재시도
대상으로 남긴다. 이는 REQ-009.4가 보존을 요구하는 기존 terminal-scoping
delta(tracked base `e57fe1c` 대비)를 되돌리거나 의미를 바꾸지 않는
**순수 추가/분기 확장**이므로 같은 commit scope 안에서 함께 다룬다
(§11.3).

```python
# orchestration_watchdog.py에 추가(기존 코드 삭제/수정 없음, 순수 추가):
CONSUMER_FENCED_MARKER = "consumer_fenced"

def _classify_runner_error(exc: Exception) -> str:
    """원본 예외 문자열(전체 커맨드라인 + 원본 stderr 포함)에서 고정
    vocabulary 안의 reason만 추출한다. 알려진 marker가 없으면
    "cli_command_failed"라는 단일 generic bounded 값으로 접는다 —
    어느 경우든 원본 텍스트를 stdout/journal에 그대로 노출하지
    않는다."""
    if CONSUMER_FENCED_MARKER in str(exc):
        return CONSUMER_FENCED_MARKER
    return "cli_command_failed"
```
호출 지점 2곳만 `str(exc)` 대신 `_classify_runner_error(exc)`를 쓰도록
바꾼다:
- `main()`의 except 블록: `print(json.dumps({"ok": False, "error":
  _classify_runner_error(exc)}, ...), file=sys.stderr)`.
- `run_loop`의 except 블록(아래, DR-I3-MAJ-05로 제어 흐름 자체가
  분기): `state.append_journal(root, run_id, {"operation":
  "watchdog_check", "outcome": "failed", "reason":
  _classify_runner_error(exc)})`(journal 필드명을 `error`에서
  `reason`으로 바꿔 "고정 vocabulary 필드"라는 의도를 명확히 한다).

`run_loop`의 except 블록 자체(DR-I3-MAJ-05 신규 delta — 기존
`except Exception: state.append_journal(...)` 한 줄을 아래 분기로
교체, `while`/`check_once`/정상 종료 경로는 무변경):
```python
except Exception as exc:
    reason = _classify_runner_error(exc)
    state.append_journal(root, run_id, {"operation": "watchdog_check",
                                         "outcome": "failed", "reason": reason})
    if reason == CONSUMER_FENCED_MARKER:
        # terminal ownership loss — 이 프로세스는 더 이상 이 run을
        # 감시할 권한이 없다. journal에는 이미 단 한 번 기록했으므로
        # (위 append_journal 한 줄), 여기서 즉시 nonzero로 종료해
        # 두 번째 check_once 호출/재시도를 구조적으로 막는다. 재시작은
        # 항상 새 프로세스(운영자/coordinator의 명시적 rebind)로만
        # 일어난다 — 이 함수 자신이 스스로를 재기동하지 않는다.
        return 1
    # generic transient failure만 기존과 동일하게 interval 뒤 재시도.
```
`_classify_runner_error`가 반환하는 값은 `CONSUMER_FENCED_MARKER` 또는
`"cli_command_failed"` 둘뿐이므로, 이 분기는 "terminal ownership
loss"와 "그 외 모든 generic transient CLI 실패"를 정확히 이분한다 —
`run_json`이 던질 수 있는 어떤 새로운 실패 형태가 추가돼도
`_classify_runner_error`가 알려진 marker를 못 찾으면 항상
`"cli_command_failed"`(재시도 대상)로 접히므로, 이 분기 자체가
깨지는 일은 없다.

부족했던 것은 이 classifier와 분기 외에는 오직 테스트 증거다
(REQ-009.3 "현재 Orca `--help`가 해당 문법을 지원하는 것만으로 PASS하지
않는다... exact argv, ... read-only peek, ... fail-closed wake 기록을
검증").

### 11.2 `tests/unit/test_orchestration_watchdog.py` 추가 테스트 8종 (문서 표현 정합성 수정)

이전 iteration의 문서는 이 섹션 제목에 "6종"이라고 썼지만 실제로는
7개 번호 항목을 나열했고, 그 중 4번·5번 항목은 각각 테스트 함수를
2개씩 포함해 실제 함수 개수는 8개였다(리뷰 "확인된 강점과 보존 조건"
항목이 이 불일치를 지적). 아래는 8개 항목 각각에 테스트 함수 정확히
1개씩만 두도록 번호를 다시 매긴 버전이다 — 내용은 이전 버전과 동일하고
번호와 개수 표현만 정정했다.

기존 9개 테스트는 그대로 둔다(REQ-009.4 "기존 delta를 되돌리거나 의미를
바꾸지 않는다" — 새 테스트는 추가만).

**1) `test_task_list_uses_exact_bound_argv`**
```python
def test_task_list_uses_exact_bound_argv(tmp_path):
    setup_state(tmp_path)
    runner = FakeRunner(tasks=[{"status": "dispatched"}])
    watchdog.check_once(tmp_path, "run_watch123", "orca", runner=runner)
    task_list_cmd = next(c for c in runner.commands if "task-list" in c)
    assert task_list_cmd == [
        "orca", "orchestration", "task-list",
        "--run", "run_watch123", "--from", "term_coord",
        "--brief", "--json",
    ]
```
(`term_coord`는 `setup_state`가 `state.init_state(tmp_path, "run_watch123",
"m4.1", "term_coord", "runtime_a", 180)`로 이미 고정한 값 — 기존
`setup_state` 헬퍼를 그대로 재사용, 신규 fixture 불필요.)

**2) `test_check_uses_exact_bound_argv`**
```python
def test_check_uses_exact_bound_argv(tmp_path):
    setup_state(tmp_path)
    runner = FakeRunner(tasks=[{"status": "dispatched"}])
    watchdog.check_once(tmp_path, "run_watch123", "orca", runner=runner)
    check_cmd = next(c for c in runner.commands if c[2] == "check")
    assert check_cmd == [
        "orca", "orchestration", "check",
        "--terminal", "term_coord", "--run", "run_watch123",
        "--peek", "--json",
    ]
```

**3) `test_check_always_includes_peek_flag`**
```python
def test_check_always_includes_peek_flag(tmp_path):
    """--peek이 빠지면 조회가 소비형(consuming)으로 바뀐다 — 이 플래그가
    항상 존재함을 회귀 고정한다."""
    setup_state(tmp_path)
    runner = FakeRunner(tasks=[])
    watchdog.check_once(tmp_path, "run_watch123", "orca", runner=runner)
    check_cmd = next(c for c in runner.commands if c[2] == "check")
    assert "--peek" in check_cmd
```

**4) `FakeRunner` 확장 — terminal-scope 격리 검증** (테스트 함수는 아래
`ScopedFakeRunner` 정의 뒤 `test_check_is_terminal_scoped_and_ignores_foreign_messages`
1개)

`FakeRunner`에 다른 terminal 소유 메시지를 함께 시뮬레이션하는 옵션을
추가하고, `--terminal`로 넘어온 값이 그 메시지의 소유자와 다르면
`AssertionError`를 던지게 해 "다른 terminal/run의 message를 소비하지
않는" 계약을 실제로 검증한다:
```python
class ScopedFakeRunner:
    def __init__(self, *, owned_terminal, tasks=None, owned_messages=None,
                 foreign_messages=None, connected=True):
        self.owned_terminal = owned_terminal
        self.tasks = tasks or []
        self.owned_messages = owned_messages or []
        self.foreign_messages = foreign_messages or []  # 다른 terminal 소유
        self.connected = connected
        self.commands = []

    def __call__(self, command):
        self.commands.append(command)
        if "task-list" in command:
            requested = command[command.index("--from") + 1]
            assert requested == self.owned_terminal, "task-list leaked cross-terminal scope"
            return {"result": {"tasks": self.tasks}}
        if command[2] == "check":
            requested = command[command.index("--terminal") + 1]
            assert requested == self.owned_terminal, "check leaked cross-terminal scope"
            return {"result": {"messages": self.owned_messages}}  # foreign_messages는 절대 반환 안 함
        if command[:2] == ["orca", "terminal"] and command[2] == "show":
            return {"result": {"terminal": {"connected": self.connected}}}
        if command[:2] == ["orca", "terminal"] and command[2] == "send":
            return {"result": {"sent": True}}
        raise AssertionError(command)


def test_check_is_terminal_scoped_and_ignores_foreign_messages(tmp_path):
    setup_state(tmp_path)
    runner = ScopedFakeRunner(
        owned_terminal="term_coord", tasks=[{"status": "dispatched"}],
        owned_messages=[],
        foreign_messages=[{"id": "msg_other", "type": "worker_done"}],
    )
    result = watchdog.check_once(tmp_path, "run_watch123", "orca", runner=runner)
    assert result["woke"] is False   # foreign_messages는 조회 대상이 아니므로 영향 없음
```

**5) `test_check_propagates_consumer_fenced_fail_closed_with_no_send`**
(DR-I2-MAJ-05 수정: generic `RuntimeError("orca cli unavailable")`
대신, 실제 `run_json`이 stale/non-owner terminal에서 만들어낼 법한
현실적인 메시지 — 전체 커맨드라인 + 원본 CLI stderr 안에
`"consumer_fenced"` marker가 섞인 형태 — 를 재현한다. exact argv를
받은 뒤 실패하는 fake이므로 "exact argv"와 "fail-closed 전파"와
"no-send"(wake를 보내는 `orca terminal send`가 전혀 호출되지 않음)
세 가지를 한 테스트에서 검증한다.)
```python
def test_check_propagates_consumer_fenced_fail_closed_with_no_send(tmp_path):
    setup_state(tmp_path)
    calls = []
    def fenced_runner(command):
        calls.append(command)
        if "task-list" in command:
            assert command == [
                "orca", "orchestration", "task-list",
                "--run", "run_watch123", "--from", "term_coord",
                "--brief", "--json",
            ]
            raise RuntimeError(
                "command failed (2): orca orchestration task-list --run "
                "run_watch123 --from term_coord --brief --json: "
                "consumer_fenced: term_coord is not the active consumer for run run_watch123")
        raise AssertionError(command)  # check/terminal send가 호출되면 실패 —
                                        # task-list 실패 이후 어떤 후속
                                        # 커맨드도 실행되지 않아야 한다(fail-closed)
    import pytest
    with pytest.raises(RuntimeError, match="consumer_fenced"):
        watchdog.check_once(tmp_path, "run_watch123", "orca", runner=fenced_runner)
    assert not any(c[:2] == ["orca", "terminal"] and c[2] == "send" for c in calls)  # no-send
```

**6) `test_check_subcommand_exits_2_with_bounded_stderr_and_no_stdout_success`**
(subprocess 수준에서 `main()`이 같은 `consumer_fenced` 실패를 exit 2로
안정 변환하고, `_classify_runner_error`(§11.1)가 원본 커맨드라인/stderr를
`"consumer_fenced"` 하나의 bounded 토큰으로 접었는지, `stdout`에
`"ok": true`가 전혀 없는지를 모두 확인한다 — 리뷰가 지적한 "return
code만 assert한다"는 결함을 stdout/stderr 캡처로 닫는다.)
```python
def test_check_subcommand_exits_2_with_bounded_stderr_and_no_stdout_success(
        tmp_path, monkeypatch, capsys):
    setup_state(tmp_path)
    def fenced_run_json(command):
        raise RuntimeError(
            "command failed (2): orca orchestration check --terminal term_coord "
            "--run run_watch123 --peek --json: consumer_fenced: stale coordinator lease")
    monkeypatch.setattr(watchdog, "run_json", fenced_run_json)
    monkeypatch.chdir(tmp_path)
    code = watchdog.main(["--root", str(tmp_path), "--run-id", "run_watch123", "check"])
    assert code == 2
    captured = capsys.readouterr()
    assert '"ok": true' not in captured.out and '"ok":true' not in captured.out  # no-success
    assert '"error": "consumer_fenced"' in captured.err or \
           '"error":"consumer_fenced"' in captured.err  # bounded reason, not raw stderr
    assert "stale coordinator lease" not in captured.err  # 원본 CLI stderr 비노출
    assert "Traceback" not in captured.err  # 스택트레이스 없음
```

**7) `test_run_loop_terminates_nonzero_after_consumer_fenced_with_exact_once_journal`**
(DR-I3-MAJ-05 재작성 — 이전 버전은 fenced 뒤 loop가 **계속** 두 번째
`check_once`를 실행하는 것을 요구했다. ownership이 거부된 consumer가
계속 retry하는 것은 안전한 복구가 아니므로, 재작성된 계약은 정반대를
요구한다: fence 후 call count=1, `orca terminal send` 미호출, journal
exact-one, process nonzero, 재시작은 항상 새 프로세스(명시적 rebind)로만.)
```python
def test_run_loop_terminates_nonzero_after_consumer_fenced_with_exact_once_journal(
        tmp_path, monkeypatch):
    setup_state(tmp_path)
    calls = {"n": 0}
    def fenced_run_json(command):
        calls["n"] += 1
        if "task-list" in command:
            raise RuntimeError(
                "command failed (2): orca orchestration task-list --run run_watch123 "
                "--from term_coord --brief --json: consumer_fenced: stale lease")
        raise AssertionError(command)  # task-list 실패 뒤 어떤 후속 커맨드도
                                        # 실행되지 않아야 한다(check, terminal
                                        # send 포함) — no-send의 구조적 증거
    monkeypatch.setattr(watchdog, "run_json", fenced_run_json)
    monkeypatch.setattr(watchdog.time, "sleep", lambda _: None)
    exit_code = watchdog.run_loop(tmp_path, "run_watch123", "orca", interval=1)
    assert exit_code != 0   # fail-closed — 성공(exit 0) 경로에 남지 않는다
    assert calls["n"] == 1  # check_once가 두 번째로 호출되지 않음(재시도 없음)
    journal_text = (state.journal_path(tmp_path, "run_watch123")).read_text(encoding="utf-8")
    fenced_lines = [line for line in journal_text.splitlines()
                     if "consumer_fenced" in line and "watchdog_check" in line]
    assert len(fenced_lines) == 1   # journal exact-one — 매 interval 반복 누적이 아니다
    assert "stale lease" not in journal_text  # journal bounded reason — 원본 stderr 비노출
    assert not (state.state_dir(tmp_path, "run_watch123") / "watchdog.stop").exists()
```
`calls["n"] == 1`은 두 가지를 동시에 증명한다 — `run_loop`이
`check_once`를 두 번째로 호출하지 않는다는 것과(재시도 없음), 그 한
번의 `check_once` 내부에서도 `task-list` 실패 이후 `check`/`orca
terminal send` 어느 것도 호출되지 않는다는 것(호출됐다면
`fenced_run_json`이 `AssertionError`를 던져 테스트 자체가 실패한다 —
no-send가 assertion 실패로 드러나는 구조). "재시작 전 명시적 rebind
필요"는 이 테스트가 직접 assert하는 대상이 아니라 §11.1의 제어
흐름 자체가 만드는 운영 계약이다: `run_loop`은 fenced 이후
`watchdog.stop`을 스스로 만들지 않고 그냥 프로세스를 종료하므로, 다음
감시는 항상 운영자/coordinator가 새 프로세스(`... run` 재실행)를
명시적으로 기동해야만 시작된다 — 이 함수 자신이 내부적으로 재시도
루프를 도는 경로가 없다.

**8) `test_bound_run_dry_run_receipt_is_durable`**
```python
def test_bound_run_dry_run_receipt_is_durable(tmp_path):
    setup_state(tmp_path)
    runner = FakeRunner(tasks=[], messages=[{"id": "msg_1", "type": "worker_done"}])
    result = watchdog.check_once(tmp_path, "run_watch123", "orca", dry_run=True, runner=runner)
    assert result["dry_run"] is True
    assert result["woke"] is False   # dry_run은 절대 send하지 않음(기존 계약)
    on_disk = watchdog.load_watchdog(
        state.state_dir(tmp_path, "run_watch123") / "watchdog_state.json")
    assert on_disk["last_check"] is not None
    assert on_disk["pid"] == os.getpid()
```

### 11.3 commit scope (REQ-009.4)

구현 phase의 첫 commit(또는 Phase 0에 준하는 독립 commit)은 다음을
**함께** 포함해야 한다 — 어느 것 하나만 커밋해 provenance가 끊기지
않도록:

1. `scripts/orchestration_watchdog.py`의 현재 working-tree delta
   (§0.2에 인용된 diff, terminal-scoping 부분은 추가 수정 없이 그대로)
   **더한** §11.1의 `_classify_runner_error` 헬퍼, `main()` 호출부
   변경(DR-I2-MAJ-05), 그리고 `run_loop`의 `except Exception` 분기를
   `consumer_fenced`(즉시 nonzero 종료)와 generic transient(기존
   interval 재시도)로 나누는 제어 흐름 변경(DR-I3-MAJ-05) — 모두
   기존 코드 삭제 없는 순수 추가/분기 확장이며, 같은 파일의 같은
   commit에 함께 포함하되 이미 승인된 terminal-scoping delta의 의미는
   바꾸지 않는다.
2. `tests/unit/test_orchestration_watchdog.py`에 §11.2의 8개 테스트
   추가(기존 9개는 무변경).
3. 커밋 메시지 또는 이 문서 §0.2/§11.1에 이미 기록된 provenance —
   tracked base `e57fe1c`, 변경 의도("task-list/check 조회를 실행
   coordinator terminal identity에 bind"하는 기존 delta +
   "runner 실패를 bounded reason으로 정규화"하는 신규 delta), 작성
   주체("root/coordinator가 Continuous Operation Readiness Gate 준비
   중 만든 M4.3 의도적 readiness fix, bounded-reason classifier는
   Iteration 2 review로 추가된 동일 scope 보강") — 를 커밋 본문에
   명시한다.
4. `venv/bin/python scripts/orchestration_watchdog.py --run-id <run-id>
   check --dry-run`의 실제 stdout(§11.2-7이 자동화하는 것과 같은 내용을
   그 구현 Task 자신의 실제 coordinator Run에 대해 한 번 더 수동
   실행)을 구현 Task의 PR 본문 또는 커밋 메시지에 첨부해 "실제
   bound-Run dry-run receipt"(REQ-009.3) 증거로 남긴다 — 자동화된
   테스트(§11.2)가 논리를 증명하고, 이 수동 transcript가 실제 운영
   Run에 대해서도 동작함을 증명하는 이중 증거다.
5. 독립 review(§13의 Codex review checklist 7번 항목 "M4.1 blocker/
   protected gate 불변"이 이 항목도 포함하도록 Traceability에 이미
   반영돼 있다 — §13 참조)를 거친 뒤에만 이 commit을 M4.3 diff에
   포함한다.

## 12. Fault injection 테스트 매트릭스 요약

| 주입 지점 | monkeypatch 대상 | 기대 결과 |
|---|---|---|
| index.faiss write 실패 | `_write_fsync`가 첫 호출에서 `OSError` | `.staging/<op>` 잔류, `versions/`/`current` 무변경 |
| index.pkl write 실패 | 위와 동일, 두 번째 호출 | 동일 |
| manifest fsync 실패 | `os.fsync`를 manifest fd에서만 raise | 동일 |
| staging dir fsync 실패 | `_fsync_dir(op_dir)`에서 raise | 동일 |
| smoke 단계 hash 불일치 | 쓰기 직후 파일 1바이트 변조 | `member_hash_mismatch`, publish 진행 안 함 |
| rename 실패(EXDEV) | `os.rename`을 `OSError(EXDEV)`로 monkeypatch | `cross_device_staging`, `.staging` 잔류 |
| rename 후 chmod 실패 | `os.chmod`가 `dest` 첫 파일에서 raise | `versions/<id>`는 이미 존재(원자적 rename 완료) — 이 케이스는 "publish 자체는 성공, 방어적 read-only화만 부분 실패"로 별도 분류하고 재시도 가능하게 설계(§4.3 chmod 실패는 `LifecycleError("chmod_partial")`로 receipt에 남기되 `dest`를 롤백하지 않음 — 디렉터리가 최종적으로 존재하고 해시가 맞으면 version 자체는 유효하기 때문) |
| activation pointer write 실패 | `_write_fsync(tmp, ...)`에서 raise | `current` 무변경, receipt 없음 |
| pointer replace 실패 | `os.replace`가 raise | `current` 무변경, receipt 없음 |
| pointer 이후 dir fsync 실패 | `_fsync_dir(index_root)`가 raise | `current`는 새 값이지만(레이스 상 이미 rename됨) receipt는 **쓰지 않음**(§4.4 불변식) — 다음 `verify`/`activate` 호출이 재검증하므로 안전 |
| 동시 lock 경쟁 | 별도 프로세스/스레드가 `.lock` 선점 | `LockTimeoutError`, exit 3, `current` 무변경 |
| disk full(ENOSPC) | `_write_fsync`가 `OSError(ENOSPC)` | write 실패 케이스와 동일 처리(잔류 staging, 무변경 current) |
| **(신규, DR-I1-CRIT-01) verify 이후 racer가 version_dir/멤버 교체** | `verify_version()` 반환 직후 별도 스레드가 `versions/<id>`를 rename하거나 `shutil.rmtree` | `load_verified_faiss`는 이미 캡처한 bytes만 사용 — racer 이후에도 결과 불변, 신규 `os.open` 0회(§3.4) |
| **(신규, DR-I1-CRIT-01) verify_version 내부 멤버 open 사이 디렉터리 교체** | `manifest.json`/`index.faiss`/`index.pkl` 세 open 사이 racer가 `versions/<id>`를 다른 디렉터리로 rename | 세 open이 모두 같은 `version_dir.fd`(§3.2) 상대이므로 영향 없음 — racer의 새 디렉터리는 조회되지 않음 |
| **(신규, DR-I1-MAJ-02) `current`가 symlink/dangling symlink** | `current -> /etc/passwd` 또는 `current -> ./nonexistent` | `TrustBoundaryError("current_pointer_symlink")`, legacy loader 호출 0회 |
| **(신규, DR-I1-MAJ-03/DR-I2-MIN-06) source_dir가 승인 pair와 불일치** | `_approved_override`로 다른 hash pair를 주입하거나(negative-only test seam) 실제 `source_dir`의 `index.faiss`/`index.pkl`을 1바이트 변조 | `TrustBoundaryError("member_hash_mismatch")`, staging/publish 진행 안 함 |
| **(신규, DR-I2-MIN-06, fixture seam은 DR-I3-MIN-06) pinned 상수와 tracked baseline 파일 불일치** | `evaluation/baselines/m3_initial.json`의 tracked bytes를 `tmp_path`에 복사한 뒤 그 임시 사본에서만 hash pair 1바이트를 변조(원본 tracked 파일은 무변경)하고, `_parse_m3_baseline_fingerprint(tampered_bytes)`의 결과를 `_PINNED_M3_APPROVED_*` 상수와 비교 | `test_tampered_baseline_copy_diverges_from_pinned_constants`가 "변조된 사본의 hash != pinned 상수"를 assert(비교 메커니즘이 실제로 변조에 민감함을 증명); `test_pinned_constants_match_tracked_m3_baseline_bytes`(positive)는 별도로 원본 tracked bytes와 상수의 일치를 assert — 둘 중 하나라도 실패하면 CI job 자체가 실패(런타임 TrustBoundaryError가 아니라 provenance 회귀 테스트 실패) |
| **(신규, DR-I1-MAJ-04) `.staging` 이름 불일치/symlink 항목** | `.staging/not-a-uuid/`, `.staging/<uuid>`가 symlink | `cleanup --include-staging --apply`가 두 항목 모두 후보에서 제외, 삭제 0건 |
| **(신규, DR-I1-MAJ-04) young staging 항목** | `.staging/<uuid>`가 `staging_min_age_seconds`보다 최근 mtime | 후보에서 제외(삭제 0건), 다음 실행에서 나이 조건 충족 시에만 후보 포함 |
| **(신규, DR-I1-MAJ-04) versions 삭제 중 racer symlink 교체** | `cleanup --apply` 실행 중 racer가 삭제 대상 디렉터리를 symlink로 교체 시도 | `_fd_relative_rmtree`가 이미 연 dirfd를 쓰므로 racer의 교체는 무시되고, 명시적 symlink 감지 시 `TrustBoundaryError("version_dir_symlink")`로 중단 |
| **(신규, DR-I1-MAJ-05) `.lock`이 이미 symlink로 존재** | activate 실행 전 공격자가 `.lock -> /tmp/evil` symlink를 미리 심어둠 | `os.open(..., O_NOFOLLOW, dir_fd=root_fd)`가 `ELOOP`로 즉시 실패, lock 미획득, `current` 무변경 |
| **(신규, DR-I1-MAJ-05) pointer replace 직후 crash(history append 전)** | `os.replace`+`_fsync_dir` 성공 직후 프로세스 kill(테스트는 `_append_history` 진입 전 `os._exit`로 시뮬레이션) | 재시작 후 `_reconcile_pending_transition`이 `actual_current == post`를 확인해 history/receipt를 `reconciled=True`로 사후 완결 |
| **(신규, DR-I1-MAJ-05) pointer replace 전 crash** | `_write_transition_journal(phase="prepared")` 직후, `os.replace` 진입 전 kill | 재시작 후 `actual_current == pre` 확인, 저널만 지우고 operation은 ABORTED(history/receipt 없음) — 운영자가 재시도 |
| **(신규, DR-I2-MAJ-01) history append 후·receipt write 전 crash** | `_append_history` 성공(fsync 포함) 직후, `_write_receipt_atomic` 진입 전 kill, 재시작 후 reconcile을 2회 연속 호출 | history 행 op_id당 정확히 1개, `_write_receipt_atomic`이 사후 완결한 receipt의 `operation_id`가 그 op_id와 일치, 두 번째 reconcile 호출은 아무것도 다시 쓰지 않음(idempotent no-op) |
| **(신규, DR-I2-MAJ-01) receipt write 후·journal unlink 전 crash** | `_write_receipt_atomic` 성공 직후, `os.unlink(journal_path)` 진입 전 kill, 재시작 후 reconcile을 2회 연속 호출 | history/receipt는 이미 완결 상태(그대로), 첫 reconcile 호출이 journal만 지우고, 두 번째 호출은 `journal_path.is_file()`이 False이므로 즉시 no-op 반환 |
| **(신규, DR-I2-MAJ-01) journal unlink 자체 도중 crash** | `os.unlink` 완료 직후·`_fsync_dir(index_root)` 진입 전 kill 시뮬레이션(unlink는 원자적이므로 관찰 가능한 상태는 "이미 지워짐" 또는 "아직 있음" 둘 중 하나) | 재시작 후 journal이 없으면 reconcile은 `None`을 반환하는 정상 종료 경로(§4.4-b 첫 줄, `not journal_path.is_file()`)로 처리되고 history/receipt는 이미 첫 완결에서 정확히 1개 |
| **(신규, DR-I1-MAJ-06) production 이미지에 web/static 미포함(회귀)** | Dockerfile에서 `COPY web/static/` 줄을 monkeypatch로 제거한 fixture 이미지 | `container_smoke.py`가 `static_asset_ok=false`로 exit 1 |
| **(신규, DR-I1-MAJ-07) `--add-host` 누락(회귀)** | `container_smoke.py`의 docker run argv에서 `--add-host` 항목 제거 | Linux 환경에서 mock Ollama 연결 실패, `mock_query_ok=false`로 exit 1 |
| **(신규, DR-I4-MAJ-01) `previous` 대수 — empty/first/second/rollback/duplicate-gap/filename mismatch** | `tests/unit/test_index_lifecycle.py::test_previous_history_algebra_matrix`가 `activation_history/` 디렉터리를 fixture로 직접 구성(락/`activate()` 없이 `_read_history_rows`/`_read_previous_from_history`만 호출) — §4.4-a-1의 11개 서브케이스 전체 | empty→`previous is None`; empty+current 존재→`activation_history_current_mismatch`; first `A→B`(레코드 1개)→`previous == A`(이전 설계는 `None`을 반환하던 버그 재현); second `B→C`→`previous == B`; rollback `C→B`→`previous == C`; sequence duplicate/gap→`activation_history_sequence_invalid`; filename↔op_id mismatch→`activation_history_filename_op_id_mismatch`; operation enum 위반→`activation_history_operation_invalid`; `latest.post_pointer != current`→`activation_history_current_mismatch` |
| **(신규, DR-I4-MAJ-01) crash window에서도 `previous` 대수 성립** | 기존 `test_crash_recovery_history_and_receipt_exact_once_matrix`(§4.4-a-1)의 세 crash 주입 지점 각각의 재시작 후 상태에서 `_read_previous_from_history(index_root, current=<재시작 후 실제 current>)`를 추가 호출 | 세 crash 지점(tmp write/fsync 전, `os.replace` 직후·parent-fsync 전, 구 JSONL 잔여 fixture) 모두에서 `previous`가 항상 성공적으로 해석되고(예외 없음) 케이스 3-5(first/second/rollback)의 기대값과 일치 |

`tests/integration/test_index_lifecycle_fault_injection.py::
test_activate_rollback_100x`는 위 표의 "정상 경로"만 100회 반복해
partial/dangling pointer 0건, `current`의 매 순간 파싱 가능성(중간에
읽어도 항상 유효한 JSON)을 확인한다(REQ-003 정량 기준 "정상
activate/rollback 100회 partial/dangling pointer 0").

## 13. Traceability 매핑 (Requirement → 설계 심볼 → 테스트)

| Requirement | 설계 심볼 | 테스트 |
|---|---|---|
| M4.3-REQ-001 | `index/manifest.py`(§2.1), `index/verification.py::ContainedDir/verify_version/load_verified_faiss/resolve_current`(§3.2-3.5, CRIT-01/MAJ-02 반영) | `test_index_manifest.py`, `test_index_verification.py::test_current_pointer_trust_matrix`, race fixture(§3.2, §3.4) |
| M4.3-REQ-002 | `index/lifecycle.py::_stage_candidate/_publish/import_legacy/_pinned_m3_approved_pair/_parse_m3_baseline_fingerprint`(§4.2-4.3, §4.7, MAJ-03/DR-I2-MIN-06/DR-I3-MIN-06/**DR-I4-MIN-03**(pinned 상수를 승인 baseline의 실제 SHA-256 값으로 치환, placeholder 제거) 반영) | `test_pinned_baseline_provenance.py::test_pinned_constants_match_tracked_m3_baseline_bytes`(positive — 이제 실제 상수 대 tracked bytes 재계산을 assert)/`test_tampered_baseline_copy_diverges_from_pinned_constants`(negative), `test_index_lifecycle.py::test_import_legacy_rejects_source_hash_mismatch`, fault injection(§12), `git diff --exit-code -- evaluation/baselines/m3_initial.*`(§15 Gate — tracked baseline bytes 무변경 보존) |
| M4.3-REQ-003 | `index/lifecycle.py::activate/_reconcile_pending_transition/_append_history/_write_receipt_atomic/_read_history_rows/_read_previous_from_history/acquire_index_lock/cleanup/_fd_relative_rmtree`(§4.4-4.6, MAJ-04/MAJ-05/DR-I2-MAJ-01/DR-I3-MAJ-01/**DR-I4-MAJ-01**(`previous`를 latest record의 `pre_pointer`에서 도출 + `latest.post_pointer==current` 검증 + exact schema/filename↔op_id/operation enum/unique-contiguous sequence fail-closed 검사) 반영) | `test_index_lifecycle.py::test_previous_history_algebra_matrix`(신규 — empty/first `A→B`/second `B→C`/rollback `C→B`/sequence duplicate/sequence gap/filename↔op_id mismatch/operation enum 위반/current mismatch 11케이스), `test_index_lifecycle_fault_injection.py::test_crash_recovery_journal_reconciles_to_consistent_state`, `test_index_lifecycle_fault_injection.py::test_crash_recovery_history_and_receipt_exact_once_matrix`(short/partial write, newline 전 crash, partial tail 재시작 매트릭스 + `previous` 대수 재확인 포함), `test_index_lifecycle.py::test_cleanup_staging_protects_unexpected_and_young_entries` |
| M4.3-REQ-004 | `cli/index_lifecycle.py`(§6, `--baseline-json` 제거·`--include-staging` 추가, `--to-previous`가 `_read_previous_from_history`의 `current`-검증 계약을 그대로 전파) | `test_index_lifecycle_cli.py` |
| M4.3-REQ-005 | `deploy/Dockerfile`(web/static COPY, MAJ-06; production stage가 `tests/`를 COPY하지 않음, **DR-I3-MAJ-02**), `.dockerignore`, `scan_image_layers.py`(`simple_qna_rag_test_seam` forbidden pattern 신규), `container_smoke.py`(`--add-host`, static/root smoke, reachability probe, harness mount + 4-neg negative activation control, MAJ-07/DR-I2-MAJ-02/**DR-I3-MAJ-02**)(§5.2-a, §7.1-7.5), `_build_embeddings`/`TestEmbeddingSeamUnavailable`(§5.2-a) | `test_scan_image_layers.py`(test-seam-leak fixture 포함), `test_container_smoke_contract.py::test_docker_run_argv_includes_add_host_and_embedding_seam_env`/`test_reachability_probe_argv_targets_mock_ping_via_host_gateway`/`test_negative_activation_argv_omits_test_seam_mount_and_pythonpath`(신규), `test_settings_inventory.py::test_deterministic_embedding_provider_without_allow_flag_rejected`, `test_rag_engine_embeddings.py::test_build_embeddings_raises_seam_unavailable_when_module_absent`(신규), `container` CI job(§8.1, 4-neg 단계) |
| M4.3-REQ-006 | `docs/operations/*.md`, `deploy_drill.py`(§7.6-7.7) | `test_deploy_drill.py` |
| M4.3-REQ-007 | `.github/workflows/ci.yml` 추가분(`if-no-files-found: error`), `write_ci_producer_receipt.py`/`assemble_m4_evidence.py::_verify_payloads/_evaluate_producer/_check_identity/_parse_and_verify_m43_payload/_payload_manifest_sha256/EXPECTED_M43_NODE_IDS`(§8, MAJ-08/DR-I2-MAJ-03/DR-I3-MAJ-03/**DR-I4-MAJ-02**(producer receipt에 `payload_manifest_sha256` canonical identity 추가, assembler output(재검증된 payload_hashes)과 receipt가 선언한 baseline copy identity를 assembly 시점에 결합) 반영) | `test_assemble_m4_evidence.py`(§8.3 27개 negative + 1 positive + `test_expected_node_ids_matches_producer_profile_node_ids` provenance — payload-manifest hash malformed/mismatch 2건 신규 포함) |
| M4.3-REQ-008 | `check_m4_baseline.py::PRODUCER_STATUS_SCHEMA/PRODUCER_EXPECTED_PAYLOAD_FILENAMES/_payload_manifest_sha256`(§9.2, MIN-09/DR-I2-MAJ-04/DR-I3-MAJ-04/**DR-I4-MAJ-02**(count 대신 job별 exact payload filename set + payload_hashes에서 독립 재계산한 payload_manifest_sha256 결합 검사) 반영) | `test_check_m4_baseline.py`(최소 24개 파라미터화 케이스 — same-count filename substitution/extra+omission 상쇄/cross-job filename swap/malformed·mismatched payload-manifest hash 5건 신규 포함) |
| M4.3-REQ-009 | §5(호환성 브리지), §11(watchdog, 문서 표현 8종으로 정정, `_classify_runner_error` bounded reason 신규, **DR-I3-MAJ-05**로 `run_loop`이 `consumer_fenced` 후 즉시 nonzero 종료하도록 분기) | `test_settings_inventory.py`(회귀), `test_orchestration_watchdog.py`(§11.2, 8개 함수 — 항목 7 `test_run_loop_terminates_nonzero_after_consumer_fenced_with_exact_once_journal`로 재작성) |

## 14. 잔여 위험과 미해결 항목

1. **`embedding_model_revision="unknown"` 고정값**(§2.2) — HuggingFace
   로컬 캐시 모델의 정확한 revision을 얻는 안정적 API가 현재 의존성
   범위에 없다. 구현 phase에서 `huggingface_hub` API로 개선 가능하면
   그렇게 하되, 불가능하면 이 고정값을 유지하고 identity hash에는
   여전히 포함해(재현성에는 영향 없음, 단지 정보성 필드가 덜 정밀할
   뿐) 리스크를 흡수한다.
2. **cross-device rename(EXDEV) 미실측**(§0.3) — 이 개발 환경에서
   두 번째 파일시스템을 만들 수 없어 실제 `OSError(EXDEV)` 발생을
   실험적으로 재현하지 못했다. `errno.EXDEV` catch는 표준 POSIX 계약에
   근거하지만, 구현 phase의 CI(단일 파일시스템 hosted runner)에서도
   이 분기를 진짜로 트리거하는 결정론적 테스트를 만들기 어렵다 —
   `os.rename`을 monkeypatch해 `OSError(EXDEV)`를 인위 주입하는
   단위 테스트로 대체한다(§12 표에 이미 반영).
3. **컨테이너 mock query의 임베딩 모델 다운로드 시간** — `container_smoke.py`가
   테스트 fixture corpus로 즉석 `build()`를 수행하려면 GitHub-hosted
   runner에서 `BAAI/bge-m3` 임베딩 모델을 받아야 한다(`python-tests`
   job도 이미 이 비용을 지불하므로 신규 위험은 아니지만, `container`
   job이 병렬로 같은 다운로드를 또 하면 CI 총 시간이 늘어난다) —
   구현 phase에서 GH Actions cache(`actions/cache`)로 모델 캐시를 공유하는
   최적화를 고려할 수 있으나, 이는 성능 최적화이지 Gate 정합성
   문제는 아니므로 이 설계의 필수 요구사항으로 삼지 않는다.
4. **`activation_history/`의 무제한 증가** — 오래 운영되는 배포에서 이
   디렉터리의 레코드 파일 개수가 계속 늘어난다(DR-I3-MAJ-01로 단일
   JSONL에서 파일당-레코드로 저장 형태가 바뀌었지만, 무제한 증가라는
   운영 성격 자체는 동일하다 — 오히려 디렉터리 엔트리 수가 늘어나므로
   `listdir()` 비용이 파일 크기가 아니라 엔트리 수에 비례하게 된다는
   점만 다르다). M4.3 범위에서는 rotation/보관/compaction 정책을
   정의하지 않는다(Requirement에 명시된 요구가 없음) — 향후 milestone의
   잠재 후속 작업으로 남긴다.
5. **`docker save`를 통한 layer scan 비용** — 대형 이미지에서 `docker save`가
   느릴 수 있다. 현재 이미지는 `python:3.11-slim` 기반의 비교적 작은
   런타임(HuggingFace 모델은 이미지에 없음)이므로 실질적 위험은 낮다고
   판단하지만, 구현 후 실측이 필요하다.
6. **`EXPECTED_M43_NODE_IDS`(assembler)와 `PROFILE_NODE_IDS`(runner)의
   이중 유지보수**(§8.2-c, DR-I3-MAJ-03) — 두 리터럴을 물리적으로
   분리한 것 자체가 독립 오라클의 핵심이지만, 그 대가로 신규 acceptance
   node를 추가할 때마다 두 곳을 각각 수동으로 갱신해야 한다.
   `test_expected_node_ids_matches_producer_profile_node_ids`가 그
   불일치를 즉시 테스트 실패로 잡아주므로 "조용히 갈라짐"은 방지되지만,
   "한쪽만 고치고 CI가 실패할 때까지 모른다"는 개발자 경험상의 마찰은
   남는다 — REQ-007 범위에서는 이 마찰을 감수하는 것이 독립성 확보의
   필요조건이라고 판단했다.

## 15. Clean 검증 명령 재확인 (Plan §4 Phase 8과 동일, 이 설계가 이름을 확정)

```bash
bash scripts/compile_lock.sh --verify
python -m pip check
python -m pytest -q
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
npm ci
npm test
npm run sync-vendor
git diff --exit-code -- web/static/vendor/
git diff --exit-code -- evaluation/baselines/m3_initial.*
python scripts/generate_field_spec.py --check
python scripts/logging_callsite_audit.py --check
python scripts/check_markdown_links.py
python -m compileall -q src scripts tests evaluation
python scripts/run_m42_acceptance.py --profile deterministic --repeat 10 --seed 4202 --output <tmp>/m42.json
python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 --output <tmp>/m43.json
python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 --inject-evidence-mismatch --output <tmp>/m43-negative.json
docker build --target test -f deploy/Dockerfile .
docker build --target production -f deploy/Dockerfile -t simple-qna-rag:m43-candidate .
python scripts/scan_image_layers.py --image simple-qna-rag:m43-candidate --output <tmp>/layer_scan.json
python scripts/container_smoke.py --image simple-qna-rag:m43-candidate --output <tmp>/container_smoke.json
python scripts/deploy_drill.py --root <tmp>/drill-root --repeat 3 --output <tmp>/drill.json
python scripts/assemble_m4_evidence.py --fresh-dir <tmp>/m4-assemble --expected-sha "$(git rev-parse HEAD)" ...
python scripts/check_m4_baseline.py --candidate <tmp>/m4-assemble/m4-baseline.json --expect-operational-blocked
git diff --check
```

Native Linux/Ollama/DDGS, M3 live regression, M4.1 live 14-gate는 이
목록에 없으며, 그 부재는 §0.1의 두 판정 경계에 따라 전체 M4 release
blocker를 닫지 않는다.

## 16. Iteration 1 리뷰 반영 — Closure Matrix

[Design_Review_Iteration_1.md](Design_Review_Iteration_1.md)의 CRITICAL
1건·MAJOR 7건·MINOR 1건을 각각 이 개정에서 어디를 어떻게 바꿔 닫았는지
ID별로 정리한다. "리뷰 지적"은 원문 요지, "반영 위치"는 이 문서의 절
번호, "핵심 수정"은 코드/스키마/argv 수준의 실제 변경을 가리킨다 —
서술만 바꾼 항목은 없다.

| ID | Severity | 리뷰 지적 | 반영 위치 | 핵심 수정 |
|---|---|---|---|---|
| DR-I1-CRIT-01 | CRITICAL | 검증 뒤 `FAISS.load_local`이 경로를 재오픈해 hash 검증과 pickle 역직렬화 사이 TOCTOU 창이 생김; `contained_open`도 조상 디렉터리를 path 기반 사전검사만 함 | §0.4-1, §3.2, §3.3, §3.4, §12(신규 2행) | `ContainedDir`/`open_contained_root`로 root-to-leaf dirfd 체인을 열어 고정(§3.2); `verify_version`이 세 멤버를 **같은 version_dir fd**에서 읽음(§3.3); `load_verified_faiss`는 `FAISS.load_local`을 전혀 호출하지 않고 `verify_version()`이 캡처한 `faiss_bytes`/`pkl_bytes`에서 `faiss.deserialize_index`+`pickle.loads`로 직접 구성(§3.4) — 재오픈 자체가 코드에서 사라짐. `grep -rn "load_local" src/`가 1건(legacy 경로)만 남는지가 회귀 감사 명령 |
| DR-I1-MAJ-02 | MAJOR | `resolve_current`가 `pointer_path.exists()`로 사전 확인 — dangling symlink도 `False`가 되어 "없음"으로 오분류되고 검증 없는 legacy pickle로 fail-open downgrade | §3.5, §12(신규 1행) | `exists()` 사전 검사를 완전히 제거하고 `os.open(..., O_NOFOLLOW, dir_fd=root_fd)`의 errno만으로 분기: `ENOENT`→`CurrentPointerMissing`(genuine absence), `ELOOP`(symlink/dangling 모두)→`TrustBoundaryError("current_pointer_symlink")`(fail-closed, legacy 호출 0회). 4-케이스 spy 테스트로 legacy loader 0-call 검증 |
| DR-I1-MAJ-03 | MAJOR | CLI가 `--baseline-json PATH`를 노출해 호출자가 임의 hash를 담은 파일을 legacy import 승인 근거로 제출 가능 — "커밋된 baseline만 신뢰" 서술과 충돌 | §4.7(신규, DR-I2-MIN-06으로 추가 개정) | production CLI에서 `--baseline-json` 플래그를 완전히 삭제. Iteration 2 MIN-06 재검토 이후 최종 형태는 런타임 파일 읽기 자체를 제거하고 `_pinned_m3_approved_pair()`가 `index/lifecycle.py`의 코드 상수(`_PINNED_M3_APPROVED_INDEX_FAISS_SHA256`/`_PINNED_M3_APPROVED_INDEX_PKL_SHA256`)만 반환한다(배포 위치 문제가 구조적으로 사라짐); 커밋된 파일과의 대응은 `tests/unit/test_pinned_baseline_provenance.py`가 test 시점에 재확인한다. 테스트 전용 override는 `_approved_override`(leading underscore, CLI argparse 미노출) Python 인자로만 접근 가능 |
| DR-I1-MAJ-04 | MAJOR | retention 삭제가 path 기반 `realpath`/`is_symlink` 검사 후 `shutil.rmtree(target)` — 검사와 삭제 사이 TOCTOU; `.staging` 실패 잔여물은 정책(TTL/owner/liveness/dry-run) 없이 무제한 누적 | §4.6-a, §4.6-b(신규), §6.1, §12(신규 3행) | `_fd_relative_rmtree`가 openat/fstatat 기반 no-follow 재귀 삭제로 교체(dirfd 고정, path 재조회 없음); `.staging` 전용 `_cleanup_staging`이 UUID 이름 정규식·non-symlink·`staging_min_age_seconds`(전역 lock이 "진행 중" 보호를 구조적으로 겸함) 조건을 모두 만족하는 항목만 후보로 삼는 명시적 dry-run/apply 정책 추가; CLI에 `--include-staging`/`--staging-min-age-seconds` 신설 |
| DR-I1-MAJ-05 | MAJOR | `_write_fsync`가 short write 미처리; lock이 `touch()` 후 path 재오픈이라 symlink 교체에 취약; pointer fsync 이후 `_append_history` 실패 시 pointer는 바뀌었지만 history/receipt가 없어 rollback/retention 대칭성 붕괴; 재시작 시 결정론적 복구 절차 없음 | §4.2, §4.4, §4.4-a/b(신규), §4.5, §12(신규 3행) | `_write_fsync`를 완전 쓰기 loop로 교체(§4.2); lock을 trusted root dirfd에서 `O_CREAT\|O_NOFOLLOW` 단일 syscall로 열고 open 후 regular-file/mode 재확인(§4.5); pointer 교체 **전** `.transition` 저널에 pre/post를 durable 기록하고 `prepared`→(replace/fsync)→`pointer_committed`→history→receipt→저널 clear 순서를 명시(§4.4); `_reconcile_pending_transition`이 모든 lifecycle 진입점 시작 시 `actual_current`만으로(phase 문자열은 신뢰하지 않음) ABORTED/COMPLETED를 결정론적으로 판정(§4.4-b) |
| DR-I1-MAJ-06 | MAJOR | production stage가 `web/`을 COPY하지 않아 M4.2 readiness(static mount 실패 우선 503)를 이 이미지로 절대 만족할 수 없음 | §7.1, §7.5 | `COPY web/static/ ./web/static/`, `COPY web/templates/ ./web/templates/`를 production stage에 추가; `container_smoke.py`가 `GET /`와 정적 자산 응답을 명시적으로 확인하고 `static_asset_ok`/`root_page_ok` bool을 receipt에 기록, 실패 시 스스로 exit 1(§7.5) — §12에 회귀 fault 행 추가 |
| DR-I1-MAJ-07 | MAJOR | Linux hosted runner의 `docker run` argv에 `host.docker.internal` 이름을 만드는 옵션이 없어 mock Ollama 연결이 목표 환경에서 애초에 불가능 | §7.3, §7.5 | 표준 `docker run` 계약과 `container_smoke.py` 양쪽에 `--add-host host.docker.internal:host-gateway`(Docker 20.10+ Linux 표준 기능) 추가; `mock_query_ok` bool이 assembler(§8.2-a)의 semantic 검증 대상이 되어 이 플래그가 빠지면 CI 자체가 FAIL — §12에 회귀 fault 행 추가 |
| DR-I1-MAJ-08 | MAJOR | assembler가 producer 최소 receipt의 identity만 확인하고 `layer_scan.json`/`container_smoke.json`/`m43.json` 등 상세 evidence는 열어보지 않음 — `needs=success`+최소 receipt만 있으면 container scan 위반/smoke 실패가 있어도 합성 PASS 가능; `upload-artifact` 기본값이 파일 부재를 경고로만 처리 | §8.1(YAML `if-no-files-found: error`), §8.2-a/b(신규), §8.3 | producer receipt 스키마에 `payloads: [{filename, sha256, size_bytes, semantic_field, semantic_expected}]`와 `semantic_status`를 추가; `assemble_m4_evidence.py::_verify_payloads`가 각 필수 payload의 존재·hash·size·**내부 semantic 필드**(`forbidden_count==0`, `mock_query_ok/root_page_ok/static_asset_ok==true`, `m43.json.status=="PASS"`)를 candidate의 자기 보고와 무관하게 재계산해 `OK`/`PAYLOAD_INVALID`로 판정; 모든 `upload-artifact` step에 `if-no-files-found: error` 추가; §8.3에 payload hash/semantic/set-mismatch negative 케이스 4건 추가(9→13). **주의**: payload entry의 `semantic_field`/`semantic_expected`는 애초에 assembler 판정에 쓰이지 않는 정보성 필드였고(semantic 판정은 항상 §8.2-b `REQUIRED_PAYLOADS` spec과 §8.2-c typed parser가 payload 파일 bytes에서 독립 재계산했다), Iteration 5가 지적한 receipt exact-schema 결함(DR-I5-MAJ-01)을 닫기 위해 Iteration 6에서 두 필드가 payload entry 스키마에서 완전히 제거됐다 — 이 행은 Iteration 1 시점의 기록이므로 그대로 보존하고, 최종 payload entry 스키마는 §20 DR-I5-MAJ-01 행과 §8.2-a/§8.2-b를 따른다 |
| DR-I1-MIN-09 | MINOR | baseline checker가 존재하는 `gates.items()`만 순회해 필수 gate 누락을 감지하지 못하고, `deterministic_status`를 producer 상태에서 재계산하지 않고 candidate 자기 보고를 신뢰 | §9.2 | 최상위/`gates` 키 집합을 **exact-match**로 먼저 검사(불일치 시 이후 필드 접근 없이 즉시 실패); `deterministic_status`/`operational_status`/`overall_release_ready` 전부를 `gates`에서 재계산해 candidate 값과 비교; boolean 필드에 `isinstance(..., bool)` 타입 검사 추가; 최소 12개 파라미터화 negative 케이스로 omission/extra-key/wrong-type/null/truthy-string 전수 검증 |
| (문서 표현, 리뷰 "강점과 보존 조건" 항목) | — | §11.2 제목이 "6종"인데 실제 번호 항목 7개(그 중 2개 항목이 테스트 함수 2개씩 포함해 실제 함수 8개)를 나열 — 구현 전 정리 권고 | §11.2 | 제목을 "8종"으로 정정하고 8개 항목 각각에 테스트 함수 정확히 1개씩만 대응하도록 번호를 다시 매김(내용 변경 없음, 표현만 정정) |

모든 행의 "핵심 수정"은 이 문서의 다른 절(§0.4, §3, §4, §7, §8, §9,
§11, §12, §13)에 이미 반영된 실제 심볼/스키마/argv 변경을 가리키며,
이 표는 그 변경들을 요구사항 ID가 아니라 **리뷰 finding ID** 기준으로
재인덱싱한 것이다. 구현 phase는 이 표의 "반영 위치" 열을 그대로
구현 대상 절로 채택할 수 있다.

## 17. Iteration 2 리뷰 반영 — Closure Matrix

[Design_Review_Iteration_2.md](Design_Review_Iteration_2.md)의 CRITICAL
0건·MAJOR 5건·MINOR 1건을 각각 이 개정(Iteration 3)에서 어디를 어떻게
바꿔 닫았는지 ID별로 정리한다. §16과 같은 형식이며, "핵심 수정"은
이번 개정에서 **새로** 도입한 심볼/스키마/argv/test oracle만 가리킨다
— §16이 이미 닫은 항목을 다시 설명하지 않는다.

| ID | Severity | 리뷰 지적 | 반영 위치 | 핵심 수정 |
|---|---|---|---|---|
| DR-I2-MAJ-01 | MAJOR | `_reconcile_pending_transition`이 `actual_current == post`일 때 `_append_history`/`_write_receipt_atomic`을 무조건 재실행 — history append 성공 직후·receipt write 전 또는 receipt write 후·journal clear 전 crash하면 같은 operation이 history에 두 번 기록돼 `_read_previous_from_history`의 "마지막에서 두 번째"가 실제 이전 pointer가 아니라 current와 같아짐(previous/rollback/retention 대수 붕괴) | §4.4-a-1(신규), §4.4, §4.4-b, §12(신규 3행) | `_append_history`/`_write_receipt_atomic`을 `operation_id` 기준 **exact-once** 헬퍼로 재정의 — 이미 같은 `op_id` 행/receipt가 있으면 물리적으로 다시 쓰지 않고 즉시 반환. `activate()`의 정상 경로와 `_reconcile_pending_transition`의 사후 완결 경로가 **정확히 같은 두 함수**를 호출하므로 어느 경로·몇 번을 재시도하든 op_id당 물리적 쓰기는 1회. `_read_previous_from_history`를 `activation_history.jsonl`의 마지막에서 두 번째 행의 `post_pointer`로 명시적으로 재정의(§4.4-a-1). history append 후·receipt write 전, receipt write 후·journal unlink 전, journal unlink 자체 도중 3개 crash 지점을 각각 재시작·재조정 2~3회 반복해 history 행 op_id당 정확히 1개, `previous=pre_pointer`, `current=post_pointer`를 검증하는 `test_crash_recovery_history_and_receipt_exact_once_matrix`(§10 `crash_recovery_journal` 노드에 추가). **주의**: 이 JSONL append 저장 형태 자체는 Iteration 3이 `PIPE_BUF`가 regular file에 적용되지 않는다는 점을 지적해(DR-I3-MAJ-01) Iteration 4에서 operation별 불변 레코드 파일(`activation_history/<op_id>.json`)로 전면 교체됐다 — 이 행은 Iteration 2 시점의 기록이므로 그대로 보존하고, 최종 형태는 §18 DR-I3-MAJ-01 행과 §4.4-a-1을 따른다 |
| DR-I2-MAJ-02 | MAJOR | mock 서버의 bind 주소/reachability가 계약화되지 않았고, 컨테이너 안에서 host가 만든 index를 query하려면 같은 embedding model을 컨테이너 안에서 초기화해야 하는데 이미지에 model/cache가 없고 read-only rootfs/제한된 `/tmp`에서 그 경로가 정의되지 않아 hosted Linux smoke가 구현 불가능 | §5.1, §5.2-a(신규), §7.5, §12 | `mock_ollama.py`가 `("0.0.0.0", 0)`에 명시적으로 bind하고 `/mock/ping` reachability probe 엔드포인트를 추가; `container_smoke.py`가 `docker exec <container> python -c ...`로 host-gateway 연결 자체를 앱 로직과 독립적으로 먼저 확인(`host_gateway_reachable` bool을 receipt/assembler semantic 검증 대상에 추가). `EMBEDDING_PROVIDER`/`ALLOW_TEST_EMBEDDING` 2-키 게이트 Settings 필드(§5.1)와 `DeterministicTestEmbeddings`(§5.2-a, `src/simple_qna_rag/deterministic_embeddings.py` 신규 파일)가 host build와 컨테이너 query 양쪽에서 같은 네트워크/모델-불필요 provider를 쓰게 해 embedding runtime 문제를 구조적으로 제거; production 기본값(`huggingface`/`False`)에서는 이 seam이 활성화될 수 없음을 negative test(`test_deterministic_embedding_provider_without_allow_flag_rejected`)와 기본값 회귀 test(`test_build_embeddings_default_uses_huggingface_provider`)로 이중 고정. exact docker argv/reachability probe argv unit test 2종 신설. **주의**: `src/simple_qna_rag/deterministic_embeddings.py` 배치 자체는 Iteration 3이 이 위치가 production 이미지에 COPY된다는 점을 지적해(DR-I3-MAJ-02) Iteration 4에서 `tests/support/simple_qna_rag_test_seam/`으로 재배치되고 물리적 봉인+negative OCI test가 추가됐다 — 이 행은 Iteration 2 시점의 기록이므로 그대로 보존하고, 최종 형태는 §18 DR-I3-MAJ-02 행과 §5.2-a를 따른다 |
| DR-I2-MAJ-03 | MAJOR | assembler가 `m43-deterministic` payload를 `m43.json.status == "PASS"` 한 필드로만 검사 — `schema`/seed/repeat/required node exact set/각 node repeat·status/negative control 결과를 재계산하지 않아 `{"status":"PASS"}` 최소 위조 파일도 통과 가능; malformed payload entry가 assembler를 KeyError로 죽일 수 있음 | §10.1(신규), §8.2-a, §8.2-c(신규), §8.3 | `run_m43_acceptance.py` 출력의 정확한 스키마(`m43-acceptance-receipt-v1`: schema/profile/seed/repeat/command/nodes/negative_control/status)를 §10.1로 명문화. `_parse_and_verify_m43_payload`(§8.2-c)가 `PROFILE_NODE_IDS`를 `run_m43_acceptance.py`에서 직접 import해 node exact set/count/status, seed=4303/repeat=10 exact match, negative control의 `REJECTED_AS_EXPECTED` 결과를 독립 재계산 — `m43.json`/`m43-negative.json` 두 payload 모두 이 typed parser로 검증. `_verify_payloads`의 payload dict comprehension을 `.get()` 기반 방어적 필터로 교체해 malformed entry가 assembler를 죽이지 않고 `payload_set_mismatch`로 typed FAIL하게 함. §8.3 negative case 7건 신설(총 13→20+1) |
| DR-I2-MAJ-04 | MAJOR | `check_m4_baseline.py`가 `producers`의 exact key/schema/status를 전혀 검사하지 않고 candidate가 자기 보고한 `gates`만으로 `deterministic_status`를 재계산 — `producers={}`이거나 전부 `MISSING`이어도 `gates`를 `PASS`로 쓰면 통과; `expected_ready`도 재계산 로컬 변수 대신 candidate의 `deterministic_status`/`operational_status`를 재사용 | §9.2 | `REQUIRED_PRODUCER_KEYS`/`PRODUCER_STATUS_ENUM`/`PRODUCER_TO_GATE_KEY`를 신설해 `producers`를 `gates`와 동급으로 exact-key/enum 검사. gate 값 자체를 `producers[job].status`에서 재계산한 뒤(`"PASS" if status=="OK" else "FAIL"`) candidate의 self-report `gates`와 비교(`gate_producer_algebra_mismatch`) — 이 비교를 통과해야만 이후 대수 계산에 진입. `deterministic_status`/`operational_status`/`overall_release_ready` 전부를 producer 파생 **로컬 변수**(`expected_deterministic`/`expected_operational`)에서만 도출해 candidate의 `gates`를 다시 참조하지 않음. `producers={}`, 전부 `MISSING`, 부분 위조(`producer=MISSING`+`gate=PASS`) 3개 필수 negative case 신설(총 12→15+) |
| DR-I2-MAJ-05 | MAJOR | watchdog 8-test 계약의 exact argv 테스트가 `consumer_fenced` 구조화 실패의 재현·전파·no-send·journal bounded reason을 검증하지 않음; 테스트 6은 stdout에 `"ok": true` 부재를 요구한다고 서술했지만 실제로는 return code만 assert | §11.1(신규 delta), §11.2(항목 5/6/7 재작성) | `orchestration_watchdog.py`에 `_classify_runner_error` 헬퍼를 순수 추가(기존 terminal-scoping delta는 무변경)해 `main()`/`run_loop`의 예외 처리가 원본 커맨드라인+stderr 대신 고정 vocabulary(`"consumer_fenced"` 또는 `"cli_command_failed"`) bounded reason만 stdout/journal에 남기도록 함. 8-test 계약의 항목 5를 `test_check_propagates_consumer_fenced_fail_closed_with_no_send`(exact argv + fail-closed 전파 + `orca terminal send` 미호출)로, 항목 6을 `test_check_subcommand_exits_2_with_bounded_stderr_and_no_stdout_success`(exit 2 + capsys로 stdout `"ok": true` 부재 실제 검증 + stderr bounded reason + 원본 stderr/traceback 비노출)로, 항목 7을 `test_run_loop_records_bounded_consumer_fenced_reason_in_journal_and_continues`(journal의 `"reason": "consumer_fenced"` bounded 필드 + 원본 텍스트 비노출 + loop 계속)로 재작성. 총 8개 항목·8개 함수 구조는 유지 |
| DR-I2-MIN-06 | MINOR | `_PINNED_M3_BASELINE_PATH`가 소스 checkout 기준 상대 경로라 `pip install --target`/컨테이너 이미지 배포 형태에서 실재하지 않을 수 있음; 테스트 이름 `rejects_tampered_or_untracked_baseline`이 실제로는 git tracked 여부를 검사하지 않아 이름과 보장이 불일치 | §4.7(전면 개정) | DR-I1-MAJ-03과 동일 절에서 함께 닫힘 — `import_legacy`의 production 경로가 런타임에 어떤 파일도 열지 않고 `_pinned_m3_approved_pair()`의 코드 상수만 참조하도록 재작성해 "이 경로가 이 배포 형태에서 실재하는가"라는 질문 자체를 제거. "커밋된 파일과 상수가 일치하는가"는 test 시점에만 필요하므로 `tests/unit/test_pinned_baseline_provenance.py::test_pinned_constants_match_tracked_m3_baseline_bytes`로 이관하고, 순수 hash-mismatch 검증은 `test_import_legacy_rejects_source_hash_mismatch`로 이름과 범위를 분리해 각 테스트 이름이 실제 보장과 일치하도록 정정 |

이 표의 여섯 행 모두 "구현 가능한 symbol/schema/argv/test oracle"
수준으로 구체화됐다 — 함수 시그니처, 새 파일 경로, exact JSON
key/enum, exact argv 리스트, 신규/재작성 테스트 함수명이 각 대응
절에 이미 존재한다. Native Linux/Ollama/DDGS, protected M3 live,
M4.1 live 14-gate는 이번 개정에서도 실행하지 않았으며, `M4.1_BLOCKED=true`,
protected M3 live `NOT_RUN`, `overall_release_ready=false`는 §0.1/§9.1의
불변식 그대로 보존된다. 이 개정은 어떤 PASS도 합성하지 않았고
self-hosted runner/environment 승인을 변경하지 않았다.

## 18. Iteration 3 리뷰 반영 — Closure Matrix

[Design_Review_Iteration_3.md](Design_Review_Iteration_3.md)의 CRITICAL
0건·MAJOR 5건·MINOR 1건을 각각 이 개정(Iteration 4)에서 어디를 어떻게
바꿔 닫았는지 ID별로 정리한다. §16/§17과 같은 형식이며, "핵심 수정"은
이번 개정에서 **새로** 도입한 심볼/스키마/argv/test oracle만 가리킨다
— §16/§17이 이미 닫은 항목을 다시 설명하지 않는다. "Requirement/Traceability"
열은 §13의 어느 행이 이 finding의 closure를 추적하는지 명시한다.

| ID | Severity | 리뷰 지적 | 반영 위치 | Requirement/Traceability | 핵심 수정(symbol/schema/argv/test oracle) |
|---|---|---|---|---|---|
| DR-I3-MAJ-01 | MAJOR | `_append_history`가 `O_APPEND` regular file에 대한 단일 `os.write`에 의존 — `PIPE_BUF` 원자성은 pipe/FIFO 계약이지 regular file complete-write 보장이 아니다. short write/newline 전 crash가 남긴 partial tail 뒤에 재시도 record가 이어붙으면 malformed 한 줄이 되고, 그 op_id는 이후에도 영구히 나타나지 않는다 | §2(디렉터리 레이아웃), §4.4-a-1(전면 재설계) | M4.3-REQ-003(§13) | history 저장 단위를 "JSONL 한 줄"에서 **operation당 하나의 불변 레코드 파일**(`activation_history/<op_id>.json`)로 교체 — §4.2와 동일한 primitive(temp full-write + fsync + `os.replace` atomic rename + parent-dir fsync)로 커밋. `_read_history_rows`/`_read_history_op_ids`는 이름 정규식(`^[0-9a-f]{32}\.json$`)으로 미완결 `.tmp.*` 잔여물을 원천 배제하므로 partial read가 구조적으로 불가능. exact-once는 `dest.exists()` 검사로, ordering은 `sequence` 필드(매 커밋 시 재계산)로 보장. 완결 파일의 파싱 실패는 새 reason `activation_history_record_corrupt`(§3.1)로 fail-closed. `test_crash_recovery_history_and_receipt_exact_once_matrix`(§10 `crash_recovery_journal` 노드)가 tmp write/fsync 단계 crash, `os.replace` 직후·parent-fsync 전 crash, 구 JSONL 잔여 fixture 세 지점을 각각 재현 |
| DR-I3-MAJ-02 | MAJOR | `DeterministicTestEmbeddings`가 `src/`(production COPY 대상) 안에 있어, 2-키 Settings 게이트를 통과할 두 env var만 설정하면 production 배포에서도 활성화 가능 — "production 경로에서 활성화 불가능"이라는 문서 주장과 실제 분기가 모순. hosted smoke도 별도 test harness가 아니라 이 모듈에 직접 의존 | §5.1(Layer 1/2 분리), §5.2-a(전면 재설계), §7.1, §7.4, §7.5(harness mount + 4-neg) | M4.3-REQ-005(§13) | 모듈을 `src/` 밖 `tests/support/simple_qna_rag_test_seam/deterministic_embeddings.py`로 이전 — production Dockerfile은 `tests/`를 어떤 stage에서도 COPY하지 않으므로 물리적으로 부재. `_build_embeddings()`가 `importlib.import_module`로 로드를 시도해 `ModuleNotFoundError`를 `TestEmbeddingSeamUnavailable`(reason=`test_embedding_seam_unavailable`)로 변환, `RAGEngine.initialize()`가 `artifact_error_reason`에 반영해 readiness 503 `artifact_test_embedding_seam_unavailable`로 fail-closed. `scan_image_layers.py`에 `simple_qna_rag_test_seam` forbidden pattern 추가(정적 이중화). hosted smoke(§7.5 4단계)는 **같은 production 이미지**에 `tests/support/`를 `:ro` bind mount + `PYTHONPATH` env로만 주입(동일 `USER`/`--read-only`/`--cap-drop`/`--security-opt` 유지). 신규 4-neg 단계가 harness 없이 같은 이미지로 별도 컨테이너를 띄워 503+정확한 reason을 bounded polling으로 확인하는 negative OCI test — `production_test_seam_sealed` bool을 receipt/assembler semantic 검증 대상에 추가 |
| DR-I3-MAJ-03 | MAJOR | assembler의 `_parse_and_verify_m43_payload`가 `PROFILE_NODE_IDS`를 `run_m43_acceptance.py`에서 직접 import — runner 버그가 필수 node를 constant와 output 양쪽에서 함께 누락하면 assembler도 같은 축소 constant로 통과시킨다(독립 재계산이 아님). `command` 값, negative receipt의 `expected_to_fail is True`, positive receipt의 `expected_to_fail is None`/`actual_exit_code is None`도 검사하지 않음 | §8.2-c(전면 재설계) | M4.3-REQ-007(§13) | assembler 자신의 review-pinned `EXPECTED_M43_NODE_IDS` 독립 상수를 신설 — `run_m43_acceptance.PROFILE_NODE_IDS`를 런타임에 import하지 않는다. 두 리터럴의 legitimate 동기화는 `test_expected_node_ids_matches_producer_profile_node_ids`(provenance 회귀, production 판정 경로와 분리)로만 확인. `doc["command"] != M43_EXPECTED_COMMAND`(assembler 자신의 `M43_REPEAT`/`M43_SEED`로 조립) exact match 추가. negative receipt는 `expected_to_fail is True`+`actual_exit_code == 1`, positive receipt는 `expected_to_fail is None`+`actual_exit_code is None`을 명시적으로 요구하도록 `_parse_and_verify_m43_payload` 재작성. §8.3에 command 변조/runner constant+output 동시 누락/expected_to_fail 위조/positive 가짜 exit code 4개 negative case 신설 |
| DR-I3-MAJ-04 | MAJOR | `check_m4_baseline.py`가 각 producer entry를 `"status" in entry`로만 검사 — 네 entry를 모두 `{"status":"OK"}`로 만든 candidate가 receipt hash/`needs_result`/payload hash/reason 없이도 통과 | §9.2(`PRODUCER_STATUS_SCHEMA` 신규) | M4.3-REQ-008(§13) | `_evaluate_producer`(§8.2-b)가 실제로 내는 producer 결과를 status별 exact key-set을 갖는 **tagged union**으로 명문화(`PRODUCER_STATUS_SCHEMA`) — `OK`는 `{status, receipt_sha256, payload_hashes}`, `FAILED_OR_SKIPPED`는 `{status, needs_result}`, `IDENTITY_MISMATCH`/`PAYLOAD_INVALID`는 `{status, reason}`, `DUPLICATE_PRODUCER`는 `{status, count}`, 나머지는 `{status}`. `set(entry) != PRODUCER_STATUS_SCHEMA[status]`이면 즉시 `producer_variant_schema_mismatch`. `OK` entry는 추가로 `receipt_sha256`이 64-hex인지, `payload_hashes`가 그 job의 선언 개수(`PRODUCER_EXPECTED_PAYLOAD_COUNT`)와 정확히 같은 64-hex 값 dict인지 검사. `test_check_m4_baseline.py`에 최소-status-only(4건)/malformed hash/count mismatch/failure-variant에 success 필드 혼입 negative case 4건 신설 |
| DR-I3-MAJ-05 | MAJOR | `consumer_fenced` bounded reason classifier는 정확했지만 `run_loop`의 `except Exception` 분기가 reason과 무관하게 항상 journal 기록 후 `interval` 뒤 재시도 — ownership이 거부된 프로세스가 계속 retry, journal도 매 interval 누적, 실행 프로세스는 성공 exit 경로에 남음 | §11.1(run_loop 제어 흐름 분기 신규), §11.2(항목 7 재작성), §11.3 | M4.3-REQ-009(§13) | `run_loop`의 예외 처리를 `reason == CONSUMER_FENCED_MARKER` 여부로 분기 — journal에 단 한 번 기록한 뒤 즉시 `return 1`(재시도 없음, 다음 감시는 항상 새 프로세스의 명시적 rebind로만 시작), generic transient(`cli_command_failed`)만 기존과 같이 `interval` 재시도 대상으로 유지. 8-test 계약 항목 7을 `test_run_loop_terminates_nonzero_after_consumer_fenced_with_exact_once_journal`로 재작성 — fence 후 `check_once` 호출 1회(재시도 없음), journal 내 `consumer_fenced` 행 exact-one, `run_loop` 반환값 nonzero, `orca terminal send` 미호출을 단일 테스트에서 검증 |
| DR-I3-MIN-06 | MINOR | §12가 baseline provenance 테스트를 "임시 복사본에서 1바이트 변조"라고 서술했지만 실제 테스트는 고정 tracked 경로를 직접 읽어 임시 복사본을 주입할 seam이 없었음. `_PINNED_M3_APPROVED_*` 상수는 여전히 `"0"*64` placeholder | §4.7(파서 분리 + negative test 신설), §12, §15(Gate 명령 추가) | M4.3-REQ-002(§13) | 파싱 로직을 `_parse_m3_baseline_fingerprint(raw: bytes) -> dict` 순수 함수로 분리 — path를 전혀 모른다. positive test(`test_pinned_constants_match_tracked_m3_baseline_bytes`)는 tracked 경로 bytes를, 신규 negative test(`test_tampered_baseline_copy_diverges_from_pinned_constants`)는 `tmp_path`에 만든 1바이트 변조 사본 bytes를 **같은 함수**에 넣어 서로 다른 결론(일치 vs 불일치)을 증명. 구현 Gate에 placeholder→실제값 치환과 `git diff --exit-code -- evaluation/baselines/m3_initial.*`(§15)를 명시 |

이 표의 여섯 행 모두 이전 두 iteration과 동일하게 "구현 가능한
symbol/schema/argv/test oracle" 수준으로 구체화됐고, 각 행이 §13
Traceability 매핑의 정확히 한 Requirement 행에 연결된다 —
Requirement.md/Traceability.md 자체는 이번 개정에서 수정하지 않았으며
(이 문서는 Design.md만 개정 대상이다), §13이 그 문서들과 이 설계
사이의 유일한 매핑 지점이라는 불변식은 그대로다. Native Linux/Ollama/
DDGS, protected M3 live, M4.1 live 14-gate는 이번 개정에서도 실행하지
않았으며, `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`,
`operational_status=BLOCKED`, `overall_release_ready=false`는 §0.1/§9.1의
불변식 그대로 보존된다. 이 개정은 어떤 PASS도 합성하지 않았고
self-hosted runner/environment 승인을 변경하지 않았다.

## 19. Iteration 4 리뷰 반영 — Closure Matrix

[Design_Review_Iteration_4.md](Design_Review_Iteration_4.md)(FAIL 9.1/10,
CRITICAL 0건·MAJOR 2건·MINOR 1건)를 각각 이 개정(Iteration 5)에서
어디를 어떻게 바꿔 닫았는지 ID별로 정리한다. §16-§18과 같은 형식이며,
"핵심 수정"은 이번 개정에서 **새로** 도입한 심볼/스키마/argv/test
oracle만 가리킨다 — §16-§18이 이미 닫은 항목을 다시 설명하지 않는다.
"Requirement/Traceability" 열은 §13의 어느 행이 이 finding의 closure를
추적하는지 명시한다.

| ID | Severity | 리뷰 지적 | 반영 위치 | Requirement/Traceability | 핵심 수정(symbol/schema/argv/test oracle) |
|---|---|---|---|---|---|
| DR-I4-MAJ-01 | MAJOR | `_read_previous_from_history`가 최신 record의 `pre_pointer`가 아니라 `rows[-2].post_pointer`를 반환 — record가 정확히 하나뿐인 최초 activation/import 직후(`pre_pointer=A`, `post_pointer=B`)에 무조건 `previous=None`을 반환해 같은 절의 fault oracle("operation당 exact-one, `previous=pre_pointer`, `current=post_pointer`")과 직접 모순되고, 첫 배포 직후 `rollback --to-previous`와 retention의 직전 version 보호를 잃는다. 또한 `_read_history_rows`가 record exact-key/type, filename의 op_id와 body `op_id` 일치, sequence uniqueness/contiguity를 검사하지 않아 ordering oracle이 단순 `r["sequence"]` 정렬에 머문다 | §3.1(신규 REASONS 5종), §4.4(narrative 갱신), §4.4-a-1(전면 재설계 — 헤더/`_read_history_rows`/`_read_previous_from_history`), §4.6-a(cleanup 호출부 2곳) | M4.3-REQ-003(§13) | `_read_history_rows`가 완결 레코드마다 exact key set(`_HISTORY_REQUIRED_KEYS`)/타입/`schema` 리터럴/`sequence` non-negative-int/`pre_pointer`·`post_pointer` 16-hex 정규식/`operation` enum(`_HISTORY_OPERATION_ENUM`)을 fail-closed 검증하고, 파일 이름의 op_id와 본문 `op_id`가 다르면 `activation_history_filename_op_id_mismatch`, 정렬 후 `sequence`가 `0..N-1` 연속 정수가 아니면(duplicate 또는 gap) `activation_history_sequence_invalid`로 즉시 거부. `_read_previous_from_history(index_root, *, current)`를 **최신(sequence 최대) committed record의 `pre_pointer`에서 직접 도출**하도록 재정의하고, 그 record의 `post_pointer == current`를 함께 검증해 불일치 시 `activation_history_current_mismatch`(신규 REASONS 5종: `activation_history_schema_invalid`/`activation_history_filename_op_id_mismatch`/`activation_history_operation_invalid`/`activation_history_sequence_invalid`/`activation_history_current_mismatch`, §3.1). `cleanup()`의 두 호출부(§4.6-a)와 CLI `--to-previous`(§4.4 narrative)가 모두 새 시그니처(`current=` 키워드 인자)로 갱신됨. `tests/unit/test_index_lifecycle.py::test_previous_history_algebra_matrix`(신규, §4.4-a-1)가 empty(레코드 0개)/empty+current 존재(모순)/first `A→B`(레코드 1개 — 이전 설계가 무조건 `None`을 반환하던 정확한 재현 케이스)/second `B→C`(레코드 2개)/rollback `C→B`(레코드 3개, `operation="rollback"`)/sequence duplicate/sequence gap/filename↔op_id mismatch/operation enum 위반/`latest.post_pointer != current` 총 11개 케이스를 파일시스템 fixture만으로 독립 검증하고, 기존 `test_crash_recovery_history_and_receipt_exact_once_matrix`(§4.4-a-1, §12)의 세 crash 주입 지점에서도 재시작 후 `previous` 대수가 성립함을 추가로 재확인 |
| DR-I4-MAJ-02 | MAJOR | `check_m4_baseline.py`의 `PRODUCER_STATUS_SCHEMA["OK"]` 검사가 `payload_hashes`의 **개수**와 값의 64-hex 형식만 확인 — `container`에 `{"a": "<64hex>", "b": "<64hex>"}`, `m43-deterministic`에 임의 두 filename을 넣어도 개수(2)만 맞으면 통과해 same-count filename substitution/extra+omission 상쇄/cross-job filename swap을 잡지 못했다. baseline candidate가 assembler가 실제로 검증한 `layer_scan.json`/`container_smoke.json` 또는 `m43.json`/`m43-negative.json` identity를 보존한다는 보장이 없어 DR-I3-MAJ-04가 요구한 receipt/payload identity summary를 exact schema로 닫지 못함 | §8.2-a(신규 `payload_manifest_sha256` 필드+helper), §8.2-b(`_evaluate_producer`/`_check_identity` 재작성), §8.3(신규 negative 2행), §9.2(`PRODUCER_STATUS_SCHEMA`/`PRODUCER_EXPECTED_PAYLOAD_FILENAMES`/`_payload_manifest_sha256`/`check()` 재작성, 신규 negative 5건) | M4.3-REQ-007(§13), M4.3-REQ-008(§13) | producer receipt 스키마(§8.2-a)에 `payload_manifest_sha256`(그 job이 선언하는 `{filename: sha256}` dict의 canonical SHA-256, `hashlib.sha256(canonical_json_bytes(payload_hashes)).hexdigest()`) 필드를 추가하고 `_check_identity`의 `required` 집합에 포함. `assemble_m4_evidence.py::_evaluate_producer`(§8.2-b)는 `_verify_payloads`가 실제 파일로 재검증한 `payload_hashes`("assembler output")에서 이 hash를 독립 재계산해 receipt가 선언한 값("baseline copy 이전 identity")과 정확히 같을 때만 `OK`를 내고(`payload_manifest_sha256_malformed`/`payload_manifest_sha256_mismatch`를 `PAYLOAD_INVALID` reason으로 신설), 그 재계산된 값을 `OK` entry의 `payload_manifest_sha256`로 함께 반환. `check_m4_baseline.py`(§9.2)는 count 상수 `PRODUCER_EXPECTED_PAYLOAD_COUNT`를 job별 review-pinned **exact filename set** `PRODUCER_EXPECTED_PAYLOAD_FILENAMES`(§8.2 `REQUIRED_PAYLOADS`와 동기화)로 교체해 `set(payload_hashes) == PRODUCER_EXPECTED_PAYLOAD_FILENAMES[job]`을 검사하고(`producer_payload_filename_set_mismatch`), `PRODUCER_STATUS_SCHEMA["OK"]`에 `payload_manifest_sha256` key를 추가해 candidate에 실려온 그 값을 `payload_hashes`에서 checker가 직접 재계산한 값과 다시 대조(`producer_payload_manifest_sha256_malformed`/`_mismatch`) — assembler output과 baseline copy의 identity를 checker 시점에도 한 번 더 결합한다. `tests/unit/test_assemble_m4_evidence.py`에 payload-manifest hash malformed/mismatch 2건(총 27 negative+1 positive+1 provenance=29개 이상), `tests/unit/test_check_m4_baseline.py`에 same-count filename substitution/extra+omission 상쇄/cross-job filename swap/payload-manifest hash malformed/mismatch 5건(총 최소 24개 파라미터화 케이스) 신설 |
| DR-I4-MIN-03 | MINOR | §4.7의 `_PINNED_M3_APPROVED_INDEX_FAISS_SHA256`/`_PINNED_M3_APPROVED_INDEX_PKL_SHA256`가 fixture seam(`_parse_m3_baseline_fingerprint` 분리, tracked-bytes positive/tampered-copy negative)은 DR-I3-MIN-06으로 이미 닫혔음에도 여전히 `"0" * 64` placeholder — 현재 승인 baseline에 이미 실제 hash(`index_faiss_sha256=c52fb288...69820`, `index_pkl_sha256=3f7217...1bb00`)가 존재하므로 "real constants" closure를 구현 단계로 미룰 이유가 없음 | §4.7(모듈 상수 코드 인용 치환, "구현 Gate 필수 조건" bullet 재작성) | M4.3-REQ-002(§13) | `_PINNED_M3_APPROVED_INDEX_FAISS_SHA256`/`_PINNED_M3_APPROVED_INDEX_PKL_SHA256`를 승인 `evaluation/baselines/m3_initial.json::reproducibility.vectorstore_fingerprint`의 실제 SHA-256 값(`c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820`/`3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00`, 이 세션이 tracked 파일을 직접 읽어 확인)로 치환 — placeholder 문자열은 문서 전체에서 제거됨. `git diff --exit-code -- evaluation/baselines/m3_initial.*`(§15) tracked baseline bytes 무변경 gate는 그대로 보존(치환 대상은 `index/lifecycle.py` 모듈 상수뿐, baseline JSON 파일 자체는 이 세션에서도 읽기만 했다). `test_pinned_baseline_provenance.py::test_pinned_constants_match_tracked_m3_baseline_bytes`(DR-I3-MIN-06이 이미 만든 fixture seam)가 이제 이 실제 상수와 tracked bytes 재계산 결과의 실질적 일치를 검증하는 유의미한 gate로 활성화된다(placeholder 상태에서는 이 positive 테스트가 항상 실패했다) |

이 표의 세 행 모두 이전 iteration들과 동일하게 "구현 가능한
symbol/schema/argv/test oracle" 수준으로 구체화됐고, 각 행이 §13
Traceability 매핑의 정확히 한 Requirement 행(REQ-002/REQ-003/REQ-007/REQ-008)에
연결된다 — Requirement.md/Traceability.md 자체는 이번 개정에서 수정하지
않았으며(이 문서는 Design.md만 개정 대상이다), §13이 그 문서들과 이
설계 사이의 유일한 매핑 지점이라는 불변식은 그대로다.

Native Linux/Ollama/DDGS, protected M3 live, M4.1 live 14-gate는 이번
개정에서도 실행하지 않았으며, `M4.1_BLOCKED=true`, protected M3 live
`NOT_RUN`, `operational_status=BLOCKED`, `overall_release_ready=false`는
§0.1/§9.1의 불변식 그대로 보존된다. 이 개정은 어떤 PASS도 합성하지
않았고 self-hosted runner/environment 승인을 변경하지 않았다. §4.7의
pinned 상수 치환은 코드 상수 리터럴을 tracked baseline의 기존 승인값과
일치시키는 문서 개정일 뿐, 승인 절차나 self-hosted/environment 설정을
바꾸지 않는다. 이번 개정은 Design.md만 수정했으며 다른 문서·코드·커밋을
동반하지 않는다.

## 20. Iteration 5 리뷰 반영 — Closure Matrix

[Design_Review_Iteration_5.md](Design_Review_Iteration_5.md)(FAIL/STOP
9.3/10, CRITICAL 0건·MAJOR 1건·MINOR 0건)를 이 개정(Iteration 6, explicit
resume)에서 어디를 어떻게 바꿔 닫았는지 정리한다. §16-§19와 같은
형식이며, "핵심 수정"은 이번 개정에서 **새로** 도입한 심볼/스키마/
reason/test oracle만 가리킨다 — §16-§19가 이미 닫은 항목을 다시
설명하지 않는다. "Requirement/Traceability" 열은 §13의 어느 행이 이
finding의 closure를 추적하는지 명시한다.

이 재개는 [milestone_dev_orchestration_guide.md](../../../milestone_dev_orchestration_guide.md)
§"Gate가 가져야 할 진행 여부 결정의 지침" 4항이 요구하는 "동일 근본
문제가 2회 연속 재발하면 즉시 중단"에 따라 Iteration 4→5 재발 이후
coordinator/user가 명시적으로 재개를 결정한 뒤 시작됐다 — Design.md
외 문서·코드·commit/push/PR/merge는 이 세션에서 다루지 않는다.

| ID | Severity | 리뷰 지적 | 반영 위치 | Requirement/Traceability | 핵심 수정(symbol/schema/reason/test oracle) |
|---|---|---|---|---|---|
| DR-I5-MAJ-01 | MAJOR | (1) `_verify_payloads`가 `payloads` 리스트를 `declared[entry["filename"]] = entry`로 즉시 dict 축약해, 같은 filename이 두 번 선언되면(동일 entry든 서로 다른 entry든) 마지막 entry가 앞 entry를 조용히 덮어써 `set(required_files) == set(declared)`가 통과하고 duplicate가 `payload_hashes`/`payload_manifest_sha256`에서 완전히 사라짐. (2) `_check_identity`가 필수 key **포함 여부**만(`required <= set(doc)`) 검사해 unknown top-level key를 가진 receipt를 통과시키고, `schema` 리터럴/`doc["job"] == 호출 producer job`/`semantic_status` enum을 전혀 검사하지 않아 다른 job이 자칭한 receipt를 그대로 옮겨 놓고 payload만 맞추는 substitution이 통과 가능. (3) pseudocode의 malformed-entry guard가 같은 `if` 문으로 두 번 연속 나타나 첫 `if` body가 없는 오탈자여서 그대로는 Python으로 구현 불가능 | §8.2(헤더/도입부 신규 문단), §8.2-a(JSON 스키마·CLI 서술 재작성, payload entry에서 `semantic_field`/`semantic_expected` 제거), §8.2-b(`RECEIPT_SCHEMA`/`RECEIPT_TOP_KEYS`/`SEMANTIC_STATUS_ENUM`/`PAYLOAD_ENTRY_KEYS`/`KNOWN_PAYLOAD_FILENAMES` 신규, `_check_identity`/`_verify_payloads`/`_evaluate_producer` 전면 재작성), §8.2-c(malformed-entry 서술 갱신), §8.3(신규 negative 10행 + 기존 1행 reason 재정의), §9.2(assembler→checker exact-identity 결합 증명 문단 신규, `check()` 로직 자체는 무변경), §16(DR-I1-MAJ-08 행에 forward 주의 추가) | M4.3-REQ-004.3(§13), M4.3-REQ-007.1/.3/.5(§13), M4.3-REQ-008.1(§13), M4.3-NFR-003/.005(§13) | **(1) exact tagged top-level schema**: `RECEIPT_SCHEMA = "m43-producer-receipt-v1"`, `RECEIPT_TOP_KEYS`(10개 key exact set), `SEMANTIC_STATUS_ENUM = {"PASS","FAIL"}`을 신설. `_check_identity(doc, job, args)`가 `set(doc) != RECEIPT_TOP_KEYS`(초과/누락 모두 `unknown_or_missing_top_level_key`), `doc["schema"] != RECEIPT_SCHEMA`(`wrong_schema`), `doc["job"] != job`(`receipt_job_mismatch`, **신규 파라미터** `job` — `_evaluate_producer`의 `REQUIRED_PRODUCERS` 순회 변수를 그대로 전달해 receipt 자기 보고가 아니라 호출자가 배정한 evidence slot과 비교), `sha`/`run_id`/`run_attempt`/`workflow_path`/`event_name`의 타입+값(bool을 int로 오인하지 않도록 `isinstance(..., bool)` 명시 제외), `semantic_status not in SEMANTIC_STATUS_ENUM`(`semantic_status_invalid`), `payload_manifest_sha256`의 타입(`payload_manifest_sha256_not_string`), `payloads`의 list 타입(`payloads_not_list`)을 dict/set 축약 **전에** 순서대로 검사한 뒤에만 `True`를 반환한다. **(2) payload entry exact schema + duplicate fail-closed**: `PAYLOAD_ENTRY_KEYS = {"filename","sha256","size_bytes"}`(이전 개정의 `semantic_field`/`semantic_expected`는 assembler 판정에 쓰인 적이 없는 정보성 필드였으므로 제거), `KNOWN_PAYLOAD_FILENAMES`(모든 job의 `REQUIRED_PAYLOADS` filename 합집합, allowlist). `_verify_payloads`가 raw `payloads` 리스트를 순회하며 각 entry의 `set(entry) != PAYLOAD_ENTRY_KEYS`(`payload_entry_schema_invalid`), `filename not in KNOWN_PAYLOAD_FILENAMES`(`payload_entry_filename_not_allowlisted`), `sha256`의 64-hex 정규식(`payload_entry_sha256_invalid`), `size_bytes`의 `isinstance(..., bool)` 우선 배제 + non-negative int(`payload_entry_size_bytes_invalid`)를 dict 축약 **전에** 개별 검사한다. 이어서 `raw_filenames = [entry["filename"] for entry in raw_payloads]`의 `len(raw_filenames) != len(set(raw_filenames))`(`payload_duplicate_filename`)로 duplicate를 — 동일 hash로 반복되든 상이한 hash로 반복되든 구분 없이 — 거부한다. 이 두 단계를 모두 통과한 뒤에만 `declared = {entry["filename"]: entry for entry in raw_payloads}`로 축약하고(이 시점엔 이미 unique/allowlisted/type-검증 완료), 기존 required-set/hash/size/semantic 재검증을 그대로 수행한 뒤 `payload_hashes`를 반환한다(`_verify_payloads`의 반환형이 2-tuple에서 3-tuple `tuple[bool, str \| None, dict[str, str] \| None]`로 변경 — canonical mapping을 이 함수가 단일하게 생성해 `_evaluate_producer`가 `doc.get("payloads", [])`를 다시 축약하던 이중 계산 지점을 제거). **(3) pseudocode 오탈자**: malformed-entry guard의 중복 `if`가 위 entry-level 순차 검사로 전면 재작성되며 사라짐 — 그대로 실행 가능한 Python이다. `tests/unit/test_assemble_m4_evidence.py`에 `test_check_identity_rejects_unknown_or_missing_top_level_key`/`test_check_identity_rejects_wrong_schema_literal`/`test_check_identity_rejects_receipt_job_mismatch`/`test_check_identity_rejects_invalid_semantic_status_enum`/`test_check_identity_rejects_non_list_payloads`/`test_verify_payloads_rejects_malformed_payload_entry_schema`/`test_verify_payloads_rejects_unknown_filename_not_allowlisted`/`test_verify_payloads_rejects_bool_as_int_size_bytes`/`test_verify_payloads_rejects_duplicate_filename_same_hash`/`test_verify_payloads_rejects_duplicate_filename_different_hash` 10개 신설(§8.3 신규 10행 1:1 대응), 기존 malformed-entry 테스트는 `payload_entry_schema_invalid` reason으로 갱신. DR-I4-MAJ-02의 same-count substitution/extra+omission 상쇄/cross-job filename swap/manifest-hash malformed/mismatch 5개 baseline-checker-level oracle(§9.2)은 그대로 보존 — 이번 개정은 그보다 앞선 assembler 단계에서 원본 receipt 자체의 identity를 exact하게 고정한다 |

### 20.1 `check_m4_baseline.py` candidate가 assembler 검증 exact identity만 전달받는 결합 증명

`check_m4_baseline.py::check()`(§9.2)는 원본 producer receipt
(`ci_producer_receipt.json`)를 어떤 경로로도 열지 않는다 — 유일한 입력은
`assemble_m4_evidence.py::assemble()`(§8.2-b)이 만들어 CI가 업로드한
`assemble/m4-baseline.json`(`--candidate`)뿐이다. 이 결합은 다음 세
symbol만으로 완전히 추적된다.

1. `assemble()`은 `REQUIRED_PRODUCERS`를 순회하며 `producers[job] =
   _evaluate_producer(job, ...)`만 실행하고, 그 반환 dict를 가공 없이
   그대로 `_build_baseline(producers, deterministic_status, args)`에
   넘긴다(§8.2-b) — `assemble()`/`_build_baseline` 어디에도 receipt의
   `payloads`를 직접 다시 읽는 코드가 없다.
2. `_evaluate_producer`의 `status == "OK"` 반환 값
   `{"status": "OK", "receipt_sha256": ..., "payload_hashes": ...,
   "payload_manifest_sha256": ...}`은 `_verify_payloads`가 (a) 원본
   `payloads` 리스트를 dict/set으로 축약하기 전에 entry exact-key/type/
   range/filename-allowlist를 통과시키고, (b) raw filename list 길이==
   unique set 길이로 duplicate(동일/상이 hash 모두)를 fail-closed
   거부하고, (c) 그 뒤 canonical mapping으로 축약해 실제 파일 hash/size/
   semantic까지 재검증한 뒤에만 반환하는 `payload_hashes`를 그대로
   담는다(§8.2-b) — 이 세 단계 중 하나라도 실패하면 함수는
   `PAYLOAD_INVALID`로 조기 반환하고 `OK` 분기 자체에 도달하지 않으므로
   `payload_hashes`/`payload_manifest_sha256` 필드가 candidate에 실릴
   가능성이 없다.
3. `check_m4_baseline.py::check()`(§9.2)는 `producers[job]`의 key 집합을
   `PRODUCER_STATUS_SCHEMA[status]`와 exact 비교한 뒤(`status != "OK"`이면
   애초에 `payload_hashes`를 읽지 않는다), `OK` entry에서만
   `set(payload_hashes) == PRODUCER_EXPECTED_PAYLOAD_FILENAMES[job]`과
   `payload_manifest_sha256 == _payload_manifest_sha256(payload_hashes)`를
   **다시** 독립 재계산해 candidate 값과 대조한다.

세 단계를 이으면: checker가 관찰하는 `payload_hashes`는 항상 "이
CI 실행에서 assembler가 원본 receipt를 exact tagged schema로 파싱하고
duplicate-free/allowlisted/실제 파일 일치까지 확인한 뒤에만 생성한
identity"이며, 원본 receipt에 DR-I5-MAJ-01이 지적한 어떤 결함(duplicate
filename, unknown key, wrong schema, job swap, malformed entry)이
있어도 그 job은 `status != "OK"`가 되어 `check()`가 그 job의
`payload_hashes`를 아예 만나지 못한다 — assembler 판정과 checker 판정
사이에 "축약된 자료를 신뢰"하는 지점이 구조적으로 존재하지 않는다.

Native Linux/Ollama/DDGS, protected M3 live, M4.1 live 14-gate는 이번
개정에서도 실행하지 않았다. `M4.1_BLOCKED=true`, protected M3 live
`NOT_RUN`, `operational_status=BLOCKED`, `overall_release_ready=false`는
§0.1/§9.1의 불변식 그대로 보존되며, 이번 개정 어디에서도 이 값들을
계산하거나 우회하는 코드를 추가하지 않았다. 이 개정은 어떤 PASS도
합성하지 않았고 self-hosted runner/environment 승인을 변경하지
않았다. §9.2 `check()`의 재계산/exact-key/algebra 로직 자체는 이번
개정에서 변경되지 않았다(§9.2 신규 문단 참조) — DR-I5-MAJ-01의 수정은
전적으로 §8.2 producer receipt parser 단계에 있다. Requirement.md/
Traceability.md 자체는 이번 개정에서 수정하지 않았으며(이 문서는
Design.md만 개정 대상이다), §13이 그 문서들과 이 설계 사이의 유일한
매핑 지점이라는 불변식은 그대로다. 이번 개정은 Design.md만 수정했으며
다른 문서·코드·commit/push/PR/merge를 동반하지 않는다.
