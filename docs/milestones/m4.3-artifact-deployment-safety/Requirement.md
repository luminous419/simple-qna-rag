# M4.3 Artifact & Deployment Safety 요구사항

상태: **구현 완료 — 로컬 결정론적 검증 PASS (Implementation_Report.md 참조), hosted CI·post-merge 증거 미발생(미커밋)**  
기준 revision: `648e3ab` (`master`, M4.2 merge)  
상위 결정: [M4 복구 결정](../m4-production-readiness/Recovery_Decision.md)  
선행 위험: [M4.1 운영 acceptance 예외](../m4.1-configuration-observability/Operational_Acceptance_Exception.md)  
승인 기준선: [M3 baseline](../../../evaluation/baselines/m3_initial.md)  
구현 보고서: [Implementation_Report.md](Implementation_Report.md)

## 1. 목적과 판정 경계

M4.3은 FAISS artifact를 provenance가 검증된 불변 version으로 만들고, 별도 staging에서
완성한 candidate만 원자적으로 활성화·rollback하며, 같은 계약을 최소 권한 OCI image와
배포/복구 runbook으로 제공한다. 또한 일반 GitHub-hosted runner에서 Python, frontend,
container와 결정론적 M4.3 Gate를 하나의 workflow/run에 묶고 M4 baseline candidate를
생성한다.

M4.3에는 서로 독립인 두 판정이 있다.

1. **M4.3 deterministic cycle**: 외부 모델·네트워크·self-hosted runner 없이 재현되는
   artifact/container/rollback/CI 계약이다. 이 범위는 독립적으로 PASS할 수 있다.
2. **전체 M4 release readiness**: M4.1 post-merge live 14-gate와 protected M3 live gate를
   포함한 운영 판정이다. 현재 `M4.1_BLOCKED=true`이며 M4.3 deterministic PASS가 이를
   해소하거나 대체하지 않는다.

M4.1의 미완료 receipt를 합성·복사·재분류하거나 protected M3 live Gate의 trigger,
runner label, environment approval을 변경해서는 안 된다. Native Linux/Ollama, DDGS 및
모든 live Gate 실행은 이번 cycle에서 제외한다. 이 제외는 전체 M4 release blocker의
면제 또는 PASS가 아니다.

## 2. 범위

포함 범위:

- canonical versioned index manifest와 build provenance
- same-filesystem staging, 검증, atomic activation과 explicit rollback
- legacy M3 index의 승인 hash 기반 read-only import
- 비루트·read-only 실행이 가능한 Linux CPU OCI production image
- 배포, 검증, restart, rollback, backup/restore와 incident recovery runbook
- 단일 GitHub Actions workflow/run의 일반 hosted Python/frontend/container/
  deterministic Gate와 fail-closed evidence assembly
- M3 reference와 M4.1/M4.2/M4.3 상태를 분리한 M4 baseline candidate

제외 범위:

- Native Linux/Ollama/DDGS, opt-in live 12-case, M3 live 14-gate의 실행
- 기존 `m3-live-regression-gate`의 self-hosted runner/environment/approval 변경
- hot reload, 무중단 다중 replica rollout, 외부 artifact registry 배포
- image signing/SBOM 의무화, 외부 vector DB, distributed queue, Kubernetes/autoscaling
- 코드 구현, commit, push, PR, merge(현재 요구사항·계획 단계)

## 3. 기능 요구사항

### M4.3-REQ-001 — canonical versioned index와 provenance

1. 각 version은 `<index-root>/versions/<version-id>/` 아래 불변 디렉터리이며 정확히
   `index.faiss`, `index.pkl`, `manifest.json`을 가진다. 활성 pointer는 version
   디렉터리 밖의 단일 `current` 항목이다.
2. canonical manifest는 schema version, version ID, 생성 시각, corpus canonical hash,
   source/chunk 수, embedding model과 revision, normalization, chunk size/overlap,
   FAISS type/dimension, settings hash, dependency lock/snapshot hash, builder Git SHA와
   dirty 상태, 두 index 파일의 size/SHA-256을 포함한다.
3. version ID는 manifest의 자기참조 필드를 제외한 canonical build identity에서
   결정론적으로 파생한다. JSON은 UTF-8, 정렬 key, 고정 separator와 trailing newline
   정책을 갖고 비유한 값·unknown schema/key를 거부한다.
4. 서비스는 manifest가 승인한 두 regular file만 읽는다. symlink, non-regular file,
   owner/mode 위반, root 탈출, hash/size/schema/settings 불일치는
   `FAISS.load_local(...allow_dangerous_deserialization=True)` 호출 전에 실패한다.
5. `index.pkl`은 신뢰된 운영자 artifact로만 허용한다. upload/URL/임의 경로에서 pickle을
   직접 로드하는 인터페이스는 만들지 않는다.

### M4.3-REQ-002 — legacy M3 import와 staging

1. legacy import는 커밋된 [M3 baseline](../../../evaluation/baselines/m3_initial.json)의
   정확한 `index.faiss`/`index.pkl` hash pair만 허용한다. CLI가 제공한 임의 expected hash는
   승인 근거가 될 수 없다.
2. import/build는 활성 경로와 다른 `<index-root>/.staging/<operation-id>/`에서 수행한다.
   staging과 versions/current는 같은 filesystem이어야 하며 cross-device activation은
   거부한다.
3. staging은 새 디렉터리 생성, 파일 write, file fsync, manifest write/fsync, directory
   fsync, hash/load smoke, immutable destination rename 순서를 완료하기 전에는 candidate가
   아니다.
4. legacy 원본과 M3 baseline은 byte 단위로 변경하지 않는다. source root 또는 모든
   path component의 symlink/ownership/mode/TOCTOU 이상은 load 전에 fail-closed한다.
5. 실패·취소·disk full 뒤 staging 잔여물은 inactive로 남아야 하며, cleanup은 검증된
   `.staging` child만 대상으로 하는 explicit dry-run-first 명령이어야 한다.

### M4.3-REQ-003 — atomic activation, rollback과 retention

1. build/import/activate/rollback/cleanup mutation은 index root의 OS advisory lock 한 개로
   직렬화한다. lock contention은 active pointer를 변경하지 않고 안정된 nonzero exit로
   즉시 또는 bounded timeout 후 실패한다.
2. activation은 완전 검증된 immutable version에 대한 새 temporary pointer를 만든 뒤
   같은 디렉터리의 atomic replace로 `current`를 교체하고 parent directory를 fsync한다.
   두 단계 pointer 또는 partial manifest 상태를 허용하지 않는다.
3. 모든 validation은 pointer 교체 전에 수행한다. write/rename/fsync/callback 어느 지점의
   실패도 이전 current의 target과 bytes/hash를 보존해야 한다. 성공 receipt는 durable
   pointer fsync 이후에만 발행한다.
4. rollback도 대상 version을 현재 설정으로 다시 검증한 뒤 같은 activation primitive로
   pointer만 교체한다. 새 index를 만들거나 실패 version을 자동 선택하지 않는다.
5. 현재 version, rollback 직전 version과 service가 사용 중이라고 명시한 version은 삭제
   금지다. retention은 dry-run 기본, 명시적 apply, root-contained regular directory,
   재검증과 lock을 요구한다.
6. M4 기본 배포는 `activate -> service restart -> readiness 확인`이다. 실행 중 process의
   in-memory index hot swap은 하지 않는다.

### M4.3-REQ-004 — lifecycle CLI와 fail-closed receipt

1. `build`, `import-legacy`, `verify`, `activate`, `rollback`, `list`, `cleanup`은 명시적
   subcommand와 안정된 exit code를 가진다. destructive mutation은 `--dry-run`/`--apply`
   경계를 명확히 하며 default는 비파괴다.
2. 성공·실패 receipt는 schema, operation ID, Git/settings/lock identity, source/target
   version, pre/post pointer, 검증 목록, 시작/종료 시각, outcome/error code를 포함한
   canonical JSON이다. exception text, absolute private path, 문서 내용과 secret은 없다.
3. receipt write/upload 실패, 필수 필드 누락, duplicate/unknown producer, hash/binding
   mismatch는 PASS가 아니다. stderr 문구나 process exit 0만으로 증거를 합성하지 않는다.
4. `CorpusManifestError`와 lifecycle 오류는 traceback 없이 안정된 code/exit로 변환하며
   active artifact 불변성을 함께 검사한다.

### M4.3-REQ-005 — 최소 권한 OCI production image

1. root `deploy/Dockerfile`은 명시적 `test`와 `production` stage를 가진 Linux CPU
   multi-stage build다. production은 `requirements.lock` hash 설치만 사용하며 builder,
   compiler, test/evaluation/runtime data를 포함하지 않는다.
2. `.dockerignore`는 `.git`, `.env*`, `runtime/`, `evaluation/reports/`, caches, local venv,
   model/index/document artifact와 secret fixture를 제외한다. 필요한 package metadata와
   README는 명시적으로 포함한다.
3. production process는 고정 non-root UID/GID로 실행하며 read-only root filesystem,
   tmpfs, `no-new-privileges`, capability drop-all에서 liveness/readiness와 mock query를
   처리한다. documents/index는 read-only mount이고 index mutation은 별도 operator
   command/write volume이다.
4. image에는 Ollama model, corpus, index, Git directory, CI report, credential이 없어야
   한다. OCI archive의 모든 layer를 path-normalize해 검사하고 traversal/whiteout을
   보수적으로 처리한다.
5. CI는 image build, package import/config, numeric non-root identity, read-only runtime,
   mock health/query, graceful stop과 layer positive/negative control을 검증한다. 실제
   Ollama image test는 deterministic container PASS의 구성요소가 아니다.

### M4.3-REQ-006 — deployment와 recovery runbook

1. runbook은 digest-pinned image, settings check, Ollama/model preflight(실행이 아닌 운영자
   절차), volume owner/mode, index verify/activate, restart, readiness와 smoke 확인을 정확한
   명령 및 기대 exit/status로 제공한다.
2. 배포 전 snapshot, current/previous version, image digest, settings/lock hash를 기록하고
   배포 후 pointer/image/readiness가 같은 release identity인지 검증한다.
3. readiness 실패, manifest/hash mismatch, corrupted staging, lock contention, disk full,
   orphan saturation, Ollama outage, container start/stop 실패별 진단과 복구 순서를 둔다.
4. rollback은 traffic 중지 또는 내부 bind 확인, 이전 image digest와 index version 검증,
   atomic pointer rollback, service restart, readiness 확인 순서다. 어느 검증이 실패해도
   더 진행하지 않는 fail-closed 중단점과 escalation 정보를 명시한다.
5. backup/restore는 immutable version 디렉터리와 manifest/hash 검증을 보존하고,
   untrusted restored pickle을 바로 활성화하지 않는다.

### M4.3-REQ-007 — 단일 workflow deterministic/hosted CI Gate

1. `.github/workflows/ci.yml` 한 workflow/run 안에서 일반 GitHub-hosted runner가
   `python-tests`, `frontend-tests`, `container`, `m43-deterministic`, `m4-assemble`을
   실행한다. 필요한 producer job은 모두 같은 `github.run_id`, `github.run_attempt`,
   `github.sha`, workflow path/event에 bind된다.
2. PR에서는 Python/frontend/container/deterministic을 실행하고 pre-merge candidate를
   assemble한다. push-to-master에서는 같은 hosted Gate를 재실행해 정확한 merge SHA의
   post-merge receipt를 생성한다.
3. `m4-assemble`은 `needs`의 모든 필수 hosted job이 success이고, 각 artifact schema/hash/
   identity가 일치하며, fresh empty assemble directory에 중복 없이 모인 경우에만
   deterministic PASS를 낸다. missing/skipped/cancelled/expired/malformed evidence는 FAIL이다.
4. 현재 protected `m3-live-regression-gate`는 이름, trigger allowlist, `runs-on:
   [self-hosted, ollama-m3]`, `environment: m3-live-regression`과 승인 정책을 그대로 둔다.
   hosted assembler는 live job의 skip/미실행을 PASS로 변환하거나 기존 M4.1 receipt를
   생성하지 않는다.
5. workflow와 assembler에는 삭제/변조, cross-run/SHA mismatch, duplicate producer,
   skipped producer, path traversal, stale artifact, synthesized PASS negative control이 있다.

### M4.3-REQ-008 — M4 baseline과 상태 분리

1. machine-readable M4 baseline은 schema/version, exact Git SHA, workflow run identity,
   M3 fingerprint reference, dependency/settings/index manifest/image digest, M4.2 deterministic
   receipt hash와 각 Gate 상태를 포함한다.
2. baseline은 `deterministic_status`와 `operational_status`를 별도 필드로 둔다.
   M4.3 deterministic Gate가 성공해도 M4.1 live receipt가 없으면
   `operational_status=BLOCKED`, `M4.1_BLOCKED=true`, `overall_release_ready=false`다.
3. `NOT_RUN`, `SKIPPED`, `UNKNOWN`, `BLOCKED`, `PASS`, `FAIL`은 서로 다른 enum이며
   비-PASS를 truthy/누락/default PASS로 처리하지 않는다.
4. baseline candidate는 post-merge hosted receipt까지 만들 수 있으나, protected live
   evidence 미실행 상태에서 M4 완료 표시, Roadmap 완료 변경 또는 release-ready 주장을
   금지한다.

### M4.3-REQ-009 — 기존 계약과 orchestration readiness fix 보존

1. M4.2 merge SHA `648e3ab`의 API, settings, logging, metrics, health, executor,
   deterministic acceptance와 M3 baseline bytes를 보존한다.
2. `scripts/orchestration_watchdog.py`의 working-tree delta는 root/coordinator가 의도적으로
   만든 M4.3 continuous-readiness fix다. tracked base `e57fe1c`의 run-only 조회를 현재
   coordinator terminal identity에 bind하도록 `task-list --from <terminal>`과
   `check --terminal <terminal>`을 추가한 provenance를 보존한다.
3. 현재 Orca `--help`가 해당 문법을 지원하는 것만으로 PASS하지 않는다.
   `tests/unit/test_orchestration_watchdog.py`에서 exact argv, 다른 terminal/run의 message를
   소비하지 않는 read-only peek, 오류의 fail-closed wake 기록을 검증하고 실제 bound-Run
   dry-run receipt와 독립 review를 거친다.
4. 이 fix와 대응 test/review evidence는 M4.3의 의도된 commit scope에 포함한다. 구현
   cycle 전까지 기존 delta를 되돌리거나 의미를 바꾸지 않는다.

## 4. 비기능 요구사항

| ID | 요구사항 |
|---|---|
| M4.3-NFR-001 재현성 | 같은 입력 identity로 manifest/version ID/evidence schema가 반복 실행에서 동일하다. 생성 시각·run ID 같은 실행 metadata는 판정 identity와 분리한다. |
| M4.3-NFR-002 신뢰성 | crash, disk full, contention, corruption, receipt 실패에서 이전 active index와 release 상태가 보존된다. |
| M4.3-NFR-003 보안 | pickle·filesystem·OCI layer·CI artifact를 명시적 trust boundary로 다루고 path/owner/mode/hash를 load 전에 검증한다. |
| M4.3-NFR-004 호환성 | M3 bytes와 M4.1/M4.2 public 계약을 보존하며 legacy import는 opt-in이다. |
| M4.3-NFR-005 검증성 | 필수 pre-merge Gate는 hosted/offline/deterministic이고 negative control이 동일 parser를 실제 실패시킨다. |
| M4.3-NFR-006 복구성 | rollback drill이 이전 image/index pair로 복귀하고 불일치 시 중단하며 RTO를 receipt에 기록한다. |

## 5. 정량 수용 기준

| Gate | PASS 조건 |
|---|---|
| manifest | canonical round-trip 100회 동일, 필수 필드/파일 hash 100%, unknown/non-finite/symlink/owner/mode/hash mismatch 전부 load 0회 |
| staging | write/manifest/fsync/rename/load-smoke 각 fault injection에서 pre/post current target와 file hash 변화 0 |
| activation/rollback | 정상 activate/rollback 100회 partial/dangling pointer 0, 대상 검증 실패 시 pointer mutation 0, lock 경쟁 mutation 1개 이하 |
| legacy | committed M3 hash pair만 import 성공, 원본/baseline byte 변화 0, 임의 hash/root/symlink/TOCTOU matrix 전부 거부 |
| receipt | missing/duplicate/malformed/cross-SHA/run/path/stale evidence 전부 FAIL, genuine receipt 한 필드 변조 negative control exit 1 |
| container | clean hosted build, non-root UID, read-only/drop-all/no-new-privileges mock smoke PASS, forbidden layer match 0, scanner positive/negative control PASS |
| runbook | clean temporary index root에서 deploy/verify/activate/restart/rollback drill을 mock profile로 3회 실행, 최종 image/index identity가 시작 identity와 같고 실패 주입은 중단점 준수 |
| regression | `pytest`, dataset validate, frontend, vendor sync, generated docs/audits, Markdown links, compile, lock verify, `git diff --check` PASS |
| hosted workflow | 같은 run의 Python/frontend/container/m43-deterministic/m4-assemble success와 artifact identity 일치 |
| M4 baseline | deterministic PASS와 operational BLOCKED를 동시에 정확히 기록; `overall_release_ready=false`, `M4.1_BLOCKED=true` |

모든 M4.3 deterministic 필수 Gate가 PASS이고 review Gate가 9.7/10 이상,
CRITICAL/MAJOR 0이어야 M4.3 deterministic cycle을 PASS로 판정한다. Native Linux/Ollama와
protected live Gate가 `NOT_RUN`인 것은 이 판정을 방해하지 않지만, 전체 M4 release
readiness는 계속 BLOCKED다.

## 6. Gate 경계

### Pre-merge Code Quality Gate

- 요구사항·설계·추적성, 전체 local/clean deterministic test와 negative control
- workflow 정적 계약 및 PR hosted Python/frontend/container/deterministic receipts
- index fault injection과 mock deployment/rollback drill
- acceptance checker의 fail-closed·fresh-directory·same-run/SHA binding

구조상 merge 후에만 얻는 exact merge SHA receipt의 부재만으로 pre-merge를 FAIL하지
않는다. 단, post-merge checker가 fail-open이거나 실행 불가능하면 pre-merge MAJOR다.

### Post-merge Operational Acceptance Gate

- `master` exact merge SHA에서 같은 hosted workflow Gate 재실행과 baseline candidate
- 별도 protected M3 live/M4.1 operational receipt 검증
- image/index/settings/dependency identity와 rollback drill receipt 검증

이 cycle은 첫 항목의 계약과 자동화를 설계하지만 live 항목을 실행하지 않는다. 따라서
M4.3 deterministic 결과와 무관하게 전체 M4 post-merge Operational Acceptance는 BLOCKED다.
